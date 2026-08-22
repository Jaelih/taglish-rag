# Supabase for the deployed app — implementation guide

**Status:** not implemented. This document is the instruction set; nothing in it has been applied to the codebase yet.

---

## 1. Goal and scope boundary

Serve the deployed Gradio app's dense retrieval from **Supabase (Postgres + pgvector)** while the local evaluation and ablation harness keeps using **FAISS**.

This split is deliberate. The benchmark half of this project depends on being fast, offline, and reproducible: the ablation sweep re-embeds and re-indexes across six axes, CI runs keyless with no network, and anyone cloning the repo must be able to reproduce the published numbers without provisioning a hosted database. Routing the eval harness through Supabase would cost all three. The deployed app has the opposite profile — one fixed config, no sweeping, and a real benefit to not shipping a corpus inside the image.

| | Local (eval, ablation, CI) | Deployed (Docker / HF Spaces) |
|---|---|---|
| Dense index | FAISS `IndexFlatIP` | Supabase pgvector |
| Chunk source | `data/processed/chunks_*.jsonl` | `chunks` table |
| Configs exercised | all 6 ablation axes | exactly one (below) |
| Credentials | none | `SUPABASE_URL`, `SUPABASE_KEY`, `GOOGLE_API_KEY` |

**Out of scope:** moving BM25 to Postgres full-text search, and routing the eval harness through Supabase. See §9.

### What already exists vs. what must be built

`src/taglish_rag/retrieval/supabase_store.py` exists but is **entirely unwired** — `SupabaseDenseIndex` is imported by nothing, `upsert()` is called by nothing, and `vector_store.backend` in `configs/retrieval.yaml` is read by nothing. `Retriever.__init__` hardcodes `DenseIndex` at `retriever.py:61`. Treat the existing file as a sketch to be rewritten, not a working component to be switched on.

---

## 2. The config the deployed app will serve

```python
RetrievalConfig(
    embedding_model="bge-m3",      # 1024-dim, normalize=true
    chunk_size=1024, overlap=128,  # 273 chunks
    mode="dense",                  # no BM25
    top_k=10,
    use_reranker=False,            # no cross-encoder
    translate_query_to_english=False,
)
```

Measured on the standard n=72 retrieval-eligible eval items:

| Config | R@1 | R@5 | MRR | EN | TL | Taglish |
|---|---|---|---|---|---|---|
| **dense 1024/128, no reranker** | **0.814** | 0.929 | 0.951 | 0.821 | 0.821 | 0.800 |
| dense 1024/128, + reranker | 0.814 | 0.933 | 0.958 | 0.821 | 0.800 | 0.821 |
| hybrid 1024/128, + reranker | 0.821 | 0.926 | 0.958 | 0.821 | 0.821 | 0.821 |
| prior published "best" (512/64 hybrid + reranker) | 0.804 | 0.940 | 0.950 | 0.821 | 0.792 | 0.800 |

Raw results are in `results/retrieval_dense_1024_norerank.json`, `retrieval_dense_1024_rerank.json`, `retrieval_hybrid_1024_rerank.json`.

**Why not the 0.821 config.** At n=72 the standard error on a proportion near 0.81 is roughly ±4.6pp; the spread across the top three is 0.7pp. They are statistically tied. The 0.821 config buys that indistinguishable difference with a full BM25 index *and* a per-query cross-encoder pass. The chosen config is the only one of the three that reduces to a single pgvector query plus one query embedding.

**Why the levers don't stack.** Dense retrieval, the cross-encoder, and query translation are three different fixes for the *same* failure — Tagalog queries against an English corpus (baseline TL R@1 0.562). Any one of them lifts TL to roughly 0.80. Applying all three is redundant work, and translation actively hurt Taglish in the sweep (0.696 vs 0.717).

Each choice also removes a deployment dependency: `mode="dense"` means no BM25 index and therefore no local corpus needed to build one; `use_reranker=False` means no `bge-reranker-v2-m3` in the container.

---

## 3. Step 1 — Database schema

Create a `migrations/001_chunks.sql` file in the repo (right now this schema exists only as a comment in a docstring, which means it is untracked and undeployable):

```sql
create extension if not exists vector;

create table chunks (
    chunk_id        text primary key,
    doc_id          text not null,
    agency          text not null,
    title           text not null,
    url             text not null,
    text            text not null,
    position        int  not null,
    chunk_size      int  not null,
    overlap         int  not null,
    embedding_model text not null,
    embedding       vector(1024) not null
);

create or replace function match_chunks(
    query_embedding vector(1024),
    match_count     int default 10
)
returns table (chunk_id text, similarity float)
language sql stable
as $$
    select c.chunk_id,
           1 - (c.embedding <=> query_embedding) as similarity
    from chunks c
    order by c.embedding <=> query_embedding
    limit match_count;
$$;
```

Three things to get right here:

**Do not create an `ivfflat` index.** The existing docstring in `supabase_store.py` recommends one; at 273 rows it is actively harmful. `ivfflat` is an *approximate* index — it would make deployed retrieval diverge from the exact FAISS `IndexFlatIP` search the 0.814 number was measured with, silently and unquantifiably. A sequential scan over 273 rows is sub-millisecond and exactly matches the eval. Revisit only if the corpus grows by orders of magnitude.

**`vector(1024)` is specific to bge-m3.** `minilm-multilingual` is 384-dim (`configs/embeddings.yaml`). The column type hard-codes a commitment to the deployed embedding model; changing models means a migration, not a config edit.

**Store chunk text and metadata, not just vectors.** Downstream code needs `title`, `url`, `agency`, and `text` for citation display (`app/app.py:34-35`) and prompt context (`crag.py:41-42`). Keeping them in the row is what lets `chunks_*.jsonl` drop out of the image entirely.

The `chunk_size` / `overlap` / `embedding_model` columns are not queried — they exist so the table is self-describing and so the push script can assert it is not mixing configs. This matters because `chunk_id` is `{doc_id}__{position}`, which **collides across chunk sizes**: the 256/32 and 1024/128 sets both contain `bir-rmc-no-38-2026-digest__0` with different text. This table holds exactly one config; do not load a second into it.

---

## 4. Step 2 — Single source of truth for the deployed config

The push script and the running app must agree on embedding model, chunk size, and overlap. If they drift, retrieval breaks silently — wrong-config vectors still return plausible-looking neighbors, and a dimension mismatch is the *lucky* case because it at least raises.

Add to `configs/retrieval.yaml`:

```yaml
# The single config the deployed app serves. scripts/push_to_supabase.py
# must upload exactly this config -- see docs/SUPABASE_MIGRATION.md.
deployed:
  embedding_model: bge-m3
  chunk_size: 1024
  overlap: 128
  mode: dense
  top_k: 10
  use_reranker: false
  translate_query_to_english: false
```

Add a loader in `src/taglish_rag/retrieval/retriever.py`:

```python
def load_deployed_config() -> RetrievalConfig:
    """The one config the deployed app serves, read from configs/retrieval.yaml
    so the app and the Supabase push script cannot drift apart."""
    from taglish_rag.config import load_yaml
    cfg = load_yaml("retrieval.yaml")["deployed"]
    return RetrievalConfig(vector_store="supabase", **cfg)
```

Also **delete the dead `VECTOR_STORE_BACKEND` line from `.env.example`.** There are currently two competing config surfaces for the same setting (`vector_store.backend` in YAML and `VECTOR_STORE_BACKEND` in env) and neither is read. Keep the YAML one; the env var only invites drift.

---

## 5. Step 3 — Rewrite `supabase_store.py`

Two classes. The dense search goes to the server; chunk metadata is loaded once at startup and cached in process.

```python
class SupabaseDenseIndex:
    """pgvector-backed dense search. Same (chunk_id, score) contract as
    retrieval.index.DenseIndex, so Retriever can use either."""

    def __init__(self, table: str = "chunks"):
        from supabase import create_client
        url, key = env("SUPABASE_URL"), env("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        self._client = create_client(url, key)
        self._table = table

    def search(self, query_vec, top_k: int) -> list[tuple[str, float]]:
        resp = self._client.rpc("match_chunks", {
            "query_embedding": query_vec.tolist(),
            "match_count": top_k,
        }).execute()
        return [(r["chunk_id"], float(r["similarity"])) for r in resp.data]

    def upsert(self, rows: list[dict], batch_size: int = 100) -> None:
        for i in range(0, len(rows), batch_size):
            self._client.table(self._table).upsert(rows[i:i + batch_size]).execute()

    def load_chunk_records(self) -> list[dict]:
        """All chunk metadata (no embeddings), shaped exactly like a row of
        data/processed/chunks_*.jsonl so Retriever's consumers are unchanged."""
        cols = "chunk_id,doc_id,agency,title,url,text,position,chunk_size,overlap"
        out, page = [], 0
        while True:
            resp = (self._client.table(self._table).select(cols)
                    .range(page * 1000, page * 1000 + 999).execute())
            if not resp.data:
                break
            out.extend(resp.data)
            page += 1
        return out
```

Changes from the current stub: batching in `upsert` (the existing version sends every row in one request), a new `load_chunk_records()`, and pagination (PostgREST caps response rows — 273 fits under the default 1000, but the loop costs nothing and prevents a silent truncation if the corpus grows).

Loading all 273 metadata rows at startup is the right call over per-query lookups: it preserves the existing `retriever.chunks` / `texts_by_id` / `doc_id_by_chunk` contract that four call sites depend on, costs a few hundred KB, and avoids a network round trip on every citation render.

---

## 6. Step 4 — Backend switch in `retriever.py`

Add the field to `RetrievalConfig`:

```python
vector_store: str = "faiss"   # faiss | supabase
```

Defaulting to `faiss` is what keeps every existing eval, ablation, and test path untouched.

Then branch in `Retriever.__init__` (currently `retriever.py:48-61`, which hardcodes both the JSONL load and `DenseIndex`):

```python
if cfg.vector_store == "supabase":
    from taglish_rag.retrieval.supabase_store import SupabaseDenseIndex
    store = SupabaseDenseIndex()
    self.chunks = store.load_chunk_records()
else:
    self.chunks = load_chunks(cfg.chunk_size, cfg.overlap)

self.chunk_ids = [c["chunk_id"] for c in self.chunks]
self.texts_by_id = {c["chunk_id"]: c["text"] for c in self.chunks}
self.doc_id_by_chunk = {c["chunk_id"]: c["doc_id"] for c in self.chunks}

self._bm25 = None
self._dense = None
if cfg.mode in ("bm25", "hybrid"):
    self._bm25 = BM25Index(self.chunk_ids, [c["text"] for c in self.chunks])
if cfg.mode in ("dense", "hybrid"):
    if cfg.vector_store == "supabase":
        self._dense = store
    else:
        embeddings = embed_texts(cfg.embedding_model, self.chunk_ids,
                                 [c["text"] for c in self.chunks], is_query=False)
        self._dense = DenseIndex(self.chunk_ids, embeddings)
```

`retrieve()` needs no changes — both backends expose `search(query_vec, top_k)` returning `list[tuple[str, float]]`.

Add a guard rejecting `vector_store="supabase"` combined with `mode` in `("bm25", "hybrid")`. It is not conceptually impossible — BM25 would just build from the fetched text — but nothing verifies the fetched corpus matches what was measured, and the deployed config never needs it. Fail loudly rather than serve an unvalidated path.

---

## 7. Step 5 — Population script

New file `scripts/push_to_supabase.py`. It must read the same `deployed:` block from §4 rather than taking its own flags — that shared read is the mechanism preventing drift.

Responsibilities, in order:

1. Load `deployed:` config; resolve chunk file `data/processed/chunks_{chunk_size}_{overlap}.jsonl`.
2. Assert `get_model_config(embedding_model)["dim"] == 1024` — a mismatch against the column type must fail here, not on first query.
3. Embed all chunk texts via the existing `embed_texts(...)` with `is_query=False`. This reuses the local `.npy` cache, so it is instant if the ablation sweep already embedded this set.
4. Build rows: every JSONL field plus `embedding_model` and `embedding` (`vec.tolist()`).
5. `upsert` in batches of 100.
6. Verify: re-read `count(*)`, assert it equals the local chunk count (273 for 1024/128). A partial upload is the failure mode most likely to go unnoticed.

Give it a `--dry-run` that does everything except the upsert, and have it refuse to run if the table already contains rows with a different `(embedding_model, chunk_size, overlap)` triple — the `chunk_id` collision described in §3 makes silent mixing a real hazard.

---

## 8. Step 6 — Application and container

### `app/app.py`

Replace line 20:

```python
_retriever = Retriever(RetrievalConfig(embedding_model="minilm-multilingual"))
```

with:

```python
_retriever = Retriever(load_deployed_config())
```

This is a substantial quality change independent of Supabase — the app currently serves `minilm-multilingual`, the *worst* model in the sweep (overall R@1 0.440, Tagalog 0.167). Moving to bge-m3 dense at 1024/128 takes the deployed app from 0.440 to 0.814.

The `next(c for c in retriever.chunks ...)` citation lookup at line 34 keeps working unchanged, since `load_chunk_records()` returns the same record shape.

### `app/requirements.txt`

- **Add** `supabase>=2.6` — it is currently only in `pyproject.toml`, so the deployed image would `ImportError` on first query.
- **Remove** `faiss-cpu>=1.8` — `import faiss` is lazy (inside `DenseIndex.__init__`, `index.py:11`) and `DenseIndex` is never constructed on the Supabase path. Saves ~30MB.
- **Keep** `rank_bm25` despite BM25 being unused: `bm25.py:7` imports `BM25Okapi` at module top and `retriever.py:11` imports `BM25Index` at module top, so removing the package breaks the import chain. The package is tiny; making those imports lazy is a valid alternative but not worth the churn.

### `Dockerfile`

- **Remove** the `COPY data/processed/chunks_512_64.jsonl` line (currently line 24) and its explanatory comment — chunk data now comes from Supabase. This is the main size win.
- **Pre-download bge-m3 at build time**, in a layer before the app code:
  ```dockerfile
  RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
  ```
  bge-m3 is roughly 2.3GB against minilm's ~470MB. Without this, the first user request after every cold start pays a multi-GB download. Baking it in trades image size for a usable cold start.
- Document the three required runtime secrets: `GOOGLE_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`.

### `docs/DEPLOY.md`

Add the Supabase prerequisite (run the migration, run the push script, set the two secrets) ahead of the existing Docker and HF Spaces sections, and add `SUPABASE_URL` / `SUPABASE_KEY` to the HF Spaces repository-secrets step.

---

## 9. Step 7 — Correct the ablation write-up

Independent of the deployment work, and arguably the more interesting result.

`README.md` currently presents 80.4% (512/64 hybrid + reranker) as "best measured," generated by `scripts/render_results.py` from `results/ablation_summary.json`. Two problems: a measured combination beats it (81.4%), and the framing implies the sweep identified a winner when the top configs are statistically tied.

What to change:

- Fold the three combination runs into the results rendering so the "best measured" row reflects an actual best.
- State the interaction finding explicitly: **one-axis-at-a-time sweeps cannot detect redundancy between levers.** Three separate axes (retrieval mode, reranker, query translation) each appeared to be a large independent win; all three were substantially fixing the same Tagalog failure, and combining them adds nothing.
- Add the n=72 confidence interval (±~4.6pp near 0.81) so readers can distinguish the real signal — BM25's Tagalog collapse at 2.9% vs dense at 79–82%, minilm at 16.7% vs bge-m3 at 56.2%, both far outside noise — from the fractions at the top of the table, which are not resolvable at this sample size.
- Note in Limitations that the deployed demo serves a config selected for cost/quality balance among statistically tied options, not a uniquely optimal one.

Reporting this honestly is a stronger result than the extra percentage point.

---

## 10. Verification

Run in order; each catches a distinct failure.

1. **Local harness untouched** — `uv run pytest tests/ -v` keyless, then `uv run python -m taglish_rag.eval.runner --mode bm25 --limit 20 --label ci_smoke`. Both must behave exactly as before; the `vector_store` default of `faiss` is what guarantees this.
2. **Schema and RPC** — in the Supabase SQL editor, call `match_chunks` with a zero vector of length 1024. Should return rows, not an error.
3. **Push, dry run first** — `python scripts/push_to_supabase.py --dry-run`, then for real. Confirm the reported row count is 273.
4. **Parity check — the important one.** Run the eval harness against the Supabase backend and confirm it reproduces the FAISS numbers:
   ```
   uv run python -m taglish_rag.eval.runner \
       --embedding-model bge-m3 --chunk-size 1024 --overlap 128 \
       --mode dense --label supabase_parity
   ```
   (with `vector_store="supabase"` forced). Expect R@1 0.814 ± rounding. **A meaningful gap means something is wrong** — most likely an approximate index was created, the wrong chunk set was uploaded, or normalization differs. Do not proceed past a mismatch.
5. **App locally against Supabase** — `uv run python app/app.py`, ask one Tagalog and one English question, confirm answers and citations render.
6. **Container** — build, run with the three secrets, repeat step 5. Confirm no `chunks_*.jsonl` in the image (`docker run --rm <img> ls data/processed/` should be empty or absent).
7. **Failure modes are loud** — unset `SUPABASE_KEY` and confirm a clear `RuntimeError` at startup rather than an empty result set at query time.

---

## 11. Known risks and gotchas

**Per-query embedding cache writes to disk.** `Retriever.retrieve` calls `embed_texts(..., [query_cache_id(query)], ...)`, and `embed_texts` unconditionally `np.save`s to `data/index/embeddings/` (`embeddings.py:52,72`). In the eval harness this is a valuable cache. In a deployed app it writes a new `.npy` for **every unique user query**, growing unbounded for the container's lifetime. Not fatal on ephemeral HF Spaces storage, but it is a slow leak and should be addressed — add a `cache: bool = True` parameter to `embed_texts` and pass `cache=False` for queries on the deployed path.

**bge-m3 still loads in-container.** Supabase stores the passage vectors, but the incoming query must be embedded with the *same* model, so the ~2.3GB model is a hard requirement. This is the main cost of choosing quality over the current minilm.

**Free-tier cold starts.** Model load plus the startup metadata fetch means the first request is slow. Consider a warm-up call at app start rather than waiting for a user.

**The service key is a secret.** Use the anon key with row-level security if the table is read-only to the app; never commit either. `.env` is gitignored — keep it that way.

**No Supabase test coverage exists**, and none is proposed here: meaningful tests need either a live project or a mocked PostgREST layer. The parity check in §10.4 is the real safety net, and it should be re-run whenever the corpus or embedding model changes.

**Unrelated pre-existing issue worth folding in:** `.gitignore:13` is `README.md` with no leading slash, so it matches at every level and silently excludes `eval/human_labels/README.md`. `.gitignore:7` also excludes `.env.example`. Changing line 13 to `/README.md` scopes it to the repo root.
