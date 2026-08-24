"""Top-level retriever: wires chunk loading, dense/BM25/hybrid search,
optional query translation, and optional reranking into one call, driven
by a plain dict of config (see configs/retrieval.yaml / configs/ablations.yaml)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from taglish_rag.config import data_path
from taglish_rag.retrieval.bm25 import BM25Index
from taglish_rag.retrieval.embeddings import embed_texts
from taglish_rag.retrieval.index import DenseIndex, fuse_hybrid
from taglish_rag.schemas import RetrievedChunk


def query_cache_id(query: str) -> str:
    return "q::" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


@dataclass
class RetrievalConfig:
    chunk_size: int = 512
    overlap: int = 64
    embedding_model: str = "bge-m3"
    mode: str = "hybrid"  # dense | bm25 | hybrid
    top_k: int = 10
    hybrid_alpha: float = 0.5
    use_reranker: bool = False
    rerank_top_n: int = 20
    translate_query_to_english: bool = False


def load_chunks(chunk_size: int, overlap: int) -> list[dict]:
    path = data_path("processed", f"chunks_{chunk_size}_{overlap}.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run python -m taglish_rag.ingest.pipeline "
            f"--chunk-size {chunk_size} --overlap {overlap}"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Retriever:
    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.chunks = load_chunks(cfg.chunk_size, cfg.overlap)
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.texts_by_id = {c["chunk_id"]: c["text"] for c in self.chunks}
        self.doc_id_by_chunk = {c["chunk_id"]: c["doc_id"] for c in self.chunks}

        self._bm25 = None
        self._dense = None
        if cfg.mode in ("bm25", "hybrid"):
            self._bm25 = BM25Index(self.chunk_ids, [c["text"] for c in self.chunks])
        if cfg.mode in ("dense", "hybrid"):
            embeddings = embed_texts(
                cfg.embedding_model, self.chunk_ids, [c["text"] for c in self.chunks], is_query=False
            )
            self._dense = DenseIndex(self.chunk_ids, embeddings)

    def _maybe_translate(self, query: str) -> str:
        if not self.cfg.translate_query_to_english:
            return query
        from taglish_rag.retrieval.translate import translate_to_english

        return translate_to_english(query)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        cfg = self.cfg
        query = self._maybe_translate(query)

        candidate_k = cfg.rerank_top_n if cfg.use_reranker else cfg.top_k

        if cfg.mode == "bm25":
            results = self._bm25.search(query, candidate_k)
        elif cfg.mode == "dense":
            qvec = embed_texts(cfg.embedding_model, [query_cache_id(query)], [query], is_query=True)
            results = self._dense.search(qvec[0], candidate_k)
        else:  # hybrid
            bm25_results = self._bm25.search(query, candidate_k)
            qvec = embed_texts(cfg.embedding_model, [query_cache_id(query)], [query], is_query=True)
            dense_results = self._dense.search(qvec[0], candidate_k)
            results = fuse_hybrid(dense_results, bm25_results, cfg.hybrid_alpha, candidate_k)

        if cfg.use_reranker:
            from taglish_rag.retrieval.rerank import rerank

            candidates = [(cid, self.texts_by_id[cid]) for cid, _ in results]
            results = rerank(query, candidates, cfg.top_k)
        else:
            results = results[: cfg.top_k]

        return [
            RetrievedChunk(chunk_id=cid, score=score, rank=i)
            for i, (cid, score) in enumerate(results)
        ]
