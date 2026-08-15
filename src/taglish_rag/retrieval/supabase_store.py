"""Supabase (Postgres + pgvector) dense store -- an alternative backend to
the local FAISS index, matching the vector-DB stack used in production at
Amdocs (see README). FAISS is the default (`VECTOR_STORE_BACKEND=faiss` in
configs/retrieval.yaml) because it needs no hosted service and keeps the
ablation sweep's eval loop fast; this module exists so the retriever can
point at a real hosted pgvector instance instead, given SUPABASE_URL/
SUPABASE_KEY, without changing anything else in the retrieval pipeline
(same chunk_id-in, (chunk_id, score) list-out contract as DenseIndex).

Requires a table created ahead of time, e.g.:

    create extension if not exists vector;
    create table chunks (
        chunk_id text primary key,
        embedding vector(1024)  -- match the configured embedding model's dim
    );
    create index on chunks using ivfflat (embedding vector_cosine_ops);

Untested against a live Supabase project in this build (no credentials
were configured -- see .env.example) -- included to demonstrate the
integration point, not as a verified-working deployment path.
"""
from __future__ import annotations

import numpy as np

from taglish_rag.config import env


class SupabaseDenseIndex:
    def __init__(self, table: str = "chunks"):
        from supabase import create_client

        url = env("SUPABASE_URL")
        key = env("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set to use this backend")
        self._client = create_client(url, key)
        self._table = table

    def upsert(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        rows = [
            {"chunk_id": cid, "embedding": vec.tolist()}
            for cid, vec in zip(chunk_ids, embeddings)
        ]
        self._client.table(self._table).upsert(rows).execute()

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        # Requires a `match_chunks` Postgres function (pgvector cosine-distance
        # RPC) defined in the Supabase project; see module docstring for the
        # accompanying schema this assumes.
        resp = self._client.rpc(
            "match_chunks",
            {"query_embedding": query_vec.tolist(), "match_count": top_k},
        ).execute()
        return [(row["chunk_id"], float(row["similarity"])) for row in resp.data]
