"""Sparse (BM25) retrieval. Critical for Taglish: catches Filipino terms
that multilingual dense embeddings under-weight relative to English."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9ñÑ]+", text.lower())


class BM25Index:
    def __init__(self, chunk_ids: list[str], texts: list[str]):
        self.chunk_ids = chunk_ids
        self._tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
