"""FAISS dense index + score-fusion hybrid retrieval (dense + BM25)."""
from __future__ import annotations

import numpy as np


class DenseIndex:
    """Cosine-similarity search via inner product over L2-normalized vectors."""

    def __init__(self, chunk_ids: list[str], embeddings: np.ndarray):
        import faiss

        self.chunk_ids = chunk_ids
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        scores, idxs = self._index.search(query_vec.reshape(1, -1), top_k)
        return [
            (self.chunk_ids[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1
        ]


def _min_max_normalize(scored: list[tuple[str, float]]) -> dict[str, float]:
    if not scored:
        return {}
    values = [s for _, s in scored]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {cid: 1.0 for cid, _ in scored}
    return {cid: (s - lo) / (hi - lo) for cid, s in scored}


def fuse_hybrid(
    dense_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    alpha: float,
    top_k: int,
) -> list[tuple[str, float]]:
    """alpha weights the dense score; (1 - alpha) weights BM25. Both score
    lists are min-max normalized to [0, 1] first since they're on
    incomparable scales (cosine similarity vs. BM25's unbounded scores)."""
    dense_norm = _min_max_normalize(dense_results)
    bm25_norm = _min_max_normalize(bm25_results)
    all_ids = set(dense_norm) | set(bm25_norm)
    fused = {
        cid: alpha * dense_norm.get(cid, 0.0) + (1 - alpha) * bm25_norm.get(cid, 0.0)
        for cid in all_ids
    }
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
