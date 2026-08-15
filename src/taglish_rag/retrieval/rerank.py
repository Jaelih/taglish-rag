"""Cross-encoder reranking (BAAI/bge-reranker-v2-m3), on/off ablation axis."""
from __future__ import annotations

from taglish_rag.config import load_yaml

_reranker_cache = {}


def _get_reranker():
    from sentence_transformers import CrossEncoder

    cfg = load_yaml("embeddings.yaml")
    hf_id = cfg["reranker"]["hf_id"]
    if hf_id not in _reranker_cache:
        _reranker_cache[hf_id] = CrossEncoder(hf_id)
    return _reranker_cache[hf_id]


def rerank(query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
    """candidates: [(chunk_id, chunk_text)]. Returns top_k (chunk_id, score) by relevance."""
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, text) for _, text in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip([cid for cid, _ in candidates], scores), key=lambda x: x[1], reverse=True)
    return [(cid, float(s)) for cid, s in ranked[:top_k]]
