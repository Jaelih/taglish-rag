"""Retrieval metrics at document granularity (see ADR in retriever.py
docstring: gold labels cite doc_id, not chunk_id, so these metrics stay
valid across the chunk-size ablation sweep)."""
from __future__ import annotations

import math


def ranked_doc_ids(chunk_ids: list[str], doc_id_by_chunk: dict[str, str]) -> list[str]:
    """Dedup chunk hits into a doc-id ranking, keeping each doc's best (first) rank."""
    seen = []
    for cid in chunk_ids:
        doc_id = doc_id_by_chunk.get(cid)
        if doc_id and doc_id not in seen:
            seen.append(doc_id)
    return seen


def recall_at_k(ranked_docs: list[str], gold_docs: list[str], k: int) -> float:
    if not gold_docs:
        return float("nan")
    top_k = set(ranked_docs[:k])
    hit = len(top_k & set(gold_docs))
    return hit / len(gold_docs)


def mrr(ranked_docs: list[str], gold_docs: list[str]) -> float:
    if not gold_docs:
        return float("nan")
    gold_set = set(gold_docs)
    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_docs: list[str], gold_docs: list[str], k: int) -> float:
    if not gold_docs:
        return float("nan")
    gold_set = set(gold_docs)
    dcg = 0.0
    for i, doc_id in enumerate(ranked_docs[:k], start=1):
        if doc_id in gold_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
