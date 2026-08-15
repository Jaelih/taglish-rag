"""Retrieval eval runner: given a RetrievalConfig, score it against
eval/taglish_rag_eval_v1.jsonl and emit a results JSON + markdown table.

Only items with a non-empty gold_doc_ids are scored for retrieval metrics
(factual, multi_hop, and ambiguous items with multiple acceptable docs).
Unanswerable items have no gold doc by design and are scored separately,
at the generation stage, on refusal accuracy.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from taglish_rag.config import eval_path, results_path
from taglish_rag.eval.metrics import mrr, ndcg_at_k, ranked_doc_ids, recall_at_k
from taglish_rag.retrieval.retriever import Retriever, RetrievalConfig
from taglish_rag.schemas import EvalItem

K_VALUES = [1, 5, 10]


def load_eval_set() -> list[EvalItem]:
    path = eval_path("taglish_rag_eval_v1.jsonl")
    with open(path, encoding="utf-8") as f:
        return [EvalItem(**json.loads(line)) for line in f if line.strip()]


def run_retrieval_eval(cfg: RetrievalConfig, label: str = "", limit: int | None = None) -> dict:
    retriever = Retriever(cfg)
    items = [i for i in load_eval_set() if i.gold_doc_ids]
    if limit:
        items = items[:limit]

    per_item = []
    t0 = time.time()
    for item in items:
        retrieved = retriever.retrieve(item.question)
        chunk_ids = [r.chunk_id for r in retrieved]
        ranked = ranked_doc_ids(chunk_ids, retriever.doc_id_by_chunk)
        row = {
            "qid": item.qid,
            "language": item.language,
            "question_type": item.question_type,
            "agency": item.agency,
        }
        for k in K_VALUES:
            row[f"recall@{k}"] = recall_at_k(ranked, item.gold_doc_ids, k)
        row["mrr"] = mrr(ranked, item.gold_doc_ids)
        row["ndcg@10"] = ndcg_at_k(ranked, item.gold_doc_ids, 10)
        per_item.append(row)
    elapsed = time.time() - t0

    def _mean(key: str, rows: list[dict]) -> float:
        vals = [r[key] for r in rows if r[key] == r[key]]  # filter NaN
        return sum(vals) / len(vals) if vals else float("nan")

    overall = {k: _mean(k, per_item) for k in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]}
    by_language = {}
    for lang in ["en", "tl", "taglish"]:
        rows = [r for r in per_item if r["language"] == lang]
        by_language[lang] = {k: _mean(k, rows) for k in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]}

    return {
        "label": label,
        "config": asdict(cfg),
        "n_items": len(items),
        "elapsed_sec": elapsed,
        "overall": overall,
        "by_language": by_language,
        "per_item": per_item,
    }


def to_markdown_row(label: str, result: dict) -> str:
    o = result["overall"]
    return (
        f"| {label} | {o['recall@1']:.3f} | {o['recall@5']:.3f} | {o['recall@10']:.3f} "
        f"| {o['mrr']:.3f} | {o['ndcg@10']:.3f} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-model", default="bge-m3")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--hybrid-alpha", type=float, default=0.5)
    ap.add_argument("--use-reranker", action="store_true")
    ap.add_argument("--translate-query", action="store_true")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = RetrievalConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.embedding_model,
        mode=args.mode,
        hybrid_alpha=args.hybrid_alpha,
        use_reranker=args.use_reranker,
        translate_query_to_english=args.translate_query,
    )
    result = run_retrieval_eval(cfg, label=args.label, limit=args.limit)

    print(f"\n=== {args.label} ===")
    print(json.dumps(result["overall"], indent=2))
    print("\nBy language:")
    print(json.dumps(result["by_language"], indent=2))

    out_path = Path(args.out) if args.out else results_path(f"retrieval_{args.label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
