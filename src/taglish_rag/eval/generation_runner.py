"""Generation eval: run naive RAG or the CRAG agent over the eval set,
score with the LLM judge, and report groundedness / correctness /
citation accuracy / refusal accuracy (the last computed directly from
question_type=="unanswerable", not from the judge, since refusal is a
simple behavioral check).

Requires GOOGLE_API_KEY in .env -- both the answer generation and the
judge call go through the Gemini backend, so this is the one eval stage
that cannot run offline. Every run stamps a `backend` field into its
result file recording which backend produced the numbers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from taglish_rag.agent.crag import run_crag, run_naive_rag
from taglish_rag.config import results_path
from taglish_rag.eval.runner import load_eval_set
from taglish_rag.generation.generator import get_generator
from taglish_rag.generation.judge import LLMJudge
from taglish_rag.retrieval.retriever import Retriever, RetrievalConfig

REFUSAL_MARKERS = ["don't know", "not answerable", "cannot answer", "no information", "not in the"]


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def run_generation_eval(use_agent: bool, limit: int | None = None, label: str = "") -> dict:
    generator = get_generator()
    backend = generator.backend
    # minilm-multilingual: smallest/fastest embedding model, so generation-eval
    # runs (which call the retriever once per eval item, then an LLM/judge call
    # on top) don't also pay bge-m3's heavier CPU cost. Swap once the ablation
    # sweep picks a winning embedding model for the "real" run.
    retriever = Retriever(RetrievalConfig(embedding_model="minilm-multilingual"))
    judge = LLMJudge(generator)

    items = load_eval_set()
    if limit:
        items = items[:limit]

    rows = []
    t0 = time.time()
    for item in items:
        if use_agent:
            state = run_crag(retriever, item.question, generator)
        else:
            state = run_naive_rag(retriever, item.question, generator)

        answer = state.get("answer", "")
        context = state.get("context", "")
        judge_score = judge.score(
            question=item.question,
            context=context,
            reference=item.gold_answer or "(unanswerable)",
            answer=answer,
        )
        predicted_refusal = _looks_like_refusal(answer)
        rows.append(
            {
                "qid": item.qid,
                "question_type": item.question_type,
                "language": item.language,
                "answer": answer,
                "groundedness": judge_score.groundedness,
                "correctness": judge_score.correctness,
                "citation_accuracy": judge_score.citation_accuracy,
                "judge_is_refusal": judge_score.is_refusal,
                "heuristic_is_refusal": predicted_refusal,
                "expected_refusal": item.question_type == "unanswerable",
                "judge_parse_ok": judge_score.parse_ok,
            }
        )
    elapsed = time.time() - t0

    def _mean(key):
        vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    unanswerable_rows = [r for r in rows if r["expected_refusal"]]
    refusal_accuracy = (
        sum(1 for r in unanswerable_rows if r["heuristic_is_refusal"]) / len(unanswerable_rows)
        if unanswerable_rows
        else float("nan")
    )

    summary = {
        "label": label or ("crag_agent" if use_agent else "naive_rag"),
        "backend": backend,
        "use_agent": use_agent,
        "n_items": len(rows),
        "elapsed_sec": elapsed,
        "groundedness": _mean("groundedness"),
        "correctness": _mean("correctness"),
        "citation_accuracy": _mean("citation_accuracy"),
        "refusal_accuracy_on_unanswerable": refusal_accuracy,
        "judge_parse_success_rate": sum(1 for r in rows if r["judge_parse_ok"]) / len(rows) if rows else 0,
        "rows": rows,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-agent", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    result = run_generation_eval(use_agent=args.use_agent, limit=args.limit, label=args.label)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))

    out_path = results_path(f"generation_{result['label']}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
