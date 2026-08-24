"""Generation eval: run naive RAG or the CRAG agent over the eval set,
score with the LLM judge, and report groundedness / correctness /
citation accuracy / refusal accuracy (the last computed directly from
question_type=="unanswerable", not from the judge, since refusal is a
simple behavioral check).

Requires GOOGLE_API_KEY in .env -- both the answer generation and the
judge call go through the Gemini backend, so this is the one eval stage
that cannot run offline. Every run stamps a `backend` field into its
result file recording which backend produced it.

Each item costs two Gemini calls (answer + judge), so a full 90-item run
is 180 requests. Against the free tier's 15 RPM that is ~12 minutes of
mostly waiting, which is why this module prints per-item progress with an
ETA and checkpoints every finished row to
`results/generation_<label>.partial.jsonl` -- an interrupted run resumes
instead of throwing away the calls it already paid for.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from taglish_rag.agent.crag import run_crag, run_naive_rag
from taglish_rag.agent.grader import LLMGrader
from taglish_rag.config import results_path
from taglish_rag.eval.runner import load_eval_set
from taglish_rag.generation.generator import get_generator
from taglish_rag.generation.judge import LLMJudge
from taglish_rag.retrieval.retriever import Retriever, RetrievalConfig

REFUSAL_MARKERS = ["don't know", "not answerable", "cannot answer", "no information", "not in the"]


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _resolve_label(use_agent: bool, label: str | None) -> str:
    return label or ("crag_agent" if use_agent else "naive_rag")


def _partial_path(label: str) -> Path:
    return results_path(f"generation_{label}.partial.jsonl")


def _load_checkpoint(path: Path) -> list[dict]:
    """Rows from a previous interrupted run. A truncated trailing line (a run
    killed mid-write) is dropped rather than crashing the restart."""
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[checkpoint] skipping malformed line in {path.name}", flush=True)
    return rows


def _append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _fmt_score(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "--"


def _summarize(
    rows: list[dict], label: str, backend: str, use_agent: bool, elapsed: float, resumed: bool
) -> dict:
    def _mean(key):
        vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    unanswerable_rows = [r for r in rows if r["expected_refusal"]]
    refusal_accuracy = (
        sum(1 for r in unanswerable_rows if r["heuristic_is_refusal"]) / len(unanswerable_rows)
        if unanswerable_rows
        else float("nan")
    )

    return {
        "label": label,
        "backend": backend,
        "use_agent": use_agent,
        "n_items": len(rows),
        # Wall time for THIS process only. On a resumed run it excludes the
        # items restored from the checkpoint, so don't read it as an
        # end-to-end timing of the full set.
        "elapsed_sec": elapsed,
        "resumed_from_checkpoint": resumed,
        "mean_latency_ms": _mean("latency_ms"),
        "groundedness": _mean("groundedness"),
        "correctness": _mean("correctness"),
        "citation_accuracy": _mean("citation_accuracy"),
        "refusal_accuracy_on_unanswerable": refusal_accuracy,
        "judge_parse_success_rate": sum(1 for r in rows if r["judge_parse_ok"]) / len(rows) if rows else 0,
        "rows": rows,
    }


def run_generation_eval(
    use_agent: bool,
    limit: int | None = None,
    label: str | None = "",
    resume: bool = True,
) -> dict:
    label = _resolve_label(use_agent, label)
    generator = get_generator()
    backend = generator.backend
    # minilm-multilingual: smallest/fastest embedding model, so generation-eval
    # runs (which call the retriever once per eval item, then an LLM/judge call
    # on top) don't also pay bge-m3's heavier CPU cost. Swap once the ablation
    # sweep picks a winning embedding model for the "real" run.
    retriever = Retriever(RetrievalConfig(embedding_model="minilm-multilingual"))
    judge = LLMJudge(generator)
    grader = LLMGrader(generator)

    items = load_eval_set()
    if limit:
        items = items[:limit]

    partial = _partial_path(label)
    rows: list[dict] = []
    if resume:
        rows = _load_checkpoint(partial)
    elif partial.exists():
        partial.unlink()

    done_qids = {r["qid"] for r in rows}
    resumed = bool(done_qids)
    if resumed:
        print(
            f"[checkpoint] resuming from {partial.name}: {len(done_qids)} item(s) already scored",
            flush=True,
        )

    pending = [item for item in items if item.qid not in done_qids]
    total = len(items)
    print(
        f"Generation eval: label={label} use_agent={use_agent} "
        f"items={total} pending={len(pending)} (~{len(pending) * 2} Gemini calls)",
        flush=True,
    )

    t0 = time.time()
    completed = 0
    try:
        for item in pending:
            completed += 1
            position = len(done_qids) + completed
            prefix = f"[{position:3d}/{total}] {item.qid} ({item.language}/{item.question_type})"
            print(f"{prefix} generating...", flush=True)

            item_t0 = time.time()
            if use_agent:
                state = run_crag(retriever, item.question, generator, grader)
            else:
                state = run_naive_rag(retriever, item.question, generator)
            latency_ms = (time.time() - item_t0) * 1000

            answer = state.get("answer", "")
            context = state.get("context", "")
            print(f"{prefix} judging...", flush=True)
            judge_score = judge.score(
                question=item.question,
                context=context,
                reference=item.gold_answer or "(unanswerable)",
                answer=answer,
            )
            predicted_refusal = _looks_like_refusal(answer)
            row = {
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
                # Wall time for retrieve(+grade+rewrite)+generate(+verify) only --
                # excludes the judge call below, so this is the fair CRAG-vs-naive
                # comparison (the judge costs the same regardless of which path
                # produced the answer).
                "latency_ms": latency_ms,
                # CRAG trace -- naive RAG never sets these (no grading/rewrite/
                # verify step), so they come back None on naive rows rather
                # than a misleading default.
                "grade": state.get("grade"),
                "verdict": state.get("verdict"),
                "corpus_gap": state.get("corpus_gap"),
                "grade_missing": state.get("grade_missing"),
                "rewrite_count": state.get("rewrite_count"),
                "citations_verified": state.get("citations_verified"),
            }
            rows.append(row)
            _append_checkpoint(partial, row)

            elapsed = time.time() - t0
            eta = (elapsed / completed) * (len(pending) - completed)
            print(
                f"{prefix} done  g={_fmt_score(row['groundedness'])} "
                f"c={_fmt_score(row['correctness'])} cite={_fmt_score(row['citation_accuracy'])}"
                f"  latency={latency_ms:.0f}ms"
                f"  | elapsed {_fmt_duration(elapsed)} eta ~{_fmt_duration(eta)}",
                flush=True,
            )
    except BaseException as exc:
        # Everything finished so far is already on disk; say so, then re-raise
        # so the caller still sees the real failure (or the Ctrl-C).
        print(
            f"\n[checkpoint] run stopped after {len(rows)}/{total} items "
            f"({type(exc).__name__}). Progress saved to {partial} -- re-run the "
            f"same command to resume.",
            flush=True,
        )
        raise

    return _summarize(rows, label, backend, use_agent, time.time() - t0, resumed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-agent", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore and discard any checkpoint from a previous interrupted run",
    )
    args = ap.parse_args()

    result = run_generation_eval(
        use_agent=args.use_agent,
        limit=args.limit,
        label=args.label,
        resume=not args.no_resume,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))

    out_path = results_path(f"generation_{result['label']}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")

    partial = _partial_path(result["label"])
    if partial.exists():
        partial.unlink()  # final results are durable now, the checkpoint is redundant


if __name__ == "__main__":
    main()
