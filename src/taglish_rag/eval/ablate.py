"""Ablation sweep: for each axis in configs/ablations.yaml, vary one
retrieval-config field at a time (holding the rest at `baseline`), score
each variant with the retrieval eval runner, and emit a single results
table + per-axis breakdown.

This is what produces the README's headline ablation table.

Results are written to results/ablation_summary.json incrementally, after
every individual run -- not just once at the end -- so a crash partway
through the sweep (e.g. one axis's model has a bug) doesn't discard the
runs that already completed. Re-running resumes from where it left off
rather than recomputing everything.
"""
from __future__ import annotations

import argparse
import json

from taglish_rag.config import load_yaml, results_path
from taglish_rag.eval.runner import run_retrieval_eval
from taglish_rag.retrieval.retriever import RetrievalConfig

SUMMARY_PATH = results_path("ablation_summary.json")


def _cfg_from_baseline(baseline: dict, overrides: dict) -> RetrievalConfig:
    fields = dict(
        chunk_size=baseline["chunk_size"],
        overlap=baseline["overlap"],
        embedding_model=baseline["embedding_model"],
        mode=baseline["retrieval_mode"],
        use_reranker=baseline["use_reranker"],
        translate_query_to_english=baseline["translate_query_to_english"],
    )
    for k, v in overrides.items():
        if k == "retrieval_mode":
            fields["mode"] = v
        elif k in ("chunk_size", "overlap"):
            fields[k] = v
        elif k == "embedding_model":
            fields["embedding_model"] = v
        elif k == "use_reranker":
            fields["use_reranker"] = v
        elif k == "translate_query_to_english":
            fields["translate_query_to_english"] = v
    return RetrievalConfig(**fields)


def _load_summary() -> dict:
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_summary(summary: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_axis(axis_name: str, values: list, baseline: dict, summary: dict, force: bool) -> None:
    done_labels = {r["label"] for r in summary.get(axis_name, [])}
    for value in values:
        if axis_name == "chunking":
            overrides = {"chunk_size": value["chunk_size"], "overlap": value["overlap"]}
            label = f"chunking_{value['chunk_size']}_{value['overlap']}"
        else:
            overrides = {axis_name: value}
            label = f"{axis_name}_{value}"

        if label in done_labels and not force:
            print(f"\n--- Skipping {label} (already in {SUMMARY_PATH.name}, use --force to redo) ---")
            continue

        cfg = _cfg_from_baseline(baseline, overrides)
        print(f"\n--- Running {label} ---")
        result = run_retrieval_eval(cfg, label=label)
        o = result["overall"]
        print(f"  R@1={o['recall@1']:.3f}  R@5={o['recall@5']:.3f}  R@10={o['recall@10']:.3f}  MRR={o['mrr']:.3f}  nDCG@10={o['ndcg@10']:.3f}")

        row = {"label": label, "overall": o, "by_language": result["by_language"], "n_items": result["n_items"]}
        summary.setdefault(axis_name, [])
        summary[axis_name] = [r for r in summary[axis_name] if r["label"] != label] + [row]
        _save_summary(summary)  # persist after every run, not just every axis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default=None, help="Run only this axis (default: all axes in configs/ablations.yaml)")
    ap.add_argument("--force", action="store_true", help="Re-run configs already present in ablation_summary.json")
    args = ap.parse_args()

    cfg = load_yaml("ablations.yaml")
    baseline = cfg["baseline"]
    summary = _load_summary()

    for axis in cfg["axes"]:
        name = axis["name"]
        if args.axis and name != args.axis:
            continue
        if name == "use_agent":
            print(f"Skipping axis '{name}' (naive RAG vs. CRAG agent is scored in the generation eval, not here)")
            continue
        run_axis(name, axis["values"], baseline, summary, args.force)

    print(f"\nAblation summary -> {SUMMARY_PATH}")
    print("\n\n=== ABLATION SUMMARY (Recall@5 / MRR / nDCG@10) ===")
    for axis, results in summary.items():
        print(f"\n## {axis}")
        for r in results:
            o = r["overall"]
            print(f"  {r['label']:35s} R@5={o['recall@5']:.3f}  MRR={o['mrr']:.3f}  nDCG@10={o['ndcg@10']:.3f}")


if __name__ == "__main__":
    main()
