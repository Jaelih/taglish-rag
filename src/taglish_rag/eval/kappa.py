"""Judge-vs-human agreement (Cohen's kappa) on the 30-item hand-labeled
sample (eval/human_labels/sample_qids.txt), per plan section 2: "hand-label
30 items yourself and report judge-human agreement... stating the judge's
limitations out loud is a maturity signal."

Usage:
  1. Run generation eval to produce results/generation_<label>.json (needs
     GOOGLE_API_KEY set in .env).
  2. Fill in eval/human_labels/human_labels.jsonl: for each sampled qid, add
     {"qid": ..., "human_acceptable": true/false} based on your own read of
     the system answer in the generation results file.
  3. Run this script to compute kappa between judge_acceptable (judge's
     correctness score thresholded at >=0.5) and human_acceptable.
"""
from __future__ import annotations

import argparse
import json

from taglish_rag.config import REPO_ROOT, results_path

SAMPLE_QIDS_PATH = REPO_ROOT / "eval" / "human_labels" / "sample_qids.txt"
HUMAN_LABELS_PATH = REPO_ROOT / "eval" / "human_labels" / "human_labels.jsonl"


def load_sample_qids() -> list[str]:
    return [
        line.strip()
        for line in SAMPLE_QIDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_human_labels() -> dict[str, bool]:
    if not HUMAN_LABELS_PATH.exists():
        return {}
    labels = {}
    for line in HUMAN_LABELS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            labels[d["qid"]] = d["human_acceptable"]
    return labels


def cohen_kappa(judge_labels: list[bool], human_labels: list[bool]) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(judge_labels, human_labels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation-result", required=True, help="Path to results/generation_*.json")
    args = ap.parse_args()

    with open(args.generation_result, encoding="utf-8") as f:
        gen_result = json.load(f)
    rows_by_qid = {r["qid"]: r for r in gen_result["rows"]}

    sample_qids = load_sample_qids()
    human_labels = load_human_labels()

    missing_human = [q for q in sample_qids if q not in human_labels]
    if missing_human:
        print(f"Missing human labels for {len(missing_human)} qids. Fill in {HUMAN_LABELS_PATH}:")
        for q in missing_human:
            print(f"  {q}")
        return

    judge_bools, human_bools = [], []
    for qid in sample_qids:
        row = rows_by_qid.get(qid)
        if row is None or row.get("correctness") is None:
            continue
        judge_bools.append(row["correctness"] >= 0.5)
        human_bools.append(human_labels[qid])

    if len(judge_bools) < 2:
        print("Not enough scored items to compute kappa.")
        return

    kappa = cohen_kappa(judge_bools, human_bools)
    print(f"Cohen's kappa (judge vs. human, n={len(judge_bools)}): {kappa:.3f}")
    print(f"Judge said 'acceptable' on {sum(judge_bools)}/{len(judge_bools)}")
    print(f"Human said 'acceptable' on {sum(human_bools)}/{len(human_bools)}")


if __name__ == "__main__":
    main()
