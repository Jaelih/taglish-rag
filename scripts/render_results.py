"""Render results/ablation_summary.json (+ results/retrieval_run.json) into
markdown tables and inject them into README.md in place of the two
placeholder comments.

Idempotent: each placeholder marker is followed by a sentinel-wrapped block
(<!-- BEGIN:GENERATED ... --> / <!-- END:GENERATED --> ); re-running replaces
that block instead of appending a new one, and the marker comment itself is
never removed.

Usage:
    uv run python scripts/render_results.py
    uv run python scripts/render_results.py --check   # exit 1 if README is stale
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "ablation_summary.json"
BASELINE_PATH = ROOT / "results" / "retrieval_run.json"
README_PATH = ROOT / "README.md"

RESULTS_MARKER = "<!-- RESULTS_TABLE_PLACEHOLDER -->"
ABLATION_MARKER = "<!-- ABLATION_TABLE_PLACEHOLDER -->"

AXIS_TITLES = {
    "embedding_model": "Embedding model",
    "chunking": "Chunk size / overlap",
    "retrieval_mode": "Retrieval mode",
    "use_reranker": "Cross-encoder reranker",
    "translate_query_to_english": "Query translation to English",
}

# Axes not yet present in ablation_summary.json (needs a configured LLM
# generator backend to run) -- listed explicitly rather than silently omitted.
PENDING_AXES = {
    "naive_vs_crag": "Naive RAG vs. LangGraph self-correcting CRAG loop",
}

# Figures made by scripts/make_figures.py, embedded next to the axis table
# they visualize. Paths are README-relative (repo root).
FIGURE_FOR_AXIS = {
    "retrieval_mode": (
        "results/figures/01_retrieval_mode_recall5.png",
        "Recall@5 by retrieval mode, by query language",
    ),
    "embedding_model": (
        "results/figures/02_embedding_model_recall1.png",
        "Recall@1 by embedding model, by query language",
    ),
    "chunking": (
        "results/figures/03_chunk_size_recall1.png",
        "Recall@1 by chunk size / overlap, by query language",
    ),
}

METRIC_COLS = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]
METRIC_HEADERS = ["Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@10"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt(x: float) -> str:
    return f"{x:.3f}"


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path} -- run the retrieval eval / ablation sweep first")
    return json.loads(path.read_text(encoding="utf-8"))


def wrap(marker: str, body: str) -> str:
    return f"{marker}\n<!-- BEGIN:GENERATED -->\n{body.strip()}\n<!-- END:GENERATED -->"


def render_results_table(summary: dict, baseline: dict) -> str:
    # "Best measured" = the single highest-recall@1 config across every
    # ablation run (each run varies exactly one axis off the shared
    # baseline config) -- not a jointly-tuned combination that was never
    # actually measured together.
    best_entry, best_axis = None, None
    for axis, entries in summary.items():
        for entry in entries:
            if best_entry is None or entry["overall"]["recall@1"] > best_entry["overall"]["recall@1"]:
                best_entry, best_axis = entry, axis

    bm25 = next(e for e in summary["retrieval_mode"] if e["label"] == "retrieval_mode_bm25")
    bm25_en = bm25["by_language"]["en"]["recall@1"]
    bm25_tl = bm25["by_language"]["tl"]["recall@1"]

    lines = []
    lines.append(
        f"**Headline finding:** BM25-only retrieval (the default in most RAG tutorials) hits "
        f"the right document on the first try {pct(bm25_en)} of the time for English questions "
        f"— and just {pct(bm25_tl)} of the time for the *same* question asked in Tagalog, "
        f"against this English-only government corpus."
    )
    lines.append("")
    lines.append(
        "| Config | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Recall@1 (EN / TL / Taglish) |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    def row(label: str, entry_overall: dict, entry_by_lang: dict) -> str:
        by_lang = " / ".join(pct(entry_by_lang[l]["recall@1"]) for l in ("en", "tl", "taglish"))
        return (
            f"| {label} | {pct(entry_overall['recall@1'])} | {pct(entry_overall['recall@5'])} | "
            f"{pct(entry_overall['recall@10'])} | {fmt(entry_overall['mrr'])} | "
            f"{fmt(entry_overall['ndcg@10'])} | {by_lang} |"
        )

    lines.append(
        row(
            f"Baseline (bge-m3, hybrid, chunk {baseline['config']['chunk_size']}/"
            f"{baseline['config']['overlap']}, no reranker)",
            baseline["overall"],
            baseline["by_language"],
        )
    )
    lines.append(
        row(
            f"**Best measured** ({best_entry['label'].removeprefix(best_axis + '_')} on the "
            f"{AXIS_TITLES.get(best_axis, best_axis)} axis, other settings at baseline)",
            best_entry["overall"],
            best_entry["by_language"],
        )
    )
    lines.append("")
    lines.append(
        "Retrieval-only numbers (n=72 retrieval-eligible items out of the 90-item eval set; "
        "the 18 deliberately unanswerable items have no gold document and are scored separately "
        "as refusal accuracy, see Limitations). Generation-stage metrics (groundedness, answer "
        "correctness, citation accuracy, refusal accuracy, judge-vs-human κ) require a configured "
        "`GOOGLE_API_KEY` and are not included here -- see Limitations."
    )
    return "\n".join(lines)


def render_ablation_table(summary: dict) -> str:
    sections = []
    for axis, entries in summary.items():
        title = AXIS_TITLES.get(axis, axis)
        lines = [f"### {title}", ""]
        lines.append(
            "| Config | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Recall@1 (EN / TL / Taglish) | n |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for entry in entries:
            label = entry["label"].removeprefix(axis + "_")
            overall, by_lang = entry["overall"], entry["by_language"]
            by_lang_str = " / ".join(pct(by_lang[l]["recall@1"]) for l in ("en", "tl", "taglish"))
            lines.append(
                f"| {label} | {pct(overall['recall@1'])} | {pct(overall['recall@5'])} | "
                f"{pct(overall['recall@10'])} | {fmt(overall['mrr'])} | {fmt(overall['ndcg@10'])} | "
                f"{by_lang_str} | {entry['n_items']} |"
            )

        if axis in FIGURE_FOR_AXIS:
            rel_path, caption = FIGURE_FOR_AXIS[axis]
            if not (ROOT / rel_path).exists():
                raise SystemExit(f"missing {rel_path} -- run scripts/make_figures.py first")
            lines.append("")
            lines.append(f"![{caption}]({rel_path})")

        sections.append("\n".join(lines))

    pending_lines = ["### Not yet run", ""]
    for _, desc in PENDING_AXES.items():
        pending_lines.append(f"- **{desc}** -- blocked on a configured `GOOGLE_API_KEY` (see `docs/DEPLOY.md`).")
    sections.append("\n".join(pending_lines))

    return "\n\n".join(sections)


def replace_block(readme: str, marker: str, body: str) -> str:
    new_block = wrap(marker, body)
    start = readme.find(marker)
    if start == -1:
        raise SystemExit(f"marker not found in README.md: {marker}")

    begin_sentinel = "<!-- BEGIN:GENERATED -->"
    end_sentinel = "<!-- END:GENERATED -->"
    after_marker = start + len(marker)
    begin_idx = readme.find(begin_sentinel, after_marker)
    end_idx = readme.find(end_sentinel, after_marker)

    # Only treat an existing sentinel block as "ours" if it immediately
    # (modulo whitespace) follows this marker, not some later marker's block.
    between = readme[after_marker:begin_idx].strip() if begin_idx != -1 else None
    if begin_idx != -1 and end_idx != -1 and between == "":
        return readme[:start] + new_block + readme[end_idx + len(end_sentinel):]
    return readme[:start] + new_block + readme[after_marker:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check",
        action="store_true",
        help="don't write README.md -- exit 1 if regenerating would change it",
    )
    args = ap.parse_args()

    summary = load(SUMMARY_PATH)
    baseline = load(BASELINE_PATH)

    readme = README_PATH.read_text(encoding="utf-8")
    readme = replace_block(readme, RESULTS_MARKER, render_results_table(summary, baseline))
    readme = replace_block(readme, ABLATION_MARKER, render_ablation_table(summary))

    if args.check:
        current = README_PATH.read_text(encoding="utf-8")
        if readme != current:
            raise SystemExit("README.md is stale -- run scripts/render_results.py")
        print("README.md results tables are up to date.")
        return

    README_PATH.write_text(readme, encoding="utf-8")
    print(f"Updated {README_PATH.relative_to(ROOT)} from {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
