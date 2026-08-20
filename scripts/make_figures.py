"""Render results/ablation_summary.json into the 2-3 headline figures the
project plan calls for, as static PNGs for the README / writeup.

Palette, mark specs, and label rules follow the project's dataviz skill:
fixed categorical hue order (en/tl/taglish -> blue/orange/aqua, the
palette's first three slots, validated all-pairs colorblind-safe), one
axis, a legend for every multi-series chart, muted ink for text, hairline
recessive gridlines.

Usage:
    uv run python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "ablation_summary.json"
OUT_DIR = ROOT / "results" / "figures"

# -- palette (references/palette.md: categorical slots 1-3, order fixed) --
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

LANG_ORDER = ["en", "tl", "taglish"]
LANG_LABEL = {"en": "English", "tl": "Tagalog", "taglish": "Taglish"}
LANG_COLOR = {"en": "#2a78d6", "tl": "#eb6834", "taglish": "#1baf7a"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"missing {SUMMARY_PATH} -- run the ablation sweep first")
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _style_axes(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_SECONDARY)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", length=0, labelsize=9.5)


def _title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="semibold", color=INK_PRIMARY, loc="left", pad=20)
    ax.text(
        0.0, 1.03, subtitle, transform=ax.transAxes,
        fontsize=9.5, color=INK_MUTED, ha="left", va="bottom",
    )


def _grouped_bar(ax, categories: list[str], series: dict[str, list[float]]) -> None:
    n_groups, n_series = len(categories), len(LANG_ORDER)
    group_width = 0.72
    bar_width = group_width / n_series
    x = list(range(n_groups))

    for i, lang in enumerate(LANG_ORDER):
        offsets = [xi - group_width / 2 + bar_width * (i + 0.5) for xi in x]
        bars = ax.bar(
            offsets, series[lang], width=bar_width * 0.88,
            color=LANG_COLOR[lang], label=LANG_LABEL[lang], zorder=3,
        )
        for rect, val in zip(bars, series[lang]):
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015,
                f"{val * 100:.0f}", ha="center", va="bottom",
                fontsize=8, color=INK_SECONDARY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, color=INK_PRIMARY)
    ax.set_ylim(0, 1.08)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
        frameon=False, fontsize=9.5, handlelength=1.0, handleheight=1.0,
        columnspacing=1.4,
    )


def fig_retrieval_mode(summary: dict) -> Path:
    entries = {e["label"].removeprefix("retrieval_mode_"): e for e in summary["retrieval_mode"]}
    order = ["dense", "bm25", "hybrid"]
    labels = {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid"}

    series = {lang: [entries[m]["by_language"][lang]["recall@5"] for m in order] for lang in LANG_ORDER}

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    _title(
        ax,
        "Hybrid retrieval recovers most of BM25's Tagalog collapse",
        "Recall@5 by retrieval mode, by query language (n=72 retrieval-eligible items)",
    )
    _grouped_bar(ax, [labels[m] for m in order], series)
    _style_axes(ax, "Recall@5")

    out = OUT_DIR / "01_retrieval_mode_recall5.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_embedding_model(summary: dict) -> Path:
    entries = {e["label"].removeprefix("embedding_model_"): e for e in summary["embedding_model"]}
    order = ["bge-m3", "multilingual-e5-large", "minilm-multilingual"]
    labels = {"bge-m3": "bge-m3", "multilingual-e5-large": "e5-large", "minilm-multilingual": "MiniLM"}

    series = {lang: [entries[m]["by_language"][lang]["recall@1"] for m in order] for lang in LANG_ORDER}

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    _title(
        ax,
        "The English-Tagalog gap holds across every embedding model",
        "Recall@1 by embedding model, by query language (n=72 retrieval-eligible items)",
    )
    _grouped_bar(ax, [labels[m] for m in order], series)
    _style_axes(ax, "Recall@1")

    out = OUT_DIR / "02_embedding_model_recall1.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_chunk_size(summary: dict) -> Path:
    entries = {e["label"].removeprefix("chunking_"): e for e in summary["chunking"]}
    order = ["256_32", "512_64", "1024_128"]
    labels = {"256_32": "256 / 32", "512_64": "512 / 64", "1024_128": "1024 / 128"}
    x = list(range(len(order)))

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    _title(
        ax,
        "Larger chunks retrieve better -- most for Tagalog",
        "Recall@1 by chunk size / overlap (tokens), by query language (n=72 retrieval-eligible items)",
    )

    for lang in LANG_ORDER:
        y = [entries[m]["by_language"][lang]["recall@1"] for m in order]
        ax.plot(
            x, y, color=LANG_COLOR[lang], linewidth=2, marker="o",
            markersize=8, markerfacecolor=LANG_COLOR[lang],
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
            label=LANG_LABEL[lang],
        )
        ax.annotate(
            f"{y[-1] * 100:.0f}%", xy=(x[-1], y[-1]), xytext=(8, 0),
            textcoords="offset points", va="center", fontsize=9.5,
            color=INK_SECONDARY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in order], fontsize=10, color=INK_PRIMARY)
    ax.set_xlim(-0.15, len(order) - 1 + 0.32)
    ax.set_ylim(0, 1.0)
    _style_axes(ax, "Recall@1")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
        frameon=False, fontsize=9.5, handlelength=1.4, columnspacing=1.4,
    )

    out = OUT_DIR / "03_chunk_size_recall1.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    paths = [fig_retrieval_mode(summary), fig_embedding_model(summary), fig_chunk_size(summary)]
    for p in paths:
        print(f"Wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
