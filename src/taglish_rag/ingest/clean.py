"""Text cleaning: whitespace normalization + cross-document boilerplate
removal.

The site-scraped HTML docs (esp. Pag-IBIG) repeat the same ~40-line nav
menu and footer verbatim on every page. Left in, that boilerplate would
dominate chunk content and pollute retrieval (every chunk from the site
would look similar). Rather than hand-writing per-site strip rules, we
detect it generically: any line that recurs across most documents *within
the same agency* is treated as template chrome and dropped.
"""
from __future__ import annotations

import re
from collections import Counter

BOILERPLATE_FREQ_THRESHOLD = 0.4  # line appears in >=40% of an agency's docs
MIN_LINE_LEN_FOR_BOILERPLATE_CHECK = 3


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_boilerplate_lines(texts: list[str]) -> set[str]:
    if len(texts) < 3:
        return set()
    doc_line_sets = [set(t.split("\n")) for t in texts]
    counts = Counter(line for lines in doc_line_sets for line in lines)
    n_docs = len(texts)
    boilerplate = {
        line
        for line, count in counts.items()
        if len(line.strip()) >= MIN_LINE_LEN_FOR_BOILERPLATE_CHECK
        and count / n_docs >= BOILERPLATE_FREQ_THRESHOLD
    }
    return boilerplate


def strip_lines(text: str, lines_to_strip: set[str]) -> str:
    kept = [line for line in text.split("\n") if line not in lines_to_strip]
    return "\n".join(kept)


def clean_corpus(doc_texts: dict[str, str]) -> dict[str, str]:
    """doc_id -> raw text, returns doc_id -> cleaned text with boilerplate removed."""
    normalized = {doc_id: normalize_whitespace(t) for doc_id, t in doc_texts.items()}
    boilerplate = find_boilerplate_lines(list(normalized.values()))
    cleaned = {
        doc_id: normalize_whitespace(strip_lines(t, boilerplate))
        for doc_id, t in normalized.items()
    }
    return cleaned
