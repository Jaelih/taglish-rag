"""Extract raw text from a downloaded source document (pdf/html-as-text)."""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
    except Exception as e:  # corrupt/encrypted PDF
        print(f"  WARN: failed to open {path}: {e}")
        return ""
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def extract_text_file(path: Path) -> str:
    """Our .txt scrapes are saved as `Title: ...\\nURL: ...\\n---\\n<body>`."""
    raw = path.read_text(encoding="utf-8")
    return raw.split("---\n", 1)[-1] if "---\n" in raw else raw


def extract(doc_type: str, path: Path) -> str:
    if doc_type == "pdf":
        return extract_pdf_text(path)
    if doc_type in ("html", "txt"):
        return extract_text_file(path)
    raise ValueError(f"Unknown doc_type: {doc_type}")
