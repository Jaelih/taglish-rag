"""End-to-end ingestion: manifests -> extract -> clean -> chunk -> jsonl.

Text extraction (esp. from large PDFs) is cached per doc_id under
data/processed/doc_text_raw and data/processed/doc_text_clean, so re-running
the chunk sweep (many chunk_size/overlap combos, see configs/chunking.yaml)
doesn't re-parse PDFs each time.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from taglish_rag.config import REPO_ROOT, data_path, load_yaml
from taglish_rag.ingest.chunk import chunk_text
from taglish_rag.ingest.clean import clean_corpus
from taglish_rag.ingest.extract import extract

RAW_DIR = data_path("raw")
PROCESSED_DIR = data_path("processed")
RAW_TEXT_CACHE = PROCESSED_DIR / "doc_text_raw"
CLEAN_TEXT_CACHE = PROCESSED_DIR / "doc_text_clean"
AGENCIES = ["bir", "philhealth", "pagibig"]


def load_manifest() -> list[dict]:
    docs = []
    for agency in AGENCIES:
        manifest_path = RAW_DIR / agency / "manifest.jsonl"
        if not manifest_path.exists():
            continue
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                docs.append(json.loads(line))
    return docs


def extract_all(docs: list[dict]) -> dict[str, str]:
    RAW_TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    texts = {}
    for doc in docs:
        cache_path = RAW_TEXT_CACHE / f"{doc['doc_id']}.txt"
        if cache_path.exists():
            texts[doc["doc_id"]] = cache_path.read_text(encoding="utf-8")
            continue
        local_path = REPO_ROOT / doc["local_path"]
        text = extract(doc["doc_type"], local_path)
        cache_path.write_text(text, encoding="utf-8")
        texts[doc["doc_id"]] = text
    return texts


def clean_all(docs: list[dict], raw_texts: dict[str, str]) -> dict[str, str]:
    CLEAN_TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    by_agency: dict[str, dict[str, str]] = defaultdict(dict)
    for doc in docs:
        by_agency[doc["agency"]][doc["doc_id"]] = raw_texts.get(doc["doc_id"], "")

    cleaned: dict[str, str] = {}
    for agency, agency_texts in by_agency.items():
        cleaned.update(clean_corpus(agency_texts))

    for doc_id, text in cleaned.items():
        (CLEAN_TEXT_CACHE / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    return cleaned


def build_chunks(
    docs: list[dict],
    cleaned_texts: dict[str, str],
    chunk_size: int,
    overlap: int,
    tokenizer: str,
) -> list[dict]:
    records = []
    for doc in docs:
        text = cleaned_texts.get(doc["doc_id"], "")
        if not text.strip():
            continue
        pieces = chunk_text(text, chunk_size, overlap, tokenizer)
        for i, piece in enumerate(pieces):
            records.append(
                {
                    "chunk_id": f"{doc['doc_id']}__{i}",
                    "doc_id": doc["doc_id"],
                    "agency": doc["agency"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "text": piece,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "position": i,
                }
            )
    return records


def run_ingest(chunk_size: int, overlap: int, tokenizer: str = "cl100k_base") -> Path:
    docs = load_manifest()
    if not docs:
        raise RuntimeError("No manifests found under data/raw/*/manifest.jsonl. Run the scrapers first.")
    print(f"Loaded {len(docs)} source docs across {AGENCIES}")

    raw_texts = extract_all(docs)
    print(f"Extracted text for {len(raw_texts)} docs")

    cleaned_texts = clean_all(docs, raw_texts)
    print("Cleaned (boilerplate-stripped) text cached")

    chunks = build_chunks(docs, cleaned_texts, chunk_size, overlap, tokenizer)
    out_path = PROCESSED_DIR / f"chunks_{chunk_size}_{overlap}.jsonl"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in chunks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks (size={chunk_size}, overlap={overlap}) -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--all-sweep-sizes", action="store_true", help="Ingest every chunk_size/overlap in configs/chunking.yaml sweep")
    args = ap.parse_args()

    cfg = load_yaml("chunking.yaml")
    if args.all_sweep_sizes:
        sizes = cfg["sweep"]["chunk_sizes"]
        overlaps = cfg["sweep"]["overlaps"]
        for size, overlap in zip(sizes, overlaps):
            run_ingest(size, overlap, cfg["default"]["tokenizer"])
    else:
        size = args.chunk_size or cfg["default"]["chunk_size"]
        overlap = args.overlap if args.overlap is not None else cfg["default"]["overlap"]
        run_ingest(size, overlap, cfg["default"]["tokenizer"])


if __name__ == "__main__":
    main()
