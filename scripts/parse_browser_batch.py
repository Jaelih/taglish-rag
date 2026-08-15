"""One-off tool: turn a saved claude-in-chrome browser_batch JSON transcript
(alternating [navigate] / [get_page_text] blocks) into individual raw corpus
files + a manifest.jsonl entry per page.

Used for sites that are JS-rendered or bot-gated against plain HTTP clients
(here: pagibigfund.gov.ph), where a real browser session was used to fetch
page text instead of `requests`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def slugify(url: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1] or "index"
    name = re.sub(r"\.html?$", "", name, flags=re.I)
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()


def parse(batch_path: Path) -> list[dict]:
    blocks = json.loads(batch_path.read_text(encoding="utf-8"))
    pages = []
    last_url = None
    for block in blocks:
        text = block.get("text", "")
        nav_match = re.match(r"\[navigate\] Navigated to (\S+)", text)
        if nav_match:
            last_url = nav_match.group(1)
            continue
        if text.startswith("[get_page_text]"):
            title_match = re.search(r"Title: (.+)", text)
            url_match = re.search(r"URL: (\S+)", text)
            url = url_match.group(1) if url_match else last_url
            title = title_match.group(1).strip() if title_match else url
            body = text.split("---\n", 1)[-1].strip()
            if "404 - File or directory not found" in body:
                continue
            pages.append({"url": url, "title": title, "body": body})
    return pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_json", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--agency", required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"
    existing = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            existing.add(json.loads(line)["url"])

    pages = parse(args.batch_json)
    written = 0
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for page in pages:
            if page["url"] in existing:
                continue
            slug = slugify(page["url"])
            out_path = args.out_dir / f"{slug}.txt"
            out_path.write_text(
                f"Title: {page['title']}\nURL: {page['url']}\n---\n{page['body']}\n",
                encoding="utf-8",
            )
            mf.write(
                json.dumps(
                    {
                        "doc_id": f"{args.agency}-{slug}",
                        "agency": args.agency,
                        "title": page["title"],
                        "url": page["url"],
                        "local_path": out_path.relative_to(args.out_dir.parents[2]).as_posix(),
                        "doc_type": "html",
                    }
                )
                + "\n"
            )
            written += 1
    print(f"Wrote {written} new pages to {args.out_dir} (skipped {len(pages) - written} dupes/404s)")


if __name__ == "__main__":
    main()
