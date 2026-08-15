"""Scrape PhilHealth circulars (server-rendered, no bot-gating) and a
handful of member-facing FAQ pages."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from taglish_rag.config import data_path

BASE = "https://www.philhealth.gov.ph"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; taglish-rag-research-bot/0.1)"}
OUT_DIR = data_path("raw", "philhealth")

CIRCULAR_YEARS = [2023, 2024, 2025, 2026]
MAX_CIRCULARS_PER_YEAR = 8

MEMBER_PAGES = [
    ("members-formal", "/members/formal/"),
    ("members-informal", "/members/informal/"),
    ("members-lifetime", "/members/lifetime/"),
    ("members-senior", "/members/senior/"),
    ("members-sponsored", "/members/sponsored/"),
]


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp


def fetch_circular_links() -> list[dict]:
    """Return [{year, filename, url}] for main PC circulars (skip TS_/Annex variants)."""
    links = []
    for year in CIRCULAR_YEARS:
        url = f"{BASE}/circulars/{year}/archives.php"
        try:
            resp = _get(url)
        except requests.RequestException as e:
            print(f"  skip {year}: {e}")
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        count = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.lower().endswith(".pdf"):
                continue
            fname = href.rsplit("/", 1)[-1]
            if not re.match(r"^PC\d{4}-\d{4}\.pdf$", fname, re.I):
                continue  # skip TS_ transmittal slips and Annex sub-pages
            links.append(
                {
                    "year": year,
                    "filename": fname,
                    "url": f"{BASE}/circulars/{year}/{href.lstrip('./')}",
                }
            )
            count += 1
            if count >= MAX_CIRCULARS_PER_YEAR:
                break
        print(f"  {year}: {count} circulars queued")
    return links


def download_pdfs(links: list[dict]) -> list[dict]:
    docs = []
    for item in links:
        doc_id = f"philhealth-{item['filename'].replace('.pdf', '').lower()}"
        local_path = OUT_DIR / item["filename"]
        try:
            resp = _get(item["url"])
        except requests.RequestException as e:
            print(f"  FAILED {item['url']}: {e}")
            continue
        local_path.write_bytes(resp.content)
        docs.append(
            {
                "doc_id": doc_id,
                "agency": "philhealth",
                "title": f"PhilHealth Circular {item['filename'].replace('.pdf', '')}",
                "url": item["url"],
                "local_path": local_path.relative_to(data_path().parent).as_posix(),
                "doc_type": "pdf",
            }
        )
        time.sleep(0.3)  # be polite
    return docs


def download_member_pages() -> list[dict]:
    docs = []
    for slug, path in MEMBER_PAGES:
        url = BASE + path
        try:
            resp = _get(url)
        except requests.RequestException as e:
            print(f"  FAILED {url}: {e}")
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        local_path = OUT_DIR / f"{slug}.txt"
        local_path.write_text(f"Title: {soup.title.string if soup.title else slug}\nURL: {url}\n---\n{text}\n", encoding="utf-8")
        docs.append(
            {
                "doc_id": f"philhealth-{slug}",
                "agency": "philhealth",
                "title": soup.title.string.strip() if soup.title else slug,
                "url": url,
                "local_path": local_path.relative_to(data_path().parent).as_posix(),
                "doc_type": "html",
            }
        )
    return docs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching circular links...")
    links = fetch_circular_links()
    print(f"Downloading {len(links)} circular PDFs...")
    docs = download_pdfs(links)
    print("Downloading member FAQ pages...")
    docs += download_member_pages()

    manifest_path = OUT_DIR / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"Wrote {len(docs)} docs, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
