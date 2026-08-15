"""Download BIR issuances (RMCs/RRs) from a curated seed list.

BIR's site (bir.gov.ph) is a client-rendered Next.js app with no static
listing pages or sitemap reachable over plain HTTP, so unlike PhilHealth we
can't crawl an index page generically. Its PDFs are served from a separate,
un-gated CDN (bir-cdn.bir.gov.ph) that *is* directly fetchable, so instead of
scraping we curate a seed list of issuance URLs (gathered via search, scoped
to individual-taxpayer topics: ITR filing/penalties, VAT/percentage tax
registration, invoicing under the Ease of Paying Taxes Act, withholding tax)
in configs/bir_seed_urls.txt and bulk-download from that list.
"""
from __future__ import annotations

import json
import re
import time
from urllib.parse import unquote

import requests

from taglish_rag.config import REPO_ROOT, data_path

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; taglish-rag-research-bot/0.1)"}
OUT_DIR = data_path("raw", "bir")
SEED_FILE = REPO_ROOT / "configs" / "bir_seed_urls.txt"


def title_from_url(url: str) -> str:
    fname = unquote(url.rsplit("/", 1)[-1]).replace(".pdf", "")
    return re.sub(r"\s+", " ", fname).strip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = [
        line.strip()
        for line in SEED_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    docs = []
    for url in urls:
        title = title_from_url(url)
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
        local_path = OUT_DIR / f"{slug}.pdf"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  FAILED {url}: {e}")
            continue
        local_path.write_bytes(resp.content)
        docs.append(
            {
                "doc_id": f"bir-{slug}",
                "agency": "bir",
                "title": title,
                "url": url,
                "local_path": local_path.relative_to(data_path().parent).as_posix(),
                "doc_type": "pdf",
            }
        )
        print(f"  OK {title} ({len(resp.content)} bytes)")
        time.sleep(0.3)

    manifest_path = OUT_DIR / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"Wrote {len(docs)} docs, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
