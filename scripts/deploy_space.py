"""Push this repo's deployed subset to the HuggingFace Space, so a code
change actually reaches https://huggingface.co/spaces/<repo_id>.

The Space is a separate git repo from this one -- pushing to GitHub does
not touch it. This uploads via the HTTP API (huggingface_hub.upload_folder)
rather than `git push`, because `hf auth login`'s stored credential isn't
automatically usable as a git credential -- see docs/DEPLOY.md.

Uploads exactly the paths the Dockerfile COPYs (see allow_patterns below),
plus docs/HF_SPACE_README.md mapped to the Space's required README.md
(this repo's own README.md is for GitHub and has no Spaces frontmatter, so
it can't be reused directly).

Usage:
    uv run python scripts/deploy_space.py                       # jaelih/taglish-rag
    uv run python scripts/deploy_space.py --repo-id someone/other-space
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_README = ROOT / "docs" / "HF_SPACE_README.md"

# Keep in lockstep with the Dockerfile's COPY list.
ALLOW_PATTERNS = [
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "app/**",
    "src/**",
    "configs/**",
    "eval/taglish_rag_eval_v1.jsonl",
    "data/processed/chunks_512_64.jsonl",
    "data/index/embeddings/bge-m3__passage__c85f0934803604dd.npy",
]
IGNORE_PATTERNS = ["**/__pycache__/**", "**/*.pyc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="jaelih/taglish-rag")
    ap.add_argument("--message", default="Update Space from repo")
    args = ap.parse_args()

    if not SPACE_README.exists():
        raise SystemExit(f"missing {SPACE_README} -- this becomes the Space's README.md")

    api = HfApi()

    print(f"Uploading {ROOT} -> spaces/{args.repo_id} ...")
    commit = api.upload_folder(
        folder_path=str(ROOT),
        repo_id=args.repo_id,
        repo_type="space",
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=args.message,
    )
    print("code:", commit)

    readme_commit = api.upload_file(
        path_or_fileobj=str(SPACE_README),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="space",
        commit_message=f"{args.message} (README)",
    )
    print("readme:", readme_commit)
    print(f"\nDone. Space rebuilds automatically: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
