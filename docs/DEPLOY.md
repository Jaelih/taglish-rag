# Deploying the demo

## Quickest: a temporary public link (no hosting)

Run the app, then put a tunnel in front of it:

```
uv run python app/app.py                              # serves 127.0.0.1:7860
cloudflared tunnel --url http://127.0.0.1:7860        # prints a public URL
```

Gives a `https://<random>.trycloudflare.com` URL for as long as both processes
run. Nothing is hosted, and every question spends your `GOOGLE_API_KEY` quota.
Good for showing a few people; not a deployment.

`winget install Cloudflare.cloudflared` if you don't have it.

### Why not `GRADIO_SHARE=1`?

`app/app.py` still supports it, but Gradio's own tunnel needs `frpc`, which
**Windows Defender blocks as `PUA:Win32/FRProxy`** — `subprocess` fails with
`PermissionError: [WinError 5] Access is denied`, and Gradio reports only the
misleading "Could not create share link. Please check your internet
connection." The network is fine; the binary is refused execution. Renaming it
to `.exe` doesn't help. Using it would mean adding a Defender exclusion for a
reverse-proxy binary — cloudflared avoids that tradeoff entirely.

## Locally / Docker

```
docker build -t taglish-rag .
docker run -p 7860:7860 --env-file .env taglish-rag
```

`GOOGLE_API_KEY` must be set in `.env` for answer generation — it is not baked into the image. Without it the app still starts and retrieval works, but the first question raises `RuntimeError: GOOGLE_API_KEY not set`.

## HuggingFace Spaces (Docker SDK)

> **Requires a PRO subscription.** As of 2026-08, creating a Docker or Gradio
> Space returns `402 Payment Required`: *"Static Spaces are free for everyone,
> but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO
> subscription."* Only static (frontend-only) Spaces remain free, which this
> app can't be. The steps below are correct and ready to run once PRO is active.

HF Spaces are their own git repos, separate from this GitHub repo. Pushing to
GitHub does **not** touch the Space — the two are unrelated remotes, and the
Space only rebuilds when its own repo gets a new commit. This repo's root
`README.md` (written for recruiters/GitHub) also can't be reused for the Space
as-is: HF requires the Space's `README.md` to carry specific YAML frontmatter
(`sdk: docker`, `app_port`, ...), which the GitHub README doesn't have. That
frontmatter's source of truth lives in this repo at `docs/HF_SPACE_README.md`.

### First-time setup

1. Create a new Space at huggingface.co/new-space, SDK = **Docker**, or:
   `hf repo create <user>/taglish-rag --type space --sdk docker --public`
2. `hf auth login` if you haven't (needs a token with **write** access).
3. In the Space's **Settings → Repository secrets**, add `GOOGLE_API_KEY` so
   the deployed demo can generate answers — or via API:
   ```python
   from huggingface_hub import HfApi
   HfApi().add_space_secret(repo_id="<user>/taglish-rag", key="GOOGLE_API_KEY", value="...")
   ```
4. Push the code (see below).

### Pushing a code change to the Space (every time after)

```
uv run python scripts/deploy_space.py --repo-id <user>/taglish-rag
```

This is **not** `git push` — `hf auth login`'s stored credential isn't
automatically usable as a git credential for the Space's git remote (attempting
a plain `git push` fails with "Invalid username or password" even after `hf
auth login`). `scripts/deploy_space.py` instead uploads via the HTTP API
(`huggingface_hub.upload_folder`), which reuses that same login. It:

- uploads exactly the paths the Dockerfile `COPY`s (kept in sync in
  `ALLOW_PATTERNS` at the top of the script) — anything else in the repo
  (raw scraped data, other chunk-size sweeps, results/, tests/) never
  reaches the Space, deliberately;
- uploads `docs/HF_SPACE_README.md` as the Space's `README.md`;
- reports "No files have been modified since last commit" and skips instead
  of pushing an empty commit, if nothing actually changed.

The Space detects the new commit and rebuilds/restarts automatically — no
separate "deploy" trigger. Watch it with:

```python
from huggingface_hub import HfApi
print(HfApi().space_info("<user>/taglish-rag").runtime.stage)
# BUILDING -> APP_STARTING -> RUNNING (or *_ERROR)
```

**Editing `app/app.py` or anything under `src/`, `configs/`, or the
Dockerfile itself** takes effect only after running the script above — a
GitHub push alone changes nothing on the live Space. Editing `pyproject.toml`
dependencies likewise needs a re-push, and forces a full rebuild (new
`uv pip install` layer) rather than a fast restart.

## GitHub Actions CI

`.github/workflows/ci.yml` runs unit tests plus a 20-question BM25-only smoke retrieval eval on every push — no model downloads or API keys needed, so it stays fast and free on GitHub-hosted runners. It is not a substitute for the full ablation sweep (`taglish_rag.eval.ablate`), which needs the larger embedding models and is run manually / on a schedule, not per-commit.
