# Deploying the demo

## Locally / Docker

```
docker build -t taglish-rag .
docker run -p 7860:7860 --env-file .env taglish-rag
```

`GOOGLE_API_KEY` must be set in `.env` for answer generation — it is not baked into the image. Without it the app still starts and retrieval works, but the first question raises `RuntimeError: GOOGLE_API_KEY not set`.

## HuggingFace Spaces (Docker SDK)

HF Spaces are their own git repos, separate from this GitHub repo, so this repo's root `README.md` (written for recruiters/GitHub) doesn't collide with the Space's `README.md` (which HF requires to carry specific YAML frontmatter). Steps:

1. Create a new Space at huggingface.co/new-space, SDK = **Docker**.
2. Clone the new (empty) Space repo locally, e.g. `git clone https://huggingface.co/spaces/<user>/taglish-rag hf-space`.
3. Copy in: `Dockerfile`, `app/`, `src/`, `configs/`, `eval/taglish_rag_eval_v1.jsonl`, `pyproject.toml`.
4. Add a Space `README.md` with frontmatter:
   ```yaml
   ---
   title: Taglish RAG
   emoji: 🇵🇭
   colorFrom: blue
   colorTo: red
   sdk: docker
   app_port: 7860
   ---
   ```
5. In the Space's **Settings → Repository secrets**, add `GOOGLE_API_KEY` so the deployed demo can generate answers.
6. `git add -A && git commit -m "deploy" && git push`.

This step requires an HF account + token and pushes to a public URL, so it's left to you to run rather than done automatically here.

## GitHub Actions CI

`.github/workflows/ci.yml` runs unit tests plus a 20-question BM25-only smoke retrieval eval on every push — no model downloads or API keys needed, so it stays fast and free on GitHub-hosted runners. It is not a substitute for the full ablation sweep (`taglish_rag.eval.ablate`), which needs the larger embedding models and is run manually / on a schedule, not per-commit.
