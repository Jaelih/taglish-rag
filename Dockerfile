FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY app/requirements.txt ./app/requirements.txt
RUN uv pip install --system --no-cache -r app/requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
# --no-deps: runtime deps already satisfied above from the slim app/requirements.txt;
# a plain `uv pip install .` here would also pull in the full ingestion-only
# dependency set (pdfplumber, beautifulsoup4, datasets, supabase, ...) that
# the deployed app never uses.
#
# -e (editable) is load-bearing, not a convenience: config.py derives
# REPO_ROOT as Path(__file__).parents[2], so a regular install puts the
# package under site-packages and REPO_ROOT resolves to the interpreter's
# lib dir -- configs/ and data/ are then never found and the app dies with
# "Config not found: /usr/local/lib/python3.11/configs/embeddings.yaml".
# Editable keeps taglish_rag at /app/src/taglish_rag, so REPO_ROOT == /app.
RUN uv pip install --system --no-cache --no-deps -e .

COPY configs ./configs
COPY app ./app
COPY eval/taglish_rag_eval_v1.jsonl ./eval/taglish_rag_eval_v1.jsonl
# Only the pre-chunked corpus the app actually queries (chunk_size=512,
# overlap=64, matching app/app.py's RetrievalConfig) -- not the raw PDFs/HTML
# or the other chunk-size sweep outputs, which the deployed app never reads.
COPY data/processed/chunks_512_64.jsonl ./data/processed/chunks_512_64.jsonl
# Precomputed bge-m3 passage embeddings for exactly that chunk set (2MB).
# embed_texts() keys its cache on a hash of the chunk-id list, so this hits
# on startup and the container never re-embeds the 514-chunk corpus.
COPY data/index/embeddings/bge-m3__passage__c85f0934803604dd.npy \
     ./data/index/embeddings/bge-m3__passage__c85f0934803604dd.npy

# Pin the HF cache inside the image rather than $HOME/.cache: the build runs
# as root but HF Spaces may start the container as another user, and a cache
# under /root would then be invisible -- silently re-downloading 2.2GB on boot.
ENV HF_HOME=/app/.cache/huggingface

# Bake the query encoder into the image. Without this the 2.2GB model
# downloads lazily on the *first question*, so the first visitor waits
# minutes and may well assume the demo is broken.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# embed_texts() writes a small .npy per unique query at runtime, so
# data/index/embeddings must stay writable whichever uid Spaces picks.
RUN mkdir -p /app/data/index/embeddings \
    && chmod -R a+rwX /app/data /app/.cache

# GOOGLE_API_KEY is deliberately not baked in -- inject it at runtime
# (`docker run -e GOOGLE_API_KEY=...` / HF Spaces secret). Without it the
# app still starts and retrieval works; answer generation raises on the
# first query. See docs/DEPLOY.md.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app/app.py"]
