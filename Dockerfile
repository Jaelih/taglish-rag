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
RUN uv pip install --system --no-cache --no-deps .

COPY configs ./configs
COPY app ./app
COPY eval/taglish_rag_eval_v1.jsonl ./eval/taglish_rag_eval_v1.jsonl
# Only the pre-chunked corpus the app actually queries (chunk_size=512,
# overlap=64, matching app/app.py's RetrievalConfig) -- not the raw PDFs/HTML
# or the other chunk-size sweep outputs, which the deployed app never reads.
COPY data/processed/chunks_512_64.jsonl ./data/processed/chunks_512_64.jsonl

# GOOGLE_API_KEY is deliberately not baked in -- inject it at runtime
# (`docker run -e GOOGLE_API_KEY=...` / HF Spaces secret). Without it the
# app still starts and retrieval works; answer generation raises on the
# first query. See docs/DEPLOY.md.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app/app.py"]
