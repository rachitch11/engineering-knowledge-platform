FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    EPHEMERAL_SESSIONS_ENABLED=true \
    SINGLE_USER_MODE=false \
    SESSION_STORE_ROOT=/tmp/ephemeral_sessions \
    MEMORY_CONSTRAINED=true \
    RERANK_ENABLED=false \
    INDEX_EMBEDDING_BATCH_SIZE=8 \
    LLM_PROVIDER=gemini \
    CHROMA_PATH=/tmp/legacy_chroma \
    BM25_INDEX_PATH=/tmp/legacy_bm25.json

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY scripts ./scripts
COPY streamlit_app.py ./

RUN chmod +x /app/scripts/start_api.sh && mkdir -p /tmp/ephemeral_sessions

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=3).raise_for_status()"

CMD ["/app/scripts/start_api.sh"]
