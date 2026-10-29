FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    EPHEMERAL_SESSIONS_ENABLED=true \
    SINGLE_USER_MODE=false \
    SESSION_STORE_ROOT=/data/ephemeral_sessions \
    LLM_PROVIDER=mock \
    CHROMA_PATH=/data/legacy_chroma \
    BM25_INDEX_PATH=/data/legacy_bm25.json

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY streamlit_app.py ./

RUN mkdir -p /data/ephemeral_sessions

VOLUME ["/data/ephemeral_sessions"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=3).raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
