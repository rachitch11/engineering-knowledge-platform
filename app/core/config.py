"""Application configuration loaded from environment variables and optional .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings for the Engineering Knowledge Intelligence Platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(
        default="Engineering Knowledge Intelligence",
        validation_alias="APP_NAME",
    )
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        validation_alias="ENVIRONMENT",
    )
    api_host: str = Field(default="127.0.0.1", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    chroma_path: str = Field(
        default="data/processed/chroma",
        validation_alias="CHROMA_PATH",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    max_upload_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        validation_alias="MAX_UPLOAD_SIZE_BYTES",
    )
    chunk_size: int = Field(default=512, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, validation_alias="CHUNK_OVERLAP")
    chroma_collection_name: str = Field(
        default="engineering_knowledge",
        validation_alias="CHROMA_COLLECTION_NAME",
    )
    bm25_index_path: str = Field(
        default="data/processed/bm25_index.json",
        validation_alias="BM25_INDEX_PATH",
    )
    retrieval_top_k: int = Field(default=5, validation_alias="RETRIEVAL_TOP_K")
    rrf_k: int = Field(default=60, validation_alias="RRF_K")
    rerank_enabled: bool = Field(default=True, validation_alias="RERANK_ENABLED")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="RERANK_MODEL",
    )
    rerank_candidate_k: int = Field(default=20, validation_alias="RERANK_CANDIDATE_K")
    rerank_top_k: int = Field(default=5, validation_alias="RERANK_TOP_K")
    max_evidence_chunks: int = Field(default=5, validation_alias="MAX_EVIDENCE_CHUNKS")
    max_context_chars: int = Field(default=8000, validation_alias="MAX_CONTEXT_CHARS")
    llm_provider: Literal["mock", "gemini", "ollama", "openai_compatible"] = Field(
        default="gemini",
        validation_alias="LLM_PROVIDER",
    )
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai",
        validation_alias="GEMINI_BASE_URL",
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
        validation_alias="GEMINI_MODEL",
    )
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="llama3.2", validation_alias="OLLAMA_MODEL")
    openai_compatible_base_url: str = Field(
        default="https://api.openai.com",
        validation_alias="OPENAI_COMPATIBLE_BASE_URL",
    )
    openai_compatible_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="OPENAI_COMPATIBLE_MODEL",
    )
    openai_compatible_api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_COMPATIBLE_API_KEY",
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        validation_alias="LLM_TIMEOUT_SECONDS",
        description="HTTP read timeout while waiting for LLM response bytes.",
    )
    llm_connect_timeout_seconds: float = Field(
        default=10.0,
        validation_alias="LLM_CONNECT_TIMEOUT_SECONDS",
    )
    llm_write_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="LLM_WRITE_TIMEOUT_SECONDS",
    )
    llm_pool_timeout_seconds: float = Field(
        default=10.0,
        validation_alias="LLM_POOL_TIMEOUT_SECONDS",
    )
    llm_max_retries: int = Field(default=2, validation_alias="LLM_MAX_RETRIES")
    llm_retry_backoff_seconds: float = Field(
        default=1.0,
        validation_alias="LLM_RETRY_BACKOFF_SECONDS",
    )
    llm_max_output_tokens: int = Field(
        default=768,
        validation_alias="LLM_MAX_OUTPUT_TOKENS",
    )
    backend_url: str = Field(
        default="http://127.0.0.1:8000",
        validation_alias="BACKEND_URL",
    )
    single_user_mode: bool = Field(default=False, validation_alias="SINGLE_USER_MODE")
    evidence_min_dense_score: float = Field(
        default=0.24,
        validation_alias="EVIDENCE_MIN_DENSE_SCORE",
    )
    max_conversation_messages: int = Field(
        default=12,
        validation_alias="MAX_CONVERSATION_MESSAGES",
    )
    max_conversation_context_chars: int = Field(
        default=4000,
        validation_alias="MAX_CONVERSATION_CONTEXT_CHARS",
    )
    ollama_num_predict: int = Field(default=512, validation_alias="OLLAMA_NUM_PREDICT")
    rerank_skip_short_queries: bool = Field(
        default=True,
        validation_alias="RERANK_SKIP_SHORT_QUERIES",
    )
    rerank_skip_max_words: int = Field(
        default=5,
        validation_alias="RERANK_SKIP_MAX_WORDS",
    )
    log_pipeline_timings: bool = Field(
        default=True,
        validation_alias="LOG_PIPELINE_TIMINGS",
    )
    ephemeral_sessions_enabled: bool = Field(
        default=True,
        validation_alias="EPHEMERAL_SESSIONS_ENABLED",
    )
    session_ttl_seconds: int = Field(
        default=3600,
        validation_alias="SESSION_TTL_SECONDS",
    )
    session_cleanup_interval_seconds: int = Field(
        default=300,
        validation_alias="SESSION_CLEANUP_INTERVAL_SECONDS",
    )
    session_store_root: str = Field(
        default="data/ephemeral_sessions",
        validation_alias="SESSION_STORE_ROOT",
    )
    index_embedding_batch_size: int = Field(
        default=32,
        validation_alias="INDEX_EMBEDDING_BATCH_SIZE",
        ge=1,
        le=256,
    )
    memory_constrained: bool = Field(
        default=False,
        validation_alias="MEMORY_CONSTRAINED",
        description="Smaller embedding batches; recommended on 512MB hosts (Render Free).",
    )

    @field_validator("memory_constrained", mode="before")
    @classmethod
    def _default_memory_constrained_on_render(cls, value: object) -> object:
        if value is not None and str(value).strip() != "":
            return value
        from app.core.memory_diagnostics import memory_constrained_default

        return memory_constrained_default()

    cors_allow_origins: str = Field(
        default="",
        validation_alias="CORS_ALLOW_ORIGINS",
        description="Comma-separated browser origins allowed to call the API (optional).",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance (safe to call on every request)."""
    return Settings()
