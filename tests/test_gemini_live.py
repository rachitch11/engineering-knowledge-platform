"""Optional live Gemini integration check (skipped without GEMINI_API_KEY)."""

from __future__ import annotations

import os

import pytest

from app.core.config import Settings, get_settings
from app.generation.exceptions import LLMProviderError
from app.main import create_app
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.reranker import MockReranker


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping live Gemini test",
)
def test_live_gemini_general_chat(tmp_path) -> None:
    get_settings.cache_clear()
    settings = Settings(
        environment="development",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        llm_provider="gemini",
        gemini_api_key=os.environ["GEMINI_API_KEY"],
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    )
    from app.generation.providers import build_llm_provider

    provider = build_llm_provider(settings)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            text = provider.generate_conversational(
                "Reply with exactly: pong",
                conversation_context="",
                timeout_seconds=120.0,
                max_context_chars=500,
                num_predict=32,
            )
            break
        except LLMProviderError as exc:
            last_error = exc
            if attempt >= 2 or "503" not in str(exc):
                raise
            import time

            time.sleep(2)
    else:
        assert last_error is not None
        raise last_error
    assert "pong" in text.lower()
    get_settings.cache_clear()
