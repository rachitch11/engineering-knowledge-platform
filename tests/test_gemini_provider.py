"""Tests for Google Gemini LLM provider (mocked HTTP; no live API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.core.config import Settings, get_settings
from app.generation.exceptions import LLMProviderError, LLMTimeoutError
from app.generation.models import EvidenceChunk
from app.generation.providers import GeminiProvider, build_llm_provider, parse_generated_json


@pytest.fixture
def gemini_settings(tmp_path) -> Settings:
    return Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        llm_provider="gemini",
        gemini_api_key="test-gemini-key",
        gemini_model="gemini-2.0-flash",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )


def test_default_llm_provider_is_gemini(tmp_path) -> None:
    get_settings.cache_clear()
    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
    )
    assert settings.llm_provider == "gemini"
    get_settings.cache_clear()


def test_build_gemini_provider_requires_api_key(tmp_path) -> None:
    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        llm_provider="gemini",
        gemini_api_key=None,
    )
    with pytest.raises(LLMProviderError, match="GEMINI_API_KEY"):
        build_llm_provider(settings)


def test_build_gemini_provider_initialization(gemini_settings: Settings) -> None:
    provider = build_llm_provider(gemini_settings)
    assert isinstance(provider, GeminiProvider)


@patch("app.generation.providers.post_with_retries")
@patch("app.generation.providers.httpx.Client")
def test_gemini_generate_answer_success(
    mock_client_cls: MagicMock,
    mock_post: MagicMock,
    gemini_settings: Settings,
) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"answer":"From the doc: Python.","citation_ids":["chunk-1"]}'
                    )
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    provider = build_llm_provider(gemini_settings)
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="d1",
            filename="a.md",
            file_type="markdown",
            chunk_index=0,
            page_number=None,
            text="Python skills",
            score=0.9,
        )
    ]
    result = provider.generate_answer(
        "What skills?",
        evidence,
        timeout_seconds=30.0,
        max_context_chars=8000,
    )
    assert "Python" in result.answer
    assert result.citation_ids == ["chunk-1"]
    call_kwargs = mock_post.call_args
    assert "chat/completions" in call_kwargs[0][1]
    assert "/v1/chat/completions" in call_kwargs[0][1] or call_kwargs[0][1].endswith(
        "/openai/chat/completions"
    )
    headers = call_kwargs[1]["headers"]
    assert headers["Authorization"] == "Bearer test-gemini-key"
    assert "GEMINI" not in str(call_kwargs)


@patch("app.generation.providers.post_with_retries")
@patch("app.generation.providers.httpx.Client")
def test_gemini_timeout(mock_client_cls: MagicMock, mock_post: MagicMock, gemini_settings: Settings) -> None:
    mock_client_cls.return_value.__enter__.return_value = MagicMock()
    mock_post.side_effect = LLMTimeoutError(
        "The language model did not respond in time. Try again, switch to gemini-3.5-flash-lite, "
        "or increase LLM_TIMEOUT_SECONDS."
    )
    provider = build_llm_provider(gemini_settings)
    with pytest.raises(LLMTimeoutError, match="did not respond in time"):
        provider.generate_answer(
            "q",
            [],
            timeout_seconds=1.0,
            max_context_chars=100,
        )


@patch("app.generation.providers.post_with_retries")
@patch("app.generation.providers.httpx.Client")
def test_gemini_http_error(
    mock_client_cls: MagicMock,
    mock_post: MagicMock,
    gemini_settings: Settings,
) -> None:
    mock_client_cls.return_value.__enter__.return_value = MagicMock()
    mock_post.side_effect = LLMProviderError("Hosted LLM request failed: 429 quota")
    provider = build_llm_provider(gemini_settings)
    with pytest.raises(LLMProviderError, match="Hosted LLM request failed"):
        provider.generate_conversational(
            "hello",
            conversation_context="",
            timeout_seconds=5.0,
            max_context_chars=500,
        )


def test_citation_validation_strips_unknown_ids() -> None:
    parsed = parse_generated_json(
        '{"answer":"ok","citation_ids":["chunk-1","fake-id"]}',
        {"chunk-1"},
    )
    assert parsed.citation_ids == ["chunk-1"]


def test_invalid_json_raises() -> None:
    with pytest.raises(LLMProviderError, match="invalid JSON"):
        parse_generated_json("not json", set())


@patch("app.core.ollama.assert_ollama_ready")
@patch("app.generation.providers.post_with_retries")
@patch("app.generation.providers.httpx.Client")
def test_gemini_mode_never_calls_ollama(
    mock_client_cls: MagicMock,
    mock_post: MagicMock,
    mock_ollama_ready: MagicMock,
    gemini_settings: Settings,
) -> None:
    mock_client_cls.return_value.__enter__.return_value = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"answer":"hi","citation_ids":[]}'}}]
    }
    mock_post.return_value = mock_response
    provider = build_llm_provider(gemini_settings)
    provider.generate_answer("q", [], timeout_seconds=5.0, max_context_chars=500)
    mock_ollama_ready.assert_not_called()


def test_llm_health_route_reports_missing_gemini_key(tmp_path) -> None:
    get_settings.cache_clear()
    from app.main import create_app
    from fastapi.testclient import TestClient
    from app.retrieval.embeddings import MockEmbedding
    from app.retrieval.reranker import MockReranker
    from app.generation.providers import MockLLMProvider

    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        llm_provider="gemini",
        gemini_api_key=None,
    )
    app = create_app(
        settings=settings,
        embedding_model=MockEmbedding(),
        reranker=MockReranker(),
        llm_provider=MockLLMProvider(),
    )
    with TestClient(app) as test_client:
        response = test_client.get("/health/llm")
        body = response.json()
        assert body["status"] == "unavailable"
        assert body["api_key_configured"] is False
        assert "GEMINI_API_KEY" in body["message"]
    get_settings.cache_clear()
