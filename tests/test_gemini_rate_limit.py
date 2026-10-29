"""Tests for Gemini rate-limit classification and default model configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.core.config import Settings
from app.generation.exceptions import LLMRateLimitError
from app.generation.gemini_errors import (
    DEFAULT_GEMINI_MODEL,
    classify_rate_limit,
    extract_google_error_text,
    should_retry_response,
)
from app.generation.llm_http import LLMHttpConfig, post_with_retries


def test_default_gemini_model_is_flash_lite(tmp_path) -> None:
    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "c"),
        bm25_index_path=str(tmp_path / "b.json"),
    )
    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.gemini_model == "gemini-3.5-flash-lite"


def test_daily_quota_429_is_not_retryable() -> None:
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
        json={
            "error": {
                "code": 429,
                "message": "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_requests_per_day",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    message, retryable = classify_rate_limit(response)
    assert retryable is False
    assert "daily" in message.lower() or "RPD" in message
    assert should_retry_response(response) is False


def test_rpm_429_is_retryable() -> None:
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://example.test"),
        json={
            "error": {
                "code": 429,
                "message": "Quota exceeded for metric: generate_content_requests_per_minute per project.",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    _, retryable = classify_rate_limit(response)
    assert retryable is True
    assert should_retry_response(response) is True


@patch("app.generation.llm_http.time.sleep")
def test_daily_quota_429_does_not_retry(mock_sleep: MagicMock) -> None:
    config = LLMHttpConfig(5, 30, 10, 5, max_retries=2, retry_backoff_seconds=0.01)
    client = MagicMock(spec=httpx.Client)
    body = {
        "error": {
            "code": 429,
            "message": "Quota exceeded for metric: generate_content_requests_per_day",
            "status": "RESOURCE_EXHAUSTED",
        }
    }
    failing = httpx.Response(
        429,
        request=httpx.Request("POST", "https://example.test"),
        json=body,
    )
    client.post.return_value = failing

    with pytest.raises(LLMRateLimitError, match="daily request quota"):
        post_with_retries(client, "https://example.test", config=config)

    assert client.post.call_count == 1
    mock_sleep.assert_not_called()


@patch("app.generation.llm_http.time.sleep")
def test_rpm_429_retries_then_raises(mock_sleep: MagicMock) -> None:
    config = LLMHttpConfig(5, 30, 10, 5, max_retries=2, retry_backoff_seconds=0.01)
    client = MagicMock(spec=httpx.Client)
    rpm_body = {
        "error": {
            "code": 429,
            "message": "Quota exceeded for generate_content_requests_per_minute",
            "status": "RESOURCE_EXHAUSTED",
        }
    }
    failing = httpx.Response(
        429,
        request=httpx.Request("POST", "https://example.test"),
        json=rpm_body,
    )
    client.post.return_value = failing

    with pytest.raises(LLMRateLimitError, match="requests-per-minute"):
        post_with_retries(client, "https://example.test", config=config)

    assert client.post.call_count == 3
    assert mock_sleep.call_count == 2


def test_extract_google_error_text_reads_message() -> None:
    response = httpx.Response(
        429,
        json={"error": {"message": "Too many requests", "status": "RESOURCE_EXHAUSTED"}},
    )
    assert "Too many requests" in extract_google_error_text(response)
