"""Tests for hosted LLM HTTP timeouts and retries."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.generation.exceptions import LLMProviderError, LLMTimeoutError
from app.generation.llm_http import LLMHttpConfig, post_with_retries


@pytest.fixture
def http_config() -> LLMHttpConfig:
    return LLMHttpConfig(
        connect_timeout_seconds=5.0,
        read_timeout_seconds=30.0,
        write_timeout_seconds=10.0,
        pool_timeout_seconds=5.0,
        max_retries=2,
        retry_backoff_seconds=0.01,
    )


def test_read_timeout_raises_clear_message(http_config: LLMHttpConfig) -> None:
    client = MagicMock(spec=httpx.Client)
    client.post.side_effect = httpx.ReadTimeout("read timed out")

    with pytest.raises(LLMTimeoutError, match="did not respond in time"):
        post_with_retries(client, "https://example.test/v1/chat/completions", config=http_config)

    assert client.post.call_count == 1


@patch("app.generation.llm_http.time.sleep")
def test_retries_transient_503_then_succeeds(
    mock_sleep: MagicMock,
    http_config: LLMHttpConfig,
) -> None:
    client = MagicMock(spec=httpx.Client)
    failing = httpx.Response(503, request=httpx.Request("POST", "https://example.test"))
    ok = httpx.Response(
        200,
        request=httpx.Request("POST", "https://example.test"),
        json={"ok": True},
    )
    client.post.side_effect = [failing, ok]

    response = post_with_retries(
        client,
        "https://example.test/v1/chat/completions",
        config=http_config,
    )
    assert response.status_code == 200
    assert client.post.call_count == 2
    mock_sleep.assert_called_once()


@patch("app.generation.llm_http.time.sleep")
def test_transient_503_exhausts_retries(
    mock_sleep: MagicMock,
    http_config: LLMHttpConfig,
) -> None:
    client = MagicMock(spec=httpx.Client)
    failing = httpx.Response(503, request=httpx.Request("POST", "https://example.test"))
    client.post.return_value = failing

    with pytest.raises(LLMProviderError, match="Hosted LLM request failed"):
        post_with_retries(client, "https://example.test/v1/chat/completions", config=http_config)

    assert client.post.call_count == 3


@patch("app.generation.providers.post_with_retries")
@patch("app.generation.providers.httpx.Client")
def test_gemini_uses_granular_timeout(
    mock_client_cls: MagicMock,
    mock_post: MagicMock,
    tmp_path,
) -> None:
    from app.core.config import Settings
    from app.generation.providers import build_llm_provider

    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        llm_provider="gemini",
        gemini_api_key="test-key",
        llm_connect_timeout_seconds=7.0,
        llm_timeout_seconds=99.0,
        llm_write_timeout_seconds=11.0,
        llm_pool_timeout_seconds=6.0,
    )
    mock_client_cls.return_value.__enter__.return_value = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"answer":"ok","citation_ids":[]}'}}]
    }
    mock_post.return_value = mock_response

    provider = build_llm_provider(settings)
    provider.generate_answer("q", [], timeout_seconds=45.0, max_context_chars=500)

    timeout_arg = mock_client_cls.call_args.kwargs["timeout"]
    assert timeout_arg.connect == 7.0
    assert timeout_arg.read == 45.0
    assert timeout_arg.write == 11.0
    assert timeout_arg.pool == 6.0
