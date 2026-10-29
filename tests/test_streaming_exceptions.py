"""Regression tests for chat streaming and global exception handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.generation.exceptions import LLMProviderError
from app.generation.providers import MockLLMProvider
from app.main import create_app
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.reranker import MockReranker


class _FailingConversationalLLM(MockLLMProvider):
    def generate_conversational(self, message: str, **kwargs: object) -> str:
        _ = (message, kwargs)
        raise LLMProviderError("Hosted LLM request failed: model unavailable")


@pytest.fixture
def dev_client(tmp_path: Path) -> TestClient:
    get_settings.cache_clear()
    settings = Settings(
        app_name="Engineering Knowledge Intelligence",
        environment="development",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        llm_provider="mock",
        ephemeral_sessions_enabled=False,
    )
    app = create_app(
        settings=settings,
        embedding_model=MockEmbedding(dimension=384),
        reranker=MockReranker(),
        llm_provider=_FailingConversationalLLM(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    get_settings.cache_clear()


def test_chat_stream_llm_failure_returns_502_before_sse_body(
    dev_client: TestClient,
) -> None:
    """LLM errors must not run after StreamingResponse headers (Starlette RuntimeError)."""
    payload = {
        "message": "Explain what a mutex is in one sentence.",
        "history": [],
    }
    with dev_client.stream("POST", "/chat/stream", json=payload) as response:
        assert response.status_code == 502
        body = response.read().decode("utf-8")
    assert body
    detail = json.loads(body)
    assert "Hosted LLM" in detail["detail"]
    assert "data: " not in body


def test_chat_stream_llm_timeout_returns_504(
    dev_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.generation.exceptions import LLMTimeoutError

    def timeout_chat(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise LLMTimeoutError(
            "The language model did not respond in time. Try again, use gemini-3.5-flash-lite, "
            "or increase LLM_TIMEOUT_SECONDS."
        )

    monkeypatch.setattr(
        "app.generation.chat_service.ChatService.chat",
        timeout_chat,
    )
    response = dev_client.post(
        "/chat/stream",
        json={"message": "Explain mutex briefly.", "history": []},
    )
    assert response.status_code == 504
    assert "did not respond in time" in response.json()["detail"]


def test_chat_llm_failure_matches_non_stream_status(dev_client: TestClient) -> None:
    payload = {
        "message": "Explain what a mutex is in one sentence.",
        "history": [],
    }
    response = dev_client.post("/chat", json=payload)
    assert response.status_code == 502


def test_unhandled_exception_handler_logs_in_development(
    dev_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise ValueError("synthetic routing failure")

    monkeypatch.setattr(
        "app.generation.chat_service.ChatService.chat",
        boom,
    )
    caplog.set_level("ERROR")
    response = dev_client.post("/chat", json={"message": "hi", "history": []})
    assert response.status_code == 500
    assert any(
        "unhandled_exception" in record.message
        and "synthetic routing failure" in record.message
        for record in caplog.records
    )


def test_iter_chat_sse_emits_error_event_when_done_payload_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.generation.chat_service import ChatService
    from app.generation.intent import greeting_response
    from app.generation.models import ChatResponse
    from app.retrieval.indexer import IndexService
    from app.retrieval.search import SearchService

    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        llm_provider="mock",
    )
    index = IndexService.from_settings(settings, embedding_model=MockEmbedding())
    service = ChatService(
        SearchService(index, settings),
        settings,
        llm_provider=MockLLMProvider(),
    )
    response = ChatResponse(
        question="hi",
        answer=greeting_response(),
        sufficient_evidence=False,
        citations=[],
        evidence=[],
        retrieval_strategy="hybrid",
        candidate_count=0,
        rerank_applied=False,
        intent="greeting",
        response_mode="general",
        retrieval_query="hi",
        grounded=False,
        timings_ms={"intent": 0},
        conversation_context_included=False,
    )

    original_dumps = json.dumps

    def flaky_dumps(obj: object, *args: object, **kwargs: object) -> str:
        if isinstance(obj, dict) and obj.get("type") == "done":
            raise RuntimeError("serialization failed")
        return original_dumps(obj, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("app.generation.chat_service.json.dumps", flaky_dumps)

    parsed: list[dict] = []
    for chunk in service.iter_chat_sse(response):
        for line in chunk.splitlines():
            if line.startswith("data: "):
                parsed.append(json.loads(line[6:]))

    assert any(event.get("type") == "token" for event in parsed)
    assert parsed[-1]["type"] == "error"
    assert parsed[-1]["detail"]
