"""Tests for general, document, and mixed response modes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.generation.intent import MessageIntent
from app.generation.providers import LLMProviderError, build_llm_provider
from app.main import create_app
from app.generation.providers import MockLLMProvider
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.reranker import MockReranker


def test_general_question_without_indexed_documents(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "What is the capital of Japan?", "history": []},
    )
    body = response.json()
    assert body["response_mode"] == "general"
    assert body["intent"] == MessageIntent.GENERAL_CHAT.value
    assert body["grounded"] is False
    assert body["candidate_count"] == 0
    assert "tokyo" in body["answer"].lower() or "general" in body["answer"].lower()


def test_general_coding_question_skips_retrieval(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"search": False}

    def fake_search(*args, **kwargs):
        called["search"] = True
        raise AssertionError("retrieval should not run")

    monkeypatch.setattr(
        "app.generation.chat_service.SearchService.search",
        fake_search,
    )
    response = client.post(
        "/chat",
        json={"message": "Explain Python decorators briefly.", "history": []},
    )
    assert response.status_code == 200
    assert called["search"] is False
    assert response.json()["response_mode"] == "general"


def test_document_specific_missing_fact(client: TestClient) -> None:
    client.post(
        "/documents/index",
        files={
            "file": (
                "resume.md",
                b"Name: Alex Developer. Skills: Python.",
                "text/markdown",
            )
        },
    )
    response = client.post(
        "/chat",
        json={"message": "What is my middle name in my resume?", "history": []},
    )
    body = response.json()
    assert body["response_mode"] == "document"
    assert body["grounded"] is True
    assert body["sufficient_evidence"] is False
    assert "searched your indexed documents" in body["answer"].lower()


def test_mixed_question(client: TestClient) -> None:
    client.post(
        "/documents/index",
        files={"file": ("resume.md", b"Skills: Python, FastAPI.", "text/markdown")},
    )
    response = client.post(
        "/chat",
        json={
            "message": "Compare my resume skills and explain FastAPI best practices",
            "history": [],
        },
    )
    body = response.json()
    assert body["intent"] == MessageIntent.MIXED.value
    assert body["response_mode"] == "mixed"
    assert "From your documents" in body["answer"]
    assert "General knowledge" in body["answer"]


def test_follow_up_general_after_general(client: TestClient) -> None:
    history = [
        {"role": "user", "content": "Explain what a REST API is."},
        {
            "role": "assistant",
            "content": "A REST API uses HTTP resources and standard verbs.",
        },
    ]
    response = client.post(
        "/chat",
        json={"message": "Why?", "history": history},
    )
    body = response.json()
    assert body["response_mode"] == "general"
    assert body["candidate_count"] == 0


def test_general_trivia_after_document_thread(client: TestClient) -> None:
    client.post(
        "/documents/index",
        files={"file": ("resume.pdf", b"Name: Alex. Skills: Python.", "application/pdf")},
    )
    history = [
        {"role": "user", "content": "What skills are in my resume?"},
        {"role": "assistant", "content": "Python appears in your indexed resume."},
    ]
    for question in (
        "What's the capital of Japan?",
        "What\u2019s the capital of Japan",
    ):
        response = client.post(
            "/chat",
            json={"message": question, "history": history},
        )
        body = response.json()
        assert body["response_mode"] == "general", question
        assert body["intent"] == MessageIntent.GENERAL_CHAT.value, question
        assert body["grounded"] is False, question
        assert body["citations"] == [], question
        assert "reference chunks" not in body["answer"].lower(), question


def test_openai_provider_requires_api_key(tmp_path) -> None:
    get_settings.cache_clear()
    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        llm_provider="openai_compatible",
        openai_compatible_api_key=None,
    )
    with pytest.raises(LLMProviderError, match="OPENAI_COMPATIBLE_API_KEY"):
        build_llm_provider(settings)
    get_settings.cache_clear()


def test_provider_error_surfaces_as_http_error(tmp_path) -> None:
    get_settings.cache_clear()
    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        llm_provider="mock",
        ephemeral_sessions_enabled=False,
    )

    class BrokenLLM(MockLLMProvider):
        def generate_conversational(self, *args, **kwargs):
            raise LLMProviderError("Provider unavailable")

    app = create_app(
        settings=settings,
        embedding_model=MockEmbedding(),
        reranker=MockReranker(),
        llm_provider=BrokenLLM(),
    )
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "Explain Python decorators.", "history": []},
        )
        assert response.status_code == 502
        assert "Provider unavailable" in response.json()["detail"]
    get_settings.cache_clear()
