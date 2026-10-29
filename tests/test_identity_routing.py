"""Regression tests for application/model identity and routing via POST /chat."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.generation.identity_responses import model_identity_response
from app.generation.intent import MessageIntent
from app.generation.providers import MockLLMProvider
from app.main import create_app
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.reranker import MockReranker


def _index_resume(client: TestClient) -> None:
    response = client.post(
        "/documents/index",
        files={
            "file": (
                "Kush_IT_Resume.md",
                b"Name: Kush Sharma. Skills: Python, FastAPI, React.",
                "text/markdown",
            )
        },
    )
    assert response.status_code == 200, response.text
    assert response.json().get("chunk_count", 0) > 0


def test_capital_of_india_general_no_retrieval(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "What is the capital of India?", "history": []},
    )
    body = response.json()
    assert body["response_mode"] == "general"
    assert body["grounded"] is False
    assert body["candidate_count"] == 0
    assert body["citations"] == []


def test_who_are_you_application_identity(client: TestClient) -> None:
    _index_resume(client)
    history = [
        {"role": "user", "content": "Summarize my resume skills"},
        {"role": "assistant", "content": "Python and FastAPI are listed."},
    ]
    response = client.post(
        "/chat",
        json={"message": "who are you", "history": history},
    )
    body = response.json()
    assert body["intent"] == MessageIntent.APPLICATION_IDENTITY.value
    assert body["response_mode"] == "general"
    assert body["grounded"] is False
    assert body["candidate_count"] == 0
    assert "Engineering Knowledge" in body["answer"]


def test_model_identity_from_config(client: TestClient, test_settings: Settings) -> None:
    _index_resume(client)
    response = client.post(
        "/chat",
        json={
            "message": "Which AI model am I talking to?",
            "history": [{"role": "user", "content": "What is on my resume?"}],
        },
    )
    body = response.json()
    assert body["intent"] == MessageIntent.MODEL_IDENTITY.value
    assert body["grounded"] is False
    assert body["candidate_count"] == 0
    assert "mock" in body["answer"].lower()
    assert "mock-llm" in body["answer"].lower()
    assert test_settings.llm_provider in body["answer"]


def test_llama_or_gpt_reports_config_not_guess(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "Are you using Llama or GPT?", "history": []},
    )
    body = response.json()
    assert body["intent"] == MessageIntent.MODEL_IDENTITY.value
    assert "configured runtime settings" in body["answer"].lower()
    assert body["citations"] == []


def test_model_typo_talking_too(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "which model i am talking too", "history": []},
    )
    body = response.json()
    assert body["intent"] == MessageIntent.MODEL_IDENTITY.value
    assert body["candidate_count"] == 0


def test_japan_after_resume_stays_general(client: TestClient) -> None:
    _index_resume(client)
    history = [
        {"role": "user", "content": "What skills are on my resume?"},
        {"role": "assistant", "content": "Python, FastAPI, and React."},
    ]
    response = client.post(
        "/chat",
        json={"message": "What is the capital of Japan?", "history": history},
    )
    body = response.json()
    assert body["response_mode"] == "general"
    assert body["grounded"] is False
    assert body["citations"] == []


def test_frameworks_follow_up_uses_documents(client: TestClient) -> None:
    _index_resume(client)
    history = [
        {"role": "user", "content": "What skills are on my resume?"},
        {"role": "assistant", "content": "Python, FastAPI, and React."},
    ]
    response = client.post(
        "/chat",
        json={"message": "Which frameworks are listed?", "history": history},
    )
    body = response.json()
    assert body["response_mode"] == "document"
    assert body["intent"] == MessageIntent.DOCUMENT_FOLLOWUP.value
    assert body["grounded"] is True
    assert body["candidate_count"] > 0


def test_who_is_kush_uses_resume_when_indexed(client: TestClient) -> None:
    _index_resume(client)
    response = client.post(
        "/chat",
        json={"message": "Who is Kush?", "history": []},
    )
    body = response.json()
    assert body["grounded"] is True
    assert body["candidate_count"] > 0


def test_model_identity_when_model_name_missing(tmp_path) -> None:
    get_settings.cache_clear()
    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        llm_provider="gemini",
        gemini_model="",
    )
    text = model_identity_response(settings)
    assert "cannot verify" in text.lower()
    get_settings.cache_clear()


def test_identity_never_calls_search(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"search": False}

    def fake_search(*args, **kwargs):
        called["search"] = True
        raise AssertionError("retrieval must not run")

    monkeypatch.setattr(
        "app.generation.chat_service.SearchService.search",
        fake_search,
    )
    for message in ("who are you", "Which LLM powers this application?"):
        response = client.post("/chat", json={"message": message, "history": []})
        assert response.status_code == 200
    assert called["search"] is False
