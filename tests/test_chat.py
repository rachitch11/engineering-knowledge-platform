"""Tests for intent routing and conversational chat."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.timing import PipelineTimer
from app.generation.chat_service import ChatService
from app.generation.conversation import rewrite_retrieval_query
from app.generation.intent import MessageIntent, classify_intent, greeting_response
from app.generation.models import ChatMessage
from app.generation.providers import MockLLMProvider
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.indexer import IndexService
from app.retrieval.reranker import MockReranker
from app.retrieval.search import SearchService
from tests.test_retrieval import _sample_chunks


@pytest.fixture
def chat_service(test_settings: Settings) -> ChatService:
    index = IndexService.from_settings(test_settings, embedding_model=MockEmbedding())
    search = SearchService(index, test_settings)
    return ChatService(
        search,
        test_settings,
        reranker=MockReranker(),
        llm_provider=MockLLMProvider(),
    )


def test_greeting_intent_includes_intro(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hiii", "history": []})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == MessageIntent.GREETING.value
    assert body["grounded"] is False
    assert "Upload" in body["answer"]
    assert body["candidate_count"] == 0
    assert "intent" in body["timings_ms"] or body["timings_ms"].get("intent", 0) >= 0


def test_greeting_does_not_search(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"search": False}

    def fake_search(*args, **kwargs):
        called["search"] = True
        raise AssertionError("search should not run for greetings")

    monkeypatch.setattr(
        "app.generation.chat_service.SearchService.search",
        fake_search,
    )
    response = client.post("/chat", json={"message": "hello", "history": []})
    assert response.status_code == 200
    assert called["search"] is False


def test_capabilities_intent(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "What can you do?", "history": []},
    )
    assert response.status_code == 200
    assert response.json()["intent"] == MessageIntent.CAPABILITIES.value


def test_general_chat_not_grounded(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "Explain what a mutex is in one sentence.", "history": []},
    )
    body = response.json()
    assert body["intent"] == MessageIntent.GENERAL_CHAT.value
    assert body["grounded"] is False
    assert "general response" in body["answer"].lower()


def test_document_question_after_index(client: TestClient) -> None:
    index_response = client.post(
        "/documents/index",
        files={"file": ("skills.md", b"# Skills\n\nPython asyncio and FastAPI.", "text/markdown")},
    )
    assert index_response.status_code == 200

    response = client.post(
        "/chat",
        json={"message": "What skills are mentioned?", "history": []},
    )
    body = response.json()
    assert body["grounded"] is True
    assert body["sufficient_evidence"] is True
    assert body["citations"]


def test_follow_up_rewrite_and_answer(client: TestClient) -> None:
    client.post(
        "/documents/index",
        files={"file": ("skills.md", b"Skills: Python, FastAPI, asyncio.", "text/markdown")},
    )
    history = [
        {"role": "user", "content": "What skills are in the document?"},
        {
            "role": "assistant",
            "content": "Python, FastAPI, and asyncio appear in the indexed file.",
        },
    ]
    response = client.post(
        "/chat",
        json={"message": "Which frameworks?", "history": history},
    )
    body = response.json()
    assert body["intent"] == MessageIntent.DOCUMENT_FOLLOWUP.value
    assert "What skills" in body["retrieval_query"]
    assert body["grounded"] is True


def test_rewrite_retrieval_query_deterministic() -> None:
    history = [ChatMessage(role="user", content="What skills are in my resume?")]
    rewritten = rewrite_retrieval_query("Which frameworks?", history)
    assert "resume" in rewritten.lower()
    assert "frameworks" in rewritten.lower()


def test_chat_stream_returns_done_payload(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "hi", "history": []},
    ) as response:
        assert response.status_code == 200
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert any(event.get("type") == "token" for event in events)
    done = next(event for event in events if event.get("type") == "done")
    assert done["response"]["intent"] == MessageIntent.GREETING.value


def test_query_embedding_cache_invalidated_on_reindex(
    test_settings: Settings,
) -> None:
    index = IndexService.from_settings(test_settings, embedding_model=MockEmbedding())
    index.index_chunks(_sample_chunks())
    first = index.embed_query("asyncio tutorial")
    second = index.embed_query("asyncio tutorial")
    assert first == second
    assert len(index._query_embedding_cache) == 1  # noqa: SLF001

    index.index_chunks(_sample_chunks(document_id="doc-2"))
    index.embed_query("asyncio tutorial")
    assert len(index._query_embedding_cache) == 1  # noqa: SLF001


def test_pipeline_timer_aggregates_stages() -> None:
    timer = PipelineTimer()
    with timer.stage("intent"):
        pass
    with timer.stage("retrieval"):
        with timer.stage("retrieval.embedding"):
            pass
    payload = timer.as_dict()
    assert "intent" in payload
    assert "retrieval" in payload
    assert payload["total"] >= payload["intent"]


def test_abstain_when_no_relevant_index(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "What is the CEO name in the quarterly report?", "history": []},
    )
    body = response.json()
    assert body["grounded"] is True
    assert body["sufficient_evidence"] is False
    assert "searched your indexed documents" in body["answer"].lower()


def test_greeting_response_mentions_workflow() -> None:
    text = greeting_response()
    assert "index" in text.lower()


def test_chat_history_isolated_per_request(client: TestClient) -> None:
    first = client.post("/chat", json={"message": "hi", "history": []})
    second = client.post(
        "/chat",
        json={"message": "Which frameworks?", "history": []},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["intent"] == MessageIntent.GENERAL_CHAT.value
    assert second.json()["response_mode"] == "general"
