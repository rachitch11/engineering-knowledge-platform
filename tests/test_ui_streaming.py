"""Regression tests for chat streaming assembly and UI session guards."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.ui_streaming import (
    assemble_answer_from_stream_events,
    incremental_chunks_from_cumulative_snapshots,
    iter_write_stream_chunks,
    should_append_assistant_message,
    should_append_user_message,
    simulate_write_stream,
)


def test_greeting_stream_assembles_single_answer(client: TestClient) -> None:
    with client.stream("POST", "/chat/stream", json={"message": "hi", "history": []}) as response:
        assert response.status_code == 200
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    token_events = [event for event in events if event.get("type") == "token"]
    assert len(token_events) >= 1
    answer, final = assemble_answer_from_stream_events(events)
    assert final is not None
    assert answer == final["answer"]
    assert answer.count("Hello") == 1
    assembled_via_stream = simulate_write_stream(
        list(iter_write_stream_chunks(iter(token_events)))
    )
    assert assembled_via_stream == answer


def test_incremental_chunks_do_not_duplicate_on_write_stream() -> None:
    events = [
        {"type": "token", "text": "Hello! "},
        {"type": "token", "text": "I'm the assistant."},
        {"type": "done", "response": {"answer": "Hello! I'm the assistant."}},
    ]
    displayed = simulate_write_stream(list(iter_write_stream_chunks(iter(events))))
    answer, _ = assemble_answer_from_stream_events(events)
    assert displayed == answer
    assert displayed == "Hello! I'm the assistant."


def test_cumulative_snapshots_would_duplicate_without_deltas() -> None:
    snapshots = ["Hello", "Hello! I'm", "Hello! I'm the assistant."]
    buggy = simulate_write_stream(snapshots)
    assert buggy != snapshots[-1]
    assert buggy.count("Hello") > 1

    fixed = simulate_write_stream(
        incremental_chunks_from_cumulative_snapshots(snapshots)
    )
    assert fixed == snapshots[-1]


def test_should_append_user_message_prevents_duplicate_turn() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert should_append_user_message(messages, "hi") is False
    assert should_append_user_message(messages, "hello") is True


def test_should_append_assistant_message_prevents_duplicate_turn() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello there."},
    ]
    assert should_append_assistant_message(messages, "Hello there.") is False


def test_conversation_turn_count_after_successful_exchange() -> None:
    messages: list[dict] = []
    user_text = "hi"
    assistant_text = "Hello! I'm the Engineering Knowledge Intelligence assistant."

    if should_append_user_message(messages, user_text):
        messages.append({"role": "user", "content": user_text})
    if should_append_assistant_message(messages, assistant_text):
        messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
                "citations": [],
            }
        )

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_citations_attached_once_in_final_payload(client: TestClient) -> None:
    client.post(
        "/documents/index",
        files={"file": ("skills.md", b"Python FastAPI skills listed here.", "text/markdown")},
    )
    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "What skills are mentioned?", "history": []},
    ) as response:
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    _, final = assemble_answer_from_stream_events(events)
    assert final is not None
    citations = final.get("citations", [])
    assert len(citations) >= 1
    chunk_ids = [item["chunk_id"] for item in citations]
    assert len(chunk_ids) == len(set(chunk_ids))
