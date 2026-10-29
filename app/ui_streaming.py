"""Pure helpers for SSE chat streaming and Streamlit write_stream assembly."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def iter_write_stream_chunks(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    """Yield incremental text pieces suitable for ``st.write_stream`` (append semantics).

    Backend contract: ``type=token`` events carry incremental ``text`` fragments.
    ``type=done`` is metadata only and is not yielded.
    """
    for event in events:
        if event.get("type") != "token":
            continue
        piece = str(event.get("text", "") or "")
        if piece:
            yield piece


def assemble_answer_from_stream_events(
    events: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    """Build full answer text and optional final payload from parsed SSE events."""
    accumulated = ""
    final_payload: dict[str, Any] | None = None
    for event in events:
        if event.get("type") == "token":
            accumulated += str(event.get("text", "") or "")
        elif event.get("type") == "done":
            response = event.get("response")
            if isinstance(response, dict):
                final_payload = response
    if final_payload is not None:
        answer = str(final_payload.get("answer", accumulated))
    else:
        answer = accumulated
    return answer, final_payload


def simulate_write_stream(chunks: list[str]) -> str:
    """Mirror Streamlit ``write_stream`` string append behavior for tests."""
    return "".join(chunks)


def incremental_chunks_from_cumulative_snapshots(snapshots: list[str]) -> list[str]:
    """Convert cumulative snapshot yields into incremental deltas (replace-style fix)."""
    deltas: list[str] = []
    previous = ""
    for snapshot in snapshots:
        if snapshot.startswith(previous):
            delta = snapshot[len(previous) :]
        else:
            delta = snapshot
        if delta:
            deltas.append(delta)
        previous = snapshot if snapshot.startswith(previous) else previous + delta
    return deltas


def should_append_user_message(messages: list[dict[str, Any]], content: str) -> bool:
    """Return False if the latest user turn already matches (rerun-safe)."""
    if not messages:
        return True
    last = messages[-1]
    return not (last.get("role") == "user" and last.get("content") == content)


def should_append_assistant_message(
    messages: list[dict[str, Any]],
    content: str,
) -> bool:
    """Return False if the latest assistant turn already matches (rerun-safe)."""
    if not messages:
        return True
    last = messages[-1]
    return not (last.get("role") == "assistant" and last.get("content") == content)
