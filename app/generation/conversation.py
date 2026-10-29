"""Conversation history handling and deterministic query rewriting."""

from __future__ import annotations

import re

from app.generation.models import ChatMessage


def trim_history(history: list[ChatMessage], max_messages: int) -> list[ChatMessage]:
    if max_messages <= 0:
        return []
    return history[-max_messages:]


def build_conversation_context(
    history: list[ChatMessage],
    *,
    max_messages: int,
    max_chars: int,
) -> str:
    """Format recent chat for the generation prompt (not treated as evidence)."""
    recent = trim_history(history, max_messages)
    if not recent:
        return ""
    lines: list[str] = []
    used = 0
    for message in recent:
        line = f"{message.role.upper()}: {message.content}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def needs_retrieval_rewrite(message: str, history: list[ChatMessage]) -> bool:
    if not history:
        return False
    text = message.strip().lower()
    if len(text.split()) >= 8 and not re.search(r"\b(that|those|it|they|them)\b", text):
        return False
    if len(text.split()) <= 6:
        return True
    return bool(re.search(r"\b(that|those|it|they|them|why|which)\b", text))


def rewrite_retrieval_query(message: str, history: list[ChatMessage]) -> str:
    """Expand follow-ups into standalone retrieval queries without an LLM."""
    if not needs_retrieval_rewrite(message, history):
        return message.strip()

    last_user = _last_user_message(history)
    if not last_user:
        return message.strip()

    cleaned = message.strip()
    lowered = cleaned.lower()
    if lowered.startswith(("which", "why", "what about", "how about")):
        return f"{last_user} {cleaned}"
    if re.search(r"\b(that|those|it|they|them)\b", lowered):
        return f"{last_user} {cleaned}"
    if len(cleaned.split()) <= 4:
        return f"{last_user} {cleaned}"
    return cleaned


def _last_user_message(history: list[ChatMessage]) -> str | None:
    for message in reversed(history):
        if message.role == "user":
            return message.content.strip()
    return None
