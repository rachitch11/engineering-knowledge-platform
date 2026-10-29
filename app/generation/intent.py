"""Deterministic intent routing before expensive retrieval."""

from __future__ import annotations

import re
from enum import Enum

from app.generation.models import ChatMessage


class MessageIntent(str, Enum):
    GREETING = "greeting"
    APPLICATION_IDENTITY = "application_identity"
    MODEL_IDENTITY = "model_identity"
    CAPABILITIES = "capabilities"
    GENERAL_CHAT = "general_chat"
    DOCUMENT_RAG = "document_rag"
    DOCUMENT_FOLLOWUP = "document_followup"
    MIXED = "mixed"


class ResponseMode(str, Enum):
    GENERAL = "general"
    DOCUMENT = "document"
    MIXED = "mixed"


_GREETING_PATTERN = re.compile(
    r"^(hi+|hello+|hey+|hii+|howdy|good (morning|afternoon|evening))(?:[!.,\s]|$)",
    re.IGNORECASE,
)
_SHORT_GREETINGS = {"hi", "hello", "hey", "hii", "hiii", "yo", "sup"}

_CAPABILITIES_PHRASES = (
    "what can you do",
    "how can i use",
    "how do i use",
    "help me use",
    "how does this work",
)

_DOCUMENT_HINTS = (
    "resume",
    "document",
    "upload",
    "uploaded",
    "indexed",
    "pdf",
    "manual",
    "report",
    "file",
    "page",
    "cite",
    "source",
    "chunk",
)

_PERSONAL_DOCUMENT_PHRASES = (
    "my resume",
    "my cv",
    "my name",
    "what's my name",
    "what is my name",
    "in my document",
    "in my file",
    "my uploaded",
    "from my pdf",
    "my education",
    "my experience",
    "summarize the education",
    "education section",
)

_MIXED_PHRASES = (
    "compare my",
    "in my resume and",
    "from my document and",
    "based on my file",
    "using my document explain",
    "in my pdf and",
)

_INDEXED_CONTENT_CUES = (
    "mentioned",
    "listed",
    "summarize",
    "summary of",
    "section",
    "skills",
    "experience",
    "education",
    "languages",
    "projects",
)

_FOLLOWUP_STARTERS = (
    "which",
    "why",
    "what about",
    "how about",
    "and ",
    "also ",
    "summarize",
    "explain that",
    "tell me more",
    "explain further",
)

_GENERAL_FACT_PATTERN = re.compile(
    r"\b(capital of|population of|currency of|largest city in|"
    r"who invented|when was|where is)\b",
    re.IGNORECASE,
)


def normalize_intent_text(text: str) -> str:
    """Lowercase and normalize punctuation for intent rules."""
    lowered = " ".join(text.split()).strip().lower()
    for mark in ("\u2019", "\u2018", "\u0060", "\u00b4"):
        lowered = lowered.replace(mark, "'")
    lowered = lowered.replace("what's", "what is").replace("who's", "who is")
    lowered = re.sub(r"\btoo\b", "to", lowered)
    return lowered


def is_standalone_general_question(message: str) -> bool:
    """True when the message should never use document retrieval."""
    from app.generation.identity_responses import (
        is_application_identity_question,
        is_model_identity_question,
    )

    if is_application_identity_question(message) or is_model_identity_question(message):
        return True
    normalized = normalize_intent_text(message)
    if _references_personal_documents(normalized) or _mentions_documents(normalized):
        return False
    if _looks_like_general_knowledge(normalized):
        return True
    return _GENERAL_FACT_PATTERN.search(normalized) is not None


def classify_intent(
    message: str,
    history: list[ChatMessage],
    *,
    indexed_chunk_count: int = 0,
) -> MessageIntent:
    from app.generation.identity_responses import (
        is_application_identity_question,
        is_model_identity_question,
    )

    lowered = normalize_intent_text(message)

    if lowered in _SHORT_GREETINGS or _GREETING_PATTERN.match(lowered):
        return MessageIntent.GREETING

    if is_application_identity_question(message):
        return MessageIntent.APPLICATION_IDENTITY

    if is_model_identity_question(message):
        return MessageIntent.MODEL_IDENTITY

    if any(phrase in lowered for phrase in _CAPABILITIES_PHRASES):
        return MessageIntent.CAPABILITIES

    if any(phrase in lowered for phrase in _MIXED_PHRASES):
        return MessageIntent.MIXED

    if is_standalone_general_question(message):
        return MessageIntent.GENERAL_CHAT

    if _looks_like_follow_up(lowered, history):
        if _history_in_document_thread(history, indexed_chunk_count):
            return MessageIntent.DOCUMENT_FOLLOWUP
        return MessageIntent.GENERAL_CHAT

    if _references_personal_documents(lowered) or _mentions_documents(lowered):
        if _also_needs_general_knowledge(lowered):
            return MessageIntent.MIXED
        return MessageIntent.DOCUMENT_RAG

    if indexed_chunk_count > 0 and _looks_like_indexed_content_question(lowered):
        return MessageIntent.DOCUMENT_RAG

    return MessageIntent.GENERAL_CHAT


def intent_to_response_mode(intent: MessageIntent) -> ResponseMode:
    if intent in (
        MessageIntent.GREETING,
        MessageIntent.APPLICATION_IDENTITY,
        MessageIntent.MODEL_IDENTITY,
        MessageIntent.CAPABILITIES,
        MessageIntent.GENERAL_CHAT,
    ):
        return ResponseMode.GENERAL
    if intent == MessageIntent.MIXED:
        return ResponseMode.MIXED
    return ResponseMode.DOCUMENT


def _looks_like_general_knowledge(text: str) -> bool:
    normalized = normalize_intent_text(text)
    patterns = (
        "explain ",
        "define ",
        "what is a ",
        "what is an ",
        "what is the capital",
        "what is the difference",
        "capital of ",
        "how does a ",
        "how does ",
        "how do ",
        "how can ",
        "write a ",
        "implement ",
        "in general",
        "not from my document",
    )
    if any(normalized.startswith(prefix) for prefix in patterns):
        return True
    if any(phrase in normalized for phrase in patterns):
        return True
    if len(normalized.split()) <= 12 and normalized.endswith("?"):
        if normalized.startswith(
            ("what is ", "why is ", "when was ", "where is ")
        ):
            if not _references_personal_documents(text):
                return True
        if normalized.startswith("who is "):
            remainder = normalized[7:].strip(" ?.!")
            if len(remainder.split()) > 2:
                return True
    return False


def _also_needs_general_knowledge(text: str) -> bool:
    markers = (
        " and explain",
        " and compare",
        " compared to",
        " versus industry",
        " in general",
        " best practices",
    )
    return any(marker in text for marker in markers)


def _looks_like_indexed_content_question(text: str) -> bool:
    from app.generation.identity_responses import (
        is_application_identity_question,
        is_model_identity_question,
    )

    if is_application_identity_question(text) or is_model_identity_question(text):
        return False
    if _looks_like_general_knowledge(text):
        return False
    if any(cue in text for cue in _INDEXED_CONTENT_CUES):
        return True
    normalized = normalize_intent_text(text)
    if normalized.startswith(("what ", "which ", "list ", "find ")):
        return True
    if normalized.startswith("who "):
        if normalized in ("who are you",) or normalized.startswith("who are you "):
            return False
        return True
    return False


def greeting_response() -> str:
    return (
        "Hello! I'm the Engineering Knowledge Intelligence assistant.\n\n"
        "Here's how you can use me:\n"
        "1. Upload a PDF, Markdown, or HTML file in the sidebar and click **Index uploaded file**.\n"
        "2. Ask questions about your indexed content — I'll answer using retrieved passages and show citations.\n"
        "3. Ask general engineering questions anytime — I answer directly without requiring indexed files.\n"
        "4. Ask follow-ups naturally (for example, \"Which frameworks?\" after a skills question).\n\n"
        "I distinguish general chat from document-grounded answers: claims about your files always come from retrieved sources."
    )


def capabilities_response() -> str:
    return (
        "I help you explore engineering documents you've indexed locally and answer general questions.\n\n"
        "**General chat:** explanations, coding help, and everyday questions (no citations required).\n"
        "**Document workflow:** upload → index → ask questions → review citations.\n"
        "**Mixed questions:** I separate facts from your files from general model knowledge.\n"
        "**Privacy:** retrieval runs locally; answers use your configured LLM provider (Gemini by default)."
    )


def _references_personal_documents(text: str) -> bool:
    return any(phrase in text for phrase in _PERSONAL_DOCUMENT_PHRASES)


def _mentions_documents(text: str) -> bool:
    return any(token in text for token in _DOCUMENT_HINTS)


def _history_mentions_documents(history: list[ChatMessage]) -> bool:
    for message in history[-6:]:
        if _mentions_documents(message.content.lower()):
            return True
        if _references_personal_documents(message.content.lower()):
            return True
    return False


def _history_in_document_thread(
    history: list[ChatMessage],
    indexed_chunk_count: int,
) -> bool:
    if not history:
        return False
    if indexed_chunk_count <= 0:
        return False
    if _history_mentions_documents(history):
        return True
    last_user = _last_user_text(history)
    if last_user and _looks_like_indexed_content_question(last_user.lower()):
        return True
    return False


def _last_user_text(history: list[ChatMessage]) -> str | None:
    for message in reversed(history):
        if message.role == "user":
            return message.content.strip()
    return None


def _looks_like_follow_up(text: str, history: list[ChatMessage]) -> bool:
    from app.generation.identity_responses import (
        is_application_identity_question,
        is_model_identity_question,
    )

    if not history:
        return False
    if is_application_identity_question(text) or is_model_identity_question(text):
        return False
    if is_standalone_general_question(text):
        return False
    if _looks_like_general_knowledge(text) and not _references_personal_documents(text):
        return False
    if text.startswith("which"):
        if is_model_identity_question(text):
            return False
        if _looks_like_indexed_content_question(text):
            return True
        if len(text.split()) <= 5 and not is_model_identity_question(text):
            return True
        return False
    if any(text.startswith(prefix) for prefix in _FOLLOWUP_STARTERS if prefix != "which"):
        return True
    if re.search(r"\b(that|those|it|they|them|previous|above)\b", text):
        return True
    word_count = len(text.split())
    if word_count <= 3:
        return True
    return False
