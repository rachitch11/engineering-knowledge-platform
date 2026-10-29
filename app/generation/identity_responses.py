"""Application and model identity detection and config-backed responses."""

from __future__ import annotations

import re

from app.core.config import Settings


def _normalized(message: str) -> str:
    from app.generation.intent import normalize_intent_text

    return normalize_intent_text(message)

_APPLICATION_PATTERNS = (
    "who are you",
    "what are you",
    "what is this app",
    "what is this application",
    "what does this app do",
    "what does this application do",
    "what is your purpose",
    "what is your role",
    "what can you help me with",
    "what can you help with",
    "tell me about yourself",
    "introduce yourself",
)

_MODEL_IDENTITY_PATTERN = re.compile(
    r"(which|what)\s+(ai\s+)?(model|llm)\b|"
    r"\bmodel\s+(are\s+you|is\s+this|am\s+i)\b|"
    r"\b(which|what)\s+provider\b|"
    r"\bllama\s+or\s+gpt\b|"
    r"\bgpt\s+or\s+llama\b|"
    r"\bam\s+i\s+talking\s+to\b|"
    r"\btalking\s+to\s+(which|what)\b|"
    r"\bwhich\s+llm\b|"
    r"\bwhat\s+llm\b",
    re.IGNORECASE,
)


def is_application_identity_question(message: str) -> bool:
    normalized = _normalized(message)
    normalized = re.sub(r"[?.!]+$", "", normalized).strip()
    if normalized in ("who are you", "what are you"):
        return True
    return any(
        normalized == phrase or normalized.startswith(phrase + " ")
        for phrase in _APPLICATION_PATTERNS
    )


def is_model_identity_question(message: str) -> bool:
    normalized = _normalized(message)
    normalized = re.sub(r"\btoo\b", "to", normalized)
    return _MODEL_IDENTITY_PATTERN.search(normalized) is not None


def configured_llm_identity(settings: Settings) -> tuple[str, str | None]:
    """Return (provider_label, model_name_or_none) from settings only."""
    provider = settings.llm_provider
    if provider == "gemini":
        return ("gemini", settings.gemini_model)
    if provider == "ollama":
        return ("ollama", settings.ollama_model)
    if provider == "openai_compatible":
        return ("openai_compatible", settings.openai_compatible_model)
    if provider == "mock":
        return ("mock", "mock-llm")
    return (provider, None)


def application_identity_response(settings: Settings) -> str:
    return (
        f"I am the assistant for **{settings.app_name}**.\n\n"
        "This is a local engineering knowledge app: you can chat for general questions, "
        "upload PDFs or Markdown, index them, and ask document-grounded questions with citations. "
        "I separate general answers from facts taken from your indexed files."
    )


def model_identity_response(settings: Settings) -> str:
    provider, model = configured_llm_identity(settings)
    if not model or not str(model).strip():
        return (
            "I cannot verify the active LLM model from configuration. "
            f"The configured provider is `{provider}`, but no model name is set. "
            "Check your `.env` or server settings."
        )
    return (
            "This chat uses configured runtime settings (not a guess from the model):\n"
            f"- **Provider:** `{provider}`\n"
            f"- **Model:** `{model}`\n\n"
            "Change these via environment variables such as `LLM_PROVIDER`, `GEMINI_MODEL`, "
        "or `OLLAMA_MODEL`."
    )
