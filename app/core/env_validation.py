"""Startup validation for staging and production deployments."""

from __future__ import annotations

import logging
import re

from app.core.config import Settings

logger = logging.getLogger(__name__)

_PLACEHOLDER_KEY = re.compile(
    r"^(?:your[_-]?|changeme|replace[_-]?me|xxx+|dummy).*$",
    re.IGNORECASE,
)


class SettingsValidationError(RuntimeError):
    """Raised when required production configuration is missing or unsafe."""


def _is_missing_secret(value: str | None) -> bool:
    cleaned = (value or "").strip()
    if not cleaned:
        return True
    return bool(_PLACEHOLDER_KEY.match(cleaned))


def validate_settings_at_startup(settings: Settings) -> None:
    """Log production misconfiguration; do not crash the process (keeps /health alive)."""
    if settings.environment not in ("production", "staging"):
        return

    if settings.llm_provider == "gemini" and _is_missing_secret(settings.gemini_api_key):
        logger.error(
            "production_config: GEMINI_API_KEY is missing while LLM_PROVIDER=gemini. "
            "The API will start; /health/llm reports unavailable until the key is set."
        )
    if settings.llm_provider == "openai_compatible" and _is_missing_secret(
        settings.openai_compatible_api_key
    ):
        logger.error(
            "production_config: OPENAI_COMPATIBLE_API_KEY is missing while "
            "LLM_PROVIDER=openai_compatible."
        )

    if not settings.ephemeral_sessions_enabled:
        logger.warning(
            "EPHEMERAL_SESSIONS_ENABLED=false in %s: all visitors share one index. "
            "On Render, set EPHEMERAL_SESSIONS_ENABLED=true and SESSION_STORE_ROOT=/tmp/ephemeral_sessions.",
            settings.environment,
        )
    else:
        logger.info(
            "Ephemeral sessions enabled (store=%s, ttl=%ss). Use a single uvicorn worker.",
            settings.session_store_root,
            settings.session_ttl_seconds,
        )

    if settings.single_user_mode and settings.ephemeral_sessions_enabled:
        logger.warning(
            "SINGLE_USER_MODE=true while ephemeral sessions are enabled; "
            "set SINGLE_USER_MODE=false for multi-visitor deployments."
        )

    if settings.memory_constrained and settings.rerank_enabled:
        logger.warning(
            "memory_constrained_with_rerank: consider RERANK_ENABLED=false on 512MB hosts."
        )
