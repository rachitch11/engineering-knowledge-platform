"""Production configuration validation tests."""

from __future__ import annotations

from app.core.config import Settings
from app.core.env_validation import validate_settings_at_startup


def test_production_gemini_missing_key_logs_only() -> None:
    settings = Settings(
        _env_file=None,
        environment="production",
        llm_provider="gemini",
        gemini_api_key=None,
        ephemeral_sessions_enabled=True,
    )
    validate_settings_at_startup(settings)


def test_production_gemini_placeholder_key_does_not_crash() -> None:
    settings = Settings(
        _env_file=None,
        environment="production",
        llm_provider="gemini",
        gemini_api_key="your_gemini_api_key_here",
        ephemeral_sessions_enabled=True,
    )
    validate_settings_at_startup(settings)


def test_production_mock_llm_does_not_require_gemini_key() -> None:
    settings = Settings(
        _env_file=None,
        environment="production",
        llm_provider="mock",
        gemini_api_key=None,
        ephemeral_sessions_enabled=True,
    )
    validate_settings_at_startup(settings)


def test_development_skips_strict_validation() -> None:
    settings = Settings(
        _env_file=None,
        environment="development",
        llm_provider="gemini",
        gemini_api_key=None,
    )
    validate_settings_at_startup(settings)
