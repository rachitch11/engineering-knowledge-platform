"""Production configuration validation tests."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.env_validation import SettingsValidationError, validate_settings_at_startup


def test_production_gemini_requires_api_key() -> None:
    settings = Settings(
        environment="production",
        llm_provider="gemini",
        gemini_api_key=None,
        ephemeral_sessions_enabled=True,
    )
    with pytest.raises(SettingsValidationError):
        validate_settings_at_startup(settings)


def test_production_gemini_rejects_placeholder_key() -> None:
    settings = Settings(
        environment="production",
        llm_provider="gemini",
        gemini_api_key="your_gemini_api_key_here",
        ephemeral_sessions_enabled=True,
    )
    with pytest.raises(SettingsValidationError):
        validate_settings_at_startup(settings)


def test_production_mock_llm_does_not_require_gemini_key() -> None:
    settings = Settings(
        environment="production",
        llm_provider="mock",
        gemini_api_key=None,
        ephemeral_sessions_enabled=True,
    )
    validate_settings_at_startup(settings)


def test_development_skips_strict_validation() -> None:
    settings = Settings(
        environment="development",
        llm_provider="gemini",
        gemini_api_key=None,
    )
    validate_settings_at_startup(settings)
