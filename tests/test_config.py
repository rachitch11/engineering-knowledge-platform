"""Tests for configuration loading."""

from app.core.config import Settings, get_settings


def test_settings_load_defaults() -> None:
    settings = Settings(
        _env_file=None,
        environment="development",
    )
    assert settings.app_name == "Engineering Knowledge Intelligence"
    assert settings.environment == "development"
    assert settings.api_port == 8000
    assert settings.llm_provider == "gemini"
    assert settings.ephemeral_sessions_enabled is True


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
