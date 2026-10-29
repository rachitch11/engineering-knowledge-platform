"""Render host configuration defaults."""

from __future__ import annotations

import os
from unittest.mock import patch

from app.core.config import Settings


def test_render_defaults_enable_ephemeral_and_disable_rerank() -> None:
    env = {
        "RENDER": "true",
        "ENVIRONMENT": "production",
        "MEMORY_CONSTRAINED": "true",
        "RERANK_ENABLED": "false",
        "EPHEMERAL_SESSIONS_ENABLED": "true",
        "SESSION_STORE_ROOT": "/tmp/ephemeral_sessions",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = Settings(_env_file=None)
    assert settings.ephemeral_sessions_enabled is True
    assert settings.rerank_enabled is False
    assert settings.memory_constrained is True
    assert settings.session_store_root == "/tmp/ephemeral_sessions"


def test_render_auto_defaults_when_env_vars_omitted() -> None:
    env = {"RENDER": "true", "ENVIRONMENT": "production"}
    with patch.dict(os.environ, env, clear=True):
        settings = Settings(_env_file=None)
    assert settings.rerank_enabled is False
    assert settings.memory_constrained is True
    assert settings.session_store_root == "/tmp/ephemeral_sessions"


def test_production_gemini_missing_key_does_not_raise() -> None:
    from app.core.env_validation import validate_settings_at_startup

    settings = Settings(
        _env_file=None,
        environment="production",
        llm_provider="gemini",
        gemini_api_key=None,
        ephemeral_sessions_enabled=True,
    )
    validate_settings_at_startup(settings)
