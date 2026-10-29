"""Tests for Streamlit backend client and error formatting."""

from __future__ import annotations

import json

import httpx
import pytest

from app.ui_errors import format_backend_error


def test_format_backend_error_uses_json_detail() -> None:
    response = httpx.Response(
        502,
        json={"detail": "Hosted LLM request failed: model unavailable"},
    )
    exc = httpx.HTTPStatusError("fail", request=httpx.Request("POST", "http://test/chat/stream"), response=response)
    message = format_backend_error(exc)
    assert "502" in message
    assert "model unavailable" in message
    assert "GEMINI" not in message


def test_format_backend_error_truncates_unknown_body() -> None:
    response = httpx.Response(500, text="x" * 1000)
    exc = httpx.HTTPStatusError("fail", request=httpx.Request("POST", "http://test/chat"), response=response)
    message = format_backend_error(exc)
    assert "500" in message
    assert len(message) < 200
