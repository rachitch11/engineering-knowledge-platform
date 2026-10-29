"""User-facing HTTP error formatting for the Streamlit UI."""

from __future__ import annotations

import json

import httpx


def format_backend_error(exc: httpx.HTTPStatusError) -> str:
    """Return a safe, user-readable message without secrets or raw tracebacks."""
    status = exc.response.status_code
    detail: str | None = None
    try:
        payload = exc.response.json()
        if isinstance(payload, dict):
            raw = payload.get("detail")
            if isinstance(raw, str):
                detail = raw
            elif raw is not None:
                detail = str(raw)
    except (json.JSONDecodeError, ValueError):
        text = (exc.response.text or "").strip()
        if text and len(text) <= 500:
            detail = text
    if detail:
        return f"Request failed ({status}): {detail}"
    return f"Request failed ({status}). Check backend logs and configuration."
