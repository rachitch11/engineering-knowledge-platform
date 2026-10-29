"""Ollama connectivity and model availability checks."""

from __future__ import annotations

from typing import Any

import httpx


def normalize_model_name(model: str) -> str:
    return model.split(":", 1)[0]


def model_is_available(requested: str, installed_names: list[str]) -> bool:
    requested_base = normalize_model_name(requested).lower()
    for name in installed_names:
        installed = name.lower()
        if installed == requested_base or installed.startswith(f"{requested_base}:"):
            return True
        if normalize_model_name(installed) == requested_base:
            return True
    return False


def check_ollama_status(
    base_url: str,
    model: str,
    *,
    timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    """Return reachability and whether the configured model appears installed."""
    url = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(f"{url}/api/tags")
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        return {
            "reachable": False,
            "model_available": False,
            "configured_model": model,
            "installed_models": [],
            "message": f"Ollama is not reachable at {url}: {exc}",
        }

    installed = [str(item.get("name", "")) for item in payload.get("models", [])]
    available = model_is_available(model, installed)
    message = "Ollama is reachable."
    if not available:
        message = (
            f"Ollama is reachable but model '{model}' was not found. "
            f"Installed: {', '.join(installed) or '(none)'}. "
            f"Run: ollama pull {normalize_model_name(model)}"
        )
    return {
        "reachable": True,
        "model_available": available,
        "configured_model": model,
        "installed_models": installed,
        "message": message,
    }


def assert_ollama_ready(base_url: str, model: str) -> None:
    status = check_ollama_status(base_url, model)
    if not status["reachable"]:
        raise ConnectionError(str(status["message"]))
    if not status["model_available"]:
        raise ConnectionError(str(status["message"]))
