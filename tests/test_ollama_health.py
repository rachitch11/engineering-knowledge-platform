"""Tests for Ollama health helpers and API route."""

import httpx
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.core.ollama import check_ollama_status, model_is_available


def test_model_is_available_matches_tags() -> None:
    assert model_is_available("llama3.2", ["llama3.2:latest"])
    assert not model_is_available("llama3.2", ["mistral:latest"])


@patch("app.core.ollama.httpx.Client")
def test_check_ollama_status_unreachable(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client
    mock_client.get.side_effect = httpx.ConnectError("connection refused")
    status = check_ollama_status("http://127.0.0.1:11434", "llama3.2")
    assert status["reachable"] is False


def test_ollama_health_route_with_mock_provider(client: TestClient) -> None:
    response = client.get("/health/ollama")
    assert response.status_code == 200
    body = response.json()
    assert "reachable" in body
    assert "configured_model" in body
