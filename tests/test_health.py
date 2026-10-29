"""Tests for root and system health-check endpoints."""

from fastapi.testclient import TestClient


def test_root_returns_app_name_and_status(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "Engineering Knowledge Intelligence"


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "Engineering Knowledge Intelligence"
    assert body["environment"] == "test"
    assert body["ephemeral_sessions_enabled"] is False


def test_openapi_docs_available(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
