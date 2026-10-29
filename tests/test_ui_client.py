"""Tests for Streamlit backend client helpers."""

from unittest.mock import MagicMock, patch

from app.ui_client import BackendClient


def test_backend_client_exposes_session_helpers() -> None:
    client = BackendClient("http://testserver")
    assert hasattr(client, "create_session")
    assert hasattr(client, "reset_session")


@patch("app.ui_client.httpx.Client")
def test_backend_client_create_session(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"session_id": "550e8400-e29b-41d4-a716-446655440000"}
    mock_client.request.return_value = mock_response

    client = BackendClient("http://testserver")
    body = client.create_session()
    assert body["session_id"].startswith("550e8400")
    mock_client.request.assert_called_with(
        "POST",
        "http://testserver/session",
        json=None,
        headers={},
    )


@patch("app.ui_client.httpx.Client")
def test_backend_client_health(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "ok"}
    mock_client.request.return_value = mock_response

    client = BackendClient("http://testserver")
    health = client.health()
    assert health["status"] == "ok"
    mock_client.request.assert_called_with(
        "GET",
        "http://testserver/health",
        json=None,
        headers={},
    )


@patch("app.ui_client.httpx.Client")
def test_backend_client_answer(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"answer": "hi", "citations": []}
    mock_client.request.return_value = mock_response

    client = BackendClient("http://testserver")
    answer = client.answer({"question": "hello"})
    assert answer["answer"] == "hi"
    mock_client.request.assert_called_with(
        "POST",
        "http://testserver/answer",
        json={"question": "hello"},
        headers={},
    )
