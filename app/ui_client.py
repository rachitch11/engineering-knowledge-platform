"""HTTP client for the Streamlit UI."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.ui_errors import format_backend_error


class BackendClient:
    """Thin wrapper around the FastAPI backend."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 180.0,
        *,
        session_id: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id = session_id

    def _headers(self) -> dict[str, str]:
        if self.session_id:
            return {"X-Session-ID": self.session_id}
        return {}

    def create_session(self) -> dict[str, Any]:
        """Register a new ephemeral backend session (POST /session)."""
        return self._request("POST", "/session")

    def reset_session(self) -> dict[str, Any]:
        """Clear this session's indexed documents and server chat history."""
        return self._request("POST", "/session/reset")

    def delete_session(self) -> None:
        """Remove this ephemeral session and its temporary index."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.delete(
                f"{self.base_url}/session",
                headers=self._headers(),
            )
            response.raise_for_status()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def llm_health(self) -> dict[str, Any]:
        return self._request("GET", "/health/llm")

    def ollama_health(self) -> dict[str, Any]:
        return self._request("GET", "/health/ollama")

    def list_documents(self) -> dict[str, Any]:
        return self._request("GET", "/documents")

    def upload_and_index(self, filename: str, content: bytes) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/documents/index",
                files={"file": (filename, content)},
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    def delete_document(self, document_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/documents/{document_id}")

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/search", json=payload)

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/chat", json=payload)

    def chat_stream(self, payload: dict[str, Any]):
        """Stream SSE events from POST /chat/stream."""
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/stream",
                json=payload,
                headers=self._headers(),
            ) as response:
                if response.status_code >= 400:
                    raw = response.read()
                    detail = raw.decode("utf-8", errors="replace")
                    raise httpx.HTTPStatusError(
                        detail,
                        request=response.request,
                        response=httpx.Response(
                            status_code=response.status_code,
                            content=raw,
                            request=response.request,
                        ),
                    )
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event.get("type") == "error":
                        detail = str(event.get("detail") or "Stream failed.")
                        raise httpx.HTTPStatusError(
                            detail,
                            request=response.request,
                            response=httpx.Response(
                                status_code=502,
                                content=json.dumps({"detail": detail}).encode(),
                                request=response.request,
                            ),
                        )
                    yield event

    def answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/answer", json=payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(
                method,
                f"{self.base_url}{path}",
                json=json,
                headers=self._headers(),
            )
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict):
                return body
            raise httpx.HTTPError(f"Unexpected response type from {path}")
