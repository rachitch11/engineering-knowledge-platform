"""Streamlit frontend for the Engineering Knowledge Intelligence Platform."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Streamlit executes this file with `app/` on sys.path, which breaks `import app.*`.
# Ensure the repository root is importable when launched via `streamlit run app/ui.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx
import streamlit as st

from app.ui_client import BackendClient
from app.ui_errors import format_backend_error
from app.ui_streaming import (
    assemble_answer_from_stream_events,
    should_append_assistant_message,
    should_append_user_message,
)


def get_backend_url() -> str:
    return os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def render_error(message: str) -> None:
    st.error(message)


def get_backend_client(base_url: str) -> BackendClient:
    cached = st.session_state.get("backend_client")
    if cached is not None and not hasattr(cached, "create_session"):
        st.session_state.pop("backend_client", None)
        st.session_state.pop("backend_session_id", None)

    if "backend_client" not in st.session_state:
        probe = BackendClient(base_url)
        session_id: str | None = None
        try:
            created = probe.create_session()
            session_id = str(created.get("session_id", "") or "") or None
        except AttributeError:
            # Stale import/cache: create session without relying on helper method.
            with httpx.Client(timeout=probe.timeout) as client:
                response = client.post(f"{base_url.rstrip('/')}/session")
                if response.status_code == 200:
                    session_id = str(response.json().get("session_id", "") or "") or None
                elif response.status_code != 404:
                    response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
        st.session_state.backend_session_id = session_id
        st.session_state.backend_client = BackendClient(base_url, session_id=session_id)
    return st.session_state.backend_client  # type: ignore[no-any-return]


def rotate_ephemeral_session(base_url: str) -> BackendClient:
    """Discard the current visitor session and register a new isolated one."""
    previous = st.session_state.get("backend_client")
    if previous is not None and getattr(previous, "session_id", None):
        try:
            previous.delete_session()
        except httpx.HTTPError:
            pass
    probe = BackendClient(base_url)
    created = probe.create_session()
    session_id = str(created.get("session_id", "") or "") or None
    st.session_state.backend_session_id = session_id
    st.session_state.backend_client = BackendClient(base_url, session_id=session_id)
    st.session_state.messages = []
    st.session_state.pop("documents", None)
    return st.session_state.backend_client  # type: ignore[no-any-return]


def main() -> None:
    st.set_page_config(
        page_title="Engineering Knowledge Intelligence",
        layout="wide",
    )
    st.title("Engineering Knowledge Intelligence Platform")
    backend_url = get_backend_url()
    st.caption(f"Backend: {backend_url}")
    client = get_backend_client(backend_url)
    ephemeral_on_backend = False
    with st.sidebar:
        st.header("Connection status")
        try:
            health = client.health()
            st.success(f"Backend: {health.get('status', 'unknown')}")
            ephemeral_on_backend = bool(health.get("ephemeral_sessions_enabled"))
        except httpx.HTTPError as exc:
            st.error(f"Backend unreachable: {exc}")
        try:
            llm = client.llm_health()
            if llm.get("status") == "ok":
                st.success(
                    f"LLM: {llm.get('llm_provider')} / {llm.get('configured_model')}"
                )
            elif llm.get("status") == "degraded":
                st.warning(llm.get("message", "LLM configuration degraded"))
            else:
                st.warning(llm.get("message", "LLM unavailable"))
        except httpx.HTTPError as exc:
            st.warning(f"Could not check LLM via backend: {exc}")

        if ephemeral_on_backend and st.session_state.get("backend_session_id"):
            st.caption(
                "Private session: your uploads and answers are isolated from other visitors."
            )
        elif ephemeral_on_backend and not st.session_state.get("backend_session_id"):
            st.warning(
                "Ephemeral mode is enabled on the API but this tab has no session id. "
                "Use **New private session** or restart the API with matching settings."
            )
        elif not ephemeral_on_backend:
            st.warning(
                "Shared index mode (legacy): all visitors share one document store. "
                "Enable `EPHEMERAL_SESSIONS_ENABLED=true` on the API and restart for "
                "per-visitor isolation (the default for this project)."
            )

        if st.button("New chat", use_container_width=True):
            try:
                if client.session_id:
                    client.reset_session()
            except httpx.HTTPError as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    render_error(format_backend_error(exc))
            st.session_state.messages = []
            st.session_state.pop("documents", None)
            st.rerun()

        if ephemeral_on_backend and st.button(
            "New private session",
            use_container_width=True,
            help="Start a brand-new isolated index (other visitors are unaffected).",
        ):
            try:
                rotate_ephemeral_session(backend_url)
            except httpx.HTTPError as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    render_error(format_backend_error(exc))
                else:
                    render_error(f"Could not create session: {exc}")
            else:
                st.rerun()

        st.header("Retrieval settings")
        strategy = st.selectbox("Strategy", ["hybrid", "dense", "bm25"], index=0)
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
        rerank = st.checkbox("Enable reranking", value=True)
        st.header("Document management")
        uploaded = st.file_uploader(
            "Upload and index",
            type=["pdf", "md", "markdown", "html", "htm"],
        )
        if uploaded is not None and st.button("Index uploaded file"):
            try:
                result = client.upload_and_index(uploaded.name, uploaded.getvalue())
                st.success(
                    f"Indexed {result.get('filename')} "
                    f"({result.get('chunk_count')} chunks)"
                )
            except httpx.HTTPError as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    render_error(format_backend_error(exc))
                else:
                    render_error(f"Indexing failed: {exc}")

        if st.button("Refresh document list"):
            st.session_state.pop("documents", None)

        try:
            documents_payload = client.list_documents()
            st.session_state["documents"] = documents_payload.get("documents", [])
        except httpx.HTTPError as exc:
            render_error(f"Could not load documents: {exc}")
            st.session_state["documents"] = []

        for doc in st.session_state.get("documents", []):
            cols = st.columns([3, 1])
            cols[0].write(f"{doc.get('filename')} ({doc.get('chunk_count')} chunks)")
            if cols[1].button("Delete", key=f"delete-{doc.get('document_id')}"):
                try:
                    client.delete_document(str(doc.get("document_id")))
                    st.success("Document deleted.")
                except httpx.HTTPError as exc:
                    if isinstance(exc, httpx.HTTPStatusError):
                        render_error(format_backend_error(exc))
                    else:
                        render_error(f"Delete failed: {exc}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for citation in message.get("citations", []):
                with st.expander(
                    f"{citation.get('filename')} "
                    f"(page {citation.get('page_number', 'n/a')})"
                ):
                    st.write(citation.get("excerpt", ""))

    prompt = st.chat_input("Message the assistant…")
    if prompt:
        if should_append_user_message(st.session_state.messages, prompt):
            st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
            if message["role"] in ("user", "assistant")
        ]
        if history and history[-1]["role"] == "user" and history[-1]["content"] == prompt:
            history = history[:-1]

        with st.chat_message("assistant"):
            try:
                stream_payload = {
                    "message": prompt,
                    "history": history,
                    "strategy": strategy,
                    "top_k": int(top_k),
                    "rerank": rerank,
                }
                captured_events: list[dict] = []

                def stream_tokens():
                    for event in client.chat_stream(stream_payload):
                        captured_events.append(event)
                        if event.get("type") == "token":
                            piece = str(event.get("text", "") or "")
                            if piece:
                                yield piece

                st.write_stream(stream_tokens)

                answer, final_payload = assemble_answer_from_stream_events(
                    captured_events
                )
                citations = final_payload.get("citations", []) if final_payload else []
                grounded = final_payload.get("grounded") if final_payload else None
                intent = final_payload.get("intent") if final_payload else None
                if intent:
                    mode = final_payload.get("response_mode", "general") if final_payload else "general"
                    st.caption(
                        f"Mode: {mode} · intent: {intent}"
                        + (" · grounded" if grounded else "")
                    )
                for citation in citations:
                    with st.expander(
                        f"Source: {citation.get('filename')} "
                        f"(page {citation.get('page_number', 'n/a')})"
                    ):
                        st.write(citation.get("excerpt", ""))
                if should_append_assistant_message(st.session_state.messages, answer):
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                        }
                    )
            except httpx.HTTPStatusError as exc:
                render_error(format_backend_error(exc))
            except httpx.HTTPError as exc:
                render_error(f"Could not reach backend API: {exc}")


if __name__ == "__main__":
    main()
