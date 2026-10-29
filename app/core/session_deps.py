"""FastAPI dependencies for ephemeral session isolation."""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.core.config import Settings, get_settings
from app.core.session_manager import SessionManager
from app.generation.answer import AnswerService
from app.generation.chat_service import ChatService
from app.generation.models import ChatMessage
from app.retrieval.indexer import IndexService
from app.retrieval.search import SearchService


def _settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def _manager(request: Request) -> SessionManager:
    manager = getattr(request.app.state, "session_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Ephemeral session mode is not initialized on this process.",
        )
    return manager


def ephemeral_sessions_enabled(request: Request) -> bool:
    return _settings(request).ephemeral_sessions_enabled


def resolve_session_id(request: Request) -> str | None:
    settings = _settings(request)
    if not settings.ephemeral_sessions_enabled:
        return None
    raw = request.headers.get("X-Session-ID", "").strip()
    if not raw:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Session-ID header. Create a session via POST /session first.",
        )
    if not SessionManager.is_valid_session_id(raw):
        raise HTTPException(status_code=400, detail="Invalid X-Session-ID format.")
    try:
        _manager(request).touch(raw)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Create a new session via POST /session.",
        ) from exc
    return raw


def get_index_service(request: Request) -> IndexService:
    session_id = resolve_session_id(request)
    if session_id is not None:
        return _manager(request).get_index_service(session_id)
    if hasattr(request.app.state, "index_service"):
        return request.app.state.index_service  # type: ignore[no-any-return]
    settings = _settings(request)
    from app.retrieval.indexer import IndexService as IndexServiceCls

    service = IndexServiceCls.from_settings(settings)
    request.app.state.index_service = service
    request.app.state.search_service = SearchService(service, settings)
    return service


def get_search_service(request: Request) -> SearchService:
    session_id = resolve_session_id(request)
    if session_id is not None:
        return _manager(request).get_search_service(session_id)
    if hasattr(request.app.state, "search_service"):
        return request.app.state.search_service  # type: ignore[no-any-return]
    index_service = get_index_service(request)
    settings = _settings(request)
    service = SearchService(index_service, settings)
    request.app.state.search_service = service
    return service


def get_chat_service(request: Request) -> ChatService:
    session_id = resolve_session_id(request)
    if session_id is not None:
        return _manager(request).get_chat_service(session_id)
    if hasattr(request.app.state, "chat_service"):
        return request.app.state.chat_service  # type: ignore[no-any-return]

    settings = _settings(request)
    search_service = get_search_service(request)
    from app.generation.deps import build_reranker
    from app.generation.providers import build_llm_provider

    reranker = getattr(request.app.state, "reranker", None) or build_reranker(settings)
    llm_provider = getattr(request.app.state, "llm_provider", None) or build_llm_provider(
        settings
    )
    service = ChatService(
        search_service,
        settings,
        reranker=reranker,
        llm_provider=llm_provider,
    )
    request.app.state.chat_service = service
    return service


def get_answer_service(request: Request) -> AnswerService:
    session_id = resolve_session_id(request)
    if session_id is not None:
        return _manager(request).get_answer_service(session_id)
    if hasattr(request.app.state, "answer_service"):
        return request.app.state.answer_service  # type: ignore[no-any-return]

    settings = _settings(request)
    search_service = get_search_service(request)
    from app.generation.deps import build_reranker
    from app.generation.providers import build_llm_provider

    reranker = getattr(request.app.state, "reranker", None) or build_reranker(settings)
    llm_provider = getattr(request.app.state, "llm_provider", None) or build_llm_provider(
        settings
    )
    service = AnswerService(
        search_service,
        settings,
        reranker=reranker,
        llm_provider=llm_provider,
    )
    request.app.state.answer_service = service
    return service


def resolve_chat_history(request: Request, client_history: list[ChatMessage]) -> list[ChatMessage]:
    session_id = resolve_session_id(request)
    if session_id is None:
        return client_history
    return _manager(request).get_chat_history(session_id)


def record_chat_turn(request: Request, user_message: str, assistant_message: str) -> None:
    if not ephemeral_sessions_enabled(request):
        return
    session_id = request.headers.get("X-Session-ID", "").strip()
    if not session_id or not SessionManager.is_valid_session_id(session_id):
        return
    _manager(request).append_chat_turn(
        session_id,
        user_message=user_message,
        assistant_message=assistant_message,
    )
