"""FastAPI dependencies for answer generation."""

from __future__ import annotations

from fastapi import Request

from app.core.config import Settings, get_settings
from app.core.session_deps import get_answer_service, get_chat_service
from app.generation.answer import AnswerService
from app.generation.chat_service import ChatService
from app.generation.providers import LLMProvider, build_llm_provider
from app.retrieval.reranker import LazyCrossEncoderReranker, MockReranker, Reranker
from app.retrieval.search import SearchService


def build_reranker(settings: Settings, reranker: Reranker | None = None) -> Reranker | None:
    if reranker is not None:
        return reranker
    if not settings.rerank_enabled:
        return None
    if settings.environment == "test":
        return MockReranker()
    return LazyCrossEncoderReranker(settings.rerank_model)


def bind_generation_services(
    app: object,
    settings: Settings,
    *,
    reranker: Reranker | None = None,
    llm_provider: LLMProvider | None = None,
) -> None:
    search_service = app.state.search_service  # type: ignore[attr-defined]
    resolved_reranker = build_reranker(settings, reranker=reranker)
    app.state.reranker = resolved_reranker  # type: ignore[attr-defined]
    app.state.llm_provider = llm_provider  # type: ignore[attr-defined]
    if llm_provider is not None:
        app.state.answer_service = AnswerService(  # type: ignore[attr-defined]
            search_service,
            settings,
            reranker=resolved_reranker,
            llm_provider=llm_provider,
        )
        app.state.chat_service = ChatService(  # type: ignore[attr-defined]
            search_service,
            settings,
            reranker=resolved_reranker,
            llm_provider=llm_provider,
        )


__all__ = [
    "AnswerService",
    "ChatService",
    "build_reranker",
    "bind_generation_services",
    "get_answer_service",
    "get_chat_service",
]
