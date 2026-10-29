"""FastAPI dependencies for indexing and search services."""

from __future__ import annotations

from app.core.config import Settings
from app.core.session_deps import get_index_service, get_search_service
from app.generation.providers import LLMProvider
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.reranker import Reranker
from app.retrieval.indexer import IndexService
from app.retrieval.search import SearchService


def bind_services(
    app: object,
    settings: Settings,
    embedding_model: EmbeddingModel | None = None,
    reranker: Reranker | None = None,
    llm_provider: LLMProvider | None = None,
) -> None:
    """Initialize shared services for tests or explicit app setup."""
    from app.generation.deps import bind_generation_services, build_reranker

    if embedding_model is not None:
        app.state.embedding_model = embedding_model  # type: ignore[attr-defined]
    app.state.reranker = build_reranker(settings, reranker)  # type: ignore[attr-defined]
    app.state.llm_provider = llm_provider  # type: ignore[attr-defined]

    if settings.ephemeral_sessions_enabled:
        return

    index_service = IndexService.from_settings(settings, embedding_model=embedding_model)
    search_service = SearchService(index_service, settings)
    app.state.index_service = index_service  # type: ignore[attr-defined]
    app.state.search_service = search_service  # type: ignore[attr-defined]
    bind_generation_services(
        app,
        settings,
        reranker=reranker,
        llm_provider=llm_provider,
    )


__all__ = [
    "bind_services",
    "get_index_service",
    "get_search_service",
]
