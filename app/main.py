"""FastAPI application entry point."""

import asyncio
import contextlib
import logging
import traceback
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.answer import answer_router
from app.api.chat import chat_router
from app.api.documents import documents_router
from app.api.routes import router as api_router
from app.api.search import search_router
from app.api.session import session_router
from app.core.config import Settings, get_settings
from app.core.env_validation import SettingsValidationError, validate_settings_at_startup
from app.ingestion.exceptions import IngestionError
from app.generation.exceptions import GenerationError, LLMProviderError
from app.retrieval.deps import bind_services
from app.retrieval.exceptions import RetrievalError

logger = logging.getLogger(__name__)


def _request_correlation_id(request: Request) -> str | None:
    for header_name in ("X-Request-ID", "X-Correlation-ID", "X-Trace-ID"):
        value = request.headers.get(header_name)
        if value:
            return value.strip()
    return getattr(request.state, "correlation_id", None)


def _log_request_exception(request: Request, exc: Exception, *, label: str) -> None:
    correlation_id = _request_correlation_id(request)
    logger.error(
        "%s method=%s path=%s exc_type=%s correlation_id=%s\n%s",
        label,
        request.method,
        request.url.path,
        type(exc).__name__,
        correlation_id or "-",
        traceback.format_exc(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load configuration at startup; run session cleanup; wipe ephemeral data on shutdown."""
    if not hasattr(app.state, "settings"):
        get_settings.cache_clear()
        app.state.settings = get_settings()
    settings: Settings = app.state.settings
    from app.core.memory_diagnostics import configure_runtime_memory, log_memory_event

    configure_runtime_memory()
    try:
        validate_settings_at_startup(settings)
    except SettingsValidationError as exc:
        logger.error("settings_validation_failed: %s", exc)

    log_memory_event(
        "startup",
        environment=settings.environment,
        ephemeral_sessions=settings.ephemeral_sessions_enabled,
        memory_constrained=settings.memory_constrained,
        rerank_enabled=settings.rerank_enabled,
        max_index_chunks=settings.max_index_chunks_per_document,
    )
    if settings.memory_constrained and settings.rerank_enabled:
        logger.warning(
            "memory_constrained_with_rerank: cross-encoder plus embedding may exceed "
            "512MB; set RERANK_ENABLED=false on small hosts if uploads are OOM-killed."
        )
    cleanup_task: asyncio.Task[None] | None = None
    if settings.ephemeral_sessions_enabled:
        from app.core.session_manager import init_session_manager

        init_session_manager(
            app,
            settings,
            embedding_model=getattr(app.state, "embedding_model", None),
            reranker=getattr(app.state, "reranker", None),
            llm_provider=getattr(app.state, "llm_provider", None),
        )

        async def _cleanup_loop() -> None:
            while True:
                await asyncio.sleep(settings.session_cleanup_interval_seconds)
                removed = app.state.session_manager.cleanup_expired()
                if removed:
                    logger.info("session_cleanup removed=%s", removed)

        cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
        if hasattr(app.state, "session_manager"):
            app.state.session_manager.shutdown_all()
        get_settings.cache_clear()


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(IngestionError)
    async def ingestion_error_handler(
        _request: Request,
        exc: IngestionError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(
        _request: Request,
        exc: RetrievalError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @app.exception_handler(GenerationError)
    async def generation_error_handler(
        _request: Request,
        exc: GenerationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @app.exception_handler(LLMProviderError)
    async def llm_provider_error_handler(
        request: Request,
        exc: LLMProviderError,
    ) -> JSONResponse:
        _log_request_exception(request, exc, label="llm_provider_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        _log_request_exception(request, exc, label="unhandled_exception")
        settings = getattr(request.app.state, "settings", None) or get_settings()
        if settings.environment == "development":
            raise exc
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


def create_app(
    settings: Settings | None = None,
    embedding_model: object | None = None,
    reranker: object | None = None,
    llm_provider: object | None = None,
) -> FastAPI:
    """Build the FastAPI application (used by uvicorn and tests)."""
    resolved = settings or get_settings()
    app = FastAPI(
        title=resolved.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    register_exception_handlers(app)
    cors_origins = [
        origin.strip()
        for origin in resolved.cors_allow_origins.split(",")
        if origin.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
    app.include_router(api_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(answer_router)
    app.include_router(chat_router)
    app.include_router(session_router)
    bind_services(
        app,
        resolved,
        embedding_model=embedding_model,  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
        llm_provider=llm_provider,  # type: ignore[arg-type]
    )
    if settings is not None:
        from app.api.routes import get_app_settings

        app.dependency_overrides[get_app_settings] = lambda: settings
    return app


app = create_app()
