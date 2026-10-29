"""HTTP route handlers for the REST API."""

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.ollama import check_ollama_status
from app.retrieval.deps import get_index_service
from app.retrieval.indexer import IndexService

router = APIRouter()


def get_app_settings() -> Settings:
    return get_settings()


class RootResponse(BaseModel):
    """Payload returned by the root endpoint."""

    app_name: str
    status: Literal["ok"]


class HealthResponse(BaseModel):
    """Payload returned by the health-check endpoint."""

    status: Literal["ok"]
    app_name: str
    environment: str
    ephemeral_sessions_enabled: bool


class DeploymentHealthResponse(BaseModel):
    """Extended health payload for deployment verification."""

    status: Literal["ok", "degraded"]
    app_name: str
    environment: str
    single_user_mode: bool
    indexed_documents: int
    indexed_chunks: int
    bm25_records: int
    chroma_path: str
    bm25_index_path: str
    llm_provider: str
    storage_ready: bool
    notes: list[str]


class OllamaHealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    llm_provider: str
    reachable: bool
    model_available: bool
    configured_model: str
    installed_models: list[str]
    message: str


@router.get("/", response_model=RootResponse, tags=["system"])
def read_root(settings: Settings = Depends(get_app_settings)) -> RootResponse:
    """Return a minimal identity response for the running API."""
    return RootResponse(app_name=settings.app_name, status="ok")


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Report whether the API process is running and configuration is loaded."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
        ephemeral_sessions_enabled=settings.ephemeral_sessions_enabled,
    )


@router.get("/health/deployment", response_model=DeploymentHealthResponse, tags=["system"])
def deployment_health_check(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> DeploymentHealthResponse:
    """Verify storage paths and index state for deployment smoke tests."""
    if settings.ephemeral_sessions_enabled:
        manager = getattr(request.app.state, "session_manager", None)
        active = manager.active_session_count() if manager is not None else 0
        store = Path(settings.session_store_root)
        store.mkdir(parents=True, exist_ok=True)
        return DeploymentHealthResponse(
            status="ok",
            app_name=settings.app_name,
            environment=settings.environment,
            single_user_mode=False,
            indexed_documents=0,
            indexed_chunks=0,
            bm25_records=0,
            chroma_path=str(store),
            bm25_index_path="(per-session under session store)",
            llm_provider=settings.llm_provider,
            storage_ready=store.exists(),
            notes=[
                "Ephemeral session mode: each visitor uses POST /session and header X-Session-ID.",
                f"Active sessions in this process: {active}.",
                "Session indexes are temporary and are not restored after restart.",
                "In-memory session registry is not shared across uvicorn workers; use one worker "
                "or accept per-worker session stickiness.",
            ],
        )

    index_service: IndexService = get_index_service(request)
    notes = [
        "Shared persistent index mode (EPHEMERAL_SESSIONS_ENABLED=false).",
        "Do not expose to untrusted multi-tenant traffic without session isolation.",
    ]
    chroma_dir = Path(settings.chroma_path)
    bm25_file = Path(settings.bm25_index_path)
    storage_ready = chroma_dir.exists() and chroma_dir.is_dir() and bm25_file.parent.exists()
    indexed_documents = len(index_service.list_documents())
    indexed_chunks = index_service.chroma.count()
    bm25_records = len(index_service.bm25.records)
    status: Literal["ok", "degraded"] = "ok"
    if not storage_ready:
        status = "degraded"
        notes.append("Storage path is missing or not writable.")
    if indexed_chunks != bm25_records:
        status = "degraded"
        notes.append("Chroma and BM25 record counts differ; rebuild may be required.")

    return DeploymentHealthResponse(
        status=status,
        app_name=settings.app_name,
        environment=settings.environment,
        single_user_mode=settings.single_user_mode,
        indexed_documents=indexed_documents,
        indexed_chunks=indexed_chunks,
        bm25_records=bm25_records,
        chroma_path=settings.chroma_path,
        bm25_index_path=settings.bm25_index_path,
        llm_provider=settings.llm_provider,
        storage_ready=storage_ready,
        notes=notes,
    )


class LLMHealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    llm_provider: str
    configured_model: str
    api_key_configured: bool
    message: str


@router.get("/health/llm", response_model=LLMHealthResponse, tags=["system"])
def llm_health_check(settings: Settings = Depends(get_app_settings)) -> LLMHealthResponse:
    """Report configured LLM provider and whether required secrets are present."""
    provider = settings.llm_provider
    if provider == "gemini":
        key_ok = bool(settings.gemini_api_key and settings.gemini_api_key.strip())
        if not key_ok:
            return LLMHealthResponse(
                status="unavailable",
                llm_provider=provider,
                configured_model=settings.gemini_model,
                api_key_configured=False,
                message=(
                    "GEMINI_API_KEY is not set. Add it to .env (see .env.example). "
                    "Gemini requires internet access and may incur usage charges."
                ),
            )
        return LLMHealthResponse(
            status="ok",
            llm_provider=provider,
            configured_model=settings.gemini_model,
            api_key_configured=True,
            message="Gemini API key is configured (connectivity not probed on this endpoint).",
        )
    if provider == "mock":
        return LLMHealthResponse(
            status="ok",
            llm_provider=provider,
            configured_model="mock-llm",
            api_key_configured=True,
            message="Using mock LLM provider for tests.",
        )
    if provider == "openai_compatible":
        key_ok = bool(
            settings.openai_compatible_api_key
            and settings.openai_compatible_api_key.strip()
        )
        return LLMHealthResponse(
            status="ok" if key_ok else "unavailable",
            llm_provider=provider,
            configured_model=settings.openai_compatible_model,
            api_key_configured=key_ok,
            message=(
                "OpenAI-compatible API key configured."
                if key_ok
                else "OPENAI_COMPATIBLE_API_KEY is not set."
            ),
        )
    if provider == "ollama":
        status = check_ollama_status(settings.ollama_base_url, settings.ollama_model)
        if not status["reachable"]:
            return LLMHealthResponse(
                status="unavailable",
                llm_provider=provider,
                configured_model=settings.ollama_model,
                api_key_configured=True,
                message=str(status["message"]),
            )
        if not status["model_available"]:
            return LLMHealthResponse(
                status="degraded",
                llm_provider=provider,
                configured_model=settings.ollama_model,
                api_key_configured=True,
                message=str(status["message"]),
            )
        return LLMHealthResponse(
            status="ok",
            llm_provider=provider,
            configured_model=settings.ollama_model,
            api_key_configured=True,
            message=str(status["message"]),
        )
    return LLMHealthResponse(
        status="degraded",
        llm_provider=provider,
        configured_model="unknown",
        api_key_configured=False,
        message=f"Unknown LLM provider '{provider}'.",
    )


@router.get("/health/ollama", response_model=OllamaHealthResponse, tags=["system"])
def ollama_health_check(settings: Settings = Depends(get_app_settings)) -> OllamaHealthResponse:
    """Check whether Ollama is reachable and the configured model is installed."""
    if settings.llm_provider != "ollama":
        return OllamaHealthResponse(
            status="ok",
            llm_provider=settings.llm_provider,
            reachable=False,
            model_available=False,
            configured_model=settings.ollama_model,
            installed_models=[],
            message=f"LLM provider is '{settings.llm_provider}', not Ollama.",
        )

    status = check_ollama_status(settings.ollama_base_url, settings.ollama_model)
    if not status["reachable"]:
        return OllamaHealthResponse(
            status="unavailable",
            llm_provider=settings.llm_provider,
            reachable=False,
            model_available=False,
            configured_model=settings.ollama_model,
            installed_models=[],
            message=str(status["message"]),
        )
    if not status["model_available"]:
        return OllamaHealthResponse(
            status="degraded",
            llm_provider=settings.llm_provider,
            reachable=True,
            model_available=False,
            configured_model=settings.ollama_model,
            installed_models=list(status["installed_models"]),
            message=str(status["message"]),
        )
    return OllamaHealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        reachable=True,
        model_available=True,
        configured_model=settings.ollama_model,
        installed_models=list(status["installed_models"]),
        message=str(status["message"]),
    )
