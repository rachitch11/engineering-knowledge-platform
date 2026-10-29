"""Session lifecycle API for ephemeral multi-visitor mode."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.session_deps import ephemeral_sessions_enabled, resolve_session_id
from app.core.session_manager import SessionManager

from app.api.routes import get_app_settings

session_router = APIRouter(prefix="/session", tags=["session"])


class CreateSessionResponse(BaseModel):
    session_id: str
    ttl_seconds: int = Field(description="Idle time before the session is deleted.")


class SessionStatusResponse(BaseModel):
    session_id: str
    active: bool = True


@session_router.post("", response_model=CreateSessionResponse)
def create_session(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> CreateSessionResponse:
    if not settings.ephemeral_sessions_enabled:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=404,
            detail="Ephemeral sessions are disabled on this deployment.",
        )
    manager: SessionManager = request.app.state.session_manager
    session_id = manager.create_session()
    return CreateSessionResponse(
        session_id=session_id,
        ttl_seconds=settings.session_ttl_seconds,
    )


@session_router.get("", response_model=SessionStatusResponse)
def session_status(
    request: Request,
    session_id: str = Depends(resolve_session_id),
) -> SessionStatusResponse:
    return SessionStatusResponse(session_id=session_id)


@session_router.post("/reset", response_model=SessionStatusResponse)
def reset_session(
    request: Request,
    session_id: str = Depends(resolve_session_id),
) -> SessionStatusResponse:
    request.app.state.session_manager.reset_session(session_id)
    return SessionStatusResponse(session_id=session_id)


@session_router.delete("", status_code=204)
def delete_session(
    request: Request,
    session_id: str = Depends(resolve_session_id),
) -> None:
    request.app.state.session_manager.destroy_session(session_id)
