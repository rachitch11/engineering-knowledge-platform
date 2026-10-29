"""Conversational chat API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.core.session_deps import (
    record_chat_turn,
    resolve_chat_history,
)
from app.generation.models import ChatRequest, ChatResponse
from app.generation.chat_service import ChatService
from app.core.session_deps import get_chat_service

chat_router = APIRouter(tags=["chat"])


@chat_router.post("/chat", response_model=ChatResponse)
def chat(
    request: Request,
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Conversation-aware grounded or general chat."""
    _ = payload.stream
    history = resolve_chat_history(request, payload.history)
    response = chat_service.chat(
        payload.message,
        history,
        strategy=payload.strategy,
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
        rerank=payload.rerank,
        document_id=payload.document_id,
        file_type=payload.file_type,
    )
    record_chat_turn(request, payload.message, response.answer)
    return response


@chat_router.post("/chat/stream")
def chat_stream(
    request: Request,
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Server-sent events stream of answer tokens and final metadata."""
    history = resolve_chat_history(request, payload.history)
    chat_response = chat_service.chat(
        payload.message,
        history,
        strategy=payload.strategy,
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
        rerank=payload.rerank,
        document_id=payload.document_id,
        file_type=payload.file_type,
    )
    record_chat_turn(request, payload.message, chat_response.answer)
    return StreamingResponse(
        chat_service.iter_chat_sse(chat_response),
        media_type="text/event-stream",
    )
