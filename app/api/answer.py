"""Grounded answer API routes."""

from fastapi import APIRouter, Depends

from app.api.routes import get_app_settings
from app.core.config import Settings
from app.generation.answer import AnswerService
from app.generation.deps import get_answer_service
from app.generation.models import AnswerRequest, AnswerResponse

answer_router = APIRouter(tags=["answer"])


@answer_router.post("/answer", response_model=AnswerResponse)
def generate_answer(
    payload: AnswerRequest,
    answer_service: AnswerService = Depends(get_answer_service),
    settings: Settings = Depends(get_app_settings),
) -> AnswerResponse:
    """Retrieve evidence, optionally rerank, and generate a grounded answer."""
    _ = settings
    return answer_service.answer(
        payload.question,
        strategy=payload.strategy,
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
        rerank=payload.rerank,
        document_id=payload.document_id,
        file_type=payload.file_type,
    )
