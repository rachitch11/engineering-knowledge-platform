"""Request and response models for grounded answer generation."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.retrieval.models import SearchHit, SearchStrategy


class EvidenceChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    chunk_index: int
    page_number: int | None = None
    text: str
    score: float


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None = None
    excerpt: str


class GeneratedAnswer(BaseModel):
    answer: str
    citation_ids: list[str] = Field(default_factory=list)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    strategy: SearchStrategy = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=100)
    rerank: bool | None = None
    document_id: str | None = None
    file_type: Literal["pdf", "markdown", "html"] | None = None

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Question must not be empty.")
        return normalized


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    strategy: SearchStrategy = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=100)
    rerank: bool | None = None
    stream: bool = False
    document_id: str | None = None
    file_type: Literal["pdf", "markdown", "html"] | None = None

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Message must not be empty.")
        return normalized


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sufficient_evidence: bool
    citations: list[Citation]
    evidence: list[EvidenceChunk]
    retrieval_strategy: SearchStrategy
    candidate_count: int
    rerank_applied: bool


class ChatResponse(AnswerResponse):
    intent: str
    response_mode: Literal["general", "document", "mixed"] = "general"
    retrieval_query: str
    grounded: bool
    timings_ms: dict[str, float] = Field(default_factory=dict)
    conversation_context_included: bool = False


class SearchHitWithRerank(SearchHit):
    rerank_score: float | None = None
    rerank_rank: int | None = None
