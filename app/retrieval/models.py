"""Data models for indexing and retrieval."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.ingestion.models import FileType, IngestResult

SearchStrategy = Literal["hybrid", "dense", "bm25"]


class ChunkRecord(BaseModel):
    """Canonical indexed chunk stored in Chroma and mirrored for BM25."""

    chunk_id: str
    document_id: str
    filename: str
    file_type: FileType
    chunk_index: int
    page_number: int | None = None
    text: str


class IndexResult(IngestResult):
    """Result of ingest plus successful indexing."""

    indexed: bool = True


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: FileType
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    strategy: SearchStrategy = "hybrid"
    document_id: str | None = None
    file_type: FileType | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Search query must not be empty.")
        return normalized


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    file_type: FileType
    chunk_index: int
    page_number: int | None = None
    text: str
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    strategy: SearchStrategy
    total: int
    results: list[SearchHit]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool
    chunks_removed: int
