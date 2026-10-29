"""Data models for parsed documents and text chunks."""

from typing import Literal

from pydantic import BaseModel, Field

FileType = Literal["pdf", "markdown", "html"]


class TextSegment(BaseModel):
    """A contiguous span of text, optionally tied to a source page."""

    text: str
    page_number: int | None = None


class ParsedContent(BaseModel):
    """Structured output from a format-specific parser."""

    filename: str
    file_type: FileType
    segments: list[TextSegment]


class DocumentChunk(BaseModel):
    """A retrieval-ready chunk with provenance metadata."""

    chunk_id: str
    document_id: str
    filename: str
    file_type: FileType
    page_number: int | None = None
    chunk_index: int = Field(ge=0)
    text: str


class IngestResult(BaseModel):
    """Result of parsing and chunking an uploaded document."""

    document_id: str
    filename: str
    file_type: FileType
    chunk_count: int
    chunks: list[DocumentChunk]
