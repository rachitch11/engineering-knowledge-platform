"""Split parsed document segments into overlapping chunks with metadata."""

from __future__ import annotations

from app.ingestion.models import DocumentChunk, ParsedContent


class ChunkingError(ValueError):
    """Raised when chunking parameters are invalid."""


def split_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into fixed-size windows with overlap between consecutive chunks."""
    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ChunkingError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("chunk_overlap must be smaller than chunk_size.")

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end]
        if piece:
            chunks.append(piece)
        if end >= text_len:
            break
        start = end - chunk_overlap
    return chunks


def chunk_document(
    document_id: str,
    parsed: ParsedContent,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Produce retrieval chunks while preserving document and page metadata."""
    result: list[DocumentChunk] = []
    chunk_index = 0

    for segment in parsed.segments:
        for piece in split_with_overlap(segment.text, chunk_size, chunk_overlap):
            result.append(
                DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                    document_id=document_id,
                    filename=parsed.filename,
                    file_type=parsed.file_type,
                    page_number=segment.page_number,
                    chunk_index=chunk_index,
                    text=piece,
                )
            )
            chunk_index += 1

    return result
