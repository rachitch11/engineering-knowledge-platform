"""Orchestrate validation, parsing, and chunking for uploaded documents."""

from __future__ import annotations

import uuid
from pathlib import PurePath

from app.core.config import Settings
from app.ingestion.chunker import chunk_document
from app.ingestion.exceptions import EmptyDocumentError, FileTooLargeError
from app.ingestion.models import IngestResult
from app.ingestion.parser import parse_document

DOCUMENT_NAMESPACE = uuid.UUID("a3f7c8e2-11b4-4f3e-9f9e-8e2b3d5c6a01")


def stable_document_id(filename: str) -> str:
    """Deterministic id so re-uploading the same filename replaces indexed content."""
    normalized_name = PurePath(filename).name.lower()
    return str(uuid.uuid5(DOCUMENT_NAMESPACE, f"engineering-knowledge:{normalized_name}"))


def ingest_bytes(
    content: bytes,
    filename: str,
    settings: Settings,
) -> IngestResult:
    """Validate upload constraints, parse content, and return metadata-rich chunks."""
    if not content:
        raise EmptyDocumentError("Uploaded file is empty.")

    if len(content) > settings.max_upload_size_bytes:
        raise FileTooLargeError(
            f"File exceeds maximum upload size of "
            f"{settings.max_upload_size_bytes} bytes."
        )

    parsed = parse_document(content, filename)
    document_id = stable_document_id(filename)
    chunks = chunk_document(
        document_id,
        parsed,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise EmptyDocumentError(f"No chunks produced from '{filename}'.")

    return IngestResult(
        document_id=document_id,
        filename=parsed.filename,
        file_type=parsed.file_type,
        chunk_count=len(chunks),
        chunks=chunks,
    )
