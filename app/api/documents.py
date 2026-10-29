"""Document upload, indexing, and management routes."""

from fastapi import APIRouter, Depends, File, Request, UploadFile, HTTPException

from app.core.config import Settings
from app.core.memory_diagnostics import log_memory_event
from app.core.security import (
    InvalidDocumentIdError,
    sanitize_upload_filename,
    validate_document_id,
    validate_upload_content_type,
)
from app.ingestion.models import IngestResult
from app.ingestion.service import ingest_bytes
from app.core.session_deps import ephemeral_sessions_enabled, resolve_session_id
from app.retrieval.deps import get_index_service
from app.retrieval.exceptions import IndexingError
from app.retrieval.indexer import IndexService
from app.retrieval.models import (
    DeleteDocumentResponse,
    DocumentListResponse,
    IndexResult,
)

from app.api.routes import get_app_settings

documents_router = APIRouter(prefix="/documents", tags=["documents"])


async def _read_upload(file: UploadFile) -> tuple[str, bytes]:
    filename = sanitize_upload_filename(file.filename or "upload")
    validate_upload_content_type(filename, file.content_type)
    content = await file.read()
    return filename, content


@documents_router.post("/upload", response_model=IngestResult)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_app_settings),
) -> IngestResult:
    """Upload a PDF, Markdown, or HTML file and return parsed text chunks."""
    filename, content = await _read_upload(file)
    return ingest_bytes(content, filename, settings)


@documents_router.post("/index", response_model=IndexResult)
async def index_document(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_app_settings),
    index_service: IndexService = Depends(get_index_service),
) -> IndexResult:
    """Parse an uploaded file and index its chunks in Chroma and BM25."""
    filename, content = await _read_upload(file)
    log_memory_event("upload_read", filename=filename, size_bytes=len(content))
    try:
        ingested = ingest_bytes(content, filename, settings)
    finally:
        del content
    try:
        if ephemeral_sessions_enabled(request):
            session_id = resolve_session_id(request)
            manager = request.app.state.session_manager
            record = manager.get_record(session_id)
            with record.lock:
                index_service.index_chunks(ingested.chunks)
        else:
            index_service.index_chunks(ingested.chunks)
    except IndexingError as exc:
        log_memory_event("index_failed", filename=filename)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return IndexResult(**ingested.model_dump(), indexed=True)


@documents_router.get("", response_model=DocumentListResponse)
def list_indexed_documents(
    index_service: IndexService = Depends(get_index_service),
) -> DocumentListResponse:
    """List documents currently stored in the vector index."""
    documents = index_service.list_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@documents_router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    index_service: IndexService = Depends(get_index_service),
) -> DeleteDocumentResponse:
    """Delete all indexed chunks for a document from Chroma and BM25."""
    try:
        safe_id = validate_document_id(document_id)
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    removed = index_service.delete_document(safe_id)
    return DeleteDocumentResponse(
        document_id=safe_id,
        deleted=True,
        chunks_removed=removed,
    )
