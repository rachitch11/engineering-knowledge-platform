"""Search API routes."""

from fastapi import APIRouter, Depends

from app.retrieval.deps import get_search_service
from app.retrieval.models import SearchRequest, SearchResponse
from app.retrieval.search import SearchService

search_router = APIRouter(prefix="/search", tags=["search"])


@search_router.post("", response_model=SearchResponse)
def search_documents(
    payload: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Search indexed chunks using dense, BM25, or hybrid retrieval."""
    return search_service.search(
        payload.query,
        top_k=payload.top_k,
        strategy=payload.strategy,
        document_id=payload.document_id,
        file_type=payload.file_type,
    )
