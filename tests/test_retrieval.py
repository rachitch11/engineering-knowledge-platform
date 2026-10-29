"""Retrieval and indexing unit/integration tests."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.ingestion.models import DocumentChunk
from app.ingestion.service import stable_document_id
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.exceptions import DocumentNotFoundError, EmptyIndexError, EmptyQueryError
from app.retrieval.indexer import IndexService
from app.retrieval.search import SearchService
from tests.ingestion_fixtures import sample_pdf_bytes


@pytest.fixture
def index_service(test_settings: Settings) -> IndexService:
    return IndexService.from_settings(test_settings, embedding_model=MockEmbedding())


@pytest.fixture
def search_service(index_service: IndexService, test_settings: Settings) -> SearchService:
    return SearchService(index_service, test_settings)


def _sample_chunks(document_id: str = "doc-1") -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"{document_id}_chunk_0000",
            document_id=document_id,
            filename="alpha.md",
            file_type="markdown",
            chunk_index=0,
            page_number=None,
            text="python asyncio event loop tutorial",
        ),
        DocumentChunk(
            chunk_id=f"{document_id}_chunk_0001",
            document_id=document_id,
            filename="alpha.md",
            file_type="markdown",
            chunk_index=1,
            page_number=None,
            text="keyword bm25 retrieval ranking example",
        ),
    ]


def test_index_and_list_documents(index_service: IndexService) -> None:
    indexed = index_service.index_chunks(_sample_chunks())
    assert indexed == 2
    documents = index_service.list_documents()
    assert len(documents) == 1
    assert documents[0].chunk_count == 2


def test_reindex_replaces_existing_chunks(index_service: IndexService) -> None:
    document_id = stable_document_id("alpha.md")
    index_service.index_chunks(_sample_chunks(document_id))
    updated = [
        DocumentChunk(
            chunk_id=f"{document_id}_chunk_0000",
            document_id=document_id,
            filename="alpha.md",
            file_type="markdown",
            chunk_index=0,
            page_number=None,
            text="updated content only",
        )
    ]
    index_service.index_chunks(updated)
    assert index_service.chroma.count() == 1
    assert index_service.bm25.records[0].text == "updated content only"


def test_delete_document_keeps_indexes_consistent(index_service: IndexService) -> None:
    document_id = stable_document_id("alpha.md")
    index_service.index_chunks(_sample_chunks(document_id))
    removed = index_service.delete_document(document_id)
    assert removed == 2
    assert index_service.chroma.count() == 0
    assert index_service.bm25.is_empty()


def test_delete_missing_document_raises(index_service: IndexService) -> None:
    with pytest.raises(DocumentNotFoundError):
        index_service.delete_document("missing-doc")


def test_bm25_search_returns_matches(search_service: SearchService, index_service: IndexService) -> None:
    index_service.index_chunks(_sample_chunks(stable_document_id("alpha.md")))
    response = search_service.search("ranking example", strategy="bm25")
    assert response.total >= 1
    assert "ranking" in response.results[0].text


def test_dense_search_returns_results(search_service: SearchService, index_service: IndexService) -> None:
    index_service.index_chunks(_sample_chunks(stable_document_id("alpha.md")))
    response = search_service.search("python asyncio", strategy="dense")
    assert response.total >= 1


def test_hybrid_search_deduplicates_chunks(search_service: SearchService, index_service: IndexService) -> None:
    index_service.index_chunks(_sample_chunks(stable_document_id("alpha.md")))
    response = search_service.search("python retrieval", strategy="hybrid")
    chunk_ids = [hit.chunk_id for hit in response.results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_metadata_filter_by_file_type(search_service: SearchService, index_service: IndexService) -> None:
    index_service.index_chunks(_sample_chunks(stable_document_id("alpha.md")))
    response = search_service.search(
        "python",
        strategy="bm25",
        file_type="markdown",
    )
    assert all(hit.file_type == "markdown" for hit in response.results)


def test_empty_query_raises(search_service: SearchService) -> None:
    with pytest.raises(EmptyQueryError):
        search_service.search("   ", strategy="bm25")


def test_dense_search_on_empty_index_raises(search_service: SearchService) -> None:
    with pytest.raises(EmptyIndexError):
        search_service.search("anything", strategy="dense")


def test_hybrid_search_on_empty_index_returns_empty(search_service: SearchService) -> None:
    response = search_service.search("anything", strategy="hybrid")
    assert response.total == 0


def test_chroma_persistence_across_service_instances(test_settings: Settings) -> None:
    document_id = stable_document_id("persist.md")
    first = IndexService.from_settings(test_settings, embedding_model=MockEmbedding())
    first.index_chunks(_sample_chunks(document_id))

    second = IndexService.from_settings(test_settings, embedding_model=MockEmbedding())
    assert second.chroma.count() == 2
    assert len(second.bm25.records) == 2


def test_stable_document_id_for_same_filename() -> None:
    first = stable_document_id("Manual.PDF")
    second = stable_document_id("manual.pdf")
    assert first == second


def test_index_api_and_search_api(
    client: TestClient,
    sample_pdf_bytes: bytes,
) -> None:
    index_response = client.post(
        "/documents/index",
        files={"file": ("report.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert index_response.status_code == 200
    body = index_response.json()
    assert body["indexed"] is True

    list_response = client.get("/documents")
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1

    search_response = client.post(
        "/search",
        json={"query": "Alpha content", "strategy": "hybrid"},
    )
    assert search_response.status_code == 200
    assert search_response.json()["total"] >= 1

    document_id = body["document_id"]
    reindex_response = client.post(
        "/documents/index",
        files={"file": ("report.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert reindex_response.status_code == 200
    assert reindex_response.json()["document_id"] == document_id

    delete_response = client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["chunks_removed"] >= 1
    assert client.get("/documents").json()["total"] == 0


def test_search_validation_error(client: TestClient) -> None:
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 422
