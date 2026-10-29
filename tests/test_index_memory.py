"""Regression tests for bounded indexing memory behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.config import Settings
from app.ingestion.models import DocumentChunk
from app.main import create_app
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.indexer import IndexService
from app.retrieval.reranker import MockReranker


def _chunks(count: int, *, document_id: str = "doc-1") -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"chunk-{index}",
            document_id=document_id,
            filename="sample.md",
            file_type="markdown",
            chunk_index=index,
            page_number=None,
            text=f"Chunk text number {index} with some keywords.",
        )
        for index in range(count)
    ]


def test_index_chunks_batches_embedding_calls(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        index_embedding_batch_size=4,
        memory_constrained=False,
    )
    encode_calls: list[int] = []

    class CountingEmbedding(MockEmbedding):
        def encode(self, texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
            encode_calls.append(len(texts))
            return super().encode(texts, batch_size=batch_size)

    index = IndexService.from_settings(settings, embedding_model=CountingEmbedding())
    indexed = index.index_chunks(_chunks(10))
    assert indexed == 10
    assert encode_calls == [4, 4, 2]


def test_index_failure_rolls_back_partial_writes(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        index_embedding_batch_size=2,
    )
    call_count = {"n": 0}

    class FlakyEmbedding(MockEmbedding):
        def encode(self, texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise ValueError("simulated OOM during embedding")
            return super().encode(texts, batch_size=batch_size)

    index = IndexService.from_settings(settings, embedding_model=FlakyEmbedding())
    with pytest.raises(Exception, match="simulated OOM"):
        index.index_chunks(_chunks(5))
    assert index.list_documents() == []


def test_bind_services_shares_one_embedding_for_ephemeral_sessions(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "unused"),
        bm25_index_path=str(tmp_path / "unused.json"),
        ephemeral_sessions_enabled=True,
        session_store_root=str(tmp_path / "sessions"),
        llm_provider="mock",
        memory_constrained=False,
    )
    shared = MockEmbedding(dimension=384)
    app = create_app(
        settings=settings,
        embedding_model=shared,
        reranker=MockReranker(),
    )
    with TestClient(app) as client:
        manager = client.app.state.session_manager
        id_a = manager.create_session()
        id_b = manager.create_session()
        embed_a = manager.get_index_service(id_a).embedding_model
        embed_b = manager.get_index_service(id_b).embedding_model
        assert embed_a is embed_b
        assert embed_a is shared


def test_index_does_not_full_chroma_resync_during_ingest(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25.json"),
        index_embedding_batch_size=3,
    )
    index = IndexService.from_settings(settings, embedding_model=MockEmbedding())
    with patch.object(index, "_sync_bm25_from_chroma") as sync:
        index.index_chunks(_chunks(7))
        sync.assert_not_called()
