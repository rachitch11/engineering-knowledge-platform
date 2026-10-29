"""Shared pytest fixtures."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app
from app.generation.providers import MockLLMProvider
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.reranker import MockReranker

pytest_plugins = ["ingestion_fixtures"]


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="Engineering Knowledge Intelligence",
        environment="test",
        chroma_path=str(tmp_path / "chroma"),
        bm25_index_path=str(tmp_path / "bm25_index.json"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        max_upload_size_bytes=1024 * 1024,
        chunk_size=50,
        chunk_overlap=10,
        retrieval_top_k=5,
        rerank_enabled=True,
        rerank_candidate_k=10,
        rerank_top_k=3,
        llm_provider="mock",
        max_evidence_chunks=3,
        ephemeral_sessions_enabled=False,
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    get_settings.cache_clear()
    app = create_app(
        settings=test_settings,
        embedding_model=MockEmbedding(dimension=384),
        reranker=MockReranker(),
        llm_provider=MockLLMProvider(),
    )
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
