"""Tests for grounded answer generation."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.generation.answer import AnswerService, build_citations, has_sufficient_evidence
from app.generation.exceptions import LLMProviderError, LLMTimeoutError
from app.generation.models import EvidenceChunk
from app.generation.providers import MockLLMProvider, parse_generated_json
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.indexer import IndexService
from app.retrieval.models import SearchHit
from app.retrieval.reranker import MockReranker
from app.retrieval.search import SearchService
from tests.test_retrieval import _sample_chunks
from app.ingestion.service import stable_document_id


class TimeoutLLM(MockLLMProvider):
    def generate_answer(self, question, evidence, *, timeout_seconds, max_context_chars):
        _ = (question, evidence, timeout_seconds, max_context_chars)
        raise LLMTimeoutError("timeout")


class InvalidCitationLLM(MockLLMProvider):
    def generate_answer(self, question, evidence, *, timeout_seconds, max_context_chars):
        _ = (question, evidence, timeout_seconds, max_context_chars)
        from app.generation.models import GeneratedAnswer

        return GeneratedAnswer(
            answer="Answer with invalid citation",
            citation_ids=["not-a-real-chunk"],
        )


@pytest.fixture
def answer_service(test_settings: Settings) -> AnswerService:
    index = IndexService.from_settings(test_settings, embedding_model=MockEmbedding())
    search = SearchService(index, test_settings)
    return AnswerService(
        search,
        test_settings,
        reranker=MockReranker(),
        llm_provider=MockLLMProvider(),
    )


def test_has_sufficient_evidence_requires_overlap() -> None:
    hits = [
        SearchHit(
            chunk_id="c1",
            document_id="d1",
            filename="f.md",
            file_type="markdown",
            chunk_index=0,
            page_number=None,
            text="completely unrelated",
            score=0.0,
        )
    ]
    assert has_sufficient_evidence("quantum physics", hits) is False


def test_parse_generated_json_filters_unknown_citation_ids() -> None:
    parsed = parse_generated_json(
        '{"answer":"ok","citation_ids":["a","b"]}',
        {"a"},
    )
    assert parsed.citation_ids == ["a"]


def test_build_citations_use_real_excerpts() -> None:
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="manual.pdf",
            file_type="pdf",
            chunk_index=0,
            page_number=2,
            text="Pressure vessel design limits and safety factors.",
            score=1.0,
        )
    ]
    citations = build_citations(evidence, ["chunk-1"])
    assert citations[0].filename == "manual.pdf"
    assert citations[0].page_number == 2
    assert citations[0].excerpt in evidence[0].text


def test_answer_with_insufficient_evidence(answer_service: AnswerService) -> None:
    response = answer_service.answer("quantum entanglement")
    assert response.sufficient_evidence is False
    assert "not have enough" in response.answer.lower()
    assert response.citations == []


def test_answer_with_indexed_evidence_and_valid_citation(
    answer_service: AnswerService,
) -> None:
    document_id = stable_document_id("alpha.md")
    answer_service._search._index.index_chunks(_sample_chunks(document_id))  # noqa: SLF001
    response = answer_service.answer("python asyncio")
    assert response.sufficient_evidence is True
    assert response.citations
    assert response.citations[0].chunk_id in {item.chunk_id for item in response.evidence}


def test_invalid_citations_are_dropped(test_settings: Settings) -> None:
    index = IndexService.from_settings(test_settings, embedding_model=MockEmbedding())
    search = SearchService(index, test_settings)
    service = AnswerService(
        search,
        test_settings,
        reranker=MockReranker(),
        llm_provider=InvalidCitationLLM(),
    )
    document_id = stable_document_id("alpha.md")
    index.index_chunks(_sample_chunks(document_id))
    response = service.answer("python asyncio")
    assert response.citations == []


def test_llm_timeout_surfaces_as_provider_error(test_settings: Settings) -> None:
    index = IndexService.from_settings(test_settings, embedding_model=MockEmbedding())
    search = SearchService(index, test_settings)
    service = AnswerService(
        search,
        test_settings,
        reranker=MockReranker(),
        llm_provider=TimeoutLLM(),
    )
    document_id = stable_document_id("alpha.md")
    index.index_chunks(_sample_chunks(document_id))
    with pytest.raises(LLMTimeoutError):
        service.answer("python asyncio")


from tests.ingestion_fixtures import sample_pdf_bytes  # noqa: F401 — pytest fixture


def test_answer_api_endpoint(client: TestClient, sample_pdf_bytes: bytes) -> None:
    index_response = client.post(
        "/documents/index",
        files={"file": ("report.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert index_response.status_code == 200

    answer_response = client.post(
        "/answer",
        json={"question": "Alpha content", "strategy": "hybrid", "rerank": True},
    )
    assert answer_response.status_code == 200
    body = answer_response.json()
    assert "answer" in body
    assert body["sufficient_evidence"] is True
    assert body["citations"]
