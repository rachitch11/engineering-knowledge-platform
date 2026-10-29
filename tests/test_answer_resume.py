"""Regression tests for resume-style factual Q&A and abstention."""

from __future__ import annotations

import re

import pytest

from app.core.config import Settings
from app.generation.answer import AnswerService, has_sufficient_evidence
from app.generation.models import GeneratedAnswer
from app.generation.providers import MockLLMProvider
from app.ingestion.service import ingest_bytes, stable_document_id
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.indexer import IndexService
from app.retrieval.models import SearchHit
from app.retrieval.reranker import MockReranker
from app.retrieval.search import SearchService

SYNTHETIC_RESUME = """
Rachit Choudhary
Software Engineer

Technical skills: Python, Django, React, HTML, CSS, JavaScript, SQL.
Experience in design and integration of software systems.
""".strip()


class ResumeGroundedMockLLM(MockLLMProvider):
    """Test double that answers from evidence text (not hardcoded production logic)."""

    def generate_answer(self, question, evidence, *, timeout_seconds, max_context_chars):
        _ = (timeout_seconds, max_context_chars)
        if not evidence:
            return GeneratedAnswer(
                answer="I do not have enough indexed evidence to answer confidently.",
                citation_ids=[],
            )
        combined = "\n".join(chunk.text for chunk in evidence)
        lowered = question.lower()

        if "name" in lowered or "who" in lowered:
            match = re.search(r"Rachit Choudhary", combined)
            if match:
                chunk = next(item for item in evidence if "Rachit Choudhary" in item.text)
                return GeneratedAnswer(
                    answer="Rachit Choudhary",
                    citation_ids=[chunk.chunk_id],
                )

        if "skill" in lowered or "programming" in lowered or "language" in lowered:
            chunk = evidence[0]
            return GeneratedAnswer(
                answer=(
                    "Technical skills include Python, Django, React, HTML, CSS, "
                    "JavaScript, and SQL."
                ),
                citation_ids=[chunk.chunk_id],
            )

        return super().generate_answer(question, evidence, timeout_seconds=timeout_seconds, max_context_chars=max_context_chars)


@pytest.fixture
def resume_answer_service(test_settings: Settings) -> AnswerService:
    resume_settings = test_settings.model_copy(
        update={"chunk_size": 4000, "chunk_overlap": 64},
    )
    index = IndexService.from_settings(resume_settings, embedding_model=MockEmbedding())
    search = SearchService(index, resume_settings)
    ingested = ingest_bytes(
        SYNTHETIC_RESUME.encode("utf-8"),
        "synthetic_resume.md",
        resume_settings,
    )
    index.index_chunks(ingested.chunks)
    return AnswerService(
        search,
        resume_settings,
        reranker=MockReranker(),
        llm_provider=ResumeGroundedMockLLM(),
    )


def test_has_sufficient_evidence_accepts_dense_retrieval_without_lexical_overlap() -> None:
    hits = [
        SearchHit(
            chunk_id="c1",
            document_id="d1",
            filename="resume.pdf",
            file_type="pdf",
            chunk_index=0,
            page_number=1,
            text="Rachit Choudhary Software Engineer Python Django",
            score=0.5,
            dense_score=0.42,
            dense_rank=1,
        )
    ]
    assert has_sufficient_evidence("What is the name of the resume guy?", hits) is True


def test_has_sufficient_evidence_rejects_unrelated_hits() -> None:
    hits = [
        SearchHit(
            chunk_id="c1",
            document_id="d1",
            filename="resume.pdf",
            file_type="pdf",
            chunk_index=0,
            page_number=1,
            text="Rachit Choudhary Software Engineer Python Django",
            score=0.01,
            dense_score=0.05,
            bm25_score=0.0,
        )
    ]
    assert has_sufficient_evidence("quantum chromodynamics coupling constant", hits) is False


def test_user_wording_name_question_when_dense_score_is_strong() -> None:
    """Regression for phrasing like 'resume guy' when retrieval is semantically correct."""
    hits = [
        SearchHit(
            chunk_id="c1",
            document_id="d1",
            filename="resume.pdf",
            file_type="pdf",
            chunk_index=0,
            page_number=1,
            text="Rachit Choudhary Software Engineer Python Django React SQL",
            score=0.4,
            dense_score=0.41,
            dense_rank=1,
        )
    ]
    question = "What's the name of the resume guy?"
    assert has_sufficient_evidence(question, hits, min_dense_score=0.24) is True


def test_resume_name_question(resume_answer_service: AnswerService) -> None:
    response = resume_answer_service.answer(
        "Who is Rachit Choudhary?",
        strategy="hybrid",
        rerank=True,
    )
    assert response.sufficient_evidence is True
    assert "Rachit Choudhary" in response.answer
    assert response.citations
    assert response.citations[0].chunk_id in {item.chunk_id for item in response.evidence}


def test_resume_skills_question(resume_answer_service: AnswerService) -> None:
    response = resume_answer_service.answer(
        "What technical skills and programming languages are listed in my resume?",
        strategy="hybrid",
        rerank=True,
    )
    assert response.sufficient_evidence is True
    assert "Python" in response.answer
    assert response.citations


def test_unrelated_question_abstains(resume_answer_service: AnswerService) -> None:
    response = resume_answer_service.answer(
        "What is the quantum chromodynamics coupling constant?",
        strategy="hybrid",
        rerank=True,
    )
    assert response.sufficient_evidence is False
    assert "not have enough" in response.answer.lower()
