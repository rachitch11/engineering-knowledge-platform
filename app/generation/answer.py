"""Grounded answer generation using retrieval, reranking, and validated citations."""

from __future__ import annotations

import re

from app.core.config import Settings
from app.generation.exceptions import GenerationError, LLMProviderError
from app.generation.models import (
    AnswerResponse,
    Citation,
    EvidenceChunk,
)
from app.generation.providers import LLMProvider
from app.retrieval.models import SearchHit
from app.retrieval.reranker import Reranker
from app.retrieval.search import SearchService


def _question_tokens(question: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "my",
        "me",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whats",
        "why",
        "with",
    }
    tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    return {token for token in tokens if token not in stopwords}


# Query terms that often appear in questions but not answers (e.g. "resume", "name").
_GENERIC_OVERLAP_TOKENS = {
    "resume",
    "cv",
    "curriculum",
    "vitae",
    "document",
    "file",
    "page",
    "data",
    "information",
    "guy",
    "person",
    "name",
    "listed",
    "list",
    "question",
}


def has_sufficient_evidence(
    question: str,
    hits: list[SearchHit],
    *,
    min_dense_score: float = 0.24,
) -> bool:
    """Decide whether retrieved chunks justify calling the LLM.

    Lexical overlap alone misses valid cases (e.g. "What is the person's name?"
    vs chunk text "Rachit Choudhary"). We also trust BM25 and dense retrieval
    signals when hybrid search has already ranked evidence.
    """
    if not hits:
        return False

    query_tokens = _question_tokens(question)
    for hit in hits[:5]:
        chunk_tokens = set(re.findall(r"[a-z0-9]+", hit.text.lower()))
        overlap = query_tokens & chunk_tokens
        meaningful_overlap = overlap - _GENERIC_OVERLAP_TOKENS
        if meaningful_overlap:
            return True
        if overlap and (hit.bm25_score or 0) > 0:
            return True

    top = hits[0]
    if top.dense_score is not None and top.dense_score >= min_dense_score:
        return True

    max_dense = max((hit.dense_score or 0.0) for hit in hits[:10])
    if max_dense >= min_dense_score:
        return True

    return False


def hits_to_evidence(hits: list[SearchHit]) -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            filename=hit.filename,
            file_type=hit.file_type,
            chunk_index=hit.chunk_index,
            page_number=hit.page_number,
            text=hit.text,
            score=hit.score,
        )
        for hit in hits
    ]


def build_citations(
    evidence: list[EvidenceChunk],
    citation_ids: list[str],
    *,
    excerpt_chars: int = 240,
) -> list[Citation]:
    evidence_map = {chunk.chunk_id: chunk for chunk in evidence}
    citations: list[Citation] = []
    for chunk_id in citation_ids:
        chunk = evidence_map.get(chunk_id)
        if chunk is None:
            continue
        excerpt = chunk.text[:excerpt_chars]
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                excerpt=excerpt,
            )
        )
    return citations


class AnswerService:
    """Retrieve, rerank, and generate grounded answers."""

    def __init__(
        self,
        search_service: SearchService,
        settings: Settings,
        *,
        reranker: Reranker | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._search = search_service
        self._settings = settings
        self._reranker = reranker
        self._llm = llm_provider

    def answer(
        self,
        question: str,
        *,
        strategy: str = "hybrid",
        top_k: int | None = None,
        candidate_k: int | None = None,
        rerank: bool | None = None,
        document_id: str | None = None,
        file_type: str | None = None,
    ) -> AnswerResponse:
        candidate_limit = candidate_k or self._settings.rerank_candidate_k
        final_k = top_k or self._settings.rerank_top_k
        use_rerank = self._settings.rerank_enabled if rerank is None else rerank

        retrieval = self._search.search(
            question,
            top_k=candidate_limit,
            strategy=strategy,  # type: ignore[arg-type]
            document_id=document_id,
            file_type=file_type,  # type: ignore[arg-type]
        )

        hits = retrieval.results
        rerank_applied = False
        if use_rerank and hits and self._reranker is not None:
            ranked_hits = self._reranker.rerank(question, hits, top_k=final_k)
            rerank_applied = True
        else:
            ranked_hits = hits[:final_k]

        if not has_sufficient_evidence(
            question,
            hits,
            min_dense_score=self._settings.evidence_min_dense_score,
        ):
            return AnswerResponse(
                question=question,
                answer=(
                    "I do not have enough relevant indexed evidence to answer this "
                    "question confidently."
                ),
                sufficient_evidence=False,
                citations=[],
                evidence=hits_to_evidence(ranked_hits),
                retrieval_strategy=retrieval.strategy,
                candidate_count=len(retrieval.results),
                rerank_applied=rerank_applied,
            )

        evidence = hits_to_evidence(ranked_hits[: self._settings.max_evidence_chunks])
        llm = self._llm
        if llm is None:
            from app.generation.providers import build_llm_provider

            llm = build_llm_provider(self._settings)

        try:
            generated = llm.generate_answer(
                question,
                evidence,
                timeout_seconds=self._settings.llm_timeout_seconds,
                max_context_chars=self._settings.max_context_chars,
            )
        except LLMProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Answer generation failed: {exc}") from exc

        citations = build_citations(evidence, generated.citation_ids)
        if not generated.answer.strip():
            return AnswerResponse(
                question=question,
                answer=(
                    "I do not have enough relevant indexed evidence to answer this "
                    "question confidently."
                ),
                sufficient_evidence=False,
                citations=[],
                evidence=evidence,
                retrieval_strategy=retrieval.strategy,
                candidate_count=len(retrieval.results),
                rerank_applied=rerank_applied,
            )

        return AnswerResponse(
            question=question,
            answer=generated.answer,
            sufficient_evidence=True,
            citations=citations,
            evidence=evidence,
            retrieval_strategy=retrieval.strategy,
            candidate_count=len(retrieval.results),
            rerank_applied=rerank_applied,
        )
