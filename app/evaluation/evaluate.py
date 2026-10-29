"""Evaluation harness for retrieval strategies and grounded answers."""

from __future__ import annotations

import time
from typing import Literal

from app.core.config import Settings
from app.evaluation.dataset import EVAL_DOCUMENTS, EVAL_QUESTIONS, EvalQuestion
from app.evaluation.metrics import (
    AnswerMetricResult,
    EvaluationReport,
    QueryMetricResult,
    aggregate_strategy,
    answer_fact_recall,
    citation_accuracy,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_chunk_ids,
)
from app.generation.answer import AnswerService
from app.generation.providers import LLMProvider
from app.ingestion.service import ingest_bytes
from app.retrieval.indexer import IndexService
from app.retrieval.models import SearchStrategy
from app.retrieval.reranker import Reranker
from app.retrieval.search import SearchService

EvalMode = Literal["mock_deterministic", "live_models"]


class EvaluationHarness:
    """Runs reproducible evaluations against a temporary index."""

    def __init__(
        self,
        settings: Settings,
        index_service: IndexService,
        search_service: SearchService,
        answer_service: AnswerService,
        *,
        mode: EvalMode = "mock_deterministic",
    ) -> None:
        self._settings = settings
        self._index = index_service
        self._search = search_service
        self._answer = answer_service
        self._mode = mode

    def seed_dataset(self) -> None:
        for document in EVAL_DOCUMENTS:
            ingested = ingest_bytes(
                document.content.encode("utf-8"),
                document.filename,
                self._settings,
            )
            self._index.index_chunks(ingested.chunks)

    def run(self, *, k: int = 5) -> EvaluationReport:
        report = EvaluationReport(mode=self._mode, k=k)
        if self._mode == "mock_deterministic":
            report.notes.append(
                "Metrics below were measured with MockEmbedding, MockReranker, "
                "and MockLLMProvider (deterministic; not live model quality)."
            )
        else:
            report.notes.append(
                "Live model mode requires manual execution with downloaded models."
            )

        answerable = [question for question in EVAL_QUESTIONS if question.answerable]
        strategies: list[tuple[str, SearchStrategy, bool]] = [
            ("dense", "dense", False),
            ("bm25", "bm25", False),
            ("hybrid", "hybrid", False),
            ("hybrid_rerank", "hybrid", True),
        ]

        for strategy_name, strategy, use_rerank in strategies:
            query_results: list[QueryMetricResult] = []
            for question in answerable:
                query_results.append(
                    self._evaluate_retrieval_question(
                        question,
                        strategy=strategy,
                        strategy_name=strategy_name,
                        use_rerank=use_rerank,
                        k=k,
                    )
                )
            report.retrieval.append(aggregate_strategy(query_results))

        for question in EVAL_QUESTIONS:
            report.answers.append(self._evaluate_answer_question(question))

        return report

    def _evaluate_retrieval_question(
        self,
        question: EvalQuestion,
        *,
        strategy: SearchStrategy,
        strategy_name: str,
        use_rerank: bool,
        k: int,
    ) -> QueryMetricResult:
        start = time.perf_counter()
        if use_rerank:
            response = self._answer.answer(
                question.question,
                strategy=strategy,
                top_k=k,
                candidate_k=max(k, self._settings.rerank_candidate_k),
                rerank=True,
            )
            hits = response.evidence
            retrieved_ids = [item.chunk_id for item in hits]
            retrieved_texts = {item.chunk_id: item.text for item in hits}
            retrieved_filenames = {item.chunk_id: item.filename for item in hits}
        else:
            search = self._search.search(
                question.question,
                top_k=k,
                strategy=strategy,
            )
            hits = search.results
            retrieved_ids = [item.chunk_id for item in hits]
            retrieved_texts = {item.chunk_id: item.text for item in hits}
            retrieved_filenames = {item.chunk_id: item.filename for item in hits}

        relevant = relevant_chunk_ids(
            retrieved_ids,
            retrieved_texts,
            question.expected_chunk_substrings,
            question.expected_filename,
            retrieved_filenames,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return QueryMetricResult(
            question_id=question.question_id,
            strategy=strategy_name,
            recall_at_k=recall_at_k(relevant, retrieved_ids, k),
            mrr=reciprocal_rank(relevant, retrieved_ids),
            ndcg_at_k=ndcg_at_k(relevant, retrieved_ids, k),
            latency_ms=elapsed_ms,
            retrieved_count=len(retrieved_ids),
        )

    def _evaluate_answer_question(self, question: EvalQuestion) -> AnswerMetricResult:
        start = time.perf_counter()
        response = self._answer.answer(
            question.question,
            strategy="hybrid",
            rerank=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        evidence_texts = {item.chunk_id: item.text for item in response.evidence}
        cited_ids = [item.chunk_id for item in response.citations]
        return AnswerMetricResult(
            question_id=question.question_id,
            answerable=question.answerable,
            sufficient_evidence=response.sufficient_evidence,
            citation_accuracy=citation_accuracy(
                cited_ids,
                evidence_texts,
                question.expected_answer_facts,
            ),
            answer_fact_recall=answer_fact_recall(
                response.answer,
                question.expected_answer_facts,
            ),
            latency_ms=elapsed_ms,
        )


def build_harness(
    settings: Settings,
    *,
    embedding_model: object | None = None,
    reranker: Reranker | None = None,
    llm_provider: LLMProvider | None = None,
    mode: EvalMode = "mock_deterministic",
) -> EvaluationHarness:
    index = IndexService.from_settings(settings, embedding_model=embedding_model)  # type: ignore[arg-type]
    search = SearchService(index, settings)
    answer = AnswerService(
        search,
        settings,
        reranker=reranker,
        llm_provider=llm_provider,
    )
    return EvaluationHarness(
        settings,
        index,
        search,
        answer,
        mode=mode,
    )
