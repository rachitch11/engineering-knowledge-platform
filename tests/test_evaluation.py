"""Tests for the evaluation harness and measured mock metrics."""

from __future__ import annotations

from app.core.config import Settings
from app.evaluation.evaluate import build_harness
from app.generation.providers import MockLLMProvider
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.reranker import MockReranker


def test_evaluation_harness_reports_mock_metrics(test_settings: Settings) -> None:
    harness = build_harness(
        test_settings,
        embedding_model=MockEmbedding(dimension=384),
        reranker=MockReranker(),
        llm_provider=MockLLMProvider(),
        mode="mock_deterministic",
    )
    harness.seed_dataset()
    report = harness.run(k=5)

    assert report.mode == "mock_deterministic"
    assert len(report.retrieval) == 4
    strategies = {item.strategy: item for item in report.retrieval}
    assert strategies["bm25"].recall_at_k >= strategies["dense"].recall_at_k
    assert strategies["hybrid_rerank"].recall_at_k >= strategies["hybrid"].recall_at_k

    answers = {item.question_id: item for item in report.answers}
    assert answers["q_unanswerable"].sufficient_evidence is False
    assert answers["q_flange_torque"].citation_accuracy == 1.0


def test_evaluation_metrics_are_recorded_not_fabricated(test_settings: Settings) -> None:
    harness = build_harness(
        test_settings,
        embedding_model=MockEmbedding(dimension=384),
        reranker=MockReranker(),
        llm_provider=MockLLMProvider(),
    )
    harness.seed_dataset()
    report = harness.run(k=3)
    for aggregate in report.retrieval:
        assert 0.0 <= aggregate.recall_at_k <= 1.0
        assert 0.0 <= aggregate.mrr <= 1.0
        assert 0.0 <= aggregate.ndcg_at_k <= 1.0
        assert aggregate.mean_latency_ms >= 0.0
        assert 0.0 <= aggregate.failure_rate <= 1.0
