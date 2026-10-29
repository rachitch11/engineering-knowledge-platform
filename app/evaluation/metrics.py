"""Retrieval and answer evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class QueryMetricResult:
    question_id: str
    strategy: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_ms: float
    retrieved_count: int


@dataclass
class StrategyAggregate:
    strategy: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_latency_ms: float
    failure_rate: float
    query_count: int


@dataclass
class AnswerMetricResult:
    question_id: str
    answerable: bool
    sufficient_evidence: bool
    citation_accuracy: float
    answer_fact_recall: float
    latency_ms: float


@dataclass
class EvaluationReport:
    mode: str
    k: int
    retrieval: list[StrategyAggregate] = field(default_factory=list)
    answers: list[AnswerMetricResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def relevant_chunk_ids(
    retrieved_ids: list[str],
    retrieved_texts: dict[str, str],
    expected_substrings: list[str],
    expected_filename: str | None,
    retrieved_filenames: dict[str, str] | None = None,
) -> set[str]:
    relevant: set[str] = set()
    for chunk_id in retrieved_ids:
        text = retrieved_texts.get(chunk_id, "")
        filename = (retrieved_filenames or {}).get(chunk_id, "")
        if expected_filename and filename.endswith(expected_filename):
            if not expected_substrings or any(part in text for part in expected_substrings):
                relevant.add(chunk_id)
                continue
        if any(part in text for part in expected_substrings):
            relevant.add(chunk_id)
    return relevant


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if relevant.intersection(retrieved[:k]) else 0.0


def reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        rel = 1.0 if chunk_id in relevant else 0.0
        dcg += rel / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def aggregate_strategy(results: list[QueryMetricResult]) -> StrategyAggregate:
    count = len(results)
    if count == 0:
        return StrategyAggregate(
            strategy="unknown",
            recall_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            mean_latency_ms=0.0,
            failure_rate=0.0,
            query_count=0,
        )
    failures = sum(1 for item in results if item.retrieved_count == 0)
    return StrategyAggregate(
        strategy=results[0].strategy,
        recall_at_k=sum(item.recall_at_k for item in results) / count,
        mrr=sum(item.mrr for item in results) / count,
        ndcg_at_k=sum(item.ndcg_at_k for item in results) / count,
        mean_latency_ms=sum(item.latency_ms for item in results) / count,
        failure_rate=failures / count,
        query_count=count,
    )


def citation_accuracy(
    cited_chunk_ids: list[str],
    evidence_texts: dict[str, str],
    expected_facts: list[str],
) -> float:
    if not cited_chunk_ids:
        return 0.0
    valid = 0
    for chunk_id in cited_chunk_ids:
        text = evidence_texts.get(chunk_id, "").lower()
        if any(fact.lower() in text for fact in expected_facts):
            valid += 1
    return valid / len(cited_chunk_ids)


def answer_fact_recall(answer: str, expected_facts: list[str]) -> float:
    if not expected_facts:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for fact in expected_facts if fact.lower() in answer_lower)
    return hits / len(expected_facts)
