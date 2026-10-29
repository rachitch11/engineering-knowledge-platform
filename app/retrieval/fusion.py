"""Rank fusion utilities for hybrid retrieval."""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_ids: list[list[str]],
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Combine ranked chunk-id lists using Reciprocal Rank Fusion."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    scores: dict[str, float] = {}
    for result_list in ranked_ids:
        for rank, chunk_id in enumerate(result_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
