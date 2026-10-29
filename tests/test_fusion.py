"""Unit tests for reciprocal rank fusion."""

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_combines_two_ranked_lists() -> None:
    fused = reciprocal_rank_fusion(
        [
            ["a", "b", "c"],
            ["b", "d", "a"],
        ],
        rrf_k=60,
    )
    assert [item[0] for item in fused[:3]] == ["b", "a", "d"]
    assert fused[0][1] > fused[1][1] > fused[2][1]


def test_rrf_deduplicates_shared_ids() -> None:
    fused = reciprocal_rank_fusion([["x"], ["x"]], rrf_k=60)
    assert fused == [("x", 2 / 61)]


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], rrf_k=0)
