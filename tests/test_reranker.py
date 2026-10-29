"""Tests for reranking behavior."""

from app.retrieval.models import SearchHit
from app.retrieval.reranker import MockReranker


def _hit(chunk_id: str, text: str, score: float = 0.1) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id="doc-1",
        filename="a.md",
        file_type="markdown",
        chunk_index=0,
        page_number=None,
        text=text,
        score=score,
    )


def test_mock_reranker_orders_by_query_overlap() -> None:
    reranker = MockReranker()
    candidates = [
        _hit("c1", "unrelated content about cats"),
        _hit("c2", "python asyncio event loop tutorial"),
    ]
    ranked = reranker.rerank("python asyncio", candidates, top_k=2)
    assert ranked[0].chunk_id == "c2"


def test_mock_reranker_respects_top_k() -> None:
    reranker = MockReranker()
    candidates = [_hit(f"c{i}", f"word{i}") for i in range(5)]
    ranked = reranker.rerank("word3", candidates, top_k=2)
    assert len(ranked) == 2
