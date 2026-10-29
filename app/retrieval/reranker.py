"""Optional cross-encoder reranking for retrieved candidates."""

from __future__ import annotations

import re
from typing import Protocol

from app.retrieval.models import SearchHit


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[SearchHit],
        *,
        top_k: int,
    ) -> list[SearchHit]: ...


class LazyCrossEncoderReranker:
    """Lazy-loads a sentence-transformers cross encoder."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[SearchHit],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        if not candidates:
            return []
        model = self._load()
        pairs = [[query, hit.text] for hit in candidates]
        scores = model.predict(pairs)  # type: ignore[union-attr]
        ranked = sorted(
            zip(candidates, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [hit for hit, _score in ranked[:top_k]]


class MockReranker:
    """Deterministic reranker based on lexical overlap (for tests)."""

    def rerank(
        self,
        query: str,
        candidates: list[SearchHit],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))

        def overlap_score(text: str) -> int:
            tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
            return len(query_tokens & tokens)

        ranked = sorted(
            candidates,
            key=lambda hit: (overlap_score(hit.text), hit.score),
            reverse=True,
        )
        return ranked[:top_k]
