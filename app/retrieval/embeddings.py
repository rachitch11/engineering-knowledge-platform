"""Local embedding generation with lazy model loading."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    """Protocol implemented by production and test embedding backends."""

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def normalize_embedding_input(text: str) -> str:
    """Normalize chunk or query text before embedding."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise ValueError("Cannot embed empty text.")
    return cleaned


class LazySentenceTransformerEmbedding:
    """Loads sentence-transformers on first encode call."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load()
        assert self._dimension is not None
        return self._dimension

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._model_name)
        self._model = model
        self._dimension = int(model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        assert self._model is not None
        normalized = [normalize_embedding_input(text) for text in texts]
        vectors = self._model.encode(  # type: ignore[union-attr]
            normalized,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=float).tolist()


class MockEmbedding:
    """Deterministic embeddings for unit tests (no model download)."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = normalize_embedding_input(text)
            digest = hashlib.sha256(normalized.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self._dimension)
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector.astype(float).tolist())
        return vectors
