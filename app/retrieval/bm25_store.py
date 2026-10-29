"""BM25 keyword index rebuilt from canonical Chroma records."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.retrieval.models import ChunkRecord


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Store:
    """In-memory BM25 index with optional JSON snapshot for faster restarts."""

    def __init__(self, persist_path: str) -> None:
        self._persist_path = Path(persist_path)
        self._records: list[ChunkRecord] = []
        self._bm25: BM25Okapi | None = None
        self._load_snapshot()

    @property
    def records(self) -> list[ChunkRecord]:
        return list(self._records)

    def is_empty(self) -> bool:
        return not self._records

    def rebuild_from_records(self, records: list[ChunkRecord]) -> None:
        self._records = list(records)
        self._rebuild_index()

    def remove_document(self, document_id: str) -> None:
        filtered = [record for record in self._records if record.document_id != document_id]
        if len(filtered) == len(self._records):
            return
        self._records = filtered
        self._rebuild_index()

    def add_records(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        self._records.extend(records)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self._records:
            self._bm25 = None
            self._persist_snapshot()
            return
        corpus = [tokenize(record.text) for record in self._records]
        self._bm25 = BM25Okapi(corpus)
        self._persist_snapshot()

    def search(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
        file_type: str | None = None,
    ) -> list[tuple[ChunkRecord, float]]:
        if not query.strip() or self._bm25 is None or not self._records:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        corpus_tokens = [tokenize(record.text) for record in self._records]
        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(self._records)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        # Fall back to BM25's top-n ordering when scores tie at zero.
        if ranked_indices and float(scores[ranked_indices[0]]) == 0.0:
            top_token_lists = self._bm25.get_top_n(
                query_tokens,
                corpus_tokens,
                n=len(corpus_tokens),
            )
            token_to_index = {tuple(tokens): index for index, tokens in enumerate(corpus_tokens)}
            ranked_indices = [
                token_to_index[tuple(tokens)]
                for tokens in top_token_lists
                if tuple(tokens) in token_to_index
            ]

        results: list[tuple[ChunkRecord, float]] = []
        for index in ranked_indices:
            record = self._records[index]
            if document_id and record.document_id != document_id:
                continue
            if file_type and record.file_type != file_type:
                continue
            results.append((record, float(scores[index])))
            if len(results) >= top_k:
                break
        return results

    def _persist_snapshot(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "records": [record.model_dump() for record in self._records],
        }
        self._persist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_snapshot(self) -> None:
        if not self._persist_path.exists():
            self._records = []
            self._bm25 = None
            return
        payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
        self._records = [ChunkRecord.model_validate(item) for item in payload["records"]]
        if not self._records:
            self._bm25 = None
            return
        corpus = [tokenize(record.text) for record in self._records]
        self._bm25 = BM25Okapi(corpus)
