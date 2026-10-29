"""Persistent Chroma vector storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from app.retrieval.models import ChunkRecord


def _page_to_metadata(page_number: int | None) -> int:
    return page_number if page_number is not None else -1


def _page_from_metadata(value: int | None) -> int | None:
    if value is None or value < 0:
        return None
    return value


class ChromaStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, persist_path: str, collection_name: str) -> None:
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self._collection.count())

    def list_document_summaries(self) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []
        payload = self._collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = {}
        for metadata in payload.get("metadatas") or []:
            if not metadata:
                continue
            document_id = str(metadata["document_id"])
            if document_id not in grouped:
                grouped[document_id] = {
                    "document_id": document_id,
                    "filename": metadata["filename"],
                    "file_type": metadata["file_type"],
                    "chunk_count": 0,
                }
            grouped[document_id]["chunk_count"] += 1
        return sorted(grouped.values(), key=lambda item: item["filename"])

    def list_all_records(self) -> list[ChunkRecord]:
        if self.count() == 0:
            return []
        payload = self._collection.get(include=["documents", "metadatas"])
        records: list[ChunkRecord] = []
        ids = payload.get("ids") or []
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=False):
            if not metadata or text is None:
                continue
            records.append(
                ChunkRecord(
                    chunk_id=str(chunk_id),
                    document_id=str(metadata["document_id"]),
                    filename=str(metadata["filename"]),
                    file_type=metadata["file_type"],  # type: ignore[arg-type]
                    chunk_index=int(metadata["chunk_index"]),
                    page_number=_page_from_metadata(
                        int(metadata["page_number"])
                        if metadata.get("page_number") is not None
                        else None
                    ),
                    text=str(text),
                )
            )
        return records

    def delete_document(self, document_id: str) -> int:
        existing = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def upsert_records(
        self,
        records: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        if len(records) != len(embeddings):
            raise ValueError("Records and embeddings length mismatch.")
        if not records:
            return
        self._collection.upsert(
            ids=[record.chunk_id for record in records],
            documents=[record.text for record in records],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": record.document_id,
                    "filename": record.filename,
                    "file_type": record.file_type,
                    "chunk_index": record.chunk_index,
                    "page_number": _page_to_metadata(record.page_number),
                }
                for record in records
            ],
        )

    def query_dense(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> tuple[list[str], list[float], list[dict[str, Any]], list[str]]:
        if self.count() == 0:
            return [], [], [], []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        # Convert cosine distance to similarity-like score for API output.
        scores = [float(1.0 - distance) for distance in distances]
        return ids, scores, metadatas, documents

    @staticmethod
    def record_from_metadata(
        chunk_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=str(metadata["document_id"]),
            filename=str(metadata["filename"]),
            file_type=metadata["file_type"],  # type: ignore[arg-type]
            chunk_index=int(metadata["chunk_index"]),
            page_number=_page_from_metadata(
                int(metadata["page_number"])
                if metadata.get("page_number") is not None
                else None
            ),
            text=text,
        )
