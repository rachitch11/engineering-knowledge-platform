"""Index documents in bounded embedding batches to limit peak RSS."""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import Settings
from app.core.memory_diagnostics import log_memory_event
from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_store import BM25Store
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.embeddings import EmbeddingModel, LazySentenceTransformerEmbedding
from app.retrieval.exceptions import DocumentNotFoundError, IndexingError
from app.retrieval.models import ChunkRecord, DocumentSummary

logger = logging.getLogger(__name__)


class IndexService:
    """Coordinates Chroma writes and BM25 rebuilds."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_model = embedding_model or LazySentenceTransformerEmbedding(
            settings.embedding_model
        )
        self._chroma = ChromaStore(
            settings.chroma_path,
            settings.chroma_collection_name,
        )
        self._bm25 = BM25Store(settings.bm25_index_path)
        self._cache_epoch = 0
        self._query_embedding_cache: dict[tuple[int, str], list[float]] = {}
        self._sync_bm25_from_chroma()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        embedding_model: EmbeddingModel | None = None,
    ) -> IndexService:
        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        Path(settings.bm25_index_path).parent.mkdir(parents=True, exist_ok=True)
        return cls(settings, embedding_model=embedding_model)

    def _embedding_batch_size(self) -> int:
        configured = self._settings.index_embedding_batch_size
        if self._settings.memory_constrained:
            return min(configured, 4)
        return configured

    def _sync_bm25_from_chroma(self) -> None:
        records = self._chroma.list_all_records()
        self._bm25.rebuild_from_records(records)

    def _records_from_chunks(self, chunks: list[DocumentChunk]) -> list[ChunkRecord]:
        return [
            ChunkRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                file_type=chunk.file_type,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                text=chunk.text,
            )
            for chunk in chunks
        ]

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Replace all chunks for the document id contained in ``chunks``."""
        if not chunks:
            raise IndexingError("Cannot index an empty chunk list.")

        document_id = chunks[0].document_id
        for chunk in chunks[1:]:
            if chunk.document_id != document_id:
                raise IndexingError("All chunks in one index call must share document_id.")

        records = self._records_from_chunks(chunks)
        max_chunks = self._settings.max_index_chunks_per_document
        if max_chunks > 0 and len(records) > max_chunks:
            raise IndexingError(
                f"Document produced {len(records)} chunks; limit is {max_chunks} on this "
                "host. Shorten the file or raise MAX_INDEX_CHUNKS_PER_DOCUMENT."
            )
        batch_size = self._embedding_batch_size()
        log_memory_event(
            "index_start",
            document_id=document_id,
            chunk_count=len(records),
            batch_size=batch_size,
            memory_constrained=self._settings.memory_constrained,
        )

        self._chroma.delete_document(document_id)
        self._bm25.remove_document(document_id)

        indexed = 0
        bm25_pending: list[ChunkRecord] = []
        try:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                try:
                    embeddings = self._embedding_model.encode(
                        [record.text for record in batch],
                        batch_size=batch_size,
                    )
                except ValueError as exc:
                    raise IndexingError(str(exc)) from exc
                self._chroma.upsert_records(batch, embeddings)
                if self._settings.memory_constrained:
                    bm25_pending.extend(batch)
                else:
                    self._bm25.add_records(batch)
                indexed += len(batch)
                del embeddings, batch
                if self._settings.memory_constrained:
                    from app.core.memory_diagnostics import release_memory

                    release_memory()
            if self._settings.memory_constrained and bm25_pending:
                self._bm25.add_records(bm25_pending)
                del bm25_pending
                from app.core.memory_diagnostics import release_memory

                release_memory()
        except IndexingError:
            self._rollback_document(document_id)
            raise
        except Exception as exc:  # noqa: BLE001
            self._rollback_document(document_id)
            raise IndexingError(f"Failed to index document: {exc}") from exc

        self.invalidate_query_cache()
        log_memory_event("index_complete", chunk_count=indexed)
        return indexed

    def _rollback_document(self, document_id: str) -> None:
        try:
            self._chroma.delete_document(document_id)
        except Exception:  # noqa: BLE001
            logger.exception("index_rollback_chroma_failed document_id=%s", document_id)
        try:
            self._bm25.remove_document(document_id)
        except Exception:  # noqa: BLE001
            logger.exception("index_rollback_bm25_failed document_id=%s", document_id)

    def delete_document(self, document_id: str) -> int:
        removed = self._chroma.delete_document(document_id)
        if removed == 0:
            raise DocumentNotFoundError(f"Document '{document_id}' is not indexed.")
        try:
            self._bm25.remove_document(document_id)
        except Exception as exc:  # noqa: BLE001
            raise IndexingError(
                f"Removed document from Chroma but BM25 update failed: {exc}"
            ) from exc
        self.invalidate_query_cache()
        return removed

    def list_documents(self) -> list[DocumentSummary]:
        summaries = self._chroma.list_document_summaries()
        return [DocumentSummary.model_validate(item) for item in summaries]

    @property
    def chroma(self) -> ChromaStore:
        return self._chroma

    @property
    def bm25(self) -> BM25Store:
        return self._bm25

    @property
    def embedding_model(self) -> EmbeddingModel:
        return self._embedding_model

    def invalidate_query_cache(self) -> None:
        self._cache_epoch += 1
        self._query_embedding_cache.clear()

    def embed_query(self, query: str) -> list[float]:
        key = (self._cache_epoch, query)
        cached = self._query_embedding_cache.get(key)
        if cached is not None:
            return cached
        vector = self._embedding_model.encode([query])[0]
        self._query_embedding_cache[key] = vector
        return vector
