"""Dense, BM25, and hybrid retrieval strategies."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from app.core.config import Settings
from app.core.timing import PipelineTimer
from app.retrieval.exceptions import EmptyIndexError, EmptyQueryError
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.indexer import IndexService
from app.retrieval.models import ChunkRecord, SearchHit, SearchResponse, SearchStrategy


def _build_metadata_filter(
    *,
    document_id: str | None,
    file_type: str | None,
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if document_id:
        clauses.append({"document_id": document_id})
    if file_type:
        clauses.append({"file_type": file_type})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class SearchService:
    """Executes dense, BM25, and hybrid search over the index."""

    def __init__(self, index_service: IndexService, settings: Settings) -> None:
        self._index = index_service
        self._settings = settings

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        strategy: SearchStrategy = "hybrid",
        document_id: str | None = None,
        file_type: str | None = None,
        timer: PipelineTimer | None = None,
    ) -> SearchResponse:
        normalized = " ".join(query.split())
        if not normalized:
            raise EmptyQueryError("Search query must not be empty.")

        limit = top_k or self._settings.retrieval_top_k
        if self._index.chroma.count() == 0:
            if strategy == "bm25" or strategy == "hybrid":
                return SearchResponse(
                    query=normalized,
                    strategy=strategy,
                    total=0,
                    results=[],
                )
            raise EmptyIndexError("No documents are indexed yet.")

        if strategy == "dense":
            hits = self._dense_search(
                normalized, limit, document_id, file_type, timer=timer
            )
        elif strategy == "bm25":
            hits = self._bm25_search(
                normalized, limit, document_id, file_type, timer=timer
            )
        else:
            hits = self._hybrid_search(
                normalized, limit, document_id, file_type, timer=timer
            )

        return SearchResponse(
            query=normalized,
            strategy=strategy,
            total=len(hits),
            results=hits,
        )

    def _dense_search(
        self,
        query: str,
        top_k: int,
        document_id: str | None,
        file_type: str | None,
        *,
        timer: PipelineTimer | None = None,
    ) -> list[SearchHit]:
        stage = timer.stage("retrieval.embedding") if timer else nullcontext()
        with stage:
            embedding = self._index.embed_query(query)
        stage2 = timer.stage("retrieval.chroma") if timer else nullcontext()
        with stage2:
            where = _build_metadata_filter(document_id=document_id, file_type=file_type)
            ids, scores, metadatas, documents = self._index.chroma.query_dense(
                embedding,
                top_k=top_k,
                where=where,
            )
        hits: list[SearchHit] = []
        for rank, (chunk_id, score, metadata, text) in enumerate(
            zip(ids, scores, metadatas, documents, strict=False),
            start=1,
        ):
            record = self._index.chroma.record_from_metadata(chunk_id, text, metadata)
            hits.append(
                SearchHit(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    filename=record.filename,
                    file_type=record.file_type,
                    chunk_index=record.chunk_index,
                    page_number=record.page_number,
                    text=record.text,
                    score=score,
                    dense_rank=rank,
                    dense_score=score,
                )
            )
        return hits

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        document_id: str | None,
        file_type: str | None,
        *,
        timer: PipelineTimer | None = None,
    ) -> list[SearchHit]:
        stage = timer.stage("retrieval.bm25") if timer else nullcontext()
        with stage:
            ranked = self._index.bm25.search(
                query,
                top_k=top_k,
                document_id=document_id,
                file_type=file_type,
            )
        hits: list[SearchHit] = []
        for rank, (record, score) in enumerate(ranked, start=1):
            hits.append(self._record_to_hit(record, score=score, bm25_rank=rank, bm25_score=score))
        return hits

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        document_id: str | None,
        file_type: str | None,
        *,
        timer: PipelineTimer | None = None,
    ) -> list[SearchHit]:
        dense_hits = self._dense_search(
            query, top_k, document_id, file_type, timer=timer
        )
        bm25_hits = self._bm25_search(
            query, top_k, document_id, file_type, timer=timer
        )

        stage = timer.stage("retrieval.fusion") if timer else nullcontext()
        with stage:
            fused = reciprocal_rank_fusion(
                [
                    [hit.chunk_id for hit in dense_hits],
                    [hit.chunk_id for hit in bm25_hits],
                ],
                rrf_k=self._settings.rrf_k,
            )

            dense_map = {hit.chunk_id: hit for hit in dense_hits}
            bm25_map = {hit.chunk_id: hit for hit in bm25_hits}
            record_map = self._records_by_id()

            hits: list[SearchHit] = []
            for chunk_id, score in fused[:top_k]:
                dense = dense_map.get(chunk_id)
                bm25 = bm25_map.get(chunk_id)
                record = record_map.get(chunk_id)
                if record is None:
                    continue
                hits.append(
                    SearchHit(
                        chunk_id=record.chunk_id,
                        document_id=record.document_id,
                        filename=record.filename,
                        file_type=record.file_type,
                        chunk_index=record.chunk_index,
                        page_number=record.page_number,
                        text=record.text,
                        score=score,
                        dense_rank=dense.dense_rank if dense else None,
                        bm25_rank=bm25.bm25_rank if bm25 else None,
                        dense_score=dense.dense_score if dense else None,
                        bm25_score=bm25.bm25_score if bm25 else None,
                    )
                )
            return hits

    def _records_by_id(self) -> dict[str, ChunkRecord]:
        return {record.chunk_id: record for record in self._index.chroma.list_all_records()}

    @staticmethod
    def _record_to_hit(
        record: ChunkRecord,
        *,
        score: float,
        dense_rank: int | None = None,
        bm25_rank: int | None = None,
        dense_score: float | None = None,
        bm25_score: float | None = None,
    ) -> SearchHit:
        return SearchHit(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            filename=record.filename,
            file_type=record.file_type,
            chunk_index=record.chunk_index,
            page_number=record.page_number,
            text=record.text,
            score=score,
            dense_rank=dense_rank,
            bm25_rank=bm25_rank,
            dense_score=dense_score,
            bm25_score=bm25_score,
        )
