"""Unit tests for text chunking."""

import pytest

from app.ingestion.chunker import ChunkingError, chunk_document, split_with_overlap
from app.ingestion.models import ParsedContent, TextSegment


def test_split_with_overlap_respects_boundaries() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = split_with_overlap(text, chunk_size=10, chunk_overlap=2)
    assert chunks == [
        "abcdefghij",
        "ijklmnopqr",
        "qrstuvwxyz",
    ]


def test_split_with_overlap_rejects_invalid_overlap() -> None:
    with pytest.raises(ChunkingError):
        split_with_overlap("abc", chunk_size=10, chunk_overlap=10)


def test_chunk_document_preserves_metadata() -> None:
    parsed = ParsedContent(
        filename="doc.pdf",
        file_type="pdf",
        segments=[
            TextSegment(text="First page text here.", page_number=1),
            TextSegment(text="Second page text here.", page_number=2),
        ],
    )
    chunks = chunk_document(
        "doc-123",
        parsed,
        chunk_size=100,
        chunk_overlap=10,
    )
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "doc-123_chunk_0000"
    assert chunks[0].document_id == "doc-123"
    assert chunks[0].filename == "doc.pdf"
    assert chunks[0].file_type == "pdf"
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_chunk_document_splits_long_page_with_overlap() -> None:
    text = "x" * 25
    parsed = ParsedContent(
        filename="long.pdf",
        file_type="pdf",
        segments=[TextSegment(text=text, page_number=3)],
    )
    chunks = chunk_document(
        "doc-456",
        parsed,
        chunk_size=10,
        chunk_overlap=2,
    )
    assert len(chunks) == 3
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert chunks[0].text == "x" * 10
    assert chunks[1].text == "x" * 10
    assert len(chunks[2].text) == 9
