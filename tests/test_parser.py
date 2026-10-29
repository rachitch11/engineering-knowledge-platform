"""Unit tests for document parsers."""

import pymupdf
import pytest

from app.ingestion.exceptions import EmptyDocumentError, UnsupportedFormatError
from app.ingestion.parser import (
    file_type_from_filename,
    parse_document,
    parse_html,
    parse_markdown,
    parse_pdf,
)


def test_file_type_from_filename_pdf() -> None:
    assert file_type_from_filename("report.PDF") == "pdf"


def test_file_type_from_filename_markdown() -> None:
    assert file_type_from_filename("notes.md") == "markdown"


def test_file_type_from_filename_html() -> None:
    assert file_type_from_filename("page.htm") == "html"


def test_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
        file_type_from_filename("data.csv")


def test_parse_pdf_page_level_extraction(sample_pdf_bytes: bytes) -> None:
    segments = parse_pdf(sample_pdf_bytes)
    assert len(segments) == 2
    assert segments[0].page_number == 1
    assert segments[1].page_number == 2
    assert "Alpha" in segments[0].text
    assert "Beta" in segments[1].text


def test_parse_markdown_extracts_text() -> None:
    content = b"# Heading\n\nImportant **markdown** content."
    segments = parse_markdown(content)
    assert len(segments) == 1
    assert segments[0].page_number is None
    assert "Heading" in segments[0].text
    assert "markdown" in segments[0].text


def test_parse_html_extracts_visible_text() -> None:
    content = b"<html><body><h1>Title</h1><p>Body text</p></body></html>"
    segments = parse_html(content)
    assert len(segments) == 1
    assert "Title" in segments[0].text
    assert "Body text" in segments[0].text


def test_parse_document_integrates_filename_and_type(sample_pdf_bytes: bytes) -> None:
    parsed = parse_document(sample_pdf_bytes, "manual.pdf")
    assert parsed.filename == "manual.pdf"
    assert parsed.file_type == "pdf"
    assert len(parsed.segments) == 2


def test_empty_pdf_raises_empty_document_error() -> None:
    document = pymupdf.open()
    document.new_page()
    empty_pdf = document.tobytes()
    document.close()
    with pytest.raises(EmptyDocumentError):
        parse_document(empty_pdf, "blank.pdf")


def test_empty_html_raises_on_parse_document() -> None:
    with pytest.raises(EmptyDocumentError):
        parse_document(b"<html><body></body></html>", "empty.html")
