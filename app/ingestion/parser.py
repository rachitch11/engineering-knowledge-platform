"""Parse PDF, Markdown, and HTML documents into page-aware text segments."""

from __future__ import annotations

import re
from pathlib import PurePath

import pymupdf
import markdown
from bs4 import BeautifulSoup

from app.ingestion.exceptions import EmptyDocumentError, UnsupportedFormatError
from app.ingestion.models import FileType, ParsedContent, TextSegment

SUPPORTED_EXTENSIONS: dict[str, FileType] = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
}


def file_type_from_filename(filename: str) -> FileType:
    """Resolve supported file type from a filename extension."""
    suffix = PurePath(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"Unsupported file format '{suffix or '(none)'}'. "
            f"Supported extensions: {supported}."
        )
    return SUPPORTED_EXTENSIONS[suffix]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_pdf(content: bytes) -> list[TextSegment]:
    """Extract text from each PDF page using PyMuPDF."""
    segments: list[TextSegment] = []
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                text = _normalize_whitespace(page.get_text("text"))
                if text:
                    segments.append(
                        TextSegment(text=text, page_number=page_index + 1)
                    )
    except Exception as exc:  # noqa: BLE001 — surface as empty/invalid document
        raise EmptyDocumentError(f"Could not read PDF: {exc}") from exc
    return segments


def parse_markdown(content: bytes) -> list[TextSegment]:
    """Convert Markdown to plain text via HTML rendering."""
    try:
        raw = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EmptyDocumentError("Markdown file must be UTF-8 encoded.") from exc
    html = markdown.markdown(raw)
    text = _normalize_whitespace(BeautifulSoup(html, "html.parser").get_text())
    if not text:
        return []
    return [TextSegment(text=text, page_number=None)]


def parse_html(content: bytes) -> list[TextSegment]:
    """Extract visible text from HTML."""
    try:
        raw = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EmptyDocumentError("HTML file must be UTF-8 encoded.") from exc
    text = _normalize_whitespace(BeautifulSoup(raw, "html.parser").get_text())
    if not text:
        return []
    return [TextSegment(text=text, page_number=None)]


def parse_document(content: bytes, filename: str) -> ParsedContent:
    """Parse raw bytes into structured segments based on filename extension."""
    file_type = file_type_from_filename(filename)
    if file_type == "pdf":
        segments = parse_pdf(content)
    elif file_type == "markdown":
        segments = parse_markdown(content)
    else:
        segments = parse_html(content)

    if not segments:
        raise EmptyDocumentError(f"No extractable text found in '{filename}'.")

    return ParsedContent(filename=filename, file_type=file_type, segments=segments)
