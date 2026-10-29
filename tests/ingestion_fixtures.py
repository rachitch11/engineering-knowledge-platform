"""Fixtures and helpers for ingestion tests."""

from __future__ import annotations

import pymupdf
import pytest


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Two-page PDF with distinct page text."""
    document = pymupdf.open()
    page_one = document.new_page()
    page_one.insert_text((72, 72), "Alpha content on page one.")
    page_two = document.new_page()
    page_two.insert_text((72, 72), "Beta content on page two.")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


@pytest.fixture
def long_text() -> str:
    return "word " * 200
