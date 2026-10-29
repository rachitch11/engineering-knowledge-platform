"""API tests for document upload and validation."""

import io

import pymupdf
from fastapi.testclient import TestClient

from tests.ingestion_fixtures import sample_pdf_bytes


def test_upload_pdf_returns_chunks(
    client: TestClient,
    sample_pdf_bytes: bytes,
) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "report.pdf"
    assert body["file_type"] == "pdf"
    assert body["chunk_count"] == len(body["chunks"])
    assert body["chunk_count"] >= 2
    pages = {chunk["page_number"] for chunk in body["chunks"]}
    assert pages == {1, 2}
    assert body["chunks"][0]["chunk_id"].startswith(body["document_id"])


def test_upload_markdown(client: TestClient) -> None:
    content = b"# Title\n\nSome markdown paragraph."
    response = client.post(
        "/documents/upload",
        files={"file": ("readme.md", io.BytesIO(content), "text/markdown")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["file_type"] == "markdown"
    assert body["chunks"][0]["page_number"] is None
    assert "Title" in body["chunks"][0]["text"]


def test_upload_empty_file_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_unsupported_format_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_upload_too_large_rejected(client: TestClient) -> None:
    oversized = b"x" * (1024 * 1024 + 1)
    response = client.post(
        "/documents/upload",
        files={"file": ("big.md", io.BytesIO(oversized), "text/markdown")},
    )
    assert response.status_code == 413


def test_upload_blank_pdf_rejected(client: TestClient) -> None:
    document = pymupdf.open()
    document.new_page()
    blank_pdf = document.tobytes()
    document.close()
    response = client.post(
        "/documents/upload",
        files={"file": ("blank.pdf", io.BytesIO(blank_pdf), "application/pdf")},
    )
    assert response.status_code == 400
