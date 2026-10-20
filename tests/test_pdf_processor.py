"""
Unit tests for PDF Processor module.
Tests PDF validation, text extraction, page counting, corrupted/empty PDF handling.
"""

import io
import pytest
import pymupdf as fitz
from utils.pdf_processor import (
    validate_pdf_upload,
    extract_text_from_pdf,
    normalize_whitespace,
    generate_document_id,
)


def create_dummy_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory PDF with specified text per page."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_normalize_whitespace():
    raw = "   This   is   a   test.\n\n\n\nAnother    paragraph.\r\nLine  with   tabs\t\there."
    normalized = normalize_whitespace(raw)
    assert "   " not in normalized
    assert "\t" not in normalized
    assert "\n\n\n" not in normalized
    assert "This is a test." in normalized
    assert "Another paragraph." in normalized


def test_validate_pdf_upload():
    valid_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj"
    is_valid, err = validate_pdf_upload("sample.pdf", valid_pdf, max_size_mb=20)
    assert is_valid is True
    assert err is None

    # Invalid extension
    is_valid, err = validate_pdf_upload("sample.docx", valid_pdf)
    assert is_valid is False
    assert "Invalid file format" in err

    # Invalid header
    is_valid, err = validate_pdf_upload("fake.pdf", b"NOT A PDF HEADER")
    assert is_valid is False
    assert "missing PDF header" in err

    # Over size limit
    is_valid, err = validate_pdf_upload("large.pdf", valid_pdf, max_size_mb=0)
    assert is_valid is False
    assert "exceeds maximum allowed size" in err


def test_extract_text_valid_multi_page_pdf():
    pdf_bytes = create_dummy_pdf_bytes([
        "Page 1: Deep learning models have shown high empirical success in NLP tasks.",
        "Page 2: Vector embeddings and RAG pipelines significantly reduce model hallucination.",
        "Page 3: Conclusion and future work on cross-document comparative analysis."
    ])

    doc = extract_text_from_pdf(pdf_bytes, "test_paper.pdf")
    assert doc.is_valid is True
    assert doc.page_count == 3
    assert len(doc.pages) == 3
    assert doc.filename == "test_paper.pdf"
    assert len(doc.document_id) == 16
    assert "Deep learning" in doc.pages[0].text
    assert doc.pages[0].page_number == 1
    assert "Vector embeddings" in doc.pages[1].text
    assert doc.pages[1].page_number == 2
    assert "Conclusion" in doc.pages[2].text
    assert doc.pages[2].page_number == 3


def test_extract_text_empty_pdf():
    # PDF with blank page (no text)
    pdf_bytes = create_dummy_pdf_bytes([""])
    doc = extract_text_from_pdf(pdf_bytes, "empty.pdf")
    assert doc.is_valid is True
    assert doc.page_count == 1
    assert doc.total_characters == 0
    assert doc.error_message is not None
    assert "No extractable text found" in doc.error_message


def test_extract_corrupted_pdf():
    corrupted_bytes = b"%PDF-corrupted-data-that-is-not-a-valid-structure"
    doc = extract_text_from_pdf(corrupted_bytes, "corrupt.pdf")
    assert doc.is_valid is False
    assert doc.error_message is not None
    assert "Failed to open or parse PDF" in doc.error_message


def test_deterministic_document_id():
    data = b"%PDF-sample-content-12345"
    id1 = generate_document_id("paper1.pdf", data)
    id2 = generate_document_id("paper1.pdf", data)
    id3 = generate_document_id("paper2.pdf", data)
    assert id1 == id2
    assert id1 != id3
