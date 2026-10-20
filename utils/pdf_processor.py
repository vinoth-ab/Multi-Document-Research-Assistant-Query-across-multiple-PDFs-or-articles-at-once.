"""
PDF Processing module using PyMuPDF (fitz).
Extracts text page by page, normalizes whitespace, detects scanned/corrupted PDFs,
and provides optional OCR fallback using Tesseract when available.
"""

import io
import re
import hashlib
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Check for Tesseract availability
_HAS_PYTESSERACT = False
try:
    import pytesseract
    from PIL import Image
    # Test if tesseract is accessible in PATH
    try:
        pytesseract.get_tesseract_version()
        _HAS_PYTESSERACT = True
    except Exception:
        _HAS_PYTESSERACT = False
except ImportError:
    _HAS_PYTESSERACT = False


@dataclass
class ExtractedPage:
    """Represents text and metadata extracted from a single PDF page."""
    page_number: int  # 1-indexed
    text: str
    char_count: int
    has_text: bool
    is_scanned: bool = False
    ocr_applied: bool = False


@dataclass
class ProcessedDocument:
    """Represents the complete processed result of an uploaded PDF."""
    document_id: str
    filename: str
    file_size_bytes: int
    page_count: int
    pages: List[ExtractedPage] = field(default_factory=list)
    is_valid: bool = True
    error_message: Optional[str] = None
    is_scanned: bool = False
    total_characters: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def file_size_mb(self) -> float:
        """Return file size formatted in megabytes."""
        return round(self.file_size_bytes / (1024 * 1024), 2)


def generate_document_id(filename: str, file_bytes: bytes) -> str:
    """Generate a deterministic unique document ID using SHA-256 of file content and name."""
    hasher = hashlib.sha256()
    hasher.update(filename.encode("utf-8", errors="ignore"))
    hasher.update(file_bytes)
    # Shorten to 16 hex characters for readable IDs
    return hasher.hexdigest()[:16]


def normalize_whitespace(text: str) -> str:
    """
    Clean and normalize excessive whitespace without destroying paragraph structure.
    - Strips trailing/leading spaces on lines
    - Collapses multiple horizontal spaces/tabs into a single space
    - Preserves double newlines (paragraphs) while collapsing 3+ newlines into 2
    """
    if not text:
        return ""
    
    # Replace non-breaking spaces and form feeds
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    
    # Collapse multiple spaces or tabs into a single space per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned_text = "\n".join(lines)
    
    # Collapse 3 or more consecutive newlines into 2
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    
    return cleaned_text.strip()


def validate_pdf_upload(
    filename: str,
    file_bytes: bytes,
    max_size_mb: int = 20
) -> Tuple[bool, Optional[str]]:
    """
    Validate PDF upload:
    - Checks file extension
    - Checks file size
    - Validates PDF magic header (%PDF-)
    """
    if not filename.lower().endswith(".pdf"):
        return False, f"Invalid file format: '{filename}' is not a PDF file."

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File '{filename}' ({size_mb:.1f} MB) exceeds maximum allowed size of {max_size_mb} MB."

    if len(file_bytes) < 4 or not file_bytes.startswith(b"%PDF"):
        return False, f"File '{filename}' does not appear to be a valid PDF document (missing PDF header)."

    return True, None


def extract_text_from_pdf(
    file_bytes: bytes,
    filename: str,
    enable_ocr_fallback: bool = True
) -> ProcessedDocument:
    """
    Extract text page by page from PDF bytes using PyMuPDF (fitz).
    Handles encrypted, corrupted, and scanned documents gracefully.
    """
    doc_id = generate_document_id(filename, file_bytes)
    file_size = len(file_bytes)

    # Lazy import pymupdf/fitz
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
    except ImportError:
        return ProcessedDocument(
            document_id=doc_id,
            filename=filename,
            file_size_bytes=file_size,
            page_count=0,
            is_valid=False,
            error_message="PyMuPDF library is not installed.",
        )

    try:
        pdf_stream = io.BytesIO(file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
    except Exception as e:
        logger.error(f"Error opening PDF '{filename}': {e}")
        return ProcessedDocument(
            document_id=doc_id,
            filename=filename,
            file_size_bytes=file_size,
            page_count=0,
            is_valid=False,
            error_message=f"Failed to open or parse PDF: {str(e)}",
        )

    # Check for encryption
    if doc.is_encrypted:
        try:
            # Try authenticating with empty password
            if not doc.authenticate(""):
                doc.close()
                return ProcessedDocument(
                    document_id=doc_id,
                    filename=filename,
                    file_size_bytes=file_size,
                    page_count=0,
                    is_valid=False,
                    error_message=f"PDF '{filename}' is password protected and cannot be read.",
                )
        except Exception:
            doc.close()
            return ProcessedDocument(
                document_id=doc_id,
                filename=filename,
                file_size_bytes=file_size,
                page_count=0,
                is_valid=False,
                error_message=f"PDF '{filename}' is encrypted with a password.",
            )

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        return ProcessedDocument(
            document_id=doc_id,
            filename=filename,
            file_size_bytes=file_size,
            page_count=0,
            is_valid=False,
            error_message=f"PDF '{filename}' contains 0 pages.",
        )

    extracted_pages: List[ExtractedPage] = []
    total_chars = 0
    scanned_pages_count = 0

    for page_idx in range(page_count):
        page_num = page_idx + 1
        page = doc[page_idx]
        raw_text = page.get_text("text") or ""
        cleaned = normalize_whitespace(raw_text)
        char_count = len(cleaned)

        is_scanned = False
        ocr_applied = False

        # If page has very little or no text (< 30 characters), check if images exist
        if char_count < 30:
            images = page.get_images()
            if images:
                is_scanned = True
                scanned_pages_count += 1
                # Attempt OCR fallback if available and enabled
                if enable_ocr_fallback and _HAS_PYTESSERACT:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_text = pytesseract.image_to_string(img)
                        ocr_cleaned = normalize_whitespace(ocr_text)
                        if len(ocr_cleaned) > char_count:
                            cleaned = ocr_cleaned
                            char_count = len(cleaned)
                            ocr_applied = True
                    except Exception as ocr_err:
                        logger.warning(f"OCR failed for {filename} page {page_num}: {ocr_err}")

        has_text = char_count > 0
        total_chars += char_count

        extracted_pages.append(
            ExtractedPage(
                page_number=page_num,
                text=cleaned,
                char_count=char_count,
                has_text=has_text,
                is_scanned=is_scanned,
                ocr_applied=ocr_applied,
            )
        )

    doc_metadata = doc.metadata or {}
    doc.close()

    is_overall_scanned = (scanned_pages_count > 0 and total_chars < 150) or (total_chars == 0)

    error_msg = None
    if total_chars == 0:
        if _HAS_PYTESSERACT:
            error_msg = "No extractable text found in this PDF (even after OCR inspection). It may contain blank or unreadable scans."
        else:
            error_msg = (
                "No extractable text found in this PDF. It appears to be a scanned document or image-only PDF. "
                "Tesseract OCR is not installed on this system for automatic text recognition."
            )

    return ProcessedDocument(
        document_id=doc_id,
        filename=filename,
        file_size_bytes=file_size,
        page_count=page_count,
        pages=extracted_pages,
        is_valid=True,
        error_message=error_msg,
        is_scanned=is_overall_scanned,
        total_characters=total_chars,
        metadata=doc_metadata,
    )
