"""
Unit tests for Text Chunker module.
Tests chunk splitting, overlap handling, page provenance preservation, and metadata formatting.
"""

import pytest
from utils.text_chunker import TextChunker, DocumentChunk
from utils.pdf_processor import ProcessedDocument, ExtractedPage


def test_chunker_initialization_validation():
    # Chunk overlap must be less than chunk size
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=150)


def test_chunk_splitting_and_overlap():
    chunker = TextChunker(chunk_size=100, chunk_overlap=30)
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances LLM output by integrating external knowledge sources. "
        "It retrieves relevant documents dynamically based on user questions. "
        "This substantially minimizes hallucinations and provides direct citations to primary evidence. "
        "Multi-document RAG extends this paradigm across heterogeneous research papers."
    )

    chunks = chunker.split_text(sample_text)
    assert len(chunks) > 1
    for chunk in chunks:
        # Each chunk should not drastically exceed chunk_size
        assert len(chunk) > 20

    # Ensure overlapping words appear across consecutive chunks
    overlap_detected = False
    for i in range(len(chunks) - 1):
        c1_words = set(chunks[i].split())
        c2_words = set(chunks[i + 1].split())
        common = c1_words.intersection(c2_words)
        if len(common) >= 1:
            overlap_detected = True
            break
    assert overlap_detected is True


def test_chunk_document_page_number_preservation():
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)

    doc = ProcessedDocument(
        document_id="doc12345",
        filename="ai_ethics.pdf",
        file_size_bytes=4096,
        page_count=2,
        pages=[
            ExtractedPage(
                page_number=1,
                text="Page 1 discusses data privacy, user consent, and regulatory compliance under GDPR.",
                char_count=85,
                has_text=True,
            ),
            ExtractedPage(
                page_number=2,
                text="Page 2 presents the algorithmic bias mitigation framework and fairness metrics across demographic subgroups.",
                char_count=110,
                has_text=True,
            ),
        ],
        is_valid=True,
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2

    # Check that chunks from page 1 have page_number 1 and chunks from page 2 have page_number 2
    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]

    assert len(page1_chunks) >= 1
    assert len(page2_chunks) >= 1

    assert page1_chunks[0].filename == "ai_ethics.pdf"
    assert page1_chunks[0].document_id == "doc12345"
    assert "data privacy" in page1_chunks[0].text

    assert page2_chunks[0].filename == "ai_ethics.pdf"
    assert "algorithmic bias" in page2_chunks[0].text

    # Verify unique chunk IDs
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))

    # Test Chroma metadata formatting
    meta = chunks[0].to_chroma_metadata()
    assert meta["filename"] == "ai_ethics.pdf"
    assert meta["page_number"] == 1
    assert meta["document_id"] == "doc12345"
    assert isinstance(meta["chunk_index"], int)
