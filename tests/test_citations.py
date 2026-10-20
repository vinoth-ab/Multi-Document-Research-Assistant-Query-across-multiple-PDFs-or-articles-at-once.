"""
Unit tests for Citations module.
Tests verified citation extraction, deduplication, missing page handling, and formatting.
"""

from utils.vector_store import RetrievedPassage
from utils.citations import CitationManager, Citation


def test_citation_extraction_and_deduplication():
    # Simulate 3 passages, two from the same page with different similarity scores
    p1 = RetrievedPassage(
        chunk_id="doc1_p1_c0",
        text="The attention mechanism calculates dynamic weights between tokens.",
        metadata={"filename": "attention.pdf", "page_number": 1, "document_id": "doc1"},
        distance=0.15,  # similarity ~0.85
    )
    p2 = RetrievedPassage(
        chunk_id="doc1_p1_c1",
        text="Multi-head attention projects queries, keys, and values into subspaces.",
        metadata={"filename": "attention.pdf", "page_number": 1, "document_id": "doc1"},
        distance=0.10,  # similarity ~0.90 (higher score)
    )
    p3 = RetrievedPassage(
        chunk_id="doc2_p4_c0",
        text="Residual connections prevent gradient degradation in deep architectures.",
        metadata={"filename": "resnet.pdf", "page_number": 4, "document_id": "doc2"},
        distance=0.25,  # similarity ~0.75
    )

    passages = [p1, p2, p3]
    citations = CitationManager.extract_citations(passages)

    # Should deduplicate p1 and p2 into a single citation for (attention.pdf, page 1)
    assert len(citations) == 2

    # First citation should be attention.pdf page 1 with the highest score
    assert citations[0].filename == "attention.pdf"
    assert citations[0].page_number == 1
    assert citations[0].similarity_score >= 0.89
    assert citations[0].formatted_citation == "[Source: attention.pdf, Page 1]"

    # Second citation is resnet.pdf page 4
    assert citations[1].filename == "resnet.pdf"
    assert citations[1].page_number == 4
    assert citations[1].formatted_citation == "[Source: resnet.pdf, Page 4]"


def test_citation_missing_page_number():
    p = RetrievedPassage(
        chunk_id="docX_c0",
        text="Document with unrecorded page number.",
        metadata={"filename": "notes.pdf", "page_number": 0, "document_id": "docX"},
        distance=0.2,
    )
    citations = CitationManager.extract_citations([p])
    assert len(citations) == 1
    assert citations[0].formatted_citation == "[Source: notes.pdf, Page Unavailable]"


def test_format_citations_summary():
    c = Citation(
        document_id="doc1",
        filename="bert_paper.pdf",
        page_number=3,
        chunk_id="doc1_p3_c0",
        snippet="BERT uses masked language modeling and next sentence prediction.",
        similarity_score=0.92,
    )
    summary_md = CitationManager.format_citations_summary([c])
    assert "Sources & Citations" in summary_md
    assert "[Source: bert_paper.pdf, Page 3]" in summary_md
    assert "92%" in summary_md
    assert "BERT uses masked language modeling" in summary_md
