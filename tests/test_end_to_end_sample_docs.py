"""
Integration tests using the generated sample research papers.
Tests multi-document extraction, indexing into ChromaDB, retrieval across multiple files,
citation mapping, and persistence across vector store reloads.
"""

from pathlib import Path
import tempfile
import pytest
from unittest.mock import MagicMock

from utils.pdf_processor import extract_text_from_pdf
from utils.text_chunker import TextChunker
from utils.embeddings import EmbeddingManager
from utils.vector_store import VectorStore
from utils.groq_client import GroqService
from utils.rag_pipeline import RAGPipeline
from utils.citations import CitationManager


@pytest.fixture
def temp_store_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir


def test_end_to_end_multi_doc_indexing_and_persistence(temp_store_dir):
    sample_dir = Path("./sample_docs")
    pdf_files = list(sample_dir.glob("*.pdf"))
    assert len(pdf_files) == 3, "Expected 3 sample PDFs generated in sample_docs"

    chunker = TextChunker(chunk_size=600, chunk_overlap=100)
    embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(persist_directory=temp_store_dir, collection_name="research_docs")

    # Ingest all 3 documents
    for pdf_path in pdf_files:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = extract_text_from_pdf(pdf_bytes, pdf_path.name)
        assert doc.is_valid is True
        assert doc.page_count == 2
        assert doc.total_characters > 200

        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2

        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_manager.embed_texts(chunk_texts)
        added = vector_store.add_chunks(chunks, embeddings)
        assert added == len(chunks)

    # Check total indexed count
    indexed_docs = vector_store.get_indexed_documents()
    assert len(indexed_docs) == 3
    filenames = [d["filename"] for d in indexed_docs]
    assert "Sparse_Transformer_NLP.pdf" in filenames
    assert "Vision_Transformers_vs_CNNs.pdf" in filenames
    assert "Federated_Learning_Healthcare_Privacy.pdf" in filenames

    # Test Query 1: Targeted privacy question
    privacy_q = "Which framework discusses HIPAA, GDPR, and differential privacy?"
    q_vec = embedding_manager.embed_query(privacy_q)
    results = vector_store.search(q_vec, top_k=3)
    assert len(results) > 0
    top_hit = results[0]
    assert "Federated_Learning_Healthcare_Privacy.pdf" == top_hit.filename
    assert top_hit.page_number in (1, 2)
    assert "GDPR" in top_hit.text or "HIPAA" in top_hit.text or "Differential Privacy" in top_hit.text

    # Test Query 2: Multi-document retrieval for performance/accuracy
    eval_q = "What empirical accuracy and benchmark results are reported?"
    q_vec2 = embedding_manager.embed_query(eval_q)
    results2 = vector_store.search(q_vec2, top_k=6)
    retrieved_files = {p.filename for p in results2}
    # Multiple distinct files should be retrieved
    assert len(retrieved_files) >= 2

    # Test Citation extraction
    citations = CitationManager.extract_citations(results2)
    assert len(citations) >= 2
    for c in citations:
        assert c.formatted_citation.startswith("[Source: ")
        assert ", Page " in c.formatted_citation
        assert c.similarity_score > 0.0

    # Test Persistence across VectorStore reload
    reloaded_store = VectorStore(persist_directory=temp_store_dir, collection_name="research_docs")
    assert reloaded_store.count() == vector_store.count()
    reloaded_docs = reloaded_store.get_indexed_documents()
    assert len(reloaded_docs) == 3

    # Test Mocked RAG Pipeline Answer Generation
    mock_groq = MagicMock(spec=GroqService)
    mock_groq.api_key = "gsk_valid_mock_key"
    mock_groq.model_name = "llama-3.3-70b-versatile"
    mock_groq.chat_completion.return_value = (
        "HealthFed-DP achieved 0.932 AUC across 5 hospital nodes under HIPAA and GDPR compliance [Source: Federated_Learning_Healthcare_Privacy.pdf, Page 2]."
    )

    pipeline = RAGPipeline(
        vector_store=reloaded_store,
        embedding_manager=embedding_manager,
        groq_service=mock_groq,
    )

    rag_res = pipeline.answer_question("How does HealthFed-DP perform in medical settings?")
    assert rag_res.has_sufficient_context is True
    assert "0.932 AUC" in rag_res.answer
    assert any(c.filename == "Federated_Learning_Healthcare_Privacy.pdf" for c in rag_res.citations)

    # Test Document Removal
    target_del_id = indexed_docs[0]["document_id"]
    reloaded_store.delete_document(target_del_id)
    assert len(reloaded_store.get_indexed_documents()) == 2
    assert reloaded_store.document_exists(target_del_id) is False
