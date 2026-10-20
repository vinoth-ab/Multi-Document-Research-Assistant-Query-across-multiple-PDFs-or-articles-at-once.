"""
Unit and Integration tests for RAG Pipeline, VectorStore, and GroqService.
Uses mock Groq API to test offline without consuming API credits.
"""

import tempfile
import pytest
from unittest.mock import MagicMock, patch

from utils.embeddings import EmbeddingManager
from utils.vector_store import VectorStore
from utils.text_chunker import DocumentChunk
from utils.groq_client import GroqService, GroqClientError
from utils.rag_pipeline import RAGPipeline, SYSTEM_PROMPT


@pytest.fixture
def temp_chroma_dir():
    # On Windows, SQLite files remain briefly locked by ChromaDB client; ignore_cleanup_errors prevents WinError 32
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager returning fast deterministic 384-dimensional vectors."""
    manager = MagicMock(spec=EmbeddingManager)
    manager.get_embedding_dimension.return_value = 384

    def fake_embed_texts(texts, batch_size=32):
        # Generate predictable vector based on text length
        return [[(len(t) % 10) * 0.1] * 384 for t in texts]

    def fake_embed_query(query):
        return [(len(query) % 10) * 0.1] * 384

    manager.embed_texts.side_effect = fake_embed_texts
    manager.embed_query.side_effect = fake_embed_query
    return manager


@pytest.fixture
def mock_groq_service():
    """Mock GroqService for offline testing."""
    service = MagicMock(spec=GroqService)
    service.api_key = "gsk_test_mock_key"
    service.model_name = "llama-3.3-70b-versatile"
    service.test_connection.return_value = (True, "Connected to Groq successfully.")
    service.chat_completion.return_value = (
        "According to the research papers, deep learning achieves 94% accuracy [Source: doc1.pdf, Page 1]."
    )
    return service


def test_vector_store_crud_and_deduplication(temp_chroma_dir, mock_embedding_manager):
    store = VectorStore(persist_directory=temp_chroma_dir, collection_name="test_col")
    assert store.count() == 0

    chunks_doc1 = [
        DocumentChunk(
            chunk_id="d1_p1_c0",
            document_id="doc1",
            filename="doc1.pdf",
            page_number=1,
            chunk_index=0,
            text="First chunk of doc1 covering transformer architecture.",
            char_count=52,
        ),
        DocumentChunk(
            chunk_id="d1_p2_c1",
            document_id="doc1",
            filename="doc1.pdf",
            page_number=2,
            chunk_index=1,
            text="Second chunk of doc1 discussing self-attention.",
            char_count=48,
        ),
    ]

    embeddings_doc1 = mock_embedding_manager.embed_texts([c.text for c in chunks_doc1])
    added = store.add_chunks(chunks_doc1, embeddings_doc1)
    assert added == 2
    assert store.count() == 2
    assert store.document_exists("doc1") is True

    # Duplicate ingestion prevention: Re-adding doc1 should not double the count
    added_again = store.add_chunks(chunks_doc1, embeddings_doc1)
    assert added_again == 2
    assert store.count() == 2

    # Add second document
    chunks_doc2 = [
        DocumentChunk(
            chunk_id="d2_p1_c0",
            document_id="doc2",
            filename="doc2.pdf",
            page_number=1,
            chunk_index=0,
            text="Doc2 discusses reinforcement learning with human feedback.",
            char_count=58,
        )
    ]
    embeddings_doc2 = mock_embedding_manager.embed_texts([c.text for c in chunks_doc2])
    store.add_chunks(chunks_doc2, embeddings_doc2)
    assert store.count() == 3

    # Check document listing
    docs = store.get_indexed_documents()
    assert len(docs) == 2
    filenames = [d["filename"] for d in docs]
    assert "doc1.pdf" in filenames
    assert "doc2.pdf" in filenames

    # Delete doc1 only, doc2 should remain intact
    store.delete_document("doc1")
    assert store.document_exists("doc1") is False
    assert store.document_exists("doc2") is True
    assert store.count() == 1

    # Clear entire collection
    store.clear_collection()
    assert store.count() == 0


def test_metadata_filtering(temp_chroma_dir, mock_embedding_manager):
    store = VectorStore(persist_directory=temp_chroma_dir, collection_name="test_filter")

    c1 = DocumentChunk(
        chunk_id="a1", document_id="paperA", filename="paperA.pdf", page_number=1,
        chunk_index=0, text="Topic A content.", char_count=16
    )
    c2 = DocumentChunk(
        chunk_id="b1", document_id="paperB", filename="paperB.pdf", page_number=1,
        chunk_index=0, text="Topic B content.", char_count=16
    )

    store.add_chunks([c1, c2], mock_embedding_manager.embed_texts([c1.text, c2.text]))

    query_vec = mock_embedding_manager.embed_query("Topic A")

    # Search with filter for paperA only
    results_a = store.search(query_vec, top_k=5, filter_document_ids=["paperA"])
    assert len(results_a) == 1
    assert results_a[0].filename == "paperA.pdf"

    # Search with filter for paperB only
    results_b = store.search(query_vec, top_k=5, filter_document_ids=["paperB"])
    assert len(results_b) == 1
    assert results_b[0].filename == "paperB.pdf"


def test_rag_pipeline_answer_flow(temp_chroma_dir, mock_embedding_manager, mock_groq_service):
    store = VectorStore(persist_directory=temp_chroma_dir, collection_name="test_pipeline")
    chunk = DocumentChunk(
        chunk_id="c1", document_id="doc1", filename="doc1.pdf", page_number=1,
        chunk_index=0, text="Deep learning model achieves 94% accuracy.", char_count=43
    )
    store.add_chunks([chunk], mock_embedding_manager.embed_texts([chunk.text]))

    pipeline = RAGPipeline(
        vector_store=store,
        embedding_manager=mock_embedding_manager,
        groq_service=mock_groq_service,
    )

    response = pipeline.answer_question("What accuracy was achieved?", top_k=2)

    assert response.has_sufficient_context is True
    assert "94% accuracy" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].filename == "doc1.pdf"
    assert response.citations[0].page_number == 1


def test_rag_insufficient_context_empty_db(temp_chroma_dir, mock_embedding_manager, mock_groq_service):
    empty_store = VectorStore(persist_directory=temp_chroma_dir, collection_name="test_empty")
    pipeline = RAGPipeline(
        vector_store=empty_store,
        embedding_manager=mock_embedding_manager,
        groq_service=mock_groq_service,
    )

    response = pipeline.answer_question("Any query?")
    assert response.has_sufficient_context is False
    assert "No relevant content found" in response.answer


def test_rag_pipeline_graceful_groq_error(temp_chroma_dir, mock_embedding_manager):
    store = VectorStore(persist_directory=temp_chroma_dir, collection_name="test_err")
    chunk = DocumentChunk(
        chunk_id="c1", document_id="d1", filename="paper.pdf", page_number=1,
        chunk_index=0, text="Sample text content.", char_count=21
    )
    store.add_chunks([chunk], mock_embedding_manager.embed_texts([chunk.text]))

    failing_groq = MagicMock(spec=GroqService)
    failing_groq.chat_completion.side_effect = GroqClientError("Rate limit exceeded")

    pipeline = RAGPipeline(
        vector_store=store,
        embedding_manager=mock_embedding_manager,
        groq_service=failing_groq,
    )

    response = pipeline.answer_question("Query?")
    assert response.has_sufficient_context is False
    assert "Rate limit exceeded" in response.answer
    assert response.error_message is not None


def test_real_embedding_manager():
    """Test actual SentenceTransformers all-MiniLM-L6-v2 model generation and dimensions."""
    real_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    assert real_manager.get_embedding_dimension() == 384

    texts = [
        "Machine learning models require robust evaluation metrics.",
        "Natural language processing uses tokenization and embeddings.",
    ]
    embeddings = real_manager.embed_texts(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    q_vec = real_manager.embed_query("What evaluation metrics are used?")
    assert len(q_vec) == 384

