"""
Vector Database module using ChromaDB.
Provides persistent storage, semantic similarity search, metadata filtering,
deduplication, and document lifecycle management (insert, delete, clear).
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from utils.text_chunker import DocumentChunk

logger = logging.getLogger(__name__)


class RetrievedPassage:
    """Represents a retrieved chunk with similarity score and metadata."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        distance: float,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.distance = distance

        # Calculate a cosine similarity score (0.0 to 1.0)
        # ChromaDB default metric is L2 distance or cosine distance depending on space
        # For normalized embeddings with cosine distance: distance is 1 - cos_sim.
        # So similarity = max(0.0, 1.0 - distance)
        self.similarity_score = max(0.0, min(1.0, 1.0 - distance))

    @property
    def filename(self) -> str:
        return self.metadata.get("filename", "Unknown Document")

    @property
    def page_number(self) -> int:
        return int(self.metadata.get("page_number", 1))

    @property
    def document_id(self) -> str:
        return self.metadata.get("document_id", "")

    @property
    def chunk_index(self) -> int:
        return int(self.metadata.get("chunk_index", 0))

    def format_citation(self) -> str:
        """Format citation string: [Source: filename.pdf, Page X]."""
        return f"[Source: {self.filename}, Page {self.page_number}]"


class VectorStore:
    """Wrapper around ChromaDB PersistentClient."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "research_documents",
    ):
        self.persist_directory = Path(persist_directory).resolve()
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """Create storage directory and initialize ChromaDB client."""
        import chromadb
        from chromadb.config import Settings

        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            # Create or get collection with cosine similarity space
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB initialized at '{self.persist_directory}', collection '{self.collection_name}'"
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise RuntimeError(f"Failed to initialize ChromaDB at {self.persist_directory}: {e}")

    def document_exists(self, document_id: str) -> bool:
        """Check whether chunks for this document_id already exist in the vector store."""
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                limit=1,
            )
            return bool(results and results["ids"] and len(results["ids"]) > 0)
        except Exception as e:
            logger.warning(f"Error checking if document exists {document_id}: {e}")
            return False

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Add document chunks and their embeddings to ChromaDB.
        Deduplicates by removing existing document chunks first if re-indexing.
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks list and embeddings list must have matching lengths.")

        # If document already exists, remove previous chunks to prevent duplicates
        doc_id = chunks[0].document_id
        if self.document_exists(doc_id):
            logger.info(f"Document {doc_id} already exists. Removing old chunks before re-indexing.")
            self.delete_document(doc_id)

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.to_chroma_metadata() for c in chunks]

        # Ingest in batches to prevent hitting any memory/payload thresholds
        batch_size = 100
        total_added = 0
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=texts[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )
            total_added += (end_idx - i)

        logger.info(f"Successfully added {total_added} chunks for document {doc_id} to ChromaDB.")
        return total_added

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedPassage]:
        """
        Perform semantic similarity search using precomputed query vector.
        Supports document ID filtering.
        """
        if self.collection.count() == 0:
            return []

        where_clause = None
        if filter_document_ids:
            if len(filter_document_ids) == 1:
                where_clause = {"document_id": filter_document_ids[0]}
            elif len(filter_document_ids) > 1:
                where_clause = {"document_id": {"$in": filter_document_ids}}

        # Bound top_k to total available chunks
        available_count = self.collection.count()
        actual_top_k = min(top_k, available_count)

        if actual_top_k <= 0:
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Search query failed in ChromaDB: {e}")
            return []

        retrieved: List[RetrievedPassage] = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for idx in range(len(ids)):
                chunk_id = ids[idx]
                text = docs[idx] if idx < len(docs) else ""
                metadata = metas[idx] if idx < len(metas) else {}
                dist = distances[idx] if idx < len(distances) else 1.0

                retrieved.append(
                    RetrievedPassage(
                        chunk_id=chunk_id,
                        text=text,
                        metadata=metadata,
                        distance=dist,
                    )
                )

        return retrieved

    def delete_document(self, document_id: str) -> bool:
        """Safely delete all chunks belonging to a specific document_id."""
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted chunks for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def clear_collection(self) -> bool:
        """Clear all indexed documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Cleared collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def count(self) -> int:
        """Return total number of chunks indexed in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Aggregate all documents currently indexed in ChromaDB.
        Returns list of metadata dicts containing:
        - document_id
        - filename
        - chunk_count
        - pages (list of distinct page numbers)
        """
        if self.collection.count() == 0:
            return []

        try:
            all_data = self.collection.get(include=["metadatas"])
            metas = all_data.get("metadatas", [])

            docs_map: Dict[str, Dict[str, Any]] = {}
            for m in metas:
                if not m:
                    continue
                d_id = m.get("document_id")
                if not d_id:
                    continue

                if d_id not in docs_map:
                    docs_map[d_id] = {
                        "document_id": d_id,
                        "filename": m.get("filename", "Unknown"),
                        "chunk_count": 0,
                        "pages": set(),
                        "source_type": m.get("source_type", "pdf"),
                    }
                docs_map[d_id]["chunk_count"] += 1
                page_num = m.get("page_number")
                if page_num:
                    docs_map[d_id]["pages"].add(int(page_num))

            result = []
            for d in docs_map.values():
                d["page_count"] = len(d["pages"])
                d["pages"] = sorted(list(d["pages"]))
                result.append(d)

            return result
        except Exception as e:
            logger.error(f"Failed to fetch indexed documents list: {e}")
            return []
