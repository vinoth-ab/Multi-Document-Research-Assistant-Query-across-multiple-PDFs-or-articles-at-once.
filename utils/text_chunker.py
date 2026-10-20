"""
Text Chunking module for Multi-Document Research Assistant.
Splits extracted document text into overlapping, semantically coherent chunks
while preserving critical provenance metadata (document ID, filename, page number).
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from utils.pdf_processor import ProcessedDocument, ExtractedPage


class DocumentChunk(BaseModel):
    """Represents an indexed chunk with rich provenance metadata."""
    chunk_id: str = Field(description="Globally unique chunk identifier")
    document_id: str = Field(description="Parent document identifier")
    filename: str = Field(description="Original document filename")
    page_number: int = Field(description="1-indexed source page number")
    chunk_index: int = Field(description="0-indexed sequence of this chunk within the document")
    source_type: str = Field(default="pdf", description="Source format")
    text: str = Field(description="Extracted chunk text content")
    char_count: int = Field(description="Length in characters")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Format metadata dictionary for ChromaDB (requires primitive types: str, int, float, bool)."""
        meta = {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "filename": str(self.filename),
            "page_number": int(self.page_number),
            "chunk_index": int(self.chunk_index),
            "source_type": str(self.source_type),
            "char_count": int(self.char_count),
        }
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)
        return meta


class TextChunker:
    """
    Intelligent text chunker supporting paragraph/sentence/word boundary awareness
    with configurable chunk size and overlap.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences or meaningful clauses while retaining delimiters."""
        # Split on sentence terminals followed by whitespace or line break
        pattern = r"(?<=[.!?])\s+"
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    def split_text(self, text: str) -> List[str]:
        """
        Split a block of text into chunks of roughly `chunk_size` characters,
        overlapping by `chunk_overlap` characters.
        Prefers splitting on paragraph breaks, then sentence breaks, then spaces.
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # Break text into paragraphs
        paragraphs = text.split("\n\n")
        units: List[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.chunk_size:
                units.append(para)
            else:
                # If paragraph exceeds chunk size, split by sentences
                sentences = self._split_into_sentences(para)
                for sent in sentences:
                    if len(sent) <= self.chunk_size:
                        units.append(sent)
                    else:
                        # If a single sentence exceeds chunk size, split by words
                        words = sent.split(" ")
                        current_word_chunk: List[str] = []
                        current_len = 0
                        for w in words:
                            if current_len + len(w) + 1 > self.chunk_size and current_word_chunk:
                                units.append(" ".join(current_word_chunk))
                                current_word_chunk = [w]
                                current_len = len(w)
                            else:
                                current_word_chunk.append(w)
                                current_len += len(w) + 1
                        if current_word_chunk:
                            units.append(" ".join(current_word_chunk))

        # Now group units into overlapping chunks
        chunks: List[str] = []
        current_chunk_units: List[str] = []
        current_length = 0

        for unit in units:
            unit_len = len(unit)
            # Check if adding this unit exceeds chunk_size
            if current_length + unit_len + 2 > self.chunk_size and current_chunk_units:
                combined_chunk = "\n\n".join(current_chunk_units).strip()
                if combined_chunk:
                    chunks.append(combined_chunk)

                # Keep overlapping units from the end
                overlap_units: List[str] = []
                overlap_len = 0
                for prev_unit in reversed(current_chunk_units):
                    if overlap_len + len(prev_unit) <= self.chunk_overlap or not overlap_units:
                        overlap_units.insert(0, prev_unit)
                        overlap_len += len(prev_unit)
                    else:
                        break

                current_chunk_units = overlap_units + [unit]
                current_length = sum(len(u) for u in current_chunk_units) + (len(current_chunk_units) - 1) * 2
            else:
                current_chunk_units.append(unit)
                current_length += unit_len + 2

        if current_chunk_units:
            final_chunk = "\n\n".join(current_chunk_units).strip()
            if final_chunk and (not chunks or final_chunk != chunks[-1]):
                chunks.append(final_chunk)

        # Final filtering of empty or near-empty chunks
        return [c for c in chunks if len(c.strip()) > 20]

    def chunk_document(self, doc: ProcessedDocument) -> List[DocumentChunk]:
        """
        Split an entire ProcessedDocument into DocumentChunks,
        page by page, ensuring each chunk retains its exact source page number.
        """
        all_chunks: List[DocumentChunk] = []
        chunk_counter = 0

        for page in doc.pages:
            if not page.has_text or not page.text.strip():
                continue

            page_text_chunks = self.split_text(page.text)
            for idx_on_page, text_segment in enumerate(page_text_chunks):
                chunk_id = f"{doc.document_id}_p{page.page_number}_c{chunk_counter}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    filename=doc.filename,
                    page_number=page.page_number,
                    chunk_index=chunk_counter,
                    source_type="pdf",
                    text=text_segment,
                    char_count=len(text_segment),
                    metadata={
                        "is_scanned": page.is_scanned,
                        "ocr_applied": page.ocr_applied,
                        "page_chunk_index": idx_on_page,
                    },
                )
                all_chunks.append(chunk)
                chunk_counter += 1

        return all_chunks
