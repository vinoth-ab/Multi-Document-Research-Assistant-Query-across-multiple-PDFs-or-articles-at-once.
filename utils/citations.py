"""
Citations module for Multi-Document Research Assistant.
Extracts, formats, and deduplicates verified source citations from retrieved chunks.
Ensures citations are strictly grounded in retrieved metadata and never hallucinated.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from utils.vector_store import RetrievedPassage


@dataclass
class Citation:
    """Represents a verified citation linked to an indexed source document page."""
    document_id: str
    filename: str
    page_number: Optional[int]
    chunk_id: str
    snippet: str
    similarity_score: float

    @property
    def formatted_citation(self) -> str:
        """Standard citation string: [Source: filename.pdf, Page X]."""
        if self.page_number is not None and self.page_number > 0:
            return f"[Source: {self.filename}, Page {self.page_number}]"
        return f"[Source: {self.filename}, Page Unavailable]"


class CitationManager:
    """Manages citation extraction, deduplication, and formatting."""

    @staticmethod
    def extract_citations(passages: List[RetrievedPassage]) -> List[Citation]:
        """
        Extract deduplicated citations from retrieved passages.
        Deduplicates by (filename, page_number) while keeping the passage with the highest similarity score.
        """
        if not passages:
            return []

        # Track best citation per (filename, page_number)
        seen_pages: Dict[Tuple[str, Optional[int]], Citation] = {}

        for passage in passages:
            fn = passage.filename
            pg = passage.page_number if passage.page_number > 0 else None
            key = (fn, pg)

            # Create citation candidate
            # Truncate snippet to first ~200 characters for concise display
            clean_snippet = passage.text.replace("\n", " ").strip()
            if len(clean_snippet) > 240:
                clean_snippet = clean_snippet[:240].rsplit(" ", 1)[0] + "..."

            candidate = Citation(
                document_id=passage.document_id,
                filename=fn,
                page_number=pg,
                chunk_id=passage.chunk_id,
                snippet=clean_snippet,
                similarity_score=passage.similarity_score,
            )

            if key not in seen_pages or candidate.similarity_score > seen_pages[key].similarity_score:
                seen_pages[key] = candidate

        # Return list sorted by highest similarity score
        citations = list(seen_pages.values())
        citations.sort(key=lambda c: c.similarity_score, reverse=True)
        return citations

    @staticmethod
    def format_citations_summary(citations: List[Citation]) -> str:
        """Format a list of citations as a clean markdown block."""
        if not citations:
            return ""

        lines = ["\n\n### 📚 Sources & Citations:"]
        for idx, cit in enumerate(citations, 1):
            score_percent = int(cit.similarity_score * 100)
            lines.append(
                f"{idx}. **{cit.formatted_citation}** (Relevance: `{score_percent}%`)\n"
                f"   > *\"{cit.snippet}\"*"
            )
        return "\n".join(lines)

    @staticmethod
    def format_inline_sources_tags(citations: List[Citation]) -> str:
        """Format concise inline citation pills for quick reference."""
        if not citations:
            return ""
        unique_tags = [c.formatted_citation for c in citations]
        return " ".join(f"`{t}`" for t in unique_tags)
