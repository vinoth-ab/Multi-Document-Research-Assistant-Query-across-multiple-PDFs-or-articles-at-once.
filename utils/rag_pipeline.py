"""
RAG Pipeline module for Multi-Document Research Assistant.
Orchestrates query embedding, semantic retrieval, prompt construction,
Groq inference, citation extraction, summarization, and cross-document comparison.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from utils.embeddings import EmbeddingManager
from utils.vector_store import VectorStore, RetrievedPassage
from utils.groq_client import GroqService, GroqClientError
from utils.citations import CitationManager, Citation

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI academic research assistant analyzing scientific papers and research documents.
Your goal is to provide rigorous, accurate, and evidence-grounded answers based strictly on the provided document passages.

CRITICAL INSTRUCTIONS:
1. Ground your answer ENTIRELY in the provided passages below.
2. If the context does not contain sufficient evidence to answer the question accurately, explicitly state:
   "Based on the provided documents, there is insufficient evidence to answer this question."
   Do NOT attempt to guess, extrapolate, or fabricate information.
3. Every factual statement or comparison should reference its source document and page number where available, using the format [Source: filename.pdf, Page X].
4. When comparing multiple documents, highlight points of agreement, divergence, distinct methodologies, and respective limitations.
5. Do NOT fabricate citations, statistical numbers, authors, or publication dates.
6. Maintain an objective, academic, and structured tone. Use markdown headings and bullet points where helpful.
"""


@dataclass
class RAGResponse:
    """Holds the complete result of a RAG query execution."""
    answer: str
    passages: List[RetrievedPassage] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    has_sufficient_context: bool = True
    error_message: Optional[str] = None
    query: str = ""
    target_documents: Optional[List[str]] = None


class RAGPipeline:
    """End-to-end RAG orchestrator for multi-document research."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        groq_service: GroqService,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.groq_service = groq_service

    def build_context_string(self, passages: List[RetrievedPassage]) -> str:
        """Construct structured context block from retrieved passages."""
        if not passages:
            return "No relevant passages were retrieved from the document database."

        context_blocks = []
        for i, p in enumerate(passages, 1):
            source_tag = f"[Passage {i} | Source: {p.filename}, Page {p.page_number}]"
            context_blocks.append(f"{source_tag}\n{p.text}\n")

        return "\n---\n".join(context_blocks)

    def answer_question(
        self,
        query: str,
        top_k: int = 5,
        filter_document_ids: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> RAGResponse:
        """
        Execute standard RAG question answering:
        1. Encode question
        2. Retrieve top-k passages from ChromaDB
        3. Format context
        4. Query Groq LLM
        5. Extract and link verified citations
        """
        if not query or not query.strip():
            return RAGResponse(
                answer="Please provide a non-empty question.",
                has_sufficient_context=False,
                query=query,
            )

        # Step 1 & 2: Embedding and Retrieval
        try:
            query_vector = self.embedding_manager.embed_query(query)
            passages = self.vector_store.search(
                query_embedding=query_vector,
                top_k=top_k,
                filter_document_ids=filter_document_ids,
            )
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return RAGResponse(
                answer="Failed to retrieve information from the document store.",
                error_message=str(e),
                has_sufficient_context=False,
                query=query,
            )

        # Check if any passages were retrieved
        if not passages:
            return RAGResponse(
                answer="No relevant content found in the indexed documents for this query. "
                       "Try rephrasing your question or uploading additional research papers.",
                passages=[],
                citations=[],
                has_sufficient_context=False,
                query=query,
            )

        # Step 3: Extract verified citations from metadata
        citations = CitationManager.extract_citations(passages)

        # Step 4: Build prompt messages
        context_str = self.build_context_string(passages)
        user_prompt = f"DOCUMENT EXCERPTS:\n{context_str}\n\nRESEARCH QUESTION:\n{query}\n\nANSWER:"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Append recent conversation history if provided (up to last 4 turns for context)
        if chat_history:
            recent_turns = chat_history[-4:]
            for turn in recent_turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_prompt})

        # Step 5: Groq Inference
        try:
            answer = self.groq_service.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
            )
            has_context = "insufficient evidence" not in answer.lower()

            return RAGResponse(
                answer=answer,
                passages=passages,
                citations=citations,
                has_sufficient_context=has_context,
                query=query,
                target_documents=filter_document_ids,
            )
        except GroqClientError as ge:
            return RAGResponse(
                answer=f"⚠️ {str(ge)}",
                passages=passages,
                citations=citations,
                has_sufficient_context=False,
                error_message=str(ge),
                query=query,
            )
        except Exception as e:
            return RAGResponse(
                answer=f"⚠️ An unexpected error occurred while generating the answer: {str(e)}",
                passages=passages,
                citations=citations,
                has_sufficient_context=False,
                error_message=str(e),
                query=query,
            )

    def summarize_document(
        self,
        document_id: str,
        filename: str,
    ) -> RAGResponse:
        """
        Generate a comprehensive, structured research summary of a specific document.
        Uses representative passages covering the document from beginning, middle, and conclusion.
        """
        # Retrieve chunks for this document
        try:
            all_chunks = self.vector_store.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            return RAGResponse(
                answer=f"Failed to access document data: {e}",
                error_message=str(e),
                has_sufficient_context=False,
            )

        docs = all_chunks.get("documents", [])
        metas = all_chunks.get("metadatas", [])

        if not docs:
            return RAGResponse(
                answer=f"No indexed content found for document '{filename}'.",
                has_sufficient_context=False,
            )

        # Select representative chunks across the document (up to 8 chunks)
        # to prevent prompt overflow while capturing Abstract, Methods, Results, Conclusion
        total_chunks = len(docs)
        if total_chunks <= 8:
            selected_indices = list(range(total_chunks))
        else:
            step = total_chunks / 8.0
            selected_indices = [min(int(i * step), total_chunks - 1) for i in range(8)]
            selected_indices = sorted(list(set(selected_indices)))

        representative_passages: List[RetrievedPassage] = []
        for idx in selected_indices:
            text = docs[idx]
            meta = metas[idx] if idx < len(metas) else {}
            representative_passages.append(
                RetrievedPassage(
                    chunk_id=meta.get("chunk_id", f"{document_id}_{idx}"),
                    text=text,
                    metadata=meta,
                    distance=0.0,
                )
            )

        citations = CitationManager.extract_citations(representative_passages)
        context_str = self.build_context_string(representative_passages)

        summary_prompt = (
            f"You are an academic researcher. Analyze the following excerpts from '{filename}' "
            f"and synthesize a structured academic executive summary with the following sections:\n"
            f"1. **Core Objective & Research Problem**\n"
            f"2. **Methodology & Key Concepts**\n"
            f"3. **Major Findings & Empirical Results**\n"
            f"4. **Limitations & Future Directions**\n\n"
            f"Document Excerpts:\n{context_str}\n\n"
            f"Ground your summary strictly in these excerpts. Include page citations where appropriate."
        )

        try:
            answer = self.groq_service.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.15,
                max_tokens=1800,
            )
            return RAGResponse(
                answer=answer,
                passages=representative_passages,
                citations=citations,
                has_sufficient_context=True,
                query=f"Summarize document: {filename}",
            )
        except GroqClientError as ge:
            return RAGResponse(
                answer=f"⚠️ {str(ge)}",
                passages=representative_passages,
                citations=citations,
                has_sufficient_context=False,
                error_message=str(ge),
            )

    def compare_documents(
        self,
        document_ids: List[str],
        comparison_aspect: str = "findings, methodologies, limitations, and conclusions",
    ) -> RAGResponse:
        """
        Cross-Document Comparison: Compares multiple selected documents across key academic dimensions.
        """
        if len(document_ids) < 2:
            return RAGResponse(
                answer="Please select at least two documents for cross-document comparison.",
                has_sufficient_context=False,
            )

        combined_passages: List[RetrievedPassage] = []

        # Retrieve top passages for each document regarding the comparison aspect
        for doc_id in document_ids:
            try:
                aspect_vector = self.embedding_manager.embed_query(
                    f"Research methodology, key findings, dataset, evaluation metrics, and limitations: {comparison_aspect}"
                )
                doc_passages = self.vector_store.search(
                    query_embedding=aspect_vector,
                    top_k=4,
                    filter_document_ids=[doc_id],
                )
                combined_passages.extend(doc_passages)
            except Exception as e:
                logger.error(f"Error retrieving comparison chunks for {doc_id}: {e}")

        if not combined_passages:
            return RAGResponse(
                answer="Could not retrieve sufficient passages across the selected documents for comparison.",
                has_sufficient_context=False,
            )

        citations = CitationManager.extract_citations(combined_passages)
        context_str = self.build_context_string(combined_passages)

        comparison_prompt = (
            f"Perform a comprehensive academic cross-document comparative analysis based on the excerpts below.\n\n"
            f"Compare the documents regarding: **{comparison_aspect}**.\n\n"
            f"Structure your response with:\n"
            f"1. **Comparative Overview Table or Summary** (Methods, Datasets/Approach, Key Findings)\n"
            f"2. **Detailed Methodological Comparison** (How do the approaches differ?)\n"
            f"3. **Results & Performance Contrast** (What findings are reported in each?)\n"
            f"4. **Shared Challenges & Divergent Limitations**\n"
            f"5. **Synthesis & Takeaways**\n\n"
            f"Include exact citations for every claim [Source: filename, Page X].\n\n"
            f"Document Excerpts:\n{context_str}\n\n"
            f"COMPARATIVE ANALYSIS:"
        )

        try:
            answer = self.groq_service.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": comparison_prompt},
                ],
                temperature=0.15,
                max_tokens=2000,
            )
            return RAGResponse(
                answer=answer,
                passages=combined_passages,
                citations=citations,
                has_sufficient_context=True,
                query=f"Compare documents regarding {comparison_aspect}",
                target_documents=document_ids,
            )
        except GroqClientError as ge:
            return RAGResponse(
                answer=f"⚠️ {str(ge)}",
                passages=combined_passages,
                citations=citations,
                has_sufficient_context=False,
                error_message=str(ge),
            )
