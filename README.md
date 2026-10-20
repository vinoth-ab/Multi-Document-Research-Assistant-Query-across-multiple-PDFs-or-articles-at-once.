# Multi-Document Research Assistant Using RAG

An AI-powered academic research assistant built with **Retrieval-Augmented Generation (RAG)**. This application enables researchers and students to upload multiple scientific papers or PDF reports simultaneously, conduct natural-language queries across all documents, synthesize cross-document comparisons, generate executive summaries, and inspect evidence with verified page-level citations.

---

## 🌟 Key Features

- **Multi-Document PDF Ingestion:** Upload multiple PDFs simultaneously with automatic size and type validation (up to 20 MB each).
- **Intelligent Page-by-Page Extraction:** Uses **PyMuPDF** (`fitz`) for fast text extraction while preserving paragraph structures, page numbers, and document boundaries.
- **Scanned PDF Detection & OCR Fallback:** Detects image-only/scanned PDFs and provides graceful notifications and optional Tesseract OCR fallback.
- **Provenance-Preserving Text Chunking:** Configurable chunking (default ~1,000 characters with 150-character overlap) maintaining exact metadata (`document_id`, `filename`, `page_number`, `chunk_index`).
- **Dense Vector Embeddings:** Encodes text and queries using HuggingFace's **Sentence Transformers** (`all-MiniLM-L6-v2`, 384 dimensions) with local caching.
- **Persistent Vector Store (ChromaDB):** Persists embeddings and chunk metadata to `./data/chroma_db`, preventing duplicate ingestion and surviving application restarts.
- **Strictly Grounded LLM Reasoning (Groq API):** Integrates the official **Groq SDK** with low temperature (0.1) and system prompts that mandate answers based strictly on retrieved passages. Supports modern models such as `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`.
- **Verified Source Citations:** Citations are generated from real retrieved chunk metadata, formatted as `[Source: filename.pdf, Page X]`, deduplicated per page, and accompanied by expandable source passage accordions.
- **Academic Research Tools:**
  - 💬 **Research Chat & Q&A:** Multi-turn conversation with follow-up support and suggested research questions.
  - ⚖️ **Cross-Document Comparison:** Side-by-side comparative analysis of methodologies, empirical results, datasets, and limitations across papers.
  - 📄 **Document Summarizer:** Synthesizes structured executive summaries (objectives, methods, results, limitations) without truncation.
  - 🔍 **Source Explorer:** Search and inspect indexed chunks and raw metadata stored in ChromaDB.

---

## 🏗️ Architecture & RAG Pipeline

```
                                      [ Upload PDFs ]
                                             │
                                             ▼
                             [ PyMuPDF Text Extraction ]
                             (Page Tracking & Cleaning)
                                             │
                                             ▼
                             [ Semantic Text Chunker ]
                            (1000 Chars / 150 Overlap)
                                             │
                                             ▼
                         [ SentenceTransformers (all-MiniLM) ]
                                             │
                                             ▼
                                  [ ChromaDB Persistence ]
                                    (./data/chroma_db)
                                             │
─────────────────────────────────────────────┼─────────────────────────────────────────────
[ User Research Query ]                      │
          │                                  │
          ▼                                  ▼
[ Query Embedding ] ──────────► [ Semantic Vector Search ] (Top-k Chunks)
                                             │
                                             ▼
                                [ Context Assembly & Citations ]
                                             │
                                             ▼
                               [ Groq API (Llama 3.3 70B) ]
                                             │
                                             ▼
                      [ Grounded Response + Verified Citations ]
```

---

## 💻 Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Core runtime |
| **Frontend** | Streamlit | Modern academic dashboard UI |
| **LLM Inference** | Groq Python SDK | Ultra-fast Llama 3.3 / Llama 3.1 inference |
| **Embeddings** | Sentence Transformers | Local `all-MiniLM-L6-v2` dense vectors |
| **Vector Database** | ChromaDB | Local persistent vector storage with cosine similarity |
| **PDF Processing** | PyMuPDF (`pymupdf`) | Robust page-by-page text parsing |
| **Configuration** | `python-dotenv` & Pydantic | Environment validation and settings management |
| **Testing** | `pytest` | Comprehensive unit and integration test suite |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.11 or compatible installed.
- Windows, macOS, or Linux.
- A free **Groq API Key** from [https://console.groq.com/keys](https://console.groq.com/keys).

---

### 2. Environment Setup

#### Option A: Windows PowerShell
```powershell
# Navigate to the project directory
cd "c:\Users\vinot\Downloads\RAG project1"

# Create a virtual environment (optional but recommended)
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

#### Option B: Windows Command Prompt (cmd)
```cmd
cd "c:\Users\vinot\Downloads\RAG project1"
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

---

### 3. Configure Environment Variables

Create a `.env` file in the root directory by copying `.env.example`:

```powershell
cp .env.example .env
```

Edit `.env` with your settings:
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=research_documents
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
DEFAULT_TOP_K=5
MAX_UPLOAD_SIZE_MB=20
MAX_FILES_PER_UPLOAD=10
```

> **Note:** If you do not wish to set a `.env` file, you can also paste your Groq API Key directly into the application's sidebar interface. The key is never logged or exposed.

---

### 4. Generate Sample Academic Papers (Optional)

To test the application immediately with realistic multi-page scientific papers, run:

```powershell
python generate_sample_papers.py
```

This creates 3 academic papers in `./sample_docs/`:
1. `Sparse_Transformer_NLP.pdf` — Linear-complexity self-attention, GLUE benchmarks.
2. `Vision_Transformers_vs_CNNs.pdf` — ResNet-50 vs. ViT-B/16 on ImageNet-1K and Jetson edge FPS.
3. `Federated_Learning_Healthcare_Privacy.pdf` — Differential privacy (RDP), Paillier encryption, HIPAA & GDPR compliance in radiology diagnostics.

---

### 5. Launch the Streamlit Web Application

```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

The repository includes a comprehensive `pytest` suite covering PDF parsing, text chunking, citation deduplication, vector store persistence, and mock RAG pipeline execution:

```powershell
python -m pytest tests/ -v
```

All 19 tests run offline and pass without consuming external Groq API credits.

---

## 📖 Usage Instructions

1. **Upload Documents:** Use the sidebar file uploader to select one or multiple PDF files. Progress bars indicate extraction, chunking, and vector indexing.
2. **Review Metrics:** The dashboard shows total indexed documents, pages processed, vector chunks, and session queries.
3. **Research Chat (Tab 1):** Ask natural-language questions across the documents. Suggested questions can be clicked directly.
4. **Inspect Citations & Passages:** Every generated answer displays verified citation tags (`[Source: paper.pdf, Page X]`). Click **Inspect Retrieved Passages** to review exact source text and similarity scores.
5. **Cross-Document Comparison (Tab 2):** Select 2 or more indexed papers and choose a dimension (e.g., Methodology, Datasets, Limitations) to generate a structured comparative analysis.
6. **Executive Summarizer (Tab 3):** Select any paper to generate an academic summary covering problem statements, methods, empirical findings, and limitations.
7. **Source Explorer (Tab 4):** Search or filter raw chunks stored in ChromaDB by keyword or document name.
8. **Document Management:** Remove individual papers using the 🗑️ button in the sidebar or use **Clear All** (with confirmation) to reset the vector database.

---

## 🔒 Security and Error Handling

- **Untrusted Document Safety:** PDF text is treated as untrusted data. Formatting instructions or prompt injection attempts inside PDFs cannot alter the system prompt.
- **API Key Protection:** The Groq API key is masked in the UI and never logged or serialized into ChromaDB or Git.
- **Corrupted & Scanned PDFs:** Gracefully handled with clear user-facing alerts; the application will not crash on corrupted, password-protected, or unreadable files.
- **Quota & Network Resilience:** Automatically detects Groq rate limits (`429`), invalid keys (`401`), and connection timeouts, providing actionable suggestions.

---

## 📁 Project Structure

```
multi_document_research_assistant/
│
├── app.py                         # Main Streamlit web application & UI
├── config.py                      # Pydantic configuration & environment loader
├── requirements.txt               # Complete Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules (secrets, venv, ChromaDB)
├── README.md                      # Comprehensive documentation
├── generate_sample_papers.py      # Script to create realistic sample papers
│
├── utils/
│   ├── __init__.py                # Package initialization
│   ├── pdf_processor.py           # PyMuPDF extraction & whitespace normalization
│   ├── text_chunker.py            # Sentence & paragraph boundary-aware chunking
│   ├── embeddings.py              # SentenceTransformers (all-MiniLM-L6-v2) manager
│   ├── vector_store.py            # Persistent ChromaDB client, search & CRUD
│   ├── groq_client.py             # Official Groq SDK wrapper with error recovery
│   ├── citations.py               # Citation extraction, deduplication & formatting
│   └── rag_pipeline.py            # Orchestrator for RAG, comparison, & summary
│
├── tests/
│   ├── test_pdf_processor.py      # PDF validation, extraction, corrupted PDF tests
│   ├── test_text_chunker.py       # Boundary splitting & page provenance tests
│   ├── test_citations.py          # Citation deduplication & formatting tests
│   ├── test_rag_pipeline.py       # VectorStore CRUD, metadata filtering & RAG tests
│   └── test_end_to_end_sample_docs.py # Real multi-doc integration & persistence tests
│
├── sample_docs/                   # Generated realistic academic PDF fixtures
└── data/
    └── .gitkeep                   # ChromaDB local storage directory
```

---

## ⚠️ Limitations & Future Work

1. **Scanned PDFs:** Requires `tesseract` binary installed in system PATH for OCR fallback on image-only PDFs.
2. **Tables and Equations:** Complex multi-column tables and mathematical LaTeX formulas are extracted as raw text by PyMuPDF; future versions could integrate specialized layout models.
3. **Multi-User Isolation:** Current deployment is tailored for single-session research or local research labs; multi-tenant production deployments can namespace ChromaDB collections by authenticated user ID.
