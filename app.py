"""
Multi-Document Research Assistant Using RAG.
A Streamlit web application for querying, comparing, and summarizing multiple research papers
with ChromaDB vector persistence, Sentence Transformers embeddings, and Groq LLM inference.
"""

import io
import os
import streamlit as st
from typing import List, Dict, Any, Optional

from config import config
from utils.pdf_processor import (
    validate_pdf_upload,
    extract_text_from_pdf,
    ProcessedDocument,
)
from utils.text_chunker import TextChunker, DocumentChunk
from utils.embeddings import EmbeddingManager
from utils.vector_store import VectorStore, RetrievedPassage
from utils.groq_client import GroqService, GroqClientError
from utils.citations import CitationManager, Citation
from utils.rag_pipeline import RAGPipeline, RAGResponse

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Multi-Document Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# DARK GLASSMORPHISM THEME — Custom CSS
# =====================================================================
st.markdown(
    """
    <style>
    /* ——— Google Font Import ——— */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ——— Root Variables ——— */
    :root {
        --bg-primary: #0B0F19;
        --bg-secondary: #111827;
        --glass-bg: rgba(17, 24, 39, 0.65);
        --glass-border: rgba(99, 102, 241, 0.18);
        --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        --glass-blur: blur(16px);
        --accent-primary: #818CF8;
        --accent-secondary: #6366F1;
        --accent-glow: rgba(129, 140, 248, 0.35);
        --accent-cyan: #22D3EE;
        --accent-emerald: #34D399;
        --accent-amber: #FBBF24;
        --accent-rose: #FB7185;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --border-subtle: rgba(148, 163, 184, 0.08);
        --gradient-hero: linear-gradient(135deg, #6366F1 0%, #8B5CF6 40%, #EC4899 100%);
        --gradient-card: linear-gradient(145deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.04) 100%);
    }

    /* ——— Global App Background ——— */
    .stApp, .main .block-container, [data-testid="stAppViewContainer"] {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ——— Sidebar Glassmorphism ——— */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.92) 0%, rgba(11, 15, 25, 0.97) 100%) !important;
        backdrop-filter: var(--glass-blur) !important;
        -webkit-backdrop-filter: var(--glass-blur) !important;
        border-right: 1px solid var(--glass-border) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        background: var(--gradient-hero);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
    }

    /* ——— Scrollbar ——— */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--accent-secondary); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

    /* ——— Hero Title ——— */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: var(--gradient-hero);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 0.25rem;
        animation: fadeSlideUp 0.6s ease-out;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 0.01em;
        animation: fadeSlideUp 0.8s ease-out;
    }

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 8px var(--accent-glow); }
        50% { box-shadow: 0 0 20px var(--accent-glow), 0 0 40px rgba(99, 102, 241, 0.12); }
    }

    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    /* ——— Glass Metric Cards ——— */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
        animation: fadeSlideUp 0.9s ease-out;
    }
    .glass-metric {
        background: var(--glass-bg);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: var(--glass-shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        border-radius: 16px 16px 0 0;
    }
    .glass-metric:nth-child(1)::before { background: linear-gradient(90deg, #6366F1, #818CF8); }
    .glass-metric:nth-child(2)::before { background: linear-gradient(90deg, #8B5CF6, #A78BFA); }
    .glass-metric:nth-child(3)::before { background: linear-gradient(90deg, #22D3EE, #67E8F9); }
    .glass-metric:nth-child(4)::before { background: linear-gradient(90deg, #34D399, #6EE7B7); }

    .glass-metric:hover {
        transform: translateY(-3px);
        box-shadow: var(--glass-shadow), 0 0 24px var(--accent-glow);
        border-color: rgba(99, 102, 241, 0.35);
    }
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.35rem;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .metric-lbl {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    /* ——— Glass Card (generic) ——— */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--glass-shadow);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* ——— Citation Tags ——— */
    .citation-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(99, 102, 241, 0.12);
        color: var(--accent-primary);
        border: 1px solid rgba(99, 102, 241, 0.25);
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-family: 'Inter', monospace;
        font-weight: 500;
        margin: 0.2rem 0.25rem 0.2rem 0;
        transition: all 0.2s ease;
    }
    .citation-tag:hover {
        background: rgba(99, 102, 241, 0.22);
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.15);
    }

    /* ——— Passage Cards ——— */
    .passage-card {
        background: rgba(17, 24, 39, 0.5);
        border-left: 3px solid var(--accent-secondary);
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.85rem;
        color: var(--text-secondary);
        font-size: 0.88rem;
        line-height: 1.6;
        transition: all 0.25s ease;
    }
    .passage-card:hover {
        border-left-color: var(--accent-primary);
        background: rgba(99, 102, 241, 0.06);
        transform: translateX(4px);
    }
    .passage-card strong {
        color: var(--accent-primary) !important;
        font-weight: 600;
    }
    .passage-card em {
        color: var(--accent-cyan) !important;
    }

    /* ——— Status Badges ——— */
    .status-badge-ok {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(52, 211, 153, 0.12);
        color: var(--accent-emerald);
        border: 1px solid rgba(52, 211, 153, 0.25);
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.73rem;
        font-weight: 600;
        animation: pulseGlow 3s infinite;
    }
    .status-badge-warn {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(251, 191, 36, 0.12);
        color: var(--accent-amber);
        border: 1px solid rgba(251, 191, 36, 0.25);
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.73rem;
        font-weight: 600;
    }

    /* ——— Tabs Styling ——— */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--glass-bg) !important;
        backdrop-filter: var(--glass-blur) !important;
        border-radius: 14px !important;
        padding: 0.3rem !important;
        gap: 0.25rem !important;
        border: 1px solid var(--glass-border) !important;
        margin-bottom: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        transition: all 0.25s ease !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(99, 102, 241, 0.08) !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        color: var(--accent-primary) !important;
        font-weight: 600 !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.12) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: var(--accent-secondary) !important;
        height: 2px !important;
        border-radius: 2px !important;
    }

    /* ——— Button Styling ——— */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1)) !important;
        color: var(--accent-primary) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.01em !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(139, 92, 246, 0.2)) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    /* Primary action buttons */
    div.stButton > button[kind="primary"] {
        background: var(--gradient-hero) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.45) !important;
        transform: translateY(-2px) !important;
    }

    /* ——— Input Fields ——— */
    .stTextInput input, .stSelectbox select, .stTextArea textarea,
    [data-baseweb="input"] input, [data-baseweb="select"] {
        background: rgba(17, 24, 39, 0.6) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.25s ease !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent-secondary) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.12) !important;
    }

    /* ——— Chat Messages ——— */
    [data-testid="stChatMessage"] {
        background: var(--glass-bg) !important;
        backdrop-filter: var(--glass-blur) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        margin-bottom: 0.75rem !important;
        padding: 1rem 1.25rem !important;
        animation: fadeSlideUp 0.4s ease-out;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: var(--text-primary) !important;
    }

    /* User vs assistant message accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-left: 3px solid var(--accent-cyan) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid var(--accent-primary) !important;
    }

    /* ——— Chat Input ——— */
    [data-testid="stChatInput"] {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px !important;
        padding: 0.2rem !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--text-primary) !important;
    }

    /* ——— Expanders ——— */
    .streamlit-expanderHeader {
        background: rgba(17, 24, 39, 0.4) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderContent {
        background: rgba(11, 15, 25, 0.5) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* ——— Slider ——— */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--accent-secondary) !important;
        box-shadow: 0 0 8px var(--accent-glow) !important;
    }
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
        background: rgba(99, 102, 241, 0.2) !important;
    }

    /* ——— File Uploader ——— */
    [data-testid="stFileUploader"] {
        background: rgba(17, 24, 39, 0.3) !important;
        border: 2px dashed rgba(99, 102, 241, 0.2) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
        background: rgba(99, 102, 241, 0.04) !important;
    }

    /* ——— Progress bar ——— */
    .stProgress > div > div {
        background: var(--gradient-hero) !important;
        border-radius: 10px !important;
    }

    /* ——— Multiselect chips ——— */
    [data-baseweb="tag"] {
        background: rgba(99, 102, 241, 0.15) !important;
        color: var(--accent-primary) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 8px !important;
    }

    /* ——— Dividers ——— */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--glass-border), transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ——— Alert and info boxes ——— */
    .stAlert, [data-testid="stNotification"] {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-secondary) !important;
    }

    /* ——— Markdown text colors ——— */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stCaption, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-primary) !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ——— Code blocks ——— */
    .stCodeBlock, code, pre {
        background: rgba(11, 15, 25, 0.7) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        color: var(--accent-cyan) !important;
    }

    /* ——— Suggested Question Pill ——— */
    .suggested-pill {
        display: inline-block;
        background: var(--gradient-card);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 0.65rem 1rem;
        color: var(--text-secondary);
        font-size: 0.82rem;
        font-weight: 400;
        cursor: pointer;
        transition: all 0.3s ease;
        line-height: 1.4;
    }
    .suggested-pill:hover {
        border-color: var(--accent-primary);
        color: var(--text-primary);
        background: rgba(99, 102, 241, 0.1);
        box-shadow: 0 0 16px var(--accent-glow);
        transform: translateY(-2px);
    }

    /* ——— Welcome Card ——— */
    .welcome-card {
        background: var(--glass-bg);
        backdrop-filter: var(--glass-blur);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: var(--glass-shadow);
        animation: fadeSlideUp 0.7s ease-out;
    }
    .welcome-card h2 {
        background: var(--gradient-hero);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        margin-bottom: 0.75rem;
    }
    .welcome-card p {
        color: var(--text-secondary) !important;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 600px;
        margin: 0 auto;
    }

    /* ——— Section Headings ——— */
    .section-heading {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-heading .icon {
        font-size: 1.4rem;
    }

    /* ——— Doc Chip in Sidebar ——— */
    .doc-chip {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.25s ease;
    }
    .doc-chip:hover {
        border-color: var(--glass-border);
        background: rgba(99, 102, 241, 0.04);
    }
    .doc-chip-name {
        font-weight: 600;
        font-size: 0.88rem;
        color: var(--text-primary);
    }
    .doc-chip-meta {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.15rem;
    }

    /* ——— Sidebar Divider ——— */
    [data-testid="stSidebar"] hr {
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.2), transparent) !important;
    }

    /* ——— Hide Streamlit Default Branding ——— */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(12px) !important;
    }

    /* ——— Checkbox ——— */
    .stCheckbox label span {
        color: var(--text-secondary) !important;
    }

    /* ——— Selectbox Dropdown ——— */
    [data-baseweb="popover"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
    }
    [data-baseweb="menu"] {
        background: var(--bg-secondary) !important;
    }
    [data-baseweb="menu"] li {
        color: var(--text-primary) !important;
    }
    [data-baseweb="menu"] li:hover {
        background: rgba(99, 102, 241, 0.12) !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --- Cached Resources ---
@st.cache_resource(show_spinner="Loading Embedding Model (all-MiniLM-L6-v2)...")
def get_cached_embedding_manager(model_name: str) -> EmbeddingManager:
    """Load and cache SentenceTransformer embedding manager."""
    return EmbeddingManager(model_name=model_name)


@st.cache_resource(show_spinner="Connecting to persistent Vector Store...")
def get_cached_vector_store(persist_dir: str, collection_name: str) -> VectorStore:
    """Initialize and cache persistent ChromaDB vector store."""
    return VectorStore(persist_directory=persist_dir, collection_name=collection_name)


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "processed_file_signatures" not in st.session_state:
    st.session_state.processed_file_signatures = set()

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = [
        "What are the central research objectives across the documents?",
        "Compare the methodologies and frameworks proposed in the papers.",
        "What empirical datasets and evaluation metrics were utilized?",
        "What are the primary limitations and future research avenues highlighted?",
    ]

if "active_user_query" not in st.session_state:
    st.session_state.active_user_query = ""


# Initialize core resources
embedding_manager = get_cached_embedding_manager(config.embedding_model)
vector_store = get_cached_vector_store(config.chroma_db_path, config.chroma_collection_name)
chunker = TextChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)


# --- Sidebar: Configuration & Document Management ---
with st.sidebar:
    # Sidebar Logo & Brand
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 0.5rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.25rem;">🔬</div>
            <div style="font-size: 1.3rem; font-weight: 800;
                 background: linear-gradient(135deg, #6366F1, #EC4899);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                 background-clip: text;">Research Hub</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;
                 letter-spacing: 0.08em; text-transform: uppercase;">AI-Powered Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # 1. Groq API Configuration
    st.subheader("🔑 Inference Settings")
    default_key = config.groq_api_key or ""
    api_key_input = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Enter your Groq API key (starts with gsk_). Can also be set in .env file.",
    )

    # Discover available models or use validated supported models list
    models_list = list(config.supported_models)
    if api_key_input:
        temp_service = GroqService(api_key=api_key_input)
        discovered = temp_service.get_available_models()
        if discovered:
            models_list = discovered

    # Determine default index (defaulting to openai/gpt-oss-120b)
    default_model = config.groq_model if config.groq_model in models_list else models_list[0]
    default_idx = models_list.index(default_model) if default_model in models_list else 0

    selected_model = st.selectbox(
        "Groq Chat Model",
        options=models_list,
        index=default_idx,
        help="Select an active Groq LLM accessible on your account.",
    )

    # Initialize Groq Service with user inputs
    groq_service = GroqService(api_key=api_key_input, model_name=selected_model)

    # Test Connection button
    col_conn1, col_conn2 = st.columns([1, 1])
    with col_conn1:
        if st.button("🔌 Test API", use_container_width=True):
            success, msg = groq_service.test_connection()
            if success:
                st.success(msg)
            else:
                st.error(msg)
    with col_conn2:
        if api_key_input:
            st.markdown("<span class='status-badge-ok'>✦ Connected</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-badge-warn'>⚠ Key Needed</span>", unsafe_allow_html=True)

    st.markdown("---")

    # 2. Retrieval Settings
    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider(
        "Retrieved Passages (Top-k)",
        min_value=1,
        max_value=15,
        value=config.default_top_k,
        help="Number of most relevant chunks retrieved for answering.",
    )

    st.markdown("---")

    # 3. Document Upload Section
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF Research Papers",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Max {config.max_files_per_upload} files, up to {config.max_upload_size_mb} MB each.",
    )

    # Handle Upload Ingestion
    if uploaded_files:
        new_files = []
        for file in uploaded_files:
            file_sig = f"{file.name}_{file.size}"
            if file_sig not in st.session_state.processed_file_signatures:
                new_files.append(file)

        if new_files:
            progress_bar = st.progress(0, text="Processing uploaded PDFs...")
            for idx, file in enumerate(new_files):
                file_bytes = file.read()
                file_sig = f"{file.name}_{file.size}"

                # Step 1: Validation
                is_valid, err_msg = validate_pdf_upload(
                    file.name, file_bytes, max_size_mb=config.max_upload_size_mb
                )
                if not is_valid:
                    st.error(f"❌ {err_msg}")
                    continue

                # Step 2: Extraction
                with st.spinner(f"Extracting text from '{file.name}'..."):
                    doc = extract_text_from_pdf(file_bytes, file.name)

                if not doc.is_valid:
                    st.error(f"❌ Error in '{file.name}': {doc.error_message}")
                    continue

                if doc.error_message:
                    st.warning(f"⚠️ Notice for '{file.name}': {doc.error_message}")

                if doc.total_characters == 0:
                    st.info(f"Skipping index for '{file.name}': No extractable text found.")
                    st.session_state.processed_file_signatures.add(file_sig)
                    continue

                # Step 3: Chunking
                chunks = chunker.chunk_document(doc)
                if not chunks:
                    st.warning(f"No textual chunks produced for '{file.name}'.")
                    continue

                # Step 4: Embedding and Storage
                with st.spinner(f"Generating embeddings and indexing '{file.name}' ({len(chunks)} chunks)..."):
                    chunk_texts = [c.text for c in chunks]
                    embeddings = embedding_manager.embed_texts(chunk_texts)
                    vector_store.add_chunks(chunks, embeddings)

                st.session_state.processed_file_signatures.add(file_sig)
                progress_bar.progress((idx + 1) / len(new_files), text=f"Indexed '{file.name}' successfully.")

            st.success(f"Processed {len(new_files)} document(s) successfully!")
            st.rerun()

    st.markdown("---")

    # 4. Indexed Documents Management
    st.subheader("📑 Indexed Documents")
    indexed_docs = vector_store.get_indexed_documents()

    if not indexed_docs:
        st.info("No documents indexed yet. Upload PDFs above to begin.")
    else:
        # Document Filter for targeted questions
        doc_options = {d["document_id"]: f"{d['filename']} ({d['chunk_count']} chunks)" for d in indexed_docs}
        selected_doc_ids = st.multiselect(
            "Filter Search Scope:",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            default=[],
            help="Select specific documents to target, or leave blank to search across ALL documents.",
        )

        st.caption(f"**Total Indexed:** {len(indexed_docs)} document(s)")

        # List individual documents with delete buttons
        with st.expander("Manage Document Collection", expanded=False):
            for doc_item in indexed_docs:
                st.markdown(
                    f"""
                    <div class="doc-chip">
                        <div class="doc-chip-name">📄 {doc_item['filename']}</div>
                        <div class="doc-chip-meta">{doc_item['page_count']} pages • {doc_item['chunk_count']} chunks</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("🗑️ Delete", key=f"del_{doc_item['document_id']}", help=f"Delete {doc_item['filename']}"):
                    vector_store.delete_document(doc_item["document_id"])
                    st.success(f"Deleted {doc_item['filename']}")
                    st.rerun()

    st.markdown("---")

    # 5. Reset and Clear Actions
    col_clr1, col_clr2 = st.columns(2)
    with col_clr1:
        if st.button("🧹 Clear Chat", use_container_width=True, help="Reset conversation history without deleting documents"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()

    with col_clr2:
        confirm_clear = st.checkbox("Confirm wipe", key="chk_confirm_clear")
        if st.button("🗑️ Clear All", use_container_width=True, disabled=not confirm_clear, help="Wipe all indexed documents from vector database"):
            vector_store.clear_collection()
            st.session_state.processed_file_signatures.clear()
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.success("Vector store wiped successfully!")
            st.rerun()

    # Sidebar footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 0.5rem 0;">
            <div style="font-size: 0.65rem; color: #475569; letter-spacing: 0.05em;">
                POWERED BY
            </div>
            <div style="font-size: 0.75rem; color: #818CF8; font-weight: 600; margin-top: 0.15rem;">
                ChromaDB • Groq • SentenceTransformers
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Main Dashboard Area ---
st.markdown('<div class="hero-title">Multi-Document Research Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Explore, compare, and understand multiple research documents with AI-powered semantic retrieval.</div>',
    unsafe_allow_html=True,
)

# Initialize RAG Pipeline
rag_pipeline = RAGPipeline(
    vector_store=vector_store,
    embedding_manager=embedding_manager,
    groq_service=groq_service,
)

# Metrics Banner — Glass Cards
total_docs_count = len(indexed_docs)
total_chunks_count = vector_store.count()
total_pages_count = sum(d["page_count"] for d in indexed_docs)

st.markdown(
    f"""
    <div class="metrics-grid">
        <div class="glass-metric">
            <div class="metric-icon">📚</div>
            <div class="metric-val">{total_docs_count}</div>
            <div class="metric-lbl">Documents</div>
        </div>
        <div class="glass-metric">
            <div class="metric-icon">📃</div>
            <div class="metric-val">{total_pages_count}</div>
            <div class="metric-lbl">Pages Processed</div>
        </div>
        <div class="glass-metric">
            <div class="metric-icon">🧬</div>
            <div class="metric-val">{total_chunks_count}</div>
            <div class="metric-lbl">Vector Chunks</div>
        </div>
        <div class="glass-metric">
            <div class="metric-icon">💬</div>
            <div class="metric-val">{st.session_state.query_count}</div>
            <div class="metric-lbl">Session Queries</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Primary Tab Navigation ---
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Research Chat & Q&A",
    "⚖️ Cross-Document Comparison",
    "📄 Document Summarizer",
    "🔍 Source Explorer",
])


# ==========================================
# TAB 1: RESEARCH CHAT & Q&A
# ==========================================
with tab1:
    if total_docs_count == 0:
        st.markdown(
            """
            <div class="welcome-card">
                <h2>👋 Welcome, Researcher</h2>
                <p>Upload one or more research papers (PDF) in the sidebar to get started.<br/>
                Once indexed, you can ask cross-document questions, compare methodologies,
                and trace citations back to the exact source text.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Suggested Questions Section
        st.markdown('<div class="section-heading"><span class="icon">💡</span> Suggested Research Questions</div>', unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, q in enumerate(st.session_state.suggested_questions):
            with cols[i]:
                if st.button(f"📌 {q}", key=f"sugg_q_{i}", use_container_width=True):
                    st.session_state.active_user_query = q

        st.markdown("---")

        # Display Conversation History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # Render verified citations if available
                if "citations" in msg and msg["citations"]:
                    st.markdown("###### 📚 Verified Source Citations:")
                    for cit in msg["citations"]:
                        st.markdown(
                            f"<span class='citation-tag'>📍 {cit['formatted']}</span> "
                            f"(Relevance: `{cit['relevance']}%`) — *\"{cit['snippet']}\"*",
                            unsafe_allow_html=True,
                        )

                # Render retrieved passages in expandable accordion
                if "passages" in msg and msg["passages"]:
                    with st.expander(f"🔎 Inspect Retrieved Passages ({len(msg['passages'])})"):
                        for p in msg["passages"]:
                            st.markdown(
                                f"<div class='passage-card'>"
                                f"<strong>{p['filename']}</strong> (Page {p['page_number']}) — "
                                f"<em>Similarity: {int(p['similarity'] * 100)}%</em><br/>"
                                f"<div style='margin-top:0.35rem;'>{p['text']}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

        # Handle Query Input (either from chat_input or suggested button)
        user_prompt = st.chat_input("Ask a question across all uploaded documents...")
        if not user_prompt and st.session_state.active_user_query:
            user_prompt = st.session_state.active_user_query
            st.session_state.active_user_query = ""

        if user_prompt:
            # Check Groq configuration
            if not groq_service.api_key:
                st.warning("⚠️ Please provide a Groq API Key in the sidebar to generate answers.")
            else:
                # Add user query to chat
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                st.session_state.query_count += 1

                with st.chat_message("user"):
                    st.markdown(user_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving relevant passages and reasoning with Groq..."):
                        # Execute RAG Pipeline
                        response = rag_pipeline.answer_question(
                            query=user_prompt,
                            top_k=top_k,
                            filter_document_ids=selected_doc_ids if 'selected_doc_ids' in locals() and selected_doc_ids else None,
                            chat_history=st.session_state.messages[:-1],
                        )

                    st.markdown(response.answer)

                    # Display citations
                    if response.citations:
                        st.markdown("###### 📚 Verified Source Citations:")
                        for cit in response.citations:
                            st.markdown(
                                f"<span class='citation-tag'>📍 {cit.formatted_citation}</span> "
                                f"(Relevance: `{int(cit.similarity_score * 100)}%`) — *\"{cit.snippet}\"*",
                                unsafe_allow_html=True,
                            )

                    # Display retrieved passages accordion
                    if response.passages:
                        with st.expander(f"🔎 Inspect Retrieved Passages ({len(response.passages)})"):
                            for p in response.passages:
                                st.markdown(
                                    f"<div class='passage-card'>"
                                    f"<strong>{p.filename}</strong> (Page {p.page_number}) — "
                                    f"<em>Similarity: {int(p.similarity_score * 100)}%</em><br/>"
                                    f"<div style='margin-top:0.35rem;'>{p.text}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    # Save to conversation state
                    msg_record = {
                        "role": "assistant",
                        "content": response.answer,
                        "citations": [
                            {
                                "formatted": c.formatted_citation,
                                "relevance": int(c.similarity_score * 100),
                                "snippet": c.snippet,
                            }
                            for c in response.citations
                        ],
                        "passages": [
                            {
                                "filename": p.filename,
                                "page_number": p.page_number,
                                "similarity": p.similarity_score,
                                "text": p.text,
                            }
                            for p in response.passages
                        ],
                    }
                    st.session_state.messages.append(msg_record)


# ==========================================
# TAB 2: CROSS-DOCUMENT COMPARISON
# ==========================================
with tab2:
    st.markdown('<div class="section-heading"><span class="icon">⚖️</span> Cross-Document Comparative Analysis</div>', unsafe_allow_html=True)
    st.write(
        "Compare research methodologies, theoretical frameworks, datasets, empirical results, "
        "and stated limitations across two or more uploaded papers."
    )

    if total_docs_count < 2:
        st.warning("Please upload at least 2 documents to enable cross-document comparative analysis.")
    else:
        doc_map = {d["document_id"]: d["filename"] for d in indexed_docs}
        selected_for_comp = st.multiselect(
            "Select Documents to Compare (minimum 2):",
            options=list(doc_map.keys()),
            format_func=lambda x: doc_map[x],
            default=list(doc_map.keys())[:min(3, len(doc_map))],
        )

        aspect_options = [
            "Methodology & Experimental Framework",
            "Key Findings & Empirical Performance",
            "Datasets & Evaluation Metrics",
            "Stated Limitations & Future Research Directions",
            "Comprehensive Synthesis across all dimensions",
            "Custom Aspect",
        ]
        chosen_aspect = st.selectbox("Comparison Dimension:", options=aspect_options)

        custom_prompt = ""
        if chosen_aspect == "Custom Aspect":
            custom_prompt = st.text_input("Specify custom comparative question:", "Compare the computational efficiency and parameter sizes.")

        if st.button("🚀 Run Comparative Analysis", type="primary", disabled=len(selected_for_comp) < 2):
            if not groq_service.api_key:
                st.error("Please enter a Groq API Key in the sidebar.")
            else:
                aspect_query = custom_prompt if chosen_aspect == "Custom Aspect" else chosen_aspect
                with st.spinner(f"Analyzing and comparing {len(selected_for_comp)} documents..."):
                    comp_response = rag_pipeline.compare_documents(
                        document_ids=selected_for_comp,
                        comparison_aspect=aspect_query,
                    )
                st.session_state.query_count += 1
                st.markdown(comp_response.answer)

                if comp_response.citations:
                    st.markdown("---")
                    st.markdown("##### 📚 Comparative Evidence Citations:")
                    for cit in comp_response.citations:
                        st.markdown(
                            f"<span class='citation-tag'>📍 {cit.formatted_citation}</span> — *\"{cit.snippet}\"*",
                            unsafe_allow_html=True,
                        )

                if comp_response.passages:
                    with st.expander(f"🔎 Supporting Passages ({len(comp_response.passages)})"):
                        for p in comp_response.passages:
                            st.markdown(
                                f"<div class='passage-card'>"
                                f"<strong>{p.filename}</strong> (Page {p.page_number})<br/>"
                                f"<div style='margin-top:0.35rem;'>{p.text}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )


# ==========================================
# TAB 3: DOCUMENT SUMMARIZER
# ==========================================
with tab3:
    st.markdown('<div class="section-heading"><span class="icon">📄</span> Document Executive Summarizer</div>', unsafe_allow_html=True)
    st.write(
        "Generate an academic executive summary of any indexed research paper, "
        "highlighting core problem statements, methodology, empirical findings, and limitations."
    )

    if total_docs_count == 0:
        st.info("Upload documents in the sidebar to generate summaries.")
    else:
        doc_dict = {d["document_id"]: f"{d['filename']} ({d['page_count']} pages)" for d in indexed_docs}
        selected_summary_doc_id = st.selectbox(
            "Select Document to Summarize:",
            options=list(doc_dict.keys()),
            format_func=lambda x: doc_dict[x],
        )

        if st.button("📑 Generate Executive Summary", type="primary"):
            if not groq_service.api_key:
                st.error("Please enter a Groq API Key in the sidebar.")
            else:
                target_filename = doc_map.get(selected_summary_doc_id, "Selected Document") if 'doc_map' in locals() else "Selected Document"
                for d in indexed_docs:
                    if d["document_id"] == selected_summary_doc_id:
                        target_filename = d["filename"]

                with st.spinner(f"Synthesizing executive summary for '{target_filename}'..."):
                    summary_res = rag_pipeline.summarize_document(
                        document_id=selected_summary_doc_id,
                        filename=target_filename,
                    )
                st.session_state.query_count += 1
                st.markdown(summary_res.answer)

                if summary_res.citations:
                    st.markdown("---")
                    st.markdown("##### 📚 Cited Pages & Evidence:")
                    for cit in summary_res.citations:
                        st.markdown(
                            f"<span class='citation-tag'>📍 {cit.formatted_citation}</span> — *\"{cit.snippet}\"*",
                            unsafe_allow_html=True,
                        )


# ==========================================
# TAB 4: SOURCE EXPLORER
# ==========================================
with tab4:
    st.markdown('<div class="section-heading"><span class="icon">🔍</span> Source Explorer & Chunk Inspector</div>', unsafe_allow_html=True)
    st.write("Browse all chunks stored in ChromaDB, inspect metadata, and review extracted text segments.")

    if total_chunks_count == 0:
        st.info("No chunks stored yet. Upload documents in the sidebar.")
    else:
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            search_filter = st.text_input("Filter chunks by keyword:", placeholder="Type a term to search within stored chunks...")
        with col_s2:
            filter_doc_name = st.selectbox(
                "Filter by Document:",
                options=["All Documents"] + [d["filename"] for d in indexed_docs],
            )

        # Retrieve raw chunks from ChromaDB
        all_raw = vector_store.collection.get(include=["documents", "metadatas"])
        raw_docs = all_raw.get("documents", [])
        raw_metas = all_raw.get("metadatas", [])
        raw_ids = all_raw.get("ids", [])

        displayed_count = 0
        for i in range(len(raw_docs)):
            text = raw_docs[i]
            meta = raw_metas[i] if i < len(raw_metas) else {}
            chunk_id = raw_ids[i] if i < len(raw_ids) else f"chunk_{i}"

            doc_fn = meta.get("filename", "Unknown")
            page_no = meta.get("page_number", "N/A")

            # Apply filters
            if filter_doc_name != "All Documents" and doc_fn != filter_doc_name:
                continue
            if search_filter and search_filter.lower() not in text.lower():
                continue

            displayed_count += 1
            if displayed_count <= 25:
                with st.expander(f"🧩 {doc_fn} | Page {page_no} (ID: `{chunk_id}`)"):
                    st.markdown(f"**Chunk Length:** `{len(text)}` characters | **Document ID:** `{meta.get('document_id', '')}`")
                    st.code(text, language="text")

        if displayed_count > 25:
            st.caption(f"Showing first 25 chunks out of {displayed_count} matching chunks.")
        elif displayed_count == 0:
            st.warning("No chunks matched your filter criteria.")
