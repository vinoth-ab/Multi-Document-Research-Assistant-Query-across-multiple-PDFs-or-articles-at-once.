"""
Configuration module for Multi-Document Research Assistant.
Handles environment variables, Pydantic validation, and application defaults.
"""

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Suppress HuggingFace symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load environment variables from .env file if available
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


class AppConfig(BaseModel):
    """Application configuration schema with validation."""

    # Groq API settings
    groq_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip() or None,
        description="Groq API key for LLM inference",
    )
    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        description="Active Groq model ID",
    )

    # Supported Groq Models for user selection
    supported_models: List[str] = Field(
        default=[
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "allam-2-7b",
        ],
        description="List of reliable Groq models",
    )

    # ChromaDB settings
    chroma_db_path: str = Field(
        default_factory=lambda: os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
        description="Local directory for ChromaDB vector persistence",
    )
    chroma_collection_name: str = Field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "research_documents"),
        description="ChromaDB collection name for indexed chunks",
    )

    # Embedding settings
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        description="HuggingFace Sentence Transformers model name",
    )

    # Chunking settings
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")),
        ge=200,
        le=4000,
        description="Target character size for text chunks",
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")),
        ge=0,
        le=500,
        description="Character overlap between consecutive chunks",
    )

    # Retrieval settings
    default_top_k: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "5")),
        ge=1,
        le=20,
        description="Default number of passages to retrieve for RAG",
    )

    # File upload limits
    max_upload_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "20")),
        ge=1,
        le=100,
        description="Maximum file size in megabytes per document",
    )
    max_files_per_upload: int = Field(
        default_factory=lambda: int(os.getenv("MAX_FILES_PER_UPLOAD", "10")),
        ge=1,
        le=30,
        description="Maximum number of files allowed per upload session",
    )

    def is_groq_configured(self) -> bool:
        """Check if Groq API key is present and non-empty."""
        return bool(self.groq_api_key and len(self.groq_api_key) > 5)

    def get_absolute_chroma_path(self) -> Path:
        """Return the absolute path for ChromaDB storage directory."""
        path = Path(self.chroma_db_path)
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        return path


# Singleton default configuration
config = AppConfig()
