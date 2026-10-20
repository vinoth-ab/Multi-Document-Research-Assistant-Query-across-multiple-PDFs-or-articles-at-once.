import os
import logging
from typing import List, Optional

# Suppress HuggingFace symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)

# Global singleton instance for cached standalone usage
_EMBEDDING_MODEL_CACHE = {}


class EmbeddingManager:
    """Manages local embedding generation via Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load and cache SentenceTransformer model."""
        if self.model_name in _EMBEDDING_MODEL_CACHE:
            self._model = _EMBEDDING_MODEL_CACHE[self.model_name]
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            model = SentenceTransformer(self.model_name)
            _EMBEDDING_MODEL_CACHE[self.model_name] = model
            self._model = model
            return self._model
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise RuntimeError(
                f"Could not load SentenceTransformer model '{self.model_name}'. "
                f"Ensure 'sentence-transformers' and 'torch' are properly installed. Error: {e}"
            )

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this model (384 for all-MiniLM-L6-v2)."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate normalized embeddings for a list of text strings.
        Returns a list of float lists suitable for ChromaDB.
        """
        if not texts:
            return []

        # sentence-transformers encode returns numpy array or list of tensors
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single search query string.
        Returns a single list of floats.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty for embedding generation.")

        embedding = self.model.encode(
            query.strip(),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()


def get_embedding_manager(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingManager:
    """Helper function to obtain an EmbeddingManager instance."""
    return EmbeddingManager(model_name=model_name)
