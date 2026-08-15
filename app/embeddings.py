"""
Embedding generation using an open-source sentence-transformers model.

The embedding model name is fully configurable via the EMBEDDING_MODEL
environment variable. The actual output vector dimension is derived
directly from the loaded model at runtime — it is never assumed to be
1536 or any other fixed number — and is exposed via
`get_embedding_dimension()` so the Pinecone index can be created with
a matching dimension.
"""

from __future__ import annotations

import logging
import threading

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger("rag_chatbot.embeddings")

_model_lock = threading.Lock()
_model: SentenceTransformer | None = None
_model_name: str | None = None


def _load_model() -> SentenceTransformer:
    global _model, _model_name
    settings = get_settings()

    with _model_lock:
        if _model is not None and _model_name == settings.embedding_model:
            return _model

        logger.info("Loading embedding model '%s' (first load may take a moment)...", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        _model_name = settings.embedding_model
        logger.info(
            "Embedding model loaded. Vector dimension = %d",
            _model.get_sentence_embedding_dimension(),
        )
        return _model


def get_embedding_dimension() -> int:
    """Return the true output dimension of the configured embedding model."""
    model = _load_model()
    dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        raise RuntimeError("Could not determine embedding dimension from the loaded model.")
    return int(dimension)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts (document chunks). Returns one vector per text."""
    if not texts:
        return []
    model = _load_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single user query."""
    model = _load_model()
    vector = model.encode([text], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)[0]
    return vector.tolist()
