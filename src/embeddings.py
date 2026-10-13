"""Lazy sentence-embedding helper. Used ONLY for suggestions and ranking — never to score.

If sentence-transformers or the model is unavailable (offline, not installed, or the
SPA_NO_EMBEDDINGS env var is set) every function degrades gracefully and returns None.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

from src.config import load_settings

log = logging.getLogger(__name__)


@lru_cache
def get_model():
    if os.environ.get("SPA_NO_EMBEDDINGS"):
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(load_settings().extraction.embedding_model)
    except Exception as e:  # noqa: BLE001 — any failure just disables embeddings
        log.warning("Embeddings unavailable (%s); continuing without them.", e)
        return None


def embed(texts: list[str]) -> np.ndarray | None:
    model = get_model()
    if model is None or not texts:
        return None
    return model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix for L2-normalised row vectors."""
    return a @ b.T
