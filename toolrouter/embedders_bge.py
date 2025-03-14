"""BGE / sentence-transformers backend.

Lazy import so the package doesn't drag torch as a hard dep.
"""
from __future__ import annotations

import numpy as np

from .embedders import BaseEmbedder


class STEmbedder(BaseEmbedder):
    """Wraps sentence-transformers; tested with BGE-small-en-v1.5 and bge-m3."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "pip install sentence-transformers") from e
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self.model.encode(texts, normalize_embeddings=True,
                              convert_to_numpy=True)
        return v.astype(np.float32)
