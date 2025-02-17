"""Embedding backends.

The router is agnostic to how vectors are produced. Two batteries-included
backends are provided; bring your own for production-quality.
"""
from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class BaseEmbedder(ABC):
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI-compatible /v1/embeddings endpoint.

    Works with OpenAI proper, Together, vLLM, etc.
    """

    def __init__(self, model: str = "text-embedding-3-small",
                 base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY") -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        self.client = OpenAI(base_url=base_url,
                             api_key=os.environ.get(api_key_env, "EMPTY"))
        self.model = model
        self.dim = 1536 if "small" in model else 3072  # rough default

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        resp = self.client.embeddings.create(model=self.model, input=texts)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        self.dim = arr.shape[1]
        return arr


_WORD = re.compile(r"\w+", re.UNICODE)


class BagOfWordsEmbedder(BaseEmbedder):
    """Cheap hashing-trick BoW. No external deps. Good for tests + tiny apps."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _WORD.findall(text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts], axis=0)
