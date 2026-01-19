"""Jina embeddings via HTTP."""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

from .embedders import BaseEmbedder


class JinaEmbedder(BaseEmbedder):
    def __init__(self, model: str = "jina-embeddings-v3",
                 api_key_env: str = "JINA_API_KEY") -> None:
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise EnvironmentError(f"{api_key_env} not set")
        self.dim = 1024

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        req = urllib.request.Request(
            "https://api.jina.ai/v1/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        arr = np.array([d["embedding"] for d in data["data"]], dtype=np.float32)
        self.dim = arr.shape[1]
        return arr
