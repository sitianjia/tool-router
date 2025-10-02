"""Optional cross-encoder rerank pass.

After top-K cosine selects 20 candidates, a cheap cross-encoder
reorders them. Costs one extra LLM-or-encoder call but improves
precision on ambiguous queries.
"""
from __future__ import annotations

from typing import Callable

from .registry import Tool


CrossEncoderFn = Callable[[str, list[str]], list[float]]


def rerank(query: str, candidates: list[Tool],
           score_fn: CrossEncoderFn, k: int = 5) -> list[Tool]:
    if not candidates:
        return []
    texts = [c.routing_text() for c in candidates]
    scores = score_fn(query, texts)
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [candidates[i] for i in order[:k]]
