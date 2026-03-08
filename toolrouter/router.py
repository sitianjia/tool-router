"""Cosine-similarity router with optional tag boosting + must-include rules.

Why this exists: passing 50 tool definitions to every LLM call burns
1-2k input tokens before the model has done any work. For most user
queries, only 3-5 tools are relevant. Pre-filtering with a cheap
embedding model usually preserves answer quality and cuts cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .embedders import BaseEmbedder
from .registry import Tool, ToolRegistry


@dataclass
class RouterConfig:
    k: int = 5
    threshold: float = 0.0
    tag_boost: dict[str, float] = field(default_factory=dict)
    always_include: list[str] = field(default_factory=list)
    never_route_out: list[str] = field(default_factory=list)


class Router:
    def __init__(
        self,
        registry: ToolRegistry,
        embedder: BaseEmbedder,
        config: Optional[RouterConfig] = None,
    ) -> None:
        if len(registry) == 0:
            raise ValueError("registry is empty")
        self.registry = registry
        self.embedder = embedder
        self.config = config or RouterConfig()
        self._tools: list[Tool] = registry.all()
        # one-shot embedding of every tool description
        self._tool_vecs = self._normalize(
            embedder.embed([t.routing_text() for t in self._tools])
        )

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    def select(self, query: str, k: Optional[int] = None) -> list[Tool]:
        """Return the top-k tools for `query`, honoring config rules."""
        k = k or self.config.k
        qv = self._normalize(self.embedder.embed([query]))[0]
        scores = self._tool_vecs @ qv         # cosine sim, since both normed

        # apply tag boost
        if self.config.tag_boost:
            for i, t in enumerate(self._tools):
                for tag in t.tags:
                    if tag in self.config.tag_boost:
                        scores[i] += self.config.tag_boost[tag]

        # threshold + sort
        order = scores.argsort()[::-1]
        picked: list[Tool] = []
        seen: set[str] = set()

        # must-includes first
        for name in self.config.always_include:
            if name in self.registry and name not in seen:
                picked.append(self.registry.get(name))
                seen.add(name)

        # cast i to python int — numpy ints break the registry dict lookups
        for i in (int(x) for x in order):
            tool = self._tools[i]
            if tool.name in self.config.never_route_out:
                continue
            if scores[i] < self.config.threshold:
                continue
            if tool.name in seen:
                continue
            picked.append(tool)
            seen.add(tool.name)
            if len(picked) >= k:
                break

        return picked

    def score(self, query: str) -> dict[str, float]:
        """Return raw scores for all tools — for debugging / tuning."""
        qv = self._normalize(self.embedder.embed([query]))[0]
        scores = self._tool_vecs @ qv
        return {t.name: float(s) for t, s in zip(self._tools, scores)}

    # --- persistence ---

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "tool_vecs.npy", self._tool_vecs)
        (p / "tool_names.txt").write_text(
            "\n".join(t.name for t in self._tools))

    def add_tool(self, tool: Tool) -> None:
        """Add a tool at runtime — re-embeds only the new description."""
        if tool.name in self.registry:
            return
        self.registry.add(tool)
        v = self._normalize(self.embedder.embed([tool.routing_text()]))
        self._tools.append(tool)
        self._tool_vecs = np.concatenate([self._tool_vecs, v], axis=0)

    def remove_tool(self, name: str) -> None:
        idx = next((i for i, t in enumerate(self._tools) if t.name == name), -1)
        if idx < 0:
            return
        self._tools.pop(idx)
        self._tool_vecs = np.delete(self._tool_vecs, idx, axis=0)
        del self.registry._by_name[name]

    @classmethod
    def load(cls, path: str | Path, registry: ToolRegistry,
             embedder: BaseEmbedder, config: Optional[RouterConfig] = None
             ) -> "Router":
        p = Path(path)
        vecs = np.load(p / "tool_vecs.npy")
        names = (p / "tool_names.txt").read_text().splitlines()
        obj = cls.__new__(cls)
        obj.registry = registry
        obj.embedder = embedder
        obj.config = config or RouterConfig()
        obj._tools = [registry.get(n) for n in names]
        obj._tool_vecs = vecs
        return obj
