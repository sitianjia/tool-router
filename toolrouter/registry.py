"""Tool definitions.

Deliberately small — we don't need to know how to *call* the tool here,
only its name, description, and (optionally) example utterances. The
runtime can hand the same descriptions to whatever agent framework
actually invokes the tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any] | None = None
    examples: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def routing_text(self) -> str:
        """Concatenated text used by embedding-based routers."""
        parts = [self.description]
        if self.examples:
            parts.append("Examples: " + " | ".join(self.examples))
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags))
        return " — ".join(parts)


class ToolRegistry:
    """Holds Tool definitions + indexes by name."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._by_name: dict[str, Tool] = {}
        for t in (tools or []):
            self.add(t)

    def add(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._by_name:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._by_name[tool.name] = tool
        return self

    def get(self, name: str) -> Tool:
        return self._by_name[name]

    def all(self) -> list[Tool]:
        return list(self._by_name.values())

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name
