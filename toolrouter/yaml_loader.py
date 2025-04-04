"""Load a ToolRegistry from a YAML file."""
from __future__ import annotations

from pathlib import Path

import yaml

from .registry import Tool, ToolRegistry


def from_yaml(path: str | Path) -> ToolRegistry:
    raw = yaml.safe_load(Path(path).read_text())
    if isinstance(raw, dict) and "tools" in raw:
        raw = raw["tools"]
    return ToolRegistry([
        Tool(
            name=t["name"],
            description=t["description"],
            parameters=t.get("parameters"),
            examples=t.get("examples", []),
            tags=t.get("tags", []),
        )
        for t in raw
    ])
