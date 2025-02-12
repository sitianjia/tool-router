"""tool-router — pick the right K tools before they ever hit the LLM."""
__version__ = "0.1.0"

from .registry import Tool, ToolRegistry
from .router import Router
from .embedders import BaseEmbedder, OpenAIEmbedder, BagOfWordsEmbedder

__all__ = [
    "Tool", "ToolRegistry", "Router",
    "BaseEmbedder", "OpenAIEmbedder", "BagOfWordsEmbedder",
]
