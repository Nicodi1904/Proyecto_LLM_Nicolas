#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import importlib
from typing import Any

from .base_chunker import BaseChunker
from .get_chunker import get_chunker
from .langchain_chunker import LangChainChunker

__all__ = [
    "BaseChunker",
    "LangChainChunker",
    "HybridSemanticChunker",
    "get_chunker",
]

_module_lookup = {"HybridSemanticChunker": ".hybrid_semantic_chunker"}
_root_module = "ibm_watsonx_ai.foundation_models.extensions.rag.chunker"


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    if name in _module_lookup:
        module = importlib.import_module(_root_module + _module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
