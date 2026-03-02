"""
SpecAgent: Agentic RAG for 3GPP Telecommunications Specifications

This package provides an agentic RAG (Retrieval-Augmented Generation) system
specifically designed for querying 3GPP Release 18 specifications using
natural language.

Key Components:
    - nodes: LangGraph nodes (router, grader, rewriter, generator, etc.)
    - graph: LangGraph workflow definition and state management
    - retrieval: Document chunking, embedding, and vector indexing
    - api: FastAPI REST endpoints
    - evaluation: RAGAS metrics and benchmark runners
    - tracing: Arize Phoenix observability integration

Example:
    >>> from specagent import query
    >>> response = query("What is the maximum number of HARQ processes in NR?")
    >>> print(response.answer)
    >>> print(response.citations)
"""

__version__ = "0.1.0"
__author__ = "Spiros Kapoulas"

from specagent.config import settings

__all__ = [
    "__version__",
    "settings",
]
