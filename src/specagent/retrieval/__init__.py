"""
Document retrieval components for the RAG pipeline.

Components:
    - chunker: Split documents into manageable chunks with metadata
    - converter: Convert local files (PDF, DOCX, etc.) to markdown
    - embeddings: Generate vector embeddings via HuggingFace API
    - indexer: FAISS index management for similarity search
"""

from specagent.retrieval.chunker import Chunk, chunk_markdown
from specagent.retrieval.converter import SUPPORTED_EXTENSIONS, convert_to_markdown
from specagent.retrieval.embeddings import HuggingFaceEmbedder
from specagent.retrieval.indexer import FAISSIndex

__all__ = [
    "Chunk",
    "chunk_markdown",
    "SUPPORTED_EXTENSIONS",
    "convert_to_markdown",
    "HuggingFaceEmbedder",
    "FAISSIndex",
]
