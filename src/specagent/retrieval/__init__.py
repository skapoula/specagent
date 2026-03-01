"""
Document retrieval components for the RAG pipeline.

Components:
    - chunker: Split documents into manageable chunks with metadata
    - converter: Convert local files (PDF, DOCX, etc.) to markdown
    - embedder: Generate vector embeddings via fastembed
    - store: LanceDB vector store with hybrid search
    - ingestor: Orchestrate document ingestion pipeline
"""

from specagent.retrieval.converter import SUPPORTED_EXTENSIONS, convert

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "convert",
]
