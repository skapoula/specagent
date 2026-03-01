"""Retriever node: Fetches relevant document chunks from LanceDB.

Embeds the query with the 'search_query: ' prefix (required by
nomic-embed-text-v1.5's asymmetric search design), then runs
hybrid BM25+vector search against the LanceDB store.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from specagent.config import settings
from specagent.graph.state import RetrievedChunk
from specagent.retrieval.resources import get_embedder, get_store

if TYPE_CHECKING:
    from specagent.graph.state import GraphState

logger = logging.getLogger(__name__)

# nomic-embed-text-v1.5 asymmetric search requires task-specific prefix
_QUERY_PREFIX = "search_query: "


def retriever_node(state: "GraphState") -> "GraphState":
    """Retrieve relevant chunks from LanceDB for the query.

    Args:
        state: Current graph state with question or rewritten_question.

    Returns:
        Updated state with retrieved_chunks populated.
    """
    query = state.get("rewritten_question") or state.get("question", "")

    if not query:
        state["error"] = "Retriever error: No query found in state"
        state["retrieved_chunks"] = []
        return state

    try:
        embedder = get_embedder()
        store = get_store()

        # Embed with query prefix (asymmetric search requirement)
        query_vector = list(next(embedder.embed([_QUERY_PREFIX + query])))

        # Hybrid search: BM25 + ANN vector
        results = store.search(
            embedding=query_vector,
            query_text=query,
            top_k=settings.retrieval_top_k,
            library=settings.default_library,
            filter=None,
        )

        retrieved_chunks: list[RetrievedChunk] = []
        for record, similarity_score in results:
            # Derive spec_id from filename stem
            # "TS38.321.docx" -> stem "TS38.321"
            stem = Path(record.source).stem

            # Deserialize section header from metadata JSON
            try:
                meta = json.loads(record.metadata or "{}")
                section = meta.get("section_header", "")
            except (json.JSONDecodeError, AttributeError):
                section = ""

            retrieved_chunks.append(
                RetrievedChunk(
                    content=record.content,
                    chunk_id=record.id,
                    doc_id=record.doc_id,
                    source=record.source,
                    title=record.title,
                    chunk_index=record.chunk_index,
                    file_type=record.file_type,
                    spec_id=stem,
                    section=section,
                    similarity_score=float(similarity_score),
                )
            )

        state["retrieved_chunks"] = retrieved_chunks
        logger.info("Retrieved %d chunks for query", len(retrieved_chunks))

    except Exception as e:
        logger.error("Retriever error: %s", e, exc_info=True)
        state["error"] = f"Retriever error: {e!s}"
        state["retrieved_chunks"] = []

    return state
