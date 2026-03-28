"""Retriever node: Fetches relevant document chunks from LanceDB.

Embeds the query with the 'search_query: ' prefix (required by
nomic-embed-text-v1.5's asymmetric search design), then runs
hybrid BM25+vector search against the LanceDB store.
"""

import json
import logging
import re
import time
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

# Matches version/release suffixes like -l00, -r17, -e15 at end of filename stem
_VERSION_SUFFIX_RE = re.compile(r"-[ler]\d+$", re.IGNORECASE)


def _normalize_spec_id(source: str) -> str:
    """Derive a normalized spec ID from a source file path.

    Strips version suffixes (e.g. -l00, -r17) and normalizes the series prefix.
    Supports both TS-series and TR-series 3GPP files.

    Examples:
        "/data/TS38.321.docx"   -> "TS38.321"
        "/data/38.321-l00.docx" -> "TS38.321"
        "/data/38.331-r17.docx" -> "TS38.331"
        "/data/TS38.521.pdf"    -> "TS38.521"
        "/data/TR38.821.docx"   -> "TR38.821"
        "/data/tr38.821-l00.docx" -> "TR38.821"
    """
    stem = Path(source).stem
    stem = _VERSION_SUFFIX_RE.sub("", stem)
    upper = stem.upper()
    if upper.startswith("TR"):
        return "TR" + stem[2:]
    if upper.startswith("TS"):
        return "TS" + stem[2:]
    return "TS" + stem  # no prefix → default to TS series


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

    embed_ms: float = 0.0
    search_ms: float = 0.0

    try:
        embedder = get_embedder()
        store = get_store()

        # Embed with query prefix (asymmetric search requirement)
        t0 = time.perf_counter()
        query_vector = list(next(embedder.embed([_QUERY_PREFIX + query])))
        embed_ms = (time.perf_counter() - t0) * 1000

        # Hybrid search: BM25 + ANN vector
        t1 = time.perf_counter()
        results = store.search(
            embedding=query_vector,
            query_text=query,
            top_k=settings.retrieval_top_k,
            library=state.get("library_filter") or settings.default_library,
            filter=None,
        )
        search_ms = (time.perf_counter() - t1) * 1000

        retrieved_chunks: list[RetrievedChunk] = []
        for record, similarity_score in results:
            # Derive normalized spec_id from source path
            # "TS38.321.docx" -> "TS38.321", "38.321-l00.docx" -> "TS38.321"
            spec_id = _normalize_spec_id(record.source)

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
                    spec_id=spec_id,
                    section=section,
                    similarity_score=float(similarity_score),
                )
            )

        from specagent.observability.models import RetrievalRecord  # noqa: PLC0415

        scores = [c.similarity_score for c in retrieved_chunks]
        rec = RetrievalRecord(
            trace_id=state.get("trace_id", ""),
            query=query,
            embed_ms=embed_ms,
            search_ms=search_ms,
            num_results=len(retrieved_chunks),
            top_similarity=max(scores) if scores else None,
            mean_similarity=sum(scores) / len(scores) if scores else None,
            rewrite_index=state.get("rewrite_count", 0),
        )
        state["retrieval_events"] = [*list(state.get("retrieval_events", [])), rec]

        state["retrieved_chunks"] = retrieved_chunks

        try:
            from specagent.tracing.rag_spans import emit_retrieval_span  # noqa: PLC0415

            emit_retrieval_span(
                embed_ms=embed_ms,
                search_ms=search_ms,
                query=query,
                results=retrieved_chunks,
                rewrite_index=state.get("rewrite_count", 0),
            )
        except Exception:
            pass  # tracing must never break retrieval

        logger.info("Retrieved %d chunks for query", len(retrieved_chunks))

    except Exception as e:
        logger.error("Retriever error: %s", e, exc_info=True)
        state["error"] = f"Retriever error: {e!s}"
        state["retrieved_chunks"] = []

    return state
