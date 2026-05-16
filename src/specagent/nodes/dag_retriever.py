"""DAG retriever node and router for the agentic RAG pipeline.

Provides:

- ``route_after_retriever``: conditional edge function — decides whether to
  activate the DAG retriever node or bypass it directly to the grader.
- ``dag_retriever_node``: LangGraph node that queries the Kuzu DAG store
  and populates ``state["dag_chunks"]`` (separate lane, bypasses grader).

Keyword heuristic recognises call-flow / procedure queries using:

- Procedural verbs and nouns: "procedure", "flow", "sequence", "steps", "call flow", etc.
- 3GPP participant names: "UE", "AMF", "gNB", "SMF", "UPF", "AUSF", "UDM", etc.
- Common procedure names: "registration", "handover", "authentication", "PDU session", etc.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Literal

from specagent.config import settings
from specagent.graph.state import GraphState, RetrievedChunk
from specagent.kuzu.resources import get_dag_store
from specagent.nodes.retriever import _normalize_spec_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword heuristic
# ---------------------------------------------------------------------------

# Procedural trigger keywords — at least one must appear in the query.
_PROCEDURE_KEYWORDS = frozenset(
    [
        "procedure",
        "call flow",
        "call-flow",
        "message flow",
        "message sequence",
        "sequence",
        "steps",
        "flow",
        "registration",
        "handover",
        "authentication",
        "pdu session",
        "attach",
        "detach",
        "mobility",
        "setup",
        "establishment",
        "release",
        "handoff",
        "during",
        "walk me through",
        "what happens when",
        "how does",
    ]
)

# 3GPP participant / network function names — presence implies a call-flow query.
_PARTICIPANT_KEYWORDS = frozenset(
    [
        "ue",
        "gnb",
        "amf",
        "smf",
        "upf",
        "ausf",
        "udm",
        "udr",
        "pcf",
        "nssf",
        "nrf",
        "seaf",
        "arpf",
        "sidf",
        "n3iwf",
        "af",
    ]
)

# Combined set for the regex (sorted longest-first avoids partial matches).
_ALL_KEYWORDS = sorted(
    _PROCEDURE_KEYWORDS | _PARTICIPANT_KEYWORDS,
    key=len,
    reverse=True,
)

_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in _ALL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _is_call_flow_query(question: str) -> bool:
    """Return True if the question appears to be about a call flow or procedure."""
    return bool(_KEYWORD_PATTERN.search(question))


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


def route_after_retriever(
    state: GraphState,
) -> Literal["dag_retriever", "grader"]:
    """Conditional edge: route to dag_retriever or directly to grader.

    Args:
        state: Current graph state (reads ``question`` / ``rewritten_question``).

    Returns:
        ``"dag_retriever"`` if the query matches the call-flow heuristic and
        DAG retrieval is enabled; otherwise ``"grader"``.
    """
    if not settings.enable_dag_retrieval:
        return "grader"

    question = state.get("rewritten_question") or state.get("question", "")
    if not question:
        return "grader"

    if _is_call_flow_query(question):
        return "dag_retriever"

    return "grader"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def dag_retriever_node(state: GraphState) -> GraphState:
    """Query the Kuzu DAG store and populate state['dag_chunks'].

    Args:
        state: Current graph state.

    Returns:
        Updated state with ``dag_chunks`` populated.
        On any error, returns state with ``dag_chunks`` unchanged (graceful degradation).
    """
    question = state.get("rewritten_question") or state.get("question", "")

    # Extract keyword tokens for the Kuzu query
    matches = _KEYWORD_PATTERN.findall(question)
    keywords: list[str] = list(dict.fromkeys(m.lower() for m in matches)) if matches else [question]

    try:
        dag_store = get_dag_store()
        dag_results = dag_store.query_dags_by_keyword(
            keywords=keywords,
            limit=settings.dag_retrieval_top_k,
        )

        dag_chunks: list[RetrievedChunk] = []
        for row in dag_results:
            dag_id: str = row.get("dag_id", "")
            mermaid = dag_store.get_dag_mermaid(dag_id)
            if mermaid is None:
                logger.debug("DAG %r has no mermaid content — skipping", dag_id)
                continue

            title: str = row.get("title", dag_id)
            source: str = row.get("source", "")
            doc_id: str = row.get("doc_id", "")

            dag_chunks.append(
                RetrievedChunk(
                    content=f"## Call Flow: {title}\n\n{mermaid}",
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, dag_id)),
                    doc_id=doc_id,
                    source=source,
                    title=title,
                    chunk_index=0,
                    file_type="docx",
                    spec_id=_normalize_spec_id(source) if source else dag_id,
                    section="Call Flow Diagram",
                    similarity_score=settings.dag_retrieval_score,
                )
            )

        state["dag_chunks"] = dag_chunks
        logger.info("DAG retriever found %d call-flow chunks for query", len(dag_chunks))

    except Exception as exc:
        logger.warning("DAG retriever failed — continuing without DAG chunks: %s", exc)
        state["dag_chunks"] = state.get("dag_chunks", [])

    return state


__all__ = ["dag_retriever_node", "route_after_retriever"]
