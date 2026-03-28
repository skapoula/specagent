"""
Rewriter node: Reformulates query for improved retrieval.

When retrieved documents have low relevance scores, the rewriter
generates a more specific query using 3GPP terminology to improve
subsequent retrieval.
"""

import logging
from typing import TYPE_CHECKING

from specagent.config import settings
from specagent.llm import get_llm

if TYPE_CHECKING:
    from specagent.graph.state import GraphState

logger = logging.getLogger(__name__)


REWRITER_PROMPT = """You are a query rewriter for a 3GPP specification search system.

Original question: {question}

The retrieval system found these documents, but they may not be relevant:
{retrieved_chunks_summary}

Rewrite the question to be more specific and likely to match 3GPP specification language.
Consider:
- Expanding acronyms (e.g., "UE" → "User Equipment (UE)")
- Adding technical context (e.g., "handover" → "RRC connection handover procedure")
- Specifying the protocol layer or interface
- Using terminology from 3GPP TS documents

Examples:
Original: "What is handover?"
Rewrite: "RRC connection reconfiguration procedure for X2 handover in 5G NR"

Original: "DRX cycle?"
Rewrite: "Discontinuous Reception (DRX) cycle configuration parameters in TS 38.331"

Original: "HARQ process?"
Rewrite: "Hybrid Automatic Repeat Request (HARQ) process configuration and timing in NR physical layer TS 38.214"

Provide only the rewritten question, nothing else."""


def rewriter_node(state: "GraphState") -> "GraphState":
    """
    Rewrite query for improved retrieval.

    Args:
        state: Current graph state with low-scoring retrieved chunks

    Returns:
        Updated state with rewritten_question and incremented rewrite_count
    """
    # Always rewrite from the original question, not rewritten_question, to avoid
    # semantic drift across multiple rewrite cycles.
    question = state.get("question", "")
    rewrite_count = state.get("rewrite_count", 0)

    # Check if we've reached the maximum number of rewrites.
    # Use the per-request override when present, otherwise fall back to the global setting.
    _override = state.get("max_rewrites_override")
    max_rewrites: int = _override if _override is not None else settings.max_rewrites
    if rewrite_count >= max_rewrites:
        # Don't rewrite if at limit
        return state

    # Get retrieved chunks for context
    retrieved_chunks = state.get("retrieved_chunks", [])

    try:
        # Build summary of retrieved chunks
        if retrieved_chunks:
            chunks_summary = "\n".join(
                [
                    f"- {chunk.content[:200]}..."
                    if len(chunk.content) > 200
                    else f"- {chunk.content}"
                    for chunk in retrieved_chunks[:5]  # Limit to first 5 chunks
                ]
            )
        else:
            chunks_summary = "(No chunks retrieved)"

        # Initialize LLM (auto-selects based on config)
        llm = get_llm()

        # Format prompt with question and chunk summary
        prompt = REWRITER_PROMPT.format(question=question, retrieved_chunks_summary=chunks_summary)

        # Call LLM to rewrite the question
        rewritten_question = llm.invoke(prompt)
        _call = llm.get_last_call()
        if _call is not None:
            _call.node = "rewriter"
            _call.trace_id = state.get("trace_id", "")
            state["llm_calls"] = [*list(state.get("llm_calls", [])), _call]
        _pre_scores = [c.similarity_score for c in state.get("retrieved_chunks", [])]
        if _pre_scores:
            logger.debug(
                "Rewriter pre-scores: top=%.3f mean=%.3f rewrite_count=%d",
                max(_pre_scores),
                sum(_pre_scores) / len(_pre_scores),
                state.get("rewrite_count", 0),
            )

        # Strip whitespace from the response
        if isinstance(rewritten_question, str):
            rewritten_question = rewritten_question.strip()

        # Update state with rewritten question
        state["rewritten_question"] = rewritten_question

        # Increment rewrite count
        state["rewrite_count"] = rewrite_count + 1

    except Exception as e:
        logger.error("Rewriter error: %s", e)
        state["error"] = f"Rewriter error: {e!s}"

    return state
