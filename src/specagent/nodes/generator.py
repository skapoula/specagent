"""
Generator node: Synthesizes answer from graded document chunks.

Uses relevant chunks as context to generate a comprehensive answer
with inline citations in the format [TS XX.XXX §Y.Z].
"""

import logging
import re
from typing import TYPE_CHECKING

from specagent.llm import get_llm
from specagent.nodes._common import format_spec_ref, record_llm_call

if TYPE_CHECKING:
    from specagent.graph.state import GraphState

logger = logging.getLogger(__name__)


GENERATOR_PROMPT = """You are a 3GPP specification expert assistant. Extract precise answers from the provided 3GPP context ONLY.

Question: {question}

Context (numbered chunks from 3GPP specs):
---
{context}
---

Rules - Follow strictly:
- Answer ONLY from the provided context - NO external knowledge
- For numerical values/parameters/procedures: Extract the EXACT value with units (e.g., "2976 bits", "80 ms", "16 bits", "29 DRBs")
- Synthesize from multiple chunks if needed - information may be spread
- ALWAYS cite inline using [TS XX.XXX §Y.Z] for every claim
- Say "I don't have enough information" ONLY if:
  • The specific parameter/value/procedure is completely absent from ALL chunks
  • Chunks discuss unrelated topics
- DO NOT refuse if the answer is present but requires reading across chunks or technical terminology

Start with the direct answer, then supporting details + citations.
Answer:"""


# Regex pattern to extract citations in format: [TS XX.XXX §Y.Z]
# Captures: TS number (with optional dash) and section number (with dots/letters)
CITATION_PATTERN = re.compile(r"\[TS\s+(\d+\.\d+(?:-\d+)?)\s+§\s*([0-9A-Za-z.]+)\]")


def generator_node(state: "GraphState") -> "GraphState":
    """
    Generate answer from graded chunks.

    Args:
        state: Current graph state with graded_chunks containing relevant docs

    Returns:
        Updated state with generation and citations populated
    """
    # Import at runtime to avoid circular imports
    from specagent.graph.state import Citation  # noqa: PLC0415

    # Get question and graded chunks from state
    question = state.get("question", "")
    graded_chunks = state.get("graded_chunks", [])
    dag_chunks = state.get("dag_chunks", [])

    # Filter for relevant chunks only
    relevant_chunks = [gc.chunk for gc in graded_chunks if gc.relevant == "yes"]

    # Sort by similarity descending so highest-relevance chunks appear first in context
    relevant_chunks.sort(key=lambda c: c.similarity_score, reverse=True)

    # Handle case where no relevant chunks and no DAG chunks are available
    if not relevant_chunks and not dag_chunks:
        state["generation"] = (
            "I don't have enough information in the available specifications "
            "to fully answer this question."
        )
        state["citations"] = []
        return state

    try:
        # Format prose chunks into context string with source metadata and numbering
        context_parts = []
        for idx, chunk in enumerate(relevant_chunks, start=1):
            # Format: **Chunk N** [TS XX.XXX §Y.Z] or [TR XX.XXX §Y.Z]: content
            source_ref = format_spec_ref(chunk.spec_id, chunk.section)
            context_parts.append(f"**Chunk {idx}** {source_ref}:\n{chunk.content}")

        # Append DAG chunks as a clearly-labelled separate section
        if dag_chunks:
            context_parts.append("\n--- Call Flow Diagrams ---")
            for dag_chunk in dag_chunks:
                source_ref = format_spec_ref(dag_chunk.spec_id, dag_chunk.section)
                context_parts.append(f"**{dag_chunk.title}** {source_ref}:\n{dag_chunk.content}")

        context = "\n\n".join(context_parts)

        # Initialize LLM (auto-selects based on config)
        # Use temperature=0.0 for deterministic outputs
        llm = get_llm(temperature=0.0)

        # Format prompt with question and context
        prompt = GENERATOR_PROMPT.format(question=question, context=context)

        # Call LLM to generate answer
        generation = llm.invoke(prompt)
        record_llm_call(state, llm, "generator")
        _call = llm.get_last_call()
        if _call is not None:
            try:
                from specagent.tracing.rag_spans import emit_llm_usage_span  # noqa: PLC0415

                emit_llm_usage_span(_call)
            except Exception:
                logger.error("Tracing span emission failed", exc_info=True)

        # Convert to string and strip whitespace
        generation = generation.strip() if isinstance(generation, str) else str(generation).strip()

        # Parse citations from the generated response
        citations = []
        for match in CITATION_PATTERN.finditer(generation):
            spec_num = match.group(1)  # e.g., "38.321" or "38.101-1"
            section = match.group(2)  # e.g., "5.4" or "5.3.7.1"
            raw_citation = match.group(0)  # Full match like "[TS 38.321 §5.4]"

            # Normalize spec_id (TS38.321 format - no spaces)
            spec_id = f"TS{spec_num.replace(' ', '')}"

            # Create Citation object
            citation = Citation(
                spec_id=spec_id,
                section=section,
                raw_citation=raw_citation,
                chunk_preview="",  # Optional field
            )
            citations.append(citation)

        # Update state
        state["generation"] = generation
        state["citations"] = citations

    except Exception as e:
        logger.error("Generator error: %s", e)
        state["error"] = f"Generator error: {e!s}"
        state["generation"] = None
        state["citations"] = []

    return state
