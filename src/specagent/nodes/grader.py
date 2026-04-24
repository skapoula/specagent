"""
Grader node: Scores relevance of retrieved chunks to the query.

For each retrieved chunk, determines:
    - relevant: "yes" or "no"
    - confidence: 0.0 to 1.0

If average confidence is below threshold, triggers query rewriting.
"""

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from specagent.llm import get_llm
from specagent.nodes._common import parse_json_object, record_llm_call

if TYPE_CHECKING:
    from specagent.graph.state import GraphState

logger = logging.getLogger(__name__)

_HIGH_SIMILARITY_AUTO_THRESHOLD = 0.82
_LOW_SIMILARITY_AUTO_THRESHOLD = 0.55


class GradeResult(BaseModel):
    """Structured output for document grading."""

    relevant: Literal["yes", "no"] = Field(
        description="Whether the document is relevant to answering the question"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for the relevance assessment"
    )


class BatchGradeResult(BaseModel):
    """Structured output for batch document grading."""

    grades: list[GradeResult] = Field(
        description="List of grade results, one per document chunk in order"
    )


BATCH_GRADER_PROMPT = """Assess relevance of document chunks to the question.

Question: {question}

Document chunks ({num_chunks}):
{documents}

Return JSON: {{"grades": [{{"relevant": "yes"/"no", "confidence": 0.0-1.0}}, ...]}}
Provide exactly {num_chunks} grades in order."""


def grader_node(state: "GraphState") -> "GraphState":  # noqa: PLR0915 — auto-grade tracking adds necessary statements to single-responsibility function
    """
    Grade retrieved chunks for relevance to the query.

    Grades only the top-3 chunks (by similarity score) for latency optimization.
    Uses similarity-based auto-grading to skip LLM calls when possible:
    - similarity > 0.82: auto "yes" with high confidence
    - similarity < 0.55: auto "no" with high confidence
    - similarity 0.55-0.82: use LLM for accurate assessment

    Args:
        state: Current graph state with retrieved_chunks populated

    Returns:
        Updated state with graded_chunks containing relevance scores
    """
    from specagent.graph.state import GradedChunk  # noqa: PLC0415

    # Use the rewritten question when available — the retriever uses the same
    # form, so grading against the original can incorrectly downgrade rewritten-
    # query chunks.
    question = state.get("rewritten_question") or state.get("question", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Only grade top-3 chunks for latency optimization
    chunks_to_grade = retrieved_chunks[:3]

    # Handle empty chunks case
    if not chunks_to_grade:
        state["graded_chunks"] = []
        state["average_confidence"] = 0.0
        return state

    try:
        # Separate chunks into auto-gradable and LLM-required
        graded_chunks = []
        llm_chunks = []  # Chunks needing LLM grading
        llm_chunk_indices = []  # Track original positions
        total_confidence = 0.0
        auto_grade_count = 0
        llm_grade_count = 0

        for i, chunk in enumerate(chunks_to_grade):
            if chunk.similarity_score > _HIGH_SIMILARITY_AUTO_THRESHOLD:
                # Auto-grade as relevant with high confidence
                grade = GradeResult(relevant="yes", confidence=chunk.similarity_score)
                graded_chunk = GradedChunk(
                    chunk=chunk, relevant=grade.relevant, confidence=grade.confidence
                )
                graded_chunks.append(graded_chunk)
                total_confidence += grade.confidence
                auto_grade_count += 1
            elif chunk.similarity_score < _LOW_SIMILARITY_AUTO_THRESHOLD:
                # Auto-grade as not relevant with high confidence
                grade = GradeResult(relevant="no", confidence=1.0 - chunk.similarity_score)
                graded_chunk = GradedChunk(
                    chunk=chunk, relevant=grade.relevant, confidence=grade.confidence
                )
                graded_chunks.append(graded_chunk)
                total_confidence += grade.confidence
                auto_grade_count += 1
            else:
                # Mid-range similarity: needs LLM grading
                llm_chunks.append(chunk)
                llm_chunk_indices.append(i)
                # Placeholder to maintain order
                graded_chunks.append(None)

        # If there are chunks requiring LLM grading, process them in batch
        if llm_chunks:
            llm = get_llm()

            # Format chunks for LLM grading
            documents_text = ""
            for i, chunk in enumerate(llm_chunks, 1):
                documents_text += f"\n--- Chunk {i} ---\n{chunk.content}\n"

            # Create batch grading prompt
            prompt = BATCH_GRADER_PROMPT.format(
                question=question, num_chunks=len(llm_chunks), documents=documents_text
            )

            # Single LLM call to grade uncertain chunks
            response = llm.invoke(prompt)
            record_llm_call(state, llm, "grader")

            batch_result = BatchGradeResult(**parse_json_object(response))

            # Verify we got the right number of grades
            if len(batch_result.grades) != len(llm_chunks):
                logger.warning(
                    "Grader: expected %d LLM grades but got %d — falling back to similarity auto-grading",
                    len(llm_chunks),
                    len(batch_result.grades),
                )
                # Fall back to similarity-based auto-grading for LLM-destined chunks
                # rather than silently dropping them, which skews average_confidence.
                midpoint = (_HIGH_SIMILARITY_AUTO_THRESHOLD + _LOW_SIMILARITY_AUTO_THRESHOLD) / 2
                for chunk, idx in zip(llm_chunks, llm_chunk_indices):
                    if chunk.similarity_score >= midpoint:
                        grade = GradeResult(relevant="yes", confidence=chunk.similarity_score)
                    else:
                        grade = GradeResult(relevant="no", confidence=1.0 - chunk.similarity_score)
                    graded_chunk = GradedChunk(
                        chunk=chunk, relevant=grade.relevant, confidence=grade.confidence
                    )
                    graded_chunks[idx] = graded_chunk
                    total_confidence += grade.confidence
                    auto_grade_count += 1
            else:
                # Insert LLM-graded chunks back into their original positions
                for chunk, grade, idx in zip(llm_chunks, batch_result.grades, llm_chunk_indices):
                    graded_chunk = GradedChunk(
                        chunk=chunk, relevant=grade.relevant, confidence=grade.confidence
                    )
                    graded_chunks[idx] = graded_chunk
                    total_confidence += grade.confidence
                llm_grade_count = len(llm_chunks)

        # Remove any None placeholders left if LLM grading was skipped unexpectedly
        graded_chunks = [gc for gc in graded_chunks if gc is not None]

        # Calculate average confidence
        average_confidence = total_confidence / len(graded_chunks) if graded_chunks else 0.0

        # Update state
        state["graded_chunks"] = graded_chunks
        state["average_confidence"] = average_confidence
        state["grader_auto_count"] = auto_grade_count
        state["grader_llm_count"] = llm_grade_count

    except Exception as e:
        logger.error("Grader error: %s", e)
        state["error"] = f"Grader error: {e!s}"
        state["graded_chunks"] = []
        state["average_confidence"] = 0.0
        state["grader_auto_count"] = 0
        state["grader_llm_count"] = 0

    return state
