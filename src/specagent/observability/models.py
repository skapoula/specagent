"""Pydantic models for structured observability events emitted by the RAG pipeline."""

import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LLMCallRecord(BaseModel):
    """One instance per LLM invocation, capturing timing and token usage."""

    node: str = Field(description="Node that invoked the LLM (e.g. 'router', 'generator')")
    trace_id: str = Field(description="UUID4 trace identifier from GraphState")
    model: str = Field(description="Model identifier")
    provider: Literal["groq", "custom_endpoint"] = Field(description="LLM provider backend")
    prompt_tokens: int | None = Field(
        description="Input token count; None if backend does not report it"
    )
    completion_tokens: int | None = Field(
        description="Output token count; None if backend does not report it"
    )
    total_tokens: int | None = Field(
        description="Total token count; None if backend does not report it"
    )
    inference_ms: float = Field(description="Wall-clock time for the LLM call in milliseconds")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="UTC timestamp when the LLM was invoked",
    )


class RetrievalRecord(BaseModel):
    """One instance per retriever_node execution, capturing latency and quality metrics."""

    trace_id: str = Field(description="UUID4 trace identifier from GraphState")
    query: str = Field(description="Effective query used for retrieval (may be rewritten)")
    embed_ms: float = Field(
        description="Wall-clock time to compute the query embedding in milliseconds"
    )
    search_ms: float = Field(description="Wall-clock time for the vector search in milliseconds")
    num_results: int = Field(description="Number of chunks returned by the store")
    top_similarity: float | None = Field(description="Highest similarity score; None if no results")
    mean_similarity: float | None = Field(description="Mean similarity score; None if no results")
    rewrite_index: int = Field(
        description="Which rewrite iteration this retrieval was (0 = original query)"
    )
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="UTC timestamp when retrieval was executed",
    )


class QueryEvent(BaseModel):
    """Summary event written once per completed query run."""

    trace_id: str = Field(
        description="UUID4 trace identifier correlating all events for this query"
    )
    question: str = Field(description="Original user question")
    route_decision: str = Field(description="Router outcome: 'retrieve' or 'reject'")
    rewrite_count: int = Field(description="Number of query rewrites performed")
    num_retrieved: int = Field(description="Total chunks retrieved")
    num_relevant: int = Field(description="Chunks graded as relevant")
    hallucination_check: str | None = Field(
        description="Hallucination check outcome or None if not reached"
    )
    total_ms: float = Field(description="Total pipeline wall-clock time in milliseconds")
    llm_calls: list[LLMCallRecord] = Field(description="All LLM calls made during this query")
    retrievals: list[RetrievalRecord] = Field(
        description="All retrieval operations performed during this query"
    )
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="UTC timestamp when the query completed",
    )
