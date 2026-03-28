"""
Pydantic models for API request and response schemas.

These models provide automatic validation and OpenAPI documentation.
"""

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for the /query endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question about 3GPP specifications",
        examples=["What is the maximum number of HARQ processes in NR?"],
    )
    verbose: bool = Field(
        default=False,
        description="Include detailed reasoning and intermediate steps",
    )
    max_rewrites: int | None = Field(
        default=None,
        ge=0,
        le=5,
        description="Maximum number of query rewrites. If omitted, uses the server default.",
    )
    library: str | None = Field(
        default=None,
        description="Library to restrict retrieval to. If None, uses the server default library.",
    )


class CitationSchema(BaseModel):
    """Schema for a source citation."""

    spec_id: str = Field(
        description="3GPP specification ID (e.g., 'TS38.331')",
        examples=["TS38.331"],
    )
    section: str = Field(
        description="Section reference within the spec",
        examples=["5.3.3"],
    )
    chunk_preview: str = Field(
        default="",
        description="Preview of the source text",
    )


class QueryMetadata(BaseModel):
    """Metadata about query processing."""

    rewrites: int = Field(
        description="Number of query rewrites performed",
    )
    chunks_retrieved: int = Field(
        description="Total chunks retrieved from index",
    )
    chunks_used: int = Field(
        description="Chunks marked as relevant and used for generation",
    )
    latency_ms: float = Field(
        description="Total processing time in milliseconds",
    )
    hallucination_check: str = Field(
        description="Result of hallucination verification",
    )
    rewritten_question: str | None = Field(
        default=None,
        description="Final rewritten query (present when query was rewritten)",
    )
    node_timings: dict | None = Field(
        default=None,
        description="Per-node execution times in milliseconds (verbose mode only)",
    )


class QueryResponse(BaseModel):
    """Response schema for the /query endpoint."""

    answer: str = Field(
        description="Generated answer with inline citations",
    )
    citations: list[CitationSchema] = Field(
        default_factory=list,
        description="Source citations for the answer",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the answer (0.0 to 1.0)",
    )
    metadata: QueryMetadata = Field(
        description="Processing metadata and statistics",
    )
    ungrounded_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the answer not fully supported by retrieved sources.",
    )
    hallucination_status: Literal["grounded", "partial", "not_grounded", "skipped", "unknown"] = Field(
        default="unknown",
        description="Hallucination check result: grounded | partial | not_grounded | skipped | unknown.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "In NR Release 18, the maximum number of HARQ processes is 16 for FDD and TDD. [TS 38.321 §5.4.1]",
                    "citations": [
                        {
                            "spec_id": "TS38.321",
                            "section": "5.4.1",
                            "chunk_preview": "The UE shall support a maximum of 16 HARQ processes...",
                        }
                    ],
                    "confidence": 0.92,
                    "metadata": {
                        "rewrites": 0,
                        "chunks_retrieved": 10,
                        "chunks_used": 3,
                        "latency_ms": 2340.5,
                        "hallucination_check": "grounded",
                    },
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(
        description="Health status",
        examples=["healthy", "degraded", "unhealthy"],
    )
    version: str = Field(
        description="API version",
    )
    index_loaded: bool = Field(
        description="Whether LanceDB store is open",
    )


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str = Field(
        description="Error code",
        examples=["off_topic", "pipeline_error", "internal_error"],
    )
    message: str = Field(
        description="Human-readable error message",
    )
    reasoning: str | None = Field(
        default=None,
        description="Additional context (e.g., router reasoning for off-topic)",
    )
