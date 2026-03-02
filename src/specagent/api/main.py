"""
FastAPI application for SpecAgent REST API.

Run with:
    uvicorn specagent.api.main:app --reload

Or use the CLI:
    specagent serve
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from specagent.api.models import (
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from specagent.config import settings
from specagent.graph.workflow import run_query
from specagent.retrieval.resources import initialize_resources

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:
        - Load FAISS index into memory (cached)
        - Initialize embedder model (cached)
        - Initialize tracing if enabled

    Shutdown:
        - Resources cleaned up on process exit
    """
    # Startup
    logger.info("Initializing SpecAgent resources...")

    try:
        status = initialize_resources()
        logger.info(f"Resource initialization status: {status}")
        logger.info("FAISS index loaded successfully")
        logger.info("Embedder initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        raise RuntimeError(f"Startup failed: {e}") from e

    # TODO: Initialize Phoenix tracing if enabled
    if settings.enable_tracing:
        logger.info("Tracing enabled but not yet implemented")

    yield

    # Shutdown
    logger.info("Shutting down SpecAgent...")
    # Resources persist until process exit (@lru_cache lifetime)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance
    """
    app = FastAPI(
        title="3GPP SpecAgent",
        description="Agentic RAG system for 3GPP telecommunications specifications",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(router)

    return app


# Router for API endpoints
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for k8s liveness/readiness probes.

    Returns:
        Health status and basic metrics
    """
    from specagent.retrieval.resources import get_store

    try:
        store = get_store()
        index_loaded = store is not None
    except Exception:
        index_loaded = False

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        index_loaded=index_loaded,  # Real status
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Off-topic query"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    tags=["Query"],
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Execute a query through the agentic RAG pipeline.

    The query flows through:
    1. Router - determines if query is about 3GPP specs
    2. Retriever - fetches relevant document chunks
    3. Grader - scores chunk relevance
    4. Rewriter - reformulates query if needed
    5. Generator - synthesizes answer with citations
    6. Hallucination check - verifies answer is grounded

    Args:
        request: Query request with question and options

    Returns:
        Answer with citations, confidence score, and metadata

    Raises:
        HTTPException: 422 if query is off-topic
        HTTPException: 500 if pipeline fails
    """
    try:
        # Run the query through the pipeline
        result = run_query(request.question)

        # Check if query was rejected
        if result.get("route_decision") == "reject":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "off_topic",
                    "message": "This question is outside 3GPP specifications. "
                    "I can help with telecom standards questions.",
                    "reasoning": result.get("route_reasoning", ""),
                },
            )

        # Check for errors
        if result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "pipeline_error", "message": result["error"]},
            )

        # Build response
        return QueryResponse(
            answer=result.get("generation", "Unable to generate answer."),
            citations=[
                {
                    "spec_id": c.spec_id,
                    "section": c.section,
                    "chunk_preview": c.chunk_preview,
                }
                for c in result.get("citations", [])
            ],
            confidence=_calculate_confidence(result),
            metadata={
                "rewrites": result.get("rewrite_count", 0),
                "chunks_retrieved": len(result.get("retrieved_chunks", [])),
                "chunks_used": len(
                    [c for c in result.get("graded_chunks", []) if c.relevant == "yes"]
                ),
                "latency_ms": result.get("processing_time_ms", 0),
                "hallucination_check": result.get("hallucination_check", "unknown"),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "message": str(e)},
        )


def _calculate_confidence(result: dict[str, Any]) -> float:
    """
    Calculate overall confidence score for the response.

    Factors:
        - Average grader confidence
        - Hallucination check result
        - Number of rewrites (more = less confident)
    """
    base_confidence = result.get("average_confidence", 0.5)

    # Penalize for hallucination issues
    hallucination_check = result.get("hallucination_check", "grounded")
    if hallucination_check == "not_grounded":
        base_confidence *= 0.5
    elif hallucination_check == "partial":
        base_confidence *= 0.8

    # Slight penalty for rewrites
    rewrite_count = result.get("rewrite_count", 0)
    base_confidence *= 1.0 - (rewrite_count * 0.05)

    return max(0.0, min(1.0, base_confidence))


# Create app instance
app = create_app()
