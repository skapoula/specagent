"""
LangGraph workflow construction and execution.

Defines the agentic RAG graph with conditional routing:
    - Router decides retrieve vs reject
    - Grader triggers rewriting if confidence is low
    - Hallucination checker can trigger regeneration

Graph visualization can be exported via get_graph_visualization().
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from specagent.config import settings
from specagent.graph.state import GraphState, create_initial_state
from specagent.nodes import (
    dag_retriever_node,
    generator_node,
    grader_node,
    hallucination_check_node,
    retriever_node,
    rewriter_node,
    route_after_retriever,
    router_node,
)

logger = logging.getLogger(__name__)


def create_timed_node(
    node_func: Callable[[GraphState], GraphState], node_name: str
) -> Callable[[GraphState], GraphState]:
    """
    Wrap a node function with timing instrumentation.

    Tracks execution time and stores it in state['node_timings'].

    Args:
        node_func: The original node function
        node_name: Name of the node for timing tracking

    Returns:
        Wrapped node function with timing
    """

    def timed_node(state: GraphState) -> GraphState:
        start_time = time.perf_counter()
        result_state = node_func(state)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update node timings in state
        if "node_timings" not in result_state:
            result_state["node_timings"] = {}

        # Track cumulative time for nodes that may run multiple times (e.g., retriever, grader in rewrite loop)
        current_time = result_state["node_timings"].get(node_name, 0.0)
        result_state["node_timings"][node_name] = current_time + elapsed_ms

        return result_state

    return timed_node


def create_traced_node(
    node_func: Callable[[GraphState], GraphState], node_name: str
) -> Callable[[GraphState], GraphState]:
    """Wrap a node with a Phoenix OTel span (no-op when tracing is disabled).

    Args:
        node_func: The original node function.
        node_name: Span name shown in Phoenix.

    Returns:
        Wrapped node function, or the original if tracing is off.
    """
    if not settings.enable_tracing:
        return node_func

    try:
        from specagent.tracing.phoenix import create_phoenix_node_wrapper  # noqa: PLC0415

        return create_phoenix_node_wrapper(node_func, node_name)
    except ImportError:
        return node_func


def should_retrieve(state: GraphState) -> Literal["retrieve", "reject"]:
    """
    Conditional edge: Route based on router decision.

    Returns:
        "retrieve" to continue with retrieval, "reject" to end
    """
    return state.get("route_decision", "reject")


def should_rewrite(state: GraphState) -> Literal["rewrite", "generate"]:
    """Decide if the query needs rewriting.

    Fast path: skip rewriting when top-3 chunks have average similarity >=
    ``settings.high_similarity_threshold``. Otherwise rewrite when
    ``average_confidence`` is low OR fewer than half the graded chunks are
    relevant, as long as ``rewrite_count < max_rewrites``.

    Returns:
        "rewrite" to reformulate the query, "generate" to proceed.
    """
    rewrite_count = state.get("rewrite_count", 0)
    _override = state.get("max_rewrites_override")
    max_rewrites = _override if _override is not None else settings.max_rewrites
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Fast heuristic: Skip rewriting if top-3 chunks have high similarity
    if retrieved_chunks:
        top_3_chunks = retrieved_chunks[:3]
        avg_similarity = sum(chunk.similarity_score for chunk in top_3_chunks) / len(top_3_chunks)

        if avg_similarity >= settings.high_similarity_threshold:
            # High-quality retrieval, skip rewriting even if grader is uncertain
            return "generate"

    # Check quality metrics from grader
    avg_confidence = state.get("average_confidence", 0.0)
    graded_chunks = state.get("graded_chunks", [])

    # Calculate percentage of relevant chunks (only if grader has run)
    if graded_chunks:
        relevant_count = sum(1 for chunk in graded_chunks if chunk.relevant == "yes")
        relevant_percentage = relevant_count / len(graded_chunks)

        # Rewrite if quality is poor AND we haven't exceeded max rewrites
        quality_is_poor = (
            avg_confidence < settings.grader_confidence_threshold
            or relevant_percentage < settings.min_relevant_chunk_percentage
        )
    else:
        # No graded chunks yet - fall back to confidence-only check
        quality_is_poor = avg_confidence < settings.grader_confidence_threshold

    if quality_is_poor and rewrite_count < max_rewrites:
        return "rewrite"

    return "generate"


def should_regenerate(state: GraphState) -> Literal["regenerate", "finish"]:
    """
    Conditional edge: Decide if answer needs regeneration.

    Triggers at most one regeneration when the hallucination check fails.
    Uses regeneration_count (incremented by hallucination_check_node) to
    prevent the generator → hallucination_check loop from running forever.

    Returns:
        "regenerate" to try again, "finish" to complete
    """
    hallucination_result = state.get("hallucination_check", "grounded")
    regeneration_count = state.get("regeneration_count", 0)

    # Allow exactly one regeneration attempt. hallucination_check_node increments
    # regeneration_count before returning, so count == 1 on the first evaluation here.
    # Guard fires when count <= 1 (first check); blocks when count >= 2 (after retry).
    if hallucination_result == "not_grounded" and regeneration_count <= 1:
        return "regenerate"

    # "partial" is intentionally not retried here. Retrying partial answers tends
    # to produce marginally different outputs at higher token cost without reliably
    # fixing the ungrounded claims. The "partial" status is surfaced to the caller
    # via QueryResponse.hallucination_status so the user can judge. Do not "fix"
    # this to also regenerate on "partial" — that risks an infinite loop if the
    # model repeatedly returns "partial" on borderline content.
    return "finish"


def build_graph() -> CompiledStateGraph:
    """Build and compile the agentic RAG graph.

    Wires router → retriever → grader → generator → hallucination_check with
    conditional edges for DAG retrieval, query rewriting, and regeneration.
    Each node is wrapped with timing and optional Phoenix OTel spans.

    Returns:
        Compiled LangGraph ready for invocation.
    """
    # Initialize graph with state schema
    workflow = StateGraph(GraphState)

    def _wrap(
        func: Callable[[GraphState], GraphState], name: str
    ) -> Callable[[GraphState], GraphState]:
        """Chain timing + Phoenix span wrappers around a node."""
        return create_timed_node(create_traced_node(func, name), name)

    # Add nodes with timing and Phoenix tracing instrumentation
    workflow.add_node("router", _wrap(router_node, "router"))
    workflow.add_node("retriever", _wrap(retriever_node, "retriever"))
    workflow.add_node("dag_retriever", _wrap(dag_retriever_node, "dag_retriever"))
    workflow.add_node("grader", _wrap(grader_node, "grader"))
    workflow.add_node("rewriter", _wrap(rewriter_node, "rewriter"))
    workflow.add_node("generator", _wrap(generator_node, "generator"))
    workflow.add_node("hallucination_check", _wrap(hallucination_check_node, "hallucination_check"))

    # Add edges from START
    workflow.add_edge(START, "router")

    # Router conditional edges
    workflow.add_conditional_edges(
        "router",
        should_retrieve,
        {
            "retrieve": "retriever",
            "reject": END,
        },
    )

    # Retriever → conditional: route to dag_retriever or bypass directly to grader
    workflow.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "dag_retriever": "dag_retriever",
            "grader": "grader",
        },
    )

    # DAG retriever always feeds into grader
    workflow.add_edge("dag_retriever", "grader")

    # Grader conditional edges
    workflow.add_conditional_edges(
        "grader",
        should_rewrite,
        {
            "rewrite": "rewriter",
            "generate": "generator",
        },
    )

    # Rewriter loops back to retriever
    workflow.add_edge("rewriter", "retriever")

    # Generator goes to hallucination check
    workflow.add_edge("generator", "hallucination_check")

    # Hallucination check conditional edges
    workflow.add_conditional_edges(
        "hallucination_check",
        should_regenerate,
        {
            "regenerate": "generator",
            "finish": END,
        },
    )

    # Compile the graph
    return workflow.compile()


_compiled_graph: CompiledStateGraph | None = None


def _get_compiled_graph() -> CompiledStateGraph:
    """Return the compiled graph, building it once and caching it."""
    global _compiled_graph  # noqa: PLW0603
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def _flush_query_journal(state: "GraphState") -> None:
    """Write QueryEvent and sub-records to the JSONL journal. Never raises."""
    try:
        from specagent.observability.journal import get_journal  # noqa: PLC0415
        from specagent.observability.models import QueryEvent  # noqa: PLC0415

        journal = get_journal()
        graded = state.get("graded_chunks", [])
        relevant_count = sum(1 for g in graded if getattr(g, "relevant", None) == "yes")
        event = QueryEvent(
            trace_id=state.get("trace_id", ""),
            question=state.get("question", ""),
            route_decision=state.get("route_decision", "reject"),
            rewrite_count=state.get("rewrite_count", 0),
            num_retrieved=len(state.get("retrieved_chunks", [])),
            num_relevant=relevant_count,
            hallucination_check=state.get("hallucination_check"),
            total_ms=state.get("processing_time_ms", 0.0),
            llm_calls=state.get("llm_calls", []),
            retrievals=state.get("retrieval_events", []),
        )
        journal.write(event)
    except Exception:
        import logging  # noqa: PLC0415

        logging.getLogger(__name__).warning("Failed to write query journal", exc_info=True)


def run_query(
    question: str,
    max_rewrites: int | None = None,
    library: str | None = None,
) -> GraphState:
    """
    Execute a query through the agentic RAG pipeline.

    Args:
        question: User's natural language question
        max_rewrites: Per-request override for max rewrites. Defaults to settings value.
        library: Per-request library filter. If provided, overrides settings.default_library.

    Returns:
        Final graph state with answer, citations, and metadata
    """
    # Create initial state
    state = create_initial_state(question)
    if max_rewrites is not None:
        state["max_rewrites_override"] = max_rewrites
    if library is not None:
        state["library_filter"] = library

    # Use cached compiled graph (compiled once per process)
    graph = _get_compiled_graph()

    start_time = time.perf_counter()

    if settings.enable_tracing:
        try:
            from opentelemetry import trace  # noqa: PLC0415

            from specagent.tracing.rag_spans import emit_query_span  # noqa: PLC0415

            tracer = trace.get_tracer("specagent.pipeline")
            with tracer.start_as_current_span("rag_pipeline") as span:
                span.set_attribute("session.id", state.get("trace_id", ""))
                final_state = graph.invoke(state)
                emit_query_span(
                    query=question,
                    answer=final_state.get("generation"),
                    trace_id=final_state.get("trace_id", ""),
                )
        except ImportError:
            final_state = graph.invoke(state)
    else:
        final_state = graph.invoke(state)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    final_state["processing_time_ms"] = elapsed_ms

    if settings.enable_query_journal:
        _flush_query_journal(final_state)

    try:
        from specagent.observability.report import build_query_report, log_report  # noqa: PLC0415

        log_report(build_query_report(final_state))
    except Exception:
        logger.error("Monitoring report failed", exc_info=True)

    return final_state


def get_graph_visualization() -> str:
    """
    Generate Mermaid diagram of the workflow.

    Returns:
        Mermaid diagram string for visualization

    Example:
        >>> mermaid = get_graph_visualization()
        >>> print(mermaid)  # Paste into mermaid.live
    """
    graph = build_graph()
    return graph.get_graph().draw_mermaid()


def save_graph_image(path: str | Path = Path("docs/architecture.png")) -> None:
    """
    Save graph visualization as PNG image.

    Requires graphviz to be installed.

    Args:
        path: Output path for the PNG file.
    """
    graph = build_graph()
    graph.get_graph().draw_png(str(Path(path).resolve()))
