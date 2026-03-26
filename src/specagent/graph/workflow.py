"""
LangGraph workflow construction and execution.

Defines the agentic RAG graph with conditional routing:
    - Router decides retrieve vs reject
    - Grader triggers rewriting if confidence is low
    - Hallucination checker can trigger regeneration

Graph visualization can be exported via get_graph_visualization().
"""

import time
from collections.abc import Callable
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from specagent.config import settings
from specagent.graph.state import GraphState, create_initial_state
from specagent.nodes import (
    generator_node,
    grader_node,
    hallucination_check_node,
    retriever_node,
    rewriter_node,
    router_node,
)


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


def should_retrieve(state: GraphState) -> Literal["retrieve", "reject"]:
    """
    Conditional edge: Route based on router decision.

    Returns:
        "retrieve" to continue with retrieval, "reject" to end
    """
    return state.get("route_decision", "reject")


def should_rewrite(state: GraphState) -> Literal["rewrite", "generate"]:
    """
    Conditional edge: Decide if query needs rewriting.

    Uses a fast similarity heuristic first, then checks quality metrics:
        1. If top-3 chunks have high similarity (>= threshold), skip rewriting
        2. Otherwise, rewrite if:
           - Average confidence is below threshold OR relevant chunk percentage is below threshold
           - Haven't exceeded max rewrites

    Returns:
        "rewrite" to reformulate query, "generate" to proceed
    """
    rewrite_count = state.get("rewrite_count", 0)
    max_rewrites = state.get("max_rewrites_override") or settings.max_rewrites
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
    regeneration_count = state.get("regeneration_count", 1)

    # Allow exactly one regeneration attempt (count is already incremented by the node)
    if hallucination_result == "not_grounded" and regeneration_count <= 1:
        return "regenerate"

    return "finish"


def build_graph() -> CompiledStateGraph:
    """
    Build and compile the agentic RAG graph.

    Graph structure:
        START
          │
          ▼
        [router]
          │
          ├── reject ──────────────────────────────────► END
          │
          └── retrieve
                │
                ▼
            [retriever]
                │
                ▼
            [grader]
                │
                ├── rewrite ──► [rewriter] ──► [retriever] (loop)
                │
                └── generate
                      │
                      ▼
                  [generator]
                      │
                      ▼
              [hallucination_check]
                      │
                      ├── regenerate ──► [generator] (retry once)
                      │
                      └── finish ──────────────────────► END

    Returns:
        Compiled LangGraph ready for execution
    """
    # Initialize graph with state schema
    workflow = StateGraph(GraphState)

    # Add nodes with timing instrumentation
    workflow.add_node("router", create_timed_node(router_node, "router"))
    workflow.add_node("retriever", create_timed_node(retriever_node, "retriever"))
    workflow.add_node("grader", create_timed_node(grader_node, "grader"))
    workflow.add_node("rewriter", create_timed_node(rewriter_node, "rewriter"))
    workflow.add_node("generator", create_timed_node(generator_node, "generator"))
    workflow.add_node(
        "hallucination_check", create_timed_node(hallucination_check_node, "hallucination_check")
    )

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

    # Retriever always goes to grader
    workflow.add_edge("retriever", "grader")

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


def run_query(question: str, max_rewrites: int | None = None) -> GraphState:
    """
    Execute a query through the agentic RAG pipeline.

    Args:
        question: User's natural language question
        max_rewrites: Per-request override for max rewrites. Defaults to settings value.

    Returns:
        Final graph state with answer, citations, and metadata
    """
    # Create initial state
    state = create_initial_state(question)
    if max_rewrites is not None:
        state["max_rewrites_override"] = max_rewrites

    # Use cached compiled graph (compiled once per process)
    graph = _get_compiled_graph()

    start_time = time.perf_counter()
    final_state = graph.invoke(state)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Add timing metadata
    final_state["processing_time_ms"] = elapsed_ms

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


def save_graph_image(path: str = "docs/architecture.png") -> None:
    """
    Save graph visualization as PNG image.

    Requires graphviz to be installed.

    Args:
        path: Output path for the PNG file
    """
    graph = build_graph()
    graph.get_graph().draw_png(path)
