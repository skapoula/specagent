"""Per-query monitoring report: aggregates metrics from GraphState after a run.

Produces a structured summary suitable for printing to stderr or logging.
Never writes to stdout.
"""

import logging
import statistics
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QueryReport:
    """Aggregated metrics for a single completed query run."""

    trace_id: str
    question: str
    route_decision: str
    rewrite_count: int
    chunks_retrieved: int
    chunks_relevant: int
    hallucination_check: str | None
    total_ms: float
    retrieval_embed_ms_list: list[float] = field(default_factory=list)
    retrieval_search_ms_list: list[float] = field(default_factory=list)
    node_timings: dict[str, float] = field(default_factory=dict)
    mean_similarity_scores: list[float] = field(default_factory=list)
    top_similarity_scores: list[float] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    llm_call_count: int = 0
    llm_calls_by_node: dict[str, int] = field(default_factory=dict)


def build_query_report(state: dict) -> QueryReport:
    """Build a QueryReport from a completed GraphState dict.

    Args:
        state: Final GraphState dict returned by run_query().

    Returns:
        QueryReport populated from all available state fields.
    """
    graded = state.get("graded_chunks", [])
    relevant_count = sum(1 for g in graded if getattr(g, "relevant", None) == "yes")

    retrieval_events = state.get("retrieval_events", [])
    embed_ms_list = [r.embed_ms for r in retrieval_events]
    search_ms_list = [r.search_ms for r in retrieval_events]
    mean_sims = [r.mean_similarity for r in retrieval_events if r.mean_similarity is not None]
    top_sims = [r.top_similarity for r in retrieval_events if r.top_similarity is not None]

    llm_calls = state.get("llm_calls", [])
    total_prompt = sum(c.prompt_tokens or 0 for c in llm_calls)
    total_completion = sum(c.completion_tokens or 0 for c in llm_calls)

    calls_by_node: dict[str, int] = {}
    for c in llm_calls:
        calls_by_node[c.node] = calls_by_node.get(c.node, 0) + 1

    return QueryReport(
        trace_id=state.get("trace_id", ""),
        question=state.get("question", ""),
        route_decision=state.get("route_decision", "unknown"),
        rewrite_count=state.get("rewrite_count", 0),
        chunks_retrieved=len(state.get("retrieved_chunks", [])),
        chunks_relevant=relevant_count,
        hallucination_check=state.get("hallucination_check"),
        total_ms=state.get("processing_time_ms", 0.0),
        retrieval_embed_ms_list=embed_ms_list,
        retrieval_search_ms_list=search_ms_list,
        node_timings=state.get("node_timings", {}),
        mean_similarity_scores=mean_sims,
        top_similarity_scores=top_sims,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        llm_call_count=len(llm_calls),
        llm_calls_by_node=calls_by_node,
    )


def format_report(report: QueryReport) -> str:
    """Format a QueryReport as a human-readable text block.

    Args:
        report: Populated QueryReport dataclass.

    Returns:
        Multi-line string suitable for logging or stderr output.
    """
    lines = [
        "=== SpecAgent Query Report ===",
        f"  Trace ID       : {report.trace_id}",
        f"  Route          : {report.route_decision}",
        f"  Rewrites       : {report.rewrite_count}",
        f"  Chunks         : {report.chunks_retrieved} retrieved / {report.chunks_relevant} relevant",
        f"  Hallucination  : {report.hallucination_check or 'skipped'}",
        f"  Total latency  : {report.total_ms:.0f} ms",
    ]

    if report.node_timings:
        lines.append("  Node timings   :")
        for node, ms in sorted(report.node_timings.items(), key=lambda x: -x[1]):
            lines.append(f"    {node:<24} {ms:.0f} ms")

    if report.retrieval_embed_ms_list:
        avg_embed = statistics.mean(report.retrieval_embed_ms_list)
        avg_search = statistics.mean(report.retrieval_search_ms_list)
        lines.append(f"  Embed latency  : {avg_embed:.1f} ms avg")
        lines.append(f"  Search latency : {avg_search:.1f} ms avg")

    if report.mean_similarity_scores:
        avg_sim = statistics.mean(report.mean_similarity_scores)
        max_sim = max(report.top_similarity_scores) if report.top_similarity_scores else 0.0
        lines.append(f"  Similarity     : mean={avg_sim:.3f}  top={max_sim:.3f}")

    if report.llm_call_count > 0:
        lines.append(
            f"  LLM calls      : {report.llm_call_count}"
            f" ({report.total_prompt_tokens} prompt"
            f" / {report.total_completion_tokens} completion tokens)"
        )
        for node, count in sorted(report.llm_calls_by_node.items()):
            lines.append(f"    {node:<24} {count} call(s)")

    lines.append("==============================")
    return "\n".join(lines)


def log_report(report: QueryReport) -> None:
    """Log the formatted report at INFO level.

    Args:
        report: Populated QueryReport to log.
    """
    logger.info("\n%s", format_report(report))
