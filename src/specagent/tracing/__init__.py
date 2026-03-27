"""
Observability and tracing integrations.

Provides OpenTelemetry-based tracing via Arize Phoenix and
LangSmith tracing for the LangGraph pipeline.
"""

from specagent.tracing.langsmith import setup_langsmith_tracing
from specagent.tracing.phoenix import create_phoenix_node_wrapper, setup_tracing, traced
from specagent.tracing.rag_spans import emit_llm_usage_span, emit_query_span, emit_retrieval_span

__all__ = [
    "create_phoenix_node_wrapper",
    "emit_llm_usage_span",
    "emit_query_span",
    "emit_retrieval_span",
    "setup_langsmith_tracing",
    "setup_tracing",
    "traced",
]
