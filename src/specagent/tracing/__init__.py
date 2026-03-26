"""
Observability and tracing integrations.

Provides OpenTelemetry-based tracing via Arize Phoenix and
LangSmith tracing for the LangGraph pipeline.
"""

from specagent.tracing.langsmith import setup_langsmith_tracing
from specagent.tracing.phoenix import setup_tracing, traced

__all__ = ["setup_langsmith_tracing", "setup_tracing", "traced"]
