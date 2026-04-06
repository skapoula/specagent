"""
LangSmith tracing integration.

Configures LangSmith tracing for the specagent pipeline. When enabled and
LANGCHAIN_API_KEY is set, all LangChain/LangGraph calls (Groq backend) are
automatically traced via LangChain's callback system. The custom_endpoint
backend is covered by the @traceable decorator on CustomEndpointLLM.

Usage:
    from specagent.tracing import setup_langsmith_tracing
    setup_langsmith_tracing()  # Call once at application startup
"""

import importlib.util
import logging
import os
import warnings

from specagent.config import settings

logger = logging.getLogger(__name__)


def setup_langsmith_tracing() -> None:
    """
    Configure LangSmith tracing.

    Checks settings.enable_langsmith and settings.langchain_api_key.
    If both are present, sets the standard LangSmith env vars that the SDK
    reads automatically (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT). Safe to call multiple times — idempotent via
    os.environ.setdefault.

    Warnings are emitted (not exceptions) so a missing key never crashes
    the application.
    """
    if not settings.enable_langsmith:
        logger.debug("LangSmith tracing disabled (ENABLE_LANGSMITH=false)")
        return

    if importlib.util.find_spec("langsmith") is None:
        warnings.warn(
            "LangSmith not installed. Install with: pip install langsmith",
            stacklevel=2,
        )
        return

    if not settings.langchain_api_key:
        warnings.warn(
            "LANGCHAIN_API_KEY is not set. LangSmith tracing will not be active. "
            "Set LANGCHAIN_API_KEY in your .env file to enable tracing.",
            stacklevel=2,
        )
        return

    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)

    logger.info(
        "LangSmith tracing enabled (project=%s)",
        os.environ["LANGCHAIN_PROJECT"],
    )
