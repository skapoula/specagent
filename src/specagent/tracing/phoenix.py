"""Arize Phoenix tracing integration.

Sets up OpenTelemetry tracing for observability of the RAG pipeline.
Traces are sent to a local Phoenix instance for visualization.
Call ``setup_tracing()`` once at application startup.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from specagent.config import settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def setup_tracing() -> None:
    """Initialize Phoenix tracing with OpenTelemetry.

    Should be called once at application startup before any
    LangChain/LangGraph operations. Requires Phoenix server running at
    ``settings.phoenix_endpoint``. No-op when ``settings.enable_tracing``
    is False.
    """
    if not settings.enable_tracing:
        return

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        # Register tracer provider with Phoenix
        tracer_provider = register(
            project_name="3gpp-specagent",
            endpoint=f"{settings.phoenix_endpoint}/v1/traces",
        )

        # Instrument LangChain for automatic tracing
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    except ImportError:
        import warnings

        warnings.warn(
            "Phoenix tracing dependencies not installed. Install with: pip install specagent[eval]"
        )
    except Exception as e:
        import warnings

        warnings.warn(f"Failed to setup tracing: {e}")


def traced(name: str | None = None) -> Callable[[F], F]:
    """Decorator that wraps a function in an OpenTelemetry span.

    Args:
        name: Span name. Defaults to the decorated function's ``__name__``.

    Returns:
        Decorated function with an OTel span around each call.
        No-op when ``settings.enable_tracing`` is False or OpenTelemetry is
        not installed.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not settings.enable_tracing:
                return func(*args, **kwargs)

            try:
                from opentelemetry import trace

                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(span_name) as span:
                    # Add function arguments as span attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)

                    result = func(*args, **kwargs)

                    # Add result type as attribute
                    span.set_attribute("result.type", type(result).__name__)

                    return result

            except ImportError:
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def add_span_attributes(**attributes: Any) -> None:
    """Add key-value attributes to the currently active OTel span.

    Args:
        **attributes: Attribute names and values. Only ``str``, ``int``,
            ``float``, and ``bool`` values are passed through directly;
            all others are coerced to ``str``.
    """
    if not settings.enable_tracing:
        return

    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        for key, value in attributes.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, str(value))

    except ImportError:
        pass


def record_exception(exception: Exception) -> None:
    """
    Record an exception in the current span.

    Args:
        exception: Exception to record
    """
    if not settings.enable_tracing:
        return

    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR))

    except ImportError:
        pass


def create_phoenix_node_wrapper(
    node_func: "Callable[[Any], Any]",
    node_name: str,
) -> "Callable[[Any], Any]":
    """Wrap a LangGraph node function in a named OpenTelemetry span.

    Creates a child span for every invocation. When tracing is disabled
    or opentelemetry is not installed, the node runs unwrapped.

    Args:
        node_func: The original node function accepting GraphState.
        node_name: Span name shown in Phoenix (e.g. 'retriever', 'generator').

    Returns:
        Wrapped node function that emits a Phoenix span on each call.
    """
    if not settings.enable_tracing:
        return node_func

    @functools.wraps(node_func)
    def wrapped(state: Any) -> Any:
        try:
            from opentelemetry import trace  # noqa: PLC0415

            tracer = trace.get_tracer("specagent.nodes")
            with tracer.start_as_current_span(node_name) as span:
                span.set_attribute("node.name", node_name)
                span.set_attribute(
                    "session.id", state.get("trace_id", "") if isinstance(state, dict) else ""
                )
                result = node_func(state)
                if isinstance(result, dict) and result.get("error"):
                    span.set_attribute("node.error", str(result["error"]))
                return result
        except ImportError:
            return node_func(state)
        except Exception:
            logger.debug("Phoenix node wrapper failed for %s", node_name, exc_info=True)
            return node_func(state)

    return wrapped
