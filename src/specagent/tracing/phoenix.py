"""
Arize Phoenix tracing integration.

Sets up OpenTelemetry tracing for observability of the RAG pipeline.
Traces are sent to a local Phoenix instance for visualization.

Usage:
    from specagent.tracing import setup_tracing
    setup_tracing()  # Call once at application startup
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from specagent.config import settings

F = TypeVar("F", bound=Callable[..., Any])


def setup_tracing() -> None:
    """
    Initialize Phoenix tracing with OpenTelemetry.

    Should be called once at application startup before any
    LangChain/LangGraph operations.

    Requires Phoenix server running at settings.phoenix_endpoint.

    Example:
        # Start Phoenix server first:
        # phoenix serve

        from specagent.tracing import setup_tracing
        setup_tracing()
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
    """
    Decorator to add tracing span to a function.

    Args:
        name: Span name (defaults to function name)

    Returns:
        Decorated function with tracing

    Example:
        @traced("my_custom_span")
        def my_function():
            ...
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
    """
    Add attributes to the current span.

    Args:
        **attributes: Key-value pairs to add as span attributes

    Example:
        add_span_attributes(
            query="What is NR?",
            chunks_retrieved=10,
        )
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
