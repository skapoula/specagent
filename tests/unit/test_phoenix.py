"""
Unit tests for Phoenix tracing integration.

Tests cover:
    - setup_tracing() initialization
    - @traced decorator functionality
    - add_span_attributes() helper
    - record_exception() helper
    - Graceful degradation when dependencies missing
"""

# ruff: noqa: PLC0415
# Note: We intentionally import modules inside test functions to control when
# they are imported relative to mocking, which is necessary for testing modules
# with conditional imports like phoenix.py

import sys
import warnings
from importlib import reload
from unittest.mock import MagicMock, Mock, patch

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_phoenix_modules():
    """Mock Phoenix and OpenTelemetry modules for testing."""
    # Create mock modules
    mock_otel = MagicMock()
    mock_register = MagicMock()
    mock_otel.register = mock_register

    mock_langchain_instrumentor = MagicMock()
    mock_langchain_module = MagicMock()
    mock_langchain_module.LangChainInstrumentor = mock_langchain_instrumentor

    mock_trace = MagicMock()
    mock_trace.get_tracer = MagicMock()
    mock_trace.get_current_span = MagicMock()
    mock_trace.Status = MagicMock()
    mock_trace.StatusCode = MagicMock()

    # Create opentelemetry module and set trace as an attribute
    mock_opentelemetry = MagicMock()
    mock_opentelemetry.trace = mock_trace

    # Create openinference modules
    mock_openinference = MagicMock()
    mock_instrumentation = MagicMock()
    mock_instrumentation.langchain = mock_langchain_module
    mock_openinference.instrumentation = mock_instrumentation

    # Create phoenix module
    mock_phoenix = MagicMock()
    mock_phoenix.otel = mock_otel

    # Install mocks into sys.modules
    sys.modules["phoenix"] = mock_phoenix
    sys.modules["phoenix.otel"] = mock_otel
    sys.modules["openinference"] = mock_openinference
    sys.modules["openinference.instrumentation"] = mock_instrumentation
    sys.modules["openinference.instrumentation.langchain"] = mock_langchain_module
    sys.modules["opentelemetry"] = mock_opentelemetry
    sys.modules["opentelemetry.trace"] = mock_trace

    yield {
        "register": mock_register,
        "LangChainInstrumentor": mock_langchain_instrumentor,
        "trace": mock_trace,
    }

    # Clean up
    for key in list(sys.modules.keys()):
        if (
            key.startswith("phoenix")
            or key.startswith("openinference")
            or key.startswith("opentelemetry")
        ):
            sys.modules.pop(key, None)


# =============================================================================
# setup_tracing() Tests
# =============================================================================


@pytest.mark.unit
def test_setup_tracing_disabled():
    """Test that setup_tracing does nothing when tracing is disabled."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = False

        # Import after mocking config
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        phoenix_module.setup_tracing()
        # Should return early without errors


@pytest.mark.unit
def test_setup_tracing_success(mock_phoenix_modules):
    """Test successful tracing setup with Phoenix."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True
        mock_config.phoenix_endpoint = "http://localhost:6006"

        mock_tracer_provider = MagicMock()
        mock_phoenix_modules["register"].return_value = mock_tracer_provider

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        phoenix_module.setup_tracing()

        # Verify Phoenix was registered
        mock_phoenix_modules["register"].assert_called_once_with(
            project_name="3gpp-specagent",
            endpoint="http://localhost:6006/v1/traces",
        )

        # Verify LangChain instrumentation
        mock_phoenix_modules["LangChainInstrumentor"].assert_called_once()
        mock_phoenix_modules["LangChainInstrumentor"]().instrument.assert_called_once_with(
            tracer_provider=mock_tracer_provider
        )


@pytest.mark.unit
def test_setup_tracing_missing_dependencies():
    """Test graceful handling when Phoenix is not installed."""
    import specagent.tracing.phoenix as phoenix_module

    # Setting a sys.modules entry to None causes any `import` of that name to
    # raise ImportError — the standard way to simulate an absent optional package.
    absent: dict = {
        "phoenix.otel": None,
        "openinference.instrumentation.langchain": None,
    }

    with (
        patch("specagent.config.settings") as mock_config,
        patch.dict(sys.modules, absent),
        warnings.catch_warnings(record=True) as w,
    ):
        mock_config.enable_tracing = True
        warnings.simplefilter("always")
        phoenix_module.setup_tracing()

        phoenix_warnings = [
            warning
            for warning in w
            if "Phoenix tracing dependencies not installed" in str(warning.message)
        ]

        assert len(phoenix_warnings) == 1
        assert "Phoenix tracing dependencies not installed" in str(phoenix_warnings[0].message)


@pytest.mark.unit
def test_setup_tracing_initialization_error(mock_phoenix_modules):
    """Test graceful handling of initialization errors."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        mock_phoenix_modules["register"].side_effect = RuntimeError("Connection failed")

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            phoenix_module.setup_tracing()

            # Should warn about setup failure
            assert len(w) == 1
            assert "Failed to setup tracing" in str(w[0].message)
            assert "Connection failed" in str(w[0].message)


# =============================================================================
# @traced Decorator Tests
# =============================================================================


@pytest.mark.unit
def test_traced_decorator_disabled():
    """Test that @traced has no effect when tracing is disabled."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = False

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        @phoenix_module.traced()
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10


@pytest.mark.unit
def test_traced_decorator_with_default_name(mock_phoenix_modules):
    """Test @traced decorator with default function name."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        mock_span = MagicMock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mock_phoenix_modules["trace"].get_tracer.return_value = mock_tracer

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        @phoenix_module.traced()
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)

        # Verify span was created with function name
        mock_tracer.start_as_current_span.assert_called_once_with("my_function")

        # Verify attributes were set
        assert mock_span.set_attribute.call_count >= 2
        mock_span.set_attribute.assert_any_call("function.name", "my_function")

        assert result == 10


@pytest.mark.unit
def test_traced_decorator_with_custom_name(mock_phoenix_modules):
    """Test @traced decorator with custom span name."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        mock_span = MagicMock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mock_phoenix_modules["trace"].get_tracer.return_value = mock_tracer

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        @phoenix_module.traced("custom_operation")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)

        # Verify span was created with custom name
        mock_tracer.start_as_current_span.assert_called_once_with("custom_operation")
        assert result == 10


@pytest.mark.unit
def test_traced_decorator_missing_dependencies():
    """Test @traced gracefully handles missing OpenTelemetry."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        # Import and reload without mocking dependencies
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        @phoenix_module.traced()
        def my_function(x: int) -> int:
            return x * 2

        # Should still work without tracing
        result = my_function(5)
        assert result == 10


@pytest.mark.unit
def test_traced_decorator_import_error_during_execution():
    """Test @traced handles ImportError during function execution (line 103)."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        # Create decorator before removing the module
        @phoenix_module.traced()
        def my_function(x: int) -> int:
            return x * 2

        # Now remove opentelemetry from sys.modules to trigger ImportError
        # when the wrapper tries to import it during execution
        import sys

        original_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith("opentelemetry"):
                original_modules[key] = sys.modules.pop(key)

        # Should fall back gracefully on ImportError during execution
        result = my_function(5)
        assert result == 10

        # Restore modules
        sys.modules.update(original_modules)


@pytest.mark.unit
def test_traced_preserves_function_metadata(mock_phoenix_modules):
    """Test that @traced preserves function name and docstring."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        @phoenix_module.traced()
        def my_function(x: int) -> int:
            """This is my docstring."""
            return x * 2

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "This is my docstring."


# =============================================================================
# add_span_attributes() Tests
# =============================================================================


@pytest.mark.unit
def test_add_span_attributes_disabled():
    """Test that add_span_attributes does nothing when tracing disabled."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = False

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        # Should not raise any errors
        phoenix_module.add_span_attributes(key="value", count=10)


@pytest.mark.unit
def test_add_span_attributes_with_primitives(mock_phoenix_modules):
    """Test adding primitive type attributes to span."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        mock_span = MagicMock()
        mock_phoenix_modules["trace"].get_current_span.return_value = mock_span

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        phoenix_module.add_span_attributes(
            query="test query",
            count=10,
            score=0.85,
            enabled=True,
        )

        # Verify attributes were set
        assert mock_span.set_attribute.call_count == 4
        mock_span.set_attribute.assert_any_call("query", "test query")
        mock_span.set_attribute.assert_any_call("count", 10)
        mock_span.set_attribute.assert_any_call("score", 0.85)
        mock_span.set_attribute.assert_any_call("enabled", True)


@pytest.mark.unit
def test_add_span_attributes_with_complex_types(mock_phoenix_modules):
    """Test adding complex types converts them to strings."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        mock_span = MagicMock()
        mock_phoenix_modules["trace"].get_current_span.return_value = mock_span

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        phoenix_module.add_span_attributes(
            chunks=[{"id": 1}, {"id": 2}],
            metadata={"key": "value"},
        )

        # Verify complex types were converted to strings
        assert mock_span.set_attribute.call_count == 2
        calls = [call[0] for call in mock_span.set_attribute.call_args_list]

        # Check that both attributes were set with string values
        assert any(call[0] == "chunks" and isinstance(call[1], str) for call in calls)
        assert any(call[0] == "metadata" and isinstance(call[1], str) for call in calls)


@pytest.mark.unit
def test_add_span_attributes_missing_dependencies():
    """Test graceful handling when OpenTelemetry not available."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        # Import and reload without mocking dependencies
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        # Should not raise errors
        phoenix_module.add_span_attributes(key="value")


@pytest.mark.unit
def test_add_span_attributes_import_error_during_execution():
    """Test add_span_attributes handles ImportError during execution (lines 136-137)."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        # Remove opentelemetry from sys.modules to trigger ImportError during execution
        import sys

        original_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith("opentelemetry"):
                original_modules[key] = sys.modules.pop(key)

        # Should not raise errors even with ImportError during execution
        phoenix_module.add_span_attributes(key="value")

        # Restore modules
        sys.modules.update(original_modules)


# =============================================================================
# record_exception() Tests
# =============================================================================


@pytest.mark.unit
def test_record_exception_disabled():
    """Test that record_exception does nothing when tracing disabled."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = False

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        # Should not raise any errors
        phoenix_module.record_exception(ValueError("test error"))


@pytest.mark.unit
def test_record_exception_success(mock_phoenix_modules):
    """Test recording exception in current span."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        mock_span = MagicMock()
        mock_phoenix_modules["trace"].get_current_span.return_value = mock_span

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        test_exception = ValueError("test error")
        phoenix_module.record_exception(test_exception)

        # Verify exception was recorded
        mock_span.record_exception.assert_called_once_with(test_exception)

        # Verify span status was set to error
        mock_span.set_status.assert_called_once()


@pytest.mark.unit
def test_record_exception_missing_dependencies():
    """Test graceful handling when OpenTelemetry not available."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        # Import and reload without mocking dependencies
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        # Should not raise errors
        phoenix_module.record_exception(ValueError("test error"))


@pytest.mark.unit
def test_record_exception_import_error_during_execution():
    """Test record_exception handles ImportError during execution (lines 157-158)."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        # Remove opentelemetry from sys.modules to trigger ImportError during execution
        import sys

        original_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith("opentelemetry"):
                original_modules[key] = sys.modules.pop(key)

        # Should not raise errors even with ImportError during execution
        phoenix_module.record_exception(ValueError("test error"))

        # Restore modules
        sys.modules.update(original_modules)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_traced_decorator_integration_with_span_attributes(mock_phoenix_modules):
    """Test @traced decorator with add_span_attributes in function body."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        mock_span = MagicMock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mock_phoenix_modules["trace"].get_tracer.return_value = mock_tracer
        mock_phoenix_modules["trace"].get_current_span.return_value = mock_span

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        @phoenix_module.traced("process_query")
        def process_query(query: str) -> str:
            phoenix_module.add_span_attributes(query=query, query_length=len(query))
            return f"Processed: {query}"

        result = process_query("test query")

        # Verify span was created
        mock_tracer.start_as_current_span.assert_called_once_with("process_query")

        # Verify custom attributes were added
        mock_span.set_attribute.assert_any_call("query", "test query")
        mock_span.set_attribute.assert_any_call("query_length", 10)

        assert result == "Processed: test query"


@pytest.mark.unit
def test_traced_decorator_with_exception(mock_phoenix_modules):
    """Test @traced decorator when function raises exception."""
    with patch("specagent.config.settings") as mock_config:
        mock_config.enable_tracing = True

        mock_span = MagicMock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mock_phoenix_modules["trace"].get_tracer.return_value = mock_tracer

        # Import and reload after mocking
        import specagent.tracing.phoenix as phoenix_module

        reload(phoenix_module)

        @phoenix_module.traced()
        def failing_function():
            raise ValueError("test error")

        # Exception should propagate
        with pytest.raises(ValueError, match="test error"):
            failing_function()

        # Span should still be created
        mock_tracer.start_as_current_span.assert_called_once()


@pytest.mark.unit
def test_traced_import_error_at_call_time_fallback():
    """traced wrapper falls back to func() when opentelemetry raises ImportError at call time (line 103)."""
    import specagent.tracing.phoenix as phoenix_module

    with patch("specagent.tracing.phoenix.settings") as ms:
        ms.enable_tracing = True

        @phoenix_module.traced()
        def fn(x):
            return x + 1

        # Force ImportError at call time by setting opentelemetry to None in sys.modules
        with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
            result = fn(2)
        assert result == 3


@pytest.mark.unit
def test_add_span_attributes_import_error_at_call_time():
    """add_span_attributes silently passes when opentelemetry raises ImportError at call time (lines 136-137)."""
    import specagent.tracing.phoenix as phoenix_module

    with patch("specagent.tracing.phoenix.settings") as ms:
        ms.enable_tracing = True
        with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
            phoenix_module.add_span_attributes(key="value")


@pytest.mark.unit
def test_record_exception_import_error_at_call_time():
    """record_exception silently passes when opentelemetry raises ImportError at call time (lines 157-158)."""
    import specagent.tracing.phoenix as phoenix_module

    with patch("specagent.tracing.phoenix.settings") as ms:
        ms.enable_tracing = True
        with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
            phoenix_module.record_exception(ValueError("test"))
