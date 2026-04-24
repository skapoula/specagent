"""Unit tests for specagent.evaluation._trace module."""

import logging

import pytest


@pytest.mark.unit
class TestSetupTraceLogging:
    def test_returns_logger(self, tmp_path):
        from specagent.evaluation._trace import setup_trace_logging

        trace = setup_trace_logging(tmp_path, "2024-01-01T12:00:00")
        assert isinstance(trace, logging.Logger)
        for handler in trace.handlers[:]:
            handler.close()
            trace.removeHandler(handler)

    def test_creates_log_file(self, tmp_path):
        from specagent.evaluation._trace import setup_trace_logging

        trace = setup_trace_logging(tmp_path, "2024-01-01T12:00:00")
        trace.info("test message")
        log_files = list(tmp_path.glob("benchmark_trace_*.log"))
        assert len(log_files) == 1
        assert "test message" in log_files[0].read_text()
        for handler in trace.handlers[:]:
            handler.close()
            trace.removeHandler(handler)

    def test_no_propagation(self, tmp_path):
        from specagent.evaluation._trace import setup_trace_logging

        trace = setup_trace_logging(tmp_path, "2024-01-01T12:00:00")
        assert trace.propagate is False
        for handler in trace.handlers[:]:
            handler.close()
            trace.removeHandler(handler)

    def test_verbose_adds_console_handler(self, tmp_path):
        from specagent.evaluation._trace import setup_trace_logging

        trace = setup_trace_logging(tmp_path, "2024-01-01T12:00:00", verbose=True)
        handler_types = [type(h) for h in trace.handlers]
        assert logging.StreamHandler in handler_types
        for handler in trace.handlers[:]:
            handler.close()
            trace.removeHandler(handler)
