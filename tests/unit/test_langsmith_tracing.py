"""Unit tests for LangSmith tracing integration."""
import pytest


@pytest.mark.unit
def test_langsmith_importable():
    """langsmith must be importable as a core dependency (not optional)."""
    import langsmith  # noqa: PLC0415 — intentional in-function import so ImportError surfaces as a test failure

    assert langsmith is not None
