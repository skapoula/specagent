"""Unit tests for LangSmith tracing integration."""
import pytest


@pytest.mark.unit
def test_langsmith_importable():
    """langsmith must be importable as a core dependency (not optional)."""
    import langsmith  # noqa: PLC0415

    assert langsmith is not None
