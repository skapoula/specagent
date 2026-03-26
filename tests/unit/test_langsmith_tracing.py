"""Unit tests for LangSmith tracing integration."""
import pytest


@pytest.mark.unit
def test_langsmith_importable():
    """langsmith must be importable as a core dependency (not optional)."""
    import langsmith  # noqa: F401, PLC0415
