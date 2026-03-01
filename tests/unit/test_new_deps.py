"""Verify new pipeline dependencies are importable after pyproject.toml update."""
import pytest


@pytest.mark.unit
def test_lancedb_importable():
    import lancedb  # noqa: F401


@pytest.mark.unit
def test_fastembed_importable():
    import fastembed  # noqa: F401


@pytest.mark.unit
def test_pyarrow_importable():
    import pyarrow  # noqa: F401
