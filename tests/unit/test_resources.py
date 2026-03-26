"""Tests for LanceDB Store + fastembed resource singletons."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_get_store_returns_store_instance():
    from specagent.retrieval.resources import clear_resource_cache, get_store

    clear_resource_cache()
    with patch("specagent.retrieval.resources.Store") as mock_cls:
        mock_cls.return_value = MagicMock()
        store = get_store()
    assert store is mock_cls.return_value


@pytest.mark.unit
def test_get_store_is_cached():
    from specagent.retrieval.resources import clear_resource_cache, get_store

    clear_resource_cache()
    with patch("specagent.retrieval.resources.Store") as mock_cls:
        mock_cls.return_value = MagicMock()
        s1 = get_store()
        s2 = get_store()
    assert s1 is s2
    mock_cls.assert_called_once()


@pytest.mark.unit
def test_get_embedder_returns_text_embedding():
    from specagent.retrieval.resources import clear_resource_cache, get_embedder

    clear_resource_cache()
    with patch("specagent.retrieval.resources.TextEmbedding") as mock_cls:
        mock_cls.return_value = MagicMock()
        result = get_embedder()
    assert result is mock_cls.return_value


@pytest.mark.unit
def test_clear_resource_cache_resets_singletons():
    from specagent.retrieval.resources import clear_resource_cache, get_store

    with patch("specagent.retrieval.resources.Store") as mock_cls:
        instances = [MagicMock(), MagicMock()]
        mock_cls.side_effect = instances
        clear_resource_cache()
        s1 = get_store()
        clear_resource_cache()
        s2 = get_store()
    assert s1 is not s2


@pytest.mark.unit
def test_initialize_resources_success():
    """initialize_resources() returns {store: True, embedder: True} on success."""
    from specagent.retrieval.resources import clear_resource_cache, initialize_resources

    clear_resource_cache()
    with (
        patch("specagent.retrieval.resources.Store"),
        patch("specagent.retrieval.resources.TextEmbedding"),
    ):
        result = initialize_resources()
    assert result == {"store": True, "embedder": True}


@pytest.mark.unit
def test_initialize_resources_store_failure_raises_runtime_error():
    """initialize_resources() raises RuntimeError when Store() fails."""
    from specagent.retrieval.resources import clear_resource_cache, initialize_resources

    clear_resource_cache()
    with patch("specagent.retrieval.resources.Store", side_effect=OSError("db gone")):
        with pytest.raises(RuntimeError, match="Failed to open LanceDB store"):
            initialize_resources()


@pytest.mark.unit
def test_initialize_resources_embedder_failure_raises_runtime_error():
    """initialize_resources() raises RuntimeError when TextEmbedding() fails."""
    from specagent.retrieval.resources import clear_resource_cache, initialize_resources

    clear_resource_cache()
    with (
        patch("specagent.retrieval.resources.Store"),
        patch("specagent.retrieval.resources.TextEmbedding", side_effect=OSError("model missing")),
    ):
        with pytest.raises(RuntimeError, match="Failed to load embedding model"):
            initialize_resources()
