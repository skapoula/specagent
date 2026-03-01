"""Tests for LanceDB Store + fastembed resource singletons."""
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
def test_get_store_returns_store_instance():
    from specagent.retrieval.resources import get_store, clear_resource_cache
    clear_resource_cache()
    with patch("specagent.retrieval.resources.Store") as mock_cls:
        mock_cls.return_value = MagicMock()
        store = get_store()
    assert store is mock_cls.return_value


@pytest.mark.unit
def test_get_store_is_cached():
    from specagent.retrieval.resources import get_store, clear_resource_cache
    clear_resource_cache()
    with patch("specagent.retrieval.resources.Store") as mock_cls:
        mock_cls.return_value = MagicMock()
        s1 = get_store()
        s2 = get_store()
    assert s1 is s2
    mock_cls.assert_called_once()


@pytest.mark.unit
def test_get_embedder_returns_text_embedding():
    from specagent.retrieval.resources import get_embedder, clear_resource_cache
    clear_resource_cache()
    with patch("specagent.retrieval.resources.TextEmbedding") as mock_cls:
        mock_cls.return_value = MagicMock()
        result = get_embedder()
    assert result is mock_cls.return_value


@pytest.mark.unit
def test_clear_resource_cache_resets_singletons():
    from specagent.retrieval.resources import get_store, clear_resource_cache
    with patch("specagent.retrieval.resources.Store") as mock_cls:
        instances = [MagicMock(), MagicMock()]
        mock_cls.side_effect = instances
        clear_resource_cache()
        s1 = get_store()
        clear_resource_cache()
        s2 = get_store()
    assert s1 is not s2
