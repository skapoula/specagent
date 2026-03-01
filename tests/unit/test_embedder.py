"""Tests for the fastembed embedder wrapper."""
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
def test_get_embedder_returns_singleton():
    """get_embedder() returns the same instance on repeated calls."""
    mock_te = MagicMock()
    with patch("specagent.retrieval.embedder.TextEmbedding", return_value=mock_te):
        from specagent.retrieval import embedder as emb_mod

        emb_mod._embedder = None  # reset singleton
        r1 = emb_mod.get_embedder()
        r2 = emb_mod.get_embedder()
    assert r1 is r2


@pytest.mark.unit
def test_embed_documents_prepends_document_prefix():
    """embed_documents() prepends 'search_document: ' to each text."""
    mock_te = MagicMock()
    mock_te.embed.return_value = iter([[0.1] * 768])

    with patch("specagent.retrieval.embedder.TextEmbedding", return_value=mock_te):
        from specagent.retrieval import embedder as emb_mod

        emb_mod._embedder = None
        emb_mod.embed_documents(["hello"])

    call_args = list(mock_te.embed.call_args[0][0])
    assert call_args == ["search_document: hello"]


@pytest.mark.unit
def test_embed_query_prepends_query_prefix():
    """embed_query() prepends 'search_query: ' to the text."""
    mock_te = MagicMock()
    mock_te.embed.return_value = iter([[0.1] * 768])

    with patch("specagent.retrieval.embedder.TextEmbedding", return_value=mock_te):
        from specagent.retrieval import embedder as emb_mod

        emb_mod._embedder = None
        emb_mod.embed_query("hello")

    call_args = list(mock_te.embed.call_args[0][0])
    assert call_args == ["search_query: hello"]
