"""Tests for the fastembed embedder wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from specagent.retrieval.exceptions import EmbeddingError


@pytest.mark.unit
def test_embed_documents_prepends_document_prefix():
    """embed_documents() prepends 'search_document: ' to each text."""
    mock_te = MagicMock()
    mock_te.embed.return_value = iter([[0.1] * 768])

    with patch("specagent.retrieval.embedder.get_embedder", return_value=mock_te):
        from specagent.retrieval import embedder as emb_mod

        emb_mod.embed_documents(["hello"])

    call_args = list(mock_te.embed.call_args[0][0])
    assert call_args == ["search_document: hello"]


@pytest.mark.unit
def test_embed_query_prepends_query_prefix():
    """embed_query() prepends 'search_query: ' to the text."""
    mock_te = MagicMock()
    mock_te.embed.return_value = iter([[0.1] * 768])

    with patch("specagent.retrieval.embedder.get_embedder", return_value=mock_te):
        from specagent.retrieval import embedder as emb_mod

        emb_mod.embed_query("hello")

    call_args = list(mock_te.embed.call_args[0][0])
    assert call_args == ["search_query: hello"]


@pytest.mark.unit
def test_embed_documents_empty_list_returns_empty_array():
    """embed_documents([]) returns shape (0, embedding_dimension) without calling model."""
    from specagent.retrieval import embedder as emb_mod

    mock_te = MagicMock()
    with patch("specagent.retrieval.embedder.get_embedder", return_value=mock_te):
        result = emb_mod.embed_documents([])
    assert result.shape[0] == 0
    mock_te.embed.assert_not_called()


@pytest.mark.unit
def test_embed_documents_vector_count_mismatch_raises_embedding_error():
    """embed_documents() raises EmbeddingError when model returns wrong vector count."""
    from specagent.retrieval import embedder as emb_mod

    mock_te = MagicMock()
    mock_te.embed.return_value = iter([[0.1] * 768, [0.2] * 768])  # 2 vecs for 1 text
    with patch("specagent.retrieval.embedder.get_embedder", return_value=mock_te):
        with pytest.raises(EmbeddingError, match="returned 2 vectors"):
            emb_mod.embed_documents(["only one text"])


@pytest.mark.unit
def test_embed_documents_reraises_embedding_error():
    """embed_documents() re-raises EmbeddingError from model without wrapping."""
    from specagent.retrieval import embedder as emb_mod

    mock_te = MagicMock()
    mock_te.embed.side_effect = EmbeddingError("inner error")
    with patch("specagent.retrieval.embedder.get_embedder", return_value=mock_te):
        with pytest.raises(EmbeddingError, match="inner error"):
            emb_mod.embed_documents(["text"])


@pytest.mark.unit
def test_embed_documents_wraps_generic_exception():
    """embed_documents() wraps non-EmbeddingError exceptions in EmbeddingError."""
    from specagent.retrieval import embedder as emb_mod

    mock_te = MagicMock()
    mock_te.embed.side_effect = RuntimeError("model crashed")
    with patch("specagent.retrieval.embedder.get_embedder", return_value=mock_te):
        with pytest.raises(EmbeddingError, match="Failed to embed"):
            emb_mod.embed_documents(["text"])


@pytest.mark.unit
def test_embed_query_empty_string_raises_embedding_error():
    """embed_query('') raises EmbeddingError without calling model."""
    from specagent.retrieval import embedder as emb_mod

    mock_te = MagicMock()
    with patch("specagent.retrieval.embedder.get_embedder", return_value=mock_te):
        with pytest.raises(EmbeddingError, match="must not be empty"):
            emb_mod.embed_query("")
    mock_te.embed.assert_not_called()


@pytest.mark.unit
def test_embed_query_whitespace_only_raises_embedding_error():
    """embed_query('   ') raises EmbeddingError without calling model."""
    from specagent.retrieval import embedder as emb_mod

    mock_te = MagicMock()
    with patch("specagent.retrieval.embedder.get_embedder", return_value=mock_te):
        with pytest.raises(EmbeddingError, match="must not be empty"):
            emb_mod.embed_query("   ")


@pytest.mark.unit
def test_embed_query_reraises_embedding_error():
    """embed_query() re-raises EmbeddingError from model without wrapping."""
    from specagent.retrieval import embedder as emb_mod

    mock_te = MagicMock()
    mock_te.embed.side_effect = EmbeddingError("inner query error")
    with patch("specagent.retrieval.embedder.get_embedder", return_value=mock_te):
        with pytest.raises(EmbeddingError, match="inner query error"):
            emb_mod.embed_query("what is HARQ?")


@pytest.mark.unit
def test_embed_query_wraps_generic_exception():
    """embed_query() wraps non-EmbeddingError exceptions in EmbeddingError."""
    from specagent.retrieval import embedder as emb_mod

    mock_te = MagicMock()
    mock_te.embed.side_effect = RuntimeError("model dead")
    with patch("specagent.retrieval.embedder.get_embedder", return_value=mock_te):
        with pytest.raises(EmbeddingError, match="Failed to embed query"):
            emb_mod.embed_query("what is HARQ?")
