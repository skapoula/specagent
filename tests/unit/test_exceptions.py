"""Tests for domain exception classes in specagent.retrieval.exceptions."""

import pytest


@pytest.mark.unit
def test_unsupported_format_error_is_exception():
    from specagent.retrieval.exceptions import UnsupportedFormatError

    with pytest.raises(UnsupportedFormatError):
        raise UnsupportedFormatError(".xyz")


@pytest.mark.unit
def test_ingestion_error_is_exception():
    from specagent.retrieval.exceptions import IngestionError

    with pytest.raises(IngestionError):
        raise IngestionError("failed to fetch")


@pytest.mark.unit
def test_store_error_is_exception():
    from specagent.retrieval.exceptions import StoreError

    with pytest.raises(StoreError):
        raise StoreError("table corrupted")


@pytest.mark.unit
def test_embedding_error_is_exception():
    from specagent.retrieval.exceptions import EmbeddingError

    with pytest.raises(EmbeddingError):
        raise EmbeddingError("model not loaded")
