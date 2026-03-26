"""Integration tests for LanceDB store."""

import json
import logging
import uuid
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def store(tmp_path):
    """Fresh Store backed by tmp LanceDB."""
    from specagent.retrieval.store import Store

    return Store(uri=str(tmp_path / "lancedb"), table_name="test_docs")


@pytest.fixture
def sample_record():
    from specagent.retrieval.store import ChunkRecord

    return ChunkRecord(
        id=str(uuid.uuid4()),
        doc_id=str(uuid.uuid4()),
        library="test-lib",
        source="/tmp/test.docx",
        content_hash="abc123",
        title="Test Document",
        content="The maximum number of HARQ processes in NR is 16.",
        embedding=[0.1] * 768,
        chunk_index=0,
        created_at="2026-03-01T00:00:00Z",
        metadata=json.dumps({"section_header": "5.4 HARQ Entity"}),
        file_type="docx",
        last_modified="2026-03-01T00:00:00Z",
        page=0,
    )


@pytest.mark.integration
def test_upsert_and_search(store, sample_record):
    """Upsert a chunk and retrieve it via search."""
    from specagent.retrieval.store import ChunkRecord

    store.upsert_chunks([sample_record])
    results = store.search(
        embedding=sample_record.embedding,
        query_text="HARQ",
        top_k=5,
        library="test-lib",
        filter=None,
    )
    assert len(results) >= 1
    record, score = results[0]
    assert isinstance(record, ChunkRecord)
    assert 0.0 <= score <= 1.0


@pytest.mark.integration
def test_empty_search_returns_empty_list(store):
    """Search on empty table returns [] not an exception."""
    results = store.search(
        embedding=[0.1] * 768,
        query_text="anything",
        top_k=5,
        library=None,
        filter=None,
    )
    assert results == []


@pytest.mark.integration
def test_delete_document(store, sample_record):
    """delete_document() removes all chunks for that doc_id."""
    store.upsert_chunks([sample_record])
    deleted = store.delete_document(sample_record.doc_id)
    assert deleted >= 1


@pytest.mark.integration
def test_find_existing_returns_none_when_missing(store):
    """find_existing() returns (None, None) when source not in DB."""
    doc_id, content_hash = store.find_existing("/nonexistent.docx", "my-lib")
    assert doc_id is None
    assert content_hash is None


@pytest.mark.integration
def test_get_document_sorted(store, sample_record):
    from specagent.retrieval.store import ChunkRecord

    rec2 = ChunkRecord(
        id=str(uuid.uuid4()),
        doc_id=sample_record.doc_id,
        library="test-lib",
        source=sample_record.source,
        content_hash="def456",
        title="T",
        content="Second.",
        embedding=[0.2] * 768,
        chunk_index=1,
        created_at="2026-03-01T00:00:00Z",
        metadata=json.dumps({}),
        file_type="docx",
        last_modified="",
        page=0,
    )
    store.upsert_chunks([sample_record, rec2])
    docs = store.get_document(sample_record.doc_id)
    assert len(docs) >= 2
    indices = [d.chunk_index for d in docs]
    assert indices == sorted(indices)


@pytest.mark.integration
def test_list_documents_per_doc(store, sample_record):
    store.upsert_chunks([sample_record])
    docs = store.list_documents(library=None, limit=10, offset=0)
    assert any(d["doc_id"] == sample_record.doc_id for d in docs)
    assert all("chunk_count" in d for d in docs)


@pytest.mark.integration
def test_list_documents_library_filter(store, sample_record):
    store.upsert_chunks([sample_record])
    docs = store.list_documents(library="test-lib", limit=10, offset=0)
    assert all(d["library"] == "test-lib" for d in docs)


@pytest.mark.integration
def test_list_documents_beyond_offset(store, sample_record):
    store.upsert_chunks([sample_record])
    assert store.list_documents(library=None, limit=10, offset=9999) == []


@pytest.mark.integration
def test_list_libraries(store, sample_record):
    store.upsert_chunks([sample_record])
    libs = store.list_libraries()
    lib = next(lb for lb in libs if lb["library"] == "test-lib")
    assert lib["document_count"] >= 1 and lib["chunk_count"] >= 1


@pytest.mark.integration
def test_find_existing_returns_ids(store, sample_record):
    store.upsert_chunks([sample_record])
    doc_id, ch = store.find_existing(sample_record.source, "test-lib")
    assert doc_id == sample_record.doc_id


@pytest.mark.unit
def test_build_where_clause_none_none():
    from specagent.retrieval.store import _build_where_clause

    assert _build_where_clause(None, None) is None


@pytest.mark.unit
def test_build_where_clause_library_only():
    from specagent.retrieval.store import _build_where_clause

    result = _build_where_clause("my-lib", None)
    assert "my-lib" in result


@pytest.mark.unit
def test_build_where_clause_filter_only():
    from specagent.retrieval.store import _build_where_clause

    result = _build_where_clause(None, {"file_type": "pdf"})
    assert "pdf" in result


@pytest.mark.unit
def test_build_where_clause_invalid_key_raises():
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import _build_where_clause

    with pytest.raises(StoreError, match="Invalid filter key"):
        _build_where_clause(None, {"bad-key!": "val"})


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for Store error paths (pure mocks, no real LanceDB)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeField:
    """Minimal stand-in for a PyArrow schema field."""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.unit
def test_open_table_exception_wraps_as_store_error():
    import lancedb

    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import _open_table

    with patch.object(lancedb, "connect", side_effect=RuntimeError("no db")):
        with pytest.raises(StoreError, match="Failed to open LanceDB"):
            _open_table("/tmp/nonexistent_test_db_xyz", "docs")


@pytest.mark.unit
def test_open_table_reraises_store_error_from_validate():
    """_open_table re-raises StoreError from _validate_embedding_dimension (line 97)."""
    import lancedb

    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import _open_table

    mock_db = MagicMock()
    mock_db.list_tables.return_value = ["docs"]
    mock_db.open_table.return_value = MagicMock()
    with (
        patch.object(lancedb, "connect", return_value=mock_db),
        patch(
            "specagent.retrieval.store._validate_embedding_dimension",
            side_effect=StoreError("dim mismatch"),
        ),
        pytest.raises(StoreError, match="dim mismatch"),
    ):
        _open_table("/tmp/test_db", "docs")


@pytest.mark.unit
def test_ensure_scalar_indexes_logs_warning_on_failure(caplog):
    from specagent.retrieval.store import _ensure_scalar_indexes

    mock_table = MagicMock()
    mock_table.create_scalar_index.side_effect = RuntimeError("no rows yet")
    with caplog.at_level(logging.WARNING, logger="specagent.retrieval.store"):
        _ensure_scalar_indexes(mock_table)
    assert mock_table.create_scalar_index.call_count == 3


@pytest.mark.unit
def test_validate_embedding_dimension_schema_error_returns_silently():
    from specagent.retrieval.store import _validate_embedding_dimension

    mock_table = MagicMock()
    mock_table.schema.field.side_effect = Exception("no embedding field")
    _validate_embedding_dimension(mock_table)  # must not raise


@pytest.mark.unit
def test_validate_embedding_dimension_mismatch_raises():
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import _validate_embedding_dimension

    mock_field = MagicMock()
    mock_field.type.list_size = 99  # wrong dimension
    mock_table = MagicMock()
    mock_table.schema.field.return_value = mock_field
    with patch("specagent.retrieval.store.settings") as ms:
        ms.embedding_dimension = 768
        with pytest.raises(StoreError, match="Embedding dimension mismatch"):
            _validate_embedding_dimension(mock_table)


@pytest.mark.unit
def test_migrate_table_adds_missing_columns():
    from specagent.retrieval.store import _migrate_table

    mock_table = MagicMock()
    mock_table.schema = [_FakeField("id"), _FakeField("content")]
    _migrate_table(mock_table)
    mock_table.add_columns.assert_called_once()
    call_arg = mock_table.add_columns.call_args[0][0]
    assert "file_type" in call_arg
    assert "last_modified" in call_arg
    assert "page" in call_arg


@pytest.mark.unit
def test_migrate_table_add_columns_failure_logs_warning(caplog):
    from specagent.retrieval.store import _migrate_table

    mock_table = MagicMock()
    mock_table.schema = [_FakeField("id")]
    mock_table.add_columns.side_effect = RuntimeError("table locked")
    with caplog.at_level(logging.WARNING, logger="specagent.retrieval.store"):
        _migrate_table(mock_table)
    assert "Schema migration failed" in caplog.text


@pytest.mark.unit
def test_build_where_clause_int_filter_value():
    from specagent.retrieval.store import _build_where_clause

    result = _build_where_clause(None, {"page": 1})
    assert result is not None
    assert "page = 1" in result


@pytest.mark.unit
def test_upsert_chunks_empty_list_returns_without_opening_table():
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    with patch("specagent.retrieval.store._open_table") as mock_ot:
        store_obj.upsert_chunks([])
        mock_ot.assert_not_called()


@pytest.mark.unit
def test_upsert_chunks_fts_failure_logs_warning(sample_record, caplog):
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.create_fts_index.side_effect = RuntimeError("fts not available")
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with caplog.at_level(logging.WARNING, logger="specagent.retrieval.store"):
            store_obj.upsert_chunks([sample_record])
    assert "FTS index rebuild failed" in caplog.text


@pytest.mark.unit
def test_upsert_chunks_add_failure_raises_store_error(sample_record):
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.add.side_effect = RuntimeError("write failed")
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with pytest.raises(StoreError, match="Failed to upsert"):
            store_obj.upsert_chunks([sample_record])


@pytest.mark.unit
def test_find_existing_exception_raises_store_error():
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.search.side_effect = RuntimeError("db read error")
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with pytest.raises(StoreError, match="find_existing failed"):
            store_obj.find_existing("/test.docx", "lib")


@pytest.mark.unit
def test_rebuild_fts_failure_logs_warning(caplog):
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.create_fts_index.side_effect = RuntimeError("fts error")
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with caplog.at_level(logging.WARNING, logger="specagent.retrieval.store"):
            store_obj.rebuild_fts_index()
    assert "FTS index rebuild failed" in caplog.text


@pytest.mark.unit
def test_rebuild_fts_success_logs_info(caplog):
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with caplog.at_level(logging.INFO, logger="specagent.retrieval.store"):
            store_obj.rebuild_fts_index()
    assert "FTS index rebuilt" in caplog.text


@pytest.mark.unit
def test_delete_exception_raises_store_error():
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.count_rows.side_effect = RuntimeError("db error")
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with pytest.raises(StoreError, match="Failed to delete"):
            store_obj.delete_document("test-doc-id")


@pytest.mark.unit
def test_search_hybrid_raises_store_error_on_fts_failure():
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.search.side_effect = RuntimeError("no fts index")
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with patch("specagent.retrieval.store.settings") as ms:
            ms.search_refine_factor = 2
            with pytest.raises(StoreError, match="Search failed"):
                store_obj.search([0.1] * 768, "query", 5, None, None)


@pytest.mark.unit
def test_search_exception_raises_store_error():
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.search.side_effect = RuntimeError("unexpected db failure")
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with patch("specagent.retrieval.store.settings") as ms:
            ms.search_refine_factor = 2
            with pytest.raises(StoreError, match="Search failed"):
                store_obj.search([0.1] * 768, "query", 5, None, None)


@pytest.mark.unit
def test_search_without_library_filter_skips_where_clause():
    """search() with library=None and filter=None sets no WHERE clause."""
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.count_rows.return_value = 1
    mock_query = MagicMock()
    mock_query.vector.return_value.text.return_value = mock_query
    mock_query.refine_factor.return_value.limit.return_value.to_list.return_value = []
    mock_table.search.return_value = mock_query
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with patch("specagent.retrieval.store.settings") as ms:
            ms.search_refine_factor = 2
            results = store_obj.search([0.1] * 768, "query", 5, None, None)
    # where is None — query.where() must NOT have been called
    mock_query.where.assert_not_called()
    assert results == []


@pytest.mark.unit
def test_get_document_exception_raises_store_error():
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.search.side_effect = RuntimeError("table error")
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with pytest.raises(StoreError, match="get_document failed"):
            store_obj.get_document("test-doc-id")


@pytest.mark.integration
def test_list_documents_counts_chunks_per_doc(store, sample_record):
    """list_documents increments chunk_count when doc_id already seen."""
    from specagent.retrieval.store import ChunkRecord

    rec2 = ChunkRecord(
        id=str(uuid.uuid4()),
        doc_id=sample_record.doc_id,
        library="test-lib",
        source=sample_record.source,
        content_hash="def456",
        title="T",
        content="Second chunk.",
        embedding=[0.2] * 768,
        chunk_index=1,
        created_at="2026-03-01T00:00:00Z",
        metadata=json.dumps({}),
        file_type="docx",
        last_modified="",
        page=0,
    )
    store.upsert_chunks([sample_record, rec2])
    docs = store.list_documents(library=None, limit=10, offset=0)
    matching = [d for d in docs if d["doc_id"] == sample_record.doc_id]
    assert len(matching) == 1
    assert matching[0]["chunk_count"] == 2


@pytest.mark.unit
def test_list_documents_exception_raises_store_error():
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.search.side_effect = RuntimeError("db error")
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with pytest.raises(StoreError, match="list_documents failed"):
            store_obj.list_documents(library=None, limit=10, offset=0)


@pytest.mark.integration
def test_list_libraries_counts_chunks_same_library(store, sample_record):
    """list_libraries increments chunk_count for multiple chunks in same library."""
    from specagent.retrieval.store import ChunkRecord

    rec2 = ChunkRecord(
        id=str(uuid.uuid4()),
        doc_id=sample_record.doc_id,
        library="test-lib",
        source=sample_record.source,
        content_hash="xyz789",
        title="T2",
        content="Chunk 2.",
        embedding=[0.3] * 768,
        chunk_index=1,
        created_at="2026-03-01T00:00:00Z",
        metadata=json.dumps({}),
        file_type="docx",
        last_modified="",
        page=0,
    )
    store.upsert_chunks([sample_record, rec2])
    libs = store.list_libraries()
    lib = next(lb for lb in libs if lb["library"] == "test-lib")
    assert lib["chunk_count"] == 2


@pytest.mark.unit
def test_list_libraries_exception_raises_store_error():
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    mock_table.search.side_effect = RuntimeError("db error")
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with pytest.raises(StoreError, match="list_libraries failed"):
            store_obj.list_libraries()


@pytest.mark.unit
def test_search_reraises_store_error_from_invalid_filter():
    """search() re-raises StoreError raised by _build_where_clause (line 419)."""
    from specagent.retrieval.exceptions import StoreError
    from specagent.retrieval.store import Store

    store_obj = Store(uri="/tmp/test_store_unit", table_name="docs")
    mock_table = MagicMock()
    with patch("specagent.retrieval.store._open_table", return_value=mock_table):
        with pytest.raises(StoreError, match="Invalid filter key"):
            store_obj.search([0.1] * 768, "q", 5, None, {"bad-key!": "val"})
