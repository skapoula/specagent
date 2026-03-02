"""Integration tests for LanceDB store."""
import json
import uuid

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
    from specagent.retrieval.store import _build_where_clause
    from specagent.retrieval.exceptions import StoreError

    with pytest.raises(StoreError, match="Invalid filter key"):
        _build_where_clause(None, {"bad-key!": "val"})
