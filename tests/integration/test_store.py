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
    store.upsert_chunks([sample_record])
    results = store.search(
        embedding=sample_record.embedding,
        query_text="HARQ",
        top_k=5,
        library="test-lib",
        filter=None,
    )
    assert len(results) >= 1


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
