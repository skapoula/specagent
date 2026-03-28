"""Unit tests for Store performance improvements (Issues 11 and 14)."""

from unittest.mock import MagicMock, patch

import pytest

from specagent.retrieval.store import Store


@pytest.mark.unit
class TestTableCaching:
    """Tests for Issue 11: cached Table object in Store._table()."""

    def test_table_object_is_cached_across_calls(self):
        """_table() returns the same object on repeated calls without re-opening."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")

        mock_table = MagicMock()

        with patch("specagent.retrieval.store._open_table", return_value=mock_table) as mock_open:
            table1 = store._table()
            table2 = store._table()

        # _open_table should only be called once; second call uses cached table
        mock_open.assert_called_once()
        assert table1 is table2
        assert table1 is mock_table


@pytest.mark.unit
class TestIsEmptyGuard:
    """Tests for Issue 14: _is_empty flag replaces count_rows() in search()."""

    def test_search_returns_empty_without_calling_count_rows(self):
        """When _is_empty is True, search() returns [] without calling count_rows."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        assert store._is_empty is True  # default state

        mock_table = MagicMock()

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            result = store.search(
                embedding=[0.1] * 768,
                query_text="test query",
                top_k=5,
                library=None,
                filter=None,
            )

        assert result == []
        mock_table.count_rows.assert_not_called()

    def test_is_empty_set_false_after_upsert(self):
        """_is_empty becomes False after upsert_chunks writes at least one chunk."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        assert store._is_empty is True

        mock_table = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.model_dump.return_value = {
            "id": "test-id",
            "doc_id": "doc-1",
            "library": "lib",
            "source": "file.pdf",
            "content_hash": "abc123",
            "title": "Test",
            "content": "chunk text",
            "embedding": [0.0] * 768,
            "chunk_index": 0,
            "created_at": "2024-01-01T00:00:00+00:00",
            "metadata": "{}",
            "file_type": "pdf",
            "last_modified": "",
            "page": 0,
        }

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            store.upsert_chunks([mock_chunk])

        assert store._is_empty is False
