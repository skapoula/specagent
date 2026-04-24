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

    def test_search_returns_empty_for_empty_table(self):
        """search() returns [] when the table is empty; count_rows called once at init."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        assert store._is_empty is True  # default state

        mock_table = MagicMock()
        mock_table.count_rows.return_value = 0  # empty table

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            result = store.search(
                embedding=[0.1] * 768,
                query_text="test query",
                top_k=5,
                library=None,
                filter=None,
            )

        assert result == []
        # count_rows is called exactly once during _table() init, not per-search call
        mock_table.count_rows.assert_called_once()

    def test_is_empty_false_after_table_access_with_data(self):
        """_is_empty is set from count_rows on first _table() access."""
        # Simulate a pre-existing database: count_rows() returns > 0 on first open.
        # A second Store instance pointing at the same URI must detect it is not empty
        # after the first _table() call, without requiring upsert_chunks to be called.
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        assert store._is_empty is True  # default before any table access

        mock_table = MagicMock()
        mock_table.count_rows.return_value = 42  # pre-existing rows

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            store._table()  # first access should call count_rows and flip the flag

        assert store._is_empty is False
        mock_table.count_rows.assert_called_once()

    def test_is_empty_stays_true_for_empty_table_on_first_access(self):
        """_is_empty remains True when count_rows returns 0 on first _table() access."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")

        mock_table = MagicMock()
        mock_table.count_rows.return_value = 0

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            store._table()

        assert store._is_empty is True

    def test_is_empty_set_false_after_upsert(self, real_chunk_record):
        """_is_empty becomes False after upsert_chunks writes at least one chunk."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        assert store._is_empty is True

        mock_table = MagicMock()

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            store.upsert_chunks([real_chunk_record])

        assert store._is_empty is False


# ──────────────────────────────────────────────────────────────────────────────
# Failing tests for bug fixes
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStoreDeleteBugFixes:
    """Regression tests for _is_empty and list_documents bug fixes."""

    def test_is_empty_resets_after_delete_library(self):
        """_is_empty must be rechecked after delete_library empties the table."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")

        mock_table = MagicMock()
        # First open: table has 5 rows → _is_empty = False
        mock_table.count_rows.return_value = 5

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            store._table()  # _is_empty = False

        assert store._is_empty is False

        # After delete, table is empty
        mock_table.count_rows.return_value = 0
        store.delete_library("mylib")

        # After deleting all rows, _is_empty must reflect the current state
        assert store._is_empty is True, (
            "_is_empty must be reset to True after delete_library empties the table"
        )

    def test_is_empty_resets_after_delete_document(self):
        """_is_empty must be rechecked after delete_document empties the table."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")

        mock_table = MagicMock()
        mock_table.count_rows.return_value = 1

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            store._table()

        assert store._is_empty is False

        mock_table.count_rows.return_value = 0
        store.delete_document("some-doc-id")

        assert store._is_empty is True

    def test_list_documents_skips_row_with_corrupt_metadata(self):
        """list_documents must continue listing when one row has invalid JSON metadata."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")

        mock_table = MagicMock()
        mock_table.count_rows.return_value = 2
        # First row has invalid JSON; second row has valid JSON
        mock_table.search.return_value.select.return_value.limit.return_value.to_list.return_value = [
            {
                "doc_id": "bad-doc",
                "source": "/specs/bad.docx",
                "title": "Bad Doc",
                "library": "lib",
                "content_hash": "abc",
                "created_at": "2024-01-01",
                "metadata": "NOT_VALID_JSON",
            },
            {
                "doc_id": "good-doc",
                "source": "38413-i30.docx",
                "title": "TS 38.413 NG Application Protocol",
                "library": "lib",
                "content_hash": "def",
                "created_at": "2024-01-02",
                "metadata": '{"key": "value"}',
            },
        ]

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            docs = store.list_documents(library=None, limit=10, offset=0)

        # The good document must still appear even though the bad one had corrupt metadata
        assert len(docs) >= 1, "list_documents must not abort on a single corrupt metadata row"
        good_docs = [d for d in docs if d["doc_id"] == "good-doc"]
        assert len(good_docs) == 1

    def test_store_has_write_lock_for_thread_safety(self):
        """Store must have a _write_lock to serialize concurrent upsert_chunks calls.

        asyncio.to_thread dispatches upsert_chunks to OS threads; without a lock
        concurrent table.add() calls corrupt the LanceDB table.
        """
        import threading

        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        assert hasattr(store, "_write_lock"), (
            "Store must expose _write_lock (threading.Lock) to serialize concurrent writes"
        )
        assert isinstance(store._write_lock, type(threading.Lock())), (
            "_write_lock must be a threading.Lock"
        )


@pytest.mark.unit
class TestTableLockThreadSafety:
    """Fix 9: _table() must use double-checked locking to prevent concurrent opens."""

    def test_table_opened_only_once_under_concurrent_access(self):
        """_table() must call _open_table exactly once even under concurrent access."""
        import threading
        import time

        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        mock_table = MagicMock()
        mock_table.count_rows.return_value = 0
        open_count: list[int] = []
        errors: list[Exception] = []

        def slow_open(uri: str, table_name: str) -> MagicMock:
            open_count.append(1)
            time.sleep(0.05)  # hold lock long enough for other threads to arrive
            return mock_table

        def worker() -> None:
            try:
                store._table()
            except Exception as exc:
                errors.append(exc)

        with patch("specagent.retrieval.store._open_table", side_effect=slow_open):
            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"
        assert len(open_count) == 1, (
            f"_open_table called {len(open_count)} times; expected exactly 1"
        )

    def test_store_has_table_lock(self):
        """Store must expose _table_lock for thread-safe lazy initialisation."""
        import threading

        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        assert hasattr(store, "_table_lock")
        assert isinstance(store._table_lock, type(threading.Lock()))


@pytest.mark.unit
class TestUpsertIsTrue:
    """Fix 2: upsert_chunks must delete old chunks before inserting new ones."""

    def test_delete_called_before_add(self, real_chunk_record):
        """upsert_chunks must call table.delete for the chunk's doc_id before table.add."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        mock_table = MagicMock()
        mock_table.count_rows.return_value = 5

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            store.upsert_chunks([real_chunk_record])

        call_names = [c[0] for c in mock_table.mock_calls]
        assert "delete" in call_names, "table.delete() must be called during upsert"
        assert "add" in call_names, "table.add() must be called during upsert"
        delete_idx = call_names.index("delete")
        add_idx = call_names.index("add")
        assert delete_idx < add_idx, "delete must happen before add"

    def test_delete_not_called_if_row_build_raises(self, real_chunk_record):
        """upsert_chunks must not delete rows if serialisation fails before insert."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        mock_table = MagicMock()
        mock_table.count_rows.return_value = 3

        bad = MagicMock()
        bad.model_dump.side_effect = ValueError("boom")
        bad.doc_id = real_chunk_record.doc_id

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            with pytest.raises(Exception):
                store.upsert_chunks([bad])

        mock_table.delete.assert_not_called()


@pytest.mark.unit
class TestRebuildFtsIndexReturnsBool:
    """Fix 11: rebuild_fts_index must return bool instead of None."""

    def test_returns_true_on_success(self):
        """rebuild_fts_index() must return True when the FTS index is rebuilt."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        mock_table = MagicMock()
        mock_table.count_rows.return_value = 1

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            result = store.rebuild_fts_index()

        assert result is True

    def test_returns_false_on_failure(self):
        """rebuild_fts_index() must return False when FTS rebuild raises."""
        store = Store(uri="/tmp/test_lancedb", table_name="test_docs")
        mock_table = MagicMock()
        mock_table.count_rows.return_value = 1
        mock_table.create_fts_index.side_effect = RuntimeError("FTS failed")

        with patch("specagent.retrieval.store._open_table", return_value=mock_table):
            result = store.rebuild_fts_index()

        assert result is False
