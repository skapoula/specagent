"""Unit tests for specagent.kuzu.connection.KuzuConnection.

The kuzu library is fully mocked — no real database is created.
Tests must pass offline and without kuzu installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from specagent.kuzu.connection import KuzuConnection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_kuzu_result(rows: list[dict]) -> MagicMock:
    """Build a mock kuzu QueryResult compatible with the get_column_names/has_next/get_next API."""
    result = MagicMock()
    columns = list(rows[0].keys()) if rows else []
    result.get_column_names.return_value = columns
    result.has_next.side_effect = [True] * len(rows) + [False]
    result.get_next.side_effect = [list(r.values()) for r in rows]
    return result


@pytest.fixture
def mock_kuzu(tmp_path):
    """Patch the kuzu module so no real database is opened."""
    with patch("specagent.kuzu.connection.kuzu") as mk:
        mk.Database.return_value = MagicMock()
        mk.Connection.return_value = MagicMock()
        # Schema init calls execute() for each DDL statement; return empty results.
        mk.Connection.return_value.execute.return_value = _make_kuzu_result([])
        yield mk


@pytest.fixture
def conn(mock_kuzu) -> KuzuConnection:
    """KuzuConnection wired to the mock kuzu library."""
    return KuzuConnection("/fake/path")


def _inner_conn(mock_kuzu: MagicMock) -> MagicMock:
    """Return the mock kuzu.Connection instance."""
    return mock_kuzu.Connection.return_value


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    """KuzuConnection.__init__"""

    @pytest.mark.unit
    def test_opens_kuzu_database_at_given_path(self, mock_kuzu) -> None:
        """KuzuConnection opens a kuzu.Database at the provided path."""
        KuzuConnection("/data/dag_store")
        mock_kuzu.Database.assert_called_once_with("/data/dag_store")

    @pytest.mark.unit
    def test_creates_kuzu_connection_from_database(self, mock_kuzu) -> None:
        """KuzuConnection creates a kuzu.Connection from the database object."""
        KuzuConnection("/data/dag_store")
        mock_kuzu.Connection.assert_called_once_with(mock_kuzu.Database.return_value)

    @pytest.mark.unit
    def test_schema_init_executes_ddl_statements(self, mock_kuzu) -> None:
        """__init__ runs DDL statements to create schema tables."""
        KuzuConnection("/data/dag_store")
        inner = _inner_conn(mock_kuzu)
        # At minimum one DDL execute() per table (5 tables)
        assert inner.execute.call_count >= 5


# ---------------------------------------------------------------------------
# execute_cypher
# ---------------------------------------------------------------------------


class TestExecuteCypher:
    """KuzuConnection.execute_cypher"""

    @pytest.mark.unit
    def test_returns_list_of_dicts(self, conn: KuzuConnection, mock_kuzu) -> None:
        """execute_cypher returns rows as dicts built from get_column_names/has_next/get_next."""
        expected = [{"dag_id": "ts23.502::registration", "title": "Registration"}]
        inner = _inner_conn(mock_kuzu)
        inner.execute.return_value = _make_kuzu_result(expected)

        result = conn.execute_cypher(
            "MATCH (d:CallFlowDag) RETURN d.dag_id AS dag_id, d.title AS title"
        )

        assert result == expected

    @pytest.mark.unit
    def test_passes_parameters_to_kuzu_execute(
        self, conn: KuzuConnection, mock_kuzu
    ) -> None:
        """execute_cypher forwards params as the parameters= keyword arg."""
        inner = _inner_conn(mock_kuzu)
        inner.execute.return_value = _make_kuzu_result([])

        conn.execute_cypher(
            "MATCH (d:CallFlowDag {dag_id: $dag_id}) RETURN d",
            {"dag_id": "abc"},
        )

        inner.execute.assert_called_with(
            "MATCH (d:CallFlowDag {dag_id: $dag_id}) RETURN d",
            parameters={"dag_id": "abc"},
        )

    @pytest.mark.unit
    def test_empty_params_default(self, conn: KuzuConnection, mock_kuzu) -> None:
        """execute_cypher passes empty dict when params is None."""
        inner = _inner_conn(mock_kuzu)
        inner.execute.return_value = _make_kuzu_result([])

        conn.execute_cypher("RETURN 1")

        inner.execute.assert_called_with("RETURN 1", parameters={})

    @pytest.mark.unit
    def test_returns_empty_list_when_no_rows(
        self, conn: KuzuConnection, mock_kuzu
    ) -> None:
        """execute_cypher returns [] when the result has no rows."""
        inner = _inner_conn(mock_kuzu)
        inner.execute.return_value = _make_kuzu_result([])

        result = conn.execute_cypher("MATCH (d:CallFlowDag) RETURN d.dag_id")

        assert result == []


# ---------------------------------------------------------------------------
# execute_cypher_write
# ---------------------------------------------------------------------------


class TestExecuteCypherWrite:
    """KuzuConnection.execute_cypher_write"""

    @pytest.mark.unit
    def test_calls_kuzu_execute_with_query_and_params(
        self, conn: KuzuConnection, mock_kuzu
    ) -> None:
        """execute_cypher_write calls kuzu conn.execute with query and parameters."""
        inner = _inner_conn(mock_kuzu)
        inner.execute.return_value = _make_kuzu_result([])

        conn.execute_cypher_write(
            "MERGE (d:CallFlowDag {dag_id: $dag_id})",
            {"dag_id": "ts23.502::registration"},
        )

        inner.execute.assert_called_with(
            "MERGE (d:CallFlowDag {dag_id: $dag_id})",
            parameters={"dag_id": "ts23.502::registration"},
        )

    @pytest.mark.unit
    def test_empty_params_default(self, conn: KuzuConnection, mock_kuzu) -> None:
        """execute_cypher_write passes empty dict when params is None."""
        inner = _inner_conn(mock_kuzu)
        inner.execute.return_value = _make_kuzu_result([])

        conn.execute_cypher_write("MERGE (n:Test {id: $id})", None)

        inner.execute.assert_called_with("MERGE (n:Test {id: $id})", parameters={})

    @pytest.mark.unit
    def test_does_not_raise_on_success(self, conn: KuzuConnection, mock_kuzu) -> None:
        """execute_cypher_write returns None without raising on success."""
        inner = _inner_conn(mock_kuzu)
        inner.execute.return_value = _make_kuzu_result([])

        result = conn.execute_cypher_write("MERGE (d:CallFlowDag {dag_id: $dag_id})", {"dag_id": "x"})

        assert result is None


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """KuzuConnection.health_check"""

    @pytest.mark.unit
    def test_returns_true_when_execute_returns_one(
        self, conn: KuzuConnection, mock_kuzu
    ) -> None:
        """health_check returns True when RETURN 1 AS health yields {health: 1}."""
        inner = _inner_conn(mock_kuzu)
        inner.execute.return_value = _make_kuzu_result([{"health": 1}])

        assert conn.health_check() is True

    @pytest.mark.unit
    def test_returns_false_on_exception(self, conn: KuzuConnection, mock_kuzu) -> None:
        """health_check returns False when execute raises."""
        inner = _inner_conn(mock_kuzu)
        inner.execute.side_effect = RuntimeError("db error")

        assert conn.health_check() is False

    @pytest.mark.unit
    def test_returns_false_on_empty_result(
        self, conn: KuzuConnection, mock_kuzu
    ) -> None:
        """health_check returns False when query returns no rows."""
        inner = _inner_conn(mock_kuzu)
        inner.execute.return_value = _make_kuzu_result([])

        assert conn.health_check() is False


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """KuzuConnection context-manager protocol."""

    @pytest.mark.unit
    def test_enter_returns_self(self, conn: KuzuConnection) -> None:
        """__enter__ returns the connection itself."""
        assert conn.__enter__() is conn

    @pytest.mark.unit
    def test_exit_does_not_raise(self, conn: KuzuConnection) -> None:
        """__exit__ completes without raising."""
        conn.__exit__(None, None, None)
