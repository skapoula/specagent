"""Integration tests for KuzuConnection — uses a real Kuzu database in tmp_path.

These fail until KuzuConnection is implemented and kuzu is installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from specagent.memgraph.connection import KuzuConnection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_DAG_PARAMS: dict = {
    "dag_id": "ts23.502::registration",
    "doc_id": "doc-001",
    "source": "TS23.502.docx",
    "title": "Registration Procedure",
    "mermaid_content": "sequenceDiagram\n    UE->>AMF: Registration Request\n",
    "prose_description": "UE registration with AMF.",
    "ingested_at": "2024-01-01T00:00:00+00:00",
}

_UPSERT_DAG = """\
MERGE (d:CallFlowDag {dag_id: $dag_id})
ON CREATE SET d.doc_id = $doc_id, d.source = $source, d.title = $title,
              d.mermaid_content = $mermaid_content,
              d.prose_description = $prose_description,
              d.ingested_at = $ingested_at
ON MATCH SET  d.doc_id = $doc_id, d.source = $source, d.title = $title,
              d.mermaid_content = $mermaid_content,
              d.prose_description = $prose_description,
              d.ingested_at = $ingested_at
"""


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_kuzu_connection_opens_without_error(tmp_path: Path) -> None:
    """KuzuConnection can be created against a fresh directory."""
    conn = KuzuConnection(tmp_path / "test.kuzu")
    assert conn is not None


@pytest.mark.integration
def test_schema_creates_node_tables(tmp_path: Path) -> None:
    """Schema init creates CallFlowDag, DagParticipant and DagStep tables."""
    conn = KuzuConnection(tmp_path / "test.kuzu")
    for label in ("CallFlowDag", "DagParticipant", "DagStep"):
        result = conn.execute_cypher(f"MATCH (n:{label}) RETURN n LIMIT 0")
        assert isinstance(result, list)


@pytest.mark.integration
def test_schema_creates_rel_tables(tmp_path: Path) -> None:
    """Schema init creates HAS_PARTICIPANT and HAS_STEP relationship tables."""
    conn = KuzuConnection(tmp_path / "test.kuzu")
    result = conn.execute_cypher(
        "MATCH (d:CallFlowDag)-[:HAS_STEP]->(s:DagStep) RETURN d.dag_id LIMIT 0"
    )
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# execute_cypher
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_execute_cypher_returns_list_of_dicts(tmp_path: Path) -> None:
    """execute_cypher returns a list of plain dicts."""
    conn = KuzuConnection(tmp_path / "test.kuzu")
    result = conn.execute_cypher("RETURN 42 AS answer")
    assert result == [{"answer": 42}]


@pytest.mark.integration
def test_execute_cypher_accepts_parameters(tmp_path: Path) -> None:
    """execute_cypher substitutes $-prefixed parameters."""
    conn = KuzuConnection(tmp_path / "test.kuzu")
    result = conn.execute_cypher("RETURN $x AS val", {"x": 99})
    assert result == [{"val": 99}]


@pytest.mark.integration
def test_execute_cypher_returns_empty_list_when_no_match(tmp_path: Path) -> None:
    """execute_cypher returns [] when the query finds no rows."""
    conn = KuzuConnection(tmp_path / "test.kuzu")
    result = conn.execute_cypher(
        "MATCH (d:CallFlowDag {dag_id: $dag_id}) RETURN d.dag_id",
        {"dag_id": "nonexistent"},
    )
    assert result == []


# ---------------------------------------------------------------------------
# execute_cypher_write
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_execute_cypher_write_upserts_dag_node(tmp_path: Path) -> None:
    """execute_cypher_write can MERGE a CallFlowDag node."""
    conn = KuzuConnection(tmp_path / "test.kuzu")
    conn.execute_cypher_write(_UPSERT_DAG, _SAMPLE_DAG_PARAMS)

    rows = conn.execute_cypher(
        "MATCH (d:CallFlowDag {dag_id: $dag_id}) RETURN d.dag_id AS dag_id",
        {"dag_id": "ts23.502::registration"},
    )
    assert rows == [{"dag_id": "ts23.502::registration"}]


@pytest.mark.integration
def test_execute_cypher_write_is_idempotent(tmp_path: Path) -> None:
    """Calling execute_cypher_write twice with same dag_id does not duplicate."""
    conn = KuzuConnection(tmp_path / "test.kuzu")
    conn.execute_cypher_write(_UPSERT_DAG, _SAMPLE_DAG_PARAMS)
    conn.execute_cypher_write(_UPSERT_DAG, _SAMPLE_DAG_PARAMS)

    rows = conn.execute_cypher("MATCH (d:CallFlowDag) RETURN d.dag_id AS dag_id")
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_health_check_returns_true(tmp_path: Path) -> None:
    """health_check returns True for an open Kuzu database."""
    conn = KuzuConnection(tmp_path / "test.kuzu")
    assert conn.health_check() is True


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_context_manager_does_not_raise(tmp_path: Path) -> None:
    """KuzuConnection works as a context manager."""
    with KuzuConnection(tmp_path / "test.kuzu") as conn:
        result = conn.execute_cypher("RETURN 1 AS ok")
    assert result == [{"ok": 1}]


# ---------------------------------------------------------------------------
# get_dag_connection singleton
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_get_dag_connection_returns_kuzu_connection(tmp_path: Path) -> None:
    """get_dag_connection() returns a KuzuConnection using KUZU_DB_PATH env var."""
    import os

    from specagent.memgraph.connection import clear_dag_connection_cache, get_dag_connection

    clear_dag_connection_cache()
    old = os.environ.get("KUZU_DB_PATH")
    try:
        os.environ["KUZU_DB_PATH"] = str(tmp_path / "singleton.kuzu")
        conn = get_dag_connection()
        assert isinstance(conn, KuzuConnection)
        assert conn.health_check() is True
    finally:
        clear_dag_connection_cache()
        if old is None:
            os.environ.pop("KUZU_DB_PATH", None)
        else:
            os.environ["KUZU_DB_PATH"] = old
