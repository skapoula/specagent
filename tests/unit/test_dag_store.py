"""Unit tests for specagent.kuzu.dag_store.CallFlowDagStore.

All Kuzu calls are mocked — no live Kuzu instance required.
Tests must pass offline.

Tests are written FIRST (TDD). Implementation does not exist yet.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from specagent.kuzu.dag_store import CallFlowDagStore
from specagent.kuzu.mermaid_parser import StepRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_MERMAID = """\
sequenceDiagram
    participant UE
    participant AMF
    UE->>AMF: Registration Request
    AMF-->>UE: Registration Accept
"""

_SAMPLE_STEPS = [
    StepRecord(step_index=0, from_actor="UE", to_actor="AMF",
               message="Registration Request", is_async=False),
    StepRecord(step_index=1, from_actor="AMF", to_actor="UE",
               message="Registration Accept", is_async=True),
]


@pytest.fixture
def mock_conn() -> MagicMock:
    """Mock KuzuConnection with no-op execute methods."""
    conn = MagicMock()
    conn.execute_cypher.return_value = []
    conn.execute_cypher_write.return_value = None
    return conn


@pytest.fixture
def store(mock_conn: MagicMock) -> CallFlowDagStore:
    """CallFlowDagStore wired to a mock connection."""
    return CallFlowDagStore(connection=mock_conn)


# ---------------------------------------------------------------------------
# store_call_flow_dag
# ---------------------------------------------------------------------------


class TestStoreCallFlowDag:
    """CallFlowDagStore.store_call_flow_dag"""

    @pytest.mark.unit
    def test_calls_execute_cypher_write(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """store_call_flow_dag calls execute_cypher_write at least once."""
        store.store_call_flow_dag(
            dag_id="ts23.502::registration",
            doc_id="doc-001",
            source="TS23.502.docx",
            title="Registration Procedure",
            mermaid_content=_SAMPLE_MERMAID,
            participants=["UE", "AMF"],
            steps=_SAMPLE_STEPS,
        )

        mock_conn.execute_cypher_write.assert_called()

    @pytest.mark.unit
    def test_passes_dag_id_as_param(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """dag_id is passed as a Cypher parameter (not interpolated)."""
        store.store_call_flow_dag(
            dag_id="ts23.502::registration",
            doc_id="doc-001",
            source="TS23.502.docx",
            title="Registration Procedure",
            mermaid_content=_SAMPLE_MERMAID,
            participants=["UE", "AMF"],
            steps=_SAMPLE_STEPS,
        )

        # Inspect first call's params dict
        call_params = mock_conn.execute_cypher_write.call_args_list[0][0][1]
        assert call_params["dag_id"] == "ts23.502::registration"

    @pytest.mark.unit
    def test_passes_doc_id_and_source_as_params(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """doc_id and source are passed as Cypher parameters."""
        store.store_call_flow_dag(
            dag_id="ts23.502::registration",
            doc_id="doc-001",
            source="TS23.502.docx",
            title="Registration Procedure",
            mermaid_content=_SAMPLE_MERMAID,
            participants=["UE", "AMF"],
            steps=_SAMPLE_STEPS,
        )

        call_params = mock_conn.execute_cypher_write.call_args_list[0][0][1]
        assert call_params["doc_id"] == "doc-001"
        assert call_params["source"] == "TS23.502.docx"

    @pytest.mark.unit
    def test_passes_participants_as_param(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """Each participant name is passed as a Cypher parameter across write calls."""
        store.store_call_flow_dag(
            dag_id="ts23.502::registration",
            doc_id="doc-001",
            source="TS23.502.docx",
            title="Registration Procedure",
            mermaid_content=_SAMPLE_MERMAID,
            participants=["UE", "AMF"],
            steps=_SAMPLE_STEPS,
        )

        all_params = [c[0][1] for c in mock_conn.execute_cypher_write.call_args_list]
        participant_names = [p["name"] for p in all_params if "name" in p]
        assert "UE" in participant_names
        assert "AMF" in participant_names

    @pytest.mark.unit
    def test_passes_steps_as_param(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """Each step's fields are passed as individual Cypher parameters."""
        store.store_call_flow_dag(
            dag_id="ts23.502::registration",
            doc_id="doc-001",
            source="TS23.502.docx",
            title="Registration Procedure",
            mermaid_content=_SAMPLE_MERMAID,
            participants=["UE", "AMF"],
            steps=_SAMPLE_STEPS,
        )

        all_params = [c[0][1] for c in mock_conn.execute_cypher_write.call_args_list]
        step_calls = [p for p in all_params if "from_actor" in p]
        assert len(step_calls) == 2
        assert any(p["from_actor"] == "UE" for p in step_calls)
        assert any(p["is_async"] is True for p in step_calls)

    @pytest.mark.unit
    def test_uses_merge_query(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """The Cypher query uses MERGE for idempotency."""
        store.store_call_flow_dag(
            dag_id="ts23.502::registration",
            doc_id="doc-001",
            source="TS23.502.docx",
            title="Registration Procedure",
            mermaid_content=_SAMPLE_MERMAID,
            participants=["UE", "AMF"],
            steps=_SAMPLE_STEPS,
        )

        cypher_query: str = mock_conn.execute_cypher_write.call_args_list[0][0][0]
        assert "MERGE" in cypher_query.upper()

    @pytest.mark.unit
    def test_does_not_raise_on_empty_steps(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """store_call_flow_dag does not raise when steps list is empty."""
        store.store_call_flow_dag(
            dag_id="ts23.502::registration",
            doc_id="doc-001",
            source="TS23.502.docx",
            title="Registration Procedure",
            mermaid_content="sequenceDiagram\n",
            participants=[],
            steps=[],
        )

        mock_conn.execute_cypher_write.assert_called()


# ---------------------------------------------------------------------------
# query_dags_by_keyword
# ---------------------------------------------------------------------------


class TestQueryDagsByKeyword:
    """CallFlowDagStore.query_dags_by_keyword"""

    @pytest.mark.unit
    def test_returns_empty_list_on_no_match(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """Returns empty list when Kuzu returns no results."""
        mock_conn.execute_cypher.return_value = []

        result = store.query_dags_by_keyword(["registration"], limit=3)

        assert result == []

    @pytest.mark.unit
    def test_returns_dag_list_on_match(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """Returns list of dicts when Kuzu returns matches."""
        mock_conn.execute_cypher.return_value = [
            {
                "dag_id": "ts23.502::registration",
                "doc_id": "doc-001",
                "source": "TS23.502.docx",
                "title": "Registration Procedure",
                "prose_description": "UE registration with AMF.",
            }
        ]

        result = store.query_dags_by_keyword(["registration"], limit=3)

        assert len(result) == 1
        assert result[0]["dag_id"] == "ts23.502::registration"

    @pytest.mark.unit
    def test_passes_keywords_as_param(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """keywords list is passed as a Cypher parameter."""
        store.query_dags_by_keyword(["registration", "AMF"], limit=5)

        call_params = mock_conn.execute_cypher.call_args[0][1]
        assert "registration" in call_params["keywords"]
        assert "AMF" in call_params["keywords"]

    @pytest.mark.unit
    def test_passes_limit_as_param(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """limit is forwarded to the Cypher query as a parameter."""
        store.query_dags_by_keyword(["handover"], limit=2)

        call_params = mock_conn.execute_cypher.call_args[0][1]
        assert call_params["limit"] == 2


# ---------------------------------------------------------------------------
# get_dag_mermaid
# ---------------------------------------------------------------------------


class TestGetDagMermaid:
    """CallFlowDagStore.get_dag_mermaid"""

    @pytest.mark.unit
    def test_returns_mermaid_string_when_found(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """Returns mermaid_content string when the DAG exists."""
        mock_conn.execute_cypher.return_value = [
            {"mermaid_content": _SAMPLE_MERMAID}
        ]

        result = store.get_dag_mermaid("ts23.502::registration")

        assert result == _SAMPLE_MERMAID

    @pytest.mark.unit
    def test_returns_none_when_not_found(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """Returns None when the DAG does not exist in Kuzu."""
        mock_conn.execute_cypher.return_value = []

        result = store.get_dag_mermaid("nonexistent::dag")

        assert result is None

    @pytest.mark.unit
    def test_passes_dag_id_as_param(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """dag_id is passed as a Cypher parameter."""
        store.get_dag_mermaid("ts23.502::registration")

        call_params = mock_conn.execute_cypher.call_args[0][1]
        assert call_params["dag_id"] == "ts23.502::registration"


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestDagStoreHealthCheck:
    """CallFlowDagStore.health_check"""

    @pytest.mark.unit
    def test_delegates_to_connection_health_check(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """health_check delegates to the underlying KuzuConnection."""
        mock_conn.health_check.return_value = True

        result = store.health_check()

        mock_conn.health_check.assert_called_once()
        assert result is True

    @pytest.mark.unit
    def test_returns_false_when_connection_fails(
        self, store: CallFlowDagStore, mock_conn: MagicMock
    ) -> None:
        """health_check returns False when connection reports failure."""
        mock_conn.health_check.return_value = False

        result = store.health_check()

        assert result is False
