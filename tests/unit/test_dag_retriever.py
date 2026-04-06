"""Unit tests for the DAG retriever node and route_after_retriever conditional edge.

Tests are written FIRST (TDD). Implementation does not exist yet.

Covers:
- GraphState.dag_chunks field exists
- route_after_retriever keyword heuristic (call-flow queries → dag_retriever)
- route_after_retriever bypass (non-call-flow queries, DAG disabled)
- dag_retriever_node populates state["dag_chunks"]
- dag_retriever_node graceful degradation when Memgraph is down
- Generator prompt receives dag_chunks (separate lane)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import GraphState, RetrievedChunk, create_initial_state
from specagent.nodes.dag_retriever import dag_retriever_node, route_after_retriever
from specagent.retrieval.exceptions import DagStoreError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(question: str, dag_retrieval_enabled: bool = True) -> GraphState:
    state = create_initial_state(question)
    state["dag_chunks"] = []
    return state


def _make_retrieved_chunk(content: str = "Some content") -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        chunk_id="chunk-001",
        doc_id="doc-001",
        source="TS23.502.docx",
        title="TS 23.502",
        chunk_index=0,
        file_type="docx",
        spec_id="TS23.502",
        section="4.2",
        similarity_score=0.80,
    )


# ---------------------------------------------------------------------------
# GraphState.dag_chunks field
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_graphstate_has_dag_chunks_field() -> None:
    """GraphState TypedDict includes a dag_chunks field."""
    state = create_initial_state("What is the registration procedure?")
    # dag_chunks is initialised in create_initial_state
    assert "dag_chunks" in state
    assert isinstance(state["dag_chunks"], list)


@pytest.mark.unit
def test_dag_chunks_defaults_to_empty_list() -> None:
    """dag_chunks is an empty list in the initial state."""
    state = create_initial_state("Some question")
    assert state["dag_chunks"] == []


# ---------------------------------------------------------------------------
# route_after_retriever — keyword heuristic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_route_returns_dag_retriever_for_procedure_query() -> None:
    """Query containing 'procedure' routes to dag_retriever."""
    state = _make_state("What is the registration procedure for 5G NR?")

    with patch("specagent.nodes.dag_retriever.settings") as mock_settings:
        mock_settings.enable_dag_retrieval = True
        result = route_after_retriever(state)

    assert result == "dag_retriever"


@pytest.mark.unit
def test_route_returns_dag_retriever_for_call_flow_query() -> None:
    """Query containing 'call flow' routes to dag_retriever."""
    state = _make_state("Show me the call flow for UE authentication")

    with patch("specagent.nodes.dag_retriever.settings") as mock_settings:
        mock_settings.enable_dag_retrieval = True
        result = route_after_retriever(state)

    assert result == "dag_retriever"


@pytest.mark.unit
def test_route_returns_dag_retriever_for_sequence_query() -> None:
    """Query containing 'sequence' routes to dag_retriever."""
    state = _make_state("Describe the message sequence during PDU session establishment")

    with patch("specagent.nodes.dag_retriever.settings") as mock_settings:
        mock_settings.enable_dag_retrieval = True
        result = route_after_retriever(state)

    assert result == "dag_retriever"


@pytest.mark.unit
def test_route_returns_dag_retriever_for_participant_name_query() -> None:
    """Query mentioning 3GPP participants (AMF, UE, gNB) routes to dag_retriever."""
    state = _make_state("What happens when the UE sends a Registration Request to the AMF?")

    with patch("specagent.nodes.dag_retriever.settings") as mock_settings:
        mock_settings.enable_dag_retrieval = True
        result = route_after_retriever(state)

    assert result == "dag_retriever"


@pytest.mark.unit
def test_route_returns_grader_for_conceptual_query() -> None:
    """Conceptual query with no procedure keywords routes to grader (bypass)."""
    state = _make_state("What is the maximum number of HARQ processes in NR?")

    with patch("specagent.nodes.dag_retriever.settings") as mock_settings:
        mock_settings.enable_dag_retrieval = True
        result = route_after_retriever(state)

    assert result == "grader"


@pytest.mark.unit
def test_route_returns_grader_when_dag_retrieval_disabled() -> None:
    """route_after_retriever always returns 'grader' when enable_dag_retrieval=False."""
    state = _make_state("Show me the registration procedure")

    with patch("specagent.nodes.dag_retriever.settings") as mock_settings:
        mock_settings.enable_dag_retrieval = False
        result = route_after_retriever(state)

    assert result == "grader"


@pytest.mark.unit
def test_route_returns_grader_for_empty_question() -> None:
    """Empty question routes to grader (no keyword match possible)."""
    state = _make_state("")

    with patch("specagent.nodes.dag_retriever.settings") as mock_settings:
        mock_settings.enable_dag_retrieval = True
        result = route_after_retriever(state)

    assert result == "grader"


# ---------------------------------------------------------------------------
# dag_retriever_node — normal operation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dag_retriever_node_populates_dag_chunks() -> None:
    """dag_retriever_node populates state['dag_chunks'] with RetrievedChunk objects."""
    state = _make_state("Show me the UE registration call flow")

    mock_dag_store = MagicMock()
    mock_dag_store.query_dags_by_keyword.return_value = [
        {
            "dag_id": "TS23.502::Figure 4.2-1 Registration",
            "doc_id": "doc-001",
            "source": "TS23.502.docx",
            "title": "Figure 4.2-1 Registration",
            "prose_description": "UE registration with AMF.",
        }
    ]
    mock_dag_store.get_dag_mermaid.return_value = (
        "```mermaid\nsequenceDiagram\n    UE->>AMF: Registration Request\n```"
    )

    with (
        patch("specagent.nodes.dag_retriever.get_dag_store", return_value=mock_dag_store),
        patch("specagent.nodes.dag_retriever.settings") as mock_settings,
    ):
        mock_settings.enable_dag_retrieval = True
        mock_settings.dag_retrieval_top_k = 1
        mock_settings.dag_retrieval_score = 0.70

        result = dag_retriever_node(state)

    assert len(result["dag_chunks"]) == 1
    chunk = result["dag_chunks"][0]
    assert isinstance(chunk, RetrievedChunk)
    assert "sequenceDiagram" in chunk.content


@pytest.mark.unit
def test_dag_retriever_node_chunk_has_correct_score() -> None:
    """DAG chunks receive the configured dag_retrieval_score."""
    state = _make_state("Registration procedure steps")

    mock_dag_store = MagicMock()
    mock_dag_store.query_dags_by_keyword.return_value = [
        {
            "dag_id": "TS23.502::Fig1",
            "doc_id": "doc-001",
            "source": "TS23.502.docx",
            "title": "Fig1",
            "prose_description": "Registration.",
        }
    ]
    mock_dag_store.get_dag_mermaid.return_value = "```mermaid\nsequenceDiagram\n```"

    with (
        patch("specagent.nodes.dag_retriever.get_dag_store", return_value=mock_dag_store),
        patch("specagent.nodes.dag_retriever.settings") as mock_settings,
    ):
        mock_settings.enable_dag_retrieval = True
        mock_settings.dag_retrieval_top_k = 1
        mock_settings.dag_retrieval_score = 0.70

        result = dag_retriever_node(state)

    assert result["dag_chunks"][0].similarity_score == pytest.approx(0.70)


@pytest.mark.unit
def test_dag_retriever_node_chunk_section_is_call_flow() -> None:
    """DAG chunks have section='Call Flow Diagram'."""
    state = _make_state("Registration call flow")

    mock_dag_store = MagicMock()
    mock_dag_store.query_dags_by_keyword.return_value = [
        {
            "dag_id": "TS23.502::Fig1",
            "doc_id": "doc-001",
            "source": "TS23.502.docx",
            "title": "Fig1",
            "prose_description": "",
        }
    ]
    mock_dag_store.get_dag_mermaid.return_value = "```mermaid\nsequenceDiagram\n```"

    with (
        patch("specagent.nodes.dag_retriever.get_dag_store", return_value=mock_dag_store),
        patch("specagent.nodes.dag_retriever.settings") as mock_settings,
    ):
        mock_settings.enable_dag_retrieval = True
        mock_settings.dag_retrieval_top_k = 1
        mock_settings.dag_retrieval_score = 0.70

        result = dag_retriever_node(state)

    assert result["dag_chunks"][0].section == "Call Flow Diagram"


@pytest.mark.unit
def test_dag_retriever_node_returns_empty_when_no_match() -> None:
    """dag_retriever_node returns empty dag_chunks when no DAGs match."""
    state = _make_state("PDU session establishment sequence")

    mock_dag_store = MagicMock()
    mock_dag_store.query_dags_by_keyword.return_value = []

    with (
        patch("specagent.nodes.dag_retriever.get_dag_store", return_value=mock_dag_store),
        patch("specagent.nodes.dag_retriever.settings") as mock_settings,
    ):
        mock_settings.enable_dag_retrieval = True
        mock_settings.dag_retrieval_top_k = 1
        mock_settings.dag_retrieval_score = 0.70

        result = dag_retriever_node(state)

    assert result["dag_chunks"] == []


@pytest.mark.unit
def test_dag_retriever_node_does_not_modify_retrieved_chunks() -> None:
    """dag_retriever_node never touches state['retrieved_chunks']."""
    existing_chunk = _make_retrieved_chunk("Vector chunk content")
    state = _make_state("Registration procedure")
    state["retrieved_chunks"] = [existing_chunk]

    mock_dag_store = MagicMock()
    mock_dag_store.query_dags_by_keyword.return_value = [
        {
            "dag_id": "TS23.502::Fig1",
            "doc_id": "doc-001",
            "source": "TS23.502.docx",
            "title": "Fig1",
            "prose_description": "",
        }
    ]
    mock_dag_store.get_dag_mermaid.return_value = "```mermaid\nsequenceDiagram\n```"

    with (
        patch("specagent.nodes.dag_retriever.get_dag_store", return_value=mock_dag_store),
        patch("specagent.nodes.dag_retriever.settings") as mock_settings,
    ):
        mock_settings.enable_dag_retrieval = True
        mock_settings.dag_retrieval_top_k = 1
        mock_settings.dag_retrieval_score = 0.70

        result = dag_retriever_node(state)

    # Vector chunks must be untouched
    assert result["retrieved_chunks"] == [existing_chunk]
    # DAG chunks in separate lane
    assert len(result["dag_chunks"]) == 1


# ---------------------------------------------------------------------------
# dag_retriever_node — graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dag_retriever_node_graceful_on_dag_store_exception() -> None:
    """dag_retriever_node does not raise when dag_store raises; dag_chunks stays empty."""
    state = _make_state("Registration call flow")

    mock_dag_store = MagicMock()
    mock_dag_store.query_dags_by_keyword.side_effect = DagStoreError("Memgraph unavailable")

    with (
        patch("specagent.nodes.dag_retriever.get_dag_store", return_value=mock_dag_store),
        patch("specagent.nodes.dag_retriever.settings") as mock_settings,
    ):
        mock_settings.enable_dag_retrieval = True
        mock_settings.dag_retrieval_top_k = 1
        mock_settings.dag_retrieval_score = 0.70

        result = dag_retriever_node(state)

    # Node must not raise and dag_chunks must be empty
    assert result["dag_chunks"] == []


@pytest.mark.unit
def test_dag_retriever_node_graceful_when_mermaid_fetch_fails() -> None:
    """If get_dag_mermaid returns None, that DAG is skipped silently."""
    state = _make_state("Registration procedure")

    mock_dag_store = MagicMock()
    mock_dag_store.query_dags_by_keyword.return_value = [
        {
            "dag_id": "TS23.502::Fig1",
            "doc_id": "doc-001",
            "source": "TS23.502.docx",
            "title": "Fig1",
            "prose_description": "",
        }
    ]
    mock_dag_store.get_dag_mermaid.return_value = None  # DAG content not found

    with (
        patch("specagent.nodes.dag_retriever.get_dag_store", return_value=mock_dag_store),
        patch("specagent.nodes.dag_retriever.settings") as mock_settings,
    ):
        mock_settings.enable_dag_retrieval = True
        mock_settings.dag_retrieval_top_k = 1
        mock_settings.dag_retrieval_score = 0.70

        result = dag_retriever_node(state)

    assert result["dag_chunks"] == []
