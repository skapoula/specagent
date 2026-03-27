"""Unit tests for retrieval observability: RetrievalRecord capture in retriever_node."""

import json
from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import create_initial_state


def _make_mock_record(source: str = "TS38.321.docx", section: str = "5.4") -> MagicMock:
    """Return a mock raw LanceDB record."""
    r = MagicMock()
    r.id = "chunk-1"
    r.doc_id = "doc-1"
    r.source = source
    r.title = "TS 38.321 MAC"
    r.content = "HARQ content."
    r.chunk_index = 0
    r.file_type = "docx"
    r.metadata = json.dumps({"section_header": section})
    return r


def _run_retriever(state, search_results):
    from specagent.nodes.retriever import retriever_node  # noqa: PLC0415

    mock_emb = MagicMock()
    mock_emb.embed.return_value = iter([[0.1] * 768])
    mock_store = MagicMock()
    mock_store.search.return_value = search_results

    with (
        patch("specagent.nodes.retriever.get_embedder", return_value=mock_emb),
        patch("specagent.nodes.retriever.get_store", return_value=mock_store),
    ):
        return retriever_node(state)


@pytest.mark.unit
class TestRetrieverObservability:
    def test_appends_retrieval_record(self):
        state = create_initial_state("What is HARQ?")
        r1 = _make_mock_record()
        r2 = _make_mock_record(source="TS38.101.docx", section="5.5")

        result = _run_retriever(state, [(r1, 0.85), (r2, 0.75)])

        events = result.get("retrieval_events", [])
        assert len(events) == 1
        rec = events[0]
        assert rec.num_results == 2
        assert rec.top_similarity == pytest.approx(0.85)
        assert rec.mean_similarity == pytest.approx(0.80)
        assert rec.embed_ms >= 0.0
        assert rec.search_ms >= 0.0
        assert rec.rewrite_index == 0

    def test_accumulates_across_rewrites(self):
        state = create_initial_state("HARQ?")
        r = _make_mock_record()

        state = _run_retriever(state, [(r, 0.80)])
        assert len(state["retrieval_events"]) == 1

        state["rewrite_count"] = 1
        state = _run_retriever(state, [(r, 0.80)])
        assert len(state["retrieval_events"]) == 2
        assert state["retrieval_events"][1].rewrite_index == 1

    def test_empty_results_null_similarity(self):
        state = create_initial_state("HARQ?")
        result = _run_retriever(state, [])

        events = result.get("retrieval_events", [])
        assert len(events) == 1
        assert events[0].top_similarity is None
        assert events[0].mean_similarity is None
        assert events[0].num_results == 0

    def test_trace_id_propagated(self):
        state = create_initial_state("HARQ?")
        r = _make_mock_record()

        result = _run_retriever(state, [(r, 0.88)])
        rec = result["retrieval_events"][0]
        assert rec.trace_id == state["trace_id"]
        assert len(rec.trace_id) == 36  # UUID4
