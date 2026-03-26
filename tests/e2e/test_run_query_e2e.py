"""
End-to-end tests for the run_query() pipeline.

These tests exercise the full LangGraph workflow from question to final state,
using mocked LLM and store to avoid real API calls or disk I/O.

All mocks are injected at the node boundary — not at the LangGraph layer —
so the conditional routing and state transitions are exercised for real.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import RetrievedChunk
from specagent.graph.workflow import run_query


def _make_chunk(
    source: str = "TS38.321.docx",
    content: str = "The maximum number of HARQ processes for NR is 16.",
    section: str = "5.4 HARQ Entity",
    similarity_score: float = 0.85,
) -> RetrievedChunk:
    """Build a minimal RetrievedChunk for testing."""
    return RetrievedChunk(
        content=content,
        chunk_id=f"{source}:0",
        doc_id="doc-uuid-1",
        source=source,
        title=source.replace(".docx", ""),
        chunk_index=0,
        file_type="docx",
        spec_id=source.replace(".docx", ""),
        section=section,
        similarity_score=similarity_score,
    )


@pytest.mark.e2e
class TestRunQueryE2E:
    """Full pipeline E2E tests with mocked LLM and store."""

    @patch("specagent.nodes.hallucination.create_llm")
    @patch("specagent.nodes.generator.create_llm")
    @patch("specagent.nodes.router.create_llm")
    @patch("specagent.nodes.retriever.get_store")
    @patch("specagent.nodes.retriever.get_embedder")
    def test_happy_path_full_pipeline(
        self,
        mock_embedder,
        mock_store,
        mock_router_llm_factory,
        mock_generator_llm_factory,
        mock_hallucination_llm_factory,
    ):
        """Happy path: question → retrieve → auto-grade → generate → grounded → answer.

        Chunk at similarity=0.85 is auto-graded (>0.82 threshold), so the
        grader node makes no LLM call. All 5 nodes execute and timings are
        recorded.
        """
        # Router: route=retrieve
        mock_router_llm = MagicMock()
        mock_router_llm.invoke.return_value = '{"route": "retrieve", "reasoning": "3GPP question"}'
        mock_router_llm_factory.return_value = mock_router_llm

        # Embedder: return a 768d vector
        mock_embedder.return_value.embed.return_value = iter([[0.1] * 768])

        # Store: return one high-similarity chunk (auto-graded, no LLM needed)
        chunk = _make_chunk(similarity_score=0.85)
        mock_store.return_value.search.return_value = [(
            MagicMock(
                id="id1",
                doc_id="doc-uuid-1",
                library="3gpp-specs",
                source="TS38.321.docx",
                content_hash="abc",
                title="TS38.321",
                content=chunk.content,
                embedding=[0.1] * 768,
                chunk_index=0,
                created_at="2026-01-01",
                metadata=json.dumps({"section_header": "5.4 HARQ Entity"}),
                file_type="docx",
                last_modified="2026-01-01",
                page=0,
            ),
            0.85,
        )]

        # Generator: return answer with citation
        mock_generator_llm = MagicMock()
        mock_generator_llm.invoke.return_value = (
            "The maximum number of HARQ processes in NR is 16. [TS 38.321 §5.4]"
        )
        mock_generator_llm_factory.return_value = mock_generator_llm

        # Hallucination check: grounded
        mock_hallucination_llm = MagicMock()
        mock_hallucination_llm.invoke.return_value = (
            '{"grounded": "yes", "ungrounded_claims": []}'
        )
        mock_hallucination_llm_factory.return_value = mock_hallucination_llm

        state = run_query("What is the maximum number of HARQ processes in NR?")

        assert state.get("route_decision") == "retrieve"
        assert state.get("generation") is not None
        assert "16" in state["generation"]
        assert state.get("hallucination_check") in ("grounded", "not_grounded", "partial")
        assert state.get("error") is None

    @patch("specagent.nodes.router.create_llm")
    def test_rejection_path(self, mock_router_llm_factory):
        """Off-topic question is rejected by the router — graph ends at router."""
        mock_router_llm = MagicMock()
        mock_router_llm.invoke.return_value = (
            '{"route": "reject", "reasoning": "Not a 3GPP question"}'
        )
        mock_router_llm_factory.return_value = mock_router_llm

        state = run_query("What is the best recipe for chocolate cake?")

        assert state.get("route_decision") == "reject"
        assert state.get("generation") is None
        assert state.get("retrieved_chunks", []) == []

    @patch("specagent.nodes.hallucination.create_llm")
    @patch("specagent.nodes.generator.create_llm")
    @patch("specagent.nodes.router.create_llm")
    @patch("specagent.nodes.retriever.get_store")
    @patch("specagent.nodes.retriever.get_embedder")
    def test_node_timings_populated(
        self,
        mock_embedder,
        mock_store,
        mock_router_llm_factory,
        mock_generator_llm_factory,
        mock_hallucination_llm_factory,
    ):
        """node_timings dict is populated with entries for each executed node."""
        mock_router_llm = MagicMock()
        mock_router_llm.invoke.return_value = '{"route": "retrieve", "reasoning": "ok"}'
        mock_router_llm_factory.return_value = mock_router_llm

        mock_embedder.return_value.embed.return_value = iter([[0.1] * 768])

        chunk = _make_chunk(similarity_score=0.85)
        mock_store.return_value.search.return_value = [(
            MagicMock(
                id="id1", doc_id="doc-uuid-1", library="3gpp-specs",
                source="TS38.321.docx", content_hash="abc", title="TS38.321",
                content=chunk.content, embedding=[0.1] * 768, chunk_index=0,
                created_at="2026-01-01",
                metadata=json.dumps({"section_header": "5.4 HARQ Entity"}),
                file_type="docx", last_modified="2026-01-01", page=0,
            ),
            0.85,
        )]

        mock_generator_llm = MagicMock()
        mock_generator_llm.invoke.return_value = "HARQ is 16. [TS 38.321 §5.4]"
        mock_generator_llm_factory.return_value = mock_generator_llm

        mock_hallucination_llm = MagicMock()
        mock_hallucination_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_hallucination_llm_factory.return_value = mock_hallucination_llm

        state = run_query("What is the maximum number of HARQ processes?")

        timings = state.get("node_timings", {})
        assert isinstance(timings, dict)
        assert len(timings) >= 4, f"Expected ≥4 node timings, got: {list(timings.keys())}"
        for node, ms in timings.items():
            assert isinstance(ms, float), f"Timing for {node!r} should be float, got {type(ms)}"
            assert ms >= 0.0

    @patch("specagent.nodes.hallucination.create_llm")
    @patch("specagent.nodes.generator.create_llm")
    @patch("specagent.nodes.grader.create_llm")
    @patch("specagent.nodes.router.create_llm")
    @patch("specagent.nodes.retriever.get_store")
    @patch("specagent.nodes.retriever.get_embedder")
    def test_grader_llm_path(
        self,
        mock_embedder,
        mock_store,
        mock_router_llm_factory,
        mock_grader_llm_factory,
        mock_generator_llm_factory,
        mock_hallucination_llm_factory,
    ):
        """Chunk at similarity=0.65 triggers the LLM grading path (0.55–0.82 range)."""
        mock_router_llm = MagicMock()
        mock_router_llm.invoke.return_value = '{"route": "retrieve", "reasoning": "ok"}'
        mock_router_llm_factory.return_value = mock_router_llm

        mock_embedder.return_value.embed.return_value = iter([[0.1] * 768])

        # similarity=0.65 → falls into 0.55–0.82 band → LLM grading required
        chunk = _make_chunk(similarity_score=0.65)
        mock_store.return_value.search.return_value = [(
            MagicMock(
                id="id1", doc_id="doc-uuid-1", library="3gpp-specs",
                source="TS38.321.docx", content_hash="abc", title="TS38.321",
                content=chunk.content, embedding=[0.1] * 768, chunk_index=0,
                created_at="2026-01-01",
                metadata=json.dumps({"section_header": "5.4 HARQ Entity"}),
                file_type="docx", last_modified="2026-01-01", page=0,
            ),
            0.65,
        )]

        # Grader LLM: grades chunk as relevant
        mock_grader_llm = MagicMock()
        mock_grader_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.80}]}'
        mock_grader_llm_factory.return_value = mock_grader_llm

        mock_generator_llm = MagicMock()
        mock_generator_llm.invoke.return_value = "HARQ is 16. [TS 38.321 §5.4]"
        mock_generator_llm_factory.return_value = mock_generator_llm

        mock_hallucination_llm = MagicMock()
        mock_hallucination_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_hallucination_llm_factory.return_value = mock_hallucination_llm

        state = run_query("What is the maximum number of HARQ processes?")

        assert state.get("generation") is not None
        # Grader LLM was actually called (once, for the single mid-range chunk)
        mock_grader_llm.invoke.assert_called_once()
        graded_chunks = state.get("graded_chunks", [])
        assert len(graded_chunks) == 1
        assert graded_chunks[0].relevant == "yes"
