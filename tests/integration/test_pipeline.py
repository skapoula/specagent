"""
Integration tests for the RAG pipeline.

These tests verify that nodes work correctly together.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestRoutingDecisions:
    """Pure function tests for routing decisions — no LLM, no I/O."""

    def test_low_confidence_triggers_rewrite(self, initial_graph_state):
        """Low grader confidence should trigger rewriter."""
        from specagent.graph.workflow import should_rewrite

        state = initial_graph_state.copy()
        state["average_confidence"] = 0.4  # Below threshold
        state["rewrite_count"] = 0

        decision = should_rewrite(state)

        assert decision == "rewrite"

    def test_high_confidence_skips_rewrite(self, initial_graph_state):
        """High grader confidence should skip rewriter."""
        from specagent.graph.workflow import should_rewrite

        state = initial_graph_state.copy()
        state["average_confidence"] = 0.8  # Above threshold
        state["rewrite_count"] = 0

        decision = should_rewrite(state)

        assert decision == "generate"


@pytest.mark.integration
class TestRetrievalPipeline:
    """Tests for the retrieval pipeline (retriever -> grader)."""

    @patch("specagent.nodes.grader.get_llm")
    def test_retriever_to_grader_flow(self, mock_create_llm, state_after_retrieval):
        """Retrieved chunks should flow to grader correctly."""
        from specagent.nodes import grader_node

        mock_llm = MagicMock()
        # scores 0.85 (auto-graded), 0.75 and 0.65 (mid-range → LLM): 2 grades needed
        mock_llm.invoke.return_value = (
            '{"grades": ['
            '{"relevant": "yes", "confidence": 0.75},'
            '{"relevant": "yes", "confidence": 0.65}'
            "]}"
        )
        mock_create_llm.return_value = mock_llm

        result = grader_node(state_after_retrieval)

        assert "graded_chunks" in result
        assert len(result["graded_chunks"]) > 0


@pytest.mark.integration
class TestGenerationPipeline:
    """Tests for the generation pipeline (generator -> hallucination check)."""

    @patch("specagent.nodes.generator.get_llm")
    def test_generator_produces_citations(self, mock_create_llm, state_after_retrieval):
        """Generator should include citations in output."""
        from specagent.graph.state import GradedChunk, RetrievedChunk
        from specagent.nodes import generator_node

        # Populate graded_chunks so the generator has relevant sources
        state = state_after_retrieval.copy()
        state["graded_chunks"] = [
            GradedChunk(
                chunk=RetrievedChunk(
                    content="The maximum number of HARQ processes for NR is 16.",
                    chunk_id="TS38.321.docx:0",
                    doc_id="doc-uuid-1",
                    source="TS38.321.docx",
                    title="TS 38.321 MAC Protocol",
                    chunk_index=0,
                    file_type="docx",
                    spec_id="TS38.321",
                    section="5.4 HARQ Entity",
                    similarity_score=0.85,
                ),
                relevant="yes",
                confidence=0.85,
            )
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "The maximum number of HARQ processes in NR is 16. [TS 38.321 §5.4]"
        )
        mock_create_llm.return_value = mock_llm

        result = generator_node(state)

        assert "generation" in result
        assert "citations" in result
        assert len(result["citations"]) > 0

    @patch("specagent.nodes.hallucination.get_llm")
    @patch("specagent.nodes.generator.get_llm")
    def test_grounded_answer_passes_check(self, mock_gen_llm, mock_hall_llm, state_after_retrieval):
        """Grounded answer should pass hallucination check."""
        from specagent.nodes import generator_node, hallucination_check_node

        mock_gen = MagicMock()
        mock_gen.invoke.return_value = (
            "The maximum number of HARQ processes in NR is 16. [TS 38.321 §5.4]"
        )
        mock_gen_llm.return_value = mock_gen

        mock_hall = MagicMock()
        mock_hall.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_hall_llm.return_value = mock_hall

        generated_state = generator_node(state_after_retrieval)
        # Ensure confidence is high enough to trigger the hallucination LLM call
        generated_state["average_confidence"] = 0.0
        result = hallucination_check_node(generated_state)

        assert result["hallucination_check"] == "grounded"
