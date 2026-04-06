"""Unit tests for rewriter node."""

from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import GraphState, RetrievedChunk, create_initial_state
from specagent.nodes.rewriter import rewriter_node


@pytest.mark.unit
class TestRewriterNode:
    """Tests for rewriter_node function."""

    def _create_state_with_low_confidence(
        self, question: str, rewrite_count: int = 0, chunks_data: list[dict] | None = None
    ) -> GraphState:
        """Helper to create state with low confidence graded chunks."""
        state = create_initial_state(question)
        state["route_decision"] = "retrieve"
        state["rewrite_count"] = rewrite_count
        state["average_confidence"] = 0.4  # Below threshold

        if chunks_data is None:
            chunks_data = [
                {"content": "The PDCCH carries downlink control information."},
                {"content": "Component carriers can be aggregated."},
            ]

        state["retrieved_chunks"] = [
            RetrievedChunk(
                content=chunk["content"],
                chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                doc_id="doc-uuid",
                source=chunk.get("source", "TS38.211.docx"),
                title=chunk.get("title", "TS 38.211"),
                chunk_index=chunk.get("chunk_index", 0),
                file_type=chunk.get("file_type", "docx"),
                spec_id=chunk.get("spec_id", "TS38.211"),
                section=chunk.get("section", "7.3"),
                similarity_score=chunk.get("similarity_score", 0.5),
            )
            for i, chunk in enumerate(chunks_data)
        ]
        return state

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_successful_rewrite(self, mock_create_llm):
        """Test rewriter successfully rewrites query."""
        # Mock LLM to return rewritten question
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "What is the maximum number of HARQ processes in 5G NR Release 18?"
        )
        mock_create_llm.return_value = mock_llm

        # Create state with low confidence (needs rewriting)
        state = self._create_state_with_low_confidence(
            "What is the max HARQ processes?", rewrite_count=0
        )

        # Call rewriter node
        result = rewriter_node(state)

        # Verify rewritten question is set
        assert (
            result["rewritten_question"]
            == "What is the maximum number of HARQ processes in 5G NR Release 18?"
        )

        # Verify rewrite count is incremented
        assert result["rewrite_count"] == 1

        # Verify LLM was called
        mock_llm.invoke.assert_called_once()

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_max_rewrites_reached(self, mock_create_llm):
        """Test rewriter stops when max rewrites limit is reached."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten question"
        mock_create_llm.return_value = mock_llm

        # Create state with rewrite_count at max (default is 1)
        state = self._create_state_with_low_confidence("What is HARQ?", rewrite_count=1)

        # Call rewriter node
        result = rewriter_node(state)

        # Verify no rewriting happened
        assert result["rewritten_question"] is None
        assert result["rewrite_count"] == 1  # Unchanged

        # Verify LLM was NOT called
        mock_llm.invoke.assert_not_called()

    @patch("specagent.nodes.rewriter.settings.max_rewrites", 2)
    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_increments_count_correctly(self, mock_create_llm):
        """Test rewriter increments rewrite count from non-zero."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Improved question about HARQ processes"
        mock_create_llm.return_value = mock_llm

        # Start with rewrite_count=1
        state = self._create_state_with_low_confidence("HARQ processes?", rewrite_count=1)

        result = rewriter_node(state)

        # Should increment to 2
        assert result["rewrite_count"] == 2
        assert result["rewritten_question"] == "Improved question about HARQ processes"

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_uses_original_question(self, mock_create_llm):
        """Test that rewriter uses original question in prompt."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten"
        mock_create_llm.return_value = mock_llm

        question = "What are the UE capabilities for carrier aggregation?"
        state = self._create_state_with_low_confidence(question)

        rewriter_node(state)

        # Verify the prompt contains the original question
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]
        assert question in prompt

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_includes_chunk_context(self, mock_create_llm):
        """Test that rewriter includes retrieved chunks in prompt."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten"
        mock_create_llm.return_value = mock_llm

        chunks = [
            {"content": "The PDCCH carries DCI information."},
            {"content": "Carrier aggregation allows multiple carriers."},
        ]
        state = self._create_state_with_low_confidence(
            "What is carrier aggregation?", chunks_data=chunks
        )

        rewriter_node(state)

        # Verify the prompt contains chunk summaries
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]
        # Should contain some reference to retrieved chunks
        assert "PDCCH" in prompt or "Carrier aggregation" in prompt

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_handles_llm_error(self, mock_create_llm):
        """Test rewriter handles LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        mock_create_llm.return_value = mock_llm

        state = self._create_state_with_low_confidence("Test question")

        result = rewriter_node(state)

        # Should populate error field
        assert "error" in result
        assert result["error"] is not None
        assert "error" in result["error"].lower()

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_preserves_other_state_fields(self, mock_create_llm):
        """Test that rewriter only modifies rewrite fields."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten question"
        mock_create_llm.return_value = mock_llm

        state = self._create_state_with_low_confidence("Test question")
        state["route_reasoning"] = "Test routing"
        state["average_confidence"] = 0.3

        result = rewriter_node(state)

        # Verify other fields are preserved
        assert result["question"] == "Test question"
        assert result["route_reasoning"] == "Test routing"
        assert result["average_confidence"] == 0.3

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_with_empty_chunks(self, mock_create_llm):
        """Test rewriter handles empty retrieved chunks."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten with no context"
        mock_create_llm.return_value = mock_llm

        state = self._create_state_with_low_confidence("Test question", chunks_data=[])

        result = rewriter_node(state)

        # Should still rewrite, just without chunk context
        assert result["rewritten_question"] == "Rewritten with no context"
        assert result["rewrite_count"] == 1

    @patch("specagent.nodes.rewriter.settings.max_rewrites", 2)
    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_limit_minus_one(self, mock_create_llm):
        """Test rewriter allows rewrite when count is one below limit."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Final rewrite"
        mock_create_llm.return_value = mock_llm

        # At count=1, should allow one more rewrite (max is 2)
        state = self._create_state_with_low_confidence("Test question", rewrite_count=1)

        result = rewriter_node(state)

        # Should allow this rewrite
        assert result["rewritten_question"] == "Final rewrite"
        assert result["rewrite_count"] == 2

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_strips_whitespace(self, mock_create_llm):
        """Test rewriter strips extra whitespace from LLM output."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "  \n  What is the maximum number of HARQ processes?  \n  "
        mock_create_llm.return_value = mock_llm

        state = self._create_state_with_low_confidence("What is HARQ?")

        result = rewriter_node(state)

        # Should strip whitespace
        assert result["rewritten_question"] == "What is the maximum number of HARQ processes?"

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_uses_correct_settings(self, mock_create_llm):
        """Test that rewriter uses correct LLM settings from config."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten"
        mock_create_llm.return_value = mock_llm

        state = self._create_state_with_low_confidence("Test question")

        rewriter_node(state)

        # Verify create_llm was called
        mock_create_llm.assert_called_once()

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_prompt_format(self, mock_create_llm):
        """Test that rewriter prompt contains all necessary components."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten"
        mock_create_llm.return_value = mock_llm

        question = "What is HARQ?"
        chunks = [{"content": "HARQ is a retransmission protocol."}]
        state = self._create_state_with_low_confidence(question, chunks_data=chunks)

        rewriter_node(state)

        prompt = mock_llm.invoke.call_args[0][0]

        # Verify prompt structure
        assert question in prompt
        assert "HARQ" in prompt  # From chunk content
        assert any(keyword in prompt.lower() for keyword in ["rewrite", "reformulate", "specific"])

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_handles_non_string_llm_response(self, mock_create_llm):
        """Test that rewriter handles non-string LLM response."""
        mock_llm = MagicMock()
        # Mock LLM returns a non-string (e.g., dict or object)
        mock_llm.invoke.return_value = {"text": "Some response"}
        mock_create_llm.return_value = mock_llm

        state = self._create_state_with_low_confidence("Test question")

        result = rewriter_node(state)

        # Should still set the rewritten_question even if it's not a string
        assert result["rewritten_question"] == {"text": "Some response"}
        assert result["rewrite_count"] == 1

    def test_rewriter_prompt_contains_few_shot_examples(self):
        """Test that REWRITER_PROMPT contains few-shot examples."""
        from specagent.nodes.rewriter import REWRITER_PROMPT

        # Verify prompt contains examples section
        assert "Example" in REWRITER_PROMPT or "example" in REWRITER_PROMPT.lower()

        # Verify prompt contains at least 2-3 few-shot examples
        # Look for common patterns in few-shot examples
        prompt_lower = REWRITER_PROMPT.lower()

        # Check for example keywords that indicate few-shot learning
        assert "original:" in prompt_lower or "original question:" in prompt_lower
        assert "rewrite" in prompt_lower or "rewritten:" in prompt_lower

    def test_rewriter_prompt_contains_telecom_terminology_examples(self):
        """Test that few-shot examples use proper 3GPP terminology."""
        from specagent.nodes.rewriter import REWRITER_PROMPT

        # Verify examples contain 3GPP-specific terminology
        prompt_lower = REWRITER_PROMPT.lower()

        # Should contain at least some telecom-specific terms in examples
        telecom_terms = ["rrc", "drx", "handover", "5g nr", "3gpp", "ts ", "nr ran"]
        found_terms = [term for term in telecom_terms if term in prompt_lower]

        # Should have at least 2 telecom terms in the examples
        assert len(found_terms) >= 2, f"Expected at least 2 telecom terms, found: {found_terms}"

    @patch("specagent.nodes.rewriter.get_llm")
    def test_rewriter_prompt_includes_examples_in_actual_call(self, mock_create_llm):
        """Test that few-shot examples are included when calling LLM."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "RRC connection reconfiguration procedure for X2 handover in 5G NR"
        )
        mock_create_llm.return_value = mock_llm

        state = self._create_state_with_low_confidence("What is handover?")

        rewriter_node(state)

        # Get the actual prompt sent to the LLM
        prompt = mock_llm.invoke.call_args[0][0]

        # Verify few-shot examples are in the prompt
        assert "Example" in prompt or "example" in prompt.lower()

        # Verify examples help guide the rewriting style
        prompt_lower = prompt.lower()
        assert "original" in prompt_lower or "rewrite" in prompt_lower


# ──────────────────────────────────────────────────────────────────────────────
# Failing tests for bug fixes
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestRewriterBugFixes:
    """Regression tests for rewriter bug fixes."""

    @patch("specagent.nodes.rewriter.get_llm")
    def test_chunk_preview_with_curly_braces_does_not_raise(self, mock_create_llm):
        """Chunk content preview containing '{...}' must not cause str.format() KeyError."""
        from specagent.graph.state import RetrievedChunk, create_initial_state  # noqa: PLC0415
        from specagent.nodes.rewriter import rewriter_node  # noqa: PLC0415

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten: resource block configuration {nrb} in NR"
        mock_llm.get_last_call.return_value = None
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("nrb?")
        state["retrieved_chunks"] = [
            RetrievedChunk(
                content="The parameter {nrb} specifies the resource block count per subcarrier spacing.",
                chunk_id="c1",
                doc_id="d1",
                source="TS38.211.docx",
                title="TS38.211",
                chunk_index=0,
                file_type="docx",
                spec_id="TS38.211",
                section="4.4",
                similarity_score=0.7,
            )
        ]

        # Must not raise KeyError
        result = rewriter_node(state)
        assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
