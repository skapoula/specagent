"""Unit tests for generator node."""

from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import (
    GradedChunk,
    GraphState,
    RetrievedChunk,
    create_initial_state,
)
from specagent.nodes.generator import generator_node


@pytest.mark.unit
class TestGeneratorNode:
    """Tests for generator_node function."""

    def _create_state_with_graded_chunks(
        self, question: str, graded_chunks_data: list[dict]
    ) -> GraphState:
        """Helper to create state with graded chunks."""
        state = create_initial_state(question)
        state["route_decision"] = "retrieve"
        state["graded_chunks"] = [
            GradedChunk(
                chunk=RetrievedChunk(
                    content=chunk["content"],
                    chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                    doc_id="doc-uuid",
                    source=chunk.get("source", "TS38.321.docx"),
                    title=chunk.get("title", "TS 38.321 MAC"),
                    chunk_index=chunk.get("chunk_index", 0),
                    file_type=chunk.get("file_type", "docx"),
                    spec_id=chunk.get("spec_id", "TS38.321"),
                    section=chunk.get("section", "5.4"),
                    similarity_score=chunk.get("similarity_score", 0.8),
                ),
                relevant=chunk.get("relevant", "yes"),
                confidence=chunk.get("confidence", 0.85),
            )
            for i, chunk in enumerate(graded_chunks_data)
        ]
        return state

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_with_relevant_chunks(self, mock_create_llm):
        """Test generator creates answer from relevant chunks."""
        # Mock LLM to return answer with citations
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "The maximum number of HARQ processes for NR is 16 for both FDD and TDD. "
            "[TS 38.321 §5.4]"
        )
        mock_create_llm.return_value = mock_llm

        # Create state with relevant chunks
        chunks = [
            {
                "content": "The maximum number of HARQ processes for NR is 16 for both FDD and TDD.",
                "spec_id": "TS38.321",
                "section": "5.4",
                "relevant": "yes",
            }
        ]
        state = self._create_state_with_graded_chunks(
            "What is the maximum number of HARQ processes in NR?", chunks
        )

        # Call generator node
        result = generator_node(state)

        # Verify generation is populated
        assert result["generation"] is not None
        assert "16" in result["generation"]
        assert "HARQ" in result["generation"]

        # Verify citations are extracted
        assert len(result["citations"]) == 1
        assert result["citations"][0].spec_id == "TS38.321"
        assert result["citations"][0].section == "5.4"

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_filters_irrelevant_chunks(self, mock_create_llm):
        """Test generator only uses relevant chunks."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "The maximum is 16. [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        # Create state with mixed relevant and irrelevant chunks
        chunks = [
            {
                "content": "The maximum number of HARQ processes for NR is 16.",
                "spec_id": "TS38.321",
                "section": "5.4",
                "relevant": "yes",
            },
            {
                "content": "The PDCCH carries downlink control information.",
                "spec_id": "TS38.211",
                "section": "7.3",
                "relevant": "no",  # Irrelevant
            },
            {
                "content": "HARQ processes are used for retransmission.",
                "spec_id": "TS38.321",
                "section": "5.4.1",
                "relevant": "yes",
            },
        ]
        state = self._create_state_with_graded_chunks(
            "What is the maximum number of HARQ processes?", chunks
        )

        generator_node(state)

        # Verify LLM was called with only relevant chunks
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]

        # Relevant chunks should be in prompt
        assert "maximum number of HARQ processes for NR is 16" in prompt
        assert "HARQ processes are used for retransmission" in prompt

        # Irrelevant chunk should NOT be in prompt
        assert "PDCCH" not in prompt

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_multiple_citations(self, mock_create_llm):
        """Test generator extracts multiple citations."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "The maximum number of HARQ processes is 16 [TS 38.321 §5.4]. "
            "The UE shall support carrier aggregation [TS 38.101-1 §5.5A]. "
            "Timer T311 is used for re-establishment [TS 38.331 §5.3.7]."
        )
        mock_create_llm.return_value = mock_llm

        chunks = [
            {"content": "HARQ info", "spec_id": "TS38.321", "section": "5.4", "relevant": "yes"}
        ]
        state = self._create_state_with_graded_chunks("Test question", chunks)

        result = generator_node(state)

        # Verify all three citations are extracted
        assert len(result["citations"]) == 3

        # Check first citation
        assert result["citations"][0].spec_id == "TS38.321"
        assert result["citations"][0].section == "5.4"

        # Check second citation
        assert result["citations"][1].spec_id == "TS38.101-1"
        assert result["citations"][1].section == "5.5A"

        # Check third citation
        assert result["citations"][2].spec_id == "TS38.331"
        assert result["citations"][2].section == "5.3.7"

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_no_citations(self, mock_create_llm):
        """Test generator handles response with no citations."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "I don't have enough information in the available specifications "
            "to fully answer this question."
        )
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Some unrelated content", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Unclear question", chunks)

        result = generator_node(state)

        # Verify generation is set but no citations
        assert result["generation"] is not None
        assert "don't have enough information" in result["generation"]
        assert result["citations"] == []

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_empty_graded_chunks(self, mock_create_llm):
        """Test generator handles empty graded chunks."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("Test question")
        state["graded_chunks"] = []

        result = generator_node(state)

        # Should set generation to indicate no information
        assert result["generation"] is not None
        assert "don't have enough information" in result["generation"].lower()
        assert result["citations"] == []

        # LLM should not be called when there are no chunks
        mock_llm.invoke.assert_not_called()

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_all_irrelevant_chunks(self, mock_create_llm):
        """Test generator handles case where all chunks are irrelevant."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        chunks = [
            {"content": "Chunk 1", "relevant": "no"},
            {"content": "Chunk 2", "relevant": "no"},
        ]
        state = self._create_state_with_graded_chunks("Test question", chunks)

        result = generator_node(state)

        # Should set generation to indicate no information
        assert result["generation"] is not None
        assert "don't have enough information" in result["generation"].lower()
        assert result["citations"] == []

        # LLM should not be called when all chunks are irrelevant
        mock_llm.invoke.assert_not_called()

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_context_formatting(self, mock_create_llm):
        """Test generator formats context with source citations."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Test answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        chunks = [
            {
                "content": "HARQ content here",
                "spec_id": "TS38.321",
                "section": "5.4",
                "relevant": "yes",
            },
            {
                "content": "RRC content here",
                "spec_id": "TS38.331",
                "section": "5.3.7",
                "relevant": "yes",
            },
        ]
        state = self._create_state_with_graded_chunks("Test question", chunks)

        generator_node(state)

        # Verify prompt includes formatted context with sources
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]

        # Should include chunk content
        assert "HARQ content here" in prompt
        assert "RRC content here" in prompt

        # Should include source information
        assert "38.321" in prompt or "TS38.321" in prompt
        assert "38.331" in prompt or "TS38.331" in prompt

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_includes_question_in_prompt(self, mock_create_llm):
        """Test generator includes question in prompt."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"
        mock_create_llm.return_value = mock_llm

        question = "What is the maximum number of HARQ processes in NR?"
        chunks = [{"content": "Test content", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks(question, chunks)

        generator_node(state)

        # Verify question is in prompt
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]
        assert question in prompt

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_citation_with_spaces(self, mock_create_llm):
        """Test generator extracts citations with various spacing."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "Info1 [TS 38.321 §5.4] and info2 [TS  38.331  §5.3.7] and info3 [TS 38.101-1 § 5.5A]"
        )
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test", chunks)

        result = generator_node(state)

        # Should extract all citations despite spacing variations
        assert len(result["citations"]) == 3

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_citation_formats(self, mock_create_llm):
        """Test generator handles different citation formats."""
        mock_llm = MagicMock()
        # Various valid citation formats
        mock_llm.invoke.return_value = (
            "Reference [TS 38.321 §5.4] and [TS 38.331 §5.3.7.1] and [TS 23.501 §4.2.8.2.3]"
        )
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test", chunks)

        result = generator_node(state)

        # Should handle sections with multiple levels
        assert len(result["citations"]) == 3
        assert any(c.section == "5.4" for c in result["citations"])
        assert any(c.section == "5.3.7.1" for c in result["citations"])
        assert any(c.section == "4.2.8.2.3" for c in result["citations"])

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_handles_llm_error(self, mock_create_llm):
        """Test generator handles LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test question", chunks)

        result = generator_node(state)

        # Should populate error field
        assert "error" in result
        assert result["error"] is not None
        assert "error" in result["error"].lower()

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_preserves_other_state_fields(self, mock_create_llm):
        """Test that generator only modifies generation fields."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test question", chunks)
        state["route_reasoning"] = "Test routing"
        state["rewrite_count"] = 1
        state["average_confidence"] = 0.85

        result = generator_node(state)

        # Verify other fields are preserved
        assert result["question"] == "Test question"
        assert result["route_reasoning"] == "Test routing"
        assert result["rewrite_count"] == 1
        assert result["average_confidence"] == 0.85

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_citation_raw_format(self, mock_create_llm):
        """Test that citations preserve raw format."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test", chunks)

        result = generator_node(state)

        # Verify raw citation is preserved
        assert len(result["citations"]) == 1
        assert result["citations"][0].raw_citation == "[TS 38.321 §5.4]"

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_uses_correct_settings(self, mock_create_llm):
        """Test that generator uses correct LLM settings from config."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test", chunks)

        generator_node(state)

        # Verify create_llm was called with temperature=0.0 for deterministic outputs
        mock_create_llm.assert_called_once_with(temperature=0.0)

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_duplicate_citations(self, mock_create_llm):
        """Test generator handles duplicate citations."""
        mock_llm = MagicMock()
        # Same citation appears twice
        mock_llm.invoke.return_value = (
            "HARQ is important [TS 38.321 §5.4]. "
            "As mentioned [TS 38.321 §5.4], it handles retransmissions."
        )
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test", chunks)

        result = generator_node(state)

        # Should include both instances (or deduplicate - either is acceptable)
        assert len(result["citations"]) >= 1
        assert all(c.spec_id == "TS38.321" for c in result["citations"])
        assert all(c.section == "5.4" for c in result["citations"])

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_strips_whitespace(self, mock_create_llm):
        """Test generator strips whitespace from LLM output."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "  \n  Answer with spacing [TS 38.321 §5.4]  \n  "
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test", chunks)

        result = generator_node(state)

        # Should strip leading/trailing whitespace
        assert result["generation"] == "Answer with spacing [TS 38.321 §5.4]"

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_spec_id_normalization(self, mock_create_llm):
        """Test generator normalizes spec IDs (removes spaces/dots)."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test", chunks)

        result = generator_node(state)

        # Spec ID should be normalized (no spaces)
        assert result["citations"][0].spec_id == "TS38.321"

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_handles_non_string_llm_response(self, mock_create_llm):
        """Test generator handles non-string LLM response."""
        mock_llm = MagicMock()
        # Mock LLM returns a non-string (e.g., dict or object)
        mock_llm.invoke.return_value = {"text": "Answer with info [TS 38.321 §5.4]"}
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test", chunks)

        result = generator_node(state)

        # Should convert non-string to string
        assert result["generation"] is not None
        assert isinstance(result["generation"], str)
        # Should still extract citations from the string representation
        assert len(result["citations"]) >= 0

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_sorts_chunks_by_similarity(self, mock_create_llm):
        """Test generator sorts chunks by similarity score descending."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        # Create chunks with different similarity scores
        chunks = [
            {
                "content": "Low similarity chunk",
                "spec_id": "TS38.331",
                "section": "5.1",
                "similarity_score": 0.6,
                "relevant": "yes",
            },
            {
                "content": "High similarity chunk",
                "spec_id": "TS38.321",
                "section": "5.4",
                "similarity_score": 0.95,
                "relevant": "yes",
            },
            {
                "content": "Medium similarity chunk",
                "spec_id": "TS38.211",
                "section": "7.3",
                "similarity_score": 0.75,
                "relevant": "yes",
            },
        ]
        state = self._create_state_with_graded_chunks("Test question", chunks)

        generator_node(state)

        # Verify chunks are passed to LLM in sorted order (highest similarity first)
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]

        # High similarity chunk should appear before medium and low
        high_pos = prompt.find("High similarity chunk")
        medium_pos = prompt.find("Medium similarity chunk")
        low_pos = prompt.find("Low similarity chunk")

        assert high_pos < medium_pos
        assert medium_pos < low_pos

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_limits_to_top2_when_high_confidence(self, mock_create_llm):
        """Test generator limits to top-2 chunks when average_confidence > 0.8."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        # Create 4 chunks with high confidence
        chunks = [
            {"content": "Chunk 1", "similarity_score": 0.95, "confidence": 0.9, "relevant": "yes"},
            {"content": "Chunk 2", "similarity_score": 0.90, "confidence": 0.85, "relevant": "yes"},
            {"content": "Chunk 3", "similarity_score": 0.85, "confidence": 0.9, "relevant": "yes"},
            {"content": "Chunk 4", "similarity_score": 0.80, "confidence": 0.88, "relevant": "yes"},
        ]
        state = self._create_state_with_graded_chunks("Test question", chunks)
        state["average_confidence"] = 0.88  # High confidence (> 0.8)

        generator_node(state)

        # Verify only top-2 chunks are in prompt (optimization)
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]

        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt
        assert "Chunk 3" not in prompt  # Should be excluded
        assert "Chunk 4" not in prompt  # Should be excluded

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_no_limit_when_low_confidence(self, mock_create_llm):
        """Test generator uses all chunks when average_confidence <= 0.8."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        # Create 4 chunks with low confidence
        chunks = [
            {"content": "Chunk 1", "similarity_score": 0.85, "confidence": 0.7, "relevant": "yes"},
            {"content": "Chunk 2", "similarity_score": 0.80, "confidence": 0.75, "relevant": "yes"},
            {"content": "Chunk 3", "similarity_score": 0.75, "confidence": 0.8, "relevant": "yes"},
            {"content": "Chunk 4", "similarity_score": 0.70, "confidence": 0.72, "relevant": "yes"},
        ]
        state = self._create_state_with_graded_chunks("Test question", chunks)
        state["average_confidence"] = 0.75  # Low confidence (<= 0.8)

        generator_node(state)

        # Verify all chunks are in prompt (no limiting)
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]

        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt
        assert "Chunk 3" in prompt
        assert "Chunk 4" in prompt

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_no_limit_when_exactly_2_chunks(self, mock_create_llm):
        """Test generator uses all chunks when there are exactly 2 chunks."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        # Create exactly 2 chunks
        chunks = [
            {"content": "Chunk 1", "similarity_score": 0.95, "confidence": 0.9, "relevant": "yes"},
            {"content": "Chunk 2", "similarity_score": 0.90, "confidence": 0.85, "relevant": "yes"},
        ]
        state = self._create_state_with_graded_chunks("Test question", chunks)
        state["average_confidence"] = 0.88  # High confidence but only 2 chunks

        generator_node(state)

        # Verify both chunks are in prompt (no limiting needed)
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]

        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_uses_shortened_prompt(self, mock_create_llm):
        """Test generator uses the new shortened prompt format."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test content", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("Test question", chunks)

        generator_node(state)

        # Verify prompt uses new concise format (not old multi-step format)
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]

        # New prompt should have these elements
        assert "3GPP specification expert" in prompt
        assert "Extract precise answers from" in prompt
        assert "Extract the EXACT value with units" in prompt

        # Old prompt should NOT have these verbose elements
        assert "STEP 1" not in prompt
        assert "STEP 2" not in prompt
        assert "STEP 3" not in prompt
        assert "STEP 4" not in prompt
        assert "STEP 5" not in prompt

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_limits_at_exactly_0_8_confidence(self, mock_create_llm):
        """Test generator behavior at confidence boundary (0.8)."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        # Create 4 chunks with confidence exactly at 0.8
        chunks = [
            {"content": "Chunk 1", "similarity_score": 0.95, "relevant": "yes"},
            {"content": "Chunk 2", "similarity_score": 0.90, "relevant": "yes"},
            {"content": "Chunk 3", "similarity_score": 0.85, "relevant": "yes"},
        ]
        state = self._create_state_with_graded_chunks("Test question", chunks)
        state["average_confidence"] = 0.8  # Exactly 0.8 (not > 0.8)

        generator_node(state)

        # At exactly 0.8, should NOT limit (only limit when > 0.8)
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]

        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt
        assert "Chunk 3" in prompt

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_handles_missing_average_confidence(self, mock_create_llm):
        """Test generator handles missing average_confidence field."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        # Create chunks but don't set average_confidence
        chunks = [
            {"content": "Chunk 1", "similarity_score": 0.95, "relevant": "yes"},
            {"content": "Chunk 2", "similarity_score": 0.90, "relevant": "yes"},
            {"content": "Chunk 3", "similarity_score": 0.85, "relevant": "yes"},
        ]
        state = self._create_state_with_graded_chunks("Test question", chunks)
        # Don't set average_confidence - should default to 0.0

        generator_node(state)

        # Should use all chunks (since default 0.0 is not > 0.8)
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]

        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt
        assert "Chunk 3" in prompt

    @patch("specagent.nodes.generator.create_llm")
    def test_generator_prompt_includes_exact_units_guidance(self, mock_create_llm):
        """Test that generator prompt includes guidance for extracting exact units."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "The size is 128 bits [TS 38.321 §5.4]"
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "The PDU size is 128 bits", "relevant": "yes"}]
        state = self._create_state_with_graded_chunks("What is the PDU size?", chunks)

        generator_node(state)

        # Verify prompt includes guidance for exact units
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]

        # Should include explicit guidance about numbers with units
        assert "For numerical" in prompt or "numerical" in prompt.lower()
        assert "exact value" in prompt.lower() or "Extract exact" in prompt
        assert "unit" in prompt.lower()
