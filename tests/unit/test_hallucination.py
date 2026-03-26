"""Unit tests for hallucination checker node."""

import re
from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import GradedChunk, GraphState, RetrievedChunk, create_initial_state
from specagent.nodes.hallucination import (
    HallucinationResult,
    _contains_numerical_or_tabular_content,
    hallucination_check_node,
)


@pytest.mark.unit
class TestHallucinationResult:
    """Tests for HallucinationResult Pydantic model."""

    def test_hallucination_result_grounded(self):
        """Test creating HallucinationResult for fully grounded answer."""
        result = HallucinationResult(grounded="yes", ungrounded_claims=[])

        assert result.grounded == "yes"
        assert result.ungrounded_claims == []

    def test_hallucination_result_not_grounded(self):
        """Test creating HallucinationResult for ungrounded answer."""
        result = HallucinationResult(
            grounded="no",
            ungrounded_claims=[
                "The UE supports 32 HARQ processes",
                "TDD requires special handling",
            ],
        )

        assert result.grounded == "no"
        assert len(result.ungrounded_claims) == 2
        assert "32 HARQ processes" in result.ungrounded_claims[0]

    def test_hallucination_result_partial(self):
        """Test creating HallucinationResult for partially grounded answer."""
        result = HallucinationResult(
            grounded="partial", ungrounded_claims=["Some minor detail not in sources"]
        )

        assert result.grounded == "partial"
        assert len(result.ungrounded_claims) == 1

    def test_hallucination_result_validation_grounded(self):
        """Test that invalid grounded values are rejected."""
        with pytest.raises(ValueError):
            HallucinationResult(grounded="maybe", ungrounded_claims=[])

    def test_hallucination_result_default_claims(self):
        """Test that ungrounded_claims defaults to empty list."""
        result = HallucinationResult(grounded="yes")
        assert result.ungrounded_claims == []


@pytest.mark.unit
class TestHallucinationCheckNode:
    """Tests for hallucination_check_node function."""

    def _create_state_with_generation(
        self, question: str, generation: str, chunks_data: list[dict]
    ) -> GraphState:
        """Helper to create state with generation and graded chunks."""
        state = create_initial_state(question)
        state["route_decision"] = "retrieve"

        # Create retrieved chunks
        retrieved_chunks = [
            RetrievedChunk(
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
            )
            for i, chunk in enumerate(chunks_data)
        ]
        state["retrieved_chunks"] = retrieved_chunks

        # Create graded chunks (mark all as relevant for simplicity)
        state["graded_chunks"] = [
            GradedChunk(chunk=chunk, relevant="yes", confidence=0.85) for chunk in retrieved_chunks
        ]

        state["generation"] = generation

        return state

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_fully_grounded(self, mock_create_llm):
        """Test hallucination check with fully grounded answer."""
        # Mock LLM to return grounded result
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        # Create state with generation that matches source
        chunks = [
            {"content": "The maximum number of HARQ processes for NR is 16 for both FDD and TDD."},
            {"content": "HARQ processes are used for retransmission handling in the MAC layer."},
        ]
        generation = "The maximum number of HARQ processes in NR is 16 for both FDD and TDD. [TS 38.321 §5.4]"

        state = self._create_state_with_generation(
            "What is the maximum number of HARQ processes in NR?", generation, chunks
        )

        # Call hallucination check node
        result = hallucination_check_node(state)

        # Verify fully grounded
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_not_grounded(self, mock_create_llm):
        """Test hallucination check with ungrounded answer."""
        # Mock LLM to return not grounded result
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "no", "ungrounded_claims": ["The UE supports up to 32 HARQ processes", "TDD mode requires special configuration"]}'
        mock_create_llm.return_value = mock_llm

        # Create state with generation that contradicts sources
        chunks = [
            {"content": "The maximum number of HARQ processes for NR is 16 for both FDD and TDD."},
        ]
        generation = (
            "The UE supports up to 32 HARQ processes. TDD mode requires special configuration."
        )

        state = self._create_state_with_generation(
            "What is the maximum number of HARQ processes in NR?", generation, chunks
        )
        # Set low confidence to trigger hallucination check (has numerical content)
        state["average_confidence"] = 0.60

        # Call hallucination check node
        result = hallucination_check_node(state)

        # Verify not grounded
        assert result["hallucination_check"] == "not_grounded"
        assert len(result["ungrounded_claims"]) == 2
        assert "32 HARQ processes" in result["ungrounded_claims"][0]

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_partial(self, mock_create_llm):
        """Test hallucination check with partially grounded answer."""
        # Mock LLM to return partial result
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "partial", "ungrounded_claims": ["Some implementation details may vary by vendor"]}'
        mock_create_llm.return_value = mock_llm

        # Create state with mostly correct generation but some unsupported claims
        chunks = [
            {"content": "The maximum number of HARQ processes for NR is 16 for both FDD and TDD."},
        ]
        generation = "The maximum number of HARQ processes in NR is 16. Some implementation details may vary by vendor."

        state = self._create_state_with_generation(
            "What is the maximum number of HARQ processes in NR?", generation, chunks
        )
        # Set low confidence to trigger hallucination check (has numerical content)
        state["average_confidence"] = 0.60

        # Call hallucination check node
        result = hallucination_check_node(state)

        # Verify partial grounding
        assert result["hallucination_check"] == "partial"
        assert len(result["ungrounded_claims"]) == 1
        assert "vendor" in result["ungrounded_claims"][0].lower()

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_no_generation(self, mock_create_llm):
        """Test hallucination check when generation is None."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        # Create state without generation
        state = create_initial_state("Test question")
        state["generation"] = None
        state["graded_chunks"] = []

        result = hallucination_check_node(state)

        # Should mark as grounded (nothing to check)
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_empty_generation(self, mock_create_llm):
        """Test hallucination check with empty string generation."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = ""
        state["graded_chunks"] = []

        result = hallucination_check_node(state)

        # Should mark as grounded (nothing to check)
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_no_chunks_grounded(self, mock_create_llm):
        """Test hallucination check with no chunks - grounded result."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Some generated answer without sources."
        state["graded_chunks"] = []

        result = hallucination_check_node(state)

        # Should mark as grounded
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_no_chunks_not_grounded(self, mock_create_llm):
        """Test hallucination check with no chunks - not grounded result."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "no", "ungrounded_claims": ["All claims are ungrounded - no sources provided"]}'
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Some generated answer without sources."
        state["graded_chunks"] = []
        state["average_confidence"] = 0.5  # Low confidence to trigger check

        result = hallucination_check_node(state)

        # Should mark as not grounded (no sources to verify against)
        assert result["hallucination_check"] == "not_grounded"
        assert len(result["ungrounded_claims"]) > 0

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_no_chunks_partial(self, mock_create_llm):
        """Test hallucination check with no chunks - partial result."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"grounded": "partial", "ungrounded_claims": ["Some claims are questionable"]}'
        )
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Some generated answer without sources."
        state["graded_chunks"] = []
        state["average_confidence"] = 0.5  # Low confidence to trigger check

        result = hallucination_check_node(state)

        # Should mark as partial
        assert result["hallucination_check"] == "partial"
        assert len(result["ungrounded_claims"]) > 0

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_no_chunks_llm_error(self, mock_create_llm):
        """Test hallucination check handles LLM errors when no chunks."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Some generated answer."
        state["graded_chunks"] = []
        state["average_confidence"] = 0.5  # Low confidence to trigger check

        result = hallucination_check_node(state)

        # Should populate error field and default to grounded
        assert "error" in result
        assert result["error"] is not None
        assert "error" in result["error"].lower()
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_llm_call_format(self, mock_create_llm):
        """Test that LLM is called with correct format."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "The maximum number of HARQ processes is 16."}]
        generation = "The answer is 16 HARQ processes."

        state = self._create_state_with_generation("Test?", generation, chunks)
        # Set low confidence to trigger hallucination check (has numerical content)
        state["average_confidence"] = 0.60

        hallucination_check_node(state)

        # Verify invoke was called with prompt containing sources and answer
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]
        assert "The maximum number of HARQ processes is 16" in prompt
        assert "The answer is 16 HARQ processes" in prompt

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_handles_llm_error(self, mock_create_llm):
        """Test hallucination check handles LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test content"}]
        state = self._create_state_with_generation("Test?", "Test answer is 5", chunks)
        state["average_confidence"] = 0.5  # Low confidence to trigger check

        result = hallucination_check_node(state)

        # Should populate error field
        assert "error" in result
        assert result["error"] is not None
        assert "error" in result["error"].lower()

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_preserves_other_state_fields(self, mock_create_llm):
        """Test that hallucination check only modifies hallucination fields."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test content"}]
        state = self._create_state_with_generation("Test?", "Test answer", chunks)
        state["route_reasoning"] = "Test routing"
        state["rewrite_count"] = 1
        state["average_confidence"] = 0.9

        result = hallucination_check_node(state)

        # Verify other fields are preserved
        assert result["question"] == "Test?"
        assert result["generation"] == "Test answer"
        assert result["route_reasoning"] == "Test routing"
        assert result["rewrite_count"] == 1
        assert result["average_confidence"] == 0.9

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_with_multiple_chunks(self, mock_create_llm):
        """Test hallucination check with multiple source chunks."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        chunks = [
            {"content": "HARQ processes: 16 maximum", "spec_id": "TS38.321", "section": "5.4"},
            {"content": "Both FDD and TDD supported", "spec_id": "TS38.321", "section": "5.4.1"},
            {"content": "MAC layer handles HARQ", "spec_id": "TS38.321", "section": "5.1"},
        ]
        generation = "NR supports 16 HARQ processes for both FDD and TDD, handled by MAC."

        state = self._create_state_with_generation("Test?", generation, chunks)
        # Set low confidence to trigger hallucination check (has numerical content)
        state["average_confidence"] = 0.60

        hallucination_check_node(state)

        # Verify all chunks were included in the prompt
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]
        assert "HARQ processes: 16 maximum" in prompt
        assert "Both FDD and TDD supported" in prompt
        assert "MAC layer handles HARQ" in prompt

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_grounded_to_state_mapping(self, mock_create_llm):
        """Test correct mapping from HallucinationResult.grounded to state values."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test content"}]

        # Test "yes" -> "grounded"
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        state = self._create_state_with_generation("Q?", "Answer is 5", chunks)
        state["average_confidence"] = 0.60  # Set low to trigger check
        result = hallucination_check_node(state)
        assert result["hallucination_check"] == "grounded"

        # Test "no" -> "not_grounded"
        mock_llm.invoke.return_value = '{"grounded": "no", "ungrounded_claims": ["claim"]}'
        state = self._create_state_with_generation("Q?", "Answer is 10", chunks)
        state["average_confidence"] = 0.60  # Set low to trigger check
        result = hallucination_check_node(state)
        assert result["hallucination_check"] == "not_grounded"

        # Test "partial" -> "partial"
        mock_llm.invoke.return_value = '{"grounded": "partial", "ungrounded_claims": ["claim"]}'
        state = self._create_state_with_generation("Q?", "Answer is 15", chunks)
        state["average_confidence"] = 0.60  # Set low to trigger check
        result = hallucination_check_node(state)
        assert result["hallucination_check"] == "partial"

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_insufficient_info_response(self, mock_create_llm):
        """Test hallucination check with 'I don't have enough information' response."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Some unrelated content"}]
        generation = "I don't have enough information in the available specifications to fully answer this question."

        state = self._create_state_with_generation("Complex question?", generation, chunks)

        result = hallucination_check_node(state)

        # Should be grounded (truthful acknowledgment of lack of info)
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_no_chunks_grounded_direct_json(self, mock_create_llm):
        """Test hallucination check with no chunks and direct JSON response (no regex match)."""
        mock_llm = MagicMock()
        # Return JSON without any surrounding text (tests the else branch on line 164)
        mock_llm.invoke.return_value = 'Some text before {"grounded": "yes", "ungrounded_claims": []} but regex should still match'
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Some generated answer."
        state["graded_chunks"] = []
        state["average_confidence"] = 0.5  # Low confidence to trigger check

        result = hallucination_check_node(state)

        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_hallucination_with_chunks_direct_json(self, mock_create_llm):
        """Test hallucination check with chunks and direct JSON response (no regex match)."""
        mock_llm = MagicMock()
        # Test the case where response needs the else branch (line 216)
        # This simulates when regex doesn't match but json.loads works directly
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test content"}]
        generation = "Answer is 42"  # Has number to trigger check
        state = self._create_state_with_generation("Test?", generation, chunks)

        result = hallucination_check_node(state)

        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []


@pytest.mark.unit
class TestContainsNumericalOrTabularContent:
    """Tests for _contains_numerical_or_tabular_content helper function."""

    def test_detects_standalone_numbers(self):
        """Test detection of standalone numbers."""
        assert _contains_numerical_or_tabular_content("The value is 16")
        assert _contains_numerical_or_tabular_content("Maximum is 100")
        assert _contains_numerical_or_tabular_content("There are 5 options")

    def test_detects_numbers_with_units(self):
        """Test detection of numbers with units."""
        assert _contains_numerical_or_tabular_content("Latency is 100ms")
        assert _contains_numerical_or_tabular_content("The frequency is 2.4GHz")
        assert _contains_numerical_or_tabular_content("Signal strength is 15dB")
        assert _contains_numerical_or_tabular_content("Distance is 5km")
        assert _contains_numerical_or_tabular_content("Bandwidth is 20MHz")

    def test_detects_percentages(self):
        """Test detection of percentages."""
        assert _contains_numerical_or_tabular_content("Success rate is 95%")
        assert _contains_numerical_or_tabular_content("Efficiency: 50%")

    def test_detects_ranges(self):
        """Test detection of ranges."""
        assert _contains_numerical_or_tabular_content("Range is 5-10")
        assert _contains_numerical_or_tabular_content("Values between 1..10")

    def test_detects_markdown_tables(self):
        """Test detection of markdown tables."""
        table_text = """
        | Parameter | Value |
        | --------- | ----- |
        | HARQ      | 16    |
        """
        assert _contains_numerical_or_tabular_content(table_text)

    def test_detects_inline_tables(self):
        """Test detection of inline table-like structures."""
        assert _contains_numerical_or_tabular_content("| Column1 | Column2 | Column3 |")

    def test_no_numerical_content_text_only(self):
        """Test that plain text without numbers returns False."""
        assert not _contains_numerical_or_tabular_content(
            "This is a purely descriptive answer with no numbers or tables."
        )
        assert not _contains_numerical_or_tabular_content(
            "The answer explains the concept without specific values."
        )

    def test_empty_string(self):
        """Test empty string returns False."""
        assert not _contains_numerical_or_tabular_content("")

    def test_detects_floats(self):
        """Test detection of floating point numbers."""
        assert _contains_numerical_or_tabular_content("The ratio is 3.14")
        assert _contains_numerical_or_tabular_content("Value: 0.5")

    def test_ignores_spec_citations(self):
        """Test that spec citations like [TS 38.321] are ignored."""
        assert not _contains_numerical_or_tabular_content(
            "According to [TS 38.321], the MAC layer handles HARQ."
        )
        assert not _contains_numerical_or_tabular_content("See [TS 23.501] for more details.")

    def test_ignores_spec_citations_with_section(self):
        """Test that spec citations with section numbers are ignored."""
        assert not _contains_numerical_or_tabular_content(
            "As specified in [TS 38.321 §5.4], the procedure is defined."
        )
        assert not _contains_numerical_or_tabular_content(
            "Reference [TS 38.321 §5.4.1] describes the behavior."
        )

    def test_ignores_spec_citations_case_insensitive(self):
        """Test that spec citations are ignored regardless of case."""
        assert not _contains_numerical_or_tabular_content(
            "According to [ts 38.321], this is specified."
        )
        assert not _contains_numerical_or_tabular_content("See [Ts 23.501] for details.")

    def test_detects_numbers_with_citations_present(self):
        """Test that actual numbers are still detected even with citations."""
        assert _contains_numerical_or_tabular_content(
            "According to [TS 38.321], the maximum is 16 HARQ processes."
        )
        assert _contains_numerical_or_tabular_content("The value is 100ms as per [TS 38.214].")

    def test_mixed_citations_and_text(self):
        """Test text with only citations and no other numbers."""
        assert not _contains_numerical_or_tabular_content(
            "See [TS 38.321] and [TS 38.214] for protocol details."
        )


@pytest.mark.unit
class TestHallucinationCheckConditional:
    """Tests for conditional hallucination check logic."""

    def _create_state_with_generation_and_confidence(
        self, generation: str, average_confidence: float, chunks_data: list[dict] | None = None
    ) -> GraphState:
        """Helper to create state with generation and confidence."""
        state = create_initial_state("Test question")
        state["generation"] = generation
        state["average_confidence"] = average_confidence

        if chunks_data:
            retrieved_chunks = [
                RetrievedChunk(
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
                )
                for i, chunk in enumerate(chunks_data)
            ]
            state["retrieved_chunks"] = retrieved_chunks
            state["graded_chunks"] = [
                GradedChunk(chunk=chunk, relevant="yes", confidence=0.85)
                for chunk in retrieved_chunks
            ]
        else:
            state["graded_chunks"] = []

        return state

    @patch("specagent.nodes.hallucination.create_llm")
    def test_skip_check_high_confidence_no_numbers(self, mock_create_llm):
        """Test that hallucination check is skipped when confidence is high and no numbers."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        generation = "This is a purely descriptive answer about concepts without specific values."
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.85
        )

        result = hallucination_check_node(state)

        # Should skip check - no LLM call
        mock_llm.invoke.assert_not_called()
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_run_check_low_confidence_no_numbers(self, mock_create_llm):
        """Test that hallucination check runs when confidence is low even without numbers."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        generation = "This is a descriptive answer."
        chunks = [{"content": "Some context"}]
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.55, chunks_data=chunks
        )

        result = hallucination_check_node(state)

        # Should run check - LLM was called
        mock_llm.invoke.assert_called_once()
        assert result["hallucination_check"] == "grounded"

    @patch("specagent.nodes.hallucination.create_llm")
    def test_run_check_high_confidence_with_numbers(self, mock_create_llm):
        """Test that hallucination check runs when generation has numbers with mid confidence."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        generation = "The maximum number of HARQ processes is 16."
        chunks = [{"content": "HARQ processes: 16"}]
        # Use 0.64 (< 0.65) to trigger check for numerical content
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.64, chunks_data=chunks
        )

        result = hallucination_check_node(state)

        # Should run check - LLM was called
        mock_llm.invoke.assert_called_once()
        assert result["hallucination_check"] == "grounded"

    @patch("specagent.nodes.hallucination.create_llm")
    def test_run_check_high_confidence_with_table(self, mock_create_llm):
        """Test that hallucination check runs when generation has table with mid confidence."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        generation = """Here's the information:
        | Parameter | Value |
        | --------- | ----- |
        | HARQ      | 16    |
        """
        chunks = [{"content": "HARQ processes: 16"}]
        # Use 0.64 (< 0.65) to trigger check for numerical/tabular content
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.64, chunks_data=chunks
        )

        result = hallucination_check_node(state)

        # Should run check - LLM was called
        mock_llm.invoke.assert_called_once()
        assert result["hallucination_check"] == "grounded"

    @patch("specagent.nodes.hallucination.create_llm")
    def test_run_check_boundary_confidence_no_numbers(self, mock_create_llm):
        """Test hallucination check runs at confidence boundary (0.7) without numbers."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        generation = "This is a descriptive answer without numbers."
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.7
        )

        result = hallucination_check_node(state)

        # At boundary with no numbers, should skip check
        mock_llm.invoke.assert_not_called()
        assert result["hallucination_check"] == "grounded"

    @patch("specagent.nodes.hallucination.create_llm")
    def test_run_check_below_boundary_confidence(self, mock_create_llm):
        """Test hallucination check runs just below confidence boundary."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        generation = "This is a descriptive answer."
        chunks = [{"content": "Some context"}]
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.69, chunks_data=chunks
        )

        result = hallucination_check_node(state)

        # Just below boundary, should run check
        mock_llm.invoke.assert_called_once()
        assert result["hallucination_check"] == "grounded"

    @patch("specagent.nodes.hallucination.create_llm")
    def test_skip_check_missing_confidence_defaults_high(self, mock_create_llm):
        """Test that missing confidence defaults to 1.0 and skips check if no numbers."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Descriptive answer without numbers."
        # Don't set average_confidence - should default to 1.0
        state["graded_chunks"] = []

        result = hallucination_check_node(state)

        # Should skip check with default high confidence
        mock_llm.invoke.assert_not_called()
        assert result["hallucination_check"] == "grounded"

    @patch("specagent.nodes.hallucination.create_llm")
    def test_run_check_with_units(self, mock_create_llm):
        """Test that numbers with units trigger hallucination check with mid confidence."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        generation = "The latency is 100ms and bandwidth is 20MHz."
        chunks = [{"content": "Latency: 100ms, Bandwidth: 20MHz"}]
        # Use 0.64 (< 0.65) to trigger check for numerical content
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.64, chunks_data=chunks
        )

        result = hallucination_check_node(state)

        # Should run check due to numerical content
        mock_llm.invoke.assert_called_once()
        assert result["hallucination_check"] == "grounded"

    @patch("specagent.nodes.hallucination.create_llm")
    @patch("specagent.nodes.hallucination.re.search")
    def test_no_chunks_json_parse_fallback(self, mock_re_search, mock_create_llm):
        """Test fallback to direct json.loads when regex doesn't match (no chunks)."""
        # Mock regex search to return None (no match)
        mock_re_search.return_value = None

        mock_llm = MagicMock()
        # Return valid JSON that somehow regex didn't match (defensive code path)
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Some answer with 42 numbers."
        state["graded_chunks"] = []
        state["average_confidence"] = 0.5

        result = hallucination_check_node(state)

        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    @patch("specagent.nodes.hallucination.re.search")
    def test_with_chunks_json_parse_fallback(self, mock_re_search, mock_create_llm):
        """Test fallback to direct json.loads when regex doesn't match (with chunks)."""

        # Mock regex search to return None (no match) for the second call
        # First call is for citation pattern in _contains_numerical_or_tabular_content
        # We need to handle multiple re.search calls
        def search_side_effect(pattern, text, *args):
            if pattern == r"\[TS\s+\d+\.\d+[^\]]*\]":
                # Citation pattern - return None (no citations)
                return None
            elif pattern == r"\{.*\}":
                # JSON pattern - return None to trigger fallback
                return None
            else:
                # Other patterns - use real search
                return re.search(pattern, text, *args) if args else re.search(pattern, text)

        mock_re_search.side_effect = search_side_effect

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test content"}]
        state = create_initial_state("Test?")
        state["generation"] = "Answer with 5 numbers"
        state["average_confidence"] = 0.9

        retrieved_chunks = [
            RetrievedChunk(
                content=chunk["content"],
                chunk_id=f"chunk_{i}",
                doc_id="doc-uuid",
                source="TS38.321.docx",
                title="TS 38.321 MAC",
                chunk_index=0,
                file_type="docx",
                spec_id="TS38.321",
                section="5.4",
                similarity_score=0.8,
            )
            for i, chunk in enumerate(chunks)
        ]
        state["retrieved_chunks"] = retrieved_chunks
        state["graded_chunks"] = [
            GradedChunk(chunk=chunk, relevant="yes", confidence=0.85) for chunk in retrieved_chunks
        ]

        result = hallucination_check_node(state)

        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_skip_check_numerical_content_confidence_0_65(self, mock_create_llm):
        """Test that hallucination check is skipped for numerical content with confidence >= 0.65."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        generation = "The maximum is 16 HARQ processes."
        chunks = [{"content": "HARQ processes: 16"}]
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.65, chunks_data=chunks
        )

        result = hallucination_check_node(state)

        # Should skip check - no LLM call (confidence >= 0.65 with numerical content)
        mock_llm.invoke.assert_not_called()
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_run_check_numerical_content_confidence_below_0_65(self, mock_create_llm):
        """Test that hallucination check runs for numerical content with confidence < 0.65."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        generation = "The maximum is 16 HARQ processes."
        chunks = [{"content": "HARQ processes: 16"}]
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.64, chunks_data=chunks
        )

        result = hallucination_check_node(state)

        # Should run check - LLM was called (confidence < 0.65 with numerical content)
        mock_llm.invoke.assert_called_once()
        assert result["hallucination_check"] == "grounded"

    @patch("specagent.nodes.hallucination.create_llm")
    def test_skip_check_non_numerical_confidence_0_70(self, mock_create_llm):
        """Test that hallucination check is skipped for non-numerical content with confidence >= 0.70."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        generation = "This describes the protocol behavior without numbers."
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.70
        )

        result = hallucination_check_node(state)

        # Should skip check - no LLM call (confidence >= 0.70 without numerical content)
        mock_llm.invoke.assert_not_called()
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.create_llm")
    def test_run_check_non_numerical_confidence_below_0_70(self, mock_create_llm):
        """Test that hallucination check runs for non-numerical content with confidence < 0.70."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grounded": "yes", "ungrounded_claims": []}'
        mock_create_llm.return_value = mock_llm

        generation = "This describes the protocol behavior."
        chunks = [{"content": "Protocol behavior description"}]
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.69, chunks_data=chunks
        )

        result = hallucination_check_node(state)

        # Should run check - LLM was called (confidence < 0.70 without numerical content)
        mock_llm.invoke.assert_called_once()
        assert result["hallucination_check"] == "grounded"

    @patch("specagent.nodes.hallucination.create_llm")
    def test_skip_check_numerical_confidence_between_0_65_and_0_70(self, mock_create_llm):
        """Test skip for numerical content with confidence in range [0.65, 0.70)."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        generation = "The value is 100ms and bandwidth is 20MHz."
        chunks = [{"content": "Latency: 100ms, Bandwidth: 20MHz"}]
        state = self._create_state_with_generation_and_confidence(
            generation=generation, average_confidence=0.67, chunks_data=chunks
        )

        result = hallucination_check_node(state)

        # Should skip check (0.67 >= 0.65 with numerical content)
        mock_llm.invoke.assert_not_called()
        assert result["hallucination_check"] == "grounded"


@pytest.mark.unit
def test_hallucination_no_json_braces_falls_to_json_loads():
    """Cover line 225: when re.search finds no {..}, falls back to json.loads."""
    mock_llm = MagicMock()
    # Response has no '{...}' pattern — re.search returns None → line 225 executes
    mock_llm.invoke.return_value = "grounded: yes no braces"

    # graded_chunk must have .relevant == "yes" and .chunk with required attrs
    inner_chunk = MagicMock()
    inner_chunk.spec_id = "TS38.321"
    inner_chunk.section = "5.4"
    inner_chunk.content = "HARQ processes."

    graded = MagicMock()
    graded.relevant = "yes"
    graded.chunk = inner_chunk

    state = {
        "question": "How many HARQ processes?",
        "generation": "16 HARQ processes",
        "graded_chunks": [graded],
        "hallucination_check": None,
        # Low confidence so the skip-threshold check doesn't short-circuit
        "average_confidence": 0.0,
    }

    with patch("specagent.nodes.hallucination.create_llm", return_value=mock_llm):
        result = hallucination_check_node(state)

    # json.loads("grounded: yes no braces") raises JSONDecodeError
    # → caught by except at line 237 → error set + fallback to "grounded"
    assert result.get("hallucination_check") == "grounded"
