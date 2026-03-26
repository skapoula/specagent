"""Unit tests for grader node."""

from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import GraphState, RetrievedChunk, create_initial_state
from specagent.nodes.grader import GradeResult, grader_node


@pytest.mark.unit
class TestGradeResult:
    """Tests for GradeResult Pydantic model."""

    def test_grade_result_relevant(self):
        """Test creating GradeResult for relevant chunk."""
        result = GradeResult(relevant="yes", confidence=0.85)

        assert result.relevant == "yes"
        assert result.confidence == 0.85

    def test_grade_result_not_relevant(self):
        """Test creating GradeResult for irrelevant chunk."""
        result = GradeResult(relevant="no", confidence=0.95)

        assert result.relevant == "no"
        assert result.confidence == 0.95

    def test_grade_result_validation_relevant(self):
        """Test that invalid relevant values are rejected."""
        with pytest.raises(ValueError):
            GradeResult(relevant="maybe", confidence=0.5)

    def test_grade_result_validation_confidence_low(self):
        """Test that confidence below 0 is rejected."""
        with pytest.raises(ValueError):
            GradeResult(relevant="yes", confidence=-0.1)

    def test_grade_result_validation_confidence_high(self):
        """Test that confidence above 1 is rejected."""
        with pytest.raises(ValueError):
            GradeResult(relevant="yes", confidence=1.5)

    def test_grade_result_confidence_boundaries(self):
        """Test that confidence at boundaries 0 and 1 is valid."""
        result_min = GradeResult(relevant="yes", confidence=0.0)
        result_max = GradeResult(relevant="no", confidence=1.0)

        assert result_min.confidence == 0.0
        assert result_max.confidence == 1.0


@pytest.mark.unit
class TestGraderNode:
    """Tests for grader_node function."""

    def _create_state_with_chunks(self, question: str, chunks_data: list[dict]) -> GraphState:
        """Helper to create state with retrieved chunks."""
        state = create_initial_state(question)
        state["route_decision"] = "retrieve"
        state["retrieved_chunks"] = [
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
        return state

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_all_relevant(self, mock_create_llm):
        """Test grader node with all relevant chunks."""
        # Mock LLM to return batch response with all relevant
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.9}, {"relevant": "yes", "confidence": 0.85}, {"relevant": "yes", "confidence": 0.8}]}'
        mock_create_llm.return_value = mock_llm

        # Create state with 3 chunks (mid-range similarity to trigger LLM)
        chunks = [
            {
                "content": "The maximum number of HARQ processes for NR is 16.",
                "similarity_score": 0.7,
            },
            {
                "content": "HARQ processes are used for retransmission handling.",
                "similarity_score": 0.65,
            },
            {"content": "Both FDD and TDD support 16 HARQ processes.", "similarity_score": 0.75},
        ]
        state = self._create_state_with_chunks(
            "What is the maximum number of HARQ processes in NR?", chunks
        )

        # Call grader node
        result = grader_node(state)

        # Verify all chunks are graded
        assert len(result["graded_chunks"]) == 3
        assert all(gc.relevant == "yes" for gc in result["graded_chunks"])

        # Verify average confidence
        expected_avg = (0.9 + 0.85 + 0.8) / 3
        assert result["average_confidence"] == pytest.approx(expected_avg)

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_all_irrelevant(self, mock_create_llm):
        """Test grader node with all irrelevant chunks."""
        # Mock LLM to return batch response with all irrelevant
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "no", "confidence": 0.95}, {"relevant": "no", "confidence": 0.9}, {"relevant": "no", "confidence": 0.85}]}'
        mock_create_llm.return_value = mock_llm

        # Create state with irrelevant chunks (mid-range similarity to trigger LLM)
        chunks = [
            {"content": "The PDCCH carries downlink control information.", "similarity_score": 0.6},
            {"content": "Component carriers can be aggregated.", "similarity_score": 0.55},
            {"content": "The F1 interface connects gNB-DU and gNB-CU.", "similarity_score": 0.65},
        ]
        state = self._create_state_with_chunks(
            "What is the maximum number of HARQ processes in NR?", chunks
        )

        # Call grader node
        result = grader_node(state)

        # Verify all chunks are graded as irrelevant
        assert len(result["graded_chunks"]) == 3
        assert all(gc.relevant == "no" for gc in result["graded_chunks"])

        # Verify average confidence
        expected_avg = (0.95 + 0.9 + 0.85) / 3
        assert result["average_confidence"] == pytest.approx(expected_avg)

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_mixed_relevance(self, mock_create_llm):
        """Test grader node with mixed relevant and irrelevant chunks."""
        # Mock LLM to return batch response with mixed results
        # Note: Only top-3 chunks are graded
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.9}, {"relevant": "no", "confidence": 0.8}, {"relevant": "yes", "confidence": 0.85}]}'
        mock_create_llm.return_value = mock_llm

        # Create state with mixed chunks (mid-range similarity to trigger LLM)
        chunks = [
            {
                "content": "The maximum number of HARQ processes for NR is 16.",
                "similarity_score": 0.7,
            },
            {
                "content": "The PDCCH carries downlink control information.",
                "similarity_score": 0.65,
            },
            {"content": "HARQ processes handle retransmissions in NR.", "similarity_score": 0.75},
            {"content": "The F1 interface is used in split architecture.", "similarity_score": 0.6},
        ]
        state = self._create_state_with_chunks(
            "What is the maximum number of HARQ processes in NR?", chunks
        )

        # Call grader node
        result = grader_node(state)

        # Verify mixed results (only top-3 graded)
        assert len(result["graded_chunks"]) == 3
        assert result["graded_chunks"][0].relevant == "yes"
        assert result["graded_chunks"][1].relevant == "no"
        assert result["graded_chunks"][2].relevant == "yes"

        # Verify average confidence
        expected_avg = (0.9 + 0.8 + 0.85) / 3
        assert result["average_confidence"] == pytest.approx(expected_avg)

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_preserves_chunk_data(self, mock_create_llm):
        """Test that grader preserves original chunk data in GradedChunk."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.85}]}'
        mock_create_llm.return_value = mock_llm

        chunks = [
            {
                "content": "Test content",
                "spec_id": "TS38.321",
                "section": "5.4.1",
                "similarity_score": 0.7,  # Mid-range to trigger LLM
                "chunk_id": "test_chunk_123",
                "source": "TS38.321.docx",
            }
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # Verify original chunk data is preserved
        graded_chunk = result["graded_chunks"][0]
        assert graded_chunk.chunk.content == "Test content"
        assert graded_chunk.chunk.spec_id == "TS38.321"
        assert graded_chunk.chunk.section == "5.4.1"
        assert graded_chunk.chunk.similarity_score == 0.7
        assert graded_chunk.chunk.chunk_id == "test_chunk_123"
        assert graded_chunk.chunk.source == "TS38.321.docx"

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_single_chunk(self, mock_create_llm):
        """Test grader with single chunk."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.88}]}'
        mock_create_llm.return_value = mock_llm

        chunks = [
            {"content": "Single test chunk", "similarity_score": 0.7}
        ]  # Mid-range to trigger LLM
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        assert len(result["graded_chunks"]) == 1
        assert result["graded_chunks"][0].relevant == "yes"
        assert result["average_confidence"] == 0.88

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_empty_chunks(self, mock_create_llm):
        """Test grader with no retrieved chunks."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        state = create_initial_state("Test question")
        state["retrieved_chunks"] = []

        result = grader_node(state)

        # Should return empty graded_chunks and 0 average confidence
        assert result["graded_chunks"] == []
        assert result["average_confidence"] == 0.0
        # LLM should not be called for empty chunks
        mock_llm.invoke.assert_not_called()

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_llm_call_format(self, mock_create_llm):
        """Test that LLM is called with correct format."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.8}]}'
        mock_create_llm.return_value = mock_llm

        chunks = [
            {"content": "Test chunk content", "similarity_score": 0.7}
        ]  # Mid-range to trigger LLM
        state = self._create_state_with_chunks("What is HARQ?", chunks)

        grader_node(state)

        # Verify invoke was called with prompt containing question and chunk
        mock_llm.invoke.assert_called_once()
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]
        assert "What is HARQ?" in prompt
        assert "Test chunk content" in prompt

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_handles_llm_error(self, mock_create_llm):
        """Test grader handles LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test chunk", "similarity_score": 0.7}]  # Mid-range to trigger LLM
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # Should populate error field
        assert "error" in result
        assert result["error"] is not None
        assert "error" in result["error"].lower()

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_preserves_other_state_fields(self, mock_create_llm):
        """Test that grader only modifies grading fields."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.8}]}'
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test chunk", "similarity_score": 0.7}]  # Mid-range to trigger LLM
        state = self._create_state_with_chunks("Test question", chunks)
        state["route_reasoning"] = "Test routing"
        state["rewrite_count"] = 1

        result = grader_node(state)

        # Verify other fields are preserved
        assert result["question"] == "Test question"
        assert result["route_reasoning"] == "Test routing"
        assert result["rewrite_count"] == 1

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_multiple_chunks_different_specs(self, mock_create_llm):
        """Test grader with chunks from different specifications."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.9}, {"relevant": "yes", "confidence": 0.85}, {"relevant": "no", "confidence": 0.7}]}'
        mock_create_llm.return_value = mock_llm

        chunks = [
            {
                "content": "HARQ info",
                "spec_id": "TS38.321",
                "section": "5.4",
                "similarity_score": 0.7,
            },
            {
                "content": "More HARQ",
                "spec_id": "TS38.331",
                "section": "7.2",
                "similarity_score": 0.65,
            },
            {
                "content": "Unrelated",
                "spec_id": "TS38.401",
                "section": "6.1",
                "similarity_score": 0.6,
            },
        ]
        state = self._create_state_with_chunks("What is HARQ?", chunks)

        result = grader_node(state)

        # Verify chunks from different specs are graded correctly
        assert len(result["graded_chunks"]) == 3
        assert result["graded_chunks"][0].chunk.spec_id == "TS38.321"
        assert result["graded_chunks"][1].chunk.spec_id == "TS38.331"
        assert result["graded_chunks"][2].chunk.spec_id == "TS38.401"

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_average_confidence_calculation(self, mock_create_llm):
        """Test that average confidence is calculated correctly."""
        mock_llm = MagicMock()
        # Only top-3 chunks are graded
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.6}, {"relevant": "yes", "confidence": 0.7}, {"relevant": "yes", "confidence": 0.8}]}'
        mock_create_llm.return_value = mock_llm

        # Create 5 chunks but only top-3 will be graded (mid-range similarity to trigger LLM)
        chunks = [{"content": f"Chunk {i}", "similarity_score": 0.55 + i * 0.05} for i in range(5)]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # Only top-3 are graded
        assert len(result["graded_chunks"]) == 3
        # Verify average is correct: (0.6 + 0.7 + 0.8) / 3 = 0.7
        expected_avg = (0.6 + 0.7 + 0.8) / 3
        assert result["average_confidence"] == pytest.approx(expected_avg)

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_auto_grade_high_similarity(self, mock_create_llm):
        """Test auto-grading for chunks with similarity > 0.82."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        # Create chunks with high similarity scores
        chunks = [
            {"content": "High similarity chunk 1", "similarity_score": 0.90},
            {"content": "High similarity chunk 2", "similarity_score": 0.87},
            {"content": "High similarity chunk 3", "similarity_score": 0.95},
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # LLM should NOT be called since all chunks are auto-graded
        mock_llm.invoke.assert_not_called()

        # All chunks should be graded as relevant
        assert len(result["graded_chunks"]) == 3
        assert all(gc.relevant == "yes" for gc in result["graded_chunks"])

        # Verify confidence scores (should equal similarity_score)
        assert result["graded_chunks"][0].confidence == pytest.approx(0.90)
        assert result["graded_chunks"][1].confidence == pytest.approx(0.87)
        assert result["graded_chunks"][2].confidence == pytest.approx(0.95)

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_auto_grade_low_similarity(self, mock_create_llm):
        """Test auto-grading for chunks with similarity < 0.55."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        # Create chunks with low similarity scores
        chunks = [
            {"content": "Low similarity chunk 1", "similarity_score": 0.3},
            {"content": "Low similarity chunk 2", "similarity_score": 0.45},
            {"content": "Low similarity chunk 3", "similarity_score": 0.2},
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # LLM should NOT be called since all chunks are auto-graded
        mock_llm.invoke.assert_not_called()

        # All chunks should be graded as not relevant
        assert len(result["graded_chunks"]) == 3
        assert all(gc.relevant == "no" for gc in result["graded_chunks"])

        # Verify confidence scores (should be 1 - similarity_score)
        assert result["graded_chunks"][0].confidence == pytest.approx(0.7)  # 1 - 0.3
        assert result["graded_chunks"][1].confidence == pytest.approx(0.55)  # 1 - 0.45
        assert result["graded_chunks"][2].confidence == pytest.approx(0.8)  # 1 - 0.2

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_mixed_auto_and_llm_grading(self, mock_create_llm):
        """Test mixed auto-grading and LLM grading for mid-range similarity."""

        mock_llm = MagicMock()
        # Only the mid-range chunk should be graded by LLM
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.75}]}'
        mock_create_llm.return_value = mock_llm

        # Create chunks with mixed similarity scores
        chunks = [
            {"content": "High similarity", "similarity_score": 0.90},  # auto: yes
            {"content": "Mid similarity", "similarity_score": 0.65},  # LLM
            {"content": "Low similarity", "similarity_score": 0.4},  # auto: no
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # LLM should be called once for the mid-range chunk
        mock_llm.invoke.assert_called_once()

        # Verify all chunks are graded
        assert len(result["graded_chunks"]) == 3

        # First chunk: auto-graded as yes (high similarity > 0.82)
        assert result["graded_chunks"][0].relevant == "yes"
        assert result["graded_chunks"][0].confidence == pytest.approx(0.90)

        # Second chunk: LLM-graded (mid similarity)
        assert result["graded_chunks"][1].relevant == "yes"
        assert result["graded_chunks"][1].confidence == 0.75

        # Third chunk: auto-graded as no (low similarity < 0.55)
        assert result["graded_chunks"][2].relevant == "no"
        assert result["graded_chunks"][2].confidence == pytest.approx(0.6)  # 1 - 0.4

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_mid_range_similarity_uses_llm(self, mock_create_llm):
        """Test that mid-range similarity (0.55-0.82) always uses LLM."""

        mock_llm = MagicMock()
        # All chunks in mid-range should be graded by LLM
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.8}, {"relevant": "no", "confidence": 0.7}, {"relevant": "yes", "confidence": 0.75}]}'
        mock_create_llm.return_value = mock_llm

        # Create chunks with mid-range similarity scores (0.55-0.82)
        chunks = [
            {"content": "Mid similarity 1", "similarity_score": 0.70},
            {"content": "Mid similarity 2", "similarity_score": 0.55},
            {"content": "Mid similarity 3", "similarity_score": 0.82},
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # LLM should be called for all mid-range chunks
        mock_llm.invoke.assert_called_once()

        # Verify all chunks are graded by LLM
        assert len(result["graded_chunks"]) == 3
        assert result["graded_chunks"][0].relevant == "yes"
        assert result["graded_chunks"][0].confidence == 0.8
        assert result["graded_chunks"][1].relevant == "no"
        assert result["graded_chunks"][1].confidence == 0.7
        assert result["graded_chunks"][2].relevant == "yes"
        assert result["graded_chunks"][2].confidence == 0.75

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_boundary_similarity_scores(self, mock_create_llm):
        """Test grading behavior at similarity score boundaries (0.55, 0.82)."""

        mock_llm = MagicMock()
        # Chunks at 0.55 and 0.82 should use LLM (3 grades for 3 chunks)
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.6}, {"relevant": "yes", "confidence": 0.8}, {"relevant": "yes", "confidence": 0.7}]}'
        mock_create_llm.return_value = mock_llm

        # Create chunks at boundary values
        chunks = [
            {"content": "At lower boundary", "similarity_score": 0.55},  # LLM (not < 0.55)
            {"content": "At upper boundary", "similarity_score": 0.82},  # LLM (not > 0.82)
            {"content": "Just above lower", "similarity_score": 0.56},  # LLM
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # All three should use LLM (boundaries are inclusive for mid-range)
        mock_llm.invoke.assert_called_once()
        assert len(result["graded_chunks"]) == 3

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_only_top_3_chunks_graded(self, mock_create_llm):
        """Test that only top-3 chunks are graded even with more retrieved."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        # Create 5 chunks, all with high similarity
        chunks = [{"content": f"Chunk {i}", "similarity_score": 0.95} for i in range(5)]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # Only top 3 should be graded
        assert len(result["graded_chunks"]) == 3
        mock_llm.invoke.assert_not_called()  # All auto-graded (high similarity)

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_json_response_without_regex_match(self, mock_create_llm):
        """Test parsing JSON response without extra text (direct JSON)."""
        mock_llm = MagicMock()
        # Return pure JSON without any wrapper text (no regex match needed)
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.85}]}'
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test chunk", "similarity_score": 0.7}]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        assert len(result["graded_chunks"]) == 1
        assert result["graded_chunks"][0].relevant == "yes"
        assert result["graded_chunks"][0].confidence == 0.85

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_handles_grade_count_mismatch(self, mock_create_llm):
        """Test error handling when LLM returns wrong number of grades."""
        mock_llm = MagicMock()
        # Return only 1 grade for 2 chunks (mismatch)
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.85}]}'
        mock_create_llm.return_value = mock_llm

        chunks = [
            {"content": "Chunk 1", "similarity_score": 0.7},
            {"content": "Chunk 2", "similarity_score": 0.65},
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # Should handle error gracefully
        assert "error" in result
        assert result["error"] is not None
        assert "Expected 2 grades but got 1" in result["error"]
        assert result["graded_chunks"] == []
        assert result["average_confidence"] == 0.0

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_handles_invalid_json_response(self, mock_create_llm):
        """Test error handling when LLM returns invalid JSON."""
        mock_llm = MagicMock()
        # Return invalid JSON that will fail parsing
        mock_llm.invoke.return_value = "This is not JSON at all"
        mock_create_llm.return_value = mock_llm

        chunks = [{"content": "Test chunk", "similarity_score": 0.7}]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # Should handle error gracefully
        assert "error" in result
        assert result["error"] is not None
        assert result["graded_chunks"] == []
        assert result["average_confidence"] == 0.0

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_auto_grade_threshold_0_82(self, mock_create_llm):
        """Test auto-grading with new threshold 0.82 instead of 0.85."""
        mock_llm = MagicMock()
        # Mock LLM for the boundary chunk at 0.82
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.80}]}'
        mock_create_llm.return_value = mock_llm

        # Create chunks at and above the new 0.82 threshold
        chunks = [
            {"content": "Chunk at 0.83", "similarity_score": 0.83},  # Should auto-grade
            {"content": "Chunk at 0.82", "similarity_score": 0.82},  # Boundary - LLM
            {"content": "Chunk at 0.84", "similarity_score": 0.84},  # Should auto-grade
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # LLM should be called once for the boundary chunk (0.82)
        mock_llm.invoke.assert_called_once()

        # All chunks should be graded
        assert len(result["graded_chunks"]) == 3

        # First and third chunks auto-graded with confidence = similarity
        assert result["graded_chunks"][0].relevant == "yes"
        assert result["graded_chunks"][0].confidence == pytest.approx(0.83)

        # Middle chunk graded by LLM
        assert result["graded_chunks"][1].relevant == "yes"
        assert result["graded_chunks"][1].confidence == pytest.approx(0.80)

        # Third chunk auto-graded
        assert result["graded_chunks"][2].relevant == "yes"
        assert result["graded_chunks"][2].confidence == pytest.approx(0.84)

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_low_similarity_penalty_threshold_0_55(self, mock_create_llm):
        """Test auto-grading with low-similarity penalty at 0.55."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        # Create chunks below the new 0.55 threshold
        chunks = [
            {"content": "Chunk at 0.54", "similarity_score": 0.54},  # Should auto-grade as no
            {"content": "Chunk at 0.50", "similarity_score": 0.50},  # Should auto-grade as no
            {"content": "Chunk at 0.40", "similarity_score": 0.40},  # Should auto-grade as no
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # LLM should NOT be called since chunks < 0.55 are auto-graded
        mock_llm.invoke.assert_not_called()

        # All chunks should be auto-graded as not relevant
        assert len(result["graded_chunks"]) == 3
        assert all(gc.relevant == "no" for gc in result["graded_chunks"])

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_mid_range_0_55_to_0_82_uses_llm(self, mock_create_llm):
        """Test that mid-range similarity (0.55-0.82) uses LLM."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.75}, {"relevant": "yes", "confidence": 0.70}]}'
        mock_create_llm.return_value = mock_llm

        # Create chunks in the new mid-range
        chunks = [
            {"content": "Chunk at 0.60", "similarity_score": 0.60},  # Mid-range - LLM
            {"content": "Chunk at 0.75", "similarity_score": 0.75},  # Mid-range - LLM
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # LLM should be called for mid-range chunks
        mock_llm.invoke.assert_called_once()

        # Verify all chunks are graded by LLM
        assert len(result["graded_chunks"]) == 2
        assert result["graded_chunks"][0].confidence == 0.75
        assert result["graded_chunks"][1].confidence == 0.70

    @patch("specagent.nodes.grader.create_llm")
    def test_grader_boundary_at_0_55(self, mock_create_llm):
        """Test grading behavior at similarity boundary (0.55)."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"grades": [{"relevant": "yes", "confidence": 0.65}]}'
        mock_create_llm.return_value = mock_llm

        # Create chunk at boundary
        chunks = [
            {"content": "Chunk at 0.55", "similarity_score": 0.55},  # Boundary - should use LLM
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # At boundary 0.55, should use LLM (not auto-grade)
        mock_llm.invoke.assert_called_once()
        assert len(result["graded_chunks"]) == 1
