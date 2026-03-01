"""
Unit tests for graph state module.
"""

import pytest

from specagent.graph.state import (
    Citation,
    GradedChunk,
    GraphState,
    RetrievedChunk,
    create_initial_state,
)


def _make_chunk(**kwargs) -> RetrievedChunk:
    """Build a RetrievedChunk with sensible defaults for tests."""
    defaults = dict(
        content="Test content",
        chunk_id="TS38.321.docx:0",
        doc_id="doc-uuid",
        source="/path/TS38.321.docx",
        title="TS 38.321 MAC",
        chunk_index=0,
        file_type="docx",
        spec_id="TS38.321",
        section="5.4",
        similarity_score=0.9,
    )
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


class TestRetrievedChunk:
    """Tests for RetrievedChunk dataclass."""

    def test_retrieved_chunk_creation(self):
        """RetrievedChunk should store all required fields."""
        chunk = _make_chunk(similarity_score=0.95, chunk_id="TS38.321.docx:0")

        assert chunk.content == "Test content"
        assert chunk.spec_id == "TS38.321"
        assert chunk.section == "5.4"
        assert chunk.similarity_score == 0.95
        assert chunk.chunk_id == "TS38.321.docx:0"

    def test_retrieved_chunk_has_source_not_source_file(self):
        """RetrievedChunk uses 'source' (LanceDB field name), not 'source_file'."""
        chunk = _make_chunk(source="/path/TS38.321.docx")
        assert chunk.source == "/path/TS38.321.docx"
        assert not hasattr(chunk, "source_file")

    def test_retrieved_chunk_has_doc_id_title_chunk_index(self):
        """RetrievedChunk includes LanceDB fields: doc_id, title, chunk_index, file_type."""
        chunk = _make_chunk(
            doc_id="doc-xyz",
            title="My Document",
            chunk_index=3,
            file_type="docx",
            spec_id="TS38.101",
            section="5.5A",
            similarity_score=0.75,
        )
        assert chunk.doc_id == "doc-xyz"
        assert chunk.title == "My Document"
        assert chunk.chunk_index == 3
        assert chunk.file_type == "docx"


class TestGradedChunk:
    """Tests for GradedChunk dataclass."""

    def test_graded_chunk_creation(self):
        """GradedChunk should wrap RetrievedChunk with grade info."""
        retrieved = _make_chunk(similarity_score=0.9)

        graded = GradedChunk(
            chunk=retrieved,
            relevant="yes",
            confidence=0.85,
        )

        assert graded.chunk.content == "Test content"
        assert graded.relevant == "yes"
        assert graded.confidence == 0.85


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self):
        """Citation should store spec reference info."""
        citation = Citation(
            spec_id="TS38.321",
            section="5.4.1",
            raw_citation="[TS 38.321 §5.4.1]",
            chunk_preview="The UE shall support a maximum of 16 HARQ processes...",
        )

        assert citation.spec_id == "TS38.321"
        assert citation.section == "5.4.1"
        assert citation.raw_citation == "[TS 38.321 §5.4.1]"


class TestGraphState:
    """Tests for GraphState TypedDict."""

    def test_create_initial_state(self, sample_question):
        """create_initial_state should set defaults correctly."""
        state = create_initial_state(sample_question)

        assert state["question"] == sample_question
        assert state["rewritten_question"] is None
        assert state["retrieved_chunks"] == []
        assert state["graded_chunks"] == []
        assert state["citations"] == []
        assert state["rewrite_count"] == 0
        assert state["generation"] is None
        assert state["error"] is None

    def test_graph_state_is_mutable(self, sample_question):
        """GraphState should allow modification of fields."""
        state = create_initial_state(sample_question)

        state["route_decision"] = "retrieve"
        state["rewrite_count"] = 1
        state["generation"] = "Test answer"

        assert state["route_decision"] == "retrieve"
        assert state["rewrite_count"] == 1
        assert state["generation"] == "Test answer"

    def test_graph_state_allows_partial(self):
        """GraphState should allow partial initialization (total=False)."""
        state: GraphState = {
            "question": "Test question",
        }

        assert state["question"] == "Test question"
        assert "generation" not in state
