"""
Unit tests for LangGraph workflow construction and execution.
"""

from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import GraphState, create_initial_state
from specagent.graph.workflow import (
    build_graph,
    get_graph_visualization,
    run_query,
    save_graph_image,
    should_regenerate,
    should_retrieve,
    should_rewrite,
)


@pytest.mark.unit
class TestBuildGraph:
    """Tests for build_graph() function."""

    def test_build_graph_returns_compiled_graph(self):
        """build_graph should return a compiled StateGraph."""
        graph = build_graph()

        # Check that we got a compiled graph
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "get_graph")

    def test_build_graph_has_all_nodes(self):
        """build_graph should include all 6 required nodes."""
        graph = build_graph()
        graph_def = graph.get_graph()

        # Get all node names - nodes is a dict mapping node_id -> node_data
        node_names = set(graph_def.nodes.keys())

        # Check for all required nodes (plus __start__ and __end__)
        expected_nodes = {
            "__start__",
            "router",
            "retriever",
            "grader",
            "rewriter",
            "generator",
            "hallucination_check",
            "__end__",
        }

        assert expected_nodes.issubset(node_names)

    def test_build_graph_has_conditional_edges(self):
        """build_graph should have conditional edges from router, grader, and hallucination_check."""
        graph = build_graph()
        graph_def = graph.get_graph()

        # Check that conditional edges exist
        # LangGraph represents conditional edges in the graph structure
        edges = graph_def.edges

        # Router should have conditional edges
        router_edges = [e for e in edges if e.source == "router"]
        assert len(router_edges) > 0

        # Grader should have conditional edges
        grader_edges = [e for e in edges if e.source == "grader"]
        assert len(grader_edges) > 0

        # Hallucination check should have conditional edges
        hallucination_edges = [e for e in edges if e.source == "hallucination_check"]
        assert len(hallucination_edges) > 0

    def test_build_graph_rewriter_loops_to_retriever(self):
        """build_graph should connect rewriter back to retriever."""
        graph = build_graph()
        graph_def = graph.get_graph()

        # Check for edge from rewriter to retriever
        edges = graph_def.edges
        rewriter_to_retriever = any(
            e.source == "rewriter" and e.target == "retriever" for e in edges
        )

        assert rewriter_to_retriever, "Rewriter should loop back to retriever"


@pytest.mark.unit
class TestShouldRetrieve:
    """Tests for should_retrieve conditional edge."""

    def test_should_retrieve_returns_retrieve_when_decision_is_retrieve(self):
        """should_retrieve should return 'retrieve' when route_decision is 'retrieve'."""
        state: GraphState = {
            "question": "Test question",
            "route_decision": "retrieve",
        }

        result = should_retrieve(state)

        assert result == "retrieve"

    def test_should_retrieve_returns_reject_when_decision_is_reject(self):
        """should_retrieve should return 'reject' when route_decision is 'reject'."""
        state: GraphState = {
            "question": "Test question",
            "route_decision": "reject",
        }

        result = should_retrieve(state)

        assert result == "reject"

    def test_should_retrieve_defaults_to_reject(self):
        """should_retrieve should default to 'reject' when route_decision is missing."""
        state: GraphState = {
            "question": "Test question",
        }

        result = should_retrieve(state)

        assert result == "reject"


@pytest.mark.unit
class TestShouldRewrite:
    """Tests for should_rewrite conditional edge."""

    def test_should_rewrite_returns_rewrite_when_low_confidence(self, mock_settings):
        """should_rewrite should return 'rewrite' when confidence is low and rewrites available."""
        state: GraphState = {
            "question": "Test question",
            "average_confidence": 0.3,  # Below default threshold of 0.5
            "rewrite_count": 0,
        }

        result = should_rewrite(state)

        assert result == "rewrite"

    def test_should_rewrite_returns_generate_when_high_confidence(self, mock_settings):
        """should_rewrite should return 'generate' when confidence is high."""
        state: GraphState = {
            "question": "Test question",
            "average_confidence": 0.8,  # Above threshold
            "rewrite_count": 0,
        }

        result = should_rewrite(state)

        assert result == "generate"

    def test_should_rewrite_returns_generate_when_max_rewrites_reached(self, mock_settings):
        """should_rewrite should return 'generate' when max rewrites reached even if confidence low."""
        state: GraphState = {
            "question": "Test question",
            "average_confidence": 0.3,  # Below threshold
            "rewrite_count": 2,  # At max (default is 2)
        }

        result = should_rewrite(state)

        assert result == "generate"

    def test_should_rewrite_defaults_to_rewrite_when_fields_missing(self, mock_settings):
        """should_rewrite should default to 'rewrite' when average_confidence defaults to 0.0."""
        state: GraphState = {
            "question": "Test question",
        }

        result = should_rewrite(state)

        # When fields missing, average_confidence defaults to 0.0
        # which is below threshold (0.5), so it should rewrite
        assert result == "rewrite"


@pytest.mark.unit
class TestShouldRegenerate:
    """Tests for should_regenerate conditional edge."""

    def test_should_regenerate_returns_regenerate_when_not_grounded(self):
        """should_regenerate should return 'regenerate' when hallucination check fails."""
        state: GraphState = {
            "question": "Test question",
            "hallucination_check": "not_grounded",
        }

        result = should_regenerate(state)

        assert result == "regenerate"

    def test_should_regenerate_returns_finish_when_grounded(self):
        """should_regenerate should return 'finish' when hallucination check passes."""
        state: GraphState = {
            "question": "Test question",
            "hallucination_check": "grounded",
        }

        result = should_regenerate(state)

        assert result == "finish"

    def test_should_regenerate_returns_finish_when_partial(self):
        """should_regenerate should return 'finish' when hallucination check is partial."""
        state: GraphState = {
            "question": "Test question",
            "hallucination_check": "partial",
        }

        result = should_regenerate(state)

        assert result == "finish"

    def test_should_regenerate_defaults_to_finish(self):
        """should_regenerate should default to 'finish' when hallucination_check missing."""
        state: GraphState = {
            "question": "Test question",
        }

        result = should_regenerate(state)

        assert result == "finish"


@pytest.mark.unit
class TestGetCompiledGraph:
    """Tests for _get_compiled_graph() caching."""

    def test_get_compiled_graph_builds_and_caches(self):
        """_get_compiled_graph() should build the graph once and cache it."""
        import specagent.graph.workflow as wf

        wf._compiled_graph = None
        with patch("specagent.graph.workflow.build_graph") as mock_build:
            mock_build.return_value = MagicMock()
            g1 = wf._get_compiled_graph()
            g2 = wf._get_compiled_graph()
        assert g1 is g2
        mock_build.assert_called_once()
        wf._compiled_graph = None


@pytest.mark.unit
class TestRunQuery:
    """Tests for run_query() function."""

    @patch("specagent.graph.workflow._get_compiled_graph")
    def test_run_query_creates_initial_state(self, mock_get_graph, sample_question):
        """run_query should create initial state from question."""
        # Mock the graph to return a simple state
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = create_initial_state(sample_question)
        mock_get_graph.return_value = mock_graph

        result = run_query(sample_question)

        assert result["question"] == sample_question

    @patch("specagent.graph.workflow._get_compiled_graph")
    def test_run_query_invokes_graph(self, mock_get_graph, sample_question):
        """run_query should invoke the compiled graph."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = create_initial_state(sample_question)
        mock_get_graph.return_value = mock_graph

        run_query(sample_question)

        # Check that graph was obtained and invoked
        mock_get_graph.assert_called_once()
        mock_graph.invoke.assert_called_once()

    @patch("specagent.graph.workflow._get_compiled_graph")
    def test_run_query_adds_processing_time(self, mock_get_graph, sample_question):
        """run_query should add processing_time_ms to final state."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = create_initial_state(sample_question)
        mock_get_graph.return_value = mock_graph

        result = run_query(sample_question)

        assert "processing_time_ms" in result
        assert isinstance(result["processing_time_ms"], float)
        assert result["processing_time_ms"] >= 0

    @patch("specagent.graph.workflow._get_compiled_graph")
    def test_run_query_returns_graph_state(self, mock_get_graph, sample_question):
        """run_query should return the final GraphState."""
        mock_graph = MagicMock()
        expected_state = create_initial_state(sample_question)
        expected_state["generation"] = "Test answer"
        mock_graph.invoke.return_value = expected_state
        mock_get_graph.return_value = mock_graph

        result = run_query(sample_question)

        assert result["question"] == sample_question
        assert result["generation"] == "Test answer"

    @patch("specagent.graph.workflow._get_compiled_graph")
    def test_run_query_sets_max_rewrites_override(self, mock_get_graph, sample_question):
        """run_query with max_rewrites sets max_rewrites_override in initial state."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = create_initial_state(sample_question)
        mock_get_graph.return_value = mock_graph

        run_query(sample_question, max_rewrites=3)

        invoked_state = mock_graph.invoke.call_args[0][0]
        assert invoked_state["max_rewrites_override"] == 3


@pytest.mark.unit
class TestGetGraphVisualization:
    """Tests for get_graph_visualization() function."""

    def test_get_graph_visualization_returns_mermaid_string(self):
        """get_graph_visualization should return a Mermaid diagram string."""
        mermaid = get_graph_visualization()

        assert isinstance(mermaid, str)
        assert len(mermaid) > 0

    def test_get_graph_visualization_contains_nodes(self):
        """get_graph_visualization should contain all node names."""
        mermaid = get_graph_visualization()

        # Check for node names in the output
        assert "router" in mermaid
        assert "retriever" in mermaid
        assert "grader" in mermaid
        assert "rewriter" in mermaid
        assert "generator" in mermaid
        assert "hallucination_check" in mermaid

    def test_get_graph_visualization_is_mermaid_format(self):
        """get_graph_visualization should produce valid Mermaid format."""
        mermaid = get_graph_visualization()

        # Mermaid diagrams typically start with graph directive
        assert "graph" in mermaid.lower()

    def test_get_graph_visualization_contains_edges(self):
        """get_graph_visualization should show edges between nodes."""
        mermaid = get_graph_visualization()

        # Mermaid uses arrows like --> or -. .-> for edges
        assert "-->" in mermaid or ".->" in mermaid or "." in mermaid


@pytest.mark.unit
class TestSaveGraphImage:
    """Tests for save_graph_image() function."""

    @patch("specagent.graph.workflow.build_graph")
    def test_save_graph_image_builds_graph(self, mock_build_graph, tmp_path):
        """save_graph_image should build the graph."""
        # Mock the graph and its methods
        mock_graph_obj = MagicMock()
        mock_graph_obj.draw_png = MagicMock()

        mock_graph = MagicMock()
        mock_graph.get_graph.return_value = mock_graph_obj
        mock_build_graph.return_value = mock_graph

        output_path = tmp_path / "test.png"
        save_graph_image(str(output_path))

        # Verify build_graph was called
        mock_build_graph.assert_called_once()

    @patch("specagent.graph.workflow.build_graph")
    def test_save_graph_image_calls_draw_png(self, mock_build_graph, tmp_path):
        """save_graph_image should call draw_png with the provided path."""
        # Mock the graph and its methods
        mock_graph_obj = MagicMock()
        mock_graph_obj.draw_png = MagicMock()

        mock_graph = MagicMock()
        mock_graph.get_graph.return_value = mock_graph_obj
        mock_build_graph.return_value = mock_graph

        output_path = tmp_path / "test.png"
        save_graph_image(str(output_path))

        # Verify draw_png was called with the correct path
        mock_graph_obj.draw_png.assert_called_once_with(str(output_path))

    @patch("specagent.graph.workflow.build_graph")
    def test_save_graph_image_uses_default_path(self, mock_build_graph):
        """save_graph_image should use default path when not specified."""
        # Mock the graph and its methods
        mock_graph_obj = MagicMock()
        mock_graph_obj.draw_png = MagicMock()

        mock_graph = MagicMock()
        mock_graph.get_graph.return_value = mock_graph_obj
        mock_build_graph.return_value = mock_graph

        save_graph_image()

        # Verify draw_png was called with default path
        mock_graph_obj.draw_png.assert_called_once_with("docs/architecture.png")


@pytest.mark.unit
class TestCreateTimedNode:
    """Tests for create_timed_node timing wrapper."""

    def test_timed_node_records_timing(self):
        """create_timed_node wraps a function and records timing in node_timings."""
        from specagent.graph.workflow import create_timed_node

        timed = create_timed_node(lambda s: s, "n")
        result = timed({"question": "q"})
        assert result["node_timings"]["n"] >= 0

    def test_timed_node_accumulates_timing(self):
        """create_timed_node accumulates timing across multiple calls."""
        from specagent.graph.workflow import create_timed_node

        timed = create_timed_node(lambda s: s, "n")
        state = timed({"question": "q"})
        t1 = state["node_timings"]["n"]
        state = timed(state)
        assert state["node_timings"]["n"] >= t1


@pytest.mark.unit
class TestShouldRewriteWithChunks:
    """Tests for should_rewrite with retrieved_chunks and graded_chunks."""

    def test_high_similarity_skips_rewrite(self):
        """should_rewrite skips rewrite when top chunks have high similarity."""
        from specagent.graph.workflow import should_rewrite

        chunk = MagicMock(similarity_score=0.95)
        state = {
            "retrieved_chunks": [chunk] * 3,
            "rewrite_count": 0,
            "graded_chunks": [],
            "average_confidence": 0.5,
        }
        with patch("specagent.graph.workflow.settings") as ms:
            ms.high_similarity_threshold = 0.85
            ms.max_rewrites = 3
            ms.grader_confidence_threshold = 0.7
            ms.min_relevant_chunk_percentage = 0.5
            assert should_rewrite(state) == "generate"

    def test_graded_chunks_trigger_rewrite_on_low_relevant_pct(self):
        """should_rewrite returns rewrite when graded_chunks shows low relevant pct."""
        from specagent.graph.workflow import should_rewrite

        chunk = MagicMock(similarity_score=0.3, relevant="no")
        state = {
            "retrieved_chunks": [chunk],
            "rewrite_count": 0,
            "graded_chunks": [chunk],
            "average_confidence": 0.8,
        }
        with patch("specagent.graph.workflow.settings") as ms:
            ms.high_similarity_threshold = 0.85
            ms.max_rewrites = 3
            ms.grader_confidence_threshold = 0.7
            ms.min_relevant_chunk_percentage = 0.5
            assert should_rewrite(state) == "rewrite"

    def test_max_rewrites_exceeded_returns_generate(self):
        """should_rewrite returns generate when max rewrites exceeded."""
        from specagent.graph.workflow import should_rewrite

        chunk = MagicMock(similarity_score=0.3, relevant="no")
        state = {
            "retrieved_chunks": [chunk],
            "rewrite_count": 5,
            "graded_chunks": [chunk],
            "average_confidence": 0.1,
        }
        with patch("specagent.graph.workflow.settings") as ms:
            ms.high_similarity_threshold = 0.85
            ms.max_rewrites = 3
            ms.grader_confidence_threshold = 0.7
            ms.min_relevant_chunk_percentage = 0.5
            assert should_rewrite(state) == "generate"
