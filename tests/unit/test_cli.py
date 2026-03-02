"""Unit tests for the CLI commands."""

import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch

from specagent.cli import app

runner = CliRunner()

_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 8000
_UVICORN_APP = "specagent.api.main:app"
_DEFAULT_LIBRARY = "3gpp-specs"


@pytest.mark.unit
def test_index_command_has_docs_dir_option():
    """specagent index --help shows --docs-dir option (not --download)."""
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "--docs-dir" in result.output
    assert "--download" not in result.output


@pytest.mark.unit
def test_index_command_has_library_option():
    """specagent index --help shows --library option."""
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "--library" in result.output


@pytest.mark.unit
def test_index_command_has_force_option():
    """specagent index --help shows --force option."""
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestServeCommand:
    def test_serve_invokes_uvicorn(self):
        """serve command calls uvicorn.run with the app path."""
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            result = runner.invoke(app, ["serve"])
        mock_uvicorn.run.assert_called_once()
        assert result.exit_code == 0

    def test_serve_custom_port(self):
        """serve --port passes the port value to uvicorn.run."""
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            runner.invoke(app, ["serve", "--port", "9000"])
        assert mock_uvicorn.run.call_args.kwargs["port"] == 9000

    def test_serve_default_host_and_port(self):
        """serve uses 0.0.0.0:8000 by default."""
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            runner.invoke(app, ["serve"])
        assert mock_uvicorn.run.call_args.kwargs["host"] == _DEFAULT_HOST
        assert mock_uvicorn.run.call_args.kwargs["port"] == _DEFAULT_PORT

    def test_serve_reload_flag(self):
        """serve --reload passes reload=True to uvicorn.run."""
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            runner.invoke(app, ["serve", "--reload"])
        assert mock_uvicorn.run.call_args.kwargs["reload"] is True

    def test_serve_app_path(self):
        """serve passes the correct app module path to uvicorn.run."""
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            runner.invoke(app, ["serve"])
        assert mock_uvicorn.run.call_args.args[0] == _UVICORN_APP


# ---------------------------------------------------------------------------
# query command
# ---------------------------------------------------------------------------

def _ok_result() -> dict[str, object]:
    """Return a typical successful GraphState-like dict."""
    return {
        "route_decision": "retrieve",
        "generation": "The answer is 16.",
        "route_reasoning": "",
        "average_confidence": 0.9,
        "citations": [],
        "graded_chunks": [],
        "retrieved_chunks": [],
        "hallucination_check": "grounded",
        "rewrite_count": 0,
        "processing_time_ms": 500.0,
        "error": None,
    }


@pytest.mark.unit
class TestQueryCommand:
    def test_query_prints_answer(self):
        """query prints the generation text on success."""
        with patch("specagent.graph.workflow.run_query", return_value=_ok_result()):
            result = runner.invoke(app, ["query", "What is HARQ?"])
        assert result.exit_code == 0
        assert "16" in result.output

    def test_query_off_topic_shows_rejection_message(self):
        """query shows rejection message when route_decision is reject."""
        r = _ok_result()
        r["route_decision"] = "reject"
        r["generation"] = None
        with patch("specagent.graph.workflow.run_query", return_value=r):
            result = runner.invoke(app, ["query", "What is Python?"])
        assert result.exit_code == 0
        assert "outside" in result.output.lower() or "reject" in result.output.lower()

    def test_query_error_in_result_does_not_crash(self):
        """query completes without unhandled exception when result contains an error."""
        r = _ok_result()
        r["error"] = "LLM timeout"
        with patch("specagent.graph.workflow.run_query", return_value=r):
            result = runner.invoke(app, ["query", "test question here"])
        assert result.exit_code == 0

    def test_query_exception_exits_with_error(self):
        """query exits non-zero or captures exception when run_query raises."""
        with patch("specagent.graph.workflow.run_query", side_effect=RuntimeError("LLM down")):
            result = runner.invoke(app, ["query", "What is HARQ?"])
        assert isinstance(result.exception, RuntimeError)
        assert "LLM down" in str(result.exception)

    def test_query_verbose_shows_metadata_table(self):
        """query --verbose renders a metadata table."""
        with patch("specagent.graph.workflow.run_query", return_value=_ok_result()):
            result = runner.invoke(app, ["query", "--verbose", "What is HARQ?"])
        assert result.exit_code == 0
        # The verbose table includes Latency and Confidence
        assert "Latency" in result.output or "Metadata" in result.output

    def test_query_verbose_rejection_shows_reasoning(self):
        """query --verbose on a rejected question prints the reasoning."""
        r = _ok_result()
        r["route_decision"] = "reject"
        r["route_reasoning"] = "Not 3GPP related."
        r["generation"] = None
        with patch("specagent.graph.workflow.run_query", return_value=r):
            result = runner.invoke(app, ["query", "--verbose", "What is Python?"])
        assert result.exit_code == 0
        assert "Not 3GPP related." in result.output

    def test_query_with_citations(self):
        """query output includes citation lines when citations are returned."""
        r = _ok_result()
        mock_citation = MagicMock()
        mock_citation.raw_citation = "3GPP TS 38.214, Section 5.1"
        r["citations"] = [mock_citation]
        with patch("specagent.graph.workflow.run_query", return_value=r):
            result = runner.invoke(app, ["query", "What is HARQ?"])
        assert result.exit_code == 0
        assert "3GPP TS 38.214" in result.output


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndexCommand:
    def test_index_calls_ingest_folder(self, tmp_path):
        """index invokes asyncio.run(ingest_folder(...)) and prints results."""
        from specagent.retrieval.ingestor import BulkIngestResult

        mock_result = BulkIngestResult(
            folder=str(tmp_path),
            library=_DEFAULT_LIBRARY,
            total_files=3,
            indexed=3,
            replaced=0,
            skipped=0,
            failed=0,
            results=[],
            errors=[],
        )

        with patch("asyncio.run", return_value=mock_result) as mock_run:
            result = runner.invoke(app, ["index", "--docs-dir", str(tmp_path)])

        mock_run.assert_called_once()
        assert result.exit_code == 0
        assert "Indexing complete" in result.output or "Indexed" in result.output

    def test_index_missing_docs_dir_exits_with_error(self, tmp_path):
        """index exits with code 1 when the target directory does not exist."""
        nonexistent = str(tmp_path / "does_not_exist")
        result = runner.invoke(app, ["index", "--docs-dir", nonexistent])
        assert result.exit_code == 1

    def test_index_force_flag_clears_library(self, tmp_path):
        """index --force calls store.list_documents and store.delete_document."""
        from specagent.retrieval.ingestor import BulkIngestResult

        mock_result = BulkIngestResult(
            folder=str(tmp_path),
            library=_DEFAULT_LIBRARY,
            total_files=0,
            indexed=0,
            replaced=0,
            skipped=0,
            failed=0,
            results=[],
            errors=[],
        )

        mock_store = MagicMock()
        mock_store.list_documents.return_value = [
            {"doc_id": "doc1"}, {"doc_id": "doc2"}
        ]

        with patch("specagent.retrieval.resources.get_store", return_value=mock_store), \
             patch("asyncio.run", return_value=mock_result):
            result = runner.invoke(
                app, ["index", "--docs-dir", str(tmp_path), "--force"]
            )

        assert result.exit_code == 0
        mock_store.delete_document.assert_called()

    def test_index_force_store_error_shows_warning(self, tmp_path):
        """index --force continues with a warning when the store raises an exception."""
        from specagent.retrieval.ingestor import BulkIngestResult

        mock_result = BulkIngestResult(
            folder=str(tmp_path),
            library=_DEFAULT_LIBRARY,
            total_files=0,
            indexed=0,
            replaced=0,
            skipped=0,
            failed=0,
            results=[],
            errors=[],
        )

        with patch("specagent.retrieval.resources.get_store", side_effect=RuntimeError("DB error")), \
             patch("asyncio.run", return_value=mock_result):
            result = runner.invoke(
                app, ["index", "--docs-dir", str(tmp_path), "--force"]
            )

        # Should warn but not crash
        assert result.exit_code == 0
        assert "Warning" in result.output or "warn" in result.output.lower()

    def test_index_with_errors_exits_nonzero(self, tmp_path):
        """index exits with code 1 when the BulkIngestResult reports errors."""
        from specagent.retrieval.ingestor import BulkIngestResult

        mock_result = BulkIngestResult(
            folder=str(tmp_path),
            library=_DEFAULT_LIBRARY,
            total_files=2,
            indexed=1,
            replaced=0,
            skipped=0,
            failed=1,
            results=[],
            errors=[{"file": "broken.docx", "error": "parse error"}],
        )

        with patch("asyncio.run", return_value=mock_result):
            result = runner.invoke(app, ["index", "--docs-dir", str(tmp_path)])

        assert result.exit_code == 1
        assert "broken.docx" in result.output

    def test_index_custom_library_flag(self, tmp_path):
        """index --library passes the library name through to ingest_folder."""
        from specagent.retrieval.ingestor import BulkIngestResult

        mock_result = BulkIngestResult(
            folder=str(tmp_path),
            library="custom-lib",
            total_files=0,
            indexed=0,
            replaced=0,
            skipped=0,
            failed=0,
            results=[],
            errors=[],
        )

        with patch("asyncio.run", return_value=mock_result) as mock_run:
            result = runner.invoke(
                app,
                ["index", "--docs-dir", str(tmp_path), "--library", "custom-lib"],
            )

        mock_run.assert_called_once()
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# benchmark command
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBenchmarkCommand:
    def test_benchmark_missing_dataset_exits_with_error(self, tmp_path):
        """benchmark exits with code 1 when the dataset file does not exist."""
        nonexistent = str(tmp_path / "no_such_benchmark.json")
        result = runner.invoke(app, ["benchmark", "--dataset", nonexistent])
        assert result.exit_code == 1

    def test_benchmark_existing_dataset_runs(self, tmp_path):
        """benchmark with an existing dataset file proceeds past the file check."""
        dataset_file = tmp_path / "bench.json"
        dataset_file.write_text("[]")

        mock_questions = []

        with patch(
            "specagent.evaluation.benchmark.load_benchmark_questions",
            return_value=mock_questions,
        ):
            result = runner.invoke(
                app, ["benchmark", "--dataset", str(dataset_file)]
            )

        # Command runs to completion (not-yet-implemented message is expected)
        assert result.exit_code == 0

    def test_benchmark_limit_option(self, tmp_path):
        """benchmark --limit message appears in output when limit is applied."""
        dataset_file = tmp_path / "bench.json"
        dataset_file.write_text("[]")

        mock_questions = [MagicMock() for _ in range(5)]

        with patch(
            "specagent.evaluation.benchmark.load_benchmark_questions",
            return_value=mock_questions,
        ):
            result = runner.invoke(
                app,
                [
                    "benchmark",
                    "--dataset",
                    str(dataset_file),
                    "--limit",
                    "3",
                ],
            )

        assert result.exit_code == 0
        assert "3" in result.output


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVersionCommand:
    def test_version_shows_version_string(self):
        """version command prints the package version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "SpecAgent" in result.output or "v" in result.output
