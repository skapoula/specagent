"""
Unit tests for benchmark runner.

Tests for:
    - Loading benchmark dataset from JSON
    - Running benchmark evaluation
    - Computing accuracy metrics by difficulty
    - Generating markdown reports
    - LLM-as-judge answer comparison
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from specagent.evaluation.benchmark import (
    BenchmarkQuestion,
    BenchmarkReport,
    BenchmarkResult,
    check_answer_correctness,
    load_benchmark_questions,
    run_benchmark,
)
from specagent.graph.state import GraphState

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_tspec_dataset(tmp_path: Path) -> Path:
    """Create sample TSpec-LLM format dataset."""
    data = {
        "question_1": {
            "question": "What is the maximum number of HARQ processes for NR?",
            "option_1": "8",
            "option_2": "16",
            "option_3": "32",
            "option_4": "64",
            "answer": "option_2: 16",
            "explanation": "The maximum number of HARQ processes is 16 for both FDD and TDD.",
            "category": "3GPP TR 38.321",
            "difficulty": "Easy",
        },
        "question_2": {
            "question": "What timer is started upon detection of radio link failure?",
            "option_1": "T300",
            "option_2": "T301",
            "option_3": "T310",
            "option_4": "T311",
            "answer": "option_4: T311",
            "explanation": "Timer T311 is started upon detection of radio link failure.",
            "category": "3GPP TR 38.331",
            "difficulty": "Intermediate",
        },
        "question_3": {
            "question": "What is the frequency range for FR1 in 5G NR?",
            "option_1": "410-7125 MHz",
            "option_2": "24250-52600 MHz",
            "option_3": "1-6 GHz",
            "option_4": "Above 24 GHz",
            "answer": "option_1: 410-7125 MHz",
            "explanation": "FR1 covers frequencies from 410 MHz to 7125 MHz.",
            "category": "3GPP TR 38.101-1",
            "difficulty": "Hard",
        },
    }

    dataset_path = tmp_path / "benchmark.json"
    with dataset_path.open("w") as f:
        json.dump(data, f, indent=2)

    return dataset_path


@pytest.fixture
def mock_graph_response():
    """Mock graph state response for benchmarking."""

    def _create_response(question: str, answer: str, latency: float = 1500.0) -> GraphState:
        return GraphState(
            question=question,
            route_decision="retrieve",
            route_reasoning="This is a 3GPP specification question",
            generation=answer,
            citations=[],
            retrieved_chunks=[],
            graded_chunks=[],
            rewrite_count=0,
            processing_time_ms=latency,
            average_confidence=0.85,
            hallucination_check="grounded",
            ungrounded_claims=[],
            error=None,
        )

    return _create_response


# =============================================================================
# Test load_benchmark_questions
# =============================================================================


def test_load_benchmark_questions_success(sample_tspec_dataset):
    """Test loading questions from TSpec-LLM format JSON."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    assert len(questions) == 3
    assert all(isinstance(q, BenchmarkQuestion) for q in questions)

    # Check first question
    q1 = questions[0]
    assert q1.id == "question_1"
    assert "HARQ processes" in q1.question
    assert q1.answer == "16"  # Parsed from "option_2: 16"
    assert q1.difficulty == "Easy"
    assert q1.category == "3GPP TR 38.321"
    assert "option_2" in q1.correct_option


def test_load_benchmark_questions_parsing_answer_format(sample_tspec_dataset):
    """Test that answer format 'option_X: text' is correctly parsed."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    # Check all answers are extracted correctly
    assert questions[0].answer == "16"
    assert questions[1].answer == "T311"
    assert questions[2].answer == "410-7125 MHz"

    # Check option numbers are preserved
    assert questions[0].correct_option == "option_2"
    assert questions[1].correct_option == "option_4"
    assert questions[2].correct_option == "option_1"


def test_load_benchmark_questions_empty_file(tmp_path):
    """Test loading from empty JSON file."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("{}")

    questions = load_benchmark_questions(empty_file)
    assert len(questions) == 0


def test_load_benchmark_questions_missing_file():
    """Test loading from non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_benchmark_questions("/nonexistent/path.json")


# =============================================================================
# Test check_answer_correctness
# =============================================================================


def test_check_answer_correctness_exact_match():
    """Test exact string match."""
    assert check_answer_correctness("16", "16", use_llm_judge=False)
    assert check_answer_correctness("T311", "T311", use_llm_judge=False)


def test_check_answer_correctness_case_insensitive():
    """Test case-insensitive matching."""
    assert check_answer_correctness("t311", "T311", use_llm_judge=False)
    assert check_answer_correctness("T311", "t311", use_llm_judge=False)


def test_check_answer_correctness_whitespace():
    """Test whitespace normalization."""
    assert check_answer_correctness("  16  ", "16", use_llm_judge=False)
    assert check_answer_correctness("16", "  16  ", use_llm_judge=False)


def test_check_answer_correctness_fuzzy_match():
    """Test fuzzy matching for similar strings."""
    # Should match "16 processes" with "16"
    assert check_answer_correctness("The answer is 16 HARQ processes", "16", use_llm_judge=False)

    # Should match when answer is embedded in response
    assert check_answer_correctness(
        "Timer T311 is used for this purpose", "T311", use_llm_judge=False
    )


def test_check_answer_correctness_incorrect():
    """Test detection of incorrect answers."""
    assert not check_answer_correctness("8", "16", use_llm_judge=False)
    assert not check_answer_correctness("T310", "T311", use_llm_judge=False)


@pytest.mark.unit
def test_check_answer_correctness_with_llm_judge():
    """Test LLM-as-judge for semantic matching."""
    with patch("specagent.evaluation.benchmark.llm_judge_answer") as mock_judge:
        mock_judge.return_value = True

        # Use an answer that won't match fuzzy matching to force LLM judge
        result = check_answer_correctness(
            "The system uses a different approach", "16", use_llm_judge=True
        )

        assert result is True
        mock_judge.assert_called_once()


# =============================================================================
# Test run_benchmark
# =============================================================================


def test_run_benchmark_basic(sample_tspec_dataset, mock_graph_response, tmp_path):
    """Test basic benchmark execution."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    # Mock the run_query function to return correct answers
    with patch("specagent.graph.workflow.run_query") as mock_run_query:
        mock_run_query.side_effect = [
            mock_graph_response("Q1", "16", 1200.0),
            mock_graph_response("Q2", "T311", 1500.0),
            mock_graph_response("Q3", "410-7125 MHz", 2000.0),
        ]

        report = run_benchmark(
            questions=questions,
            limit=None,
            output_dir=tmp_path / "results",
            skip_health_check=True,
        )

        assert report.total_questions == 3
        assert report.correct_answers == 3
        assert report.accuracy == 1.0
        assert len(report.results) == 3


def test_run_benchmark_with_limit(sample_tspec_dataset, mock_graph_response, tmp_path):
    """Test benchmark with question limit."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    with patch("specagent.graph.workflow.run_query") as mock_run_query:
        with patch("specagent.evaluation.benchmark.llm_judge_answer", return_value=False):
            mock_run_query.return_value = mock_graph_response("Q", "16", 1000.0)

            report = run_benchmark(
                questions=questions,
                limit=2,
                output_dir=tmp_path / "results",
                skip_health_check=True,
            )

            assert report.total_questions == 2
            assert len(report.results) == 2


def test_run_benchmark_accuracy_by_difficulty(sample_tspec_dataset, mock_graph_response, tmp_path):
    """Test accuracy computation by difficulty level."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    with patch("specagent.graph.workflow.run_query") as mock_run_query:
        with patch("specagent.evaluation.benchmark.llm_judge_answer", return_value=False):
            # Return correct for Easy, incorrect for Intermediate and Hard
            mock_run_query.side_effect = [
                mock_graph_response("Q1", "16", 1000.0),  # Easy - correct
                mock_graph_response("Q2", "T310", 1000.0),  # Intermediate - wrong
                mock_graph_response("Q3", "Wrong", 1000.0),  # Hard - wrong
            ]

            report = run_benchmark(
                questions=questions,
                limit=None,
                output_dir=tmp_path / "results",
                skip_health_check=True,
            )

            assert report.accuracy_by_difficulty["Easy"] == 1.0
            assert report.accuracy_by_difficulty["Intermediate"] == 0.0
            assert report.accuracy_by_difficulty["Hard"] == 0.0


def test_run_benchmark_saves_results(sample_tspec_dataset, mock_graph_response, tmp_path):
    """Test that benchmark saves results to JSON and markdown."""
    questions = load_benchmark_questions(sample_tspec_dataset)
    output_dir = tmp_path / "results"

    with patch("specagent.graph.workflow.run_query") as mock_run_query:
        mock_run_query.return_value = mock_graph_response("Q", "16", 1000.0)

        report = run_benchmark(
            questions=questions,
            limit=1,
            output_dir=output_dir,
            skip_health_check=True,
        )

        # Check that output directory was created
        assert output_dir.exists()

        # Check that JSON file was created
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) == 1

        # Check that markdown file was created
        md_files = list(output_dir.glob("*.md"))
        assert len(md_files) == 1


def test_run_benchmark_handles_errors(sample_tspec_dataset, tmp_path):
    """Test that benchmark handles errors gracefully."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    with patch("specagent.graph.workflow.run_query") as mock_run_query:
        # Simulate an error in the pipeline
        error_state = GraphState(
            question="Q1",
            error="Pipeline failed",
            route_decision="reject",
            processing_time_ms=100.0,
            generation=None,
        )
        mock_run_query.return_value = error_state

        report = run_benchmark(
            questions=questions,
            limit=1,
            output_dir=tmp_path / "results",
            skip_health_check=True,
        )

        # Should still generate report with error recorded
        assert report.total_questions == 1
        assert report.results[0].error == "Pipeline failed"
        assert not report.results[0].is_correct


# =============================================================================
# Test BenchmarkReport
# =============================================================================


def test_benchmark_report_to_dict():
    """Test conversion of report to dictionary."""
    result = BenchmarkResult(
        question_id="q1",
        question="Test?",
        expected_answer="16",
        generated_answer="16",
        is_correct=True,
        confidence=0.85,
        latency_ms=1500.0,
        difficulty="Easy",
        rewrites=0,
        error=None,
    )

    report = BenchmarkReport(
        timestamp="2024-01-01T12:00:00",
        total_questions=1,
        correct_answers=1,
        accuracy=1.0,
        accuracy_by_difficulty={"Easy": 1.0},
        average_latency_ms=1500.0,
        average_confidence=0.85,
        results=[result],
    )

    report_dict = report.to_dict()

    assert report_dict["total_questions"] == 1
    assert report_dict["accuracy"] == 1.0
    assert len(report_dict["results"]) == 1
    assert report_dict["results"][0]["is_correct"] is True


def test_benchmark_report_to_markdown():
    """Test markdown report generation."""
    result1 = BenchmarkResult(
        question_id="q1",
        question="What is X?",
        expected_answer="16",
        generated_answer="16",
        is_correct=True,
        confidence=0.85,
        latency_ms=1500.0,
        difficulty="Easy",
    )

    result2 = BenchmarkResult(
        question_id="q2",
        question="What is Y?",
        expected_answer="T311",
        generated_answer="T310",
        is_correct=False,
        confidence=0.70,
        latency_ms=2000.0,
        difficulty="Hard",
    )

    report = BenchmarkReport(
        timestamp="2024-01-01T12:00:00",
        total_questions=2,
        correct_answers=1,
        accuracy=0.5,
        accuracy_by_difficulty={"Easy": 1.0, "Hard": 0.0},
        average_latency_ms=1750.0,
        average_confidence=0.775,
        results=[result1, result2],
    )

    markdown = report.to_markdown()

    # Check header
    assert "# SpecAgent Benchmark Report" in markdown

    # Check summary table
    assert "Total Questions | 2" in markdown
    assert "Correct Answers | 1" in markdown
    assert "50.0%" in markdown

    # Check difficulty breakdown
    assert "Easy" in markdown
    assert "Hard" in markdown

    # Check failed questions section
    assert "Failed Questions" in markdown
    assert "q2" in markdown
    assert "What is Y?" in markdown


def test_benchmark_report_markdown_no_failures():
    """Test markdown report when all questions pass."""
    result = BenchmarkResult(
        question_id="q1",
        question="Test?",
        expected_answer="16",
        generated_answer="16",
        is_correct=True,
        confidence=0.85,
        latency_ms=1500.0,
        difficulty="Easy",
    )

    report = BenchmarkReport(
        timestamp="2024-01-01T12:00:00",
        total_questions=1,
        correct_answers=1,
        accuracy=1.0,
        accuracy_by_difficulty={"Easy": 1.0},
        average_latency_ms=1500.0,
        average_confidence=0.85,
        results=[result],
    )

    markdown = report.to_markdown()

    assert "No failed questions!" in markdown


# =============================================================================
# Test BenchmarkQuestion
# =============================================================================


def test_benchmark_question_creation():
    """Test BenchmarkQuestion dataclass."""
    question = BenchmarkQuestion(
        id="q1",
        question="What is the answer?",
        answer="42",
        difficulty="medium",
        category="Test Category",
        correct_option="option_1",
        spec_references=["TS38.321"],
    )

    assert question.id == "q1"
    assert question.answer == "42"
    assert question.difficulty == "medium"


# =============================================================================
# Test Confidence Analysis
# =============================================================================


def test_compute_confidence_distribution():
    """Test confidence distribution computation."""
    from specagent.evaluation.benchmark import compute_confidence_distribution

    results = [
        BenchmarkResult(
            question_id="q1",
            question="Q1",
            expected_answer="A1",
            generated_answer="A1",
            is_correct=True,
            confidence=0.95,
            latency_ms=1000,
            difficulty="Easy",
        ),
        BenchmarkResult(
            question_id="q2",
            question="Q2",
            expected_answer="A2",
            generated_answer="A2",
            is_correct=True,
            confidence=0.85,
            latency_ms=1000,
            difficulty="Easy",
        ),
        BenchmarkResult(
            question_id="q3",
            question="Q3",
            expected_answer="A3",
            generated_answer="A3",
            is_correct=True,
            confidence=0.75,
            latency_ms=1000,
            difficulty="Medium",
        ),
        BenchmarkResult(
            question_id="q4",
            question="Q4",
            expected_answer="A4",
            generated_answer="A4",
            is_correct=False,
            confidence=0.45,
            latency_ms=1000,
            difficulty="Hard",
        ),
        BenchmarkResult(
            question_id="q5",
            question="Q5",
            expected_answer="A5",
            generated_answer="A5",
            is_correct=False,
            confidence=0.25,
            latency_ms=1000,
            difficulty="Hard",
        ),
    ]

    distribution = compute_confidence_distribution(results)

    # Should have 5 bins: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    assert len(distribution) == 5

    # Check bins
    assert distribution["0.0-0.2"] == 0
    assert distribution["0.2-0.4"] == 1  # q5 at 0.25
    assert distribution["0.4-0.6"] == 1  # q4 at 0.45
    assert distribution["0.6-0.8"] == 1  # q3 at 0.75
    assert distribution["0.8-1.0"] == 2  # q1 at 0.95, q2 at 0.85


def test_compute_confidence_distribution_empty():
    """Test confidence distribution with no results."""
    from specagent.evaluation.benchmark import compute_confidence_distribution

    distribution = compute_confidence_distribution([])

    assert len(distribution) == 5
    assert all(count == 0 for count in distribution.values())


def test_confidence_statistics():
    """Test confidence statistics computation."""
    from specagent.evaluation.benchmark import compute_confidence_stats

    results = [
        BenchmarkResult(
            question_id="q1",
            question="Q1",
            expected_answer="A1",
            generated_answer="A1",
            is_correct=True,
            confidence=0.9,
            latency_ms=1000,
            difficulty="Easy",
        ),
        BenchmarkResult(
            question_id="q2",
            question="Q2",
            expected_answer="A2",
            generated_answer="A2",
            is_correct=True,
            confidence=0.8,
            latency_ms=1000,
            difficulty="Medium",
        ),
        BenchmarkResult(
            question_id="q3",
            question="Q3",
            expected_answer="A3",
            generated_answer="A3",
            is_correct=False,
            confidence=0.5,
            latency_ms=1000,
            difficulty="Hard",
        ),
    ]

    stats = compute_confidence_stats(results)

    assert stats["mean"] == pytest.approx(0.7333, abs=0.001)
    assert stats["median"] == 0.8
    assert stats["min"] == 0.5
    assert stats["max"] == 0.9
    assert stats["std"] == pytest.approx(0.2082, abs=0.001)


def test_confidence_by_correctness():
    """Test confidence analysis by answer correctness."""
    from specagent.evaluation.benchmark import analyze_confidence_by_correctness

    results = [
        BenchmarkResult(
            question_id="q1",
            question="Q1",
            expected_answer="A1",
            generated_answer="A1",
            is_correct=True,
            confidence=0.9,
            latency_ms=1000,
            difficulty="Easy",
        ),
        BenchmarkResult(
            question_id="q2",
            question="Q2",
            expected_answer="A2",
            generated_answer="A2",
            is_correct=True,
            confidence=0.85,
            latency_ms=1000,
            difficulty="Medium",
        ),
        BenchmarkResult(
            question_id="q3",
            question="Q3",
            expected_answer="A3",
            generated_answer="Wrong",
            is_correct=False,
            confidence=0.6,
            latency_ms=1000,
            difficulty="Hard",
        ),
        BenchmarkResult(
            question_id="q4",
            question="Q4",
            expected_answer="A4",
            generated_answer="Wrong",
            is_correct=False,
            confidence=0.4,
            latency_ms=1000,
            difficulty="Hard",
        ),
    ]

    analysis = analyze_confidence_by_correctness(results)

    assert analysis["correct"]["mean"] == pytest.approx(0.875, abs=0.001)
    assert analysis["correct"]["count"] == 2
    assert analysis["incorrect"]["mean"] == pytest.approx(0.5, abs=0.001)
    assert analysis["incorrect"]["count"] == 2


def test_benchmark_report_includes_confidence_distribution():
    """Test that BenchmarkReport includes confidence distribution."""
    result1 = BenchmarkResult(
        question_id="q1",
        question="Q1",
        expected_answer="A1",
        generated_answer="A1",
        is_correct=True,
        confidence=0.9,
        latency_ms=1000,
        difficulty="Easy",
    )
    result2 = BenchmarkResult(
        question_id="q2",
        question="Q2",
        expected_answer="A2",
        generated_answer="A2",
        is_correct=False,
        confidence=0.5,
        latency_ms=1000,
        difficulty="Hard",
    )

    report = BenchmarkReport(
        timestamp="2024-01-01T12:00:00",
        total_questions=2,
        correct_answers=1,
        accuracy=0.5,
        accuracy_by_difficulty={"Easy": 1.0, "Hard": 0.0},
        average_latency_ms=1000.0,
        average_confidence=0.7,
        results=[result1, result2],
        confidence_distribution={"0.8-1.0": 1, "0.4-0.6": 1},
        confidence_stats={"mean": 0.7, "median": 0.7, "min": 0.5, "max": 0.9},
    )

    assert "confidence_distribution" in report.to_dict()
    assert "confidence_stats" in report.to_dict()


def test_benchmark_report_markdown_includes_confidence():
    """Test that markdown report includes confidence analysis."""
    result = BenchmarkResult(
        question_id="q1",
        question="Q1",
        expected_answer="A1",
        generated_answer="A1",
        is_correct=True,
        confidence=0.9,
        latency_ms=1000,
        difficulty="Easy",
    )

    report = BenchmarkReport(
        timestamp="2024-01-01T12:00:00",
        total_questions=1,
        correct_answers=1,
        accuracy=1.0,
        accuracy_by_difficulty={"Easy": 1.0},
        average_latency_ms=1000.0,
        average_confidence=0.9,
        results=[result],
        confidence_distribution={"0.8-1.0": 1},
        confidence_stats={"mean": 0.9, "median": 0.9, "min": 0.9, "max": 0.9, "std": 0.0},
    )

    markdown = report.to_markdown()

    assert "Confidence Distribution" in markdown
    assert "0.8-1.0" in markdown
    assert "Confidence Statistics" in markdown


# =============================================================================
# Additional gap-filling tests
# =============================================================================


def test_load_benchmark_questions_answer_no_colon(tmp_path):
    """load_benchmark_questions() handles answers without ':' separator."""
    data = {"q1": {"question": "What is X?", "answer": "42", "difficulty": "Easy"}}
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(data))
    qs = load_benchmark_questions(path)
    assert qs[0].answer == "42"
    assert qs[0].correct_option == ""


def test_compute_confidence_stats_empty_results():
    """compute_confidence_stats() returns zeros for empty results list."""
    from specagent.evaluation.benchmark import compute_confidence_stats

    result = compute_confidence_stats([])
    assert result == {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}


def test_setup_trace_logging_verbose(tmp_path):
    """setup_trace_logging() adds console handler when verbose=True."""
    from specagent.evaluation.benchmark import setup_trace_logging

    logger = setup_trace_logging(tmp_path, "2026-03-01T12:00:00", verbose=True)
    assert len(logger.handlers) == 2  # file + console
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)


def test_run_benchmark_health_check_failure(tmp_path):
    """run_benchmark raises RuntimeError when health check fails."""
    with (
        patch(
            "specagent.llm.factory.check_llm_health",
            return_value=(False, "endpoint down"),
        ),
        pytest.raises(RuntimeError, match="LLM endpoint unavailable"),
    ):
        run_benchmark(
            questions=[BenchmarkQuestion(id="q1", question="Q?", answer="A", difficulty="Easy")],
            output_dir=tmp_path / "results",
            skip_health_check=False,
        )


def test_run_benchmark_health_check_success_prints_message(tmp_path):
    """run_benchmark continues and prints success when health check passes (line 415)."""
    qs = [BenchmarkQuestion(id="q1", question="Q?", answer="A", difficulty="Easy")]
    mock_state = {
        "question": "Q?",
        "route_decision": "retrieve",
        "route_reasoning": "",
        "generation": "A",
        "citations": [],
        "retrieved_chunks": [],
        "graded_chunks": [],
        "rewrite_count": 0,
        "processing_time_ms": 100.0,
        "average_confidence": 0.9,
        "hallucination_check": "grounded",
        "ungrounded_claims": [],
        "error": None,
        "node_timings": {},
        "llm_calls": [],
    }
    with (
        patch(
            "specagent.llm.factory.check_llm_health",
            return_value=(True, "endpoint OK"),
        ),
        patch("specagent.graph.workflow.run_query", return_value=mock_state),
        patch("specagent.evaluation.benchmark.check_answer_correctness", return_value=True),
    ):
        report = run_benchmark(
            questions=qs,
            output_dir=tmp_path / "results",
            skip_health_check=False,
        )
    assert report.total_questions == 1


def test_run_benchmark_rejected_question(tmp_path):
    """run_benchmark records rejection when route_decision is 'reject'."""
    qs = [BenchmarkQuestion(id="q1", question="Who is POTUS?", answer="A", difficulty="Easy")]
    mock_state = {
        "question": "Who is POTUS?",
        "route_decision": "reject",
        "route_reasoning": "Not a 3GPP question",
        "generation": "",
        "citations": [],
        "retrieved_chunks": [],
        "graded_chunks": [],
        "rewrite_count": 0,
        "processing_time_ms": 50.0,
        "average_confidence": 0.0,
        "hallucination_check": "grounded",
        "ungrounded_claims": [],
        "error": None,
        "node_timings": {},
        "llm_calls": [],
    }
    with patch("specagent.graph.workflow.run_query", return_value=mock_state):
        report = run_benchmark(
            questions=qs, output_dir=tmp_path / "results", skip_health_check=True
        )
    assert report.results[0].error == "Question was rejected by router"


def test_run_benchmark_verbose_paths(tmp_path):
    """run_benchmark covers verbose logging for route_reasoning, graded_chunks, rewrites, timings."""
    qs = [BenchmarkQuestion(id="q1", question="Q?", answer="16", difficulty="Easy")]
    graded_chunk = MagicMock()
    graded_chunk.relevant = "yes"
    mock_state = {
        "question": "Q?",
        "route_decision": "retrieve",
        "route_reasoning": "Relevant 3GPP question",
        "generation": "",  # empty → covers 516->521 False branch
        "citations": [],
        "retrieved_chunks": [],
        "graded_chunks": [graded_chunk],
        "rewrite_count": 2,
        "processing_time_ms": 1000.0,
        "average_confidence": 0.75,
        "hallucination_check": "grounded",
        "ungrounded_claims": [],
        "error": None,
        "node_timings": {"router": 50.0, "retriever": 300.0},
        "llm_calls": [],
    }
    with (
        patch("specagent.graph.workflow.run_query", return_value=mock_state),
        patch("specagent.evaluation.benchmark.check_answer_correctness", return_value=False),
    ):
        report = run_benchmark(
            questions=qs,
            output_dir=tmp_path / "results",
            skip_health_check=True,
            verbose=True,
        )
    assert report.total_questions == 1


def test_run_benchmark_exception_in_query(tmp_path):
    """run_benchmark catches per-question exceptions and records them as errors."""
    qs = [BenchmarkQuestion(id="q1", question="Q?", answer="A", difficulty="Easy")]
    with patch("specagent.graph.workflow.run_query", side_effect=RuntimeError("graph exploded")):
        report = run_benchmark(
            questions=qs, output_dir=tmp_path / "results", skip_health_check=True
        )
    assert report.results[0].error == "graph exploded"
    assert report.results[0].is_correct is False


def test_run_benchmark_empty_questions_skips_difficulty_block(tmp_path):
    """run_benchmark with limit=0 leaves accuracy_by_difficulty empty (covers 623->632 False branch)."""
    qs = [BenchmarkQuestion(id="q1", question="Q?", answer="A", difficulty="Easy")]
    report = run_benchmark(
        questions=qs,
        limit=0,
        output_dir=tmp_path / "results",
        skip_health_check=True,
    )
    assert report.total_questions == 0
    assert report.accuracy_by_difficulty == {}


def test_check_answer_correctness_word_based_match():
    """check_answer_correctness uses word-set matching for multi-word answers."""
    result = check_answer_correctness(
        "The HARQ process uses 16 processes in NR",
        "HARQ 16",
        use_llm_judge=False,
    )
    assert result is True


def test_llm_judge_answer_yes_with_content_attribute():
    """llm_judge_answer returns True when LLM response has .content='yes'."""
    from specagent.evaluation.benchmark import llm_judge_answer

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "yes"
    mock_llm.invoke.return_value = mock_response
    with patch("specagent.llm.factory.get_llm", return_value=mock_llm):
        assert llm_judge_answer("16 HARQ processes", "16") is True


def test_llm_judge_answer_no_str_response():
    """llm_judge_answer returns False when LLM returns string 'no'."""
    from specagent.evaluation.benchmark import llm_judge_answer

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "no"
    with patch("specagent.llm.factory.get_llm", return_value=mock_llm):
        assert llm_judge_answer("completely wrong answer", "16") is False


def test_llm_judge_answer_exception_falls_back_to_fuzzy():
    """llm_judge_answer falls back to substring match when LLM raises."""
    from specagent.evaluation.benchmark import llm_judge_answer

    with patch("specagent.llm.factory.get_llm", side_effect=RuntimeError("no llm")):
        # "16" in "16 processes" → True
        assert llm_judge_answer("16 processes", "16") is True
        # "16" not in "wrong answer" → False
        assert llm_judge_answer("wrong answer", "16") is False
