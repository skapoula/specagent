"""Unit tests for specagent.evaluation.stats module."""

import pytest

from specagent.evaluation.models import BenchmarkResult


def _make_result(confidence: float, is_correct: bool) -> BenchmarkResult:
    return BenchmarkResult(
        question_id="q1",
        question="q",
        expected_answer="a",
        generated_answer="a",
        is_correct=is_correct,
        confidence=confidence,
        latency_ms=100.0,
        difficulty="Easy",
    )


@pytest.mark.unit
class TestComputeConfidenceDistribution:
    def test_bins_correct(self):
        from specagent.evaluation.stats import compute_confidence_distribution

        results = [
            _make_result(0.1, True),
            _make_result(0.3, True),
            _make_result(0.5, True),
            _make_result(0.7, True),
            _make_result(0.9, True),
        ]
        dist = compute_confidence_distribution(results)
        assert dist["0.0-0.2"] == 1
        assert dist["0.2-0.4"] == 1
        assert dist["0.4-0.6"] == 1
        assert dist["0.6-0.8"] == 1
        assert dist["0.8-1.0"] == 1

    def test_empty_results(self):
        from specagent.evaluation.stats import compute_confidence_distribution

        dist = compute_confidence_distribution([])
        assert all(v == 0 for v in dist.values())


@pytest.mark.unit
class TestComputeConfidenceStats:
    def test_returns_all_keys(self):
        from specagent.evaluation.stats import compute_confidence_stats

        results = [_make_result(0.6, True), _make_result(0.8, True)]
        stats = compute_confidence_stats(results)
        assert set(stats) == {"mean", "median", "min", "max", "std"}

    def test_empty_returns_zeros(self):
        from specagent.evaluation.stats import compute_confidence_stats

        stats = compute_confidence_stats([])
        assert stats == {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

    def test_single_result_std_zero(self):
        from specagent.evaluation.stats import compute_confidence_stats

        stats = compute_confidence_stats([_make_result(0.7, True)])
        assert stats["std"] == 0.0
        assert stats["mean"] == pytest.approx(0.7)


@pytest.mark.unit
class TestAnalyzeConfidenceByCorrectness:
    def test_separates_correct_incorrect(self):
        from specagent.evaluation.stats import analyze_confidence_by_correctness

        results = [
            _make_result(0.9, True),
            _make_result(0.8, True),
            _make_result(0.2, False),
        ]
        analysis = analyze_confidence_by_correctness(results)
        assert analysis["correct"]["count"] == 2
        assert analysis["incorrect"]["count"] == 1
        assert analysis["correct"]["mean"] > analysis["incorrect"]["mean"]

    def test_all_correct(self):
        from specagent.evaluation.stats import analyze_confidence_by_correctness

        results = [_make_result(0.8, True), _make_result(0.9, True)]
        analysis = analyze_confidence_by_correctness(results)
        assert analysis["correct"]["count"] == 2
        assert analysis["incorrect"]["count"] == 0
