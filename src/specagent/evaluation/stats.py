"""Confidence statistics and accuracy analysis utilities."""

import statistics
from typing import Any

from specagent.evaluation.models import BenchmarkResult


def compute_confidence_distribution(results: list[BenchmarkResult]) -> dict[str, int]:
    """
    Compute confidence distribution histogram.

    Groups scores into five bins: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0].

    Args:
        results: List of benchmark results

    Returns:
        Dictionary mapping bin labels to counts
    """
    bins: dict[str, int] = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0,
    }
    for result in results:
        c = result.confidence
        if c < 0.2:
            bins["0.0-0.2"] += 1
        elif c < 0.4:
            bins["0.2-0.4"] += 1
        elif c < 0.6:
            bins["0.4-0.6"] += 1
        elif c < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    return bins


def compute_confidence_stats(results: list[BenchmarkResult]) -> dict[str, float]:
    """
    Compute confidence statistics (mean, median, min, max, std).

    Args:
        results: List of benchmark results

    Returns:
        Dictionary with mean, median, min, max, std
    """
    if not results:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    confidences = [r.confidence for r in results]
    return {
        "mean": statistics.mean(confidences),
        "median": statistics.median(confidences),
        "min": min(confidences),
        "max": max(confidences),
        "std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
    }


def analyze_confidence_by_correctness(
    results: list[BenchmarkResult],
) -> dict[str, dict[str, float]]:
    """
    Analyze confidence levels by answer correctness.

    Args:
        results: List of benchmark results

    Returns:
        Dictionary with 'correct' and 'incorrect' statistics
    """
    correct = [r for r in results if r.is_correct]
    incorrect = [r for r in results if not r.is_correct]
    correct_confidences = [r.confidence for r in correct] if correct else [0.0]
    incorrect_confidences = [r.confidence for r in incorrect] if incorrect else [0.0]
    return {
        "correct": {
            "mean": statistics.mean(correct_confidences),
            "count": len(correct),
        },
        "incorrect": {
            "mean": statistics.mean(incorrect_confidences),
            "count": len(incorrect),
        },
    }


def compute_report_metrics(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Compute all aggregate metrics needed to populate a BenchmarkReport.

    Args:
        results: List of benchmark results

    Returns:
        Dict whose keys match the BenchmarkReport fields (excluding timestamp and results).
    """
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / total if total > 0 else 0.0
    accuracy_by_difficulty: dict[str, float] = {}
    for diff in ["Easy", "Intermediate", "Hard"]:
        diff_results = [r for r in results if r.difficulty == diff]
        if diff_results:
            accuracy_by_difficulty[diff] = sum(1 for r in diff_results if r.is_correct) / len(
                diff_results
            )
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0.0
    avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0.0
    all_timings: dict[str, list[float]] = {}
    for r in results:
        for node, ms in r.node_timings.items():
            all_timings.setdefault(node, []).append(ms)
    avg_node_timings = {node: sum(ts) / len(ts) for node, ts in all_timings.items()}
    return {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "accuracy_by_difficulty": accuracy_by_difficulty,
        "average_latency_ms": avg_latency,
        "average_confidence": avg_confidence,
        "confidence_distribution": compute_confidence_distribution(results),
        "confidence_stats": compute_confidence_stats(results),
        "average_node_timings": avg_node_timings,
    }
