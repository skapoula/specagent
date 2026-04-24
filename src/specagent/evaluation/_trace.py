"""Benchmark trace logger setup."""

import logging
import sys
from pathlib import Path
from typing import Any


def setup_trace_logging(output_dir: Path, timestamp: str, verbose: bool = False) -> logging.Logger:
    """
    Set up a dedicated trace logger for benchmark execution.

    Creates a file handler (always) and an optional console handler (verbose=True).
    Caller is responsible for closing handlers after use.

    Args:
        output_dir: Directory to write the trace log file
        timestamp: ISO timestamp used to name the log file
        verbose: If True, also stream trace output to stdout

    Returns:
        Configured trace logger
    """
    trace = logging.getLogger("benchmark_trace")
    trace.setLevel(logging.INFO)
    trace.propagate = False
    trace.handlers.clear()

    ts = timestamp.replace(":", "-").split(".")[0]
    log_path = output_dir / f"benchmark_trace_{ts}.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    trace.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        trace.addHandler(console_handler)

    return trace


def log_benchmark_summary(trace: logging.Logger, metrics: dict[str, Any]) -> None:
    """Write benchmark summary to the trace logger."""
    trace.info("=" * 80)
    trace.info("BENCHMARK SUMMARY")
    trace.info("=" * 80)
    trace.info(f"Total Questions: {metrics['total_questions']}")
    trace.info(f"Correct Answers: {metrics['correct_answers']}")
    trace.info(f"Accuracy: {metrics['accuracy']:.1%}")
    trace.info(f"Average Latency: {metrics['average_latency_ms']:.0f}ms")
    trace.info(f"Average Confidence: {metrics['average_confidence']:.2f}")
    trace.info("")
    if metrics["accuracy_by_difficulty"]:
        trace.info("Accuracy by Difficulty:")
        for diff in ["Easy", "Intermediate", "Hard"]:
            if diff in metrics["accuracy_by_difficulty"]:
                trace.info(f"  {diff}: {metrics['accuracy_by_difficulty'][diff]:.1%}")
        trace.info("")
    if metrics["average_node_timings"]:
        trace.info("Node Timing Breakdown (averages):")
        for node, avg_ms in sorted(metrics["average_node_timings"].items()):
            trace.info(f"  {node}: {avg_ms:.0f}ms")
        trace.info("")
    trace.info("")
