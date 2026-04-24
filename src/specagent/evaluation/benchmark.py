"""TSpec-LLM benchmark runner orchestration."""

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from specagent.evaluation._trace import log_benchmark_summary, setup_trace_logging
from specagent.evaluation.judge import check_answer_correctness
from specagent.evaluation.models import (
    BenchmarkQuestion,
    BenchmarkReport,
    BenchmarkResult,
)
from specagent.evaluation.stats import (
    compute_report_metrics,
)

logger = logging.getLogger(__name__)


def load_benchmark_questions(dataset_path: Path) -> list[BenchmarkQuestion]:
    """Load benchmark questions from a JSON file.

    Args:
        dataset_path: Path to the JSON benchmark dataset.

    Returns:
        List of BenchmarkQuestion objects.
    """
    with dataset_path.open() as f:
        data: list[dict[str, Any]] = json.load(f)
    return [
        BenchmarkQuestion(
            id=item.get("id", ""),
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            difficulty=item.get("difficulty", ""),
            correct_option=item.get("correct_option", ""),
            spec_references=item.get("spec_references", []),
            category=item.get("category", ""),
        )
        for item in data
    ]


def _perform_health_check() -> None:
    from specagent.llm.factory import check_llm_health  # noqa: PLC0415

    logger.info("Performing LLM endpoint health check...")
    is_healthy, message = check_llm_health(timeout=30)
    if not is_healthy:
        logger.error(
            "LLM endpoint health check failed: %s. "
            "The benchmark cannot proceed with an unavailable endpoint. "
            "To skip this check (not recommended), use --skip-health-check flag.",
            message,
        )
        raise RuntimeError(f"LLM endpoint unavailable: {message}")
    logger.info("LLM endpoint healthy: %s", message)


def _exception_result(question: BenchmarkQuestion, exc: Exception) -> BenchmarkResult:
    return BenchmarkResult(
        question_id=question.id,
        question=question.question,
        expected_answer=question.answer,
        generated_answer="",
        is_correct=False,
        confidence=0.0,
        latency_ms=0.0,
        difficulty=question.difficulty,
        error=str(exc),
    )


def _build_error_result(
    question: BenchmarkQuestion,
    state: dict[str, Any],
    elapsed_ms: float,
    generated_answer: str,
    trace: logging.Logger,
) -> BenchmarkResult:
    error_msg = state.get("error")
    trace.info(f"  → Error: {error_msg}")
    trace.info(f"  → Result: ✗ ERROR (latency: {elapsed_ms:.0f}ms)")
    trace.info("")
    return BenchmarkResult(
        question_id=question.id,
        question=question.question,
        expected_answer=question.answer,
        generated_answer=generated_answer,
        is_correct=False,
        confidence=0.0,
        latency_ms=state.get("processing_time_ms", elapsed_ms),
        difficulty=question.difficulty,
        rewrites=state.get("rewrite_count", 0),
        error=error_msg,
        node_timings=state.get("node_timings", {}),
    )


def _build_rejection_result(
    question: BenchmarkQuestion,
    state: dict[str, Any],
    elapsed_ms: float,
    trace: logging.Logger,
) -> BenchmarkResult:
    trace.info(f"  → Result: ✗ REJECTED by router (latency: {elapsed_ms:.0f}ms)")
    trace.info("")
    return BenchmarkResult(
        question_id=question.id,
        question=question.question,
        expected_answer=question.answer,
        generated_answer="",
        is_correct=False,
        confidence=0.0,
        latency_ms=state.get("processing_time_ms", elapsed_ms),
        difficulty=question.difficulty,
        rewrites=state.get("rewrite_count", 0),
        error="Question was rejected by router",
        node_timings=state.get("node_timings", {}),
    )


def _log_success_details(
    trace: logging.Logger,
    state: dict[str, Any],
    generated_answer: str,
    is_correct: bool,
    confidence: float,
    elapsed_ms: float,
    verbose: bool,
) -> None:
    retrieved = state.get("retrieved_chunks", [])
    graded = state.get("graded_chunks", [])
    trace.info(f"  → Retrieved: {len(retrieved)} chunks")
    if graded:
        relevant = len([c for c in graded if c.relevant == "yes"])
        trace.info(f"  → Grading: {relevant} relevant, {len(graded) - relevant} filtered")
    rewrite_count = state.get("rewrite_count", 0)
    if rewrite_count > 0:
        trace.info(f"  → Rewrites: {rewrite_count}")
    if generated_answer:
        preview = generated_answer[:100] + ("..." if len(generated_answer) > 100 else "")
        trace.info(f"  → Generated: {preview}")
    icon, label = ("✓", "CORRECT") if is_correct else ("✗", "INCORRECT")
    trace.info(
        f"  → Result: {icon} {label} (confidence: {confidence:.2f}, latency: {elapsed_ms:.0f}ms)"
    )
    if verbose:
        node_timings = state.get("node_timings", {})
        if node_timings:
            trace.info("  → Timing breakdown:")
            for name, ms in sorted(node_timings.items()):
                trace.info(f"      {name}: {ms:.0f}ms")
        llm_calls = state.get("llm_calls", [])
        if llm_calls:
            total_llm = sum(c.inference_ms for c in llm_calls)
            trace.info(f"  → LLM inference time: {total_llm:.0f}ms ({len(llm_calls)} calls)")
    trace.info("")


def _build_success_result(
    question: BenchmarkQuestion,
    state: dict[str, Any],
    generated_answer: str,
    elapsed_ms: float,
    trace: logging.Logger,
    verbose: bool,
) -> BenchmarkResult:
    is_correct = check_answer_correctness(generated_answer, question.answer, use_llm_judge=True)
    confidence = state.get("average_confidence", 0.0)
    rewrite_count = state.get("rewrite_count", 0)
    _log_success_details(
        trace, state, generated_answer, is_correct, confidence, elapsed_ms, verbose
    )
    return BenchmarkResult(
        question_id=question.id,
        question=question.question,
        expected_answer=question.answer,
        generated_answer=generated_answer,
        is_correct=is_correct,
        confidence=confidence,
        latency_ms=state.get("processing_time_ms", elapsed_ms),
        difficulty=question.difficulty,
        rewrites=rewrite_count,
        error=None,
        node_timings=state.get("node_timings", {}),
    )


def _run_single_question(
    question: BenchmarkQuestion,
    idx: int,
    total: int,
    trace: logging.Logger,
    verbose: bool,
) -> BenchmarkResult:
    from specagent.graph.workflow import run_query  # noqa: PLC0415

    trace.info(f"[Q{idx}/{total}] {question.question}")
    trace.info(f"  Expected: {question.answer}")
    trace.info(f"  Difficulty: {question.difficulty}")
    try:
        start = time.perf_counter()
        state = run_query(question.question)
        elapsed_ms = (time.perf_counter() - start) * 1000
        route_decision = state.get("route_decision", "unknown")
        route_reasoning = state.get("route_reasoning", "")
        trace.info(f"  → Router: {route_decision}")
        if route_reasoning and verbose:
            trace.info(f"    Reasoning: {route_reasoning}")
        generated_answer = state.get("generation") or ""
        if state.get("error"):
            return _build_error_result(question, state, elapsed_ms, generated_answer, trace)
        if route_decision == "reject":
            return _build_rejection_result(question, state, elapsed_ms, trace)
        return _build_success_result(question, state, generated_answer, elapsed_ms, trace, verbose)
    except Exception as e:
        trace.info(f"  → Exception: {e!s}")
        trace.info("  → Result: ✗ EXCEPTION")
        trace.info("")
        return _exception_result(question, e)


def _save_report(
    report: BenchmarkReport,
    output_path: Path,
    timestamp: str,
    trace: logging.Logger,
) -> None:
    ts = timestamp.replace(":", "-").split(".")[0]
    json_path = output_path / f"benchmark_{ts}.json"
    md_path = output_path / f"benchmark_{ts}.md"
    with json_path.open("w") as f:
        json.dump(report.to_dict(), f, indent=2)
    with md_path.open("w") as f:
        f.write(report.to_markdown())
    trace.info("Output Files:")
    trace.info(f"  JSON: {json_path}")
    trace.info(f"  Markdown: {md_path}")
    trace.info(f"  Trace: {output_path / f'benchmark_trace_{ts}.log'}")
    trace.info("")
    trace.info("=" * 80)
    trace.info("Benchmark complete!")
    trace.info("=" * 80)


def run_benchmark(
    questions: list[BenchmarkQuestion],
    limit: int | None = None,
    output_dir: str | Path = "evaluation/results",
    skip_health_check: bool = False,
    verbose: bool = False,
) -> BenchmarkReport:
    """Run TSpec-LLM benchmark evaluation and save JSON + Markdown reports."""
    if not skip_health_check:
        _perform_health_check()

    if limit is not None:
        questions = questions[:limit]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    trace = setup_trace_logging(output_path, timestamp, verbose)

    trace.info("=" * 80)
    trace.info(f"SpecAgent Benchmark Trace - {timestamp}")
    trace.info("=" * 80)
    trace.info(f"Total questions: {len(questions)}")
    trace.info(f"Output directory: {output_path}")
    trace.info("")

    results = [
        _run_single_question(q, idx, len(questions), trace, verbose)
        for idx, q in enumerate(questions, 1)
    ]

    metrics = compute_report_metrics(results)
    log_benchmark_summary(trace, metrics)
    report = BenchmarkReport(timestamp=timestamp, results=results, **metrics)
    _save_report(report, output_path, timestamp, trace)

    for handler in trace.handlers[:]:
        handler.close()
        trace.removeHandler(handler)

    return report
