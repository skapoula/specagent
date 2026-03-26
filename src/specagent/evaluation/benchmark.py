"""
TSpec-LLM benchmark runner.

Runs the SpecAgent pipeline against the TSpec-LLM benchmark
questions and computes accuracy by difficulty level.
"""

import json
import logging
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuestion:
    """A question from the TSpec-LLM benchmark."""

    id: str
    question: str
    answer: str  # Ground truth (parsed from "option_X: text")
    difficulty: str  # Easy, Intermediate, Hard
    correct_option: str = ""  # e.g., "option_2"
    spec_references: list[str] = field(default_factory=list)
    category: str = ""


@dataclass
class BenchmarkResult:
    """Result for a single benchmark question."""

    question_id: str
    question: str
    expected_answer: str
    generated_answer: str
    is_correct: bool
    confidence: float
    latency_ms: float
    difficulty: str
    rewrites: int = 0
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""

    timestamp: str
    total_questions: int
    correct_answers: int
    accuracy: float
    accuracy_by_difficulty: dict[str, float]
    average_latency_ms: float
    average_confidence: float
    results: list[BenchmarkResult]
    confidence_distribution: dict[str, int] = field(default_factory=dict)
    confidence_stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "accuracy": self.accuracy,
            "accuracy_by_difficulty": self.accuracy_by_difficulty,
            "average_latency_ms": self.average_latency_ms,
            "average_confidence": self.average_confidence,
            "confidence_distribution": self.confidence_distribution,
            "confidence_stats": self.confidence_stats,
            "results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "generated_answer": r.generated_answer,
                    "is_correct": r.is_correct,
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms,
                    "difficulty": r.difficulty,
                    "rewrites": r.rewrites,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# SpecAgent Benchmark Report",
            "",
            f"**Date:** {self.timestamp}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Questions | {self.total_questions} |",
            f"| Correct Answers | {self.correct_answers} |",
            f"| **Accuracy** | **{self.accuracy:.1%}** |",
            f"| Average Latency | {self.average_latency_ms:.0f}ms |",
            f"| Average Confidence | {self.average_confidence:.2f} |",
            "",
            "## Accuracy by Difficulty",
            "",
            "| Difficulty | Accuracy |",
            "|------------|----------|",
        ]

        for difficulty, acc in sorted(self.accuracy_by_difficulty.items()):
            lines.append(f"| {difficulty.capitalize()} | {acc:.1%} |")

        # Add confidence analysis section
        if self.confidence_distribution:
            lines.extend(
                [
                    "",
                    "## Confidence Analysis",
                    "",
                    "### Confidence Statistics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                ]
            )

            for metric, value in sorted(self.confidence_stats.items()):
                if metric == "std":
                    lines.append(f"| Standard Deviation | {value:.3f} |")
                else:
                    lines.append(f"| {metric.capitalize()} | {value:.3f} |")

            lines.extend(
                [
                    "",
                    "### Confidence Distribution",
                    "",
                    "Frequency of confidence scores assigned to generated answers:",
                    "",
                    "| Confidence Range | Count | Percentage |",
                    "|------------------|-------|------------|",
                ]
            )

            for range_label, count in self.confidence_distribution.items():
                percentage = (count / self.total_questions * 100) if self.total_questions > 0 else 0
                lines.append(f"| {range_label} | {count} | {percentage:.1f}% |")

        lines.extend(
            [
                "",
                "## Failed Questions",
                "",
            ]
        )

        failed = [r for r in self.results if not r.is_correct]
        if failed:
            for r in failed[:10]:  # Limit to first 10
                lines.extend(
                    [
                        f"### {r.question_id}",
                        "",
                        f"**Question:** {r.question}",
                        "",
                        f"**Expected:** {r.expected_answer}",
                        "",
                        f"**Generated:** {r.generated_answer}",
                        "",
                        f"**Confidence:** {r.confidence:.2f}",
                        "",
                    ]
                )
        else:
            lines.append("No failed questions! 🎉")

        return "\n".join(lines)


def load_benchmark_questions(path: str | Path) -> list[BenchmarkQuestion]:
    """
    Load benchmark questions from TSpec-LLM format JSON file.

    Expected format:
    {
        "question_1": {
            "question": "...",
            "option_1": "...",
            "option_2": "...",
            "option_3": "...",
            "option_4": "...",
            "answer": "option_2: text",
            "explanation": "...",
            "category": "3GPP TR ...",
            "difficulty": "Easy|Intermediate|Hard"
        },
        ...
    }

    Args:
        path: Path to benchmark JSON file

    Returns:
        List of BenchmarkQuestion objects
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    questions = []
    for question_id, q_data in data.items():
        # Parse answer format: "option_2: 16" -> answer="16", correct_option="option_2"
        answer_text = q_data["answer"]
        if ":" in answer_text:
            correct_option, answer = answer_text.split(":", 1)
            correct_option = correct_option.strip()
            answer = answer.strip()
        else:
            correct_option = ""
            answer = answer_text

        questions.append(
            BenchmarkQuestion(
                id=question_id,
                question=q_data["question"],
                answer=answer,
                difficulty=q_data.get("difficulty", "Intermediate"),
                correct_option=correct_option,
                spec_references=[],
                category=q_data.get("category", ""),
            )
        )

    return questions


def compute_confidence_distribution(results: list[BenchmarkResult]) -> dict[str, int]:
    """
    Compute confidence distribution histogram.

    Groups confidence scores into 5 bins: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]

    Args:
        results: List of benchmark results

    Returns:
        Dictionary mapping bin labels to counts
    """
    bins = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0,
    }

    for result in results:
        confidence = result.confidence
        if confidence < 0.2:
            bins["0.0-0.2"] += 1
        elif confidence < 0.4:
            bins["0.2-0.4"] += 1
        elif confidence < 0.6:
            bins["0.4-0.6"] += 1
        elif confidence < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1

    return bins


def compute_confidence_stats(results: list[BenchmarkResult]) -> dict[str, float]:
    """
    Compute confidence statistics.

    Args:
        results: List of benchmark results

    Returns:
        Dictionary with mean, median, min, max, std
    """
    if not results:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }

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


def setup_trace_logging(output_dir: Path, timestamp: str, verbose: bool = False) -> logging.Logger:
    """
    Setup trace logging for benchmark execution.

    Creates a dedicated trace logger that writes to a file and optionally to console.

    Args:
        output_dir: Directory to save trace log
        timestamp: Timestamp for log filename
        verbose: If True, also output to console

    Returns:
        Configured trace logger
    """
    trace_logger = logging.getLogger("benchmark_trace")
    trace_logger.setLevel(logging.INFO)
    trace_logger.propagate = False  # Don't propagate to root logger

    # Remove existing handlers
    trace_logger.handlers.clear()

    # File handler - always write to file
    log_filename = f"benchmark_trace_{timestamp.replace(':', '-').split('.')[0]}.log"
    log_path = output_dir / log_filename
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(file_formatter)
    trace_logger.addHandler(file_handler)

    # Console handler - only if verbose
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        trace_logger.addHandler(console_handler)

    return trace_logger


def run_benchmark(
    questions: list[BenchmarkQuestion],
    limit: int | None = None,
    output_dir: str | Path = "evaluation/results",
    skip_health_check: bool = False,
    verbose: bool = False,
) -> BenchmarkReport:
    """
    Run benchmark evaluation.

    Args:
        questions: List of benchmark questions
        limit: Maximum number of questions to run (for testing)
        output_dir: Directory to save results
        skip_health_check: Skip LLM endpoint health check (default: False)
        verbose: If True, output detailed trace to console (default: False)

    Returns:
        BenchmarkReport with all results

    Raises:
        RuntimeError: If LLM endpoint health check fails
    """
    from specagent.graph.workflow import run_query
    from specagent.llm.custom_endpoint import check_llm_endpoint_health

    # Perform health check before starting benchmark
    if not skip_health_check:
        print("\nPerforming LLM endpoint health check...")
        is_healthy, message = check_llm_endpoint_health(timeout=30)

        if not is_healthy:
            error_msg = (
                f"\n✗ LLM endpoint health check failed: {message}\n"
                f"  The benchmark cannot proceed with an unavailable endpoint.\n"
                f"  Please check the endpoint status and try again.\n"
                f"  To skip this check (not recommended), use --skip-health-check flag."
            )
            print(error_msg)
            raise RuntimeError(f"LLM endpoint unavailable: {message}")

        print(f"✓ {message}\n")

    # Apply limit if specified
    if limit is not None:
        questions = questions[:limit]

    # Setup trace logging
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    trace = setup_trace_logging(output_path, timestamp, verbose)

    # Log benchmark header
    trace.info("=" * 80)
    trace.info(f"SpecAgent Benchmark Trace - {timestamp}")
    trace.info("=" * 80)
    trace.info(f"Total questions: {len(questions)}")
    trace.info(f"Output directory: {output_path}")
    trace.info("")

    # Initialize results
    results: list[BenchmarkResult] = []

    # Run each question through the pipeline
    for idx, question in enumerate(questions, 1):
        trace.info(f"[Q{idx}/{len(questions)}] {question.question}")
        trace.info(f"  Expected: {question.answer}")
        trace.info(f"  Difficulty: {question.difficulty}")

        try:
            # Execute pipeline
            import time

            start_time = time.time()
            state = run_query(question.question)
            elapsed_ms = (time.time() - start_time) * 1000

            # Log router decision
            route_decision = state.get("route_decision", "unknown")
            route_reasoning = state.get("route_reasoning", "")
            trace.info(f"  → Router: {route_decision}")
            if route_reasoning and verbose:
                trace.info(f"    Reasoning: {route_reasoning}")

            # Extract generated answer
            generated_answer = state.get("generation", "")

            # Handle errors or rejections
            if state.get("error"):
                error_msg = state.get("error")
                trace.info(f"  → Error: {error_msg}")
                trace.info(f"  → Result: ✗ ERROR (latency: {elapsed_ms:.0f}ms)")
                trace.info("")
                results.append(
                    BenchmarkResult(
                        question_id=question.id,
                        question=question.question,
                        expected_answer=question.answer,
                        generated_answer=generated_answer or "",
                        is_correct=False,
                        confidence=0.0,
                        latency_ms=state.get("processing_time_ms", elapsed_ms),
                        difficulty=question.difficulty,
                        rewrites=state.get("rewrite_count", 0),
                        error=error_msg,
                    )
                )
                continue

            if state.get("route_decision") == "reject":
                trace.info(f"  → Result: ✗ REJECTED by router (latency: {elapsed_ms:.0f}ms)")
                trace.info("")
                results.append(
                    BenchmarkResult(
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
                    )
                )
                continue

            # Log retrieval info
            retrieved_chunks = state.get("retrieved_chunks", [])
            graded_chunks = state.get("graded_chunks", [])
            trace.info(f"  → Retrieved: {len(retrieved_chunks)} chunks")
            if graded_chunks:
                relevant_count = len([c for c in graded_chunks if c.relevant == "yes"])
                trace.info(
                    f"  → Grading: {relevant_count} relevant, {len(graded_chunks) - relevant_count} filtered"
                )

            # Log rewrites
            rewrite_count = state.get("rewrite_count", 0)
            if rewrite_count > 0:
                trace.info(f"  → Rewrites: {rewrite_count}")

            # Log generation
            if generated_answer:
                answer_preview = (
                    generated_answer[:100] + "..."
                    if len(generated_answer) > 100
                    else generated_answer
                )
                trace.info(f"  → Generated: {answer_preview}")

            # Check correctness
            is_correct = check_answer_correctness(
                generated_answer,
                question.answer,
                use_llm_judge=True,
            )

            # Log result
            confidence = state.get("average_confidence", 0.0)
            result_icon = "✓" if is_correct else "✗"
            result_text = "CORRECT" if is_correct else "INCORRECT"
            trace.info(
                f"  → Result: {result_icon} {result_text} (confidence: {confidence:.2f}, latency: {elapsed_ms:.0f}ms)"
            )

            # Log timing breakdown if available
            node_timings = state.get("node_timings", {})
            if node_timings and verbose:
                trace.info("  → Timing breakdown:")
                for node_name, node_time in sorted(node_timings.items()):
                    trace.info(f"      {node_name}: {node_time:.0f}ms")

            # Log LLM inference times if available
            llm_times = state.get("llm_inference_times", [])
            if llm_times and verbose:
                total_llm_time = sum(t.get("inference_ms", 0) for t in llm_times)
                trace.info(
                    f"  → LLM inference time: {total_llm_time:.0f}ms ({len(llm_times)} calls)"
                )

            trace.info("")

            results.append(
                BenchmarkResult(
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
                )
            )

        except Exception as e:
            # Handle unexpected errors
            trace.info(f"  → Exception: {e!s}")
            trace.info("  → Result: ✗ EXCEPTION")
            trace.info("")
            results.append(
                BenchmarkResult(
                    question_id=question.id,
                    question=question.question,
                    expected_answer=question.answer,
                    generated_answer="",
                    is_correct=False,
                    confidence=0.0,
                    latency_ms=0.0,
                    difficulty=question.difficulty,
                    rewrites=0,
                    error=str(e),
                )
            )

    # Compute metrics
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r.is_correct)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0

    # Accuracy by difficulty
    accuracy_by_difficulty: dict[str, float] = {}
    for difficulty in ["Easy", "Intermediate", "Hard"]:
        difficulty_results = [r for r in results if r.difficulty == difficulty]
        if difficulty_results:
            correct = sum(1 for r in difficulty_results if r.is_correct)
            accuracy_by_difficulty[difficulty] = correct / len(difficulty_results)

    # Average metrics
    average_latency_ms = (
        sum(r.latency_ms for r in results) / total_questions if total_questions > 0 else 0.0
    )
    average_confidence = (
        sum(r.confidence for r in results) / total_questions if total_questions > 0 else 0.0
    )

    # Confidence analysis
    confidence_distribution = compute_confidence_distribution(results)
    confidence_stats = compute_confidence_stats(results)

    # Log summary
    trace.info("=" * 80)
    trace.info("BENCHMARK SUMMARY")
    trace.info("=" * 80)
    trace.info(f"Total Questions: {total_questions}")
    trace.info(f"Correct Answers: {correct_answers}")
    trace.info(f"Accuracy: {accuracy:.1%}")
    trace.info(f"Average Latency: {average_latency_ms:.0f}ms")
    trace.info(f"Average Confidence: {average_confidence:.2f}")
    trace.info("")

    if accuracy_by_difficulty:
        trace.info("Accuracy by Difficulty:")
        for difficulty in ["Easy", "Intermediate", "Hard"]:
            if difficulty in accuracy_by_difficulty:
                diff_acc = accuracy_by_difficulty[difficulty]
                trace.info(f"  {difficulty}: {diff_acc:.1%}")
        trace.info("")

    # Compute aggregate node timing statistics
    all_node_timings: dict[str, list[float]] = {}
    for result in results:
        # Note: We don't have direct access to node_timings from results
        # This would require storing node_timings in BenchmarkResult
        # For now, skip aggregate timing in summary
        pass

    trace.info("")

    # Create report
    report = BenchmarkReport(
        timestamp=timestamp,
        total_questions=total_questions,
        correct_answers=correct_answers,
        accuracy=accuracy,
        accuracy_by_difficulty=accuracy_by_difficulty,
        average_latency_ms=average_latency_ms,
        average_confidence=average_confidence,
        results=results,
        confidence_distribution=confidence_distribution,
        confidence_stats=confidence_stats,
    )

    # Save results (output_path already created earlier)
    # output_path = Path(output_dir)  # Already created above
    # output_path.mkdir(parents=True, exist_ok=True)  # Already done

    # Save JSON
    json_filename = f"benchmark_{timestamp.replace(':', '-').split('.')[0]}.json"
    json_path = output_path / json_filename
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Save Markdown
    md_filename = f"benchmark_{timestamp.replace(':', '-').split('.')[0]}.md"
    md_path = output_path / md_filename
    with open(md_path, "w") as f:
        f.write(report.to_markdown())

    # Log output file paths
    trace.info("Output Files:")
    trace.info(f"  JSON: {json_path}")
    trace.info(f"  Markdown: {md_path}")
    trace_log_path = (
        output_path / f"benchmark_trace_{timestamp.replace(':', '-').split('.')[0]}.log"
    )
    trace.info(f"  Trace: {trace_log_path}")
    trace.info("")
    trace.info("=" * 80)
    trace.info("Benchmark complete!")
    trace.info("=" * 80)

    # Close trace logger handlers
    for handler in trace.handlers[:]:
        handler.close()
        trace.removeHandler(handler)

    return report


def check_answer_correctness(
    generated: str,
    expected: str,
    use_llm_judge: bool = True,
) -> bool:
    """
    Check if generated answer matches expected answer.

    Uses fuzzy matching since exact string match is too strict.

    Args:
        generated: Generated answer text
        expected: Expected/ground truth answer
        use_llm_judge: Whether to use LLM for semantic comparison

    Returns:
        True if answers match, False otherwise
    """
    # Normalize both strings
    generated_norm = generated.strip().lower()
    expected_norm = expected.strip().lower()

    # 1. Try exact match
    if generated_norm == expected_norm:
        return True

    # 2. Try fuzzy string matching - check if expected is contained in generated
    # This handles cases like "The answer is 16" containing "16"
    if expected_norm in generated_norm:
        return True

    # 3. Try word-based matching for multi-word answers
    expected_words = set(expected_norm.split())
    generated_words = set(generated_norm.split())

    # If all expected words are in generated, consider it a match
    if expected_words and expected_words.issubset(generated_words):
        return True

    # 4. Use LLM judge for semantic comparison if requested
    if use_llm_judge:
        return llm_judge_answer(generated, expected)

    return False


def llm_judge_answer(generated: str, expected: str) -> bool:
    """
    Use LLM to judge if generated answer is semantically equivalent to expected.

    Args:
        generated: Generated answer
        expected: Expected answer

    Returns:
        True if answers are semantically equivalent
    """
    from specagent.llm.factory import get_llm

    prompt = f"""You are evaluating answers to technical questions about 3GPP specifications.

Question: Does the generated answer convey the same information as the expected answer?

Expected Answer: {expected}
Generated Answer: {generated}

Respond with ONLY "yes" or "no".

If the generated answer contains the expected answer or conveys the same key information (even with additional context), respond "yes".
If the generated answer is incorrect, contradicts the expected answer, or is missing the key information, respond "no".

Response:"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)

        # Extract text from response
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)

        # Check if response contains "yes"
        return "yes" in response_text.lower().strip()

    except Exception:
        # If LLM judge fails, fall back to fuzzy matching
        return expected.lower() in generated.lower()
