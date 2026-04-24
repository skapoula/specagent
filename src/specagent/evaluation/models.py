"""Benchmark data models and dataset loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    node_timings: dict[str, float] = field(default_factory=dict)


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
    average_node_timings: dict[str, float] = field(default_factory=dict)

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
            "average_node_timings": self.average_node_timings,
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

    def _header_and_summary_lines(self) -> list[str]:
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
        return lines

    def _confidence_section_lines(self) -> list[str]:
        if not self.confidence_distribution:
            return []
        lines = [
            "",
            "## Confidence Analysis",
            "",
            "### Confidence Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for metric, value in sorted(self.confidence_stats.items()):
            label = "Standard Deviation" if metric == "std" else metric.capitalize()
            lines.append(f"| {label} | {value:.3f} |")
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
            pct = (count / self.total_questions * 100) if self.total_questions > 0 else 0
            lines.append(f"| {range_label} | {count} | {pct:.1f}% |")
        return lines

    def _timing_section_lines(self) -> list[str]:
        if not self.average_node_timings:
            return []
        lines = [
            "",
            "## Node Timing Breakdown",
            "",
            "Average time spent per node across all questions:",
            "",
            "| Node | Average (ms) |",
            "|------|-------------|",
        ]
        for node, avg_ms in sorted(self.average_node_timings.items()):
            lines.append(f"| {node} | {avg_ms:.0f} |")
        return lines

    def _failures_section_lines(self) -> list[str]:
        lines = ["", "## Failed Questions", ""]
        failed = [r for r in self.results if not r.is_correct]
        if not failed:
            lines.append("No failed questions! 🎉")
            return lines
        for r in failed[:10]:
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
        return lines

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = (
            self._header_and_summary_lines()
            + self._confidence_section_lines()
            + self._timing_section_lines()
            + self._failures_section_lines()
        )
        return "\n".join(lines)


def _parse_question(question_id: str, q_data: dict) -> BenchmarkQuestion:
    """Parse one raw JSON entry into a BenchmarkQuestion."""
    answer_text = q_data["answer"]
    if ":" in answer_text:
        correct_option, answer = answer_text.split(":", 1)
        correct_option, answer = correct_option.strip(), answer.strip()
    else:
        correct_option, answer = "", answer_text
    return BenchmarkQuestion(
        id=question_id,
        question=q_data["question"],
        answer=answer,
        difficulty=q_data.get("difficulty", "Intermediate"),
        correct_option=correct_option,
        spec_references=[],
        category=q_data.get("category", ""),
    )


def load_benchmark_questions(path: str | Path) -> list[BenchmarkQuestion]:
    """Load benchmark questions from a TSpec-LLM format JSON file.

    Args:
        path: Path to benchmark JSON file

    Returns:
        List of BenchmarkQuestion objects
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)
    return [_parse_question(qid, q_data) for qid, q_data in data.items()]
