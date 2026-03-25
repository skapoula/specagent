"""
Evaluation metrics using RAGAS and custom implementations.

Metrics:
    - Faithfulness: Is the answer grounded in retrieved context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved chunks relevant?
    - Context Recall: Did we retrieve all needed information?
    - Retrieval metrics: Recall@k, MRR
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Type variable for generic retry decorator
T = TypeVar("T")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if the exception looks like a rate limit error."""
    msg = str(exc).lower()
    return any(p in msg for p in ["rate limit", "429", "too many requests", "ratelimit"])


def _retry_on_rate_limit(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to retry function calls on API rate limit errors.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s).
    Catches common rate limit exceptions from LLM inference APIs.
    """

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit_error(e):
                logger.warning("Rate limit hit, retrying: %s", e)
                time.sleep(1)  # Additional small delay
            raise

    return wrapper


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float | None = None  # Requires ground truth

    @property
    def overall_score(self) -> float:
        """Weighted average of all metrics."""
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.25,
            "context_precision": 0.2,
            "context_recall": 0.25,
        }
        return (
            self.faithfulness * weights["faithfulness"]
            + self.answer_relevancy * weights["answer_relevancy"]
            + self.context_precision * weights["context_precision"]
            + self.context_recall * weights["context_recall"]
        )


@dataclass
class RetrievalMetrics:
    """Results from retrieval evaluation."""

    recall_at_5: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank

    @property
    def summary(self) -> dict[str, float]:
        """Return metrics as dictionary."""
        return {
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
        }


def evaluate_e2e(
    test_dataset: list[dict[str, Any]],
    include_correctness: bool = False,  # noqa: ARG001
) -> EvaluationResult:
    """
    Run end-to-end RAGAS evaluation.

    Args:
        test_dataset: List of dicts with keys:
            - question: str
            - answer: str (generated)
            - contexts: list[str] (retrieved chunks)
            - ground_truth: str (optional, for correctness)
        include_correctness: Whether to compute answer correctness (reserved for future use)

    Returns:
        EvaluationResult with all metric scores

    Raises:
        ValueError: If dataset is empty or missing required fields
        ImportError: If ragas package is not installed
    """
    try:
        from datasets import Dataset  # type: ignore[import-untyped]  # noqa: PLC0415
        from ragas import evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        raise ImportError(
            "RAGAS dependencies not found. Install with: pip install 'specagent[eval]'"
        ) from e

    if not test_dataset:
        raise ValueError("test_dataset cannot be empty")

    # Validate required fields
    required_fields = {"question", "answer", "contexts"}
    for i, sample in enumerate(test_dataset):
        missing = required_fields - set(sample.keys())
        if missing:
            raise ValueError(f"Sample {i} missing required fields: {missing}")

    # Format dataset for RAGAS (convert to Dataset format)
    # RAGAS expects specific column names
    ragas_data = {
        "question": [s["question"] for s in test_dataset],
        "answer": [s["answer"] for s in test_dataset],
        "contexts": [s["contexts"] for s in test_dataset],
    }

    # Add ground_truth if available for context_recall
    has_ground_truth = all("ground_truth" in s for s in test_dataset)
    if has_ground_truth:
        ragas_data["ground_truth"] = [s["ground_truth"] for s in test_dataset]

    dataset = Dataset.from_dict(ragas_data)

    # Select metrics based on available data
    metrics = [faithfulness, answer_relevancy, context_precision]

    # Only include context_recall if ground_truth is available
    if has_ground_truth:
        metrics.append(context_recall)

    # Run RAGAS evaluation with retry logic
    @_retry_on_rate_limit
    def _run_evaluation() -> Any:
        return evaluate(dataset, metrics=metrics)

    results = _run_evaluation()

    # Extract metric scores (RAGAS returns dict with metric names as keys)
    return EvaluationResult(
        faithfulness=float(results.get("faithfulness", 0.0)),
        answer_relevancy=float(results.get("answer_relevancy", 0.0)),
        context_precision=float(results.get("context_precision", 0.0)),
        context_recall=float(results.get("context_recall", 0.0)) if has_ground_truth else 0.0,
        answer_correctness=None,  # Not implemented in current RAGAS metrics
    )


def evaluate_retrieval(
    queries: list[str],
    retrieved_docs: list[list[str]],
    ground_truth_docs: list[list[str]],
    k_values: list[int] | None = None,
) -> RetrievalMetrics:
    """
    Evaluate retrieval quality.

    Args:
        queries: List of query strings
        retrieved_docs: List of retrieved doc IDs per query
        ground_truth_docs: List of relevant doc IDs per query
        k_values: Values of k for Recall@k (default: [5, 10])

    Returns:
        RetrievalMetrics with Recall@k and MRR

    Raises:
        ValueError: If input lists have different lengths
    """
    if k_values is None:
        k_values = [5, 10]

    # Validate inputs
    if len(retrieved_docs) != len(ground_truth_docs):
        raise ValueError(
            f"Mismatch: {len(retrieved_docs)} retrieved vs "
            f"{len(ground_truth_docs)} ground_truth"
        )
    if len(queries) != len(retrieved_docs):
        raise ValueError(
            f"Mismatch: {len(queries)} queries vs " f"{len(retrieved_docs)} retrieved_docs"
        )

    if not queries:
        raise ValueError("queries cannot be empty")

    # Compute Recall@k for each k value
    recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

    for retrieved, relevant in zip(retrieved_docs, ground_truth_docs, strict=True):
        for k in k_values:
            recall_k = calculate_recall_at_k(retrieved, relevant, k)
            recall_scores[k].append(recall_k)

    # Compute MRR across all queries
    mrr_scores = [
        calculate_mrr(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_docs, ground_truth_docs, strict=True)
    ]

    # Return aggregated metrics (mean across all queries)
    return RetrievalMetrics(
        recall_at_5=sum(recall_scores[5]) / len(recall_scores[5]) if 5 in k_values else 0.0,
        recall_at_10=sum(recall_scores[10]) / len(recall_scores[10])
        if 10 in k_values
        else 0.0,
        mrr=sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
    )


def calculate_recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """
    Calculate Recall@k for a single query.

    Args:
        retrieved: Ordered list of retrieved doc IDs
        relevant: Set of relevant doc IDs
        k: Number of top results to consider

    Returns:
        Recall score (0.0 to 1.0)
    """
    if not relevant:
        return 0.0

    top_k = set(retrieved[:k])
    relevant_set = set(relevant)

    return len(top_k & relevant_set) / len(relevant_set)


def calculate_mrr(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank for a single query.

    Args:
        retrieved: Ordered list of retrieved doc IDs
        relevant: Set of relevant doc IDs

    Returns:
        Reciprocal rank (0.0 to 1.0)
    """
    relevant_set = set(relevant)

    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / i

    return 0.0
