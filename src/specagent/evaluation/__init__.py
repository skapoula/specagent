"""
Evaluation framework for SpecAgent.

Components:
    - metrics: RAGAS metrics for RAG evaluation
    - benchmark: TSpec-LLM benchmark runner
"""

from specagent.evaluation.benchmark import run_benchmark
from specagent.evaluation.metrics import (
    evaluate_e2e,
    evaluate_retrieval,
)

__all__ = [
    "evaluate_e2e",
    "evaluate_retrieval",
    "run_benchmark",
]
