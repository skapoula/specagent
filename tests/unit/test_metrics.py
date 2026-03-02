"""Tests for evaluation metrics."""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestEvaluationResult:
    def test_overall_score_all_ones(self):
        from specagent.evaluation.metrics import EvaluationResult

        r = EvaluationResult(
            faithfulness=1.0,
            answer_relevancy=1.0,
            context_precision=1.0,
            context_recall=1.0,
        )
        assert r.overall_score == pytest.approx(1.0)

    def test_overall_score_all_zeros(self):
        from specagent.evaluation.metrics import EvaluationResult

        r = EvaluationResult(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
        )
        assert r.overall_score == pytest.approx(0.0)

    def test_overall_score_weighted(self):
        from specagent.evaluation.metrics import EvaluationResult

        # weights: faithfulness=0.3, relevancy=0.25, precision=0.2, recall=0.25
        r = EvaluationResult(
            faithfulness=1.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
        )
        assert r.overall_score == pytest.approx(0.3)

    def test_answer_correctness_defaults_to_none(self):
        from specagent.evaluation.metrics import EvaluationResult

        r = EvaluationResult(
            faithfulness=0.5,
            answer_relevancy=0.5,
            context_precision=0.5,
            context_recall=0.5,
        )
        assert r.answer_correctness is None

    def test_answer_correctness_can_be_set(self):
        from specagent.evaluation.metrics import EvaluationResult

        r = EvaluationResult(
            faithfulness=0.5,
            answer_relevancy=0.5,
            context_precision=0.5,
            context_recall=0.5,
            answer_correctness=0.9,
        )
        assert r.answer_correctness == pytest.approx(0.9)


@pytest.mark.unit
class TestRetrievalMetrics:
    def test_summary_keys(self):
        from specagent.evaluation.metrics import RetrievalMetrics

        m = RetrievalMetrics(recall_at_5=0.8, recall_at_10=0.9, mrr=0.75)
        assert set(m.summary.keys()) == {"recall@5", "recall@10", "mrr"}

    def test_summary_values(self):
        from specagent.evaluation.metrics import RetrievalMetrics

        m = RetrievalMetrics(recall_at_5=0.8, recall_at_10=0.9, mrr=0.75)
        assert m.summary["recall@5"] == pytest.approx(0.8)
        assert m.summary["recall@10"] == pytest.approx(0.9)
        assert m.summary["mrr"] == pytest.approx(0.75)


@pytest.mark.unit
class TestCalculateRecallAtK:
    def test_all_relevant(self):
        from specagent.evaluation.metrics import calculate_recall_at_k

        assert calculate_recall_at_k(["a", "b"], ["a", "b"], k=5) == 1.0

    def test_none_in_window(self):
        from specagent.evaluation.metrics import calculate_recall_at_k

        assert calculate_recall_at_k(["x", "y"], ["a"], k=2) == 0.0

    def test_empty_relevant(self):
        from specagent.evaluation.metrics import calculate_recall_at_k

        assert calculate_recall_at_k(["a"], [], k=5) == 0.0

    def test_k_limits_window(self):
        from specagent.evaluation.metrics import calculate_recall_at_k

        # "c" is at index 2, but k=2 only checks first 2 (indices 0,1)
        assert calculate_recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0

    def test_partial_recall(self):
        from specagent.evaluation.metrics import calculate_recall_at_k

        # retrieved: ["a","b","c"], relevant: ["a","b","d"], k=3
        # top_k = {"a","b","c"}, relevant = {"a","b","d"} => intersection={"a","b"}
        # recall = 2/3
        result = calculate_recall_at_k(["a", "b", "c"], ["a", "b", "d"], k=3)
        assert result == pytest.approx(2 / 3)


@pytest.mark.unit
class TestCalculateMRR:
    def test_first_hit(self):
        from specagent.evaluation.metrics import calculate_mrr

        assert calculate_mrr(["a", "b"], ["a"]) == 1.0

    def test_second_hit(self):
        from specagent.evaluation.metrics import calculate_mrr

        assert calculate_mrr(["x", "a"], ["a"]) == pytest.approx(0.5)

    def test_no_hit(self):
        from specagent.evaluation.metrics import calculate_mrr

        assert calculate_mrr(["x", "y"], ["a"]) == 0.0

    def test_empty_retrieved(self):
        from specagent.evaluation.metrics import calculate_mrr

        assert calculate_mrr([], ["a"]) == 0.0


@pytest.mark.unit
class TestEvaluateRetrieval:
    def test_basic(self):
        from specagent.evaluation.metrics import evaluate_retrieval

        m = evaluate_retrieval(["q"], [["a"]], [["a"]])
        assert m.recall_at_5 == 1.0

    def test_retrieved_gt_mismatch(self):
        from specagent.evaluation.metrics import evaluate_retrieval

        with pytest.raises(ValueError, match="Mismatch"):
            evaluate_retrieval(["q"], [["a"], ["b"]], [["a"]])

    def test_queries_retrieved_mismatch(self):
        from specagent.evaluation.metrics import evaluate_retrieval

        with pytest.raises(ValueError, match="Mismatch"):
            evaluate_retrieval(["q1", "q2"], [["a"]], [["a"]])

    def test_empty_raises(self):
        from specagent.evaluation.metrics import evaluate_retrieval

        with pytest.raises(ValueError, match="empty"):
            evaluate_retrieval([], [], [])

    def test_custom_k(self):
        from specagent.evaluation.metrics import evaluate_retrieval

        m = evaluate_retrieval(["q"], [["a"]], [["a"]], k_values=[5, 10])
        assert m.recall_at_5 == 1.0
        assert m.recall_at_10 == 1.0

    def test_k_not_including_5(self):
        from specagent.evaluation.metrics import evaluate_retrieval

        # When k_values does not include 5, recall_at_5 should be 0.0
        m = evaluate_retrieval(["q"], [["a"]], [["a"]], k_values=[10])
        assert m.recall_at_5 == 0.0
        assert m.recall_at_10 == 1.0

    def test_k_not_including_10(self):
        from specagent.evaluation.metrics import evaluate_retrieval

        # When k_values does not include 10, recall_at_10 should be 0.0
        m = evaluate_retrieval(["q"], [["a"]], [["a"]], k_values=[5])
        assert m.recall_at_5 == 1.0
        assert m.recall_at_10 == 0.0

    def test_mrr_multiple_queries(self):
        from specagent.evaluation.metrics import evaluate_retrieval

        # query1: "a" is at rank 1 => mrr=1.0
        # query2: "b" is at rank 2 => mrr=0.5
        # mean mrr = 0.75
        m = evaluate_retrieval(
            ["q1", "q2"],
            [["a", "x"], ["x", "b"]],
            [["a"], ["b"]],
        )
        assert m.mrr == pytest.approx(0.75)


@pytest.mark.unit
class TestEvaluateE2E:
    def test_import_error(self):
        from specagent.evaluation.metrics import evaluate_e2e

        with patch.dict(
            sys.modules,
            {"ragas": None, "datasets": None, "ragas.metrics": None},
        ):
            with pytest.raises(ImportError, match="RAGAS"):
                evaluate_e2e(
                    [{"question": "q", "answer": "a", "contexts": ["c"]}]
                )

    def test_empty_dataset_raises(self):
        from specagent.evaluation.metrics import evaluate_e2e

        mock_ragas = MagicMock()
        mock_datasets = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "ragas": mock_ragas,
                "datasets": mock_datasets,
                "ragas.metrics": MagicMock(),
            },
        ):
            with pytest.raises(ValueError, match="empty"):
                evaluate_e2e([])

    def test_missing_fields_raises(self):
        from specagent.evaluation.metrics import evaluate_e2e

        mock_ragas = MagicMock()
        mock_datasets = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "ragas": mock_ragas,
                "datasets": mock_datasets,
                "ragas.metrics": MagicMock(),
            },
        ):
            with pytest.raises(ValueError, match="missing required fields"):
                evaluate_e2e([{"question": "q"}])

    def test_success_without_ground_truth(self):
        from specagent.evaluation.metrics import evaluate_e2e

        mock_ragas = MagicMock()
        mock_datasets = MagicMock()
        mock_ragas_metrics = MagicMock()

        fake_results = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        mock_ragas.evaluate.return_value = fake_results
        mock_datasets.Dataset.from_dict.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "ragas": mock_ragas,
                "datasets": mock_datasets,
                "ragas.metrics": mock_ragas_metrics,
            },
        ):
            result = evaluate_e2e(
                [{"question": "q", "answer": "a", "contexts": ["c"]}]
            )

        assert result.faithfulness == pytest.approx(0.9)
        assert result.answer_relevancy == pytest.approx(0.8)
        assert result.context_precision == pytest.approx(0.7)
        # No ground_truth => context_recall defaults to 0.0
        assert result.context_recall == pytest.approx(0.0)
        assert result.answer_correctness is None

    def test_success_with_ground_truth(self):
        from specagent.evaluation.metrics import evaluate_e2e

        mock_ragas = MagicMock()
        mock_datasets = MagicMock()
        mock_ragas_metrics = MagicMock()

        fake_results = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        mock_ragas.evaluate.return_value = fake_results
        mock_datasets.Dataset.from_dict.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "ragas": mock_ragas,
                "datasets": mock_datasets,
                "ragas.metrics": mock_ragas_metrics,
            },
        ):
            result = evaluate_e2e(
                [
                    {
                        "question": "q",
                        "answer": "a",
                        "contexts": ["c"],
                        "ground_truth": "gt",
                    }
                ]
            )

        assert result.faithfulness == pytest.approx(0.9)
        assert result.context_recall == pytest.approx(0.6)


@pytest.mark.unit
class TestRetryOnRateLimit:
    def test_rate_limit_error_triggers_print_and_reraise(self, capsys):
        from specagent.evaluation.metrics import _retry_on_rate_limit

        call_count = 0

        @_retry_on_rate_limit
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("rate limit exceeded, 429")

        with pytest.raises(Exception, match="rate limit"):
            failing_func()

        # Should have retried (3 attempts total)
        assert call_count == 3
        captured = capsys.readouterr()
        assert "Rate limit hit" in captured.out

    def test_non_rate_limit_error_still_retries_via_tenacity(self):
        from specagent.evaluation.metrics import _retry_on_rate_limit

        call_count = 0

        @_retry_on_rate_limit
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("some other error")

        with pytest.raises(ValueError, match="some other error"):
            failing_func()

        # tenacity's retry_if_exception_type(Exception) is broad — it matches
        # ValueError too, so tenacity retries all 3 attempts even though the
        # inner code does not print "Rate limit hit" for non-rate-limit errors.
        assert call_count == 3

    def test_success_returns_value(self):
        from specagent.evaluation.metrics import _retry_on_rate_limit

        @_retry_on_rate_limit
        def success_func():
            return 42

        assert success_func() == 42
