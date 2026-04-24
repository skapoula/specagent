"""Unit tests for specagent.evaluation.judge module."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestCheckAnswerCorrectness:
    def test_exact_match(self):
        from specagent.evaluation.judge import check_answer_correctness

        assert check_answer_correctness("16", "16", use_llm_judge=False) is True

    def test_case_insensitive_exact(self):
        from specagent.evaluation.judge import check_answer_correctness

        assert check_answer_correctness("YES", "yes", use_llm_judge=False) is True

    def test_expected_contained_in_generated(self):
        from specagent.evaluation.judge import check_answer_correctness

        assert (
            check_answer_correctness("The answer is 16 subcarriers", "16", use_llm_judge=False)
            is True
        )

    def test_word_subset_match(self):
        from specagent.evaluation.judge import check_answer_correctness

        assert check_answer_correctness("NR SA mode", "NR SA", use_llm_judge=False) is True

    def test_no_match_without_llm(self):
        from specagent.evaluation.judge import check_answer_correctness

        assert check_answer_correctness("something unrelated", "16", use_llm_judge=False) is False

    def test_calls_llm_judge_when_fuzzy_fails(self):
        from specagent.evaluation.judge import check_answer_correctness

        with patch("specagent.evaluation.judge.llm_judge_answer", return_value=True) as mock_judge:
            result = check_answer_correctness("wrong answer", "correct", use_llm_judge=True)
        mock_judge.assert_called_once_with("wrong answer", "correct")
        assert result is True


@pytest.mark.unit
class TestLlmJudgeAnswer:
    def test_yes_response_returns_true(self):
        from specagent.evaluation.judge import llm_judge_answer

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "yes"
        with patch("specagent.evaluation.judge.get_llm", return_value=mock_llm):
            assert llm_judge_answer("sixteen", "16") is True

    def test_no_response_returns_false(self):
        from specagent.evaluation.judge import llm_judge_answer

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "no"
        with patch("specagent.evaluation.judge.get_llm", return_value=mock_llm):
            assert llm_judge_answer("something wrong", "16") is False

    def test_llm_failure_falls_back_to_fuzzy(self):
        from specagent.evaluation.judge import llm_judge_answer

        with patch("specagent.evaluation.judge.get_llm", side_effect=RuntimeError("api down")):
            # expected "16" is not in generated "other" → False
            assert llm_judge_answer("other", "16") is False
            # expected "16" is in generated "answer is 16" → True
            assert llm_judge_answer("answer is 16", "16") is True
