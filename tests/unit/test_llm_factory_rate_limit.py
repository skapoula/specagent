"""Unit tests for _GroqAdapter rate-limit handling — TDD RED phase."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _make_openai_rate_limit_error(
    status_code: int = 429, retry_after: str | None = "5"
) -> Any:
    """Create a minimal fake openai.RateLimitError-like exception."""
    try:
        import openai  # noqa: PLC0415

        headers = {}
        if retry_after is not None:
            headers["retry-after"] = retry_after

        response = MagicMock()
        response.status_code = status_code
        response.headers = headers

        exc = openai.RateLimitError(
            message="Rate limit exceeded",
            response=response,
            body={"error": {"message": "Rate limit exceeded"}},
        )
        return exc
    except ImportError:
        pytest.skip("openai package not installed")


def _make_openai_api_error(status_code: int) -> Any:
    """Create a fake openai.APIStatusError for non-429 codes."""
    try:
        import openai  # noqa: PLC0415

        response = MagicMock()
        response.status_code = status_code
        response.headers = {}

        if status_code == 503:
            exc = openai.InternalServerError(
                message=f"HTTP {status_code}",
                response=response,
                body={"error": {"message": f"HTTP {status_code}"}},
            )
        else:
            exc = openai.BadRequestError(
                message=f"HTTP {status_code}",
                response=response,
                body={"error": {"message": f"HTTP {status_code}"}},
            )
        return exc
    except ImportError:
        pytest.skip("openai package not installed")


@pytest.mark.unit
class TestIsLLMRetryable:
    """Tests for the _is_llm_retryable() helper."""

    def test_returns_true_for_429(self) -> None:
        """429 with Retry-After <= 3600 s is retryable."""
        from specagent.llm.factory import _is_llm_retryable  # noqa: PLC0415

        exc = _make_openai_rate_limit_error(status_code=429, retry_after="30")
        assert _is_llm_retryable(exc) is True

    def test_returns_false_for_daily_quota_exceeded(self) -> None:
        """429 with Retry-After > 3600 s is not retryable (daily quota)."""
        from specagent.llm.factory import _is_llm_retryable  # noqa: PLC0415

        exc = _make_openai_rate_limit_error(status_code=429, retry_after="7200")
        assert _is_llm_retryable(exc) is False

    def test_returns_true_for_503(self) -> None:
        """503 server error is retryable."""
        from specagent.llm.factory import _is_llm_retryable  # noqa: PLC0415

        exc = _make_openai_api_error(status_code=503)
        assert _is_llm_retryable(exc) is True

    def test_returns_false_for_400(self) -> None:
        """400 bad request is not retryable."""
        from specagent.llm.factory import _is_llm_retryable  # noqa: PLC0415

        exc = _make_openai_api_error(status_code=400)
        assert _is_llm_retryable(exc) is False

    def test_returns_false_for_non_openai_exception(self) -> None:
        """Non-openai exceptions are not retryable."""
        from specagent.llm.factory import _is_llm_retryable  # noqa: PLC0415

        assert _is_llm_retryable(ValueError("not an API error")) is False


@pytest.mark.unit
class TestWaitLLMRetryAfter:
    """Tests for the _wait_llm_retry_after() tenacity wait callable."""

    def _make_retry_state(self, exc: BaseException, attempt: int = 1) -> Any:
        """Build a minimal tenacity RetryCallState-like object."""
        state = MagicMock()
        state.attempt_number = attempt
        state.outcome = MagicMock()
        state.outcome.exception.return_value = exc
        return state

    def test_uses_retry_after_header_when_present(self) -> None:
        """Returns the Retry-After header value when present and <= 3600."""
        from specagent.llm.factory import _wait_llm_retry_after  # noqa: PLC0415

        exc = _make_openai_rate_limit_error(status_code=429, retry_after="15")
        state = self._make_retry_state(exc, attempt=1)
        wait = _wait_llm_retry_after(state)
        assert wait == pytest.approx(15.0)

    def test_falls_back_to_exponential_when_no_retry_after(self) -> None:
        """Returns capped exponential backoff when Retry-After header is absent."""
        from specagent.llm.factory import _wait_llm_retry_after  # noqa: PLC0415

        exc = _make_openai_rate_limit_error(status_code=429, retry_after=None)
        state = self._make_retry_state(exc, attempt=1)
        wait = _wait_llm_retry_after(state)
        # attempt 1 → 2^1 * 2 = 4.0
        assert wait == pytest.approx(4.0)

    def test_exponential_backoff_is_capped_at_60s(self) -> None:
        """Exponential backoff is capped at 60 s."""
        from specagent.llm.factory import _wait_llm_retry_after  # noqa: PLC0415

        exc = _make_openai_rate_limit_error(status_code=429, retry_after=None)
        state = self._make_retry_state(exc, attempt=10)
        wait = _wait_llm_retry_after(state)
        assert wait <= 60.0


@pytest.mark.unit
class TestGroqAdapterRateLimit:
    """Tests for _GroqAdapter.invoke() throttle and retry integration."""

    def _make_adapter(self, mock_model: Any) -> Any:
        """Build a _GroqAdapter with a mock ChatOpenAI model."""
        from specagent.llm.factory import _GroqAdapter  # noqa: PLC0415

        return _GroqAdapter(mock_model)

    def test_acquire_called_before_model_invoke(self) -> None:
        """_get_llm_rate_limiter().acquire() is called before model.invoke()."""
        mock_model = MagicMock()
        response = MagicMock()
        response.content = "answer"
        response.usage_metadata = None
        mock_model.invoke.return_value = response

        mock_limiter = MagicMock()
        call_order: list[str] = []
        mock_limiter.acquire.side_effect = lambda *a, **kw: call_order.append("acquire")
        mock_model.invoke.side_effect = lambda *a, **kw: (
            call_order.append("invoke"),
            response,
        )[1]

        adapter = self._make_adapter(mock_model)
        # Patch at the source module where the function is defined
        with patch("specagent.llm.groq_rate_limiter._get_llm_rate_limiter", return_value=mock_limiter):
            adapter.invoke("hello")

        assert call_order[0] == "acquire", "acquire() must be called before invoke()"

    def test_acquire_receives_token_estimate_from_settings(self) -> None:
        """acquire() is called with settings.groq_llm_tokens_per_call_estimate."""
        mock_model = MagicMock()
        response = MagicMock()
        response.content = "answer"
        response.usage_metadata = None
        mock_model.invoke.return_value = response

        mock_limiter = MagicMock()
        adapter = self._make_adapter(mock_model)

        mock_settings = MagicMock()
        mock_settings.groq_llm_tokens_per_call_estimate = 1234
        mock_settings.groq_llm_max_retries = 1
        mock_settings.groq_model = "test-model"

        with (
            patch("specagent.llm.groq_rate_limiter._get_llm_rate_limiter", return_value=mock_limiter),
            patch("specagent.config.settings", mock_settings),
        ):
            adapter.invoke("hello")

        mock_limiter.acquire.assert_called_once_with(1234)

    def test_retries_on_429_and_returns_on_success(self) -> None:
        """When model.invoke raises 429 once then succeeds, the result is returned."""
        rate_limit_exc = _make_openai_rate_limit_error(status_code=429, retry_after="0")
        success_response = MagicMock()
        success_response.content = "final answer"
        success_response.usage_metadata = None

        mock_model = MagicMock()
        mock_model.invoke.side_effect = [rate_limit_exc, success_response]

        mock_limiter = MagicMock()
        adapter = self._make_adapter(mock_model)

        mock_settings = MagicMock()
        mock_settings.groq_llm_tokens_per_call_estimate = 6000
        mock_settings.groq_llm_max_retries = 3
        mock_settings.groq_model = "test-model"

        with (
            patch("specagent.llm.groq_rate_limiter._get_llm_rate_limiter", return_value=mock_limiter),
            patch("specagent.config.settings", mock_settings),
            patch("specagent.llm.factory._wait_llm_retry_after", return_value=0.0),
        ):
            result = adapter.invoke("hello")

        assert result == "final answer"
        assert mock_model.invoke.call_count == 2

    def test_raises_after_max_retries_exhausted(self) -> None:
        """Exception propagates after groq_llm_max_retries attempts."""
        rate_limit_exc = _make_openai_rate_limit_error(status_code=429, retry_after="0")

        mock_model = MagicMock()
        mock_model.invoke.side_effect = rate_limit_exc

        mock_limiter = MagicMock()
        adapter = self._make_adapter(mock_model)

        mock_settings = MagicMock()
        mock_settings.groq_llm_tokens_per_call_estimate = 6000
        mock_settings.groq_llm_max_retries = 2
        mock_settings.groq_model = "test-model"

        with (
            patch("specagent.llm.groq_rate_limiter._get_llm_rate_limiter", return_value=mock_limiter),
            patch("specagent.config.settings", mock_settings),
            patch("specagent.llm.factory._wait_llm_retry_after", return_value=0.0),
        ):
            import openai  # noqa: PLC0415

            with pytest.raises(openai.RateLimitError):
                adapter.invoke("hello")

        assert mock_model.invoke.call_count == 2  # max_retries=2
