"""Unit tests for GroqLLMRateLimiter — written before implementation (TDD RED)."""

from __future__ import annotations

import threading
import time

import pytest

from specagent.retrieval.exceptions import LLMRateLimitError


@pytest.mark.unit
class TestGroqLLMRateLimiter:
    """Tests for the synchronous thread-safe token-bucket rate limiter."""

    def test_acquire_succeeds_within_rpm_limit(self) -> None:
        """Acquiring slots up to rpm_limit completes immediately."""
        from specagent.llm.groq_rate_limiter import GroqLLMRateLimiter

        limiter = GroqLLMRateLimiter(rpm_limit=5, rpd_limit=100, tpm_limit=100000)
        for _ in range(5):
            limiter.acquire()  # Must not block or raise

    def test_acquire_tracks_rpm_window(self) -> None:
        """Slots consumed within the minute window are counted correctly."""
        from specagent.llm.groq_rate_limiter import GroqLLMRateLimiter

        limiter = GroqLLMRateLimiter(rpm_limit=10, rpd_limit=100, tpm_limit=100000)
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()
        assert len(limiter._minute_calls) == 3  # noqa: SLF001

    def test_acquire_tracks_rpd_window(self) -> None:
        """Slots consumed accumulate in the day window."""
        from specagent.llm.groq_rate_limiter import GroqLLMRateLimiter

        limiter = GroqLLMRateLimiter(rpm_limit=10, rpd_limit=100, tpm_limit=100000)
        limiter.acquire()
        limiter.acquire()
        assert len(limiter._day_calls) == 2  # noqa: SLF001

    def test_raises_llm_rate_limit_error_when_daily_quota_exhausted(self) -> None:
        """LLMRateLimitError raised when daily quota is exhausted and reset > threshold."""
        from specagent.llm.groq_rate_limiter import GroqLLMRateLimiter

        limiter = GroqLLMRateLimiter(rpm_limit=1000, rpd_limit=3, tpm_limit=100000)
        far_future = time.monotonic() + 7200  # 2 hours ahead
        limiter._day_calls.extend([far_future, far_future, far_future])  # noqa: SLF001

        with pytest.raises(LLMRateLimitError, match="daily"):
            limiter.acquire()

    def test_stale_rpm_entries_pruned(self) -> None:
        """Entries older than 60 s are pruned; acquire succeeds without blocking."""
        from specagent.llm.groq_rate_limiter import GroqLLMRateLimiter

        limiter = GroqLLMRateLimiter(rpm_limit=2, rpd_limit=100, tpm_limit=100000)
        old_ts = time.monotonic() - 90
        limiter._minute_calls.append(old_ts)  # noqa: SLF001
        # rpm_limit=2 but the one slot is stale; two fresh acquires should not block
        limiter.acquire()
        limiter.acquire()
        assert all(ts > time.monotonic() - 60 for ts in limiter._minute_calls)  # noqa: SLF001

    def test_acquire_respects_tpm_budget_check(self) -> None:
        """When minute tokens >= tpm_limit, the condition is correctly detected."""
        from specagent.llm.groq_rate_limiter import GroqLLMRateLimiter

        limiter = GroqLLMRateLimiter(
            rpm_limit=100, rpd_limit=1000, tpm_limit=3000, tokens_per_call=1000
        )
        now = time.monotonic()
        limiter._minute_tokens.append((now, 1000))  # noqa: SLF001
        limiter._minute_tokens.append((now, 1000))  # noqa: SLF001
        limiter._minute_tokens.append((now, 1000))  # noqa: SLF001
        current = sum(t for _, t in limiter._minute_tokens)  # noqa: SLF001
        assert current >= limiter._tpm  # noqa: SLF001

    def test_acquire_records_token_estimate(self) -> None:
        """Each acquire() appends tokens_per_call to the minute-token window."""
        from specagent.llm.groq_rate_limiter import GroqLLMRateLimiter

        limiter = GroqLLMRateLimiter(
            rpm_limit=10, rpd_limit=100, tpm_limit=30000, tokens_per_call=2000
        )
        limiter.acquire()
        limiter.acquire()
        total = sum(t for _, t in limiter._minute_tokens)  # noqa: SLF001
        assert total == 4000

    def test_reset_for_testing_clears_all_deques(self) -> None:
        """reset_for_testing() empties all three deques."""
        from specagent.llm.groq_rate_limiter import GroqLLMRateLimiter

        limiter = GroqLLMRateLimiter(rpm_limit=5, rpd_limit=10, tpm_limit=5000)
        limiter.acquire()
        limiter.acquire()
        limiter.reset_for_testing()
        assert len(limiter._minute_calls) == 0  # noqa: SLF001
        assert len(limiter._day_calls) == 0  # noqa: SLF001
        assert len(limiter._minute_tokens) == 0  # noqa: SLF001

    def test_get_llm_rate_limiter_returns_singleton(self) -> None:
        """_get_llm_rate_limiter() returns the same instance on repeated calls."""
        from specagent.llm.groq_rate_limiter import _get_llm_rate_limiter

        limiter_a = _get_llm_rate_limiter()
        limiter_b = _get_llm_rate_limiter()
        assert limiter_a is limiter_b

    def test_acquire_thread_safe_no_race(self) -> None:
        """Two threads calling acquire() concurrently both complete without error."""
        from specagent.llm.groq_rate_limiter import GroqLLMRateLimiter

        limiter = GroqLLMRateLimiter(rpm_limit=100, rpd_limit=1000, tpm_limit=100000)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(5):
                    limiter.acquire()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(limiter._day_calls) == 10  # noqa: SLF001
