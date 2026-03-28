"""Unit tests for GroqVisionRateLimiter — written before implementation (TDD RED)."""

from __future__ import annotations

import asyncio
import time

import pytest

from specagent.retrieval.exceptions import VisionError


@pytest.mark.unit
class TestGroqVisionRateLimiter:
    """Tests for the async token-bucket rate limiter."""

    async def test_acquire_succeeds_within_rpm_limit(self) -> None:
        """Acquiring slots up to rpm_limit completes immediately."""
        from specagent.retrieval.groq_rate_limiter import GroqVisionRateLimiter

        limiter = GroqVisionRateLimiter(rpm_limit=5, rpd_limit=100)
        for _ in range(5):
            await limiter.acquire()  # Must not block or raise

    async def test_acquire_tracks_minute_window(self) -> None:
        """Slots consumed within the minute window are counted correctly."""
        from specagent.retrieval.groq_rate_limiter import GroqVisionRateLimiter

        limiter = GroqVisionRateLimiter(rpm_limit=3, rpd_limit=100)
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        # Minute deque should now have 3 entries
        assert len(limiter._minute_calls) == 3  # noqa: SLF001

    async def test_raises_vision_error_when_daily_limit_exhausted(self) -> None:
        """VisionError raised when daily quota is exhausted and reset > threshold."""
        from specagent.retrieval.groq_rate_limiter import GroqVisionRateLimiter

        limiter = GroqVisionRateLimiter(rpm_limit=1000, rpd_limit=3)
        # Fill day deque with timestamps far in the future to simulate exhaustion
        far_future = time.monotonic() + 7200  # 2 hours ahead
        limiter._day_calls.extend([far_future, far_future, far_future])  # noqa: SLF001

        with pytest.raises(VisionError, match="daily"):
            await limiter.acquire()

    async def test_reset_for_testing_clears_all_counters(self) -> None:
        """reset_for_testing() empties both deques."""
        from specagent.retrieval.groq_rate_limiter import GroqVisionRateLimiter

        limiter = GroqVisionRateLimiter(rpm_limit=5, rpd_limit=10)
        await limiter.acquire()
        await limiter.acquire()
        limiter.reset_for_testing()
        assert len(limiter._minute_calls) == 0  # noqa: SLF001
        assert len(limiter._day_calls) == 0  # noqa: SLF001

    async def test_old_minute_entries_are_pruned(self) -> None:
        """Entries older than 60 s are removed from the minute window on acquire."""
        from specagent.retrieval.groq_rate_limiter import GroqVisionRateLimiter

        limiter = GroqVisionRateLimiter(rpm_limit=2, rpd_limit=100)
        # Insert a timestamp from 90 s ago (outside the 60 s window)
        old_ts = time.monotonic() - 90
        limiter._minute_calls.append(old_ts)  # noqa: SLF001
        # Now rpm_limit=2 but one slot is stale; acquire should succeed without blocking
        await limiter.acquire()
        await limiter.acquire()
        # Stale entry should have been pruned, leaving only the 2 fresh ones
        assert all(ts > time.monotonic() - 60 for ts in limiter._minute_calls)  # noqa: SLF001

    def test_get_rate_limiter_returns_singleton(self) -> None:
        """_get_rate_limiter() returns the same instance on repeated calls."""
        from specagent.retrieval.groq_rate_limiter import _get_rate_limiter

        limiter_a = _get_rate_limiter()
        limiter_b = _get_rate_limiter()
        assert limiter_a is limiter_b
