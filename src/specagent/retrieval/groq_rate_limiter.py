"""Async token-bucket rate limiter for the Groq vision API free-tier limits."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from specagent.retrieval.exceptions import VisionError

logger = logging.getLogger(__name__)

_MINUTE_WINDOW: float = 60.0
_DAY_WINDOW: float = 86400.0
# If the daily quota is exhausted and the oldest entry resets in more than this
# many seconds, raise immediately rather than sleeping indefinitely.
_MAX_WAIT_SECONDS: float = 3600.0


class GroqVisionRateLimiter:
    """Async token-bucket rate limiter for the Groq vision API.

    Enforces two independent rolling-window limits:

    * **Per-minute**: ``rpm_limit`` requests in any 60-second window.
    * **Per-day**: ``rpd_limit`` requests in any 86 400-second window.

    ``acquire()`` must be awaited before each API call.  The limiter is safe
    for concurrent coroutines within a single process (``asyncio.Lock``), but
    is **not** safe across multiple processes.
    """

    def __init__(self, rpm_limit: int = 30, rpd_limit: int = 1000) -> None:
        """Initialise the limiter with the configured per-minute and per-day caps."""
        self._rpm = rpm_limit
        self._rpd = rpd_limit
        self._minute_calls: deque[float] = deque()
        self._day_calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a request slot is available under both rate limits.

        Prunes expired entries from both windows on each call, then sleeps
        only as long as necessary when a window is saturated.

        Raises:
            VisionError: If the daily quota is exhausted and the reset would
                require waiting longer than one hour.
        """
        async with self._lock:
            now = time.monotonic()

            # Prune entries outside their respective windows
            _prune(self._minute_calls, now - _MINUTE_WINDOW)
            _prune(self._day_calls, now - _DAY_WINDOW)

            # Check daily limit first — if exhausted beyond threshold, raise
            if len(self._day_calls) >= self._rpd:
                wait = (self._day_calls[0] + _DAY_WINDOW) - now
                if wait > _MAX_WAIT_SECONDS:
                    raise VisionError(
                        f"Groq vision daily quota exhausted. "
                        f"Next slot available in {wait / 3600:.1f} h. "
                        "Reduce batch size or wait until tomorrow."
                    )
                logger.warning("Daily vision quota reached; sleeping %.0f s", wait)
                await asyncio.sleep(max(0.0, wait))
                now = time.monotonic()
                _prune(self._day_calls, now - _DAY_WINDOW)

            # Handle per-minute limit
            if len(self._minute_calls) >= self._rpm:
                wait = (self._minute_calls[0] + _MINUTE_WINDOW) - now
                if wait > 0:
                    logger.debug("Vision RPM limit reached; sleeping %.1f s", wait)
                    await asyncio.sleep(wait)
                now = time.monotonic()
                _prune(self._minute_calls, now - _MINUTE_WINDOW)

            self._minute_calls.append(now)
            self._day_calls.append(now)

    def reset_for_testing(self) -> None:
        """Clear all call timestamps. Use only in tests."""
        self._minute_calls.clear()
        self._day_calls.clear()


def _prune(dq: deque[float], cutoff: float) -> None:
    """Remove entries from the left of *dq* that are older than *cutoff*."""
    while dq and dq[0] < cutoff:
        dq.popleft()


_rate_limiter: GroqVisionRateLimiter | None = None


def _get_rate_limiter() -> GroqVisionRateLimiter:
    """Return the process-wide rate limiter singleton, initialised on first call."""
    global _rate_limiter  # noqa: PLW0603
    if _rate_limiter is None:
        from specagent.config import settings  # noqa: PLC0415

        _rate_limiter = GroqVisionRateLimiter(
            rpm_limit=settings.vision_rpm_limit,
            rpd_limit=settings.vision_rpd_limit,
        )
    return _rate_limiter
