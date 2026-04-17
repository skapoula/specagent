"""Synchronous thread-safe sliding-window rate limiter for the Groq LLM API.

This is the synchronous counterpart to ``retrieval/groq_rate_limiter.py`` (which
is async and used by the vision pipeline).  The LLM path uses LangChain's
``ChatOpenAI`` which is synchronous, so ``threading.Lock`` is used here instead
of ``asyncio.Lock``.

Enforces three independent rolling-window limits:

* **Per-minute requests**: ``rpm_limit`` requests in any 60-second window.
* **Per-day requests**: ``rpd_limit`` requests in any 86 400-second window.
* **Per-minute tokens**: ``tpm_limit`` estimated tokens in any 60-second window,
  using ``tokens_per_call`` as a conservative per-request estimate.

``acquire()`` blocks the calling thread until a slot is available under all limits.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from specagent.retrieval.exceptions import LLMRateLimitError

logger = logging.getLogger(__name__)

_MINUTE_WINDOW: float = 60.0
_DAY_WINDOW: float = 86400.0
# If the daily quota is exhausted and the oldest entry resets in more than this
# many seconds, raise immediately rather than sleeping indefinitely.
_MAX_WAIT_SECONDS: float = 3600.0


class GroqLLMRateLimiter:
    """Thread-safe sliding-window rate limiter for the synchronous Groq LLM path.

    Safe for use from multiple threads within a single process.  Not safe across
    multiple processes (no shared state).  specagent defaults to ``api_workers=1``
    so single-process operation is the normal deployment.
    """

    def __init__(
        self,
        rpm_limit: int = 30,
        rpd_limit: int = 1000,
        tpm_limit: int = 30000,
        tokens_per_call: int = 6000,
    ) -> None:
        """Initialise the limiter with the configured per-minute, per-day, and TPM caps."""
        self._rpm = rpm_limit
        self._rpd = rpd_limit
        self._tpm = tpm_limit
        self._tokens_per_call = tokens_per_call
        self._minute_calls: deque[float] = deque()
        self._day_calls: deque[float] = deque()
        # Each entry is (timestamp, token_count) for TPM tracking.
        self._minute_tokens: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int = 0) -> None:
        """Block until a request slot is available under all three rate limits.

        Prunes expired entries from all windows on each call, then sleeps only
        as long as necessary when a window is saturated.

        Args:
            estimated_tokens: Token count to reserve against the TPM budget.
                If 0, falls back to the configured ``tokens_per_call`` default.

        Raises:
            LLMRateLimitError: If the daily quota is exhausted and the reset would
                require waiting longer than one hour.
        """
        tokens = estimated_tokens if estimated_tokens > 0 else self._tokens_per_call

        with self._lock:
            now = time.monotonic()

            # Prune entries outside their respective windows
            _prune(self._minute_calls, now - _MINUTE_WINDOW)
            _prune(self._day_calls, now - _DAY_WINDOW)
            _prune_tokens(self._minute_tokens, now - _MINUTE_WINDOW)

            # Check daily limit first — if exhausted beyond threshold, raise
            if len(self._day_calls) >= self._rpd:
                wait = (self._day_calls[0] + _DAY_WINDOW) - now
                if wait > _MAX_WAIT_SECONDS:
                    raise LLMRateLimitError(
                        f"Groq LLM daily quota exhausted. "
                        f"Next slot available in {wait / 3600:.1f} h. "
                        "Reduce query rate or wait until tomorrow."
                    )
                logger.warning("Daily LLM quota reached; sleeping %.0f s", wait)
                time.sleep(max(0.0, wait))
                now = time.monotonic()
                _prune(self._day_calls, now - _DAY_WINDOW)

            # Handle per-minute request limit
            if len(self._minute_calls) >= self._rpm:
                wait = (self._minute_calls[0] + _MINUTE_WINDOW) - now
                if wait > 0:
                    logger.warning("LLM RPM limit reached; sleeping %.1f s", wait)
                    time.sleep(wait)
                now = time.monotonic()
                _prune(self._minute_calls, now - _MINUTE_WINDOW)

            # Handle per-minute token budget (TPM)
            current_minute_tokens = sum(t for _, t in self._minute_tokens)
            if current_minute_tokens + tokens > self._tpm and self._minute_tokens:
                wait = (self._minute_tokens[0][0] + _MINUTE_WINDOW) - now
                if wait > 0:
                    logger.warning(
                        "LLM TPM budget reached (%d/%d tokens/min); sleeping %.1f s",
                        current_minute_tokens,
                        self._tpm,
                        wait,
                    )
                    time.sleep(wait)
                now = time.monotonic()
                _prune_tokens(self._minute_tokens, now - _MINUTE_WINDOW)

            self._minute_calls.append(now)
            self._day_calls.append(now)
            self._minute_tokens.append((now, tokens))
            logger.debug(
                "LLM slot acquired: rpm=%d/%d rpd=%d/%d tpm_est=%d/%d",
                len(self._minute_calls),
                self._rpm,
                len(self._day_calls),
                self._rpd,
                sum(t for _, t in self._minute_tokens),
                self._tpm,
            )

    def reset_for_testing(self) -> None:
        """Clear all call timestamps and token records. Use only in tests."""
        self._minute_calls.clear()
        self._day_calls.clear()
        self._minute_tokens.clear()


def _prune(dq: deque[float], cutoff: float) -> None:
    """Remove entries from the left of *dq* that are older than *cutoff*."""
    while dq and dq[0] < cutoff:
        dq.popleft()


def _prune_tokens(dq: deque[tuple[float, int]], cutoff: float) -> None:
    """Remove (timestamp, tokens) entries from the left of *dq* older than *cutoff*."""
    while dq and dq[0][0] < cutoff:
        dq.popleft()


_llm_rate_limiter: GroqLLMRateLimiter | None = None
_init_lock = threading.Lock()


def _get_llm_rate_limiter() -> GroqLLMRateLimiter:
    """Return the process-wide LLM rate limiter singleton, initialised on first call."""
    global _llm_rate_limiter  # noqa: PLW0603
    if _llm_rate_limiter is None:
        with _init_lock:
            if _llm_rate_limiter is None:
                from specagent.config import settings  # noqa: PLC0415

                _llm_rate_limiter = GroqLLMRateLimiter(
                    rpm_limit=settings.groq_llm_rpm_limit,
                    rpd_limit=settings.groq_llm_rpd_limit,
                    tpm_limit=settings.groq_llm_tpm_limit,
                    tokens_per_call=settings.groq_llm_tokens_per_call_estimate,
                )
    return _llm_rate_limiter
