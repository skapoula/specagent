"""Unit tests for _vision_helpers — helper functions for Groq vision API processing."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest


@pytest.mark.unit
class TestIsRetryable:
    """Tests for _is_retryable() covering Retry-After aware 429 handling."""

    def test_429_without_retry_after_is_retryable(self) -> None:
        """429 with no Retry-After header is a transient rate limit — retry."""
        from specagent.retrieval._vision_helpers import _is_retryable

        resp = httpx.Response(429)
        exc = httpx.HTTPStatusError("rate limited", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True

    def test_429_with_short_retry_after_is_retryable(self) -> None:
        """429 with Retry-After <= 3600 s is a transient TPM/RPM limit — retry."""
        from specagent.retrieval._vision_helpers import _is_retryable

        resp = httpx.Response(429, headers={"retry-after": "261"})
        exc = httpx.HTTPStatusError("rate limited", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True

    def test_429_with_retry_after_over_threshold_is_not_retryable(self) -> None:
        """429 with Retry-After > 3600 s indicates daily quota exhaustion — do not retry."""
        from specagent.retrieval._vision_helpers import _is_retryable

        resp = httpx.Response(429, headers={"retry-after": "7200"})
        exc = httpx.HTTPStatusError("daily quota", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is False

    def test_503_is_retryable(self) -> None:
        """503 is always a transient server error."""
        from specagent.retrieval._vision_helpers import _is_retryable

        resp = httpx.Response(503)
        exc = httpx.HTTPStatusError("unavailable", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True

    def test_504_is_retryable(self) -> None:
        """504 gateway timeout is transient."""
        from specagent.retrieval._vision_helpers import _is_retryable

        resp = httpx.Response(504)
        exc = httpx.HTTPStatusError("gateway timeout", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True

    def test_400_is_not_retryable(self) -> None:
        """400 is a client error — not transient."""
        from specagent.retrieval._vision_helpers import _is_retryable

        resp = httpx.Response(400)
        exc = httpx.HTTPStatusError("bad request", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is False

    def test_timeout_is_retryable(self) -> None:
        """Network timeouts are transient."""
        from specagent.retrieval._vision_helpers import _is_retryable

        assert _is_retryable(httpx.TimeoutException("timed out")) is True

    def test_connect_error_is_retryable(self) -> None:
        """Connection errors are transient."""
        from specagent.retrieval._vision_helpers import _is_retryable

        assert _is_retryable(httpx.ConnectError("refused")) is True

    def test_value_error_is_not_retryable(self) -> None:
        """Non-HTTP exceptions are not retryable."""
        from specagent.retrieval._vision_helpers import _is_retryable

        assert _is_retryable(ValueError("nope")) is False

    def test_retry_after_exactly_at_threshold_is_retryable(self) -> None:
        """Retry-After == _MAX_RETRY_AFTER (not >) is still retryable."""
        from specagent.retrieval._vision_helpers import _is_retryable, _MAX_RETRY_AFTER

        resp = httpx.Response(429, headers={"retry-after": str(_MAX_RETRY_AFTER)})
        exc = httpx.HTTPStatusError("rate limited", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True
