"""TDD entry point for specagent/llm/factory.py.

Comprehensive rate-limit tests are in test_llm_factory_rate_limit.py.
Existing adapter/create_llm tests are in test_llm_factory.py.
This file satisfies the project TDD hook (test_factory.py required for factory.py).
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_is_llm_retryable_importable() -> None:
    """_is_llm_retryable is a public helper exported from factory module."""
    from specagent.llm.factory import _is_llm_retryable  # noqa: PLC0415

    assert callable(_is_llm_retryable)


@pytest.mark.unit
def test_wait_llm_retry_after_importable() -> None:
    """_wait_llm_retry_after is a public helper exported from factory module."""
    from specagent.llm.factory import _wait_llm_retry_after  # noqa: PLC0415

    assert callable(_wait_llm_retry_after)
