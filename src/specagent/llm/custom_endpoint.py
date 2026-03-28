"""
Custom LLM client for OpenAI-compatible inference endpoints.

Provides a simple wrapper for local/custom inference endpoints that implement
the OpenAI chat completions API format.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import requests
from langsmith import traceable

if TYPE_CHECKING:
    from specagent.observability.models import LLMCallRecord

logger = logging.getLogger(__name__)


class CustomEndpointLLM:
    """LLM client for OpenAI-compatible endpoints."""

    def __init__(
        self,
        endpoint_url: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 120,  # Increased from 30s to 120s for slow LLMs
        max_retries: int = 3,  # Retry for serverless cold starts
        retry_delay: float = 2.0,  # Initial retry delay in seconds
    ):
        """
        Initialize custom endpoint client.

        Args:
            endpoint_url: Full URL to the /v1/chat/completions endpoint
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for 502/503 errors
            retry_delay: Initial delay between retries (uses exponential backoff)
        """
        self.endpoint_url = endpoint_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._tls = threading.local()

    def invoke(self, prompt: str) -> str:
        """
        Call the LLM with a prompt.

        Implements retry logic with exponential backoff for serverless endpoints
        that may return 502/503 errors during cold starts.

        Args:
            prompt: The input prompt text

        Returns:
            The generated response text

        Raises:
            requests.HTTPError: If the API request fails after all retries
        """
        result_text, _ = self.invoke_with_timing(prompt)
        return result_text

    @traceable(name="custom_endpoint_llm", run_type="llm")
    def invoke_with_timing(self, prompt: str) -> tuple[str, float]:
        """
        Call the LLM with a prompt and return timing information.

        Implements retry logic with exponential backoff for serverless endpoints
        that may return 502/503 errors during cold starts.

        Args:
            prompt: The input prompt text

        Returns:
            Tuple of (response_text, inference_time_ms)

        Raises:
            requests.HTTPError: If the API request fails after all retries
        """
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.perf_counter()
                response = requests.post(self.endpoint_url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                inference_ms = (time.perf_counter() - start_time) * 1000

                result = response.json()
                result_text = result["choices"][0]["message"]["content"]
                try:
                    from specagent.observability.models import LLMCallRecord  # noqa: PLC0415

                    usage = result.get("usage", {})
                    self._tls.last_call = LLMCallRecord(
                        node="",
                        trace_id="",
                        model=result.get("model", "unknown"),
                        provider="custom_endpoint",
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                        inference_ms=inference_ms,
                    )
                except Exception:
                    self._tls.last_call = None
                return result_text, inference_ms

            except requests.HTTPError as e:
                last_exception = e
                # Retry on 502/503 (Bad Gateway/Service Unavailable) for serverless cold starts
                if e.response.status_code in (502, 503, 504):
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2**attempt)  # Exponential backoff
                        logger.warning(
                            "Endpoint returned %s, retrying in %.1fs (attempt %d/%d)",
                            e.response.status_code,
                            delay,
                            attempt + 1,
                            self.max_retries,
                        )
                        time.sleep(delay)
                        continue
                # For other errors or last attempt, re-raise
                raise

            except (requests.Timeout, requests.ConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        "Connection error: %s, retrying in %.1fs (attempt %d/%d)",
                        e,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise

        # If we get here, all retries failed
        if last_exception:  # pragma: no cover
            raise last_exception
        raise RuntimeError("All retry attempts failed")

    def get_last_call(self) -> LLMCallRecord | None:
        """Return the LLMCallRecord from the most recent successful invoke(), or None."""
        return getattr(self._tls, "last_call", None)

    def health_check(self, timeout: int = 30) -> tuple[bool, str]:
        """
        Perform a quick health check on the LLM endpoint.

        Sends a minimal test prompt to verify the endpoint is responsive.
        Uses a shorter timeout than normal invocations for fast failure detection.

        Args:
            timeout: Health check timeout in seconds (default: 30s)

        Returns:
            Tuple of (is_healthy: bool, message: str)
            - (True, "Endpoint healthy") if successful
            - (False, error_message) if endpoint is down or unresponsive
        """
        test_payload = {
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.0,
            "max_tokens": 1,
        }

        try:
            logger.info("Performing health check on endpoint: %s", self.endpoint_url)
            response = requests.post(self.endpoint_url, json=test_payload, timeout=timeout)
            response.raise_for_status()

            # Verify response structure
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                elapsed = response.elapsed.total_seconds()
                logger.info("Endpoint health check passed (%.2fs)", elapsed)
                return True, f"Endpoint healthy (responded in {elapsed:.2f}s)"
            else:
                error_msg = "Endpoint returned invalid response structure"
                logger.warning("Health check: %s", error_msg)
                return False, error_msg

        except requests.Timeout:
            error_msg = f"Endpoint timed out after {timeout}s"
            logger.error("Health check failed: %s", error_msg)
            return False, error_msg

        except requests.ConnectionError as e:
            error_msg = f"Connection failed: {e!s}"
            logger.error("Health check failed: %s", error_msg)
            return False, error_msg

        except requests.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.reason}"
            logger.error("Health check failed: %s", error_msg)
            return False, error_msg

        except Exception as e:
            error_msg = f"Unexpected error: {e!s}"
            logger.error("Health check failed: %s", error_msg)
            return False, error_msg


def create_custom_llm(
    endpoint_url: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> CustomEndpointLLM:
    """
    Create a custom LLM client.

    Args:
        endpoint_url: OpenAI-compatible endpoint URL. If None, uses settings.
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Configured CustomEndpointLLM instance
    """
    if endpoint_url is None:
        from specagent.config import settings

        # Default to configured endpoint or fallback
        endpoint_url = str(settings.custom_endpoint_url)

    return CustomEndpointLLM(
        endpoint_url=endpoint_url, temperature=temperature, max_tokens=max_tokens
    )


def check_llm_endpoint_health(timeout: int = 30) -> tuple[bool, str]:
    """
    Perform a health check on the configured LLM endpoint.

    This is a convenience function for checking endpoint availability before
    running benchmarks or other operations that require the LLM.

    Args:
        timeout: Health check timeout in seconds (default: 30s)

    Returns:
        Tuple of (is_healthy: bool, message: str)

    Example:
        >>> is_healthy, msg = check_llm_endpoint_health()
        >>> if not is_healthy:
        >>>     print(f"Endpoint unavailable: {msg}")
        >>>     sys.exit(1)
    """
    from specagent.config import settings

    endpoint_url = settings.custom_endpoint_url

    # Create temporary client for health check
    client = CustomEndpointLLM(endpoint_url=endpoint_url)
    return client.health_check(timeout=timeout)
