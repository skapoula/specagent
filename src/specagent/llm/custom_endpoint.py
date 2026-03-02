"""
Custom LLM client for OpenAI-compatible inference endpoints.

Provides a simple wrapper for local/custom inference endpoints that implement
the OpenAI chat completions API format.
"""

import json
import logging
import time
from typing import Optional

import requests

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
                response = requests.post(
                    self.endpoint_url, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                inference_ms = (time.perf_counter() - start_time) * 1000

                result = response.json()
                return result["choices"][0]["message"]["content"], inference_ms

            except requests.HTTPError as e:
                last_exception = e
                # Retry on 502/503 (Bad Gateway/Service Unavailable) for serverless cold starts
                if e.response.status_code in (502, 503, 504):
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Endpoint returned {e.response.status_code}, "
                            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(delay)
                        continue
                # For other errors or last attempt, re-raise
                raise

            except (requests.Timeout, requests.ConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Connection error: {str(e)}, "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue
                raise

        # If we get here, all retries failed
        if last_exception:  # pragma: no cover
            raise last_exception
        raise RuntimeError("All retry attempts failed")

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
            logger.info(f"Performing health check on endpoint: {self.endpoint_url}")
            response = requests.post(
                self.endpoint_url, json=test_payload, timeout=timeout
            )
            response.raise_for_status()

            # Verify response structure
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                logger.info(f"✓ Endpoint health check passed ({response.elapsed.total_seconds():.2f}s)")
                return True, f"Endpoint healthy (responded in {response.elapsed.total_seconds():.2f}s)"
            else:
                error_msg = "Endpoint returned invalid response structure"
                logger.warning(f"✗ {error_msg}")
                return False, error_msg

        except requests.Timeout:
            error_msg = f"Endpoint timed out after {timeout}s"
            logger.error(f"✗ {error_msg}")
            return False, error_msg

        except requests.ConnectionError as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(f"✗ {error_msg}")
            return False, error_msg

        except requests.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.reason}"
            logger.error(f"✗ {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"✗ {error_msg}")
            return False, error_msg


def create_custom_llm(
    endpoint_url: Optional[str] = None,
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
        endpoint_url = getattr(
            settings,
            "custom_endpoint_url",
            "http://qwen3-4b-predictor.ml-serving.10.0.1.2.sslip.io:30750/v1/chat/completions",
        )

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

    endpoint_url = getattr(
        settings,
        "custom_endpoint_url",
        "http://qwen3-4b-predictor.ml-serving.10.0.1.2.sslip.io:30750/v1/chat/completions",
    )

    # Create temporary client for health check
    client = CustomEndpointLLM(endpoint_url=endpoint_url)
    return client.health_check(timeout=timeout)
