"""Unit tests for groq_vision_client — written before implementation (TDD RED)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from specagent.retrieval.exceptions import ConfigurationError, VisionError


def _make_image(placeholder: str = "image0.png", n_bytes: int = 15 * 1024) -> "ExtractedImage":
    """Build a minimal ExtractedImage for testing."""
    from tests.conftest import _make_png_bytes
    from specagent.retrieval.docx_image_extractor import ExtractedImage

    return ExtractedImage(
        placeholder_name=placeholder,
        media_filename="image1.png",
        image_bytes=_make_png_bytes(n_bytes=n_bytes),
        mime_type="image/png",
    )


def _groq_response(image_type: str, content: str) -> dict:
    """Build a minimal Groq chat completion response dict."""
    payload = {"type": image_type, "content": content}
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload),
                }
            }
        ]
    }


@pytest.mark.unit
class TestAnalyzeImage:
    """Tests for analyze_image()."""

    async def test_raises_configuration_error_for_empty_api_key(self) -> None:
        """analyze_image raises ConfigurationError when api_key is empty."""
        from specagent.retrieval.groq_vision_client import analyze_image

        image = _make_image()
        with pytest.raises(ConfigurationError, match="api_key"):
            await analyze_image(image, api_key="")

    async def test_returns_mermaid_for_call_flow_diagram(self, httpx_mock) -> None:
        """Call flow diagrams produce a mermaid fenced block."""
        mermaid_content = "```mermaid\nsequenceDiagram\n  A->>B: message\n```"
        httpx_mock.add_response(json=_groq_response("call_flow", mermaid_content))

        from specagent.retrieval.groq_vision_client import analyze_image

        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=AsyncMock()),
        ):
            result = await analyze_image(_make_image(), api_key="test-key")

        assert result.image_type == "call_flow"
        assert "mermaid" in result.markdown_content
        assert result.skipped is False

    async def test_returns_markdown_table_for_table_type(self, httpx_mock) -> None:
        """Table images produce Markdown table content."""
        table_content = "| Col A | Col B |\n|---|---|\n| 1 | 2 |"
        httpx_mock.add_response(json=_groq_response("table", table_content))

        from specagent.retrieval.groq_vision_client import analyze_image

        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=AsyncMock()),
        ):
            result = await analyze_image(_make_image(), api_key="test-key")

        assert result.image_type == "table"
        assert "|" in result.markdown_content

    async def test_returns_text_for_screenshot_text_type(self, httpx_mock) -> None:
        """Screenshot text images produce extracted text."""
        httpx_mock.add_response(
            json=_groq_response("screenshot_text", "Some extracted text from screenshot.")
        )

        from specagent.retrieval.groq_vision_client import analyze_image

        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=AsyncMock()),
        ):
            result = await analyze_image(_make_image(), api_key="test-key")

        assert result.image_type == "screenshot_text"
        assert "extracted" in result.markdown_content

    async def test_placeholder_name_preserved_in_result(self, httpx_mock) -> None:
        """The result carries back the original placeholder_name."""
        httpx_mock.add_response(json=_groq_response("other", "A decorative banner."))

        from specagent.retrieval.groq_vision_client import analyze_image

        image = _make_image(placeholder="image3.png")
        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=AsyncMock()),
        ):
            result = await analyze_image(image, api_key="test-key")

        assert result.placeholder_name == "image3.png"

    async def test_raises_vision_error_after_all_retries(self, httpx_mock) -> None:
        """VisionError is raised when all retry attempts return 503."""
        from specagent.retrieval.groq_vision_client import analyze_image

        # stop_after_attempt(5) means exactly 5 HTTP calls
        for _ in range(5):
            httpx_mock.add_response(status_code=503)

        with (
            patch(
                "specagent.retrieval.groq_vision_client._get_rate_limiter",
                return_value=MagicMock(acquire=AsyncMock()),
            ),
            pytest.raises(VisionError),
        ):
            await analyze_image(_make_image(), api_key="test-key")

    async def test_malformed_json_falls_back_to_other_type(self, httpx_mock) -> None:
        """Non-JSON response body is treated as image_type='other'."""
        httpx_mock.add_response(
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "not valid json {"}}
                ]
            }
        )

        from specagent.retrieval.groq_vision_client import analyze_image

        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=AsyncMock()),
        ):
            result = await analyze_image(_make_image(), api_key="test-key")

        assert result.image_type == "other"
        assert result.skipped is False

    async def test_rate_limiter_acquire_called_before_http(self, httpx_mock) -> None:
        """rate_limiter.acquire() is called before the HTTP request."""
        call_order: list[str] = []

        acquire_mock = AsyncMock(side_effect=lambda: call_order.append("acquire"))
        httpx_mock.add_response(
            json=_groq_response("other", "Logo"),
            # Use a callback to record when HTTP is called
        )

        from specagent.retrieval.groq_vision_client import analyze_image

        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=acquire_mock),
        ):
            await analyze_image(_make_image(), api_key="test-key")

        assert call_order[0] == "acquire"


@pytest.mark.unit
class TestIsRetryable:
    """Tests for _is_retryable() helper."""

    def test_429_is_retryable(self) -> None:
        from specagent.retrieval.groq_vision_client import _is_retryable
        import httpx

        resp = httpx.Response(429)
        exc = httpx.HTTPStatusError("rate limited", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True

    def test_503_is_retryable(self) -> None:
        from specagent.retrieval.groq_vision_client import _is_retryable
        import httpx

        resp = httpx.Response(503)
        exc = httpx.HTTPStatusError("server error", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True

    def test_400_is_not_retryable(self) -> None:
        from specagent.retrieval.groq_vision_client import _is_retryable
        import httpx

        resp = httpx.Response(400)
        exc = httpx.HTTPStatusError("bad request", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is False

    def test_timeout_is_retryable(self) -> None:
        from specagent.retrieval.groq_vision_client import _is_retryable
        import httpx

        assert _is_retryable(httpx.TimeoutException("timed out")) is True

    def test_value_error_is_not_retryable(self) -> None:
        from specagent.retrieval.groq_vision_client import _is_retryable

        assert _is_retryable(ValueError("nope")) is False
