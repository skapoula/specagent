"""Groq vision API client for image analysis during .docx OCR ingestion."""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from specagent.retrieval.docx_image_extractor import ExtractedImage
from specagent.retrieval.exceptions import ConfigurationError, VisionError
from specagent.retrieval.groq_rate_limiter import _get_rate_limiter

logger = logging.getLogger(__name__)

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_KNOWN_IMAGE_TYPES = frozenset(["call_flow_diagram", "table", "screenshot_text", "other"])

_VISION_PROMPT = (
    "Analyze this image and respond with JSON only — no markdown wrapper, no explanation. "
    "Classify it as exactly one of these types: "
    "call_flow_diagram (a sequence diagram or call flow showing message exchanges between "
    "network entities), "
    "table (a table of data or parameters), "
    "screenshot_text (a screenshot containing readable text), "
    "or other (anything else). "
    'Respond as a JSON object: {"type": "<class>", "content": "<extracted content>"}. '
    "For call_flow_diagram: content MUST be a Mermaid DAG fenced code block: "
    "```mermaid\\ngraph TD\\n  A[Entity A] -->|message| B[Entity B]\\n```. "
    "For table: content must be a Markdown table. "
    "For screenshot_text: content must be the extracted text as Markdown. "
    "For other: content is a one-sentence plain-English description."
)


class ImageAnalysisResult(BaseModel):
    """Result of analysing a single image via the Groq vision API."""

    placeholder_name: str
    """MarkItDown placeholder that this result will replace, e.g. ``image0.png``."""

    markdown_content: str
    """Extracted content: Mermaid DAG, Markdown table, plain text, or description."""

    image_type: str
    """One of: ``call_flow_diagram``, ``table``, ``screenshot_text``, ``other``."""

    skipped: bool = False
    """``True`` when the image was not sent to the API (size filter, unsupported type…)."""

    skip_reason: str = ""
    """Human-readable reason when ``skipped=True``."""


async def analyze_image(
    image: ExtractedImage,
    api_key: str,
    model: str = _DEFAULT_MODEL,
) -> ImageAnalysisResult:
    """Send an image to the Groq vision API and return structured Markdown.

    Acquires a rate-limit slot before each attempt.  Uses ``tenacity`` for
    retry on transient server errors (429, 503, 504, timeout).

    Args:
        image: :class:`~specagent.retrieval.docx_image_extractor.ExtractedImage`
            containing raw bytes and metadata.
        api_key: Groq API key.  Never logged.
        model: Groq vision model identifier.

    Returns:
        :class:`ImageAnalysisResult` with ``markdown_content`` populated.

    Raises:
        ConfigurationError: If ``api_key`` is empty.
        VisionError: If the API call fails after all retries.
    """
    if not api_key:
        raise ConfigurationError(
            "api_key must be non-empty to call the Groq vision API."
        )

    await _get_rate_limiter().acquire()

    encoded = base64.b64encode(image.image_bytes).decode("ascii")
    data_url = f"data:{image.mime_type};base64,{encoded}"

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    async def _call() -> ImageAnalysisResult:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                _GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _VISION_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_url},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()

        raw_content = response.json()["choices"][0]["message"]["content"]
        return _parse_response(image.placeholder_name, raw_content)

    try:
        return await _call()
    except (ConfigurationError, VisionError):
        raise
    except Exception as exc:
        raise VisionError(
            f"Vision API failed for {image.placeholder_name!r} after retries: {exc}"
        ) from exc


def _parse_response(placeholder_name: str, raw_content: str) -> ImageAnalysisResult:
    """Parse the model's text response into an ImageAnalysisResult.

    Falls back to ``image_type='other'`` with the raw text as content if JSON
    parsing fails or the type field is unrecognised.
    """
    try:
        data = json.loads(raw_content)
        image_type = data.get("type", "other")
        if image_type not in _KNOWN_IMAGE_TYPES:
            image_type = "other"
        content = str(data.get("content", raw_content))
    except (json.JSONDecodeError, AttributeError):
        logger.warning(
            "Non-JSON response from vision API for %r; treating as 'other'",
            placeholder_name,
        )
        image_type = "other"
        content = raw_content

    return ImageAnalysisResult(
        placeholder_name=placeholder_name,
        markdown_content=content,
        image_type=image_type,
    )


def _is_retryable(exc: BaseException) -> bool:
    """Return ``True`` for transient HTTP errors that warrant a retry."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))
