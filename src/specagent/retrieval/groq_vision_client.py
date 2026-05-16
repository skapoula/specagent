"""Groq vision API client for image analysis during .docx OCR ingestion."""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt

if TYPE_CHECKING:
    from specagent.retrieval.docx_image_extractor import ExtractedImage
from specagent.config import settings
from specagent.retrieval._vision_helpers import (
    _KNOWN_DIAGRAM_TYPES,
    _MERMAID_SUBTYPE,
    _fix_mermaid_header,
    _is_retryable,
)
from specagent.retrieval._vision_prompts import (
    _RESPONSE_JSON_SCHEMA,
    _SYSTEM_PROMPT,
    _USER_MESSAGE_TEXT,
)
from specagent.retrieval.exceptions import ConfigurationError, VisionError
from specagent.retrieval.groq_rate_limiter import _get_rate_limiter

logger = logging.getLogger(__name__)

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _wait_retry_after(retry_state: Any) -> float:
    """Tenacity wait callable that respects Groq's ``Retry-After`` header.

    For 429 responses Groq returns a ``Retry-After`` header with the exact
    number of seconds to wait before the rate-limit window resets.  Using
    that value avoids the exponential back-off loop that would otherwise keep
    hitting the API every few seconds — burning quota on calls that will all
    return 429 anyway.

    Falls back to capped exponential back-off for non-429 errors or when the
    header is absent/unparseable.
    """
    exc = retry_state.outcome.exception()
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        retry_after_raw = exc.response.headers.get("retry-after", "")
        try:
            wait = float(retry_after_raw)
            if wait > 0:
                logger.warning("Groq 429 rate limit; sleeping %.1f s (Retry-After header)", wait)
                return wait
        except (ValueError, TypeError):
            pass
    # Fallback: capped exponential (attempt 1→4 s, 2→8 s … capped at 60 s)
    return min(2.0**retry_state.attempt_number * 2.0, 60.0)


class ImageAnalysisResult(BaseModel):
    """Result of analysing a single image via the Groq vision API."""

    placeholder_name: str
    """MarkItDown placeholder that this result will replace, e.g. ``image0.png``."""

    markdown_content: str
    """Extracted content: Mermaid DAG, Markdown table, plain text, or description."""

    image_type: str
    """One of: call_flow, state_machine, block_diagram, flowchart, network_topology, table, screenshot_text, other."""

    skipped: bool = False
    """``True`` when the image was not sent to the API (size filter, unsupported type…)."""

    skip_reason: str = ""
    """Human-readable reason when ``skipped=True``."""

    prose_fallback: str = ""
    """One-sentence plain-English description of the image, always populated
    for diagram types. Used as fallback when Mermaid validation fails."""


async def analyze_image(
    image: ExtractedImage,
    api_key: str,
    model: str | None = None,
) -> ImageAnalysisResult:
    """Send an image to the Groq vision API and return structured Markdown.

    Acquires a rate-limit slot on every attempt (including retries).  Uses
    ``tenacity`` for retry on transient server errors (429, 503, 504, timeout).

    Args:
        image: :class:`~specagent.retrieval.docx_image_extractor.ExtractedImage`
            containing raw bytes and metadata.
        api_key: Groq API key.  Never logged.
        model: Groq vision model identifier.  Defaults to ``settings.vision_model``.

    Returns:
        :class:`ImageAnalysisResult` with ``markdown_content`` populated.

    Raises:
        ConfigurationError: If ``api_key`` is empty.
        VisionError: If the API call fails after all retries.
    """
    if not api_key:
        raise ConfigurationError("api_key must be non-empty to call the Groq vision API.")

    _model = model or settings.vision_model
    encoded = base64.b64encode(image.image_bytes).decode("ascii")
    data_url = f"data:{image.mime_type};base64,{encoded}"
    _headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=_wait_retry_after,
        reraise=True,
    )
    async def _call() -> ImageAnalysisResult:
        await _get_rate_limiter().acquire()
        async with httpx.AsyncClient(timeout=30.0) as client:
            body: dict = {
                "model": _model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _USER_MESSAGE_TEXT},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": _RESPONSE_JSON_SCHEMA,
                },
                "max_tokens": settings.vision_max_tokens,
                "temperature": 0.0,
            }
            response = await client.post(_GROQ_CHAT_URL, headers=_headers, json=body)
            # Fallback: if Groq rejects response_format, retry without it
            if response.status_code == 400 and "response_format" in response.text:
                logger.warning("response_format not supported by model; retrying without it")
                body = {k: v for k, v in body.items() if k != "response_format"}
                response = await client.post(_GROQ_CHAT_URL, headers=_headers, json=body)
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


async def correct_mermaid_diagram(
    image: ExtractedImage,
    prior_attempt: str,
    validation_errors: str,
    diagram_type: str,
    api_key: str,
    model: str | None = None,
) -> ImageAnalysisResult:
    """Re-submit an image with validation errors to obtain a corrected Mermaid diagram.

    Sends a two-message conversation: system prompt + user message containing
    the original image, the prior failed attempt, and the validation errors.
    The diagram_type is locked — the model cannot reclassify the image.

    Args:
        image: The original ExtractedImage (re-sent for visual context).
        prior_attempt: The Mermaid block that failed validation.
        validation_errors: Human-readable description of the failures.
        diagram_type: Locked image_type from the first analysis attempt.
        api_key: Groq API key. Never logged.
        model: Groq vision model identifier.  Defaults to ``settings.vision_model``.

    Returns:
        ImageAnalysisResult with image_type locked to diagram_type.

    Raises:
        ConfigurationError: If api_key is empty.
        VisionError: If the API call fails after retries.
    """
    if not api_key:
        raise ConfigurationError("api_key must be non-empty to call the Groq vision API.")

    _model = model or settings.vision_model
    encoded = base64.b64encode(image.image_bytes).decode("ascii")
    data_url = f"data:{image.mime_type};base64,{encoded}"

    correction_text = (
        f"This image was previously classified as '{diagram_type}'. "
        f"A Mermaid diagram was generated but failed validation with these errors:\n\n"
        f"{validation_errors}\n\n"
        f"Previous attempt:\n{prior_attempt}\n\n"
        f"Re-analyze the image and return a corrected Mermaid diagram of type "
        f"'{diagram_type}'. Return the same JSON schema as before."
    )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=_wait_retry_after,
        reraise=True,
    )
    async def _call() -> ImageAnalysisResult:
        await _get_rate_limiter().acquire()
        _headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body: dict = {
            "model": _model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": correction_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": _RESPONSE_JSON_SCHEMA,
            },
            "max_tokens": settings.vision_max_tokens,
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(_GROQ_CHAT_URL, headers=_headers, json=body)
            if response.status_code == 400 and "response_format" in response.text:
                body = {k: v for k, v in body.items() if k != "response_format"}
                response = await client.post(_GROQ_CHAT_URL, headers=_headers, json=body)
            response.raise_for_status()

        raw_content = response.json()["choices"][0]["message"]["content"]
        result = _parse_response(image.placeholder_name, raw_content)
        # Lock diagram_type — correction cannot reclassify
        return result.model_copy(update={"image_type": diagram_type})

    try:
        return await _call()
    except (ConfigurationError, VisionError):
        raise
    except Exception as exc:
        raise VisionError(
            f"Correction API failed for {image.placeholder_name!r} after retries: {exc}"
        ) from exc


def _parse_response(placeholder_name: str, raw_content: str) -> ImageAnalysisResult:
    """Parse the model's text response into an ImageAnalysisResult.

    Falls back to ``image_type='other'`` with the raw text as content if JSON
    parsing fails or the type field is unrecognised.
    """
    try:
        data = json.loads(raw_content)
        image_type = data.get("type", "other")
        if image_type not in _KNOWN_DIAGRAM_TYPES:
            image_type = "other"
        content = str(data.get("content", raw_content))
        prose_fallback = str(data.get("prose_fallback", ""))

        # For diagram types: ensure the Mermaid block uses the correct header keyword
        if image_type in _MERMAID_SUBTYPE:
            content = _fix_mermaid_header(content, _MERMAID_SUBTYPE[image_type])

    except (json.JSONDecodeError, AttributeError):
        logger.warning(
            "Non-JSON response from vision API for %r; treating as 'other'",
            placeholder_name,
        )
        image_type = "other"
        content = raw_content
        prose_fallback = ""

    return ImageAnalysisResult(
        placeholder_name=placeholder_name,
        markdown_content=content,
        image_type=image_type,
        prose_fallback=prose_fallback,
    )
