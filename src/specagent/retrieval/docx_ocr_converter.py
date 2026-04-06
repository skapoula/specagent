"""Two-pass OCR-enhanced .docx → Markdown converter."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from specagent.config import settings
from specagent.retrieval.converter import convert
from specagent.retrieval.docx_image_extractor import ExtractedImage, extract_images
from specagent.retrieval.emf_converter import EMF_MIME_TYPES, convert_emf_to_jpeg
from specagent.retrieval.exceptions import IngestionError, UnsupportedFormatError, VisionError
from specagent.retrieval.groq_vision_client import (
    ImageAnalysisResult,
    analyze_image,
    correct_mermaid_diagram,
)
from specagent.retrieval.mermaid_validator import validate_mermaid

logger = logging.getLogger(__name__)

# Matches MarkItDown image placeholders: ![image](imageN.ext) or ![](imageN.ext)
_IMAGE_PLACEHOLDER_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# MIME types accepted by the Groq vision API.
# Windows vector formats (EMF, WMF) and other non-web-native types are excluded —
# the API returns a 400 for them, which is non-retryable and wastes quota.
_VISION_SUPPORTED_MIME_TYPES = frozenset([
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
])

_DIAGRAM_TYPES_REQUIRING_VALIDATION = frozenset([
    "call_flow",
    "state_machine",
    "block_diagram",
    "flowchart",
    "network_topology",
])

# Diagram types that are stored as DAGs (signal/call-flow diagrams only).
_DAG_DIAGRAM_TYPES = frozenset(["call_flow"])


@dataclass
class ExtractedDiagram:
    """A call-flow or signal-flow diagram extracted from a .docx file.

    Produced by :func:`convert_docx_with_ocr` and consumed by the ingestor
    to store the diagram as a DAG in Memgraph.
    """

    image_type: str
    """Diagram type, e.g. ``call_flow``."""

    mermaid_content: str
    """Validated Mermaid block for this diagram."""

    prose_description: str
    """One-sentence plain-English description (from ``prose_fallback``)."""

    caption: str
    """Figure caption text extracted from the .docx file (may be empty)."""

    placeholder_name: str
    """MarkItDown placeholder, e.g. ``image0.png`` (used as fallback ID)."""


def _prose_fallback_result(
    result: ImageAnalysisResult, placeholder_name: str
) -> ImageAnalysisResult:
    """Return result with markdown_content replaced by prose_fallback or a marker."""
    fallback = (
        result.prose_fallback
        or f"_[Diagram: {placeholder_name} — validation failed]_"
    )
    return result.model_copy(update={"markdown_content": fallback})


async def _apply_mermaid_validation(
    result: ImageAnalysisResult,
    image: ExtractedImage,
    api_key: str,
) -> ImageAnalysisResult:
    """Validate and optionally correct Mermaid content; fall back to prose on failure."""
    if result.image_type not in _DIAGRAM_TYPES_REQUIRING_VALIDATION:
        return result

    valid, reason = validate_mermaid(result.markdown_content)
    if valid:
        return result

    logger.warning(
        "Mermaid validation failed for %s: %s — requesting correction from VLM",
        image.placeholder_name,
        reason,
    )

    try:
        corrected = await correct_mermaid_diagram(
            image=image,
            prior_attempt=result.markdown_content,
            validation_errors=reason,
            diagram_type=result.image_type,
            api_key=api_key,
        )
    except VisionError as exc:
        logger.warning("Correction API call failed for %s: %s — using prose fallback",
                       image.placeholder_name, exc)
        return _prose_fallback_result(result, image.placeholder_name)

    valid2, reason2 = validate_mermaid(corrected.markdown_content)
    if valid2:
        return corrected

    logger.warning("Corrected Mermaid still invalid for %s: %s — using prose fallback",
                   image.placeholder_name, reason2)
    return _prose_fallback_result(corrected, image.placeholder_name)


async def convert_docx_with_ocr(
    docx_path: Path,
    api_key: str,
) -> tuple[str, list[ExtractedDiagram]]:
    """Run a two-pass OCR-enhanced conversion for a single ``.docx`` file.

    **Pass 1** — MarkItDown converts the document to Markdown, leaving image
    placeholders such as ``![image](image0.png)``.

    **Pass 2** — Images are extracted from the ZIP archive and analysed via
    the Groq vision API.  Each result is stitched back into the Markdown at
    the corresponding placeholder position.

    Size filtering:
    * Images smaller than ``settings.vision_min_image_bytes`` (default 10 KB)
      are skipped as logos / decorative elements.
    * Images larger than ``settings.vision_max_image_bytes`` (default 20 MB)
      are skipped to avoid oversized payloads.

    Args:
        docx_path: Resolved path to the ``.docx`` file.
        api_key: Groq API key for vision calls.

    Returns:
        Tuple of ``(markdown, diagrams)`` where:
        - ``markdown`` is the enriched Markdown string with placeholders replaced.
        - ``diagrams`` is a list of :class:`ExtractedDiagram` for call-flow
          diagrams found in the document (used by the ingestor for DAG storage).

    Raises:
        UnsupportedFormatError: If ``docx_path`` suffix is not ``.docx``.
        IngestionError: If Pass 1 (MarkItDown) produces empty output.
    """
    if docx_path.suffix.lower() != ".docx":
        raise UnsupportedFormatError(
            f"convert_docx_with_ocr expects a .docx file; got {docx_path.suffix!r}"
        )

    # ── Pass 1: MarkItDown → Markdown with placeholders ────────────────────
    raw_markdown = await asyncio.to_thread(convert, docx_path)
    if not raw_markdown.strip():
        raise IngestionError(f"Pass 1 produced empty output for {docx_path.name!r}")

    # ── Pass 2a: Extract images from ZIP ───────────────────────────────────
    try:
        images: list[ExtractedImage] = await asyncio.to_thread(extract_images, docx_path)
    except IngestionError:
        logger.warning("Image extraction failed for %s; using Pass 1 output only", docx_path.name)
        return raw_markdown, []

    if not images:
        logger.debug("No images in %s; skipping vision pass", docx_path.name)
        return raw_markdown, []

    # ── Pass 2b: Analyse images (sequential — rate-limiter paces calls) ────
    # Results are keyed by the image's sequential index in the extracted list.
    # This matches MarkItDown's placeholder order regardless of the URL format
    # it uses (filename-based or data-URI-based) — see _stitch() for details.
    results: dict[int, ImageAnalysisResult] = {}
    for idx, raw_image in enumerate(images):
        image = await _prepare_image(raw_image, docx_path.name)
        if image is None:
            continue
        try:
            result = await analyze_image(image, api_key=api_key)
            result = await _apply_mermaid_validation(result, image, api_key)
            results[idx] = result
            logger.info(
                "Analysed %s (index %d) in %s: type=%s",
                image.placeholder_name,
                idx,
                docx_path.name,
                result.image_type,
            )
        except VisionError as exc:
            logger.warning(
                "Vision analysis failed for %s (index %d) in %s: %s — keeping placeholder",
                image.placeholder_name,
                idx,
                docx_path.name,
                exc,
            )

    captions = {i: images[i].caption for i in range(len(images)) if images[i].caption}

    # ── Collect call-flow diagrams for DAG storage ─────────────────────────
    diagrams: list[ExtractedDiagram] = [
        ExtractedDiagram(
            image_type=result.image_type,
            mermaid_content=result.markdown_content,
            prose_description=result.prose_fallback,
            caption=captions.get(idx, ""),
            placeholder_name=result.placeholder_name,
        )
        for idx, result in results.items()
        if result.image_type in _DAG_DIAGRAM_TYPES and not result.skipped
    ]

    return _stitch(raw_markdown, results, captions), diagrams


async def _prepare_image(raw_image: ExtractedImage, doc_name: str) -> ExtractedImage | None:
    """Apply size filtering and EMF conversion, returning the image ready for vision API.

    EMF/WMF images are converted to JPEG first because their raw vector bytes
    are compact regardless of visual complexity — size filtering must run on
    the JPEG output.  Raster images are size-filtered on their raw bytes.

    Returns ``None`` when the image should be skipped.
    """
    if raw_image.mime_type in EMF_MIME_TYPES:
        image = await _convert_emf_image(raw_image, doc_name)
        if image is None:
            return None
        if len(image.image_bytes) > settings.vision_max_image_bytes:
            logger.warning(
                "Skipping %s in %s: %d bytes > max %d (post-conversion)",
                image.placeholder_name,
                doc_name,
                len(image.image_bytes),
                settings.vision_max_image_bytes,
            )
            return None
    else:
        img_size = len(raw_image.image_bytes)
        if img_size < settings.vision_min_image_bytes:
            logger.debug(
                "Skipping %s in %s: %d bytes < min %d (likely logo/icon)",
                raw_image.placeholder_name,
                doc_name,
                img_size,
                settings.vision_min_image_bytes,
            )
            return None
        if img_size > settings.vision_max_image_bytes:
            logger.warning(
                "Skipping %s in %s: %d bytes > max %d",
                raw_image.placeholder_name,
                doc_name,
                img_size,
                settings.vision_max_image_bytes,
            )
            return None
        image = raw_image

    if image.mime_type not in _VISION_SUPPORTED_MIME_TYPES:
        logger.debug(
            "Skipping %s in %s: mime_type %r not supported by vision API",
            image.placeholder_name,
            doc_name,
            image.mime_type,
        )
        return None
    return image


async def _convert_emf_image(image: ExtractedImage, doc_name: str) -> ExtractedImage | None:
    """Rasterize an EMF/WMF ExtractedImage to JPEG via PyMuPDF.

    Returns a new ExtractedImage with ``image_bytes`` replaced by JPEG bytes
    and ``mime_type`` set to ``"image/jpeg"``, or ``None`` when conversion
    fails (in which case the caller should skip this image).
    """
    filetype = "wmf" if image.mime_type in {"image/wmf", "image/x-wmf"} else "emf"
    try:
        jpeg_bytes = await asyncio.to_thread(convert_emf_to_jpeg, image.image_bytes, filetype)
        logger.debug(
            "Converted %s EMF→JPEG for %s (%d → %d bytes)",
            image.placeholder_name,
            doc_name,
            len(image.image_bytes),
            len(jpeg_bytes),
        )
        return image.model_copy(update={"image_bytes": jpeg_bytes, "mime_type": "image/jpeg"})
    except IngestionError as exc:
        logger.warning(
            "EMF conversion failed for %s in %s: %s — keeping placeholder",
            image.placeholder_name,
            doc_name,
            exc,
        )
        return None


def _stitch(
    markdown: str,
    results: dict[int, ImageAnalysisResult],
    captions: dict[int, str] | None = None,
) -> str:
    """Replace image placeholders with analysed Markdown content.

    Matching is done by sequential counter. Optionally prepends a
    ``**Figure: <caption>**`` heading when a caption is available.

    Args:
        markdown: Raw Markdown with ``![alt](url)`` placeholders.
        results: Mapping of sequential image index → analysis result.
        captions: Optional mapping of sequential image index → caption text.

    Returns:
        Markdown with analysed placeholders replaced by their content.
    """
    if captions is None:
        captions = {}
    counter = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal counter
        idx = counter
        counter += 1
        if idx in results and not results[idx].skipped:
            content = results[idx].markdown_content
            caption = captions.get(idx, "")
            if caption:
                return f"\n**Figure: {caption}**\n\n{content}"
            return content
        return match.group(0)

    return _IMAGE_PLACEHOLDER_RE.sub(_replace, markdown)
