"""Convert local files to Markdown text via MarkItDown."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from specagent.retrieval.exceptions import IngestionError, UnsupportedFormatError

if TYPE_CHECKING:
    from markitdown import MarkItDown

logger = logging.getLogger(__name__)

# Extensions MarkItDown[all] can handle. Checked before calling to give a clear error.
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".xml",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".mp3",
    ".wav",
    ".ogg",
    ".m4a",
    ".zip",
}

_md: "MarkItDown | None" = None


def _get_markitdown() -> "MarkItDown":
    """Return the MarkItDown singleton, initialising on first call."""
    global _md  # noqa: PLW0603
    if _md is None:
        from markitdown import MarkItDown  # noqa: PLC0415 — lazy import keeps MarkItDown optional

        _md = MarkItDown()
    return _md


async def convert_docx_ocr(
    source: Path,
    api_key: str,
) -> "tuple[str, list]":
    """Convert a .docx file to Markdown using the two-pass OCR pipeline.

    Delegates to :func:`~specagent.retrieval.docx_ocr_converter.convert_docx_with_ocr`.
    Imported here so that ``ingestor.py`` only needs to import from ``converter``.

    Args:
        source: Path to the ``.docx`` file.
        api_key: Groq API key for vision calls.

    Returns:
        Tuple of ``(markdown, diagrams)`` — see
        :func:`~specagent.retrieval.docx_ocr_converter.convert_docx_with_ocr`.

    Raises:
        UnsupportedFormatError: If source is not a .docx file.
        IngestionError: If the conversion fails entirely.
    """
    from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr  # noqa: PLC0415

    return await convert_docx_with_ocr(source, api_key=api_key)


def convert(source: Path) -> str:
    """Convert a local file to Markdown text.

    Args:
        source: Path to the local file to convert.

    Returns:
        Markdown text extracted from the file, or an empty string if the
        file contains no extractable text content.

    Raises:
        UnsupportedFormatError: If the file has no extension or an unsupported one.
        IngestionError: If MarkItDown fails to convert the file.
    """
    source = source.resolve()
    ext = source.suffix.lower()

    if ext == "":
        raise UnsupportedFormatError(
            f"No file extension detected for {source.name!r} — cannot determine format."
        )
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file extension: {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    logger.debug("Converting %s (ext=%s)", source, ext)
    try:
        result = _get_markitdown().convert(str(source))
        text = result.text_content or ""
    except Exception as exc:
        raise IngestionError(f"Failed to convert {source.name!r}: {exc}") from exc

    if not text:
        logger.warning("Converted %s produced empty text content", source)
    else:
        logger.debug("Converted %s → %d chars", source, len(text))
    return text
