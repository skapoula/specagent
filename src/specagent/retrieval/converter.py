"""
File format converter using MarkItDown.

Converts local files (PDF, DOCX, PPTX, XLSX, etc.) to markdown strings
for ingestion into the chunking pipeline.
"""

from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".html", ".htm", ".csv", ".epub", ".zip"}
)

ALL_SUPPORTED_EXTENSIONS: frozenset[str] = SUPPORTED_EXTENSIONS | frozenset({".md"})

_converter_instance = None


def _get_converter():  # type: ignore[no-untyped-def]
    """Return a lazy singleton MarkItDown instance."""
    global _converter_instance
    if _converter_instance is None:
        from markitdown import MarkItDown

        _converter_instance = MarkItDown()
    return _converter_instance


def convert_to_markdown(file_path: Path) -> str:
    """
    Convert a file to a markdown string.

    For .md files, reads the content directly. For all other supported
    formats, delegates to MarkItDown for conversion.

    Args:
        file_path: Path to the file to convert.

    Returns:
        Markdown string representation of the file content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix not in ALL_SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            f"Supported: {sorted(ALL_SUPPORTED_EXTENSIONS)}"
        )

    if suffix == ".md":
        return file_path.read_text(encoding="utf-8")

    converter = _get_converter()
    result = converter.convert(str(file_path))
    return result.text_content
