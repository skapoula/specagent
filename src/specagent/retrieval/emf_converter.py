"""Rasterize EMF/WMF image bytes to JPEG using PyMuPDF."""

import logging

from specagent.retrieval.exceptions import IngestionError

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150

# MIME types that identify Windows metafile formats handled by this module.
# Includes both IANA-registered and commonly-emitted variants.
EMF_MIME_TYPES: frozenset[str] = frozenset(
    {
        "image/x-emf",
        "image/emf",
        "image/x-wmf",
        "image/wmf",
    }
)


def convert_emf_to_jpeg(emf_bytes: bytes, filetype: str = "emf", dpi: int = _DEFAULT_DPI) -> bytes:
    """Rasterize EMF or WMF image bytes to JPEG using PyMuPDF.

    Args:
        emf_bytes: Raw bytes of an EMF or WMF file.
        filetype: PyMuPDF filetype hint — ``"emf"`` or ``"wmf"``.
        dpi: Rasterization resolution. 150 dpi is sufficient for vision-API
            analysis; use 300 for print-quality output.

    Returns:
        JPEG bytes of the rasterized first page.

    Raises:
        IngestionError: If PyMuPDF cannot open or render the metafile data.
    """
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415 — deferred to keep startup lean
    except ImportError as exc:
        raise IngestionError("PyMuPDF is not installed; cannot convert EMF/WMF images") from exc

    try:
        doc = fitz.open(stream=emf_bytes, filetype=filetype)
    except Exception as exc:
        raise IngestionError(f"EMF rasterization failed: {exc}") from exc

    try:
        page = doc[0]
        pixmap = page.get_pixmap(dpi=dpi)
        jpeg_bytes: bytes = pixmap.tobytes("jpeg")
        return jpeg_bytes
    except Exception as exc:
        raise IngestionError(f"EMF rasterization failed: {exc}") from exc
    finally:
        doc.close()
