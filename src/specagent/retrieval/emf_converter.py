"""Rasterize EMF/WMF image bytes to JPEG using Inkscape."""

import io
import logging
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from specagent.retrieval.exceptions import IngestionError

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150
_INKSCAPE_BIN = "inkscape"
_INKSCAPE_TIMEOUT = 60  # seconds; generous for large/complex EMF files


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
    """Rasterize EMF or WMF image bytes to JPEG using Inkscape.

    Writes the input bytes to a temporary file, invokes Inkscape to export a
    PNG, then converts the PNG to JPEG via Pillow.

    Args:
        emf_bytes: Raw bytes of an EMF or WMF file.
        filetype: File extension hint — ``"emf"`` or ``"wmf"``.
        dpi: Rasterization resolution. 150 dpi is sufficient for vision-API
            analysis; use 300 for print-quality output.

    Returns:
        JPEG bytes of the rasterized image.

    Raises:
        IngestionError: If Inkscape is not installed, fails to convert,
            exceeds the timeout, or Pillow cannot encode the result.
    """
    suffix = f".{filetype.lstrip('.')}"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_in = Path(tmpdir) / f"input{suffix}"
            tmp_out = Path(tmpdir) / "output.png"
            tmp_in.write_bytes(emf_bytes)

            try:
                result = subprocess.run(
                    [
                        _INKSCAPE_BIN,
                        "--export-type=png",
                        f"--export-filename={tmp_out}",
                        f"--export-dpi={dpi}",
                        str(tmp_in),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=_INKSCAPE_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                raise IngestionError(
                    f"Inkscape timed out after {_INKSCAPE_TIMEOUT}s converting {filetype} image"
                ) from exc
            if result.returncode != 0 or not tmp_out.exists():
                raise IngestionError(f"Inkscape exited {result.returncode}: {result.stderr[:200]}")

            png_bytes = tmp_out.read_bytes()

    except IngestionError:
        raise
    except FileNotFoundError as exc:
        raise IngestionError(
            "Inkscape is not installed or not on PATH; cannot convert EMF/WMF images"
        ) from exc
    except Exception as exc:
        raise IngestionError(f"EMF rasterization failed: {exc}") from exc

    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as exc:
        raise IngestionError(f"EMF rasterization failed: JPEG encoding error: {exc}") from exc
