"""Extract embedded images from .docx files (which are ZIP archives)."""

from __future__ import annotations

import logging
import mimetypes
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from pydantic import BaseModel

from specagent.retrieval.exceptions import IngestionError

logger = logging.getLogger(__name__)

# Register Windows metafile MIME types — Python's mimetypes module does not
# know these on all platforms, causing EMF/WMF files to get a None MIME type.
mimetypes.add_type("image/x-emf", ".emf")
mimetypes.add_type("image/x-wmf", ".wmf")

_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_IMAGE_TYPE_SUFFIX = "/image"
_RELS_PATH = "word/_rels/document.xml.rels"
_MEDIA_PREFIX = "word/media/"


class ExtractedImage(BaseModel):
    """A single image extracted from a .docx ZIP archive."""

    placeholder_name: str
    """MarkItDown placeholder name, e.g. ``image0.png`` (sequential, 0-based)."""

    media_filename: str
    """Actual filename inside ``word/media/``, e.g. ``image1.jpeg``."""

    image_bytes: bytes
    """Raw bytes of the image file."""

    mime_type: str
    """MIME type inferred from media_filename, e.g. ``image/png``."""

    model_config = {"arbitrary_types_allowed": True}


def extract_images(docx_path: Path) -> list[ExtractedImage]:
    """Extract all embedded images from a .docx file.

    Opens the file as a ZIP archive, reads ``word/_rels/document.xml.rels``
    to determine image relationship order (which drives MarkItDown's
    placeholder numbering), then reads raw bytes from ``word/media/``.

    Args:
        docx_path: Path to the ``.docx`` file to inspect.

    Returns:
        List of :class:`ExtractedImage` in relationship-index order
        (matching MarkItDown's ``image0.png``, ``image1.png``, ... numbering).
        Returns an empty list when the document contains no images or when
        the relationships file is absent.

    Raises:
        IngestionError: If the file cannot be opened as a valid ZIP archive.
    """
    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            image_rels = _parse_image_relationships(zf)
            return _read_image_bytes(zf, image_rels)
    except zipfile.BadZipFile as exc:
        raise IngestionError(
            f"Cannot open {docx_path.name!r} as a ZIP archive — "
            "the file may be corrupt or is not a valid .docx."
        ) from exc
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Unexpected error extracting images from {docx_path.name!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_image_relationships(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return (rel_id, media_filename) pairs for image relationships, sorted by Id.

    Returns an empty list when ``word/_rels/document.xml.rels`` is absent.
    """
    names = zf.namelist()
    if _RELS_PATH not in names:
        return []

    xml_text = zf.read(_RELS_PATH)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("Could not parse %s: %s — skipping image extraction", _RELS_PATH, exc)
        return []

    image_rels: list[tuple[str, str]] = []
    for rel in root.findall(f"{{{_REL_NS}}}Relationship"):
        rel_type = rel.get("Type", "")
        if not rel_type.endswith(_IMAGE_TYPE_SUFFIX):
            continue
        rel_id = rel.get("Id", "")
        target = rel.get("Target", "")
        # Target is relative to word/, e.g. "media/image1.png"
        media_filename = Path(target).name
        image_rels.append((rel_id, media_filename))

    # Sort by numeric suffix of Id (rId1, rId2, …) for stable ordering
    image_rels.sort(key=lambda pair: _rel_id_sort_key(pair[0]))
    return image_rels


def _read_image_bytes(
    zf: zipfile.ZipFile,
    image_rels: list[tuple[str, str]],
) -> list[ExtractedImage]:
    """Build an ExtractedImage list from sorted relationship pairs and ZIP contents."""
    zip_names = set(zf.namelist())
    results: list[ExtractedImage] = []

    for index, (_, media_filename) in enumerate(image_rels):
        zip_entry = f"{_MEDIA_PREFIX}{media_filename}"
        if zip_entry not in zip_names:
            logger.warning(
                "Relationship references %r but entry %r is absent from ZIP — skipping",
                media_filename,
                zip_entry,
            )
            continue

        img_bytes = zf.read(zip_entry)
        mime_type, _ = mimetypes.guess_type(media_filename)
        if mime_type is None:
            mime_type = "application/octet-stream"

        results.append(
            ExtractedImage(
                placeholder_name=f"image{index}.png",
                media_filename=media_filename,
                image_bytes=img_bytes,
                mime_type=mime_type,
            )
        )

    return results


def _rel_id_sort_key(rel_id: str) -> int:
    """Return the numeric suffix of a relationship Id for sorting.

    ``rId3`` → 3.  Ids with no trailing digits fall back to 0.
    """
    match = re.search(r"\d+$", rel_id)
    return int(match.group()) if match else 0
