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
_WORDML_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_DRAWINGML_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_DRAWING_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_IMAGE_TYPE_SUFFIX = "/image"
_RELS_PATH = "word/_rels/document.xml.rels"
_MEDIA_PREFIX = "word/media/"
_DOCUMENT_XML_PATH = "word/document.xml"


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

    caption: str = ""
    """Caption text from a Caption-style paragraph following the image, if any."""

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
            caption_map = _extract_caption_map(zf)
            return _read_image_bytes(zf, image_rels, caption_map)
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


def _caption_for_drawing_para(
    para: ET.Element,
    next_para: ET.Element,
    embed_attr: str,
    ppr_tag: str,
    pstyle_tag: str,
    wordml_ns: str,
    r_tag: str,
    t_tag: str,
) -> tuple[str, str] | None:
    """Return (rel_id, caption_text) if para has a drawing followed by a Caption paragraph.

    Returns None if the drawing has no r:embed, next_para is not a Caption, or caption is empty.
    """
    drawing_tag = f"{{{wordml_ns}}}drawing"
    drawing = para.find(f".//{drawing_tag}")
    if drawing is None:
        return None
    blip = drawing.find(f".//*[@{embed_attr}]")
    if blip is None:
        return None
    rel_id = blip.get(embed_attr)
    if not rel_id:
        return None
    ppr = next_para.find(ppr_tag)
    if ppr is None:
        return None
    pstyle = ppr.find(pstyle_tag)
    if pstyle is None or pstyle.get(f"{{{wordml_ns}}}val") != "Caption":
        return None
    texts = [
        t.text or ""
        for r in next_para.findall(f".//{r_tag}")
        for t in r.findall(t_tag)
    ]
    caption_text = "".join(texts).strip()
    return (rel_id, caption_text) if caption_text else None


def _extract_caption_map(zf: zipfile.ZipFile) -> dict[str, str]:
    """Return a mapping of relationship-ID → caption text from word/document.xml.

    Parses Caption-style paragraphs that immediately follow drawing paragraphs.
    Returns an empty dict if the document XML is missing or malformed.
    """
    if _DOCUMENT_XML_PATH not in zf.namelist():
        return {}
    try:
        root = ET.fromstring(zf.read(_DOCUMENT_XML_PATH))
    except ET.ParseError:
        logger.warning("Failed to parse %s; captions unavailable", _DOCUMENT_XML_PATH)
        return {}

    body = root.find(f"{{{_WORDML_NS}}}body")
    if body is None:
        return {}

    p_tag = f"{{{_WORDML_NS}}}p"
    ppr_tag = f"{{{_WORDML_NS}}}pPr"
    pstyle_tag = f"{{{_WORDML_NS}}}pStyle"
    r_tag = f"{{{_WORDML_NS}}}r"
    t_tag = f"{{{_WORDML_NS}}}t"
    embed_attr = f"{{{_DRAWING_REL_NS}}}embed"

    paragraphs = [child for child in body if child.tag == p_tag]
    caption_map: dict[str, str] = {}
    for idx, para in enumerate(paragraphs[:-1]):
        result = _caption_for_drawing_para(
            para, paragraphs[idx + 1], embed_attr, ppr_tag, pstyle_tag, _WORDML_NS, r_tag, t_tag
        )
        if result is not None:
            rel_id, caption_text = result
            caption_map[rel_id] = caption_text

    return caption_map


def _read_image_bytes(
    zf: zipfile.ZipFile,
    image_rels: list[tuple[str, str]],
    caption_map: dict[str, str] | None = None,
) -> list[ExtractedImage]:
    """Build an ExtractedImage list from sorted relationship pairs and ZIP contents."""
    if caption_map is None:
        caption_map = {}
    zip_names = set(zf.namelist())
    results: list[ExtractedImage] = []

    for index, (rel_id, media_filename) in enumerate(image_rels):
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
                caption=caption_map.get(rel_id, ""),
            )
        )

    return results


def _rel_id_sort_key(rel_id: str) -> int:
    """Return the numeric suffix of a relationship Id for sorting.

    ``rId3`` → 3.  Ids with no trailing digits fall back to 0.
    """
    match = re.search(r"\d+$", rel_id)
    return int(match.group()) if match else 0
