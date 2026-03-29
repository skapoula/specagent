"""Unit tests for docx_image_extractor — written before implementation (TDD RED)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tests.conftest import make_docx_zip, _make_png_bytes
from specagent.retrieval.exceptions import IngestionError


@pytest.mark.unit
class TestExtractImages:
    """Tests for extract_images() and the ExtractedImage model."""

    def test_returns_empty_for_docx_with_no_images(self, docx_no_images: Path) -> None:
        """A .docx with no embedded images returns an empty list."""
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(docx_no_images)
        assert result == []

    def test_extracts_single_png_image(self, docx_one_image: Path, large_png: bytes) -> None:
        """A .docx with one PNG returns exactly one ExtractedImage."""
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(docx_one_image)
        assert len(result) == 1
        assert result[0].placeholder_name == "image0.png"
        assert result[0].media_filename == "image1.png"
        assert result[0].image_bytes == large_png
        assert result[0].mime_type == "image/png"

    def test_placeholder_names_are_sequential(self, docx_three_images: Path) -> None:
        """Three images get placeholder names image0.png, image1.png, image2.png."""
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(docx_three_images)
        assert len(result) == 3
        assert [r.placeholder_name for r in result] == [
            "image0.png",
            "image1.png",
            "image2.png",
        ]

    def test_images_returned_in_relationship_order(self, tmp_path: Path, large_png: bytes) -> None:
        """Images are ordered by relationship Id (rId1, rId2, ...), not filesystem order."""
        p = tmp_path / "ordered.docx"
        p.write_bytes(
            make_docx_zip(
                images=[
                    ("alpha.png", large_png),
                    ("beta.png", large_png),
                ]
            )
        )
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(p)
        assert result[0].media_filename == "alpha.png"
        assert result[1].media_filename == "beta.png"

    def test_mime_type_for_jpeg(self, tmp_path: Path, large_png: bytes) -> None:
        """JPEG images get mime_type image/jpeg."""
        p = tmp_path / "jpeg_doc.docx"
        p.write_bytes(make_docx_zip(images=[("photo.jpeg", large_png)]))
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(p)
        assert result[0].mime_type == "image/jpeg"

    def test_skips_missing_media_entry_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If relationship references a file absent from word/media/, skip it and log WARNING."""
        import logging

        # Build a docx where rels references image1.png but the media file is absent
        buf = io.BytesIO()
        _IMAGE_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", "<root/>")
            zf.writestr(
                "word/_rels/document.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'<Relationship Id="rId1" Type="{_IMAGE_NS}" Target="media/missing.png"/>'
                "</Relationships>",
            )
            # Intentionally NOT writing word/media/missing.png
        p = tmp_path / "missing_media.docx"
        p.write_bytes(buf.getvalue())

        from specagent.retrieval.docx_image_extractor import extract_images

        with caplog.at_level(logging.WARNING, logger="specagent.retrieval.docx_image_extractor"):
            result = extract_images(p)

        assert result == []
        assert any("missing.png" in r.message for r in caplog.records)

    def test_raises_ingestion_error_for_invalid_zip(self, tmp_path: Path) -> None:
        """A file that isn't a valid ZIP raises IngestionError."""
        p = tmp_path / "corrupt.docx"
        p.write_bytes(b"this is not a zip file at all")
        from specagent.retrieval.docx_image_extractor import extract_images

        with pytest.raises(IngestionError, match="ZIP"):
            extract_images(p)

    def test_image_bytes_content_preserved(self, tmp_path: Path) -> None:
        """Raw bytes of the extracted image match exactly what was embedded."""
        sentinel = _make_png_bytes(n_bytes=15 * 1024)
        p = tmp_path / "sentinel.docx"
        p.write_bytes(make_docx_zip(images=[("sentinel.png", sentinel)]))
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(p)
        assert result[0].image_bytes == sentinel

    def test_no_rels_file_returns_empty(self, tmp_path: Path) -> None:
        """A .docx ZIP that has no _rels/document.xml.rels returns empty list."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", "<root/>")
            # Deliberately omit word/_rels/document.xml.rels
        p = tmp_path / "no_rels.docx"
        p.write_bytes(buf.getvalue())
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(p)
        assert result == []

    def test_emf_gets_emf_mime_type(self, tmp_path: Path, large_png: bytes) -> None:
        """A .emf media file gets mime_type 'image/x-emf' (not 'image/jpeg')."""
        p = tmp_path / "emf_doc.docx"
        p.write_bytes(make_docx_zip(images=[("diagram.emf", large_png)]))
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(p)
        assert len(result) == 1
        assert result[0].mime_type == "image/x-emf"

    def test_wmf_gets_wmf_mime_type(self, tmp_path: Path, large_png: bytes) -> None:
        """A .wmf media file gets mime_type 'image/x-wmf'."""
        p = tmp_path / "wmf_doc.docx"
        p.write_bytes(make_docx_zip(images=[("chart.wmf", large_png)]))
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(p)
        assert len(result) == 1
        assert result[0].mime_type == "image/x-wmf"

    def test_unknown_extension_defaults_to_octet_stream(self, tmp_path: Path, large_png: bytes) -> None:
        """An unrecognised extension falls back to 'application/octet-stream'."""
        p = tmp_path / "unknown_doc.docx"
        p.write_bytes(make_docx_zip(images=[("figure.zzunknownzz", large_png)]))
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(p)
        assert len(result) == 1
        assert result[0].mime_type == "application/octet-stream"


@pytest.mark.unit
class TestCaptionExtraction:
    """Tests for caption metadata populated on ExtractedImage."""

    def test_caption_extracted_for_single_image(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """Caption text from a Caption-style paragraph populates ExtractedImage.caption."""
        from tests.conftest import make_docx_zip_with_caption
        from specagent.retrieval.docx_image_extractor import extract_images

        p = tmp_path / "captioned.docx"
        p.write_bytes(
            make_docx_zip_with_caption(
                image_filename="image1.png",
                image_bytes=large_png,
                caption_text="Figure 3: Network Architecture",
            )
        )
        result = extract_images(p)
        assert len(result) == 1
        assert result[0].caption == "Figure 3: Network Architecture"

    def test_no_caption_returns_empty_string(
        self, docx_one_image: Path
    ) -> None:
        """Image in a docx with no Caption paragraph gets caption=''."""
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(docx_one_image)
        assert len(result) == 1
        assert result[0].caption == ""

    def test_malformed_document_xml_returns_empty_caption(
        self, tmp_path: Path, large_png: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unparseable word/document.xml logs WARNING and leaves caption=''."""
        import logging

        buf = io.BytesIO()
        _IMAGE_NS = (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        )
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", "<<<not xml>>>")
            zf.writestr(
                "word/_rels/document.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'<Relationship Id="rId1" Type="{_IMAGE_NS}" Target="media/image1.png"/>'
                "</Relationships>",
            )
            zf.writestr("word/media/image1.png", large_png)
        p = tmp_path / "bad_xml.docx"
        p.write_bytes(buf.getvalue())

        from specagent.retrieval.docx_image_extractor import extract_images

        with caplog.at_level(logging.WARNING, logger="specagent.retrieval.docx_image_extractor"):
            result = extract_images(p)

        assert result[0].caption == ""
        assert any("captions unavailable" in r.message for r in caplog.records)

    def test_caption_map_ignores_non_caption_paragraphs(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """Paragraphs without Caption style are not mistaken for captions."""
        _W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        _R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        _A = "http://schemas.openxmlformats.org/drawingml/2006/main"
        _PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
        _IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        document_xml = (
            f'<w:document xmlns:w="{_W}" xmlns:r="{_R}"><w:body>'
            f'<w:p><w:r><w:drawing><a:blip xmlns:a="{_A}" r:embed="rId1"/></w:drawing></w:r></w:p>'
            "<w:p><w:pPr><w:pStyle w:val=\"Normal\"/></w:pPr>"
            "<w:r><w:t>Not a caption</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", document_xml)
            zf.writestr(
                "word/_rels/document.xml.rels",
                f'<Relationships xmlns="{_PKG}">'
                f'<Relationship Id="rId1" Type="{_IMAGE_REL}" Target="media/image1.png"/>'
                "</Relationships>",
            )
            zf.writestr("word/media/image1.png", large_png)
        p = tmp_path / "no_caption_style.docx"
        p.write_bytes(buf.getvalue())

        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(p)
        assert result[0].caption == ""
