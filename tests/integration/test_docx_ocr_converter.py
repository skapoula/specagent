"""Integration tests for docx_ocr_converter — written before implementation (TDD RED)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from specagent.retrieval.exceptions import IngestionError, UnsupportedFormatError


def _make_result(placeholder: str, content: str, image_type: str = "other"):
    """Convenience builder for ImageAnalysisResult."""
    from specagent.retrieval.groq_vision_client import ImageAnalysisResult

    return ImageAnalysisResult(
        placeholder_name=placeholder,
        markdown_content=content,
        image_type=image_type,
    )


def _make_skipped_result(placeholder: str, reason: str = "too small"):
    """Build a skipped ImageAnalysisResult."""
    from specagent.retrieval.groq_vision_client import ImageAnalysisResult

    return ImageAnalysisResult(
        placeholder_name=placeholder,
        markdown_content="",
        image_type="other",
        skipped=True,
        skip_reason=reason,
    )


@pytest.mark.integration
class TestConvertDocxWithOcr:
    """Tests for convert_docx_with_ocr()."""

    async def test_raises_for_non_docx_extension(self, tmp_path: Path) -> None:
        """UnsupportedFormatError raised when path is not a .docx file."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        p = tmp_path / "spec.pdf"
        p.write_bytes(b"%PDF-1.4")
        with pytest.raises(UnsupportedFormatError, match=".pdf"):
            await convert_docx_with_ocr(p, api_key="key")

    async def test_pass1_only_when_no_images(self, docx_no_images: Path) -> None:
        """When the docx has no images, output equals the Pass 1 markdown unchanged."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        with patch(
            "specagent.retrieval.docx_ocr_converter.convert",
            return_value="# Title\n\nNo images here.",
        ):
            result = await convert_docx_with_ocr(docx_no_images, api_key="key")

        assert result == "# Title\n\nNo images here."

    async def test_placeholder_replaced_with_mermaid(self, docx_one_image: Path) -> None:
        """Call flow diagram placeholder is replaced with Mermaid block."""
        mermaid = "```mermaid\ngraph TD\n  UE-->gNB\n```"
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Before\n\n![image](image0.png)\n\nAfter",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=_make_result("image0.png", mermaid, "call_flow_diagram")),
            ),
        ):
            result = await convert_docx_with_ocr(docx_one_image, api_key="key")

        assert "mermaid" in result
        assert "![image](image0.png)" not in result
        assert "Before" in result
        assert "After" in result

    async def test_small_image_skipped_below_threshold(
        self, tmp_path: Path, small_png: bytes
    ) -> None:
        """Images under vision_min_image_bytes are not sent to the API."""
        from tests.conftest import make_docx_zip
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        p = tmp_path / "tiny_logo.docx"
        p.write_bytes(make_docx_zip(images=[("logo.png", small_png)]))

        analyze_mock = AsyncMock()
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Header\n\n![image](image0.png)\n\nFooter",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.settings.vision_min_image_bytes",
                10 * 1024,  # 10 KB threshold
            ),
        ):
            result = await convert_docx_with_ocr(p, api_key="key")

        # analyze_image must NOT have been called for the tiny image
        analyze_mock.assert_not_called()
        # Original placeholder preserved
        assert "![image](image0.png)" in result

    async def test_emf_image_converted_and_sent_to_vision_api(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """EMF images are rasterized to JPEG and passed to the vision API."""
        from tests.conftest import make_docx_zip
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        emf_image = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="image1.emf",
            image_bytes=large_png,
            mime_type="image/x-emf",
        )
        fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 200

        analyze_mock = AsyncMock(
            return_value=_make_result("image0.png", "EMF diagram content", "call_flow_diagram")
        )
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Diagram\n\n![image](image0.png)\n\nCaption",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[emf_image],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.convert_emf_to_jpeg",
                return_value=fake_jpeg,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
        ):
            p = tmp_path / "emf_doc.docx"
            p.write_bytes(make_docx_zip())
            result = await convert_docx_with_ocr(p, api_key="key")

        analyze_mock.assert_called_once()
        assert "EMF diagram content" in result

    async def test_wmf_image_converted_and_sent_to_vision_api(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """WMF images are rasterized to JPEG and passed to the vision API."""
        from tests.conftest import make_docx_zip
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        wmf_image = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="image1.wmf",
            image_bytes=large_png,
            mime_type="image/x-wmf",
        )
        fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 200

        analyze_mock = AsyncMock(
            return_value=_make_result("image0.png", "WMF chart content", "table_figure")
        )
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[wmf_image],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.convert_emf_to_jpeg",
                return_value=fake_jpeg,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
        ):
            p = tmp_path / "wmf_doc.docx"
            p.write_bytes(make_docx_zip())
            result = await convert_docx_with_ocr(p, api_key="key")

        analyze_mock.assert_called_once()
        assert "WMF chart content" in result

    async def test_emf_conversion_failure_preserves_placeholder(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """When EMF→JPEG conversion fails, the placeholder is preserved and vision API not called."""
        from tests.conftest import make_docx_zip
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr
        from specagent.retrieval.exceptions import IngestionError

        emf_image = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="image1.emf",
            image_bytes=large_png,
            mime_type="image/x-emf",
        )

        analyze_mock = AsyncMock()
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Diagram\n\n![image](image0.png)\n\nCaption",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[emf_image],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.convert_emf_to_jpeg",
                side_effect=IngestionError("bad EMF data"),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
        ):
            p = tmp_path / "bad_emf.docx"
            p.write_bytes(make_docx_zip())
            result = await convert_docx_with_ocr(p, api_key="key")

        analyze_mock.assert_not_called()
        assert "![image](image0.png)" in result

    async def test_supported_png_mime_type_not_skipped(
        self, docx_one_image: Path
    ) -> None:
        """PNG images (web-native) are passed through to the vision API."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        analyze_mock = AsyncMock(
            return_value=_make_result("image0.png", "extracted text", "screenshot_text")
        )
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
        ):
            await convert_docx_with_ocr(docx_one_image, api_key="key")

        analyze_mock.assert_called_once()

    async def test_large_image_above_max_threshold_skipped(
        self, tmp_path: Path
    ) -> None:
        """Images above vision_max_image_bytes are skipped and placeholder preserved."""
        from tests.conftest import _make_png_bytes, make_docx_zip
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        huge = _make_png_bytes(n_bytes=25 * 1024 * 1024)  # 25 MB
        p = tmp_path / "huge.docx"
        p.write_bytes(make_docx_zip(images=[("huge.png", huge)]))

        analyze_mock = AsyncMock()
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.settings.vision_max_image_bytes",
                20 * 1024 * 1024,  # 20 MB limit
            ),
        ):
            result = await convert_docx_with_ocr(p, api_key="key")

        analyze_mock.assert_not_called()
        assert "![image](image0.png)" in result

    async def test_vision_error_falls_back_to_placeholder(
        self, docx_one_image: Path
    ) -> None:
        """A VisionError for one image keeps its original placeholder."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr
        from specagent.retrieval.exceptions import VisionError

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Text\n\n![image](image0.png)\n\nMore",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(side_effect=VisionError("API down")),
            ),
        ):
            result = await convert_docx_with_ocr(docx_one_image, api_key="key")

        assert "![image](image0.png)" in result
        assert "Text" in result

    async def test_raises_ingestion_error_if_pass1_empty(
        self, docx_no_images: Path
    ) -> None:
        """IngestionError raised when MarkItDown returns empty Markdown."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        with patch("specagent.retrieval.docx_ocr_converter.convert", return_value="   "):
            with pytest.raises(IngestionError, match="empty"):
                await convert_docx_with_ocr(docx_no_images, api_key="key")

    async def test_three_images_all_replaced(self, docx_three_images: Path) -> None:
        """All three placeholders in a document are replaced."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        md = (
            "# Doc\n"
            "![image](image0.png)\n"
            "Middle text\n"
            "![image](image1.png)\n"
            "More text\n"
            "![image](image2.png)\n"
        )

        async def fake_analyze(image, *, api_key, model=None):
            return _make_result(image.placeholder_name, f"[{image.placeholder_name}]")

        with (
            patch("specagent.retrieval.docx_ocr_converter.convert", return_value=md),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                side_effect=fake_analyze,
            ),
        ):
            result = await convert_docx_with_ocr(docx_three_images, api_key="key")

        assert "[image0.png]" in result
        assert "[image1.png]" in result
        assert "[image2.png]" in result
        assert "![image](" not in result  # no original placeholders remain

    async def test_images_analyzed_sequentially_in_order(
        self, docx_three_images: Path
    ) -> None:
        """analyze_image is called in image0, image1, image2 order."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        call_order: list[str] = []

        async def recording_analyze(image, *, api_key, model=None):
            call_order.append(image.placeholder_name)
            return _make_result(image.placeholder_name, "x")

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value=(
                    "![image](image0.png)\n"
                    "![image](image1.png)\n"
                    "![image](image2.png)\n"
                ),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                side_effect=recording_analyze,
            ),
        ):
            await convert_docx_with_ocr(docx_three_images, api_key="key")

        assert call_order == ["image0.png", "image1.png", "image2.png"]

    async def test_iana_emf_mime_type_converted(self, tmp_path: Path, large_png: bytes) -> None:
        """IANA-registered 'image/emf' (without x- prefix) triggers conversion."""
        from tests.conftest import make_docx_zip
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        emf_image = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="image1.emf",
            image_bytes=large_png,
            mime_type="image/emf",
        )
        fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 200

        analyze_mock = AsyncMock(
            return_value=_make_result("image0.png", "IANA EMF content", "call_flow_diagram")
        )
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[emf_image],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.convert_emf_to_jpeg",
                return_value=fake_jpeg,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
        ):
            p = tmp_path / "iana_emf.docx"
            p.write_bytes(make_docx_zip())
            result = await convert_docx_with_ocr(p, api_key="key")

        analyze_mock.assert_called_once()
        assert "IANA EMF content" in result

    async def test_iana_wmf_mime_type_converted(self, tmp_path: Path, large_png: bytes) -> None:
        """IANA-registered 'image/wmf' (without x- prefix) triggers conversion."""
        from tests.conftest import make_docx_zip
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        wmf_image = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="image1.wmf",
            image_bytes=large_png,
            mime_type="image/wmf",
        )
        fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 200

        analyze_mock = AsyncMock(
            return_value=_make_result("image0.png", "IANA WMF content", "table_figure")
        )
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[wmf_image],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.convert_emf_to_jpeg",
                return_value=fake_jpeg,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
        ):
            p = tmp_path / "iana_wmf.docx"
            p.write_bytes(make_docx_zip())
            result = await convert_docx_with_ocr(p, api_key="key")

        analyze_mock.assert_called_once()
        assert "IANA WMF content" in result

    # ── Sequential index-based matching (P0 fix) ───────────────────────────

    async def test_data_uri_placeholder_replaced_by_index(
        self, docx_one_image: Path
    ) -> None:
        """data:image/... URI placeholders are replaced using sequential index."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        analyze_mock = AsyncMock(
            return_value=_make_result("image0.png", "OCR content from data URI", "call_flow_diagram")
        )
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Before\n\n![](data:image/x-emf;base64,AAAA)\n\nAfter",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
        ):
            result = await convert_docx_with_ocr(docx_one_image, api_key="key")

        assert "OCR content from data URI" in result
        assert "data:image/x-emf" not in result
        assert "Before" in result
        assert "After" in result

    async def test_skipped_image_slot_preserved_in_counter(
        self, tmp_path: Path, small_png: bytes, large_png: bytes
    ) -> None:
        """Skipped image at index 0 preserves its placeholder; index 1 is still replaced."""
        from tests.conftest import make_docx_zip
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        small_image = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="image1.png",
            image_bytes=small_png,  # too small → skipped
            mime_type="image/png",
        )
        large_image = ExtractedImage(
            placeholder_name="image1.png",
            media_filename="image2.png",
            image_bytes=large_png,
            mime_type="image/png",
        )

        p = tmp_path / "two_images.docx"
        p.write_bytes(make_docx_zip())
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value=(
                    "![](data:image/png;base64,SMALL)\n\n"
                    "![](data:image/png;base64,LARGE)\n"
                ),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[small_image, large_image],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.settings.vision_min_image_bytes",
                10 * 1024,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(
                    return_value=_make_result("image1.png", "second image content", "other")
                ),
            ),
        ):
            result = await convert_docx_with_ocr(p, api_key="key")

        # First placeholder (small, skipped) is preserved verbatim
        assert "data:image/png;base64,SMALL" in result
        # Second placeholder (large, analysed) is replaced
        assert "second image content" in result
        assert "data:image/png;base64,LARGE" not in result

    async def test_caption_appears_in_stitched_output(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """When ExtractedImage.caption is non-empty, **Figure: ...** heading is prepended."""
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        captioned_image = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="image1.png",
            image_bytes=large_png,
            mime_type="image/png",
            caption="Figure 3: Network Architecture",
        )
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Before\n\n![image](image0.png)\n\nAfter",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[captioned_image],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=_make_result("image0.png", "diagram content")),
            ),
        ):
            p = tmp_path / "captioned.docx"
            p.write_bytes(b"placeholder")
            result = await convert_docx_with_ocr(p, api_key="key")

        assert "**Figure: Figure 3: Network Architecture**" in result
        assert "diagram content" in result

    async def test_no_caption_stitches_without_label(
        self, docx_one_image: Path
    ) -> None:
        """When ExtractedImage.caption is empty, no Figure: heading is emitted."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=_make_result("image0.png", "content")),
            ),
        ):
            result = await convert_docx_with_ocr(docx_one_image, api_key="key")

        assert "**Figure:" not in result
        assert "content" in result

    async def test_index_matching_independent_of_placeholder_url(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """Stitch correctly handles data-URI URL formats using only sequential position."""
        from tests.conftest import make_docx_zip
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        img_a = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="image1.png",
            image_bytes=large_png,
            mime_type="image/png",
        )
        img_b = ExtractedImage(
            placeholder_name="image1.png",
            media_filename="image2.png",
            image_bytes=large_png,
            mime_type="image/png",
        )

        async def recording_analyze(image, *, api_key, model=None):
            return _make_result(image.placeholder_name, f"content-{image.placeholder_name}")

        p = tmp_path / "two_emf.docx"
        p.write_bytes(make_docx_zip())
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value=(
                    "![](data:image/x-emf;base64,FIRST)\n"
                    "![](data:image/x-emf;base64,SECOND)\n"
                ),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[img_a, img_b],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                side_effect=recording_analyze,
            ),
        ):
            result = await convert_docx_with_ocr(p, api_key="key")

        assert "content-image0.png" in result
        assert "content-image1.png" in result
        assert "data:image/x-emf" not in result
