"""Integration tests for docx_ocr_converter using real 3GPP .docx files."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from specagent.retrieval.exceptions import IngestionError, UnsupportedFormatError
from tests.conftest import DOCX_SMALL, _make_png_bytes


def _make_result(placeholder: str, content: str, image_type: str = "other"):
    """Convenience builder for ImageAnalysisResult."""
    from specagent.retrieval.groq_vision_client import ImageAnalysisResult

    return ImageAnalysisResult(
        placeholder_name=placeholder,
        markdown_content=content,
        image_type=image_type,
    )


def _make_png_image(placeholder: str, *, size: int = 20 * 1024, caption: str = ""):
    """Build a fake ExtractedImage with a large-enough PNG (no Inkscape needed)."""
    from specagent.retrieval.docx_image_extractor import ExtractedImage

    return ExtractedImage(
        placeholder_name=placeholder,
        media_filename=placeholder,
        image_bytes=_make_png_bytes(n_bytes=size),
        mime_type="image/png",
        caption=caption,
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
            markdown, _ = await convert_docx_with_ocr(docx_no_images, api_key="key")

        assert markdown == "# Title\n\nNo images here."

    async def test_placeholder_replaced_with_mermaid(self) -> None:
        """Call flow diagram placeholder is replaced with Mermaid block in real .docx."""
        mermaid = "```mermaid\nsequenceDiagram\n  A->>B: message\n```"
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Before\n\n![image](image0.png)\n\nAfter",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[_make_png_image("image0.png")],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=_make_result("image0.png", mermaid, "call_flow")),
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        assert "mermaid" in markdown
        assert "![image](image0.png)" not in markdown
        assert "Before" in markdown
        assert "After" in markdown

    async def test_small_image_skipped_below_threshold(self, small_png: bytes) -> None:
        """Images under vision_min_image_bytes are not sent to the API."""
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        tiny_image = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="logo.png",
            image_bytes=small_png,
            mime_type="image/png",
        )
        analyze_mock = AsyncMock()

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Header\n\n![image](image0.png)\n\nFooter",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[tiny_image],
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
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        # analyze_image must NOT have been called for the tiny image
        analyze_mock.assert_not_called()
        # Original placeholder preserved
        assert "![image](image0.png)" in markdown

    async def test_emf_image_converted_and_sent_to_vision_api(self, large_png: bytes) -> None:
        """EMF images are rasterized to JPEG and passed to the vision API."""
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
            return_value=_make_result("image0.png", "EMF diagram content", "call_flow")
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
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                AsyncMock(
                    return_value=_make_result(
                        "image0.png",
                        "```mermaid\nsequenceDiagram\n  A->>B: EMF diagram content\n```",
                        "call_flow",
                    )
                ),
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        analyze_mock.assert_called_once()
        assert "EMF diagram content" in markdown

    async def test_wmf_image_converted_and_sent_to_vision_api(self, large_png: bytes) -> None:
        """WMF images are rasterized to JPEG and passed to the vision API."""
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
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        analyze_mock.assert_called_once()
        assert "WMF chart content" in markdown

    async def test_emf_conversion_failure_preserves_placeholder(self, large_png: bytes) -> None:
        """When EMF→JPEG conversion fails, the placeholder is preserved."""
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
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        analyze_mock.assert_not_called()
        assert "![image](image0.png)" in markdown

    async def test_supported_png_mime_type_not_skipped(self) -> None:
        """PNG images in the real .docx are passed through to the vision API."""
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
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[_make_png_image("image0.png")],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
        ):
            await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        analyze_mock.assert_called_once()

    async def test_large_image_above_max_threshold_skipped(self) -> None:
        """Images above vision_max_image_bytes are skipped and placeholder preserved."""
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        huge_image = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="huge.png",
            image_bytes=_make_png_bytes(n_bytes=25 * 1024 * 1024),  # 25 MB
            mime_type="image/png",
        )
        analyze_mock = AsyncMock()
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[huge_image],
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
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        analyze_mock.assert_not_called()
        assert "![image](image0.png)" in markdown

    async def test_vision_error_falls_back_to_placeholder(self) -> None:
        """A VisionError for one image keeps its original placeholder (real .docx)."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr
        from specagent.retrieval.exceptions import VisionError

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Text\n\n![image](image0.png)\n\nMore",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[_make_png_image("image0.png")],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(side_effect=VisionError("API down")),
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        assert "![image](image0.png)" in markdown
        assert "Text" in markdown

    async def test_raises_ingestion_error_if_pass1_empty(self, docx_no_images: Path) -> None:
        """IngestionError raised when MarkItDown returns empty Markdown (synthetic docx)."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        with patch("specagent.retrieval.docx_ocr_converter.convert", return_value="   "):
            with pytest.raises(IngestionError, match="empty"):
                await convert_docx_with_ocr(docx_no_images, api_key="key")

    async def test_multiple_images_all_replaced(self) -> None:
        """All placeholders in a real .docx document with multiple images are replaced."""
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
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[
                    _make_png_image("image0.png"),
                    _make_png_image("image1.png"),
                    _make_png_image("image2.png"),
                ],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                side_effect=fake_analyze,
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        assert "[image0.png]" in markdown
        assert "[image1.png]" in markdown
        assert "[image2.png]" in markdown
        assert "![image](" not in markdown  # no original placeholders remain

    async def test_images_analyzed_sequentially_in_order(self) -> None:
        """analyze_image is called in image0, image1, image2 order (real .docx)."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        call_order: list[str] = []

        async def recording_analyze(image, *, api_key, model=None):
            call_order.append(image.placeholder_name)
            return _make_result(image.placeholder_name, "x")

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value=("![image](image0.png)\n![image](image1.png)\n![image](image2.png)\n"),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[
                    _make_png_image("image0.png"),
                    _make_png_image("image1.png"),
                    _make_png_image("image2.png"),
                ],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                side_effect=recording_analyze,
            ),
        ):
            await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        assert call_order == ["image0.png", "image1.png", "image2.png"]

    async def test_iana_emf_mime_type_converted(self, large_png: bytes) -> None:
        """IANA-registered 'image/emf' (without x- prefix) triggers conversion."""
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
            return_value=_make_result("image0.png", "IANA EMF content", "call_flow")
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
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                AsyncMock(
                    return_value=_make_result(
                        "image0.png",
                        "```mermaid\nsequenceDiagram\n  A->>B: IANA EMF content\n```",
                        "call_flow",
                    )
                ),
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        analyze_mock.assert_called_once()
        assert "IANA EMF content" in markdown

    async def test_iana_wmf_mime_type_converted(self, large_png: bytes) -> None:
        """IANA-registered 'image/wmf' (without x- prefix) triggers conversion."""
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
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        analyze_mock.assert_called_once()
        assert "IANA WMF content" in markdown

    # ── Sequential index-based matching (P0 fix) ───────────────────────────

    async def test_data_uri_placeholder_replaced_by_index(self) -> None:
        """data:image/... URI placeholders are replaced using sequential index (real .docx)."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        analyze_mock = AsyncMock(
            return_value=_make_result("image0.png", "OCR content from data URI", "call_flow")
        )
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Before\n\n![](data:image/x-emf;base64,AAAA)\n\nAfter",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[_make_png_image("image0.png")],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                analyze_mock,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                AsyncMock(
                    return_value=_make_result(
                        "image0.png",
                        "```mermaid\nsequenceDiagram\n  A->>B: OCR content from data URI\n```",
                        "call_flow",
                    )
                ),
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        assert "OCR content from data URI" in markdown
        assert "data:image/x-emf" not in markdown
        assert "Before" in markdown
        assert "After" in markdown

    async def test_skipped_image_slot_preserved_in_counter(
        self, small_png: bytes, large_png: bytes
    ) -> None:
        """Skipped image at index 0 preserves placeholder; index 1 is still replaced."""
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

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value=(
                    "![](data:image/png;base64,SMALL)\n\n![](data:image/png;base64,LARGE)\n"
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
                AsyncMock(return_value=_make_result("image1.png", "second image content", "other")),
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        # First placeholder (small, skipped) is preserved verbatim
        assert "data:image/png;base64,SMALL" in markdown
        # Second placeholder (large, analysed) is replaced
        assert "second image content" in markdown
        assert "data:image/png;base64,LARGE" not in markdown

    async def test_caption_appears_in_stitched_output(self, large_png: bytes) -> None:
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
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        assert "**Figure: Figure 3: Network Architecture**" in markdown
        assert "diagram content" in markdown

    async def test_no_caption_stitches_without_label(self) -> None:
        """When ExtractedImage.caption is empty, no Figure: heading is emitted (real .docx)."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[_make_png_image("image0.png")],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=_make_result("image0.png", "content")),
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        assert "**Figure:" not in markdown
        assert "content" in markdown

    async def test_invalid_mermaid_triggers_correction_call(self) -> None:
        """When analyze_image returns invalid Mermaid, correct_mermaid_diagram is called (real .docx)."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult

        bad_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="```mermaid\nBADCONTENT\n```",
            image_type="call_flow",
            prose_fallback="A call flow diagram.",
        )
        good_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="```mermaid\nsequenceDiagram\n  A->>B: msg\n  B-->>A: ack\n```",
            image_type="call_flow",
            prose_fallback="A call flow diagram.",
        )
        correction_mock = AsyncMock(return_value=good_result)

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[_make_png_image("image0.png")],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=bad_result),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                correction_mock,
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        correction_mock.assert_called_once()
        assert "sequenceDiagram" in markdown

    async def test_valid_mermaid_skips_correction(self) -> None:
        """When first Mermaid result is valid, correct_mermaid_diagram is never called (real .docx)."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult

        good_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content=(
                "```mermaid\nsequenceDiagram\n  UE->>gNB: attach\n  gNB-->>UE: ok\n```"
            ),
            image_type="call_flow",
            prose_fallback="Attach procedure.",
        )
        correction_mock = AsyncMock()

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=good_result),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                correction_mock,
            ),
        ):
            await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        correction_mock.assert_not_called()

    async def test_correction_failure_falls_back_to_prose(self) -> None:
        """When corrected Mermaid is still invalid, prose_fallback is used (real .docx)."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult

        bad_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="```mermaid\nBADCONTENT\n```",
            image_type="call_flow",
            prose_fallback="A network call flow showing UE and gNB.",
        )
        still_bad = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="```mermaid\nSTILLBAD\n```",
            image_type="call_flow",
            prose_fallback="A network call flow showing UE and gNB.",
        )

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[_make_png_image("image0.png")],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=bad_result),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                AsyncMock(return_value=still_bad),
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        assert "A network call flow showing UE and gNB." in markdown
        assert "STILLBAD" not in markdown

    async def test_non_diagram_type_skips_validation(self) -> None:
        """table and screenshot_text results pass through without validation (real .docx)."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult

        table_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="| Col A | Col B |\n|---|---|\n| 1 | 2 |",
            image_type="table",
            prose_fallback="A parameter table.",
        )
        correction_mock = AsyncMock()

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[_make_png_image("image0.png")],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=table_result),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                correction_mock,
            ),
        ):
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        correction_mock.assert_not_called()
        assert "| Col A |" in markdown

    async def test_index_matching_independent_of_placeholder_url(self, large_png: bytes) -> None:
        """Stitch correctly handles data-URI URL formats using only sequential position."""
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

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value=(
                    "![](data:image/x-emf;base64,FIRST)\n![](data:image/x-emf;base64,SECOND)\n"
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
            markdown, _ = await convert_docx_with_ocr(DOCX_SMALL, api_key="key")

        assert "content-image0.png" in markdown
        assert "content-image1.png" in markdown
        assert "data:image/x-emf" not in markdown
