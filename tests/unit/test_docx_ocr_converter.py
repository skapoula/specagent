"""Unit tests for docx_ocr_converter — TDD for Issues 2, 5, and 7."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Issue 2: _prose_fallback_result must set skipped=True (TDD)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProseFallbackResult:
    def _make_call_flow_result(self) -> ImageAnalysisResult:
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult

        return ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="```mermaid\nsequenceDiagram\n  A->>B: msg\n  B-->>A: ok\n```",
            image_type="call_flow",
            prose_fallback="A call flow between A and B.",
        )

    def test_prose_fallback_result_sets_skipped_true(self) -> None:
        """_prose_fallback_result must mark result as skipped."""
        from specagent.retrieval.docx_ocr_converter import _prose_fallback_result

        result = _prose_fallback_result(self._make_call_flow_result(), "image0.png")
        assert result.skipped is True

    def test_prose_fallback_result_sets_skip_reason(self) -> None:
        """_prose_fallback_result must populate skip_reason."""
        from specagent.retrieval.docx_ocr_converter import _prose_fallback_result

        result = _prose_fallback_result(self._make_call_flow_result(), "image0.png")
        assert result.skip_reason == "mermaid_validation_failed_prose_fallback"

    def test_prose_fallback_result_uses_prose_fallback_text(self) -> None:
        """markdown_content is replaced by prose_fallback when present."""
        from specagent.retrieval.docx_ocr_converter import _prose_fallback_result

        result = _prose_fallback_result(self._make_call_flow_result(), "image0.png")
        assert result.markdown_content == "A call flow between A and B."

    def test_prose_fallback_result_uses_placeholder_when_no_prose(self) -> None:
        """markdown_content falls back to a marker when prose_fallback is empty."""
        from specagent.retrieval.docx_ocr_converter import _prose_fallback_result
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult

        result_no_prose = ImageAnalysisResult(
            placeholder_name="image1.png",
            markdown_content="bad mermaid",
            image_type="call_flow",
            prose_fallback="",
        )
        result = _prose_fallback_result(result_no_prose, "image1.png")
        assert "image1.png" in result.markdown_content
        assert result.skipped is True


# ---------------------------------------------------------------------------
# Issue 5: _warn_if_inkscape_missing logs when Inkscape absent (TDD)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWarnIfInkscapeMissing:
    def test_logs_warning_when_inkscape_not_found(self, caplog) -> None:
        """_warn_if_inkscape_missing logs a WARNING when shutil.which returns None."""
        from specagent.retrieval.docx_ocr_converter import _warn_if_inkscape_missing

        with (
            patch("shutil.which", return_value=None),
            caplog.at_level(logging.WARNING, logger="specagent.retrieval.docx_ocr_converter"),
        ):
            _warn_if_inkscape_missing()

        assert any("Inkscape" in r.message for r in caplog.records)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_no_warning_when_inkscape_present(self, caplog) -> None:
        """_warn_if_inkscape_missing must not log when Inkscape is on PATH."""
        from specagent.retrieval.docx_ocr_converter import _warn_if_inkscape_missing

        with (
            patch("shutil.which", return_value="/usr/bin/inkscape"),
            caplog.at_level(logging.WARNING, logger="specagent.retrieval.docx_ocr_converter"),
        ):
            _warn_if_inkscape_missing()

        assert not any("Inkscape" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Issue 7: _stitch() uses placeholder_name key, not sequential counter (TDD)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStitch:
    def _make_result(
        self,
        placeholder: str,
        content: str,
        image_type: str = "other",
        skipped: bool = False,
    ) -> ImageAnalysisResult:
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult

        return ImageAnalysisResult(
            placeholder_name=placeholder,
            markdown_content=content,
            image_type=image_type,
            skipped=skipped,
        )

    def test_stitch_replaces_by_placeholder_name_not_order(self) -> None:
        """_stitch must match on URL/placeholder_name, not sequential counter."""
        from specagent.retrieval.docx_ocr_converter import _stitch

        markdown = "![image](image1.png)\n\n![image](image0.png)"
        results = {
            "image0.png": self._make_result("image0.png", "Content for image0"),
            "image1.png": self._make_result("image1.png", "Content for image1"),
        }
        stitched = _stitch(markdown, results)
        # image1.png appears first in markdown but should get its own content
        assert stitched.index("Content for image1") < stitched.index("Content for image0")

    def test_stitch_skipped_result_not_replaced(self) -> None:
        """_stitch must keep original placeholder for skipped results."""
        from specagent.retrieval.docx_ocr_converter import _stitch

        markdown = "![image](image0.png)"
        results = {
            "image0.png": self._make_result("image0.png", "Skipped content", skipped=True),
        }
        stitched = _stitch(markdown, results)
        assert "![image](image0.png)" in stitched

    def test_stitch_prepends_caption_when_present(self) -> None:
        """_stitch prepends **Figure: caption** when captions dict is provided."""
        from specagent.retrieval.docx_ocr_converter import _stitch

        markdown = "![image](image0.png)"
        results = {"image0.png": self._make_result("image0.png", "The content")}
        captions = {"image0.png": "Flow diagram"}
        stitched = _stitch(markdown, results, captions)
        assert "**Figure: Flow diagram**" in stitched
        assert "The content" in stitched

    def test_stitch_unknown_placeholder_left_unchanged(self) -> None:
        """_stitch leaves placeholders with no matching result untouched."""
        from specagent.retrieval.docx_ocr_converter import _stitch

        markdown = "![image](image99.png)"
        results: dict = {}
        stitched = _stitch(markdown, results)
        assert "![image](image99.png)" in stitched
