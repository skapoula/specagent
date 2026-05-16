"""Tests for the OCR dispatch path added to ingestor.ingest()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import DOCX_SMALL


def _make_store_mock() -> MagicMock:
    """Return a minimal store mock that satisfies ingest()'s API."""
    store = MagicMock()
    store.find_existing.return_value = (None, None)
    store.upsert_chunks.return_value = None
    store.rebuild_fts_index.return_value = None
    return store


def _embed_documents_side_effect(texts: list):
    import numpy as np

    return np.zeros((len(texts), 768), dtype=np.float32)


@pytest.mark.integration
class TestIngestOcrDispatch:
    """Verify ingest() routes to convert_docx_ocr when OCR is enabled."""

    async def test_uses_ocr_converter_when_enabled(self) -> None:
        """When enable_docx_ocr=True and groq_api_key is set, convert_docx_ocr is called."""
        from specagent.retrieval.ingestor import ingest

        ocr_mock = AsyncMock(return_value=("# OCR Markdown\n\nSome text content here.", []))

        with (
            patch("specagent.retrieval.ingestor.get_store", return_value=_make_store_mock()),
            patch(
                "specagent.retrieval.ingestor.embed_documents",
                side_effect=_embed_documents_side_effect,
            ),
            patch("specagent.retrieval.ingestor.settings.enable_docx_ocr", True),
            patch("specagent.retrieval.ingestor.settings.groq_api_key", "test-api-key"),
            patch("specagent.retrieval.ingestor.convert_docx_ocr", ocr_mock),
        ):
            result = await ingest(DOCX_SMALL, library="test-lib")

        ocr_mock.assert_called_once()
        assert result.status in ("indexed", "replaced")

    async def test_skips_ocr_when_disabled(self) -> None:
        """When enable_docx_ocr=False, standard convert() is called instead."""
        from specagent.retrieval.ingestor import ingest

        standard_convert = MagicMock(return_value="# Standard Markdown\n\nContent here.")
        ocr_mock = AsyncMock()

        with (
            patch("specagent.retrieval.ingestor.get_store", return_value=_make_store_mock()),
            patch(
                "specagent.retrieval.ingestor.embed_documents",
                side_effect=_embed_documents_side_effect,
            ),
            patch("specagent.retrieval.ingestor.settings.enable_docx_ocr", False),
            patch("specagent.retrieval.ingestor.convert", standard_convert),
            patch("specagent.retrieval.ingestor.convert_docx_ocr", ocr_mock),
        ):
            result = await ingest(DOCX_SMALL, library="test-lib")

        ocr_mock.assert_not_called()
        standard_convert.assert_called_once()
        assert result.status in ("indexed", "replaced")

    async def test_skips_ocr_when_api_key_missing(self) -> None:
        """When groq_api_key is empty, standard convert() is used even if OCR is enabled."""
        from specagent.retrieval.ingestor import ingest

        standard_convert = MagicMock(return_value="# Markdown\n\nContent.")
        ocr_mock = AsyncMock()

        with (
            patch("specagent.retrieval.ingestor.get_store", return_value=_make_store_mock()),
            patch(
                "specagent.retrieval.ingestor.embed_documents",
                side_effect=_embed_documents_side_effect,
            ),
            patch("specagent.retrieval.ingestor.settings.enable_docx_ocr", True),
            patch("specagent.retrieval.ingestor.settings.groq_api_key", ""),
            patch("specagent.retrieval.ingestor.convert", standard_convert),
            patch("specagent.retrieval.ingestor.convert_docx_ocr", ocr_mock),
        ):
            result = await ingest(DOCX_SMALL, library="test-lib")

        ocr_mock.assert_not_called()
        standard_convert.assert_called_once()

    async def test_non_docx_file_never_uses_ocr(self, tmp_path: Path) -> None:
        """Non-.docx files always use standard convert() regardless of OCR settings."""
        from specagent.retrieval.ingestor import ingest

        # Use a real minimal markdown file for the non-docx path
        md_file = tmp_path / "notes.md"
        md_file.write_text("# NR Satellite Access\n\nContent about NR satellite access node.\n")

        ocr_mock = AsyncMock()
        standard_convert = MagicMock(
            return_value="# NR Satellite Access\n\nContent about NR satellite access node."
        )

        with (
            patch("specagent.retrieval.ingestor.get_store", return_value=_make_store_mock()),
            patch(
                "specagent.retrieval.ingestor.embed_documents",
                side_effect=_embed_documents_side_effect,
            ),
            patch("specagent.retrieval.ingestor.settings.enable_docx_ocr", True),
            patch("specagent.retrieval.ingestor.settings.groq_api_key", "test-key"),
            patch("specagent.retrieval.ingestor.convert", standard_convert),
            patch("specagent.retrieval.ingestor.convert_docx_ocr", ocr_mock),
        ):
            result = await ingest(md_file, library="test-lib")

        ocr_mock.assert_not_called()
        assert result.status in ("indexed", "replaced")


# ---------------------------------------------------------------------------
# Real-API pipeline tests — one component at a time
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.real_api
@pytest.mark.slow
class TestIngestKuzuStorage:
    """Pipeline component tests: extract → convert → Groq → Kuzu.

    Each test exercises one stage so failures are immediately localised.
    Requires GROQ_API_KEY and inkscape on PATH.
    Run: pytest -m real_api -v
    """

    def test_extract_images_returns_three(
        self,
        inkscape_available: None,
    ) -> None:
        """extract_images() reads the first 3 images from the docx ZIP."""
        from specagent.retrieval.docx_image_extractor import extract_images

        images = extract_images(DOCX_SMALL)[:3]

        assert 1 <= len(images) <= 3
        for img in images:
            assert img.image_bytes
            assert img.mime_type
            assert img.placeholder_name

    def test_inkscape_converts_emf_to_jpeg(
        self,
        inkscape_available: None,
    ) -> None:
        """First EMF image in the docx is rasterised to JPEG bytes by Inkscape."""
        from specagent.retrieval.docx_image_extractor import extract_images
        from specagent.retrieval.emf_converter import EMF_MIME_TYPES, convert_emf_to_jpeg

        images = extract_images(DOCX_SMALL)[:3]
        emf_images = [i for i in images if i.mime_type in EMF_MIME_TYPES]
        if not emf_images:
            pytest.skip("No EMF images in the first 3 frames of this docx")

        img = emf_images[0]
        filetype = "wmf" if "wmf" in img.mime_type else "emf"
        jpeg_bytes = convert_emf_to_jpeg(img.image_bytes, filetype)

        assert jpeg_bytes[:3] == b"\xff\xd8\xff"  # JPEG magic bytes
        assert len(jpeg_bytes) > 500

    async def test_groq_analyzes_one_image(
        self,
        inkscape_available: None,
        groq_api_key: str,
    ) -> None:
        """Groq Vision classifies the first suitable image and returns a result."""
        from specagent.retrieval.docx_image_extractor import extract_images
        from specagent.retrieval.docx_ocr_converter import _prepare_image
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult, analyze_image

        images = extract_images(DOCX_SMALL)[:3]
        prepared = None
        for raw in images:
            prepared = await _prepare_image(raw, DOCX_SMALL.name)
            if prepared is not None:
                break

        if prepared is None:
            pytest.skip("No images passed size/type filters in the first 3")

        result = await analyze_image(prepared, api_key=groq_api_key)

        assert isinstance(result, ImageAnalysisResult)
        assert result.image_type in (
            "call_flow",
            "architecture",
            "table",
            "chart",
            "diagram",
            "other",
            "skip",
        )

    async def test_ingest_ocr_stores_diagrams_in_kuzu(
        self,
        tmp_path: Path,
        inkscape_available: None,
        groq_api_key: str,
    ) -> None:
        """Full ingest(): call_flow diagrams detected by Groq are persisted to Kuzu."""
        import numpy as np

        from specagent.kuzu.connection import KuzuConnection
        from specagent.kuzu.dag_store import CallFlowDagStore
        from specagent.retrieval import docx_image_extractor as _mod
        from specagent.retrieval.ingestor import ingest

        kuzu_path = tmp_path / "kuzu"
        real_extract = _mod.extract_images

        def _fake_embed(texts):
            return np.zeros((len(texts), 768), dtype=np.float32)

        with (
            patch("specagent.retrieval.ingestor.get_store", return_value=_make_store_mock()),
            patch("specagent.retrieval.ingestor.embed_documents", side_effect=_fake_embed),
            patch("specagent.retrieval.ingestor.settings.enable_docx_ocr", True),
            patch("specagent.retrieval.ingestor.settings.groq_api_key", groq_api_key),
            patch("specagent.retrieval.ingestor.settings.enable_dag_storage", True),
            patch("specagent.retrieval.ingestor.settings.data_dir", tmp_path),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                side_effect=lambda p: real_extract(p)[:3],
            ),
            patch(
                "specagent.retrieval.ingestor.get_dag_store",
                return_value=CallFlowDagStore(KuzuConnection(kuzu_path)),
            ),
        ):
            result = await ingest(DOCX_SMALL, library="test-lib")

        assert result.status in ("indexed", "replaced")

        # If Groq classified any of the 3 images as call_flow they are in Kuzu.
        # Zero call_flow diagrams is also a valid outcome for this docx.
        dag_store = CallFlowDagStore(KuzuConnection(kuzu_path))
        stored = dag_store.query_dags_by_keyword(keywords=[""], limit=50)
        for dag in stored:
            assert dag["dag_id"]
            assert dag["source"] == str(DOCX_SMALL)
            mermaid = dag_store.get_dag_mermaid(dag["dag_id"])
            assert mermaid
