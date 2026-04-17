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
