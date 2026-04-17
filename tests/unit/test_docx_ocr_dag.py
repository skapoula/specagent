"""Unit tests for the DAG injection behaviour added to the OCR converter pipeline.

Covers:
- convert_docx_with_ocr returns (markdown, diagrams) — not just str
- ExtractedDiagram dataclass structure
- Only call_flow image_type diagrams appear in diagrams list
- Skipped images are excluded
- Ingestor Step 3b: DAG storage called for each call_flow diagram
- Ingestor Step 3b: no DAG storage when enable_dag_storage=False
- Ingestor Step 3b: pipeline continues if dag_store raises

All tests mock vision API and Memgraph — pass offline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from specagent.retrieval.docx_ocr_converter import ExtractedDiagram, convert_docx_with_ocr
from specagent.retrieval.exceptions import DagStoreError
from specagent.retrieval.groq_vision_client import ImageAnalysisResult
from specagent.retrieval.ingestor import ingest
from tests.conftest import DOCX_SMALL

# ---------------------------------------------------------------------------
# ExtractedDiagram dataclass
# ---------------------------------------------------------------------------


class TestExtractedDiagram:
    """ExtractedDiagram structure."""

    @pytest.mark.unit
    def test_has_required_fields(self) -> None:
        """ExtractedDiagram has image_type, mermaid_content, prose_description, caption."""
        diagram = ExtractedDiagram(
            image_type="call_flow",
            mermaid_content="```mermaid\nsequenceDiagram\n    UE->>AMF: Reg\n```",
            prose_description="UE registration with AMF.",
            caption="Figure 4.2-1: Registration procedure",
            placeholder_name="image0.png",
        )

        assert diagram.image_type == "call_flow"
        assert "sequenceDiagram" in diagram.mermaid_content
        assert diagram.caption == "Figure 4.2-1: Registration procedure"
        assert diagram.placeholder_name == "image0.png"


# ---------------------------------------------------------------------------
# convert_docx_with_ocr return type
# ---------------------------------------------------------------------------


class TestConvertDocxWithOcrReturnType:
    """convert_docx_with_ocr returns (markdown, diagrams)."""

    @pytest.mark.unit
    async def test_returns_tuple_of_markdown_and_diagrams(self) -> None:
        """Return value is a tuple (str, list[ExtractedDiagram]) for a real .docx."""
        with patch("specagent.retrieval.docx_ocr_converter.convert", return_value="# Test\n"):
            markdown, diagrams = await convert_docx_with_ocr(DOCX_SMALL, api_key="test-key")

        assert isinstance(markdown, str)
        assert isinstance(diagrams, list)

    @pytest.mark.unit
    async def test_call_flow_diagram_in_diagrams_list(self) -> None:
        """A validated call_flow result appears in the diagrams list."""
        call_flow_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="```mermaid\nsequenceDiagram\n    UE->>AMF: Reg\n```",
            image_type="call_flow",
            prose_fallback="UE registration with AMF.",
        )

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="# Test\n![](image1.emf)\n",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                new_callable=AsyncMock,
                return_value=call_flow_result,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter._apply_mermaid_validation",
                new_callable=AsyncMock,
                return_value=call_flow_result,
            ),
        ):
            _, diagrams = await convert_docx_with_ocr(DOCX_SMALL, api_key="test-key")

        assert len(diagrams) >= 1
        assert diagrams[0].image_type == "call_flow"

    @pytest.mark.unit
    async def test_non_call_flow_diagram_excluded(self) -> None:
        """A 'table' image_type does not appear in the diagrams list."""
        table_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="| Col1 | Col2 |\n|---|---|\n| A | B |",
            image_type="table",
        )

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="# Test\n![](image1.emf)\n",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                new_callable=AsyncMock,
                return_value=table_result,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter._apply_mermaid_validation",
                new_callable=AsyncMock,
                return_value=table_result,
            ),
        ):
            _, diagrams = await convert_docx_with_ocr(DOCX_SMALL, api_key="test-key")

        assert diagrams == []

    @pytest.mark.unit
    async def test_skipped_image_excluded_from_diagrams(self) -> None:
        """A skipped ImageAnalysisResult does not appear in the diagrams list."""
        skipped_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="",
            image_type="call_flow",
            skipped=True,
            skip_reason="Too small",
        )

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="# Test\n![](image1.emf)\n",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                new_callable=AsyncMock,
                return_value=skipped_result,
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter._apply_mermaid_validation",
                new_callable=AsyncMock,
                return_value=skipped_result,
            ),
        ):
            _, diagrams = await convert_docx_with_ocr(DOCX_SMALL, api_key="test-key")

        assert diagrams == []


# ---------------------------------------------------------------------------
# Ingestor Step 3b — DAG storage
# ---------------------------------------------------------------------------


class TestIngestorDagStorage:
    """ingestor.ingest() Step 3b: store call-flow diagrams as DAGs."""

    def _make_call_flow_diagram(self, caption: str = "Figure 1: Registration") -> ExtractedDiagram:
        return ExtractedDiagram(
            image_type="call_flow",
            mermaid_content="```mermaid\nsequenceDiagram\n    UE->>AMF: Reg\n```",
            prose_description="UE registration.",
            caption=caption,
            placeholder_name="image0.png",
        )

    @pytest.mark.unit
    async def test_dag_store_called_for_call_flow_diagram(self) -> None:
        """store_call_flow_dag is called when a call_flow diagram is returned."""
        diagram = self._make_call_flow_diagram()
        mock_dag_store = MagicMock()

        with (
            patch("specagent.retrieval.ingestor.settings") as mock_settings,
            patch(
                "specagent.retrieval.ingestor.convert_docx_ocr",
                new_callable=AsyncMock,
                return_value=("# TS38.108\n\nSome text content here.\n", [diagram]),
            ),
            patch("specagent.retrieval.ingestor.get_store") as mock_get_store,
            patch("specagent.retrieval.ingestor.embed_documents", return_value=[[0.0] * 768]),
            patch(
                "specagent.retrieval.ingestor.chunk_with_metadata",
                return_value=[("Some text content here.", "Section 4")],
            ),
            patch("specagent.retrieval.ingestor.get_dag_store", return_value=mock_dag_store),
        ):
            mock_settings.enable_docx_ocr = True
            mock_settings.groq_api_key = "test-key"
            mock_settings.enable_dag_storage = True

            store = MagicMock()
            store.find_existing.return_value = (None, None)
            store.upsert_chunks.return_value = None
            store.rebuild_fts_index.return_value = None
            mock_get_store.return_value = store

            await ingest(source=DOCX_SMALL, library="test")

        mock_dag_store.store_call_flow_dag.assert_called_once()

    @pytest.mark.unit
    async def test_dag_id_uses_doc_name_and_caption(self) -> None:
        """dag_id is constructed as '{doc_name}::{caption}'."""
        diagram = self._make_call_flow_diagram(caption="Figure 4.2-1 Registration")
        mock_dag_store = MagicMock()

        with (
            patch("specagent.retrieval.ingestor.settings") as mock_settings,
            patch(
                "specagent.retrieval.ingestor.convert_docx_ocr",
                new_callable=AsyncMock,
                return_value=("# Content\n\nSome text content here.\n", [diagram]),
            ),
            patch("specagent.retrieval.ingestor.get_store") as mock_get_store,
            patch("specagent.retrieval.ingestor.embed_documents", return_value=[[0.0] * 768]),
            patch(
                "specagent.retrieval.ingestor.chunk_with_metadata",
                return_value=[("Some text content here.", "Section 4")],
            ),
            patch("specagent.retrieval.ingestor.get_dag_store", return_value=mock_dag_store),
        ):
            mock_settings.enable_docx_ocr = True
            mock_settings.groq_api_key = "test-key"
            mock_settings.enable_dag_storage = True

            store = MagicMock()
            store.find_existing.return_value = (None, None)
            store.upsert_chunks.return_value = None
            store.rebuild_fts_index.return_value = None
            mock_get_store.return_value = store

            await ingest(source=DOCX_SMALL, library="test")

        call_kwargs = mock_dag_store.store_call_flow_dag.call_args[1]
        assert call_kwargs["dag_id"] == "38108-i40::Figure 4.2-1 Registration"

    @pytest.mark.unit
    async def test_dag_storage_skipped_when_disabled(self) -> None:
        """No DAG storage when enable_dag_storage=False."""
        diagram = self._make_call_flow_diagram()
        mock_dag_store = MagicMock()

        with (
            patch("specagent.retrieval.ingestor.settings") as mock_settings,
            patch(
                "specagent.retrieval.ingestor.convert_docx_ocr",
                new_callable=AsyncMock,
                return_value=("# Content\n\nText.\n", [diagram]),
            ),
            patch("specagent.retrieval.ingestor.get_store") as mock_get_store,
            patch("specagent.retrieval.ingestor.embed_documents", return_value=[[0.0] * 768]),
            patch(
                "specagent.retrieval.ingestor.chunk_with_metadata",
                return_value=[("Text.", "Section 4")],
            ),
            patch("specagent.retrieval.ingestor.get_dag_store", return_value=mock_dag_store),
        ):
            mock_settings.enable_docx_ocr = True
            mock_settings.groq_api_key = "test-key"
            mock_settings.enable_dag_storage = False

            store = MagicMock()
            store.find_existing.return_value = (None, None)
            store.upsert_chunks.return_value = None
            store.rebuild_fts_index.return_value = None
            mock_get_store.return_value = store

            await ingest(source=DOCX_SMALL, library="test")

        mock_dag_store.store_call_flow_dag.assert_not_called()

    @pytest.mark.unit
    async def test_pipeline_continues_when_dag_store_raises(self) -> None:
        """Ingest succeeds even if store_call_flow_dag raises DagStoreError."""
        diagram = self._make_call_flow_diagram()
        mock_dag_store = MagicMock()
        mock_dag_store.store_call_flow_dag.side_effect = DagStoreError("Memgraph down")

        with (
            patch("specagent.retrieval.ingestor.settings") as mock_settings,
            patch(
                "specagent.retrieval.ingestor.convert_docx_ocr",
                new_callable=AsyncMock,
                return_value=("# Content\n\nText.\n", [diagram]),
            ),
            patch("specagent.retrieval.ingestor.get_store") as mock_get_store,
            patch("specagent.retrieval.ingestor.embed_documents", return_value=[[0.0] * 768]),
            patch(
                "specagent.retrieval.ingestor.chunk_with_metadata",
                return_value=[("Text.", "Section 4")],
            ),
            patch("specagent.retrieval.ingestor.get_dag_store", return_value=mock_dag_store),
        ):
            mock_settings.enable_docx_ocr = True
            mock_settings.groq_api_key = "test-key"
            mock_settings.enable_dag_storage = True

            store = MagicMock()
            store.find_existing.return_value = (None, None)
            store.upsert_chunks.return_value = None
            store.rebuild_fts_index.return_value = None
            mock_get_store.return_value = store

            # Must not raise
            result = await ingest(source=DOCX_SMALL, library="test")

        assert result.status == "indexed"
