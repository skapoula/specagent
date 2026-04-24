"""Integration tests for the ingestion pipeline."""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import DOCX_SMALL


@pytest.mark.integration
async def test_ingest_returns_indexed_status():
    """ingest() returns IngestResult with status='indexed' for a new file."""
    import numpy as np

    from specagent.retrieval.ingestor import ingest

    with (
        patch("specagent.retrieval.ingestor.get_store") as mock_store_fn,
        patch("specagent.retrieval.ingestor.embed_documents") as mock_embed,
        patch("specagent.retrieval.ingestor.chunk_with_metadata") as mock_chunk,
        patch("specagent.retrieval.ingestor.convert") as mock_convert,
    ):
        mock_convert.return_value = "# 3GPP TS 38.108\n\nSome content about NR."
        mock_chunk.return_value = [("# 3GPP TS 38.108\n\nSome content about NR.", "Section 1")]
        mock_store = MagicMock()
        mock_store.find_existing.return_value = (None, None)
        mock_store_fn.return_value = mock_store
        mock_embed.return_value = np.array([[0.1] * 768], dtype=np.float32)

        result = await ingest(source=DOCX_SMALL, library="test-lib")

    assert result.status == "indexed"


@pytest.mark.integration
async def test_ingest_skips_unchanged_file():
    """ingest() returns status='skipped' when content hash matches existing."""
    raw_bytes = DOCX_SMALL.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    from specagent.retrieval.ingestor import ingest

    with (
        patch("specagent.retrieval.ingestor.get_store") as mock_store_fn,
    ):
        mock_store = MagicMock()
        mock_store.find_existing.return_value = ("existing-doc-id", content_hash)
        mock_store_fn.return_value = mock_store

        result = await ingest(source=DOCX_SMALL, library="test-lib")

    assert result.status == "skipped"


@pytest.mark.integration
async def test_ingest_records_include_section_header():
    """Ingested ChunkRecords include section_header in metadata JSON."""
    import numpy as np

    captured_chunks = []

    from specagent.retrieval.ingestor import ingest

    def capture_upsert(chunks, **_kwargs):
        captured_chunks.extend(chunks)

    with (
        patch("specagent.retrieval.ingestor.get_store") as mock_store_fn,
        patch("specagent.retrieval.ingestor.embed_documents") as mock_embed,
        patch("specagent.retrieval.ingestor.chunk_with_metadata") as mock_chunk,
        patch("specagent.retrieval.ingestor.convert") as mock_convert,
    ):
        mock_convert.return_value = "# 3GPP TS 38.108\n\nContent about NR satellite access."
        mock_chunk.return_value = [
            ("# 3GPP TS 38.108", "3GPP TS 38.108"),
            ("Content about NR satellite access.", "3GPP TS 38.108"),
        ]
        mock_store = MagicMock()
        mock_store.find_existing.return_value = (None, None)
        mock_store.upsert_chunks.side_effect = capture_upsert
        mock_store_fn.return_value = mock_store
        mock_embed.return_value = np.array([[0.1] * 768, [0.1] * 768], dtype=np.float32)

        await ingest(source=DOCX_SMALL, library="test-lib")

    assert len(captured_chunks) >= 1
    meta = json.loads(captured_chunks[0].metadata)
    assert "section_header" in meta


@pytest.mark.unit
def test_get_store_singleton():
    from specagent.retrieval.resources import clear_resource_cache, get_store

    clear_resource_cache()
    with patch("specagent.retrieval.resources.Store") as mock_cls:
        mock_cls.return_value = MagicMock()
        s1 = get_store()
        s2 = get_store()
    assert s1 is s2
    mock_cls.assert_called_once()
    clear_resource_cache()


@pytest.mark.unit
def test_extract_title_from_heading():
    from specagent.retrieval.ingestor import _extract_title

    assert _extract_title("# My Title\n\nContent.", "/f.docx") == "My Title"


@pytest.mark.unit
def test_extract_title_fallback():
    from specagent.retrieval.ingestor import _extract_title

    assert _extract_title("no heading", "/docs/spec.docx") == "spec.docx"


@pytest.mark.integration
async def test_ingest_missing_file(tmp_path):
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with patch("specagent.retrieval.ingestor.get_store", return_value=mock_store):
        with pytest.raises(IngestionError, match="Cannot read file"):
            await ingest(tmp_path / "missing.docx", library="test")


@pytest.mark.integration
async def test_ingest_skips_on_same_hash():
    from specagent.retrieval.ingestor import ingest

    h = hashlib.sha256(DOCX_SMALL.read_bytes()).hexdigest()
    mock_store = MagicMock()
    mock_store.find_existing.return_value = ("existing-id", h)
    with patch("specagent.retrieval.ingestor.get_store", return_value=mock_store):
        result = await ingest(DOCX_SMALL, library="test")
    assert result.status == "skipped"


@pytest.mark.integration
async def test_ingest_empty_text_raises():
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.convert", return_value=""),
    ):
        with pytest.raises(IngestionError, match="No text"):
            await ingest(DOCX_SMALL, library="test")


@pytest.mark.integration
async def test_ingest_no_chunks_raises():
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.convert", return_value="text"),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[]),
    ):
        with pytest.raises(IngestionError, match="No usable chunks"):
            await ingest(DOCX_SMALL, library="test")


@pytest.mark.integration
async def test_ingest_folder_bad_path():
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest_folder

    with pytest.raises(IngestionError, match="not found"):
        await ingest_folder("/nonexistent/xyz", library="test")


@pytest.mark.integration
async def test_ingest_stat_oserror_sets_empty_last_modified():
    """When stat() raises OSError, last_modified is set to empty string."""
    import numpy as np

    from specagent.retrieval.ingestor import ingest

    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch(
            "specagent.retrieval.ingestor.embed_documents",
            return_value=np.array([[0.1] * 768], dtype=np.float32),
        ),
        patch("specagent.retrieval.ingestor.convert", return_value="# T\n\nContent."),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("content", "")]),
        patch("pathlib.Path.stat", side_effect=OSError("no access")),
    ):
        result = await ingest(DOCX_SMALL, library="test")
    assert result.status == "indexed"


@pytest.mark.integration
async def test_ingest_convert_unsupported_format_reraises(tmp_path):
    """When convert() raises UnsupportedFormatError, it is re-raised unchanged."""
    from specagent.retrieval.exceptions import UnsupportedFormatError
    from specagent.retrieval.ingestor import ingest

    bad_file = tmp_path / "file.xyz"
    bad_file.write_bytes(b"content")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch(
            "specagent.retrieval.ingestor.convert",
            side_effect=UnsupportedFormatError(".xyz"),
        ),
        pytest.raises(UnsupportedFormatError),
    ):
        await ingest(bad_file, library="test")


@pytest.mark.integration
async def test_ingest_convert_exception_raises_ingestion_error():
    """When convert() raises a generic exception, IngestionError is raised."""
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.convert", side_effect=RuntimeError("parse error")),
    ):
        with pytest.raises(IngestionError, match="Conversion failed"):
            await ingest(DOCX_SMALL, library="test")


@pytest.mark.integration
async def test_ingest_chunk_exception_raises_ingestion_error():
    """When chunk_with_metadata() raises, IngestionError is raised."""
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.convert", return_value="some content"),
        patch(
            "specagent.retrieval.ingestor.chunk_with_metadata",
            side_effect=RuntimeError("chunk fail"),
        ),
        pytest.raises(IngestionError, match="Chunking failed"),
    ):
        await ingest(DOCX_SMALL, library="test")


@pytest.mark.integration
async def test_ingest_embed_exception_raises_ingestion_error():
    """When embedding fails, IngestionError is raised."""
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.embed_documents", side_effect=RuntimeError("OOM")),
        patch("specagent.retrieval.ingestor.convert", return_value="content"),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        with pytest.raises(IngestionError, match="Embedding failed"):
            await ingest(DOCX_SMALL, library="test")


@pytest.mark.integration
async def test_ingest_store_write_exception_raises_ingestion_error():
    """When upsert_chunks() raises, IngestionError is raised."""
    import numpy as np

    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    mock_store.upsert_chunks.side_effect = RuntimeError("write failed")
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch(
            "specagent.retrieval.ingestor.embed_documents",
            return_value=np.array([[0.1] * 768], dtype=np.float32),
        ),
        patch("specagent.retrieval.ingestor.convert", return_value="content"),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        with pytest.raises(IngestionError, match="Store write failed"):
            await ingest(DOCX_SMALL, library="test")


@pytest.mark.integration
async def test_ingest_replaces_existing_document():
    """ingest() deletes old doc and returns status='replaced' when doc exists."""
    import numpy as np

    from specagent.retrieval.ingestor import ingest

    mock_store = MagicMock()
    mock_store.find_existing.return_value = ("old-doc-id", "old-hash-xyz")
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch(
            "specagent.retrieval.ingestor.embed_documents",
            return_value=np.array([[0.1] * 768], dtype=np.float32),
        ),
        patch("specagent.retrieval.ingestor.convert", return_value="content"),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        result = await ingest(DOCX_SMALL, library="test")
    assert result.status == "replaced"
    mock_store.delete_document.assert_called_once_with("old-doc-id")


@pytest.mark.integration
async def test_ingest_replace_delete_failure_logs_warning():
    """When delete of old doc fails, a warning is logged but ingest still succeeds."""
    import numpy as np

    from specagent.retrieval.ingestor import ingest

    mock_store = MagicMock()
    mock_store.find_existing.return_value = ("old-doc-id", "old-hash-xyz")
    mock_store.delete_document.side_effect = RuntimeError("delete failed")
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch(
            "specagent.retrieval.ingestor.embed_documents",
            return_value=np.array([[0.1] * 768], dtype=np.float32),
        ),
        patch("specagent.retrieval.ingestor.convert", return_value="content"),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        result = await ingest(DOCX_SMALL, library="test")
    assert result.status == "replaced"


@pytest.mark.integration
async def test_ingest_folder_with_docx_file():
    """ingest_folder() processes a real .docx file via _ingest_one."""
    import numpy as np

    from specagent.retrieval.ingestor import ingest_folder

    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch(
            "specagent.retrieval.ingestor.embed_documents",
            return_value=np.array([[0.1] * 768], dtype=np.float32),
        ),
        patch("specagent.retrieval.ingestor.convert", return_value="# Title\n\nContent here."),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        result = await ingest_folder(DOCX_SMALL.parent, library="test")
    assert result.total_files >= 1
    assert result.indexed >= 1
    assert result.failed == 0


@pytest.mark.integration
async def test_ingest_folder_empty_logs_warning(tmp_path):
    """ingest_folder() logs a warning when no supported files are found."""
    from specagent.retrieval.ingestor import ingest_folder

    (tmp_path / "file.xyz").write_bytes(b"unsupported")
    result = await ingest_folder(tmp_path, library="test")
    assert result.total_files == 0
    assert result.indexed == 0


@pytest.mark.integration
async def test_ingest_folder_errors_collected(tmp_path):
    """ingest_folder() collects per-file errors without raising."""
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest_folder

    (tmp_path / "bad.docx").write_bytes(b"not a real docx")
    with patch("specagent.retrieval.ingestor.ingest", side_effect=IngestionError("boom")):
        result = await ingest_folder(tmp_path, library="test")
    assert result.failed == 1
    assert len(result.errors) == 1


@pytest.mark.integration
async def test_ingest_folder_fts_rebuild_failure_logs_warning():
    """ingest_folder() logs a warning when FTS rebuild fails after successful ingests."""
    import numpy as np

    from specagent.retrieval.ingestor import ingest_folder

    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    mock_store.rebuild_fts_index.side_effect = RuntimeError("fts failed")
    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=mock_store),
        patch(
            "specagent.retrieval.ingestor.embed_documents",
            return_value=np.array([[0.1] * 768], dtype=np.float32),
        ),
        patch("specagent.retrieval.ingestor.convert", return_value="# Title\n\nContent here."),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        result = await ingest_folder(DOCX_SMALL.parent, library="test")
    # Ingest should still succeed even if FTS rebuild fails
    assert result.indexed >= 1


# ---------------------------------------------------------------------------
# 3GPP release folder organisation
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ingest_copies_docx_to_release_folder(tmp_path):
    """ingest() copies the source .docx to 3gpp_rel_XX/docx/ inside data_dir."""
    import numpy as np

    from specagent.retrieval.ingestor import ingest

    with (
        patch("specagent.retrieval.ingestor.get_store") as mock_store_fn,
        patch("specagent.retrieval.ingestor.embed_documents") as mock_embed,
        patch("specagent.retrieval.ingestor.chunk_with_metadata") as mock_chunk,
        patch("specagent.retrieval.ingestor.convert") as mock_convert,
        patch("specagent.retrieval.ingestor.settings") as mock_settings,
    ):
        mock_settings.enable_docx_ocr = False
        mock_settings.groq_api_key = None
        mock_settings.enable_dag_storage = False
        mock_settings.data_dir = tmp_path
        mock_convert.return_value = "# 3GPP TS 38.108\n\nContent."
        mock_chunk.return_value = [("# 3GPP TS 38.108\n\nContent.", "Section 1")]
        mock_store = MagicMock()
        mock_store.find_existing.return_value = (None, None)
        mock_store_fn.return_value = mock_store
        mock_embed.return_value = np.array([[0.1] * 768], dtype=np.float32)

        await ingest(source=DOCX_SMALL, library="test-lib")

    docx_dest = tmp_path / "3gpp_rel_18" / "docx" / f"{DOCX_SMALL.stem}_rel18.docx"
    assert docx_dest.exists(), f"Expected docx copy at {docx_dest}"


@pytest.mark.integration
async def test_ingest_writes_markdown_to_release_folder(tmp_path):
    """ingest() writes the converted Markdown to 3gpp_rel_XX/md/ inside data_dir."""
    import numpy as np

    from specagent.retrieval.ingestor import ingest

    md_content = "# 3GPP TS 38.108\n\nContent about NR."

    with (
        patch("specagent.retrieval.ingestor.get_store") as mock_store_fn,
        patch("specagent.retrieval.ingestor.embed_documents") as mock_embed,
        patch("specagent.retrieval.ingestor.chunk_with_metadata") as mock_chunk,
        patch("specagent.retrieval.ingestor.convert") as mock_convert,
        patch("specagent.retrieval.ingestor.settings") as mock_settings,
    ):
        mock_settings.enable_docx_ocr = False
        mock_settings.groq_api_key = None
        mock_settings.enable_dag_storage = False
        mock_settings.data_dir = tmp_path
        mock_convert.return_value = md_content
        mock_chunk.return_value = [("chunk", "Section 1")]
        mock_store = MagicMock()
        mock_store.find_existing.return_value = (None, None)
        mock_store_fn.return_value = mock_store
        mock_embed.return_value = np.array([[0.1] * 768], dtype=np.float32)

        await ingest(source=DOCX_SMALL, library="test-lib")

    md_dest = tmp_path / "3gpp_rel_18" / "md" / f"{DOCX_SMALL.stem}_rel18.md"
    assert md_dest.exists(), f"Expected markdown at {md_dest}"
    assert md_dest.read_text() == md_content


@pytest.mark.integration
async def test_ingest_stores_release_in_chunk_metadata(tmp_path):
    """ingest() adds 'release' key to each ChunkRecord's metadata JSON."""
    import numpy as np

    captured_chunks = []

    from specagent.retrieval.ingestor import ingest

    def capture_upsert(chunks, **_kwargs):
        captured_chunks.extend(chunks)

    with (
        patch("specagent.retrieval.ingestor.get_store") as mock_store_fn,
        patch("specagent.retrieval.ingestor.embed_documents") as mock_embed,
        patch("specagent.retrieval.ingestor.chunk_with_metadata") as mock_chunk,
        patch("specagent.retrieval.ingestor.convert") as mock_convert,
        patch("specagent.retrieval.ingestor.settings") as mock_settings,
    ):
        mock_settings.enable_docx_ocr = False
        mock_settings.groq_api_key = None
        mock_settings.enable_dag_storage = False
        mock_settings.data_dir = tmp_path
        mock_convert.return_value = "# 3GPP TS 38.108\n\nContent."
        mock_chunk.return_value = [("chunk one", "Sec 1"), ("chunk two", "Sec 2")]
        mock_store = MagicMock()
        mock_store.find_existing.return_value = (None, None)
        mock_store.upsert_chunks.side_effect = capture_upsert
        mock_store_fn.return_value = mock_store
        mock_embed.return_value = np.array([[0.1] * 768, [0.1] * 768], dtype=np.float32)

        await ingest(source=DOCX_SMALL, library="test-lib")

    assert captured_chunks, "No chunks were captured"
    for chunk in captured_chunks:
        meta = json.loads(chunk.metadata)
        assert meta.get("release") == 18
        assert chunk.release == 18


@pytest.mark.integration
async def test_ingest_skips_release_folder_for_non_3gpp_file(tmp_path):
    """ingest() does not create release folders for files with non-3GPP names."""
    import numpy as np

    from specagent.retrieval.ingestor import ingest

    plain_docx = tmp_path / "report.docx"
    plain_docx.write_bytes(DOCX_SMALL.read_bytes())

    with (
        patch("specagent.retrieval.ingestor.get_store") as mock_store_fn,
        patch("specagent.retrieval.ingestor.embed_documents") as mock_embed,
        patch("specagent.retrieval.ingestor.chunk_with_metadata") as mock_chunk,
        patch("specagent.retrieval.ingestor.convert") as mock_convert,
        patch("specagent.retrieval.ingestor.settings") as mock_settings,
    ):
        mock_settings.enable_docx_ocr = False
        mock_settings.groq_api_key = None
        mock_settings.enable_dag_storage = False
        mock_settings.data_dir = tmp_path
        mock_convert.return_value = "# Report\n\nContent."
        mock_chunk.return_value = [("chunk", "Section")]
        mock_store = MagicMock()
        mock_store.find_existing.return_value = (None, None)
        mock_store_fn.return_value = mock_store
        mock_embed.return_value = np.array([[0.1] * 768], dtype=np.float32)

        result = await ingest(source=plain_docx, library="test-lib")

    assert result.status == "indexed"
    # No 3gpp_rel_* folder should have been created
    rel_folders = list(tmp_path.glob("3gpp_rel_*"))
    assert rel_folders == [], f"Unexpected release folders: {rel_folders}"
