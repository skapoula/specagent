"""Integration tests for the ingestion pipeline."""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def md_file(tmp_path):
    f = tmp_path / "spec.md"
    f.write_text("# Section 5.4\n\nContent about HARQ processes.\n")
    return f


@pytest.mark.integration
async def test_ingest_returns_indexed_status(md_file):
    """ingest() returns IngestResult with status='indexed' for a new file."""
    from specagent.retrieval.ingestor import ingest

    with (
        patch("specagent.retrieval.ingestor._get_store") as mock_store_fn,
        patch("specagent.retrieval.ingestor.get_embedder") as mock_emb_fn,
        patch("specagent.retrieval.ingestor.chunk_with_metadata") as mock_chunk,
    ):
        mock_chunk.return_value = [
            ("# Section 5.4\n\nContent about HARQ processes.", "Section 5.4")
        ]
        mock_store = MagicMock()
        mock_store.find_existing.return_value = (None, None)
        mock_store_fn.return_value = mock_store
        mock_emb = MagicMock()
        mock_emb.embed.return_value = iter([[0.1] * 768])
        mock_emb_fn.return_value = mock_emb

        result = await ingest(source=md_file, library="test-lib")

    assert result.status == "indexed"


@pytest.mark.integration
async def test_ingest_skips_unchanged_file(md_file):
    """ingest() returns status='skipped' when content hash matches existing."""
    raw_bytes = md_file.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    from specagent.retrieval.ingestor import ingest

    with (
        patch("specagent.retrieval.ingestor._get_store") as mock_store_fn,
        patch("specagent.retrieval.ingestor.get_embedder") as mock_emb_fn,
    ):
        mock_store = MagicMock()
        mock_store.find_existing.return_value = ("existing-doc-id", content_hash)
        mock_store_fn.return_value = mock_store
        mock_emb_fn.return_value = MagicMock()

        result = await ingest(source=md_file, library="test-lib")

    assert result.status == "skipped"


@pytest.mark.integration
async def test_ingest_records_include_section_header(md_file):
    """Ingested ChunkRecords include section_header in metadata JSON."""
    captured_chunks = []

    from specagent.retrieval.ingestor import ingest

    def capture_upsert(chunks):
        captured_chunks.extend(chunks)

    with (
        patch("specagent.retrieval.ingestor._get_store") as mock_store_fn,
        patch("specagent.retrieval.ingestor.get_embedder") as mock_emb_fn,
        patch("specagent.retrieval.ingestor.chunk_with_metadata") as mock_chunk,
    ):
        mock_chunk.return_value = [
            ("# Section 5.4", "Section 5.4"),
            ("Content about HARQ processes.", "Section 5.4"),
        ]
        mock_store = MagicMock()
        mock_store.find_existing.return_value = (None, None)
        mock_store.upsert_chunks.side_effect = capture_upsert
        mock_store_fn.return_value = mock_store
        mock_emb = MagicMock()
        mock_emb.embed.return_value = iter([[0.1] * 768, [0.1] * 768])
        mock_emb_fn.return_value = mock_emb

        await ingest(source=md_file, library="test-lib")

    assert len(captured_chunks) >= 1
    meta = json.loads(captured_chunks[0].metadata)
    assert "section_header" in meta


@pytest.mark.unit
def test_get_store_singleton():
    from specagent.retrieval import ingestor

    ingestor._store = None
    with patch("specagent.retrieval.ingestor.Store") as mock_cls:
        mock_cls.return_value = MagicMock()
        s1 = ingestor._get_store()
        s2 = ingestor._get_store()
    assert s1 is s2
    mock_cls.assert_called_once()
    ingestor._store = None


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
    with patch("specagent.retrieval.ingestor._get_store", return_value=mock_store):
        with pytest.raises(IngestionError, match="Cannot read file"):
            await ingest(tmp_path / "missing.md", library="test")


@pytest.mark.integration
async def test_ingest_skips_on_same_hash(tmp_path):
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "doc.md"
    md.write_text("# T\n\nContent.")
    h = hashlib.sha256(md.read_bytes()).hexdigest()
    mock_store = MagicMock()
    mock_store.find_existing.return_value = ("existing-id", h)
    with patch("specagent.retrieval.ingestor._get_store", return_value=mock_store):
        result = await ingest(md, library="test")
    assert result.status == "skipped"


@pytest.mark.integration
async def test_ingest_empty_text_raises(tmp_path):
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "empty.md"
    md.write_text("   ")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.convert", return_value=""),
    ):
        with pytest.raises(IngestionError, match="No text"):
            await ingest(md, library="test")


@pytest.mark.integration
async def test_ingest_no_chunks_raises(tmp_path):
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "doc.md"
    md.write_text("# T\n\nText.")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.convert", return_value="text"),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[]),
    ):
        with pytest.raises(IngestionError, match="No usable chunks"):
            await ingest(md, library="test")


@pytest.mark.integration
async def test_ingest_folder_bad_path():
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest_folder

    with pytest.raises(IngestionError, match="not found"):
        await ingest_folder("/nonexistent/xyz", library="test")


@pytest.mark.integration
async def test_ingest_stat_oserror_sets_empty_last_modified(tmp_path):
    """When stat() raises OSError, last_modified is set to empty string."""
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "doc.md"
    md.write_text("# T\n\nContent.")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    mock_emb = MagicMock()
    mock_emb.embed.return_value = iter([[0.1] * 768])
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.get_embedder", return_value=mock_emb),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("content", "")]),
        patch("pathlib.Path.stat", side_effect=OSError("no access")),
    ):
        result = await ingest(md, library="test")
    assert result.status == "indexed"


@pytest.mark.integration
async def test_ingest_convert_unsupported_format_reraises(tmp_path):
    """When convert() raises UnsupportedFormatError, it is re-raised unchanged."""
    from specagent.retrieval.exceptions import UnsupportedFormatError
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "doc.xyz"
    md.write_text("content")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch(
            "specagent.retrieval.ingestor.convert",
            side_effect=UnsupportedFormatError(".xyz"),
        ),
        pytest.raises(UnsupportedFormatError),
    ):
        await ingest(md, library="test")


@pytest.mark.integration
async def test_ingest_convert_exception_raises_ingestion_error(tmp_path):
    """When convert() raises a generic exception, IngestionError is raised."""
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "doc.md"
    md.write_text("content")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.convert", side_effect=RuntimeError("parse error")),
    ):
        with pytest.raises(IngestionError, match="Conversion failed"):
            await ingest(md, library="test")


@pytest.mark.integration
async def test_ingest_chunk_exception_raises_ingestion_error(tmp_path):
    """When chunk_with_metadata() raises, IngestionError is raised."""
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "doc.md"
    md.write_text("# T\n\nContent.")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.convert", return_value="some content"),
        patch(
            "specagent.retrieval.ingestor.chunk_with_metadata",
            side_effect=RuntimeError("chunk fail"),
        ),
        pytest.raises(IngestionError, match="Chunking failed"),
    ):
        await ingest(md, library="test")


@pytest.mark.integration
async def test_ingest_embed_exception_raises_ingestion_error(tmp_path):
    """When embedding fails, IngestionError is raised."""
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "doc.md"
    md.write_text("# T\n\nContent.")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    mock_emb = MagicMock()
    mock_emb.embed.side_effect = RuntimeError("OOM")
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.get_embedder", return_value=mock_emb),
        patch("specagent.retrieval.ingestor.convert", return_value="content"),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        with pytest.raises(IngestionError, match="Embedding failed"):
            await ingest(md, library="test")


@pytest.mark.integration
async def test_ingest_store_write_exception_raises_ingestion_error(tmp_path):
    """When upsert_chunks() raises, IngestionError is raised."""
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "doc.md"
    md.write_text("# T\n\nContent.")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    mock_store.upsert_chunks.side_effect = RuntimeError("write failed")
    mock_emb = MagicMock()
    mock_emb.embed.return_value = iter([[0.1] * 768])
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.get_embedder", return_value=mock_emb),
        patch("specagent.retrieval.ingestor.convert", return_value="content"),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        with pytest.raises(IngestionError, match="Store write failed"):
            await ingest(md, library="test")


@pytest.mark.integration
async def test_ingest_replaces_existing_document(tmp_path):
    """ingest() deletes old doc and returns status='replaced' when doc exists."""
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "doc.md"
    md.write_text("# T\n\nNew content.")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = ("old-doc-id", "old-hash-xyz")
    mock_emb = MagicMock()
    mock_emb.embed.return_value = iter([[0.1] * 768])
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.get_embedder", return_value=mock_emb),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        result = await ingest(md, library="test")
    assert result.status == "replaced"
    mock_store.delete_document.assert_called_once_with("old-doc-id")


@pytest.mark.integration
async def test_ingest_replace_delete_failure_logs_warning(tmp_path):
    """When delete of old doc fails, a warning is logged but ingest still succeeds."""
    from specagent.retrieval.ingestor import ingest

    md = tmp_path / "doc.md"
    md.write_text("# T\n\nNew content.")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = ("old-doc-id", "old-hash-xyz")
    mock_store.delete_document.side_effect = RuntimeError("delete failed")
    mock_emb = MagicMock()
    mock_emb.embed.return_value = iter([[0.1] * 768])
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.get_embedder", return_value=mock_emb),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        result = await ingest(md, library="test")
    assert result.status == "replaced"


@pytest.mark.integration
async def test_ingest_folder_with_md_file(tmp_path):
    """ingest_folder() processes supported files via _ingest_one."""
    from specagent.retrieval.ingestor import ingest_folder

    md = tmp_path / "spec.md"
    md.write_text("# Title\n\nContent here.")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    mock_emb = MagicMock()
    mock_emb.embed.return_value = iter([[0.1] * 768])
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.get_embedder", return_value=mock_emb),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        result = await ingest_folder(tmp_path, library="test")
    assert result.total_files == 1
    assert result.indexed == 1
    assert result.failed == 0


@pytest.mark.integration
async def test_ingest_folder_empty_logs_warning(tmp_path):
    """ingest_folder() logs a warning when no supported files are found."""
    from specagent.retrieval.ingestor import ingest_folder

    (tmp_path / "file.xyz").write_text("unsupported")
    result = await ingest_folder(tmp_path, library="test")
    assert result.total_files == 0
    assert result.indexed == 0


@pytest.mark.integration
async def test_ingest_folder_errors_collected(tmp_path):
    """ingest_folder() collects per-file errors without raising."""
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest_folder

    md = tmp_path / "bad.md"
    md.write_text("content")
    with patch("specagent.retrieval.ingestor.ingest", side_effect=IngestionError("boom")):
        result = await ingest_folder(tmp_path, library="test")
    assert result.failed == 1
    assert len(result.errors) == 1


@pytest.mark.integration
async def test_ingest_folder_fts_rebuild_failure_logs_warning(tmp_path):
    """ingest_folder() logs a warning when FTS rebuild fails after successful ingests."""
    from specagent.retrieval.ingestor import ingest_folder

    md = tmp_path / "spec.md"
    md.write_text("# Title\n\nContent here.")
    mock_store = MagicMock()
    mock_store.find_existing.return_value = (None, None)
    mock_store.rebuild_fts_index.side_effect = RuntimeError("fts failed")
    mock_emb = MagicMock()
    mock_emb.embed.return_value = iter([[0.1] * 768])
    with (
        patch("specagent.retrieval.ingestor._get_store", return_value=mock_store),
        patch("specagent.retrieval.ingestor.get_embedder", return_value=mock_emb),
        patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[("chunk", "")]),
    ):
        result = await ingest_folder(tmp_path, library="test")
    # Ingest should still succeed even if FTS rebuild fails
    assert result.indexed == 1
    assert result.failed == 0
