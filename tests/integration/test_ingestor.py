"""Integration tests for the ingestion pipeline."""
import hashlib
import json
import pytest
from unittest.mock import MagicMock, patch


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
        patch("specagent.retrieval.ingestor._get_embedder") as mock_emb_fn,
        patch("specagent.retrieval.ingestor.chunk_with_metadata") as mock_chunk,
    ):
        mock_chunk.return_value = [("# Section 5.4\n\nContent about HARQ processes.", "Section 5.4")]
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
        patch("specagent.retrieval.ingestor._get_embedder") as mock_emb_fn,
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
        patch("specagent.retrieval.ingestor._get_embedder") as mock_emb_fn,
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
def test_get_embedder_singleton():
    from specagent.retrieval import ingestor

    ingestor._embedder = None
    with patch("fastembed.TextEmbedding") as mock_cls:
        mock_cls.return_value = MagicMock()
        e1 = ingestor._get_embedder()
        e2 = ingestor._get_embedder()
    assert e1 is e2
    ingestor._embedder = None


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
    with patch("specagent.retrieval.ingestor._get_store", return_value=mock_store), \
         patch("specagent.retrieval.ingestor.convert", return_value=""):
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
    with patch("specagent.retrieval.ingestor._get_store", return_value=mock_store), \
         patch("specagent.retrieval.ingestor.convert", return_value="text"), \
         patch("specagent.retrieval.ingestor.chunk_with_metadata", return_value=[]):
        with pytest.raises(IngestionError, match="No usable chunks"):
            await ingest(md, library="test")


@pytest.mark.integration
async def test_ingest_folder_bad_path():
    from specagent.retrieval.exceptions import IngestionError
    from specagent.retrieval.ingestor import ingest_folder

    with pytest.raises(IngestionError, match="not found"):
        await ingest_folder("/nonexistent/xyz", library="test")
