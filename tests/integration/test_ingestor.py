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
