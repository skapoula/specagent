"""Tests for ingestor helpers using real docx files — no pipeline mocks."""

import pytest

from specagent.retrieval.exceptions import IngestionError
from specagent.retrieval.ingestor import (
    _extract_title,
    _mkdir,
    _read_last_modified,
    _write_release_files,
)
from tests.conftest import DOCX_SMALL

# ---------------------------------------------------------------------------
# _read_last_modified — filesystem stat with OSError fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_last_modified_returns_iso_string(tmp_path):
    f = tmp_path / "doc.docx"
    f.write_bytes(b"x")
    result = _read_last_modified(f)
    assert result  # non-empty
    assert "T" in result  # ISO 8601 datetime separator


@pytest.mark.unit
def test_read_last_modified_falls_back_to_empty_string_and_warns(tmp_path, caplog):
    import logging
    from unittest.mock import patch

    f = tmp_path / "missing.docx"
    with (
        patch("pathlib.Path.stat", side_effect=OSError("no stat")),
        caplog.at_level(logging.WARNING, logger="specagent.retrieval.ingestor"),
    ):
        result = _read_last_modified(f)

    assert result == ""
    assert any("last_modified" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _extract_title — pure string function, no I/O
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_title_returns_first_heading():
    assert _extract_title("# My Title\n\nContent.", "/f.docx") == "My Title"


@pytest.mark.unit
def test_extract_title_strips_multiple_hashes():
    assert _extract_title("## Section 2\n\nContent.", "/f.docx") == "Section 2"


@pytest.mark.unit
def test_extract_title_falls_back_to_filename():
    assert _extract_title("no heading here", "/docs/spec.docx") == "spec.docx"


@pytest.mark.unit
def test_extract_title_truncates_long_heading():
    long = "A" * 300
    result = _extract_title(f"# {long}\n\nContent.", "/f.docx")
    assert len(result) <= 200


# ---------------------------------------------------------------------------
# _mkdir — pure filesystem function, no pipeline involvement
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mkdir_creates_nested_directory(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    _mkdir(target)
    assert target.is_dir()


@pytest.mark.unit
def test_mkdir_tolerates_existing_directory(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    _mkdir(target)  # must not raise
    assert target.is_dir()


@pytest.mark.unit
def test_mkdir_raises_ingestion_error_on_permission_denied(tmp_path):
    from unittest.mock import patch

    with (
        patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")),
        pytest.raises(IngestionError, match="Permission denied"),
    ):
        _mkdir(tmp_path / "new_dir")


@pytest.mark.unit
def test_mkdir_raises_ingestion_error_on_oserror(tmp_path):
    from unittest.mock import patch

    with (
        patch("pathlib.Path.mkdir", side_effect=OSError("filesystem error")),
        pytest.raises(IngestionError, match="Failed to create directory"),
    ):
        _mkdir(tmp_path / "new_dir")


# ---------------------------------------------------------------------------
# _write_release_files — uses real DOCX_SMALL as source file
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_release_files_copies_real_docx_and_writes_md(tmp_path):
    """_write_release_files copies the real .docx and writes the markdown text."""
    docx_dest = tmp_path / "3gpp_rel_18" / "docx" / "38108-i40_rel18.docx"
    md_dest = tmp_path / "3gpp_rel_18" / "md" / "38108-i40_rel18.md"
    text = "# 3GPP TS 38.108\n\nSome converted content."

    _write_release_files(DOCX_SMALL, text, docx_dest, md_dest)

    assert docx_dest.exists(), "docx copy must exist at the release destination"
    assert docx_dest.stat().st_size == DOCX_SMALL.stat().st_size
    assert md_dest.read_text(encoding="utf-8") == text


@pytest.mark.unit
def test_write_release_files_skips_copy_when_same_resolved_path(tmp_path):
    """When source and docx_dest resolve to the same path, no copy is attempted."""
    # Use DOCX_SMALL itself as both source and destination
    md_dest = tmp_path / "38108-i40.md"
    original_mtime = DOCX_SMALL.stat().st_mtime

    _write_release_files(DOCX_SMALL, "text", DOCX_SMALL, md_dest)

    # The source file must be completely unchanged
    assert DOCX_SMALL.stat().st_mtime == original_mtime
    assert md_dest.read_text() == "text"


@pytest.mark.unit
def test_write_release_files_creates_parent_directories(tmp_path):
    """_write_release_files creates parent directories that do not yet exist."""
    docx_dest = tmp_path / "deep" / "nested" / "path" / "dest.docx"
    md_dest = tmp_path / "other" / "nested" / "dest.md"

    _write_release_files(DOCX_SMALL, "# Title", docx_dest, md_dest)

    assert docx_dest.exists()
    assert md_dest.exists()


# ---------------------------------------------------------------------------
# End-to-end ingest() with real docx file and real store (no pipeline mocks)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ingest_real_docx_indexes_successfully(tmp_path):
    """ingest() on a real 3GPP .docx produces a non-empty indexed result."""
    from specagent.retrieval.ingestor import ingest
    from specagent.retrieval.store import Store

    store = Store(uri=str(tmp_path / "db"), table_name="docs")

    from unittest.mock import patch

    with patch("specagent.retrieval.ingestor.get_store", return_value=store):
        result = await ingest(source=DOCX_SMALL, library="test-real")

    assert result.status == "indexed"
    assert result.chunk_count > 0
    assert result.source == str(DOCX_SMALL)
    assert result.library == "test-real"


@pytest.mark.integration
async def test_ingest_real_docx_dedup_skips_unchanged(tmp_path):
    """Ingesting the same real .docx twice returns 'skipped' on the second call."""
    from specagent.retrieval.ingestor import ingest
    from specagent.retrieval.store import Store

    store = Store(uri=str(tmp_path / "db"), table_name="docs")

    from unittest.mock import patch

    with patch("specagent.retrieval.ingestor.get_store", return_value=store):
        first = await ingest(source=DOCX_SMALL, library="test-real")
        second = await ingest(source=DOCX_SMALL, library="test-real")

    assert first.status == "indexed"
    assert second.status == "skipped"
    assert second.chunk_count == 0


@pytest.mark.integration
async def test_ingest_real_docx_stores_release_number(tmp_path):
    """Ingesting a 3GPP .docx tagged with rel-18 sets release=18 on all ChunkRecords."""
    import json

    from specagent.retrieval.ingestor import ingest
    from specagent.retrieval.store import Store

    store = Store(uri=str(tmp_path / "db"), table_name="docs")

    from unittest.mock import patch

    with (
        patch("specagent.retrieval.ingestor.get_store", return_value=store),
        patch("specagent.retrieval.ingestor.settings") as mock_settings,
    ):
        mock_settings.enable_docx_ocr = False
        mock_settings.groq_api_key = None
        mock_settings.enable_dag_storage = False
        mock_settings.data_dir = tmp_path
        result = await ingest(source=DOCX_SMALL, library="test-real")

    assert result.status == "indexed"
    chunks = store.get_document(result.doc_id)
    assert all(c.release == 18 for c in chunks)
    assert all("release" in json.loads(c.metadata) for c in chunks)


@pytest.mark.integration
async def test_ingest_real_docx_section_headers_populated(tmp_path):
    """Every ChunkRecord has a non-empty section_header in its metadata."""
    import json

    from specagent.retrieval.ingestor import ingest
    from specagent.retrieval.store import Store

    store = Store(uri=str(tmp_path / "db"), table_name="docs")

    from unittest.mock import patch

    with patch("specagent.retrieval.ingestor.get_store", return_value=store):
        result = await ingest(source=DOCX_SMALL, library="test-real")

    chunks = store.get_document(result.doc_id)
    headers = [json.loads(c.metadata).get("section_header", "") for c in chunks]
    # At least some chunks must have a non-empty section header
    assert any(h for h in headers), "Expected at least one chunk with a section header"
