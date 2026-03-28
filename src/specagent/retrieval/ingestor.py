"""Orchestrates the ingestion pipeline: read → convert → chunk → embed → store."""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from specagent.config import settings
from specagent.retrieval.chunker import chunk_with_metadata
from specagent.retrieval.converter import SUPPORTED_EXTENSIONS, convert, convert_docx_ocr
from specagent.retrieval.embedder import embed_documents
from specagent.retrieval.exceptions import IngestionError, UnsupportedFormatError
from specagent.retrieval.resources import get_store
from specagent.retrieval.store import ChunkRecord

logger = logging.getLogger(__name__)


class IngestResult(BaseModel):
    """Result returned from a single ingest call."""

    status: str  # "indexed" | "replaced" | "skipped"
    doc_id: str
    source: str
    library: str
    chunk_count: int


class BulkIngestResult(BaseModel):
    """Result returned from a bulk folder ingest call."""

    folder: str
    library: str
    total_files: int
    indexed: int
    replaced: int
    skipped: int
    failed: int
    results: list[IngestResult]
    errors: list[dict]  # [{"file": str, "error": str}]


async def ingest(  # noqa: PLR0912, PLR0915 — pre-existing complexity; pipeline steps need a single function
    source: Path | str,
    library: str,
    metadata: dict | None = None,
    *,
    rebuild_fts: bool = True,
) -> IngestResult:
    """Run the full ingestion pipeline for a local file.

    Args:
        source: Local Path or str path to ingest.
        library: Library name to index the document under.
        metadata: Optional user-supplied key-value metadata.

    Returns:
        IngestResult describing what happened (indexed / replaced / skipped).

    Raises:
        IngestionError: If reading, converting, chunking, or storing fails.
        UnsupportedFormatError: If the file format is not supported.
    """
    store = get_store()
    path = Path(source) if not isinstance(source, Path) else source
    source_str = str(path)

    # ── 1. Read raw bytes ──────────────────────────────────────────────────────
    try:
        raw_bytes = await asyncio.to_thread(path.read_bytes)
    except OSError as e:
        raise IngestionError(f"Cannot read file {source_str!r}") from e

    file_type = path.suffix.lstrip(".").lower() or "unknown"
    try:
        mtime = path.stat().st_mtime
        last_modified = datetime.fromtimestamp(mtime, UTC).isoformat()
    except OSError:
        last_modified = ""

    # ── 2. Dedup check ─────────────────────────────────────────────────────────
    new_hash = hashlib.sha256(raw_bytes).hexdigest()
    existing_doc_id, existing_hash = await asyncio.to_thread(
        store.find_existing, source_str, library
    )

    if existing_hash == new_hash:
        logger.info("Skipping %s — content unchanged (hash=%s)", source_str, new_hash[:8])
        return IngestResult(
            status="skipped",
            doc_id=existing_doc_id or "",
            source=source_str,
            library=library,
            chunk_count=0,
        )

    ingest_status = "replaced" if existing_doc_id is not None else "indexed"

    # ── 3. Convert to Markdown ─────────────────────────────────────────────────
    try:
        if file_type == "docx" and settings.enable_docx_ocr and settings.groq_api_key:
            text = await convert_docx_ocr(path, api_key=settings.groq_api_key)
        else:
            text = await asyncio.to_thread(convert, path)
    except UnsupportedFormatError:
        raise
    except Exception as e:
        raise IngestionError(f"Conversion failed for {source_str!r}") from e

    if not text.strip():
        raise IngestionError(f"No text could be extracted from {source_str!r}")

    title = _extract_title(text, source_str)

    # ── 4. Chunk, extracting section headers per chunk ─────────────────────────
    try:
        chunk_pairs = await asyncio.to_thread(chunk_with_metadata, text)
    except Exception as e:
        raise IngestionError(f"Chunking failed for {source_str!r}") from e

    if not chunk_pairs:
        raise IngestionError(f"No usable chunks produced from {source_str!r}")

    chunk_texts = [ct for ct, _ in chunk_pairs]
    section_headers = [sh for _, sh in chunk_pairs]

    # ── 5. Embed ───────────────────────────────────────────────────────────────
    try:
        embeddings = list(await asyncio.to_thread(embed_documents, chunk_texts))
    except Exception as e:
        raise IngestionError(f"Embedding failed for {source_str!r}") from e

    # ── 6. Build records with per-chunk section header in metadata ─────────────
    doc_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    records = []
    for i, (chunk_text, section_header) in enumerate(
        zip(chunk_texts, section_headers, strict=True)
    ):
        chunk_meta = dict(metadata or {})
        chunk_meta["section_header"] = section_header
        emb = embeddings[i]
        records.append(
            ChunkRecord(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                library=library,
                source=source_str,
                content_hash=new_hash,
                title=title,
                content=chunk_text,
                embedding=list(emb),
                chunk_index=i,
                created_at=now,
                metadata=json.dumps(chunk_meta),
                file_type=file_type,
                last_modified=last_modified,
                page=0,
            )
        )

    try:
        await asyncio.to_thread(store.upsert_chunks, records, rebuild_fts=rebuild_fts)
    except Exception as e:
        raise IngestionError(f"Store write failed for {source_str!r}") from e

    # ── 7. Delete old version only after new write succeeds ────────────────────
    if ingest_status == "replaced" and existing_doc_id is not None:
        logger.info(
            "Replacing %s in library %r — deleting old doc_id=%s",
            source_str,
            library,
            existing_doc_id,
        )
        try:
            await asyncio.to_thread(store.delete_document, existing_doc_id)
        except Exception:
            logger.warning(
                "New version of %s written (doc_id=%s) but old doc_id=%s could not be deleted",
                source_str,
                doc_id,
                existing_doc_id,
            )

    logger.info(
        "%s %s → %d chunks in library %r (doc_id=%s)",
        ingest_status,
        source_str,
        len(records),
        library,
        doc_id,
    )
    return IngestResult(
        status=ingest_status,
        doc_id=doc_id,
        source=source_str,
        library=library,
        chunk_count=len(records),
    )


async def _ingest_one(
    sem: asyncio.Semaphore,
    file_path: Path,
    library: str,
    metadata: dict | None,
) -> IngestResult:
    """Ingest one file under a concurrency semaphore."""
    async with sem:
        return await ingest(source=file_path, library=library, metadata=metadata, rebuild_fts=False)


async def ingest_folder(
    folder: Path | str,
    library: str,
    metadata: dict | None = None,
    recursive: bool = True,
    max_concurrency: int = 4,
) -> BulkIngestResult:
    """Ingest all supported documents in a folder concurrently.

    Args:
        folder: Path to the folder to scan.
        library: Library name to index documents under.
        metadata: Optional user-supplied key-value metadata.
        recursive: Whether to scan subdirectories recursively.
        max_concurrency: Maximum files to ingest simultaneously.

    Returns:
        BulkIngestResult with per-file results and error summary.

    Raises:
        IngestionError: If the folder path does not exist or is not a directory.
    """
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise IngestionError(f"Folder not found or not a directory: {str(folder)!r}")

    pattern = "**/*" if recursive else "*"
    candidates = sorted(
        p
        for p in folder_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not candidates:
        logger.warning(
            "No supported files found in %s (recursive=%s). Supported: %s",
            folder_path,
            recursive,
            ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )

    sem = asyncio.Semaphore(max(1, max_concurrency))
    raw = await asyncio.gather(
        *[_ingest_one(sem, p, library, metadata) for p in candidates],
        return_exceptions=True,
    )

    results: list[IngestResult] = []
    errors: list[dict] = []
    for file_path, outcome in zip(candidates, raw, strict=True):
        if isinstance(outcome, BaseException):
            logger.warning("Failed to ingest %s: %s", file_path, outcome)
            errors.append({"file": str(file_path), "error": str(outcome)})
        else:
            results.append(outcome)

    # Rebuild FTS index once after all writes/deletes are complete
    # (avoids O(N²) rebuilds for bulk ingest; runs even on all-skip runs
    # so that external deletes are reflected in hybrid search)
    if candidates:
        try:
            await asyncio.to_thread(get_store().rebuild_fts_index)
        except Exception as fts_err:
            logger.warning("FTS index rebuild after ingest_folder failed: %s", fts_err)

    return BulkIngestResult(
        folder=str(folder_path),
        library=library,
        total_files=len(candidates),
        indexed=sum(1 for r in results if r.status == "indexed"),
        replaced=sum(1 for r in results if r.status == "replaced"),
        skipped=sum(1 for r in results if r.status == "skipped"),
        failed=len(errors),
        results=results,
        errors=errors,
    )


def _extract_title(text: str, source: str) -> str:
    """Infer a document title from the first Markdown heading or source path."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:200]
    return source.split("/")[-1].split("\\")[-1][:200]
