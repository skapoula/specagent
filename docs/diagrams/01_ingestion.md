# Ingestion Pipeline State Machine

The ingestion pipeline (`ingestor.py`) is an async 7-step process: read → convert →
postprocess → chunk → embed → build records → store. It supports both single-file
(`ingest()`) and bulk-folder (`ingest_folder()`) modes. Bulk mode uses `asyncio.gather`
with a `Semaphore` for bounded concurrency and defers FTS index rebuilding to a single
call after all files complete.

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> ReadingBytes : ingest(source, library) called

    ReadingBytes --> ReadFailed : OSError
    ReadingBytes --> HashComputing : raw bytes read

    note right of ReadingBytes
        last_modified computed here via path.stat().st_mtime.
        OSError silently falls back to "" — no warning is logged.
        file_type derived from path.suffix here (before hashing).
    end note

    ReadFailed --> [*] : raises IngestionError

    HashComputing --> Deduplicating : SHA-256 computed

    Deduplicating --> Skipped : existing_hash == new_hash
    Deduplicating --> ConvertingToMarkdown : hash differs or new document

    Skipped --> [*] : returns IngestResult(status="skipped")

    state ConvertingToMarkdown {
        [*] --> CheckingDocxOCR
        CheckingDocxOCR --> OCRPath : file_type==docx AND enable_docx_ocr AND groq_api_key non-empty string
        CheckingDocxOCR --> StandardPath : all other files or OCR disabled

        OCRPath --> MarkdownReady : convert_docx_ocr() returns (text, diagrams)\nboth always co-produced; diagrams may be []
        StandardPath --> MarkdownReady : convert(path) via MarkItDown; diagrams stays []

        MarkdownReady --> [*]
    }

    ConvertingToMarkdown --> ConversionFailed : UnsupportedFormatError
    ConvertingToMarkdown --> ConversionFailed : Exception from converter
    ConvertingToMarkdown --> ConversionFailed : empty text after conversion

    ConversionFailed --> [*] : raises UnsupportedFormatError or IngestionError

    ConvertingToMarkdown --> PostprocessingMarkdown : text non-empty

    PostprocessingMarkdown --> StoringDAGs : postprocess() applies 4 transforms;\nthen _extract_title() derives title from first heading or filename

    note right of PostprocessingMarkdown
        _extract_title() scans for the first line starting with '#'.
        Falls back to path.name if none found. Truncated to 200 chars.
        Populates ChunkRecord.title on every chunk.
    end note

    state StoringDAGs {
        [*] --> CheckDAGEnabled
        CheckDAGEnabled --> StoringOCRDiagrams : enable_dag_storage AND diagrams non-empty
        CheckDAGEnabled --> StoringProseDags : enable_dag_storage AND diagrams == []\n(standard .docx; all non-.docx; OCR path when no diagrams found)
        CheckDAGEnabled --> DAGsSkipped : enable_dag_storage = false

        StoringOCRDiagrams --> [*] : best-effort, never raises
        StoringProseDags --> [*] : best-effort, never raises
        DAGsSkipped --> [*]
    }

    StoringDAGs --> WritingReleaseFiles : file_type==docx AND release_paths() non-None

    WritingReleaseFiles --> ReleaseWriteFailed : _mkdir raises IngestionError (PermissionError or OSError)\nor Exception writing .docx or .md copy
    WritingReleaseFiles --> Chunking : files written; release parsed from filename here\n(e.g. 'j' → Rel-19); stored in chunk_meta and ChunkRecord.release
    StoringDAGs --> Chunking : non-docx path, or .docx with no parseable 3GPP release in filename\n(release defaults to 0 for all chunks)

    ReleaseWriteFailed --> [*] : raises IngestionError

    Chunking --> ChunkFailed : Exception from chunk_with_metadata()
    Chunking --> ChunkFailed : zero chunks produced
    Chunking --> Embedding : chunk_pairs list non-empty

    ChunkFailed --> [*] : raises IngestionError

    Embedding --> EmbedFailed : Exception from embed_documents()
    Embedding --> BuildingRecords : embeddings array returned

    EmbedFailed --> [*] : raises IngestionError

    BuildingRecords --> StoringChunks : ChunkRecord list assembled (one per chunk)

    note right of StoringChunks
        upsert_chunks internal sequence (not atomic):
        1. Serialise records to dicts — if this raises, the table is untouched.
        2. DELETE existing rows by doc_id (one DELETE per doc_id in batch).
        3. INSERT new rows via table.add().
        Failure between steps 2 and 3 leaves no chunks for that doc_id.

        FTS rebuild mode:
        ingest() single-file — rebuild_fts=True: FTS rebuilt inline after each write.
        ingest_folder() bulk — rebuild_fts=False: FTS skipped here;
        store.rebuild_fts_index() called once after asyncio.gather() completes,
        but only when at least one candidate file was found (if candidates: guard).
    end note

    StoringChunks --> StoreFailed : Exception from store.upsert_chunks()
    StoringChunks --> DeletingOldVersion : status=="replaced" — delete old doc_id
    StoringChunks --> Done : status=="indexed"

    StoreFailed --> [*] : raises IngestionError

    DeletingOldVersion --> DeleteWarned : Exception on old doc delete (non-fatal)
    DeletingOldVersion --> Done : old version deleted
    DeleteWarned --> Done : continues — new version already written

    Done --> [*] : returns IngestResult(status="indexed"|"replaced")

    note right of Done
        ingest_folder() folder-level step (not part of the per-file pipeline):
        after asyncio.gather() completes across all files,
        store.rebuild_fts_index() is called once — but only when
        candidates was non-empty. Returns True/False; never raises.
    end note
```
