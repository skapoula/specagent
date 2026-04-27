# PLAN: Fix All Issues in `docs/diagrams/01_ingestion.md`

**Status:** Ready for implementation  
**Target file:** `docs/diagrams/01_ingestion.md`  
**Scope:** Diagram-only — no source code changes required.

---

## Source evidence

All line numbers verified against current source before prescribing fixes.

| Issue              | Source location                                                            | Finding                                                                                                                                                                                                                                                                                    |
| ------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| #1                 | `ingestor.py:128-129`, `ingestor.py:188`                                   | `_extract_title()` runs immediately after `postprocess()`; `title` populates every `ChunkRecord.title`                                                                                                                                                                                     |
| #2                 | `ingestor.py:140-144`, `ingestor.py:178-179`, `ingestor.py:197`            | `release=0` default set at line 140; overwritten at line 144 inside `if paths is not None:` only; propagates into `chunk_meta["release"]` and `ChunkRecord.release` for all chunks                                                                                                         |
| #3                 | `ingestor.py:114`, `ingestor.py:117`                                       | `text, diagrams = await convert_docx_ocr(...)` — single tuple return; no code path yields one without the other                                                                                                                                                                            |
| #4                 | `ingestor.py:114`, `ingestor.py:119`, `ingestor.py:132`, `ingestor.py:136` | `diagrams = []` on standard path; `not diagrams` is `True` for standard `.docx`, all non-`.docx` files, **and** OCR path when no diagrams found — all three reach `StoringProseDags`                                                                                                       |
| #5                 | `ingestor.py:116`                                                          | Guard is `if settings.groq_api_key:` — Python truthiness; empty string `""` is falsy and routes to `StandardPath`                                                                                                                                                                          |
| #6                 | `store.py:265-267`, `store.py:289-297`                                     | `upsert_chunks` serialises rows first (safe), then `table.delete()` per doc_id, then `table.add()` — explicit delete-then-insert, not atomic                                                                                                                                               |
| #7                 | `ingestor.py:251`, `ingestor.py:316-320`, `store.py:403-419`               | `_ingest_one` passes `rebuild_fts=False`; `ingest_folder` calls `store.rebuild_fts_index()` once after `asyncio.gather`, guarded by `if candidates:`                                                                                                                                       |
| #8                 | `ingestor.py:416-417`, `ingestor.py:430-433`                               | `_write_release_files` calls `_mkdir()` before any file copy; `_mkdir` raises `IngestionError` on both `PermissionError` and `OSError`                                                                                                                                                     |
| #9                 | `ingestor.py:89-93`                                                        | `last_modified` is computed inside the "Read raw bytes" step (step 1 per pipeline comment at line 82); `OSError` silently falls back to `""` with no log                                                                                                                                   |
| #10 (pre-existing) | `ingestor.py:141`, `spec_filename.py:49-51`                                | `release_paths()` returns `None` for any `.docx` whose stem has no parseable 3GPP release; the file skips `WritingReleaseFiles` and goes directly to `Chunking` — same path as non-`.docx` files, but the existing `StoringDAGs --> Chunking : non-docx path` label is wrong for this case |

---

## Edit blocks

### Block A — Issues #1, #2, #10: postprocessing note, `release` propagation, `.docx`-no-release path

**A1** — Extend the `PostprocessingMarkdown → StoringDAGs` transition label:

```
PostprocessingMarkdown --> StoringDAGs : postprocess() applies 4 transforms;\nthen _extract_title() derives title from first heading or filename
```

**A2** — Add a note on `PostprocessingMarkdown` immediately after A1:

```
note right of PostprocessingMarkdown
    _extract_title() scans for the first line starting with '#'.
    Falls back to path.name if none found. Truncated to 200 chars.
    Populates ChunkRecord.title on every chunk.
end note
```

**A3** — Fix `WritingReleaseFiles → Chunking` label (preserve the conditional; add `release` note):

```
WritingReleaseFiles --> Chunking : files written; release parsed from filename here\n(e.g. 'j' → Rel-19); stored in chunk_meta and ChunkRecord.release
```

**A4** — Fix the pre-existing `StoringDAGs → Chunking` label (issue #10):

```
StoringDAGs --> Chunking : non-docx path, or .docx with no parseable 3GPP release in filename\n(release defaults to 0 for all chunks)
```

---

### Block B — Issue #3: `DiagramsExtracted` co-occurrence

**B1** — Replace the two OCR-path transitions with one annotated transition:

```
OCRPath --> MarkdownReady : convert_docx_ocr() returns (text, diagrams)\nboth always co-produced; diagrams may be []
```

**B2** — Annotate the standard path to make `diagrams = []` explicit:

```
StandardPath --> MarkdownReady : convert(path) via MarkItDown; diagrams stays []
```

**B3** — Delete both of these lines (the state and the transition that feeds it):

```
OCRPath --> DiagramsExtracted : OCR path also yields diagrams list   ← removed by B1
DiagramsExtracted --> [*]                                             ← delete this line
```

---

### Block C — Issue #4: `StoringProseDags` fires in three cases, not two

**C1** — Replace the `CheckDAGEnabled → StoringProseDags` guard:

```
CheckDAGEnabled --> StoringProseDags : enable_dag_storage AND diagrams == []\n(standard .docx; all non-.docx; OCR path when no diagrams found)
```

---

### Block D — Issue #5: `groq_api_key` is a truthiness check, not a boolean flag

**D1** — Fix the `CheckingDocxOCR → OCRPath` guard:

```
CheckingDocxOCR --> OCRPath : file_type==docx AND enable_docx_ocr AND groq_api_key non-empty string
```

---

### Block E — Issues #6 and #7 (merged): single `note right of StoringChunks`

> **Why merged:** `stateDiagram-v2` does not allow two `note right of` blocks for the same state. Both annotations must live in one note.

**E1** — Add a single `note right of StoringChunks` covering the delete-then-insert sequence (issue #6) and the FTS rebuild mode split (issue #7):

```
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
```

---

### Block F — Issue #7: `ingest_folder` FTS rebuild is folder-level, not per-file

> **Why `Done --> RebuildingFTS` is wrong:** `Done` is the terminal of an individual file's pipeline. The FTS rebuild fires at the `ingest_folder` orchestration level, after `asyncio.gather` across all files. Expressing it as a state transition from `Done` implies every file triggers a rebuild and makes `Done --> [*]` unreachable on the bulk path — both are wrong.

**F1** — Add `note right of Done` instead of a state transition:

```
note right of Done
    ingest_folder() folder-level step (not part of the per-file pipeline):
    after asyncio.gather() completes across all files, store.rebuild_fts_index()
    is called once — but only when candidates was non-empty.
    Returns True on success, False on failure; never raises.
end note
```

---

### Block G — Issues #8 and #9: `_mkdir` error modes and `last_modified` fallback

**G1** — Add note on `ReadingBytes` (not `HashComputing` — `last_modified` is computed in the "Read raw bytes" step, lines 89-93, before hash computation begins at line 96):

```
note right of ReadingBytes
    last_modified computed here via path.stat().st_mtime.
    OSError silently falls back to "" — no warning is logged.
    file_type derived from path.suffix here (before hashing).
end note
```

**G2** — Extend the `WritingReleaseFiles → ReleaseWriteFailed` label to include `_mkdir` failures:

```
WritingReleaseFiles --> ReleaseWriteFailed : _mkdir raises IngestionError (PermissionError or OSError)\nor Exception writing .docx or .md copy
```

---

## Implementation checklist (15 atomic changes)

- [ ] **A1** — Extend `PostprocessingMarkdown → StoringDAGs` label to include `_extract_title()`
- [ ] **A2** — Insert `note right of PostprocessingMarkdown` block
- [ ] **A3** — Fix `WritingReleaseFiles → Chunking` label: preserve conditional, add release propagation note
- [ ] **A4** — Fix `StoringDAGs → Chunking` label: add `.docx` with no parseable release case
- [ ] **B1** — Replace `OCRPath → MarkdownReady` + `OCRPath → DiagramsExtracted` with single co-occurrence transition
- [ ] **B2** — Annotate `StandardPath → MarkdownReady` with `diagrams stays []`
- [ ] **B3** — Delete `DiagramsExtracted --> [*]` line (B1 removes the feeding transition)
- [ ] **C1** — Replace `StoringProseDags` guard to cover all three cases including OCR-with-empty-diagrams
- [ ] **D1** — Change `groq_api_key set` → `groq_api_key non-empty string`
- [ ] **E1** — Insert single merged `note right of StoringChunks` (delete-then-insert + FTS modes)
- [ ] **F1** — Add `note right of Done` for folder-level FTS rebuild; do NOT add `Done --> RebuildingFTS`
- [ ] **G1** — Insert `note right of ReadingBytes` for `last_modified` and `file_type` computation
- [ ] **G2** — Prepend `_mkdir` error modes to `WritingReleaseFiles → ReleaseWriteFailed` label

---

## Complete corrected diagram source

Replace the full contents of `/workspace/specagent/docs/diagrams/01_ingestion.md` with:

````markdown
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
````
