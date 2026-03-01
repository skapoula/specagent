# Design: Replace Ingestion & Indexing Pipeline with mcpvectordb

**Date:** 2026-03-01
**Approach:** B — Unified Data Model
**Status:** Approved

---

## Context

specagent currently uses FAISS (384d, CPU in-memory) backed by `sentence-transformers/all-MiniLM-L6-v2`
for its ingestion and retrieval pipeline. The HuggingFace TSpec-LLM dataset provides the source .md files.

This design replaces that pipeline with the one implemented in mcpvectordb, which provides:
- LanceDB (768d, embedded, persistent on disk)
- `nomic-ai/nomic-embed-text-v1.5` via ONNX/fastembed (no PyTorch required for embeddings)
- Token-aware chunking with section-header extraction
- Hybrid BM25 + vector search (LanceDB FTS + ANN)
- SHA256 deduplication (skip/replace on re-ingest)
- Async ingestion pipeline with concurrency control

The user will supply .docx files instead of the HuggingFace dataset.
All existing RAG features (grader thresholds, citation extraction, rewriting, hallucination check) are preserved.

---

## Files Deleted

| File | Reason |
|------|--------|
| `retrieval/indexer.py` | FAISSIndex — replaced by LanceDB Store |
| `retrieval/embeddings.py` | sentence-transformers embedders — replaced by fastembed |
| `retrieval/data_ingestion.py` | HuggingFace TSpec-LLM downloader — user supplies files |

---

## Files Replaced In-Place (new content from mcpvectordb)

| File | Source | Modifications |
|------|--------|---------------|
| `retrieval/converter.py` | mcpvectordb verbatim | None |
| `retrieval/chunker.py` | mcpvectordb + specagent | Add `_extract_section_headers()` helper (ported from current specagent chunker); inject `section_header` into each chunk's metadata dict before JSON-serializing into `ChunkRecord.metadata` |
| `retrieval/embedder.py` | mcpvectordb verbatim | None |
| `retrieval/store.py` | mcpvectordb verbatim | None |
| `retrieval/resources.py` | Rewritten | Two singletons: `get_store() -> Store`, `get_embedder() -> TextEmbedding` |

---

## Files Added

| File | Source |
|------|--------|
| `retrieval/ingestor.py` | mcpvectordb verbatim |
| `retrieval/exceptions.py` | mcpvectordb verbatim |

---

## Files Modified (surgical updates)

### `graph/state.py`

Replace `RetrievedChunk` dataclass with unified model:

```python
@dataclass
class RetrievedChunk:
    # Core content
    content: str

    # LanceDB provenance (direct from ChunkRecord)
    chunk_id: str        # ChunkRecord.id (UUID4)
    doc_id: str          # ChunkRecord.doc_id
    source: str          # full file path (was: source_file)
    title: str           # document title (first heading)
    chunk_index: int     # position within document
    file_type: str       # "docx", "pdf", etc.

    # Derived at retrieval time
    spec_id: str         # from source filename stem: "TS38.321"
    section: str         # from metadata["section_header"]

    # Search quality
    similarity_score: float
```

`GradedChunk` and `GraphState` are unchanged — they reference `RetrievedChunk` by name.

### `nodes/retriever.py`

Replace `get_faiss_index()` + `index.search()` with `get_store()` + `store.search()`.

The retriever calls `get_embedder()` to embed the query with the `"search_query: "` prefix
(required by nomic-embed-text-v1.5's asymmetric search design), then calls `store.search()`.

For each `ChunkRecord` in results:
- `spec_id` = `Path(record.source).stem` (same derivation as today)
- `section` = `json.loads(record.metadata).get("section_header", "")` (new field)
- All other fields map directly

The embedder is **not** kept as a resource singleton alongside the store — fastembed's
`TextEmbedding` is already thread-safe and caches the ONNX model internally; wrapping it
in `lru_cache` is sufficient.

### `nodes/grader.py`

Rename `chunk.source_file` → `chunk.source` throughout. No other changes.
Similarity thresholds (0.55, 0.82, 0.85) remain in place as candidates for retuning
after the first evaluation run with the new embedder.

### `nodes/generator.py`

Rename `chunk.source_file` → `chunk.source` throughout. No other changes.
Citation regex `r'\[TS\s+(\d+\.\d+(?:-\d+)?)\s+§\s*([0-9A-Za-z.]+)\]'` unchanged.

### `config.py`

**Removed settings:**
- `faiss_index_path`, `metadata_path`
- `use_local_embeddings`
- `embedding_model` default (`all-MiniLM-L6-v2`)

**Added settings:**

| Setting | Default | Purpose |
|---------|---------|---------|
| `LANCEDB_URI` | `data/lancedb` | LanceDB storage path |
| `LANCEDB_TABLE_NAME` | `documents` | Table name |
| `DEFAULT_LIBRARY` | `3gpp-specs` | Library name for 3GPP corpus |
| `DOCS_DIR` | `data/docs` | Where user places input files |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Updated model ID |
| `EMBEDDING_DIMENSION` | `768` | Updated vector dimension |
| `HYBRID_SEARCH_ENABLED` | `true` | BM25 + vector hybrid |
| `SEARCH_REFINE_FACTOR` | `10` | ANN re-rank candidates |
| `CHUNK_SIZE_TOKENS` | `512` | Chunk target (token-based now) |
| `CHUNK_OVERLAP_TOKENS` | `64` | Overlap (token-based now) |
| `CHUNK_MIN_TOKENS` | `50` | Minimum chunk size |
| `EMBEDDING_BATCH_SIZE` | `32` | fastembed batch size |
| `HTTP_TIMEOUT_SECONDS` | `10` | URL fetch timeout (ingestor) |

### `cli.py`

Replace `specagent index` command:

```bash
# Old:
specagent index [--download] [--force] [--use-git]

# New:
specagent index [--docs-dir PATH] [--library NAME] [--force] [--max-concurrency N]
```

- `--docs-dir`: directory containing .docx (and other) files (default: `DOCS_DIR` from config)
- `--library`: LanceDB library name (default: `DEFAULT_LIBRARY`)
- `--force`: delete existing library, then re-index all files
- `--max-concurrency`: parallel ingestion workers (default: 4)
- Implementation: calls `asyncio.run(ingest_folder(...))` from `retrieval/ingestor.py`

The `--download` flag is removed. `data_ingestion.py` is deleted.

### `pyproject.toml`

```toml
# Remove:
"faiss-cpu>=1.7"
"sentence-transformers>=2.7"
"langchain-text-splitters>=0.2"

# Add:
"lancedb>=0.13"
"fastembed>=0.3"
"pyarrow>=15"
"tantivy>=0.22"
```

---

## Data Flow (After)

```
User places .docx files in data/docs/
          ↓
specagent index
          ↓
ingest_folder(data/docs/, library="3gpp-specs")
  For each file (up to 4 concurrent):
    1. Read bytes → SHA256 → dedup check
    2. MarkItDown → markdown string
    3. Token-aware chunking → list[str]
       + section_header extraction per chunk
    4. fastembed.embed_documents() → NDArray[float32, 768d]
    5. Build ChunkRecord(s) with metadata={"section_header": "..."}
    6. LanceDB upsert → FTS index rebuild
          ↓
LanceDB: data/lancedb/documents table
```

**Query Flow (After):**

```
POST /query {"question": "..."}
          ↓
retriever_node:
  fastembed.embed_query("search_query: " + question) → 768d vector
  store.search(embedding, query_text, top_k=10, library="3gpp-specs")
    → hybrid BM25 + ANN → list[ChunkRecord]
  → map to list[RetrievedChunk] (derive spec_id, deserialize section_header)
          ↓
grader_node (unchanged logic, field rename only)
          ↓
generator_node (unchanged logic, field rename only)
          ↓
hallucination_check_node (unchanged)
```

---

## What is NOT Changed

- LangGraph workflow structure (`graph/workflow.py`) — unchanged
- Router node (`nodes/router.py`) — unchanged
- Grader scoring logic — unchanged (thresholds may need retuning post-deployment)
- Generator prompt and citation regex — unchanged
- Hallucination check node — unchanged
- API endpoints (`api/main.py`, `api/models.py`) — unchanged
- LLM factory (`llm/factory.py`) — unchanged
- All tests structure — updated to match new schema

---

## Data Layout

```
specagent/
└── data/
    ├── docs/          ← user places .docx (and other) files here
    └── lancedb/       ← LanceDB embedded database (gitignored)
```

---

## Similarity Threshold Retuning Note

The grader node uses cosine similarity thresholds tuned for `all-MiniLM-L6-v2` (384d):
- Auto-grade "yes" if score > 0.82
- Auto-grade "no" if score < 0.55
- Skip grading if top-3 similarity ≥ 0.85

`nomic-embed-text-v1.5` (768d) generally produces higher cosine similarity for relevant
content. After the first benchmark run, evaluate whether thresholds need adjustment.
These are the only values likely to need tuning post-migration.
