# Implement Retrieval Component

Implement a retrieval pipeline component (chunker, embeddings, or resources).

## Process

1. **Identify component**: chunker | embeddings | resources
2. **Review placeholder** in `src/specagent/retrieval/{component}.py`
3. **Write tests first** in `tests/unit/test_{component}.py`
4. **Implement component** following existing patterns
5. **Verify**:
   ```bash
   pytest tests/unit/test_{component}.py -v
   mypy src/specagent/retrieval/{component}.py
   ```

## Component Specifications

### Chunker (`chunker.py`)
- Use `langchain.text_splitter.RecursiveCharacterTextSplitter`
- Preserve markdown section headers
- Extract spec_id from filename (e.g., "TS38.321.md" → "TS38.321")
- Default chunk_size=512, overlap=64

### Embeddings (`embedder.py`)
- Use `fastembed` for local ONNX embeddings (no API calls needed)
- Implement retry with exponential backoff (use `tenacity`)
- Batch requests (default batch_size=32)
- Normalize vectors for cosine similarity

### Resources (`resources.py`)
- Expose `get_store() -> Store` and `get_embedder() -> TextEmbedding` as `@lru_cache` singletons
- `Store` wraps LanceDB: `upsert_chunks()`, `search()` (hybrid BM25+vector), `delete_document()`
- `TextEmbedding` uses fastembed with `nomic-ai/nomic-embed-text-v1.5` (768d, ONNX, local)
- Expose `clear_resource_cache()` for test teardown (calls `cache_clear()` on both singletons)
- LanceDB URI and table name come from `settings.lancedb_uri` / `settings.lancedb_table_name`

## Dependencies
Only use packages already in pyproject.toml:
- httpx, tenacity, lancedb, fastembed
