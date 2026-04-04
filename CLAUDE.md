# SpecAgent

> **Scope:** PROJECT-LEVEL — inherits org-wide policy from `/workspace/CLAUDE.md`.
> Rules here extend or override global where they conflict.
> Personal overrides go in `CLAUDE.local.md` (auto-gitignored).

<!-- Global context loaded automatically via directory traversal — no import needed. -->

Agentic RAG system for 3GPP telecommunications specifications. Helps telecom engineers query Release 18 specs using natural language.

## Project Goal

Build a question-answering system that:
- Achieves **85%+ accuracy** on TSpec-LLM benchmark (baseline naive RAG: 71-75%)
- Responds in **<3 seconds** (P95)
- Provides **traceable citations** to source specifications

## Tech Stack

- **Framework**: LangGraph for agentic orchestration
- **Vector Store**: LanceDB (embedded, persistent, hybrid BM25+vector)
- **Embeddings**: `nomic-ai/nomic-embed-text-v1.5` via fastembed (ONNX, 768d, local)
- **LLM**: `meta-llama/llama-4-scout-17b-16e-instruct` via Groq cloud API (or local GGUF)
- **API**: FastAPI
- **Observability**: Arize Phoenix with OpenTelemetry
- **Evaluation**: RAGAS metrics

## Project Structure

```
src/specagent/
├── config.py         # Pydantic Settings (all env vars, @lru_cache singleton)
├── cli.py            # Typer CLI (serve, query, index, benchmark, download-model, version)
├── nodes/            # LangGraph nodes (router, retriever, grader, rewriter, generator, hallucination)
├── graph/
│   ├── state.py      # GraphState TypedDict, RetrievedChunk, GradedChunk, Citation dataclasses
│   └── workflow.py   # build_graph(), run_query(), conditional edges, timing wrappers
├── retrieval/
│   ├── store.py      # LanceDB Store (ChunkRecord schema, hybrid BM25+vector search, CRUD)
│   ├── ingestor.py   # Async 7-step ingest pipeline (read→dedup→convert→chunk→embed→write→delete-old)
│   ├── chunker.py    # Token-aware recursive chunker with section header propagation
│   ├── embedder.py   # embed_documents() / embed_query() with nomic prefix handling
│   ├── converter.py  # MarkItDown-based file-to-Markdown (22 supported extensions)
│   ├── resources.py  # @lru_cache singletons: get_store(), get_embedder()
│   └── exceptions.py # Domain exceptions: UnsupportedFormatError, IngestionError, StoreError, EmbeddingError
├── llm/
│   ├── factory.py    # create_llm() dispatching to Groq or custom endpoint
│   └── custom_endpoint.py  # CustomEndpointLLM (OpenAI-compatible, retry, @traceable)
├── api/
│   ├── main.py       # FastAPI app, /health + /query endpoints, lifespan startup
│   └── models.py     # QueryRequest, QueryResponse, HealthResponse Pydantic schemas
├── observability/
│   ├── models.py     # LLMCallRecord, RetrievalRecord, QueryEvent
│   ├── journal.py    # Thread-safe rotating JSONL journal (QueryJournal)
│   └── report.py     # build_query_report() / log_report(): per-query metrics summary
├── evaluation/
│   ├── benchmark.py  # TSpec-LLM runner + LLM judge, JSON+MD report output
│   └── metrics.py    # RAGAS metrics
└── tracing/
    ├── phoenix.py    # Arize Phoenix + OpenTelemetry setup
    ├── langsmith.py  # LangSmith tracing setup
    └── rag_spans.py  # emit_retrieval_span(), emit_llm_usage_span(), emit_query_span()
```

## Commands

```bash
# Development
pip install -e ".[dev,eval]"        # Install with dev + eval dependencies
pytest                               # Run all tests (unit + integration, excludes slow)
pytest -m unit                       # Run unit tests only
pytest -m integration                # Run integration tests (real LanceDB)
pytest -m e2e                        # Run end-to-end pipeline tests
pytest --cov=src/specagent           # Run with coverage report
ruff check src/ tests/               # Lint
ruff format src/ tests/              # Format
mypy src/specagent                   # Type check

# Application
specagent serve                      # Start FastAPI server (port 8000)
specagent query "question"           # Run single query
specagent query "question" --verbose # Run with timing + metadata output
specagent index                      # Build LanceDB index from data/docs/
specagent index --docs-dir PATH      # Ingest from custom directory
specagent index --library NAME       # Tag chunks with a library name
specagent index --force              # Re-ingest even if content hash unchanged
specagent index --max-concurrency 4  # Control parallel ingest workers (default: 4)
specagent benchmark                  # Run TSpec-LLM evaluation
specagent download-model             # Download ONNX embedding model to local cache
specagent version                    # Print version string

# Docker
docker-compose up                    # Start API (port 8000) + Phoenix (port 6006)
docker-compose --profile ui up       # Also start Gradio UI (port 7860)
docker-compose up -d                 # Background mode
```

## Key Patterns

### Node Signature
All nodes follow: `def node_name(state: GraphState) -> GraphState`

Each node is wrapped by `create_timed_node()` in `workflow.py`, which accumulates elapsed milliseconds into `state["node_timings"]`.

### Structured Output
Use Pydantic models with LLM:
```python
class RouteDecision(BaseModel):
    route: Literal["retrieve", "reject"]
    reasoning: str

result = llm.with_structured_output(RouteDecision).invoke(prompt)
```

### Configuration
Always load from `settings`:
```python
from specagent.config import settings
model = settings.embedding_model
```

### Embedding Prefixes (nomic asymmetric model)
```python
# Ingestion
"search_document: " + chunk_text

# Query time
"search_query: " + query_text
```

### Grader Auto-Grade Thresholds
- Similarity > 0.82 → auto-grade as relevant (no LLM call)
- Similarity < 0.55 → auto-grade as not relevant (no LLM call)
- Mid-range (0.55–0.82) → single batched LLM call for all mid-range chunks

### Hallucination Skip Thresholds
- avg_confidence >= 0.65 (numerical/tabular content) or 0.70 (other) → skip hallucination check
- Content type detected by scanning for number patterns and markdown table structures

### Per-Request Overrides
Pass `max_rewrites_override` and `library_filter` in initial state to override global settings per query. The API exposes both via `QueryRequest.max_rewrites` and `QueryRequest.library`.

## Environment

Required:
```bash
GROQ_API_KEY=...                  # LLM backend (Groq cloud)
```

Optional (observability):
```bash
LANGCHAIN_API_KEY=...             # LangSmith tracing
LANGCHAIN_TRACING_V2=true
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006  # Arize Phoenix (default: docker-compose)
```

See `config.py` for the full list of settings and their defaults. All settings are configurable via environment variables or a `.env` file.

## Constraints

- **Memory**: 4GB RAM limit (k8s pod constraint)
- **API**: Groq free tier (rate limited — 30K TPM / 500K TPD)
- **Index**: Must fit in memory (~1.5GB for 500K vectors)

## Testing

- Tests use fixtures from `tests/conftest.py`
- Mock external APIs with `pytest-httpx`
- Mark tests: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`

## Do Not

Extends global Do Not. Project-specific hard rules:

| Rule | Reason |
|---|---|
| **No LLM calls outside LangGraph nodes** | Business logic in ad-hoc LLM calls bypasses tracing, state management, and retry logic |
| **No bypassing grader thresholds** (0.55/0.82) | These were tuned for recall/precision balance on the TSpec-LLM benchmark — changing them requires re-evaluation |
| **No real network calls in tests** | Use `pytest-httpx` to mock Groq/embedding endpoints — tests must pass offline |
| **No writing to the user's real LanceDB in tests** | Always use `tmp_path` fixtures from `conftest.py` |
| **No changing `EMBEDDING_MODEL`** without a full re-index | Embeddings from different models are incompatible; search silently returns garbage |
| **No `print()` in `src/`** | Use `logging.getLogger(__name__)` — print pollutes structured logs and breaks stdio-based tooling |
| **No skipping `specagent download-model`** on a fresh env | fastembed downloads the ONNX model on first use, which blocks the first query for 30+ seconds |

---

## References

- PRD: See `docs/prd-3gpp-agentic-rag.md` for full requirements
- Evaluation: See `docs/prd-evaluation-addendum.md` for testing strategy
- Development Guide: See `docs/claude-code-development-guide.md` for workflow
- Overview (exec/end-user): See `docs/overview.md`
- Developer Guide: See `docs/developer-guide.md`
