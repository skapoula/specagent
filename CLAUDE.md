# SpecAgent

> **Scope:** PROJECT-LEVEL — inherits org-wide policy from `/workspace/CLAUDE.md`.
> Personal overrides go in `CLAUDE.local.md` (auto-gitignored).

Agentic RAG system for 3GPP telecommunications specifications. Helps telecom engineers query Release 18 specs using natural language.

## Project Goal

- **85%+ accuracy** on TSpec-LLM benchmark (baseline naive RAG: 71-75%)
- **<3 seconds** P95 response
- **Traceable citations** to source specifications

## Tech Stack

- **Framework**: LangGraph
- **Vector Store**: LanceDB (embedded, hybrid BM25+vector)
- **Embeddings**: `nomic-ai/nomic-embed-text-v1.5` via fastembed (ONNX, 768d, local)
- **LLM**: `meta-llama/llama-4-scout-17b-16e-instruct` via Groq
- **API**: FastAPI
- **Observability**: Arize Phoenix + OpenTelemetry
- **Evaluation**: RAGAS metrics

## Project Structure

```
src/specagent/
├── config.py         # Pydantic Settings (@lru_cache singleton)
├── cli.py            # Typer CLI (serve, query, index, benchmark, download-model, version)
├── nodes/            # LangGraph nodes (router, retriever, grader, rewriter, generator, hallucination)
├── graph/
│   ├── state.py      # GraphState TypedDict, RetrievedChunk, GradedChunk, Citation
│   └── workflow.py   # build_graph(), run_query(), conditional edges
├── retrieval/
│   ├── store.py      # LanceDB Store (ChunkRecord, hybrid search, CRUD)
│   ├── ingestor.py   # Async 7-step ingest pipeline
│   ├── chunker.py    # Token-aware recursive chunker with section header propagation
│   ├── embedder.py   # embed_documents() / embed_query() with nomic prefix handling
│   ├── converter.py  # MarkItDown file-to-Markdown (22 extensions)
│   ├── resources.py  # @lru_cache singletons: get_store(), get_embedder()
│   └── exceptions.py # UnsupportedFormatError, IngestionError, StoreError, EmbeddingError
├── kuzu/             # Kuzu embedded graph store for call-flow DAGs
├── llm/
│   ├── factory.py    # create_llm() → _GroqAdapter | CustomEndpointLLM
│   └── custom_endpoint.py
├── api/
│   ├── main.py       # FastAPI app, /health + /query, lifespan startup
│   └── models.py     # QueryRequest, QueryResponse, HealthResponse
├── observability/    # LLMCallRecord, QueryJournal (JSONL), build_query_report()
├── evaluation/       # TSpec-LLM runner + LLM judge, RAGAS metrics
└── tracing/          # Arize Phoenix, LangSmith, OTel spans
```

## Commands

```bash
# Setup
uv sync                              # install all dependencies
specagent download-model             # REQUIRED on first setup (caches ONNX model)

# Development
pytest                               # full suite (excludes slow)
pytest -m unit                       # unit tests only
pytest -m integration                # real LanceDB (tmp)
pytest -m e2e                        # full pipeline
pytest --cov=src/specagent           # with coverage
ruff check src/ tests/ && ruff format src/ tests/
mypy src/specagent

# Application
specagent serve                      # FastAPI server (port 8000)
specagent query "question"           # single query
specagent query "question" --verbose # with timing + metadata
specagent index                      # build LanceDB index from data/docs/
specagent index --docs-dir PATH --library NAME --force --max-concurrency 4
specagent benchmark                  # run TSpec-LLM evaluation

# Docker
docker-compose up                    # API (port 8000) + Phoenix (port 6006)
docker-compose --profile ui up       # + Gradio UI (port 7860)
```

## Key Patterns

- **Node signature:** `def node_name(state: GraphState) -> GraphState` — nodes must never raise; write errors to `state["error"]`.
- **Structured output:** `llm.with_structured_output(MyPydanticModel).invoke(prompt)`
- **Config:** always `from specagent.config import settings`
- **Embedding prefixes:** `"search_document: " + chunk_text` at ingest; `"search_query: " + query_text` at query time
- **Grader thresholds:** similarity > 0.82 → auto-yes; < 0.55 → auto-no; mid-range → batched LLM call
- **Hallucination skip:** avg_confidence ≥ 0.65 (numerical/tabular) or ≥ 0.70 (other) → skip check
- **Per-request overrides:** `max_rewrites_override` and `library_filter` in initial state

See [developer-guide.md](docs/developer-guide.md) for full module reference and design decisions.

## Environment

```bash
GROQ_API_KEY=...                              # required
LANGCHAIN_API_KEY=...                         # optional: LangSmith
LANGCHAIN_TRACING_V2=true                     # optional
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006  # optional (default: docker-compose)
```

Full settings list in `config.py`. All configurable via env vars or `.env`.

## Do Not

| Rule                                                    | Reason                                                                  |
| ------------------------------------------------------- | ----------------------------------------------------------------------- |
| **No LLM calls outside LangGraph nodes**                | Bypasses tracing, state management, retry logic                         |
| **No bypassing grader thresholds** (0.55/0.82)          | Tuned for TSpec-LLM recall/precision — changing requires re-evaluation  |
| **No real network calls in tests**                      | Use `pytest-httpx` — tests must pass offline                            |
| **No writing to real LanceDB in tests**                 | Always use `tmp_path` fixtures from `conftest.py`                       |
| **No changing `EMBEDDING_MODEL`** without full re-index | Different models produce incompatible vectors; search silently degrades |
| **No `print()` in `src/`**                              | Use `logging.getLogger(__name__)`                                       |
| **No skipping `specagent download-model`** on fresh env | fastembed blocks first query for 30+ s if model not cached              |
