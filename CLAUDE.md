# SpecAgent

Agentic RAG system for 3GPP telecommunications specifications. Helps telecom engineers query Release 18 specs using natural language.

## Coding Rules

See `.claude/rules/` for domain-specific guidelines:
- `api.md` - FastAPI endpoint patterns
- `langgraph.md` - LangGraph node implementation
- `testing.md` - Test-first development standards

Claude reads relevant rules automatically when working in those areas.

## Project Goal

Build a question-answering system that:
- Achieves **85%+ accuracy** on TSpec-LLM benchmark (baseline naive RAG: 71-75%)
- Responds in **<3 seconds** (P95)
- Provides **traceable citations** to source specifications

## Tech Stack

- **Framework**: LangGraph for agentic orchestration
- **Vector Store**: LanceDB (embedded, persistent, hybrid BM25+vector)
- **Embeddings**: `nomic-ai/nomic-embed-text-v1.5` via fastembed (ONNX, 768d, local)
- **LLM**: `Qwen/Qwen2.5-3B-Instruct` via HuggingFace Inference API (or local GGUF)
- **API**: FastAPI
- **Observability**: Arize Phoenix with OpenTelemetry
- **Evaluation**: RAGAS metrics

## Project Structure

```
src/specagent/
├── config.py         # Pydantic Settings (environment config)
├── cli.py            # Typer CLI
├── nodes/            # LangGraph nodes (router, grader, rewriter, generator, hallucination)
├── graph/            # State definition and workflow assembly
├── retrieval/        # Chunking, embeddings, FAISS indexer
├── api/              # FastAPI endpoints
├── evaluation/       # RAGAS metrics, benchmark runner
└── tracing/          # Phoenix integration
```

## Commands

```bash
# Development
pip install -e ".[dev,eval]"   # Install with dev dependencies
pytest                          # Run all tests
pytest -m unit                  # Run unit tests only
ruff check src/ tests/          # Lint
ruff format src/ tests/         # Format
mypy src/specagent              # Type check

# Application
specagent serve                 # Start FastAPI server (port 8000)
specagent query "question"      # Run single query
specagent index                 # Build LanceDB index from data/docs/
specagent index --force         # Rebuild index from scratch
specagent index --docs-dir PATH # Ingest from custom directory
specagent benchmark             # Run evaluation

# Docker
docker-compose up               # Start API + Phoenix
docker-compose up -d            # Background mode
```

## Implementation Status

Implemented (ready to use):
- `config.py` - Pydantic Settings
- `graph/state.py` - GraphState TypedDict
- `graph/workflow.py` - LangGraph skeleton
- `api/main.py` - FastAPI application
- `cli.py` - Typer CLI
- `tracing/phoenix.py` - Phoenix integration
- `tests/conftest.py` - Pytest fixtures

Needs implementation (placeholders exist):
- `retrieval/chunker.py` - Document chunking
- `retrieval/embeddings.py` - HuggingFace embeddings client
- `retrieval/indexer.py` - FAISS index management
- `nodes/*.py` - All LangGraph nodes

## Key Patterns

### Node Signature
All nodes follow: `def node_name(state: GraphState) -> GraphState`

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

## Constraints

- **Memory**: 4GB RAM limit (k8s pod constraint)
- **API**: HuggingFace free tier (rate limited)
- **Index**: Must fit in memory (~1.5GB for 500K vectors)

## Testing

- Tests use fixtures from `tests/conftest.py`
- Mock external APIs with `pytest-httpx`
- Mark tests: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`

## References

- PRD: See `docs/prd-3gpp-agentic-rag.md` for full requirements
- Evaluation: See `docs/prd-evaluation-addendum.md` for testing strategy
- Development Guide: See `docs/claude-code-development-guide.md` for workflow
