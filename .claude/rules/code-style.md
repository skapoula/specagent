# Code Style — specagent
# Extends /workspace/.claude/rules/master-code-style.md (loaded automatically).
# Only project-specific additions and overrides are listed here.

## Data Structures

- Use **Pydantic models** for all data that crosses a module boundary
  (API request/response schemas, LanceDB records, config).
- Use `dataclasses` only for simple internal structs with no validation
  (e.g. `RetrievedChunk`, `GradedChunk`, `Citation` — see `graph/state.py`).
- Never pass raw `dict` objects across module boundaries — define a model.

## Async

- FastAPI endpoints and the ingest pipeline are async.
- All I/O-bound operations (LanceDB reads/writes, HTTP fetches, file reads) must be
  `async` or run in a thread pool via `asyncio.to_thread()` — never block the event loop.
- Use `asyncio.to_thread()` for blocking calls (fastembed, markitdown, synchronous LanceDB).
- Do not mix sync and async code in the same function.

## Error Handling (additions)

- Domain exception classes live in `retrieval/exceptions.py`:
  `UnsupportedFormatError`, `IngestionError`, `StoreError`, `EmbeddingError`.
- FastAPI handlers must catch domain exceptions and raise `HTTPException` with
  structured detail — never let domain exceptions propagate to the framework unhandled.
- LangGraph nodes must handle errors gracefully and store them in state rather than
  raising exceptions that would abort the graph.
