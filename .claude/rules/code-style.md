---
# No paths: field — applies to all files in mcpvectordb
---

# Code Style — mcpvectordb

## Language & Runtime

- Python 3.11+ only. Use 3.11+ features freely: `tomllib`, `match`/`case`,
  `ExceptionGroup`, `Self` type, `LiteralString`.
- Do not use deprecated stdlib modules (`distutils`, `imp`, `optparse`).

## Type Hints

- Type hints are required on **all** public function signatures — both parameters
  and return type. No bare `def foo(x):` on public APIs.
- Use `X | Y` union syntax (PEP 604), not `Union[X, Y]`.
- Use `X | None` instead of `Optional[X]`.
- Use built-in generics: `list[str]`, `dict[str, int]`, not `List`, `Dict` from `typing`.
- Use `typing.TYPE_CHECKING` guard for imports only needed for annotations.
- Private/internal helpers may omit hints when the types are obvious from context,
  but public module-level functions never may.
- IMPORTANT: Do not add or remove type annotations without an explicit instruction.

## Formatting & Linting (Ruff)

- Ruff handles both formatting and linting. Do not use Black or Flake8 separately.
- Line length: 88 (Black-compatible).
- Run: `uv run ruff check . && uv run ruff format .`
- Ruff rules enabled (set in `pyproject.toml`): `E`, `F`, `I`, `N`, `UP`, `B`, `SIM`.
  - `I` — isort-compatible import ordering
  - `UP` — pyupgrade: modernise syntax automatically
  - `B` — flake8-bugbear: catch common mistakes
  - `SIM` — flake8-simplify: remove unnecessary complexity
- Do not add `# noqa` suppressions without a comment explaining why.

## Imports

- Standard library first, then third-party, then local — separated by blank lines.
  Ruff's `I` rules enforce this automatically.
- Prefer explicit imports: `from pathlib import Path`, not `import pathlib`.
- Never use wildcard imports (`from module import *`).
- Avoid circular imports. If you feel the need, restructure instead.

## Data Structures

- Use **Pydantic models** for all data that crosses a module boundary
  (function return values, MCP tool inputs/outputs, config, LanceDB records).
- Use `dataclasses` only for simple internal structs with no validation.
- Never pass raw `dict` objects across module boundaries — define a model.
- Use `pydantic.Field` with a `description=` for every field on MCP-facing models;
  Claude Desktop uses these descriptions to understand the tool schema.

## Filesystem

- Use `pathlib.Path` for every filesystem operation — never `os.path`, `os.getcwd()`,
  or raw string concatenation for paths.
- Resolve paths early: call `.resolve()` when accepting a path from user input or config.
- Use `Path.open()` rather than the built-in `open()` for consistency.

## Logging

- Use the stdlib `logging` module. Never use `print()` anywhere in `src/`.
- Get a logger per module: `logger = logging.getLogger(__name__)`.
- Log levels:
  - `DEBUG` — internal state, per-chunk progress, timing
  - `INFO` — document ingested, search executed, server started
  - `WARNING` — unsupported format attempted, slow operation, retry
  - `ERROR` — unrecoverable failure, exception caught at boundary
- Never log sensitive data: file contents, embeddings, user metadata values.
- CRITICAL: In `stdio` transport mode, ALL output must go to `stderr` or a log file.
  A single byte on `stdout` outside the MCP protocol will silently break Claude Desktop.

## Error Handling

- Define custom exception classes in `exceptions.py` for domain errors:
  `UnsupportedFormatError`, `IngestionError`, `StoreError`, `EmbeddingError`.
- Catch exceptions at module boundaries, log with context, re-raise as domain errors.
- Never swallow exceptions silently (`except Exception: pass` is forbidden).
- Use `raise X from e` to preserve the original traceback when re-raising.
- MCP tool handlers must catch all exceptions and return structured error responses
  rather than letting exceptions propagate to the MCP framework unhandled.

## Docstrings

- All public functions, classes, and modules require a docstring.
- Minimum: one-line summary ending with a period.
- For functions with non-obvious parameters or return values, use Google style:
  ```python
  def search(query: str, top_k: int = 5) -> list[ChunkRecord]:
      """Search the index and return the top-k most relevant chunks.

      Args:
          query: Natural language search query.
          top_k: Maximum number of results to return.

      Returns:
          List of ChunkRecord objects sorted by relevance descending.

      Raises:
          StoreError: If the LanceDB table cannot be read.
      """
  ```
- Private functions (`_name`) may use a one-liner only.

## Async

- `server.py` and `transport.py` are async (MCP SDK is async).
- All I/O-bound operations (LanceDB reads/writes, HTTP fetches, file reads) must be
  `async` or run in a thread pool via `asyncio.to_thread()` — never block the event loop.
- Use `asyncio.to_thread()` for blocking calls in libraries that don't support async
  (e.g. sentence-transformers, markitdown, synchronous LanceDB operations).
- Do not mix sync and async code in the same function.

## General

- Prefer early returns to reduce nesting. Maximum nesting depth: 3 levels.
- Maximum function length: 40 lines. Extract sub-functions if longer.
- Maximum file length: 300 lines. Split into modules if longer.
- Named constants over magic numbers or magic strings.
- No dead code, commented-out blocks, or unused imports before committing.
- Match the existing style of any file you are editing. Do not reformat unrelated lines.
