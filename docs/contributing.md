# Contributing to SpecAgent

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker + Docker Compose (for the full stack)
- `inkscape` (optional — required only for `.docx` OCR tests)

## Local Setup

1. Clone the repo and enter the directory.
2. Install all dependencies:
   ```bash
   uv sync
   ```
3. Download the embedding model (required before first run):
   ```bash
   uv run specagent download-model
   ```
4. Copy the environment template and add your keys:
   ```bash
   cp .env.example .env   # then set GROQ_API_KEY
   ```
5. Verify the setup:
   ```bash
   uv run specagent --version
   uv run pytest -m unit
   ```

## Running Tests

```bash
uv run pytest                          # full suite (excludes slow)
uv run pytest -m unit                  # unit tests only
uv run pytest -m integration           # real LanceDB (tmp_path)
uv run pytest -m real_api -v           # live Groq + Inkscape required
uv run pytest --cov=src/specagent      # with coverage report
```

Minimum coverage target: **70%** across `src/specagent/`.

## Project Structure

```
src/specagent/
├── api/          FastAPI app and Pydantic request/response models
├── cli.py        Typer CLI (serve, query, index, benchmark, download-model)
├── config.py     Pydantic Settings singleton
├── graph/        LangGraph state schema and workflow
├── kuzu/         Kuzu graph DB — call-flow DAG storage and retrieval
├── llm/          LLM factory and Groq rate limiter
├── nodes/        LangGraph nodes (one file per node)
├── observability/ Query journal and monitoring report
├── retrieval/    Ingestor, chunker, embedder, converter, LanceDB store
├── tracing/      Phoenix OTel and LangSmith integrations
└── evaluation/   TSpec-LLM benchmark runner and RAGAS metrics
tests/
├── unit/         Pure function tests — no I/O
├── integration/  Real LanceDB and file-system tests (tmp_path fixtures)
└── conftest.py   Shared fixtures including real-API helpers
```

## Making Changes

Follow the TDD cycle — write a failing test first, then implement:

1. Create `tests/unit/test_<module>.py` with at least one failing test.
2. Run it and confirm it fails for the right reason.
3. Implement the minimum code to make it pass.
4. Run the full suite: `uv run pytest`.
5. Run the linter: `uv run ruff check . && uv run ruff format .`
6. Run the type checker: `uv run mypy src/specagent`

Never delete or weaken tests to make the suite green — fix the code.

## Submitting a Pull Request

1. Branch off `main`: `git checkout -b feat/<short-description>`
2. Commit with Conventional Commits format: `feat(scope): subject` — max 72 chars, imperative mood.
3. Open a PR against `main`. CI must be green before review.
4. A maintainer will review and merge.

## Code Standards

Extracted from `pyproject.toml` and `.claude/rules/`:

- **Formatter/linter:** Ruff (`ruff check . && ruff format .`), line length 88.
- **Type hints:** required on all public function signatures; use `X | Y`, `X | None`, built-in generics (`list[str]`, not `List[str]`).
- **Logging:** `logging.getLogger(__name__)` — never `print()` in `src/`.
- **Paths:** `pathlib.Path` everywhere — never `os.path`.
- **Async:** all I/O-bound operations must be `async` or run via `asyncio.to_thread()`.
- **Max function length:** 40 lines. **Max file length:** 300 lines.
- **No hardcoded secrets** — use environment variables or `.env`.
