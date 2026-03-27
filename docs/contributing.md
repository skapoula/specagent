# Contributing to SpecAgent

## Prerequisites

- Python 3.11+
- `uv` package manager (or `pip` with a virtual environment)
- A Groq API key for integration and e2e tests
- Docker (optional, for running the full stack locally)
- `git` with pre-commit hooks support

## Local Setup

1. Clone the repository and enter the directory:
   ```bash
   git clone <repository-url>
   cd specagent
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
3. Install all dependencies including dev and eval extras:
   ```bash
   pip install -e ".[dev,eval]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```
5. Copy the environment template and set your API key:
   ```bash
   cp .env.example .env
   # Edit .env: set GROQ_API_KEY
   ```
6. Download the embedding model:
   ```bash
   specagent download-model
   ```
7. Run the test suite to verify the setup:
   ```bash
   pytest -m unit
   ```

## Running Tests

```bash
# All tests (requires GROQ_API_KEY for e2e)
pytest

# Unit tests only (no external dependencies)
pytest -m unit

# Integration tests (writes to tmp LanceDB, no LLM calls)
pytest -m integration

# E2E tests (mocked LLM + store, no real API calls)
pytest -m e2e

# Run a single file
pytest tests/unit/test_router.py -v

# With coverage report
pytest --cov=src/specagent --cov-report=html
```

Coverage minimum: **80%**. Run `pytest --cov-fail-under=80` to enforce it locally.

## Project Structure

```
specagent/
├── src/specagent/
│   ├── api/            # FastAPI app, routes, request/response models
│   ├── graph/          # LangGraph workflow definition and state schema
│   ├── nodes/          # One file per agent node (router, retriever, grader, ...)
│   ├── retrieval/      # LanceDB store, embedder, ingestor, chunker, converter
│   ├── llm/            # LLM factory (Groq + custom endpoint)
│   ├── observability/  # Query journal, telemetry models, report formatting
│   ├── tracing/        # Phoenix, LangSmith, and RAG span integrations
│   ├── evaluation/     # Benchmark runner and RAGAS metrics
│   ├── cli.py          # Typer CLI entry point
│   └── config.py       # Settings singleton (Pydantic, env-var backed)
├── tests/
│   ├── unit/           # Fast tests, no I/O
│   ├── integration/    # Multi-component tests with real LanceDB in tmp_path
│   └── e2e/            # Full pipeline with mocked LLM and store
├── docs/               # User and developer documentation
├── k8s/                # Kubernetes manifests
├── scripts/            # Data download and utility scripts
├── pyproject.toml      # Dependencies, pytest config, ruff config
└── docker-compose.yml  # Local dev stack (API + Phoenix)
```

## Making Changes

- Always work on a feature branch: `git checkout -b feat/your-feature`
- Follow the existing patterns in the file you are editing.
- Add or update tests for any changed behaviour.
- Run linting and formatting before committing:
  ```bash
  ruff check src/ tests/
  ruff format src/ tests/
  ```
- Run type checking:
  ```bash
  mypy src/specagent
  ```

## Submitting a Pull Request

1. Push your branch: `git push -u origin feat/your-feature`
2. Open a pull request targeting `master` (or the release branch for version work).
3. Ensure all CI checks pass before requesting review.
4. Request review from a maintainer — do not merge without approval.

## Code Standards

Extracted from `pyproject.toml` and `.ruff.toml`:

- Line length: 88 characters (Ruff default).
- Ruff rules: `E`, `F`, `I`, `N`, `UP`, `B`, `SIM` — enforced on all commits via pre-commit.
- Type hints required on all public function signatures.
- Use `X | Y` union syntax, not `Union[X, Y]`.
- Use `pathlib.Path` for all filesystem operations.
- Use `logging` (never `print()`) in `src/`.
- Pydantic models for all data crossing module boundaries.
- Maximum function length: 40 lines. Maximum file length: 300 lines.
- Never swallow exceptions silently — use `raise X from e`.
- No hardcoded secrets. All credentials via environment variables.
