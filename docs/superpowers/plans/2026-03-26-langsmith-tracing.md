# LangSmith Tracing & Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LangSmith tracing to the specagent pipeline so every query — across both LLM backends — is visible in the LangSmith UI with node-level spans, inputs/outputs, and latency.

**Architecture:** Add three Settings fields (`enable_langsmith`, `langchain_api_key`, `langchain_project`) consumed by a new `tracing/langsmith.py` module that follows the same pattern as the existing `tracing/phoenix.py`. Apply `@traceable` from the `langsmith` package to `CustomEndpointLLM.invoke_with_timing()` to cover the custom backend; the Groq backend auto-traces through LangChain's callback system once the env vars are active. Wire `setup_langsmith_tracing()` into the FastAPI lifespan and the relevant CLI commands.

**Tech Stack:** `langsmith>=0.1.0` (moving to core deps), `pydantic-settings` (already present), `langchain-openai` (already present for Groq path)

---

## File Map

| Action | Path |
|---|---|
| Modify | `pyproject.toml` |
| Modify | `src/specagent/config.py` |
| Modify | `.env.example` |
| **Create** | `src/specagent/tracing/langsmith.py` |
| Modify | `src/specagent/tracing/__init__.py` |
| Modify | `src/specagent/llm/custom_endpoint.py` |
| Modify | `src/specagent/api/main.py` |
| Modify | `src/specagent/cli.py` |
| **Create** | `tests/unit/test_langsmith_tracing.py` |

---

## Task 1: Add `langsmith` to Core Dependencies

**Files:**
- Modify: `pyproject.toml` (line 33–64 `[project.dependencies]`, and line 86–97 `[eval]`)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_langsmith_tracing.py  (create new file)
"""Unit tests for LangSmith tracing integration."""
import pytest


@pytest.mark.unit
def test_langsmith_importable():
    """langsmith must be importable as a core dependency (not optional)."""
    import langsmith  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py::test_langsmith_importable -v
```

Expected: passes if `langsmith` is already installed; this test documents the requirement.

- [ ] **Step 3: Move `langsmith` to core deps in `pyproject.toml`**

In `pyproject.toml`, move `"langsmith>=0.1.0"` from the `[project.optional-dependencies] eval` block into the `[project.dependencies]` list (keep it in `[eval]` too so existing `pip install specagent[eval]` still works — just add it to both):

```toml
dependencies = [
    # ... existing entries ...
    "requests>=2.32.0",
    # Tracing
    "langsmith>=0.1.0",
]
```

Also ensure it stays in `[eval]` (no change needed — adding to core is additive).

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /workspace/specagent && pip install -e ".[dev,eval]" -q && pytest tests/unit/test_langsmith_tracing.py::test_langsmith_importable -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /workspace/specagent && git add pyproject.toml tests/unit/test_langsmith_tracing.py
git commit -m "feat(deps): promote langsmith to core dependencies"
```

---

## Task 2: Add LangSmith Settings Fields

**Files:**
- Modify: `src/specagent/config.py` (Observability section, ~line 270)
- Modify: `.env.example` (Observability section, ~line 100)

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_langsmith_tracing.py`)

```python
import os
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_settings_enable_langsmith_defaults_true():
    """enable_langsmith defaults to True."""
    from specagent.config import get_settings
    get_settings.cache_clear()
    with patch.dict(os.environ, {}, clear=False):
        s = get_settings()
    assert s.enable_langsmith is True


@pytest.mark.unit
def test_settings_langchain_api_key_defaults_empty():
    """langchain_api_key defaults to empty string."""
    from specagent.config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    assert s.langchain_api_key == ""


@pytest.mark.unit
def test_settings_langchain_project_defaults():
    """langchain_project defaults to '3gpp-specagent'."""
    from specagent.config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    assert s.langchain_project == "3gpp-specagent"


@pytest.mark.unit
def test_settings_reads_langchain_api_key_from_env():
    """langchain_api_key is populated from LANGCHAIN_API_KEY env var."""
    from specagent.config import get_settings
    get_settings.cache_clear()
    with patch.dict(os.environ, {"LANGCHAIN_API_KEY": "ls-test-key"}):
        s = get_settings()
    assert s.langchain_api_key == "ls-test-key"
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_reads_langchain_project_from_env():
    """langchain_project is populated from LANGCHAIN_PROJECT env var."""
    from specagent.config import get_settings
    get_settings.cache_clear()
    with patch.dict(os.environ, {"LANGCHAIN_PROJECT": "my-project"}):
        s = get_settings()
    assert s.langchain_project == "my-project"
    get_settings.cache_clear()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py -k "settings" -v
```

Expected: FAIL — `Settings` has no `enable_langsmith`, `langchain_api_key`, or `langchain_project` fields.

- [ ] **Step 3: Add the three fields to `config.py`**

In `src/specagent/config.py`, inside the `Settings` class, append to the Observability section (after `enable_tracing` at line ~279):

```python
    # LangSmith tracing (https://smith.langchain.com)
    # Set LANGCHAIN_API_KEY to activate. LANGCHAIN_TRACING_V2 and
    # LANGCHAIN_PROJECT are set by setup_langsmith_tracing() from these values.
    enable_langsmith: bool = Field(
        default=True,
        description="Enable LangSmith tracing. Requires LANGCHAIN_API_KEY to be set.",
    )
    langchain_api_key: str = Field(
        default="",
        description="LangSmith API key (env var: LANGCHAIN_API_KEY).",
    )
    langchain_project: str = Field(
        default="3gpp-specagent",
        description="LangSmith project name (env var: LANGCHAIN_PROJECT).",
    )
```

- [ ] **Step 4: Update `.env.example`**

In `.env.example`, append to the Observability section (after `ENABLE_TRACING=true`):

```bash
# LangSmith tracing (https://smith.langchain.com)
# Set LANGCHAIN_API_KEY to your LangSmith API key to activate tracing.
# Get a free key at: https://smith.langchain.com/settings
ENABLE_LANGSMITH=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=3gpp-specagent
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py -k "settings" -v
```

Expected: all 5 PASS

- [ ] **Step 6: Commit**

```bash
cd /workspace/specagent && git add src/specagent/config.py .env.example tests/unit/test_langsmith_tracing.py
git commit -m "feat(config): add LangSmith settings fields (enable_langsmith, langchain_api_key, langchain_project)"
```

---

## Task 3: Create `tracing/langsmith.py`

**Files:**
- Create: `src/specagent/tracing/langsmith.py`
- Test: `tests/unit/test_langsmith_tracing.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_langsmith_tracing.py`)

```python
import sys
import warnings
from importlib import reload
from unittest.mock import MagicMock, patch


@pytest.mark.unit
def test_setup_langsmith_disabled():
    """setup_langsmith_tracing does nothing when enable_langsmith is False."""
    import specagent.tracing.langsmith as ls_module
    reload(ls_module)

    with patch("specagent.tracing.langsmith.settings") as mock_cfg:
        mock_cfg.enable_langsmith = False
        # Should return without touching os.environ
        original_env = os.environ.copy()
        ls_module.setup_langsmith_tracing()
        assert "LANGCHAIN_TRACING_V2" not in os.environ or \
               os.environ.get("LANGCHAIN_TRACING_V2") == original_env.get("LANGCHAIN_TRACING_V2")


@pytest.mark.unit
def test_setup_langsmith_no_api_key_warns():
    """setup_langsmith_tracing warns when LANGCHAIN_API_KEY is not set."""
    import specagent.tracing.langsmith as ls_module
    reload(ls_module)

    with patch("specagent.tracing.langsmith.settings") as mock_cfg:
        mock_cfg.enable_langsmith = True
        mock_cfg.langchain_api_key = ""
        mock_cfg.langchain_project = "3gpp-specagent"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ls_module.setup_langsmith_tracing()

        matching = [x for x in w if "LANGCHAIN_API_KEY" in str(x.message)]
        assert len(matching) == 1


@pytest.mark.unit
def test_setup_langsmith_sets_env_vars():
    """setup_langsmith_tracing sets LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT."""
    import specagent.tracing.langsmith as ls_module
    reload(ls_module)

    env_before = {
        k: v for k, v in os.environ.items()
        if k in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT")
    }
    for k in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
        os.environ.pop(k, None)

    try:
        with patch("specagent.tracing.langsmith.settings") as mock_cfg:
            mock_cfg.enable_langsmith = True
            mock_cfg.langchain_api_key = "ls-test-key"
            mock_cfg.langchain_project = "my-test-project"

            ls_module.setup_langsmith_tracing()

            assert os.environ["LANGCHAIN_API_KEY"] == "ls-test-key"
            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            assert os.environ["LANGCHAIN_PROJECT"] == "my-test-project"
    finally:
        # Restore original env
        for k in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
            os.environ.pop(k, None)
        os.environ.update(env_before)


@pytest.mark.unit
def test_setup_langsmith_does_not_overwrite_existing_project():
    """setup_langsmith_tracing respects a pre-existing LANGCHAIN_PROJECT."""
    import specagent.tracing.langsmith as ls_module
    reload(ls_module)

    env_before = {
        k: v for k, v in os.environ.items()
        if k in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT")
    }
    os.environ["LANGCHAIN_PROJECT"] = "already-set"
    for k in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY"):
        os.environ.pop(k, None)

    try:
        with patch("specagent.tracing.langsmith.settings") as mock_cfg:
            mock_cfg.enable_langsmith = True
            mock_cfg.langchain_api_key = "ls-test-key"
            mock_cfg.langchain_project = "3gpp-specagent"

            ls_module.setup_langsmith_tracing()

            # Pre-existing project must not be overwritten
            assert os.environ["LANGCHAIN_PROJECT"] == "already-set"
    finally:
        for k in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
            os.environ.pop(k, None)
        os.environ.update(env_before)


@pytest.mark.unit
def test_setup_langsmith_missing_package_warns():
    """setup_langsmith_tracing warns gracefully when langsmith is not installed."""
    import specagent.tracing.langsmith as ls_module
    reload(ls_module)

    with patch("specagent.tracing.langsmith.settings") as mock_cfg:
        mock_cfg.enable_langsmith = True
        mock_cfg.langchain_api_key = "ls-test-key"
        mock_cfg.langchain_project = "3gpp-specagent"

        with patch.dict(sys.modules, {"langsmith": None}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ls_module.setup_langsmith_tracing()

            matching = [x for x in w if "langsmith" in str(x.message).lower()]
            assert len(matching) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py -k "setup_langsmith" -v
```

Expected: FAIL — `specagent.tracing.langsmith` module does not exist.

- [ ] **Step 3: Create `src/specagent/tracing/langsmith.py`**

```python
"""
LangSmith tracing integration.

Configures LangSmith tracing for the specagent pipeline. When enabled and
LANGCHAIN_API_KEY is set, all LangChain/LangGraph calls (Groq backend) are
automatically traced via LangChain's callback system. The custom_endpoint
backend is covered by the @traceable decorator on CustomEndpointLLM.

Usage:
    from specagent.tracing import setup_langsmith_tracing
    setup_langsmith_tracing()  # Call once at application startup
"""

import logging
import os
import warnings

from specagent.config import settings

logger = logging.getLogger(__name__)


def setup_langsmith_tracing() -> None:
    """
    Configure LangSmith tracing.

    Checks settings.enable_langsmith and settings.langchain_api_key.
    If both are present, sets the standard LangSmith env vars that the SDK
    reads automatically (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT). Safe to call multiple times — idempotent via
    os.environ.setdefault.

    Warnings are emitted (not exceptions) so a missing key never crashes
    the application.
    """
    if not settings.enable_langsmith:
        logger.debug("LangSmith tracing disabled (ENABLE_LANGSMITH=false)")
        return

    try:
        import langsmith  # noqa: F401
    except ImportError:
        warnings.warn(
            "LangSmith not installed. Install with: pip install langsmith",
            stacklevel=2,
        )
        return

    if not settings.langchain_api_key:
        warnings.warn(
            "LANGCHAIN_API_KEY is not set. LangSmith tracing will not be active. "
            "Set LANGCHAIN_API_KEY in your .env file to enable tracing.",
            stacklevel=2,
        )
        return

    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)

    logger.info(
        "LangSmith tracing enabled (project=%s)",
        os.environ["LANGCHAIN_PROJECT"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py -k "setup_langsmith" -v
```

Expected: all 5 PASS

- [ ] **Step 5: Commit**

```bash
cd /workspace/specagent && git add src/specagent/tracing/langsmith.py tests/unit/test_langsmith_tracing.py
git commit -m "feat(tracing): add LangSmith tracing module with setup_langsmith_tracing()"
```

---

## Task 4: Export `setup_langsmith_tracing` from `tracing/__init__.py`

**Files:**
- Modify: `src/specagent/tracing/__init__.py`

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_langsmith_tracing.py`)

```python
@pytest.mark.unit
def test_setup_langsmith_tracing_exported_from_tracing_package():
    """setup_langsmith_tracing is importable from specagent.tracing."""
    from specagent.tracing import setup_langsmith_tracing
    assert callable(setup_langsmith_tracing)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py::test_setup_langsmith_tracing_exported_from_tracing_package -v
```

Expected: FAIL — `ImportError: cannot import name 'setup_langsmith_tracing'`

- [ ] **Step 3: Update `src/specagent/tracing/__init__.py`**

Replace the file content:

```python
"""
Observability and tracing integrations.

Provides OpenTelemetry-based tracing via Arize Phoenix and
LangSmith tracing for the LangGraph pipeline.
"""

from specagent.tracing.langsmith import setup_langsmith_tracing
from specagent.tracing.phoenix import setup_tracing, traced

__all__ = ["setup_tracing", "setup_langsmith_tracing", "traced"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py::test_setup_langsmith_tracing_exported_from_tracing_package -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /workspace/specagent && git add src/specagent/tracing/__init__.py tests/unit/test_langsmith_tracing.py
git commit -m "feat(tracing): export setup_langsmith_tracing from tracing package"
```

---

## Task 5: Apply `@traceable` to `CustomEndpointLLM.invoke_with_timing()`

**Files:**
- Modify: `src/specagent/llm/custom_endpoint.py`
- Test: `tests/unit/test_custom_endpoint.py`

**Why:** The Groq backend auto-traces through `langchain_openai.ChatOpenAI`'s LangChain callbacks. The custom endpoint uses raw `requests.post()` and is invisible to LangSmith without an explicit `@traceable` annotation. The decorator is a no-op when `LANGCHAIN_TRACING_V2` is not `"true"`, so existing tests are unaffected.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_custom_endpoint.py`)

```python
from unittest.mock import patch, MagicMock
import pytest


@pytest.mark.unit
def test_invoke_with_timing_is_traceable():
    """invoke_with_timing carries a langsmith traceable wrapper."""
    from langsmith import traceable
    from specagent.llm.custom_endpoint import CustomEndpointLLM

    # The langsmith @traceable decorator sets __wrapped__ or the function is
    # callable — verify it still returns correct (text, ms) without tracing active.
    llm = CustomEndpointLLM("http://test/v1/chat/completions")
    ok_response = MagicMock()
    ok_response.json.return_value = {"choices": [{"message": {"content": "traced"}}]}
    ok_response.raise_for_status = MagicMock()

    with patch("specagent.llm.custom_endpoint.requests.post", return_value=ok_response):
        text, ms = llm.invoke_with_timing("p")

    assert text == "traced"
    assert isinstance(ms, float)
```

- [ ] **Step 2: Run test to verify it currently passes** (baseline — the test should pass before and after the change)

```bash
cd /workspace/specagent && pytest tests/unit/test_custom_endpoint.py::TestInvoke::test_timing tests/unit/test_custom_endpoint.py::test_invoke_with_timing_is_traceable -v
```

Expected: Both PASS (test_timing already passes; new test also passes since the function works correctly with or without `@traceable`).

- [ ] **Step 3: Apply `@traceable` to `invoke_with_timing` in `src/specagent/llm/custom_endpoint.py`**

Add the import at the top of the file (after the existing imports):

```python
from langsmith import traceable
```

Then apply the decorator to `invoke_with_timing` (line 65):

```python
    @traceable(name="custom_endpoint_llm", run_type="llm")
    def invoke_with_timing(self, prompt: str) -> tuple[str, float]:
```

The decorator goes directly above the method definition, inside the class body. The full method signature becomes:

```python
    @traceable(name="custom_endpoint_llm", run_type="llm")
    def invoke_with_timing(self, prompt: str) -> tuple[str, float]:
        """
        Call the LLM with a prompt and return timing information.
        ...
        """
```

- [ ] **Step 4: Run all custom endpoint tests to verify they pass**

```bash
cd /workspace/specagent && pytest tests/unit/test_custom_endpoint.py -v
```

Expected: ALL PASS — `@traceable` is a no-op when `LANGCHAIN_TRACING_V2` is not set in the test environment.

- [ ] **Step 5: Commit**

```bash
cd /workspace/specagent && git add src/specagent/llm/custom_endpoint.py tests/unit/test_custom_endpoint.py
git commit -m "feat(llm): apply @traceable to CustomEndpointLLM.invoke_with_timing for LangSmith coverage"
```

---

## Task 6: Wire `setup_langsmith_tracing()` into the API Lifespan

**Files:**
- Modify: `src/specagent/api/main.py` (lifespan function, ~line 202)

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_langsmith_tracing.py`)

```python
@pytest.mark.unit
def test_api_lifespan_calls_setup_langsmith_tracing():
    """FastAPI lifespan calls setup_langsmith_tracing() when enable_langsmith is True."""
    from unittest.mock import AsyncMock, patch, MagicMock
    from fastapi.testclient import TestClient

    mock_langsmith_setup = MagicMock()
    mock_phoenix_setup = MagicMock()
    mock_init = MagicMock(return_value={"store": "ok", "embedder": "ok"})

    with (
        patch("specagent.api.main.initialize_resources", mock_init),
        patch("specagent.api.main.settings") as mock_cfg,
        patch("specagent.tracing.phoenix.setup_tracing", mock_phoenix_setup),
        patch("specagent.tracing.langsmith.setup_langsmith_tracing", mock_langsmith_setup),
    ):
        mock_cfg.enable_tracing = False
        mock_cfg.enable_langsmith = True

        from specagent.api.main import create_app
        app = create_app()

        with TestClient(app):
            pass  # triggers lifespan startup

    mock_langsmith_setup.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py::test_api_lifespan_calls_setup_langsmith_tracing -v
```

Expected: FAIL — `setup_langsmith_tracing` is not called from the lifespan yet.

- [ ] **Step 3: Update the lifespan in `src/specagent/api/main.py`**

In the `lifespan()` function (around line 202), after the existing Phoenix tracing block:

```python
    if settings.enable_tracing:
        from specagent.tracing.phoenix import setup_tracing  # noqa: PLC0415

        setup_tracing()

    if settings.enable_langsmith:
        from specagent.tracing.langsmith import setup_langsmith_tracing  # noqa: PLC0415

        setup_langsmith_tracing()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py::test_api_lifespan_calls_setup_langsmith_tracing -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /workspace/specagent && git add src/specagent/api/main.py tests/unit/test_langsmith_tracing.py
git commit -m "feat(api): call setup_langsmith_tracing() in FastAPI lifespan"
```

---

## Task 7: Wire `setup_langsmith_tracing()` into the CLI

**Files:**
- Modify: `src/specagent/cli.py` (`query` and `benchmark` commands)

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_langsmith_tracing.py`)

```python
@pytest.mark.unit
def test_cli_query_calls_setup_langsmith_tracing(mock_settings):
    """CLI `query` command calls setup_langsmith_tracing() before run_query()."""
    from typer.testing import CliRunner
    from unittest.mock import MagicMock, patch

    runner = CliRunner()
    mock_setup = MagicMock()
    mock_run_query = MagicMock(return_value={
        "route_decision": "reject",
        "route_reasoning": "off-topic",
    })

    with (
        patch("specagent.cli.setup_langsmith_tracing", mock_setup),
        patch("specagent.cli.run_query", mock_run_query),
    ):
        from specagent.cli import app
        result = runner.invoke(app, ["query", "What is NR?"])

    assert result.exit_code == 0
    mock_setup.assert_called_once()


@pytest.mark.unit
def test_cli_benchmark_calls_setup_langsmith_tracing(mock_settings):
    """CLI `benchmark` command calls setup_langsmith_tracing() before running."""
    from typer.testing import CliRunner
    from unittest.mock import MagicMock, patch

    runner = CliRunner()
    mock_setup = MagicMock()
    mock_benchmark = MagicMock(return_value=MagicMock(
        total_questions=0, correct=0, accuracy=0.0,
        average_latency_ms=0.0, p95_latency_ms=0.0,
    ))

    with (
        patch("specagent.cli.setup_langsmith_tracing", mock_setup),
        patch("specagent.cli.run_benchmark", mock_benchmark),
    ):
        from specagent.cli import app
        result = runner.invoke(app, ["benchmark", "--questions-file", "/dev/null"])

    # setup was called even if benchmark errors due to empty file
    mock_setup.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py -k "cli" -v
```

Expected: FAIL — `setup_langsmith_tracing` not imported or called in `cli.py`.

- [ ] **Step 3: Update `src/specagent/cli.py`**

At the top of the file, after existing imports, add a module-level import:

```python
from specagent.tracing.langsmith import setup_langsmith_tracing
```

In the `query()` command (line ~49), call `setup_langsmith_tracing()` before `run_query()`:

```python
@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run a single query through the pipeline."""
    from specagent.graph.workflow import run_query

    setup_langsmith_tracing()
    console.print(f"[blue]Question:[/blue] {question}\n")
    ...
```

In the `benchmark()` command, similarly call `setup_langsmith_tracing()` near the top of the function body (before any pipeline calls).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/specagent && pytest tests/unit/test_langsmith_tracing.py -k "cli" -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
cd /workspace/specagent && pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```

Expected: all previously-passing tests still pass; coverage ≥ 80%.

- [ ] **Step 6: Commit**

```bash
cd /workspace/specagent && git add src/specagent/cli.py tests/unit/test_langsmith_tracing.py
git commit -m "feat(cli): call setup_langsmith_tracing() in query and benchmark commands"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - ✅ `langsmith` in core deps (Task 1)
  - ✅ `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` documented in `.env.example` (Task 2)
  - ✅ Groq path: auto-traces via LangChain callbacks once env vars are active (Tasks 2–3)
  - ✅ Custom endpoint path: `@traceable` on `invoke_with_timing` (Task 5)
  - ✅ API entry point wired (Task 6)
  - ✅ CLI `query` and `benchmark` commands wired (Task 7)
  - ✅ Graceful no-op when key is missing or package absent (Task 3)

- [x] **No placeholders** — all steps contain complete code.

- [x] **Type consistency** — `setup_langsmith_tracing() -> None` used consistently across all tasks.
