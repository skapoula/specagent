"""Unit tests for LangSmith tracing integration."""
import os
import sys
import warnings
from importlib import reload
from unittest.mock import patch

import pytest

from specagent.config import get_settings


@pytest.mark.unit
def test_langsmith_importable():
    """langsmith must be importable as a core dependency (not optional)."""
    import langsmith  # noqa: PLC0415 — intentional in-function import so ImportError surfaces as a test failure

    assert langsmith is not None


@pytest.mark.unit
def test_settings_enable_langsmith_defaults_true():
    """enable_langsmith defaults to True."""
    from specagent.config import Settings  # noqa: PLC0415

    s = Settings()
    assert s.enable_langsmith is True


@pytest.mark.unit
def test_settings_langchain_api_key_defaults_empty():
    """langchain_api_key defaults to empty string."""
    from specagent.config import Settings  # noqa: PLC0415

    s = Settings()
    assert s.langchain_api_key == ""


@pytest.mark.unit
def test_settings_langchain_project_defaults():
    """langchain_project defaults to '3gpp-specagent'."""
    from specagent.config import Settings  # noqa: PLC0415

    s = Settings()
    assert s.langchain_project == "3gpp-specagent"


@pytest.mark.unit
def test_settings_reads_langchain_api_key_from_env():
    """langchain_api_key is populated via constructor kwargs (shell env excluded)."""
    from specagent.config import Settings  # noqa: PLC0415

    s = Settings(langchain_api_key="ls-test-key")
    assert s.langchain_api_key == "ls-test-key"


@pytest.mark.unit
def test_settings_reads_langchain_project_from_env():
    """langchain_project is populated via constructor kwargs (shell env excluded)."""
    from specagent.config import Settings  # noqa: PLC0415

    s = Settings(langchain_project="my-project")
    assert s.langchain_project == "my-project"


@pytest.mark.unit
def test_setup_langsmith_disabled():
    """setup_langsmith_tracing does nothing when enable_langsmith is False."""
    import specagent.tracing.langsmith as ls_module  # noqa: PLC0415 — reload pattern requires in-function import
    reload(ls_module)
    with patch("specagent.tracing.langsmith.settings") as mock_cfg:
        mock_cfg.enable_langsmith = False
        ls_module.setup_langsmith_tracing()
    # No exception = pass


@pytest.mark.unit
def test_setup_langsmith_no_api_key_warns():
    """setup_langsmith_tracing warns when langchain_api_key is empty."""
    import specagent.tracing.langsmith as ls_module  # noqa: PLC0415 — reload pattern requires in-function import
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
    import specagent.tracing.langsmith as ls_module  # noqa: PLC0415 — reload pattern requires in-function import
    reload(ls_module)
    saved = {k: os.environ.pop(k, None) for k in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT")}
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
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.unit
def test_setup_langsmith_does_not_overwrite_existing_project():
    """setup_langsmith_tracing respects a pre-existing LANGCHAIN_PROJECT env var."""
    import specagent.tracing.langsmith as ls_module  # noqa: PLC0415 — reload pattern requires in-function import
    reload(ls_module)
    saved = {k: os.environ.pop(k, None) for k in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT")}
    os.environ["LANGCHAIN_PROJECT"] = "already-set"
    try:
        with patch("specagent.tracing.langsmith.settings") as mock_cfg:
            mock_cfg.enable_langsmith = True
            mock_cfg.langchain_api_key = "ls-test-key"
            mock_cfg.langchain_project = "3gpp-specagent"
            ls_module.setup_langsmith_tracing()
        assert os.environ["LANGCHAIN_PROJECT"] == "already-set"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.unit
def test_setup_langsmith_missing_package_warns():
    """setup_langsmith_tracing warns gracefully when langsmith is not installed."""
    import specagent.tracing.langsmith as ls_module  # noqa: PLC0415 — reload pattern requires in-function import
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


@pytest.mark.unit
def test_setup_langsmith_tracing_exported_from_tracing_package():
    """setup_langsmith_tracing is importable from specagent.tracing."""
    from specagent.tracing import (  # noqa: PLC0415 — intentional late import for test isolation
        setup_langsmith_tracing,
    )
    assert callable(setup_langsmith_tracing)


@pytest.mark.unit
def test_api_lifespan_calls_setup_langsmith_tracing():
    """FastAPI lifespan calls setup_langsmith_tracing() when settings.enable_langsmith is True."""
    from unittest.mock import (  # noqa: PLC0415 — reload pattern requires in-function import
        MagicMock,
        patch,
    )

    from fastapi.testclient import (  # noqa: PLC0415 — reload pattern requires in-function import
        TestClient,
    )

    mock_langsmith = MagicMock()
    mock_init = MagicMock(return_value={"store": "ok", "embedder": "ok"})

    with (
        patch("specagent.api.main.initialize_resources", mock_init),
        patch("specagent.api.main.settings") as mock_cfg,
        patch("specagent.tracing.langsmith.setup_langsmith_tracing", mock_langsmith),
    ):
        mock_cfg.enable_tracing = False
        mock_cfg.enable_langsmith = True
        mock_cfg.cors_allow_origins = ["http://localhost:3000"]
        mock_cfg.api_host = "0.0.0.0"
        mock_cfg.api_port = 8000

        from importlib import reload  # noqa: PLC0415 — reload pattern requires in-function import

        from specagent.api import (  # noqa: PLC0415 — reload pattern requires in-function import
            main as api_module,
        )
        reload(api_module)

        with TestClient(api_module.app):
            pass

    mock_langsmith.assert_called_once()


@pytest.mark.unit
def test_cli_query_calls_setup_langsmith_tracing():
    """CLI query command calls setup_langsmith_tracing() before run_query()."""
    from unittest.mock import (  # noqa: PLC0415 — in-function import for test isolation
        MagicMock,
        patch,
    )

    from typer.testing import CliRunner  # noqa: PLC0415 — in-function import for test isolation

    runner = CliRunner()
    mock_setup = MagicMock()
    mock_run_query = MagicMock(return_value={
        "route_decision": "reject",
        "route_reasoning": "off-topic",
    })

    with (
        patch("specagent.cli.setup_langsmith_tracing", mock_setup),
        patch("specagent.graph.workflow.run_query", mock_run_query),
    ):
        from specagent.cli import app  # noqa: PLC0415 — in-function import for test isolation
        result = runner.invoke(app, ["query", "What is NR?"])

    assert result.exit_code == 0
    mock_setup.assert_called_once()


@pytest.mark.unit
def test_cli_benchmark_calls_setup_langsmith_tracing(tmp_path):
    """CLI benchmark command calls setup_langsmith_tracing()."""
    import json  # noqa: PLC0415 — in-function import for test isolation
    from unittest.mock import (  # noqa: PLC0415 — in-function import for test isolation
        MagicMock,
        patch,
    )

    from typer.testing import CliRunner  # noqa: PLC0415 — in-function import for test isolation

    runner = CliRunner()
    mock_setup = MagicMock()

    # Create a minimal valid benchmark dataset file
    dataset_file = tmp_path / "bench.json"
    dataset_file.write_text(json.dumps([]))

    with patch("specagent.cli.setup_langsmith_tracing", mock_setup):
        from specagent.cli import app  # noqa: PLC0415 — in-function import for test isolation
        runner.invoke(app, ["benchmark", "--dataset", str(dataset_file)])

    mock_setup.assert_called_once()
