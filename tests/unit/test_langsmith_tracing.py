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
    get_settings.cache_clear()
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ENABLE_LANGSMITH", None)
        s = get_settings()
    assert s.enable_langsmith is True
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_langchain_api_key_defaults_empty():
    """langchain_api_key defaults to empty string."""
    get_settings.cache_clear()
    with patch.dict(os.environ, {"LANGCHAIN_API_KEY": ""}, clear=False):
        s = get_settings()
    assert s.langchain_api_key == ""
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_langchain_project_defaults():
    """langchain_project defaults to '3gpp-specagent'."""
    get_settings.cache_clear()
    with patch.dict(os.environ, {}, clear=False):
        s = get_settings()
    assert s.langchain_project == "3gpp-specagent"
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_reads_langchain_api_key_from_env():
    """langchain_api_key is populated from LANGCHAIN_API_KEY env var."""
    get_settings.cache_clear()
    with patch.dict(os.environ, {"LANGCHAIN_API_KEY": "ls-test-key"}):
        s = get_settings()
    assert s.langchain_api_key == "ls-test-key"
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_reads_langchain_project_from_env():
    """langchain_project is populated from LANGCHAIN_PROJECT env var."""
    get_settings.cache_clear()
    with patch.dict(os.environ, {"LANGCHAIN_PROJECT": "my-project"}):
        s = get_settings()
    assert s.langchain_project == "my-project"
    get_settings.cache_clear()


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
