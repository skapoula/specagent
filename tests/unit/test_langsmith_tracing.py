"""Unit tests for LangSmith tracing integration."""
import os
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_langsmith_importable():
    """langsmith must be importable as a core dependency (not optional)."""
    import langsmith  # noqa: PLC0415 — intentional in-function import so ImportError surfaces as a test failure

    assert langsmith is not None


@pytest.mark.unit
def test_settings_enable_langsmith_defaults_true():
    """enable_langsmith defaults to True."""
    from specagent.config import get_settings  # noqa: PLC0415, I001 — intentional in-function import to allow cache_clear() before construction
    get_settings.cache_clear()
    s = get_settings()
    assert s.enable_langsmith is True
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_langchain_api_key_defaults_empty():
    """langchain_api_key defaults to empty string."""
    from specagent.config import get_settings  # noqa: PLC0415, I001 — intentional in-function import to allow cache_clear() before construction
    get_settings.cache_clear()
    with patch.dict(os.environ, {"LANGCHAIN_API_KEY": ""}, clear=False):
        s = get_settings()
    assert s.langchain_api_key == ""
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_langchain_project_defaults():
    """langchain_project defaults to '3gpp-specagent'."""
    from specagent.config import get_settings  # noqa: PLC0415, I001 — intentional in-function import to allow cache_clear() before construction
    get_settings.cache_clear()
    with patch.dict(os.environ, {}, clear=False):
        s = get_settings()
    assert s.langchain_project == "3gpp-specagent"
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_reads_langchain_api_key_from_env():
    """langchain_api_key is populated from LANGCHAIN_API_KEY env var."""
    from specagent.config import get_settings  # noqa: PLC0415, I001 — intentional in-function import to allow cache_clear() before construction
    get_settings.cache_clear()
    with patch.dict(os.environ, {"LANGCHAIN_API_KEY": "ls-test-key"}):
        s = get_settings()
    assert s.langchain_api_key == "ls-test-key"
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_reads_langchain_project_from_env():
    """langchain_project is populated from LANGCHAIN_PROJECT env var."""
    from specagent.config import get_settings  # noqa: PLC0415, I001 — intentional in-function import to allow cache_clear() before construction
    get_settings.cache_clear()
    with patch.dict(os.environ, {"LANGCHAIN_PROJECT": "my-project"}):
        s = get_settings()
    assert s.langchain_project == "my-project"
    get_settings.cache_clear()
