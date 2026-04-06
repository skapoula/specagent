"""Regression test for langsmith.py: setup_langsmith_tracing must use setdefault."""

import os

import pytest

_LANGSMITH_KEYS = ("LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT")


@pytest.mark.unit
def test_setup_langsmith_does_not_overwrite_existing_api_key():
    """setup_langsmith_tracing must use setdefault for LANGCHAIN_API_KEY.

    If LANGCHAIN_API_KEY is already set in the environment, calling
    setup_langsmith_tracing() must not overwrite it with the value from settings.
    """
    import specagent.tracing.langsmith as ls_module  # noqa: PLC0415
    from importlib import reload  # noqa: PLC0415
    from unittest.mock import patch  # noqa: PLC0415

    reload(ls_module)
    # Save and clear all keys that setup_langsmith_tracing() touches
    saved = {k: os.environ.pop(k, None) for k in _LANGSMITH_KEYS}
    os.environ["LANGCHAIN_API_KEY"] = "pre-existing-key"
    try:
        with patch("specagent.tracing.langsmith.settings") as mock_cfg:
            mock_cfg.enable_langsmith = True
            mock_cfg.langchain_api_key = "settings-key"
            mock_cfg.langchain_project = "proj"
            ls_module.setup_langsmith_tracing()
        # The pre-existing value must be preserved
        assert os.environ["LANGCHAIN_API_KEY"] == "pre-existing-key"
    finally:
        # Restore all env vars that the function may have touched
        for k in _LANGSMITH_KEYS:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
