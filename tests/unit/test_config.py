"""Unit tests for configuration module."""

import os
from unittest.mock import patch

import pytest


class TestSettings:
    """Tests for Settings class."""

    def test_settings_loads_from_env(self, mock_settings):
        """Settings should load values from environment variables."""
        assert mock_settings.chunk_size_tokens == 512
        assert mock_settings.enable_tracing is False

    def test_settings_paths_are_resolved(self, mock_settings):
        """Path settings should be resolved to absolute paths."""
        assert mock_settings.data_dir.is_absolute()
        assert mock_settings.lancedb_uri.is_absolute()

    def test_get_settings_is_cached(self):
        """get_settings should return cached instance."""
        with patch.dict(os.environ, {}):
            from specagent.config import get_settings

            # Clear cache first
            get_settings.cache_clear()

            settings1 = get_settings()
            settings2 = get_settings()

            assert settings1 is settings2


class TestNewPipelineSettings:
    """Tests for LanceDB/fastembed pipeline settings added in v0.3."""

    @pytest.mark.unit
    def test_lancedb_uri_has_default(self):
        """lancedb_uri field exists and defaults to a path containing 'lancedb'."""
        from specagent.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert hasattr(s, "lancedb_uri")
        assert "lancedb" in str(s.lancedb_uri).lower()

    @pytest.mark.unit
    def test_embedding_dimension_is_768(self):
        """embedding_dimension defaults to 768 for nomic-embed-text-v1.5."""
        from specagent.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.embedding_dimension == 768

    @pytest.mark.unit
    def test_chunk_size_tokens_has_default(self):
        """chunk_size_tokens field exists and defaults to 512."""
        from specagent.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert hasattr(s, "chunk_size_tokens")
        assert s.chunk_size_tokens == 512

    @pytest.mark.unit
    def test_chunk_overlap_tokens_must_be_less_than_chunk_size_tokens(self):
        """chunk_overlap_tokens >= chunk_size_tokens must raise a validation error."""
        import os

        from pydantic import ValidationError

        with patch.dict(
            os.environ,
            {"CHUNK_SIZE_TOKENS": "256", "CHUNK_OVERLAP_TOKENS": "256"},
            clear=False,
        ):
            from specagent.config import Settings

            with pytest.raises((ValidationError, ValueError)):
                Settings()
