"""Unit tests for LLM factory."""

from unittest.mock import MagicMock, patch

import pytest

from specagent.llm.factory import LLMProtocol, create_llm


@pytest.mark.unit
def test_llm_protocol_invoke_is_callable():
    """LLMProtocol.invoke stub is reachable (covers Protocol method body)."""
    result = LLMProtocol.invoke(None, "test prompt")  # type: ignore[arg-type]
    assert result is None


@pytest.mark.unit
class TestCreateLLM:
    """Tests for create_llm factory function."""

    @patch('langchain_huggingface.HuggingFaceEndpoint')
    @patch('specagent.config.settings')
    def test_create_llm_default_temperature(self, mock_settings, mock_hf_endpoint):
        """Test create_llm uses settings.llm_temperature by default."""
        # Configure mock settings for HuggingFace backend
        mock_settings.use_custom_endpoint = False
        mock_settings.use_local_llm = False
        mock_settings.llm_model = "Qwen/Qwen2.5-3B-Instruct"
        mock_settings.hf_api_key_value = "test-key"
        mock_settings.llm_temperature = 0.7
        mock_settings.llm_max_tokens = 512

        # Mock HuggingFaceEndpoint constructor
        mock_hf_endpoint.return_value = MagicMock()

        # Call without temperature parameter
        create_llm()

        # Verify HuggingFaceEndpoint was called with default temperature
        mock_hf_endpoint.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-3B-Instruct",
            huggingfacehub_api_token="test-key",
            temperature=0.7,
            max_new_tokens=512
        )

    @patch('langchain_huggingface.HuggingFaceEndpoint')
    @patch('specagent.config.settings')
    def test_create_llm_custom_temperature(self, mock_settings, mock_hf_endpoint):
        """Test create_llm accepts custom temperature parameter."""
        # Configure mock settings for HuggingFace backend
        mock_settings.use_custom_endpoint = False
        mock_settings.use_local_llm = False
        mock_settings.llm_model = "Qwen/Qwen2.5-3B-Instruct"
        mock_settings.hf_api_key_value = "test-key"
        mock_settings.llm_temperature = 0.7
        mock_settings.llm_max_tokens = 512

        # Mock HuggingFaceEndpoint constructor
        mock_hf_endpoint.return_value = MagicMock()

        # Call with custom temperature
        create_llm(temperature=0.0)

        # Verify HuggingFaceEndpoint was called with custom temperature
        mock_hf_endpoint.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-3B-Instruct",
            huggingfacehub_api_token="test-key",
            temperature=0.0,  # Override
            max_new_tokens=512
        )

    @patch('specagent.llm.custom_endpoint.CustomEndpointLLM')
    @patch('specagent.config.settings')
    def test_create_llm_custom_endpoint_default_temperature(
        self, mock_settings, mock_custom_endpoint
    ):
        """Test create_llm with custom endpoint uses default temperature."""
        # Configure mock settings for custom endpoint
        mock_settings.use_custom_endpoint = True
        mock_settings.custom_endpoint_url = "http://localhost:8000"
        mock_settings.llm_temperature = 0.8
        mock_settings.llm_max_tokens = 1024

        # Mock CustomEndpointLLM constructor
        mock_custom_endpoint.return_value = MagicMock()

        # Call without temperature parameter
        create_llm()

        # Verify CustomEndpointLLM was called with default temperature
        mock_custom_endpoint.assert_called_once_with(
            endpoint_url="http://localhost:8000",
            temperature=0.8,
            max_tokens=1024,
            timeout=120,
            max_retries=5,
            retry_delay=5.0
        )

    @patch('specagent.llm.custom_endpoint.CustomEndpointLLM')
    @patch('specagent.config.settings')
    def test_create_llm_custom_endpoint_custom_temperature(
        self, mock_settings, mock_custom_endpoint
    ):
        """Test create_llm with custom endpoint accepts custom temperature."""
        # Configure mock settings for custom endpoint
        mock_settings.use_custom_endpoint = True
        mock_settings.custom_endpoint_url = "http://localhost:8000"
        mock_settings.llm_temperature = 0.8
        mock_settings.llm_max_tokens = 1024

        # Mock CustomEndpointLLM constructor
        mock_custom_endpoint.return_value = MagicMock()

        # Call with custom temperature
        create_llm(temperature=0.0)

        # Verify CustomEndpointLLM was called with custom temperature
        mock_custom_endpoint.assert_called_once_with(
            endpoint_url="http://localhost:8000",
            temperature=0.0,  # Override
            max_tokens=1024,
            timeout=120,
            max_retries=5,
            retry_delay=5.0
        )

    @patch('specagent.config.settings')
    def test_create_llm_local_llm_raises_not_implemented(self, mock_settings):
        """Test create_llm raises NotImplementedError for local LLM."""
        # Configure mock settings for local LLM (not yet supported)
        mock_settings.use_custom_endpoint = False
        mock_settings.use_local_llm = True

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="Local GGUF model support"):
            create_llm()

    @patch('langchain_huggingface.HuggingFaceEndpoint')
    @patch('specagent.config.settings')
    def test_create_llm_temperature_zero(self, mock_settings, mock_hf_endpoint):
        """Test create_llm handles temperature=0.0 correctly (not None)."""
        # Configure mock settings
        mock_settings.use_custom_endpoint = False
        mock_settings.use_local_llm = False
        mock_settings.llm_model = "Qwen/Qwen2.5-3B-Instruct"
        mock_settings.hf_api_key_value = "test-key"
        mock_settings.llm_temperature = 0.5
        mock_settings.llm_max_tokens = 512

        # Mock HuggingFaceEndpoint constructor
        mock_hf_endpoint.return_value = MagicMock()

        # Call with temperature=0.0 (should not fall back to settings)
        create_llm(temperature=0.0)

        # Verify temperature=0.0 was used (not 0.5 from settings)
        mock_hf_endpoint.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-3B-Instruct",
            huggingfacehub_api_token="test-key",
            temperature=0.0,  # Should be 0.0, not 0.5
            max_new_tokens=512
        )

    @patch('langchain_huggingface.HuggingFaceEndpoint')
    @patch('specagent.config.settings')
    def test_create_llm_temperature_none_uses_settings(self, mock_settings, mock_hf_endpoint):
        """Test create_llm with temperature=None uses settings value."""
        # Configure mock settings
        mock_settings.use_custom_endpoint = False
        mock_settings.use_local_llm = False
        mock_settings.llm_model = "Qwen/Qwen2.5-3B-Instruct"
        mock_settings.hf_api_key_value = "test-key"
        mock_settings.llm_temperature = 0.9
        mock_settings.llm_max_tokens = 512

        # Mock HuggingFaceEndpoint constructor
        mock_hf_endpoint.return_value = MagicMock()

        # Call with explicit temperature=None
        create_llm(temperature=None)

        # Verify settings.llm_temperature was used
        mock_hf_endpoint.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-3B-Instruct",
            huggingfacehub_api_token="test-key",
            temperature=0.9,  # From settings
            max_new_tokens=512
        )
