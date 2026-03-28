"""Unit tests for LLM factory."""

from unittest.mock import MagicMock, patch

import pytest

from specagent.llm.factory import LLMProtocol, _GroqAdapter, create_llm, get_llm


@pytest.mark.unit
def test_llm_protocol_invoke_is_callable():
    """LLMProtocol.invoke stub is reachable (covers Protocol method body)."""
    result = LLMProtocol.invoke(None, "test prompt")  # type: ignore[arg-type]
    assert result is None


@pytest.mark.unit
class TestCreateLLM:
    """Tests for create_llm factory function."""

    # ------------------------------------------------------------------
    # Groq provider
    # ------------------------------------------------------------------

    @patch("langchain_openai.ChatOpenAI")
    @patch("specagent.config.settings")
    def test_create_llm_groq_returns_adapter(self, mock_settings, mock_chat_openai):
        """create_llm with llm_provider='groq' returns a _GroqAdapter."""
        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = "gsk_test"
        mock_settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        mock_settings.groq_max_tokens = 1024
        mock_settings.groq_reasoning_effort = ""
        mock_settings.llm_temperature = 0.1
        mock_chat_openai.return_value = MagicMock()

        llm = create_llm()

        assert isinstance(llm, _GroqAdapter)

    @patch("langchain_openai.ChatOpenAI")
    @patch("specagent.config.settings")
    def test_create_llm_groq_uses_correct_base_url(self, mock_settings, mock_chat_openai):
        """Groq ChatOpenAI is pointed at the Groq API base URL."""
        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = "gsk_test"
        mock_settings.groq_model = "llama-3.1-8b-instant"
        mock_settings.groq_max_tokens = 1024
        mock_settings.groq_reasoning_effort = ""
        mock_settings.llm_temperature = 0.0
        mock_chat_openai.return_value = MagicMock()

        create_llm(temperature=0.0)

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "https://api.groq.com/openai/v1"
        assert call_kwargs["model"] == "llama-3.1-8b-instant"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["model_kwargs"]["max_tokens"] == 1024

    @patch("langchain_openai.ChatOpenAI")
    @patch("specagent.config.settings")
    def test_create_llm_groq_reasoning_effort_included_when_set(
        self, mock_settings, mock_chat_openai
    ):
        """reasoning_effort is passed in model_kwargs when non-empty."""
        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = "gsk_test"
        mock_settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        mock_settings.groq_max_tokens = 1024
        mock_settings.groq_reasoning_effort = "medium"
        mock_settings.llm_temperature = 0.1
        mock_chat_openai.return_value = MagicMock()

        create_llm()

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["model_kwargs"]["reasoning_effort"] == "medium"

    @patch("langchain_openai.ChatOpenAI")
    @patch("specagent.config.settings")
    def test_create_llm_groq_no_reasoning_effort_when_empty(self, mock_settings, mock_chat_openai):
        """reasoning_effort is omitted from model_kwargs when empty string."""
        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = "gsk_test"
        mock_settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        mock_settings.groq_max_tokens = 1024
        mock_settings.groq_reasoning_effort = ""
        mock_settings.llm_temperature = 0.1
        mock_chat_openai.return_value = MagicMock()

        create_llm()

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs["model_kwargs"]

    @patch("specagent.config.settings")
    def test_create_llm_groq_raises_when_no_api_key(self, mock_settings):
        """create_llm raises ValueError when llm_provider='groq' and no API key."""
        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = ""
        mock_settings.llm_temperature = 0.1

        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            create_llm()

    # ------------------------------------------------------------------
    # _GroqAdapter
    # ------------------------------------------------------------------

    def test_groq_adapter_invoke_returns_string(self):
        """_GroqAdapter.invoke() calls the inner model and returns str content."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="hello from groq")
        adapter = _GroqAdapter(mock_model)

        result = adapter.invoke("test prompt")

        assert result == "hello from groq"
        assert isinstance(result, str)

    def test_groq_adapter_invoke_converts_non_string_content(self):
        """_GroqAdapter.invoke() converts list content to str."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content=["part1", "part2"])
        adapter = _GroqAdapter(mock_model)

        result = adapter.invoke("test")

        assert isinstance(result, str)

    # ------------------------------------------------------------------
    # custom_endpoint provider
    # ------------------------------------------------------------------

    @patch("specagent.llm.custom_endpoint.CustomEndpointLLM")
    @patch("specagent.config.settings")
    def test_create_llm_custom_endpoint_default_temperature(
        self, mock_settings, mock_custom_endpoint
    ):
        """create_llm with llm_provider='custom_endpoint' uses default temperature."""
        mock_settings.llm_provider = "custom_endpoint"
        mock_settings.use_custom_endpoint = False
        mock_settings.custom_endpoint_url = "http://localhost:8000"
        mock_settings.llm_temperature = 0.8
        mock_settings.llm_max_tokens = 1024
        mock_custom_endpoint.return_value = MagicMock()

        create_llm()

        mock_custom_endpoint.assert_called_once_with(
            endpoint_url="http://localhost:8000",
            temperature=0.8,
            max_tokens=1024,
            timeout=120,
            max_retries=5,
            retry_delay=5.0,
        )

    @patch("specagent.llm.custom_endpoint.CustomEndpointLLM")
    @patch("specagent.config.settings")
    def test_create_llm_custom_endpoint_custom_temperature(
        self, mock_settings, mock_custom_endpoint
    ):
        """create_llm with llm_provider='custom_endpoint' accepts custom temperature."""
        mock_settings.llm_provider = "custom_endpoint"
        mock_settings.use_custom_endpoint = False
        mock_settings.custom_endpoint_url = "http://localhost:8000"
        mock_settings.llm_temperature = 0.8
        mock_settings.llm_max_tokens = 1024
        mock_custom_endpoint.return_value = MagicMock()

        create_llm(temperature=0.0)

        mock_custom_endpoint.assert_called_once_with(
            endpoint_url="http://localhost:8000",
            temperature=0.0,
            max_tokens=1024,
            timeout=120,
            max_retries=5,
            retry_delay=5.0,
        )

    @patch("specagent.llm.custom_endpoint.CustomEndpointLLM")
    @patch("specagent.config.settings")
    def test_create_llm_legacy_use_custom_endpoint_bool(self, mock_settings, mock_custom_endpoint):
        """Legacy use_custom_endpoint=True still routes to CustomEndpointLLM."""
        mock_settings.llm_provider = "local"  # provider not groq/custom...
        mock_settings.use_custom_endpoint = True  # ...but legacy bool overrides
        mock_settings.custom_endpoint_url = "http://localhost:9000"
        mock_settings.llm_temperature = 0.5
        mock_settings.llm_max_tokens = 512
        mock_custom_endpoint.return_value = MagicMock()

        create_llm()

        mock_custom_endpoint.assert_called_once()

    # ------------------------------------------------------------------
    # unknown provider
    # ------------------------------------------------------------------

    @patch("specagent.config.settings")
    def test_create_llm_unknown_provider_raises_value_error(self, mock_settings):
        """create_llm raises ValueError for an unrecognised llm_provider."""
        mock_settings.llm_provider = "unknown_provider"
        mock_settings.use_custom_endpoint = False

        with pytest.raises(ValueError, match="Unknown llm_provider"):
            create_llm()


@pytest.mark.unit
class TestCheckLlmHealth:
    """Tests for check_llm_health provider-aware dispatcher."""

    @patch("specagent.config.settings")
    def test_groq_with_api_key_returns_healthy(self, mock_settings):
        """Groq provider with API key set → healthy, no network call."""
        from specagent.llm.factory import check_llm_health

        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = "gsk_test_key"
        is_healthy, message = check_llm_health()
        assert is_healthy is True
        assert "Groq" in message

    @patch("specagent.config.settings")
    def test_groq_without_api_key_returns_unhealthy(self, mock_settings):
        """Groq provider with no API key → unhealthy."""
        from specagent.llm.factory import check_llm_health

        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = ""
        is_healthy, message = check_llm_health()
        assert is_healthy is False
        assert "GROQ_API_KEY" in message

    @patch("specagent.config.settings")
    @patch("specagent.llm.custom_endpoint.check_llm_endpoint_health")
    def test_custom_endpoint_delegates_to_endpoint_health(self, mock_health, mock_settings):
        """custom_endpoint provider delegates to check_llm_endpoint_health."""
        from specagent.llm.factory import check_llm_health

        mock_settings.llm_provider = "custom_endpoint"
        mock_health.return_value = (True, "Endpoint healthy (responded in 0.12s)")
        is_healthy, message = check_llm_health(timeout=5)
        mock_health.assert_called_once_with(timeout=5)
        assert is_healthy is True

    @patch("specagent.config.settings")
    @patch("specagent.llm.custom_endpoint.check_llm_endpoint_health")
    def test_custom_endpoint_unhealthy_propagates(self, mock_health, mock_settings):
        """Unhealthy custom_endpoint result propagates correctly."""
        from specagent.llm.factory import check_llm_health

        mock_settings.llm_provider = "custom_endpoint"
        mock_health.return_value = (False, "Connection failed: timeout")
        is_healthy, message = check_llm_health()
        assert is_healthy is False
        assert "Connection failed" in message


@pytest.mark.unit
class TestGetLlmCache:
    """Tests for get_llm() caching behaviour."""

    @patch("langchain_openai.ChatOpenAI")
    @patch("specagent.config.settings")
    def test_get_llm_returns_same_instance_on_repeated_calls(self, mock_settings, mock_chat_openai):
        """get_llm() should return the same object on subsequent calls."""
        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = "gsk_test"
        mock_settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        mock_settings.groq_max_tokens = 1024
        mock_settings.groq_reasoning_effort = ""
        mock_settings.llm_temperature = 0.1
        mock_chat_openai.return_value = MagicMock()

        get_llm.cache_clear()
        instance1 = get_llm()
        instance2 = get_llm()
        assert instance1 is instance2
        get_llm.cache_clear()

    @patch("langchain_openai.ChatOpenAI")
    @patch("specagent.config.settings")
    def test_get_llm_cache_keys_by_temperature(self, mock_settings, mock_chat_openai):
        """Different temperatures produce different cached instances."""
        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = "gsk_test"
        mock_settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        mock_settings.groq_max_tokens = 1024
        mock_settings.groq_reasoning_effort = ""
        mock_settings.llm_temperature = 0.1
        mock_chat_openai.return_value = MagicMock()

        get_llm.cache_clear()
        default = get_llm()
        cold = get_llm(temperature=0.0)
        assert default is not cold
        get_llm.cache_clear()

    def test_groq_adapter_last_call_thread_safe(self):  # noqa: PLC0415 — test-only local import
        """Each thread sees its own last_call via threading.local."""
        import threading

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="hello", usage_metadata=None)
        adapter = _GroqAdapter(mock_model)

        results: dict[str, object] = {}
        barrier = threading.Barrier(2)

        def thread_fn(name: str, record: object) -> None:
            barrier.wait()
            adapter._tls.last_call = record
            barrier.wait()
            results[name] = adapter.get_last_call()

        sentinel_a = object()
        sentinel_b = object()
        t1 = threading.Thread(target=thread_fn, args=("a", sentinel_a))
        t2 = threading.Thread(target=thread_fn, args=("b", sentinel_b))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["a"] is sentinel_a
        assert results["b"] is sentinel_b
