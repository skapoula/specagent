"""Unit tests for LLM adapter observability: get_last_call on Groq and CustomEndpoint adapters."""

from unittest.mock import MagicMock, patch

import pytest

from specagent.observability.models import LLMCallRecord


def _minimal_llm_record(**kwargs) -> LLMCallRecord:
    defaults: dict = {
        "node": "",
        "trace_id": "",
        "model": "x",
        "provider": "groq",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "inference_ms": 1.0,
    }
    defaults.update(kwargs)
    return LLMCallRecord(**defaults)


@pytest.mark.unit
class TestGroqAdapterLastCall:
    def _make_adapter(self):
        from specagent.llm.factory import _GroqAdapter  # noqa: PLC0415

        chat_model = MagicMock()
        return _GroqAdapter(chat_model)

    def test_get_last_call_none_before_invoke(self):
        adapter = self._make_adapter()
        assert adapter.get_last_call() is None

    def test_invoke_sets_last_call(self):
        from langchain_core.messages import AIMessage  # noqa: PLC0415

        adapter = self._make_adapter()
        mock_response = AIMessage(content="hi")
        mock_response.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        adapter._model.invoke.return_value = mock_response

        mock_limiter = MagicMock()
        with (
            patch("specagent.config.settings") as ms,
            patch("specagent.llm.groq_rate_limiter._get_llm_rate_limiter", return_value=mock_limiter),
        ):
            ms.groq_model = "llama-4-scout"
            ms.groq_llm_tokens_per_call_estimate = 6000
            ms.groq_llm_max_retries = 1
            adapter.invoke("test prompt")

        rec = adapter.get_last_call()
        assert rec is not None
        assert rec.prompt_tokens == 10
        assert rec.completion_tokens == 5
        assert rec.total_tokens == 15
        assert rec.provider == "groq"
        assert rec.inference_ms >= 0

    def test_invoke_handles_none_usage(self):
        from langchain_core.messages import AIMessage  # noqa: PLC0415

        adapter = self._make_adapter()
        mock_response = AIMessage(content="hi")
        mock_response.usage_metadata = None
        adapter._model.invoke.return_value = mock_response

        mock_limiter = MagicMock()
        with (
            patch("specagent.config.settings") as ms,
            patch("specagent.llm.groq_rate_limiter._get_llm_rate_limiter", return_value=mock_limiter),
        ):
            ms.groq_model = "llama-4-scout"
            ms.groq_llm_tokens_per_call_estimate = 6000
            ms.groq_llm_max_retries = 1
            adapter.invoke("test prompt")

        rec = adapter.get_last_call()
        assert rec is not None
        assert rec.prompt_tokens is None
        assert rec.completion_tokens is None


@pytest.mark.unit
class TestCustomEndpointLastCall:
    def _make_endpoint(self):
        from specagent.llm.custom_endpoint import CustomEndpointLLM  # noqa: PLC0415

        return CustomEndpointLLM(endpoint_url="http://test.local/v1/chat/completions")

    def test_get_last_call_none_before_invoke(self):
        endpoint = self._make_endpoint()
        assert endpoint.get_last_call() is None

    def test_invoke_with_timing_sets_last_call(self):
        endpoint = self._make_endpoint()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "answer"}}],
            "model": "qwen",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
            },
        }

        with patch("requests.post", return_value=mock_resp):
            endpoint.invoke("test prompt")

        rec = endpoint.get_last_call()
        assert rec is not None
        assert rec.prompt_tokens == 20
        assert rec.completion_tokens == 8
        assert rec.provider == "custom_endpoint"
