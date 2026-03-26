"""
LLM factory for creating LLM instances based on configuration.

Provides a unified interface for creating LLM clients regardless of backend
(Groq, custom OpenAI-compatible endpoint).

Backend selection is controlled by the ``llm_provider`` setting:

    "groq"            → Groq cloud inference API (default)
    "custom_endpoint" → Self-hosted OpenAI-compatible endpoint (CustomEndpointLLM)

All returned objects implement :class:`LLMProtocol` — ``invoke(prompt) -> str``.
"""

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LLMProtocol(Protocol):
    """Protocol that all LLM clients must implement."""

    def invoke(self, prompt: str) -> str:
        """Call the LLM with a prompt and return the response."""
        ...


class _GroqAdapter:
    """Wraps langchain_openai.ChatOpenAI (pointed at Groq) to satisfy LLMProtocol.

    LangChain's ChatOpenAI returns an AIMessage from .invoke(), not a plain
    string.  This adapter extracts the text content so callers receive a str,
    matching the interface expected by all specagent nodes.
    """

    def __init__(self, chat_model: Any) -> None:
        self._model = chat_model

    def invoke(self, prompt: str) -> str:
        """Send prompt to Groq and return the response as a plain string."""
        from langchain_core.messages import HumanMessage  # noqa: PLC0415

        response = self._model.invoke([HumanMessage(content=prompt)])
        content = response.content
        return content if isinstance(content, str) else str(content)


def create_llm(temperature: float | None = None) -> LLMProtocol:
    """Create an LLM client based on configuration settings.

    Args:
        temperature: Optional temperature override (0.0-2.0). If None, uses
            ``settings.llm_temperature``.

    Returns:
        LLM client that implements :class:`LLMProtocol`.

    Raises:
        ValueError: If ``llm_provider`` is ``"groq"`` and ``groq_api_key`` is empty,
            or if an unrecognised provider is configured.
    """
    from specagent.config import settings  # noqa: PLC0415

    temp = temperature if temperature is not None else settings.llm_temperature
    provider = settings.llm_provider

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "groq_api_key must be set when llm_provider is 'groq'. "
                "Set the GROQ_API_KEY environment variable to your Groq API key."
            )
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        from pydantic import SecretStr  # noqa: PLC0415

        model_kwargs: dict[str, Any] = {}
        if settings.groq_reasoning_effort:
            model_kwargs["reasoning_effort"] = settings.groq_reasoning_effort

        logger.debug("Creating Groq LLM: model=%s", settings.groq_model)
        chat_model = ChatOpenAI(  # type: ignore[call-arg]  # max_tokens missing from stubs
            model=settings.groq_model,
            api_key=SecretStr(settings.groq_api_key),
            base_url="https://api.groq.com/openai/v1",
            temperature=temp,
            timeout=60,
            max_tokens=settings.groq_max_tokens,
            model_kwargs=model_kwargs,
        )
        return _GroqAdapter(chat_model)

    elif provider == "custom_endpoint" or settings.use_custom_endpoint:
        from specagent.llm.custom_endpoint import CustomEndpointLLM  # noqa: PLC0415

        logger.debug("Creating custom-endpoint LLM: url=%s", settings.custom_endpoint_url)
        return CustomEndpointLLM(
            endpoint_url=settings.custom_endpoint_url,
            temperature=temp,
            max_tokens=settings.llm_max_tokens,
            timeout=120,
            max_retries=5,
            retry_delay=5.0,
        )

    else:
        raise ValueError(
            f"Unknown llm_provider {provider!r}. "
            "Set llm_provider='groq' or llm_provider='custom_endpoint'."
        )


def check_llm_health(timeout: int = 30) -> tuple[bool, str]:
    """Check LLM backend health based on the configured provider.

    For Groq: verifies the API key is present (no network call).
    For custom_endpoint: performs an HTTP health check against the endpoint.

    Args:
        timeout: Timeout in seconds for the HTTP check (custom_endpoint only).

    Returns:
        Tuple of (is_healthy: bool, message: str).
    """
    from specagent.config import settings  # noqa: PLC0415

    if settings.llm_provider == "groq":
        if settings.groq_api_key:
            return True, "Groq API key present"
        return False, "GROQ_API_KEY is not set"

    from specagent.llm.custom_endpoint import check_llm_endpoint_health  # noqa: PLC0415

    return check_llm_endpoint_health(timeout=timeout)


# Alias for backwards compatibility
get_llm = create_llm
