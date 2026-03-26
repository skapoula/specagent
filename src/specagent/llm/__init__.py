"""LLM clients for specagent."""

from specagent.llm.custom_endpoint import CustomEndpointLLM, create_custom_llm
from specagent.llm.factory import LLMProtocol, create_llm

__all__ = ["CustomEndpointLLM", "LLMProtocol", "create_custom_llm", "create_llm"]
