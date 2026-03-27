"""OpenInference-compatible span attribute emission for RAG pipeline stages.

Uses openinference semantic convention attribute keys so Phoenix can parse
retrieval documents, LLM token counts, and query metadata from the OTel spans.

All functions are no-ops when opentelemetry is not installed or no span is active.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from specagent.graph.state import RetrievedChunk
    from specagent.observability.models import LLMCallRecord

logger = logging.getLogger(__name__)

# OpenInference semantic convention attribute keys.
# Defined as module constants so this file is importable without the
# openinference-semantic-conventions package installed.
_ATTR_INPUT_VALUE = "input.value"
_ATTR_OUTPUT_VALUE = "output.value"
_ATTR_DOC_CONTENT = "document.content"
_ATTR_DOC_SCORE = "document.score"
_ATTR_DOC_ID = "document.id"
_ATTR_DOC_METADATA = "document.metadata"
_ATTR_LLM_PROMPT_TOKENS = "llm.token_count.prompt"
_ATTR_LLM_COMPLETION_TOKENS = "llm.token_count.completion"
_ATTR_LLM_TOTAL_TOKENS = "llm.token_count.total"
_ATTR_LLM_MODEL_NAME = "llm.model_name"
_ATTR_EMBEDDING_LATENCY_MS = "embedding.latency_ms"
_ATTR_SEARCH_LATENCY_MS = "retrieval.latency_ms"
_ATTR_NUM_DOCUMENTS = "retrieval.num_documents"
_ATTR_MEAN_SCORE = "retrieval.mean_similarity_score"
_ATTR_TOP_SCORE = "retrieval.top_similarity_score"
_ATTR_REWRITE_INDEX = "retrieval.rewrite_index"


def emit_retrieval_span(
    embed_ms: float,
    search_ms: float,
    query: str,
    results: "list[RetrievedChunk]",
    rewrite_index: int = 0,
) -> None:
    """Set retrieval document attributes on the current OTel span.

    Args:
        embed_ms: Embedding latency in milliseconds.
        search_ms: Vector search latency in milliseconds.
        query: Query text used for this retrieval.
        results: Retrieved chunks with similarity scores.
        rewrite_index: Which rewrite iteration produced this retrieval.
    """
    try:
        from opentelemetry import trace  # noqa: PLC0415

        span = trace.get_current_span()
        if not span.is_recording():
            return

        span.set_attribute(_ATTR_INPUT_VALUE, query)
        span.set_attribute(_ATTR_EMBEDDING_LATENCY_MS, embed_ms)
        span.set_attribute(_ATTR_SEARCH_LATENCY_MS, search_ms)
        span.set_attribute(_ATTR_NUM_DOCUMENTS, len(results))
        span.set_attribute(_ATTR_REWRITE_INDEX, rewrite_index)

        scores = [r.similarity_score for r in results]
        if scores:
            span.set_attribute(_ATTR_TOP_SCORE, max(scores))
            span.set_attribute(_ATTR_MEAN_SCORE, sum(scores) / len(scores))

        for i, chunk in enumerate(results):
            prefix = f"retrieval.documents.{i}"
            span.set_attribute(f"{prefix}.{_ATTR_DOC_CONTENT}", chunk.content[:500])
            span.set_attribute(f"{prefix}.{_ATTR_DOC_SCORE}", chunk.similarity_score)
            span.set_attribute(f"{prefix}.{_ATTR_DOC_ID}", chunk.chunk_id)
            span.set_attribute(
                f"{prefix}.{_ATTR_DOC_METADATA}",
                f"spec={chunk.spec_id} section={chunk.section}",
            )

    except ImportError:
        pass
    except Exception:
        logger.debug("emit_retrieval_span failed", exc_info=True)


def emit_llm_usage_span(call_record: "LLMCallRecord") -> None:
    """Set LLM token usage attributes on the current OTel span.

    Args:
        call_record: LLMCallRecord with token counts and model info.
    """
    try:
        from opentelemetry import trace  # noqa: PLC0415

        span = trace.get_current_span()
        if not span.is_recording():
            return

        span.set_attribute(_ATTR_LLM_MODEL_NAME, call_record.model)
        if call_record.prompt_tokens is not None:
            span.set_attribute(_ATTR_LLM_PROMPT_TOKENS, call_record.prompt_tokens)
        if call_record.completion_tokens is not None:
            span.set_attribute(_ATTR_LLM_COMPLETION_TOKENS, call_record.completion_tokens)
        if call_record.total_tokens is not None:
            span.set_attribute(_ATTR_LLM_TOTAL_TOKENS, call_record.total_tokens)

    except ImportError:
        pass
    except Exception:
        logger.debug("emit_llm_usage_span failed", exc_info=True)


def emit_query_span(query: str, answer: str | None, trace_id: str) -> None:
    """Set top-level query input/output attributes on the current OTel span.

    Args:
        query: Original user question.
        answer: Generated answer text, if any.
        trace_id: UUID4 trace identifier.
    """
    try:
        from opentelemetry import trace  # noqa: PLC0415

        span = trace.get_current_span()
        if not span.is_recording():
            return

        span.set_attribute(_ATTR_INPUT_VALUE, query)
        span.set_attribute("session.id", trace_id)
        if answer:
            span.set_attribute(_ATTR_OUTPUT_VALUE, answer[:1000])

    except ImportError:
        pass
    except Exception:
        logger.debug("emit_query_span failed", exc_info=True)
