"""Fastembed embedder with document/query prefix support for nomic-embed-text-v1.5."""

import logging

import numpy as np
from numpy.typing import NDArray

from specagent.config import settings
from specagent.retrieval.exceptions import EmbeddingError
from specagent.retrieval.resources import get_embedder

logger = logging.getLogger(__name__)

# nomic-embed-text-v1.5 uses task prefixes for asymmetric search.
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


def embed_documents(texts: list[str]) -> NDArray[np.float32]:
    """Embed a list of document texts with the 'search_document:' prefix.

    Args:
        texts: Document chunk strings to embed.

    Returns:
        Float32 array of shape (len(texts), embedding_dimension).

    Raises:
        EmbeddingError: If the model fails or returns an unexpected vector count.
    """
    if not texts:
        return np.empty((0, settings.embedding_dimension), dtype=np.float32)
    try:
        prefixed = [_DOC_PREFIX + t for t in texts]
        vecs = list(get_embedder().embed(prefixed, batch_size=settings.embedding_batch_size))
        if len(vecs) != len(texts):
            raise EmbeddingError(f"Model returned {len(vecs)} vectors for {len(texts)} documents")
        result = np.array(vecs, dtype=np.float32)
        logger.debug("Embedded %d chunks → %s float32", len(texts), result.shape)
        return result
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"Failed to embed {len(texts)} documents") from e


def embed_query(query: str) -> NDArray[np.float32]:
    """Embed a single query string with the 'search_query:' prefix.

    Args:
        query: Natural-language search query.

    Returns:
        Float32 array of shape (embedding_dimension,).

    Raises:
        EmbeddingError: If query is empty or the model fails.
    """
    if not query.strip():
        raise EmbeddingError("Query must not be empty")
    try:
        prefixed = _QUERY_PREFIX + query
        vecs = list(get_embedder().embed([prefixed]))
        return np.array(vecs[0], dtype=np.float32)
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"Failed to embed query ({len(query)} chars)") from e
