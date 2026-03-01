"""Singleton resource management for LanceDB Store and fastembed embedder.

Provides cached instances of expensive resources. Uses lru_cache to ensure
resources are initialized once and reused across all queries.

Usage:
    store = get_store()          # Opens LanceDB on first call; instant thereafter
    embedder = get_embedder()    # Loads ONNX model on first call; instant thereafter
    clear_resource_cache()       # Reset in tests
"""

import logging
from functools import lru_cache

from fastembed import TextEmbedding

from specagent.config import settings
from specagent.retrieval.store import Store

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_store() -> Store:
    """Get or create the global LanceDB Store instance.

    Returns:
        Store: Ready-to-use LanceDB store.
    """
    logger.info("Opening LanceDB store at %s", settings.lancedb_uri)
    store = Store(
        uri=str(settings.lancedb_uri),
        table_name=settings.lancedb_table_name,
    )
    logger.info("LanceDB store ready")
    return store


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    """Get or create the global fastembed TextEmbedding instance.

    Returns:
        TextEmbedding: Ready-to-use fastembed embedder.
    """
    logger.info("Loading embedding model: %s", settings.embedding_model)
    embedder = TextEmbedding(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("Embedding model loaded")
    return embedder


def initialize_resources() -> dict[str, bool]:
    """Explicitly initialize all resources for eager loading at API startup.

    Returns:
        dict: {"store": bool, "embedder": bool}

    Raises:
        RuntimeError: If any resource fails to initialize.
    """
    status: dict[str, bool] = {}

    try:
        get_store()
        status["store"] = True
    except Exception as e:
        status["store"] = False
        raise RuntimeError(f"Failed to open LanceDB store: {e}") from e

    try:
        get_embedder()
        status["embedder"] = True
    except Exception as e:
        status["embedder"] = False
        raise RuntimeError(f"Failed to load embedding model: {e}") from e

    return status


def clear_resource_cache() -> None:
    """Clear all cached resources. Used in tests to reset state between cases."""
    get_store.cache_clear()
    get_embedder.cache_clear()
    logger.debug("Resource cache cleared")
