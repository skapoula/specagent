"""Singleton resource accessors for the Kuzu DAG store."""

from __future__ import annotations

from functools import lru_cache

from specagent.kuzu.connection import KuzuConnection, get_dag_connection
from specagent.kuzu.dag_store import CallFlowDagStore


@lru_cache(maxsize=1)
def get_dag_store() -> CallFlowDagStore:
    """Return the singleton :class:`CallFlowDagStore` for specagent.

    Reuses the singleton :func:`get_dag_connection` so only one
    :class:`KuzuConnection` (and one on-disk database handle) exists per process.
    Clear the cache with :func:`clear_dag_store_cache` in tests.
    """
    conn: KuzuConnection = get_dag_connection()
    return CallFlowDagStore(connection=conn)


def clear_dag_store_cache() -> None:
    """Clear the lru_cache on get_dag_store (used in tests)."""
    get_dag_store.cache_clear()


__all__ = ["clear_dag_store_cache", "get_dag_store"]
