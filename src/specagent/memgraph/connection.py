"""Kuzu embedded graph database connection for the specagent DAG store.

Replaces the Bolt-based MemgraphConnection with an in-process Kuzu database.
No external server or port required — the database is a directory on disk.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import kuzu

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL — executed once on first open, idempotent on reconnect.
# ---------------------------------------------------------------------------

_DDL_STATEMENTS: list[str] = [
    """\
CREATE NODE TABLE IF NOT EXISTS CallFlowDag (
    dag_id STRING PRIMARY KEY,
    doc_id STRING,
    source STRING,
    title STRING,
    mermaid_content STRING,
    prose_description STRING,
    ingested_at STRING
)""",
    """\
CREATE NODE TABLE IF NOT EXISTS DagParticipant (
    name STRING PRIMARY KEY
)""",
    """\
CREATE NODE TABLE IF NOT EXISTS DagStep (
    step_id STRING PRIMARY KEY,
    dag_id STRING,
    step_index INT64,
    from_actor STRING,
    to_actor STRING,
    message STRING,
    is_async BOOLEAN
)""",
    "CREATE REL TABLE IF NOT EXISTS HAS_PARTICIPANT (FROM CallFlowDag TO DagParticipant)",
    "CREATE REL TABLE IF NOT EXISTS HAS_STEP (FROM CallFlowDag TO DagStep)",
]


class KuzuConnection:
    """Embedded Kuzu graph database connection for specagent.

    Opens (or creates) a Kuzu database at *db_path* and initialises the
    DAG-store schema on first use. No Bolt server or network port required.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Open or create the Kuzu database at *db_path*.

        Args:
            db_path: Filesystem path to the Kuzu database directory.
                     Created automatically if it does not exist.
        """
        self._db = kuzu.Database(str(db_path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()
        logger.debug("KuzuConnection opened at %s", db_path)

    def _init_schema(self) -> None:
        """Create node and relationship tables if they do not already exist."""
        for ddl in _DDL_STATEMENTS:
            self._conn.execute(ddl)

    def execute_cypher(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as plain dicts.

        Args:
            query: Cypher query string.
            params: Query parameters (``$param`` substitutions).

        Returns:
            List of result records as plain dicts.
        """
        result = self._conn.execute(query, parameters=params or {})
        column_names = result.get_column_names()
        rows: list[dict[str, Any]] = []
        while result.has_next():
            rows.append(dict(zip(column_names, result.get_next())))
        return rows

    def execute_cypher_write(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Execute a write Cypher query.

        Args:
            query: Cypher write query string.
            params: Query parameters.
        """
        self._conn.execute(query, parameters=params or {})

    def health_check(self) -> bool:
        """Return True if the Kuzu database is reachable."""
        try:
            result = self.execute_cypher("RETURN 1 AS health")
            return len(result) == 1 and result[0].get("health") == 1
        except Exception:
            return False

    def close(self) -> None:
        """No-op — Kuzu connections are closed when garbage collected."""

    def __enter__(self) -> KuzuConnection:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@lru_cache(maxsize=1)
def get_dag_connection() -> KuzuConnection:
    """Return the singleton KuzuConnection for specagent.

    Reads the database path from :data:`specagent.config.settings`.
    Clear the cache with :func:`clear_dag_connection_cache` in tests.
    """
    from specagent.config import settings  # noqa: PLC0415

    return KuzuConnection(settings.kuzu_db_path)


def clear_dag_connection_cache() -> None:
    """Clear the lru_cache on get_dag_connection (used in tests)."""
    get_dag_connection.cache_clear()


__all__ = [
    "KuzuConnection",
    "clear_dag_connection_cache",
    "get_dag_connection",
]
