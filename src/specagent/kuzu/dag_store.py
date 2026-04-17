"""CallFlowDagStore — domain DAG operations against the Kuzu embedded graph store.

Stores and retrieves call-flow diagrams extracted from 3GPP spec .docx files.
Uses parameterised Cypher queries throughout (no string interpolation).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from specagent.kuzu.connection import KuzuConnection
    from specagent.kuzu.mermaid_parser import StepRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

_UPSERT_DAG_QUERY = """\
MERGE (d:CallFlowDag {dag_id: $dag_id})
ON CREATE SET d.doc_id = $doc_id,
              d.source = $source,
              d.title = $title,
              d.mermaid_content = $mermaid_content,
              d.prose_description = $prose_description,
              d.ingested_at = $ingested_at
ON MATCH SET  d.doc_id = $doc_id,
              d.source = $source,
              d.title = $title,
              d.mermaid_content = $mermaid_content,
              d.prose_description = $prose_description,
              d.ingested_at = $ingested_at
"""

_UPSERT_PARTICIPANT_QUERY = "MERGE (p:DagParticipant {name: $name})"

_LINK_PARTICIPANT_QUERY = """\
MATCH (d:CallFlowDag {dag_id: $dag_id})
MATCH (p:DagParticipant {name: $name})
MERGE (d)-[:HAS_PARTICIPANT]->(p)
"""

_UPSERT_STEP_QUERY = """\
MERGE (s:DagStep {step_id: $step_id})
ON CREATE SET s.dag_id = $dag_id,
              s.step_index = $step_index,
              s.from_actor = $from_actor,
              s.to_actor = $to_actor,
              s.message = $message,
              s.is_async = $is_async
ON MATCH SET  s.dag_id = $dag_id,
              s.step_index = $step_index,
              s.from_actor = $from_actor,
              s.to_actor = $to_actor,
              s.message = $message,
              s.is_async = $is_async
"""

_LINK_STEP_QUERY = """\
MATCH (d:CallFlowDag {dag_id: $dag_id})
MATCH (s:DagStep {step_id: $step_id})
MERGE (d)-[:HAS_STEP]->(s)
"""

_QUERY_BY_KEYWORD = """\
MATCH (d:CallFlowDag)-[:HAS_STEP]->(s:DagStep)
WHERE any(kw IN $keywords WHERE toLower(s.message) CONTAINS toLower(kw))
   OR any(kw IN $keywords WHERE toLower(d.prose_description) CONTAINS toLower(kw))
   OR any(kw IN $keywords WHERE toLower(d.title) CONTAINS toLower(kw))
RETURN DISTINCT d.dag_id AS dag_id,
       d.doc_id AS doc_id,
       d.source AS source,
       d.title AS title,
       d.prose_description AS prose_description
LIMIT $limit
"""

_GET_MERMAID = """\
MATCH (d:CallFlowDag {dag_id: $dag_id})
RETURN d.mermaid_content AS mermaid_content
"""


class CallFlowDagStore:
    """Domain-level DAG operations for call-flow diagrams in the Kuzu graph store."""

    def __init__(self, connection: KuzuConnection) -> None:
        """Initialise with an existing :class:`~specagent.kuzu.connection.KuzuConnection`.

        Args:
            connection: Active Kuzu connection.
        """
        self._conn = connection

    def store_call_flow_dag(
        self,
        *,
        dag_id: str,
        doc_id: str,
        source: str,
        title: str,
        mermaid_content: str,
        participants: list[str],
        steps: list[StepRecord],
        prose_description: str = "",
    ) -> None:
        """Persist a call-flow DAG to the graph store (idempotent via MERGE).

        Args:
            dag_id: Unique identifier: ``"{doc_name}::{caption}"``.
            doc_id: LanceDB document UUID linking back to the source document.
            source: File path of the source ``.docx`` file.
            title: Human-readable diagram title (caption or heading).
            mermaid_content: Full validated Mermaid ``sequenceDiagram`` block.
            participants: Deduplicated list of participant names.
            steps: Ordered list of :class:`~specagent.kuzu.mermaid_parser.StepRecord`.
            prose_description: One-sentence plain-English description (``prose_fallback``).
        """
        ingested_at = datetime.now(UTC).isoformat()

        # 1. Upsert the DAG node with all scalar properties.
        self._conn.execute_cypher_write(
            _UPSERT_DAG_QUERY,
            {
                "dag_id": dag_id,
                "doc_id": doc_id,
                "source": source,
                "title": title,
                "mermaid_content": mermaid_content,
                "prose_description": prose_description,
                "ingested_at": ingested_at,
            },
        )

        # 2. Upsert each participant node, then link it to the DAG.
        for name in participants:
            self._conn.execute_cypher_write(_UPSERT_PARTICIPANT_QUERY, {"name": name})
            self._conn.execute_cypher_write(
                _LINK_PARTICIPANT_QUERY, {"dag_id": dag_id, "name": name}
            )

        # 3. Upsert each step node, then link it to the DAG.
        for step in steps:
            step_id = f"{dag_id}::{step.step_index}"
            self._conn.execute_cypher_write(
                _UPSERT_STEP_QUERY,
                {
                    "step_id": step_id,
                    "dag_id": dag_id,
                    "step_index": step.step_index,
                    "from_actor": step.from_actor,
                    "to_actor": step.to_actor,
                    "message": step.message,
                    "is_async": step.is_async,
                },
            )
            self._conn.execute_cypher_write(
                _LINK_STEP_QUERY, {"dag_id": dag_id, "step_id": step_id}
            )

        logger.info("Stored DAG %r (%d steps)", dag_id, len(steps))

    def query_dags_by_keyword(
        self,
        keywords: list[str],
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Return DAGs whose steps or title/description match any keyword.

        Args:
            keywords: List of search terms (case-insensitive substring match).
            limit: Maximum number of results to return.

        Returns:
            List of dicts with keys: ``dag_id``, ``doc_id``, ``source``,
            ``title``, ``prose_description``.
        """
        return self._conn.execute_cypher(
            _QUERY_BY_KEYWORD,
            {"keywords": keywords, "limit": limit},
        )

    def get_dag_mermaid(self, dag_id: str) -> str | None:
        """Retrieve the stored Mermaid block for a DAG by its ID.

        Args:
            dag_id: The unique DAG identifier.

        Returns:
            The Mermaid content string, or ``None`` if the DAG does not exist.
        """
        results = self._conn.execute_cypher(_GET_MERMAID, {"dag_id": dag_id})
        if not results:
            return None
        return results[0].get("mermaid_content")

    def health_check(self) -> bool:
        """Return True if the underlying Kuzu connection is healthy."""
        return self._conn.health_check()


__all__ = ["CallFlowDagStore"]
