-- Kuzu schema for specagent call-flow DAG store.
--
-- This file is kept for reference only. The schema is applied automatically
-- by KuzuConnection._init_schema() on first open — you do not need to run
-- this file manually.
--
-- Node tables:
--   CallFlowDag      — one per extracted call-flow diagram
--   DagParticipant   — unique network-function / actor name (e.g. "AMF", "UE")
--   DagStep          — one message-arrow per step in the sequence diagram
--
-- Relationship tables:
--   (:CallFlowDag)-[:HAS_PARTICIPANT]->(:DagParticipant)
--   (:CallFlowDag)-[:HAS_STEP]->(:DagStep)

CREATE NODE TABLE IF NOT EXISTS CallFlowDag (
    dag_id STRING PRIMARY KEY,
    doc_id STRING,
    source STRING,
    title STRING,
    mermaid_content STRING,
    prose_description STRING,
    ingested_at STRING
);

CREATE NODE TABLE IF NOT EXISTS DagParticipant (
    name STRING PRIMARY KEY
);

-- step_id is a composite key: "{dag_id}::{step_index}" — ensures idempotent re-ingestion.
CREATE NODE TABLE IF NOT EXISTS DagStep (
    step_id STRING PRIMARY KEY,
    dag_id STRING,
    step_index INT64,
    from_actor STRING,
    to_actor STRING,
    message STRING,
    is_async BOOLEAN
);

CREATE REL TABLE IF NOT EXISTS HAS_PARTICIPANT (FROM CallFlowDag TO DagParticipant);

CREATE REL TABLE IF NOT EXISTS HAS_STEP (FROM CallFlowDag TO DagStep);
