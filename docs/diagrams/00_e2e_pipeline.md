# End-to-End Pipeline State Machine

SpecAgent consists of two independent pipelines: the **Ingestion Pipeline** (offline, triggered by `specagent index`) that reads documents into LanceDB, and the **Query Pipeline** (online, triggered via API or CLI) that runs an agentic LangGraph RAG loop. This diagram shows how the two pipelines relate and the overall system lifecycle from cold start to steady-state operation.

```mermaid
stateDiagram-v2
    [*] --> Uninitialized

    state Uninitialized {
        [*] --> AwaitingResourceInit
    }

    Uninitialized --> ResourcesReady : initialize_resources() succeeds\n(store + embedder singletons loaded)
    Uninitialized --> StartupFailed : initialize_resources() raises RuntimeError

    StartupFailed --> [*]

    state ResourcesReady {
        [*] --> Idle

        Idle --> IngestionRunning : CLI index command\nor API ingest call
        Idle --> QueryRunning : CLI query / POST /query

        state IngestionRunning {
            [*] --> ScanningFolder
            ScanningFolder --> IngestionConcurrent : candidates found
            ScanningFolder --> FTSRebuild : no candidates (still rebuilds FTS)
            IngestionConcurrent --> FTSRebuild : all asyncio.gather tasks complete
            FTSRebuild --> IngestionDone
            IngestionDone --> [*]
        }

        state QueryRunning {
            [*] --> RouterNode
            RouterNode --> RetrieverNode : route_decision = retrieve
            RouterNode --> QueryRejected : route_decision = reject
            QueryRejected --> [*]

            RetrieverNode --> DAGRetrieverNode : call-flow query AND dag_retrieval enabled
            RetrieverNode --> GraderNode : direct path

            DAGRetrieverNode --> GraderNode

            GraderNode --> RewriterNode : quality_is_poor AND rewrite_count < max_rewrites
            GraderNode --> GeneratorNode : quality OK or rewrite limit reached

            RewriterNode --> RetrieverNode : loop back with rewritten_question

            GeneratorNode --> HallucinationCheckNode
            HallucinationCheckNode --> GeneratorNode : not_grounded AND regeneration_count <= 1
            HallucinationCheckNode --> QueryDone : grounded / partial / unknown\nor regeneration_count > 1

            QueryDone --> [*]
        }

        IngestionRunning --> Idle
        QueryRunning --> Idle
    }
```
