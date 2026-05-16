# Developer Guide

Technical architecture reference for contributors and integrators. For setup and commands see [Contributing](contributing.md); for REST endpoints see [API Reference](api-reference.md).

---

## System Architecture

SpecAgent is a four-layer system: an **entry layer** (CLI + FastAPI), an **agentic pipeline** (LangGraph), a **retrieval layer** (LanceDB hybrid search + optional Kuzu DAG store), and a **support layer** (LLM factory, observability, evaluation).

### C4 Context

```{figure} diagrams/c4-context.svg
:alt: C4 Context diagram
:align: center

SpecAgent in relation to its users and external systems.
```

### C4 Container

```{figure} diagrams/c4-container.svg
:alt: C4 Container diagram
:align: center

Runtime containers: API server, LangGraph pipeline, LanceDB vector store, fastembed ONNX model, Kuzu graph DB, Groq LLM API.
```

### C4 Component

```{figure} diagrams/c4-component.svg
:alt: C4 Component diagram
:align: center

LangGraph pipeline components: router, retriever, DAG retriever, grader, rewriter, generator, hallucination checker.
```

---

## Query Pipeline

The query pipeline is a compiled LangGraph `StateGraph`. Every node has the signature `node(state: GraphState) -> GraphState` and **must never raise** — errors are written to `state["error"]`.

### Workflow

```{mermaid}
flowchart TD
    START([START]) --> router

    router -->|reject| END1([END])
    router -->|retrieve| retriever

    retriever -->|call-flow + DAG enabled| dag_retriever
    retriever -->|other| grader
    dag_retriever --> grader

    grader -->|poor quality AND rewrites left| rewriter
    grader -->|quality OK or limit reached| generator
    rewriter --> retriever

    generator --> hallucination_check
    hallucination_check -->|not_grounded ≤1 retry| generator
    hallucination_check -->|grounded / partial / unknown| END2([END])
```

### Node Reference

::::{grid} 1 1 2 2
:gutter: 2

:::{card} router
**LLM calls:** 1 (`RouteDecision`)

**Reads:** `question`

**Writes:** `route_decision`, `route_reasoning`

Classifies query as `"retrieve"` (3GPP-related) or `"reject"` (off-topic). Defaults to `"retrieve"` on LLM error.
:::

:::{card} retriever
**LLM calls:** 0 (embedding only)

**Reads:** `question` / `rewritten_question`, `library_filter`

**Writes:** `retrieved_chunks`, `retrieval_events`

Hybrid BM25 + ANN search against LanceDB. Embeds query with `"search_query: "` prefix (nomic asymmetric requirement). Returns top-`retrieval_top_k` (default 10) chunks.
:::

:::{card} dag_retriever _(optional)_
**LLM calls:** 0 (Kuzu only)

**Reads:** `question` / `rewritten_question`

**Writes:** `dag_chunks`

Reached only when `ENABLE_DAG_RETRIEVAL=true` AND the question contains call-flow keywords (`procedure`, `UE`, `AMF`, `gNB`, …). Queries Kuzu by keyword; results injected as `RetrievedChunk` with `section="Call Flow Diagram"`. Gracefully degrades on Kuzu error.
:::

:::{card} grader
**LLM calls:** 0–1 (batched `BatchGradeResult` for mid-range)

**Reads:** `question` / `rewritten_question`, `retrieved_chunks`

**Writes:** `graded_chunks`, `average_confidence`, `grader_auto_count`, `grader_llm_count`

Grades only top-3 chunks by similarity. Auto-grades on extremes; batches mid-range into one LLM call.

| Similarity  | Decision  | Confidence         |
| ----------- | --------- | ------------------ |
| `> 0.82`    | auto-yes  | `= similarity`     |
| `< 0.55`    | auto-no   | `= 1 − similarity` |
| `0.55–0.82` | LLM batch | from model         |

:::

:::{card} rewriter
**LLM calls:** 1

**Reads:** `question` (always original), `retrieved_chunks`, `rewrite_count`

**Writes:** `rewritten_question`, `rewrite_count`

Reformulates query with 3GPP terminology hints and context from the first 5 chunks. Always rewrites from the original question, never from a prior rewrite.
:::

:::{card} generator
**LLM calls:** 1 (`temperature=0.0`)

**Reads:** `question`, `graded_chunks`, `dag_chunks`

**Writes:** `generation`, `citations`

Filters to `relevant="yes"` chunks, sorts by similarity descending, appends DAG chunks as a separate section. Parses `[TS XX.XXX §Y.Z]` citations via regex.
:::

:::{card} hallucination_check
**LLM calls:** 0–1 (skipped above confidence threshold)

**Reads:** `generation`, `average_confidence`, `graded_chunks`

**Writes:** `hallucination_check`, `ungrounded_claims`, `regeneration_count`

Skip thresholds:

- Numerical/tabular content: skip when `average_confidence ≥ 0.65`
- Other content: skip when `average_confidence ≥ 0.70`

Maps LLM output `"yes"→"grounded"`, `"no"→"not_grounded"`, `"partial"→"partial"`.
:::
::::

### Conditional Edges

| Edge                    | Condition                                                                                   | Targets                                       |
| ----------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `should_retrieve`       | `route_decision`                                                                            | `retrieve` → retriever · `reject` → END       |
| `route_after_retriever` | `enable_dag_retrieval` + keyword heuristic                                                  | `dag_retriever` or `grader`                   |
| `should_rewrite`        | top-3 avg similarity ≥ `high_similarity_threshold` (0.85) → fast skip; else quality metrics | `rewrite` → rewriter · `generate` → generator |
| `should_regenerate`     | `hallucination_check == "not_grounded"` AND `regeneration_count ≤ 1`                        | `regenerate` → generator · `finish` → END     |

---

## Pipeline State (`GraphState`)

`GraphState` is `TypedDict(total=False)` — all fields are optional and populated incrementally as the pipeline executes.

::::{tab-set}
:::{tab-item} Input & Routing
| Field | Type | Set by |
|---|---|---|
| `question` | `str` | caller |
| `route_decision` | `Literal["retrieve", "reject"]` | router |
| `route_reasoning` | `str` | router |
| `library_filter` | `str \| None` | caller override |
| `max_rewrites_override` | `int \| None` | caller override |
:::

:::{tab-item} Retrieval & Grading
| Field | Type | Set by |
|---|---|---|
| `rewritten_question` | `str \| None` | rewriter |
| `retrieved_chunks` | `list[RetrievedChunk]` | retriever |
| `dag_chunks` | `list[RetrievedChunk]` | dag_retriever |
| `graded_chunks` | `list[GradedChunk]` | grader |
| `average_confidence` | `float` | grader |
| `grader_auto_count` | `int` | grader |
| `grader_llm_count` | `int` | grader |
| `rewrite_count` | `int` | rewriter |
:::

:::{tab-item} Generation & Verification
| Field | Type | Set by |
|---|---|---|
| `generation` | `str \| None` | generator |
| `citations` | `list[Citation]` | generator |
| `hallucination_check` | `Literal["grounded","not_grounded","partial","unknown"]` | hallucination_check |
| `ungrounded_claims` | `list[str]` | hallucination_check |
| `regeneration_count` | `int` | hallucination_check |
:::

:::{tab-item} Metadata
| Field | Type | Set by |
|---|---|---|
| `error` | `str \| None` | any node on failure |
| `processing_time_ms` | `float` | `run_query()` |
| `node_timings` | `dict[str, float]` | `create_timed_node()` wrapper |
| `trace_id` | `str` | `create_initial_state()` |
| `llm_calls` | `list[LLMCallRecord]` | nodes making LLM calls |
| `retrieval_events` | `list[RetrievalRecord]` | retriever |
:::
::::

### Data Shapes

**`RetrievedChunk`** (dataclass)

| Field              | Type    | Description                            |
| ------------------ | ------- | -------------------------------------- |
| `content`          | `str`   | Chunk text                             |
| `chunk_id`         | `str`   | UUID per chunk                         |
| `doc_id`           | `str`   | UUID shared across a document's chunks |
| `source`           | `str`   | Absolute file path                     |
| `title`            | `str`   | Document title                         |
| `chunk_index`      | `int`   | Zero-based position in document        |
| `file_type`        | `str`   | Extension (`docx`, `pdf`, …)           |
| `spec_id`          | `str`   | Normalised 3GPP ID, e.g. `TS38.321`    |
| `section`          | `str`   | Nearest Markdown heading               |
| `similarity_score` | `float` | Cosine similarity 0.0–1.0              |

**`GradedChunk`** — wraps `RetrievedChunk` + `relevant: Literal["yes","no"]` + `confidence: float`

**`Citation`** — `spec_id`, `section`, `raw_citation` (as it appears in text), `chunk_preview`

---

## Ingestion Pipeline

`ingest(source, library)` is an async 7-step pipeline. `ingest_folder()` runs up to `max_concurrency` (default 4) files in parallel via `asyncio.Semaphore` and rebuilds the FTS index once at the end.

```{mermaid}
flowchart TD
    A([ingest called]) --> B[1 · Read bytes\nextract file_type + last_modified]
    B --> C{2 · SHA-256 dedup}
    C -->|hash unchanged| SKIP([return status=skipped])
    C -->|new or changed| D

    D{docx + OCR enabled?}
    D -->|yes| E[3a · convert_docx_ocr\nGroq Vision → text + diagrams]
    D -->|no| F[3b · MarkItDown convert\ndiagrams = empty]

    E --> G[_extract_title from cover-page table\nthen postprocess strips TOC]
    F --> G

    G --> H{enable_dag_storage?}
    H -->|diagrams found| I[3b · Store OCR diagrams → Kuzu\nfire-and-forget]
    H -->|no diagrams| J[3c · Extract prose call-flows → Kuzu\nfire-and-forget]
    H -->|disabled| K

    I --> K[3d · Copy .docx + .md\nto data/3gpp_rel_NN/]
    J --> K

    K --> L[4 · chunk_with_metadata\ntoken-aware recursive split]
    L --> M[5 · embed_documents\n768-d nomic vectors]
    M --> N[6 · Build ChunkRecord list]
    N --> O[7 · store.upsert_chunks\nwrite-then-delete-old]
    O --> DONE([return IngestResult])
```

### Write-then-delete Semantics

New chunks are written **before** old ones are deleted. A failed write leaves the existing index entry intact. This means a brief period of overlap exists during replace — never a gap.

---

## Chunking Algorithm

`chunk_with_metadata(text)` returns `list[tuple[str, str]]` — (chunk_text, section_header).

```{mermaid}
flowchart TD
    T([text input]) --> S{Recursive split\nhierarchy}
    S -->|"\\n\\n"| M1[Merge with overlap\nchunk_size_tokens=512\noverlap=64]
    S -->|"\\n" if oversized| M2[Recurse deeper]
    S -->|" " if oversized| M3[Recurse deeper]
    S -->|"" last resort| W[Sliding token window]

    M1 --> F[Filter: drop chunks\n< chunk_min_tokens=50]
    M2 --> F
    M3 --> F
    W --> F

    F -->|all below min| P2[Preserve raw chunks\nas fallback]
    F -->|at least one ok| P1[Attach nearest\nMarkdown heading]
    P2 --> P1

    P1 --> OUT([list of chunk, section pairs])
```

**Tokenizer:** Singleton loaded from HuggingFace cache (`tokenizer.json`) via `tokenizers.Tokenizer`. Wrapped by `_TokenizersAdapter` to expose a `transformers`-compatible interface. Loaded lazily on first call with double-checked locking.

**Separator hierarchy:** `["\n\n", "\n", " ", ""]` — semantic boundaries tried first; character-level only as last resort.

**Overlap:** When merging, splits are trimmed from the front until `current_len ≤ overlap_tokens`. Token lengths are cached in a parallel list to avoid redundant tokeniser calls.

---

## Vector Store (LanceDB)

### Schema

```
id             string          chunk-level UUID
doc_id         string          document-level UUID (groups all chunks from one file)
library        string          indexing namespace (e.g. "3gpp-specs")
source         string          absolute file path
content_hash   string          SHA-256 of raw bytes (dedup key)
title          string          document title
content        string          chunk text (BM25 indexed)
embedding      list<float32>   768-d nomic-embed-text-v1.5 vector
chunk_index    int64           zero-based position in document
created_at     string          ISO 8601
metadata       string          JSON dict — includes section_header, release
file_type      string          extension without dot
last_modified  string          ISO 8601 file mtime
page           int64           1-indexed page (0 = not applicable)
release        int64           3GPP release number (0 = unknown)
```

### Hybrid Search

`Store.search()` issues a single `query_type="hybrid"` LanceDB query:

- **Vector leg:** ANN search on `embedding` column
- **BM25 leg:** full-text search on `content` column (FTS index)
- **Re-ranking:** `refine_factor` (default 10) candidates re-scored against BM25
- **Filtering:** optional `WHERE library = ?` clause
- **Similarity:** `max(0.0, 1.0 − L2_distance)` — returned with each chunk

Write operations are serialised via `_write_lock` to make `asyncio.to_thread` calls safe. The FTS index is rebuilt inline after single-file ingest (`rebuild_fts=True`) and deferred to a single call after bulk ingest (`rebuild_fts=False`).

---

## Kuzu DAG Store

Call-flow diagrams extracted from `.docx` images (via Groq Vision OCR) or prose are stored as a property graph in Kuzu.

### Schema

```{mermaid}
graph LR
    D[CallFlowDag\n─────────────────\ndag_id key\ndoc_id\nsource\ntitle\nmermaid_content\nprose_description\ningested_at]

    P[DagParticipant\n─────────────\nname key]

    S[DagStep\n──────────────────\nstep_id key\ndag_id\nstep_index\nfrom_actor\nto_actor\nmessage\nis_async]

    D -->|HAS_PARTICIPANT| P
    D -->|HAS_STEP| S
```

| Node             | Key       | Notable fields                                                   |
| ---------------- | --------- | ---------------------------------------------------------------- |
| `CallFlowDag`    | `dag_id`  | `mermaid_content` (validated Mermaid block), `prose_description` |
| `DagParticipant` | `name`    | actor name e.g. `UE`, `AMF`, `gNB`                               |
| `DagStep`        | `step_id` | `from_actor`, `to_actor`, `message`, `is_async` (dashed arrows)  |

`is_async=True` for dashed Mermaid arrows (`-->>`, `--x`, `-->`); `False` for solid (`->>`, `-x`).

### Keyword Query

```cypher
UNWIND $keywords AS kw
MATCH (d:CallFlowDag)-[:HAS_STEP]->(s:DagStep)
WHERE toLower(s.message) CONTAINS toLower(kw)
   OR toLower(d.prose_description) CONTAINS toLower(kw)
   OR toLower(d.title) CONTAINS toLower(kw)
RETURN DISTINCT d.dag_id, d.doc_id, d.source, d.title, d.prose_description
LIMIT $limit
```

All Kuzu queries use parameterised placeholders — no string interpolation.

---

## Configuration Reference

`Settings` (`config.py`) is a `pydantic_settings.BaseSettings` singleton exposed as `from specagent.config import settings`. Loaded from `/workspace/.env` and `.env`; environment variables override file values.

::::{tab-set}
:::{tab-item} Grading & Rewriting
| Setting | Default | Effect |
|---|---|---|
| `grader_confidence_threshold` | `0.60` | Rewrite triggered when avg confidence below this |
| `min_relevant_chunk_percentage` | `0.50` | Rewrite triggered when relevant fraction below this |
| `high_similarity_threshold` | `0.85` | Top-3 avg ≥ this → skip rewrite entirely |
| `max_rewrites` | `1` | Per-pipeline cap (overridable per request 0–5) |
:::

:::{tab-item} Retrieval
| Setting | Default | Effect |
|---|---|---|
| `retrieval_top_k` | `10` | Chunks fetched per hybrid search |
| `search_refine_factor` | `10` | ANN re-ranking candidates |
| `similarity_threshold` | `0.30` | Minimum similarity to include a chunk |
| `default_library` | `"3gpp-specs"` | Library when no `library_filter` supplied |
:::

:::{tab-item} Hallucination
| Setting | Default | Effect |
|---|---|---|
| `hallucination_skip_threshold` | `0.70` | Skip check when avg_confidence ≥ this (non-numerical) |
| `hallucination_numerical_threshold` | `0.65` | Skip threshold for numerical/tabular answers |
:::

:::{tab-item} Chunking & Embedding
| Setting | Default | Effect |
|---|---|---|
| `chunk_size_tokens` | `512` | Maximum tokens per chunk |
| `chunk_overlap_tokens` | `64` | Overlap between consecutive chunks |
| `chunk_min_tokens` | `50` | Chunks shorter than this are filtered |
| `embedding_batch_size` | `32` | Documents per embed call |
| `embedding_model` | `nomic-ai/nomic-embed-text-v1.5` | 768-d ONNX model |
:::

:::{tab-item} LLM & DAG
| Setting | Default | Effect |
|---|---|---|
| `groq_model` | `meta-llama/llama-4-scout-17b-16e-instruct` | Generation model |
| `vision_model` | same | Diagram OCR model |
| `enable_dag_storage` | `True` | Persist call-flow diagrams to Kuzu |
| `enable_dag_retrieval` | `False` | Use Kuzu in the query pipeline |
| `dag_retrieval_top_k` | `1` | DAGs fetched per keyword query |
| `dag_retrieval_score` | `0.70` | Similarity assigned to DAG chunks |
:::
::::

### Confidence Calculation (API)

After the pipeline completes, `POST /query` derives a single `confidence` score:

```
base = average_confidence
if hallucination_check == "not_grounded": base *= 0.5
if hallucination_check == "partial":      base *= 0.8
base *= (1.0 − rewrite_count × 0.05)
confidence = clamp(base, 0.0, 1.0)
```

---

## Full System State Machine

```{mermaid}
stateDiagram-v2
    [*] --> Uninitialized

    Uninitialized --> ResourcesReady : initialize_resources() succeeds
    Uninitialized --> StartupFailed : initialize_resources() raises RuntimeError
    StartupFailed --> [*]

    state ResourcesReady {
        [*] --> Idle
        Idle --> IngestionRunning : specagent index / ingest API
        Idle --> QueryRunning : specagent query / POST /query

        state IngestionRunning {
            [*] --> ScanningFolder
            ScanningFolder --> IngestionConcurrent : candidates found
            IngestionConcurrent --> FTSRebuild : all tasks complete
            FTSRebuild --> [*]
        }

        state QueryRunning {
            [*] --> RouterNode
            RouterNode --> Rejected : reject
            RouterNode --> RetrieverNode : retrieve
            Rejected --> [*]
            RetrieverNode --> DAGRetrieverNode : call-flow + DAG enabled
            RetrieverNode --> GraderNode : direct path
            DAGRetrieverNode --> GraderNode
            GraderNode --> RewriterNode : poor quality AND rewrites left
            GraderNode --> GeneratorNode : ok or limit reached
            RewriterNode --> RetrieverNode
            GeneratorNode --> HallucinationCheckNode
            HallucinationCheckNode --> GeneratorNode : not_grounded ≤1 retry
            HallucinationCheckNode --> [*] : grounded / partial / unknown
        }

        IngestionRunning --> Idle
        QueryRunning --> Idle
    }
```

---

## Design Decisions

**`total=False` TypedDict state.** Nodes populate fields incrementally without requiring upstream nodes to initialise everything. This makes the graph easy to extend without touching existing nodes.

**Grader caps at top-3 chunks.** Latency over completeness — top-3 by similarity gives sufficient signal for the rewrite decision. The high-similarity shortcut (`≥ 0.85`) bypasses the LLM grader entirely for high-quality queries.

**Content-adaptive hallucination threshold.** Numerical/tabular claims (`0.65`) use a lower skip threshold than prose (`0.70`) because numeric errors are harder to hallucinate plausibly but more damaging if wrong.

**Write-then-delete replace semantics.** `upsert_chunks()` writes new rows first, then deletes old ones by `doc_id`. A failed write leaves the existing entry intact — never a gap in the index.

**`@lru_cache` singletons for I/O resources.** `get_store()`, `get_embedder()`, and `get_llm()` are cached at the process level. Tests call `clear_resource_cache()` to reset between cases — no global mutable state that leaks between tests.

**Asymmetric embedding prefixes.** `nomic-embed-text-v1.5` requires `"search_document: "` at ingest and `"search_query: "` at query time. Omitting or swapping degrades retrieval measurably.

**`ENABLE_DAG_RETRIEVAL` defaults to `False`.** The DAG retrieval path adds a Kuzu query round-trip. It is off by default until enough call-flow diagrams are indexed to make the keyword heuristic reliable.

---

## Extending SpecAgent

### Adding a LangGraph node

1. Write a failing test in `tests/unit/test_<node>.py`.
2. Create `src/specagent/nodes/<node>.py` with `def <node>(state: GraphState) -> GraphState`.
3. Export from `src/specagent/nodes/__init__.py`.
4. Register in `workflow.py`: `workflow.add_node("<node>", _wrap(<node>, "<node>"))`.
5. Add conditional edges and any new `GraphState` fields in `state.py`.

### Adding a file format

Add the extension to `SUPPORTED_EXTENSIONS` in `retrieval/converter.py` and handle it in `convert()`. Chunking, embedding, and storage are format-agnostic.

### Adding an LLM backend

Implement `invoke(prompt: str) -> str` and `get_last_call() -> LLMCallRecord | None`, then add a branch in `llm/factory.py::create_llm()` and extend `LLMProvider` in `config.py`.
