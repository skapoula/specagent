# API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication is required by default. CORS is restricted to origins listed in the `CORS_ALLOW_ORIGINS` environment variable (default: `http://localhost:3000`).

## Endpoints

### GET /health

**Description:** Returns the current health status and whether the vector index has been loaded.

**Authentication:** None

**Request Example:**

```text
GET /health
```

**Response Example:**

```json
{
  "status": "ok",
  "version": "0.3.0",
  "index_loaded": true
}
```

**Response Fields:**

| Field          | Type    | Description                                    |
| -------------- | ------- | ---------------------------------------------- |
| `status`       | string  | Always `"ok"` when the server is running       |
| `version`      | string  | The installed SpecAgent version                |
| `index_loaded` | boolean | Whether the LanceDB index has been initialized |

---

### POST /query

**Description:** Runs a natural-language question through the full agentic RAG pipeline and returns a cited answer.

**Authentication:** None

**Parameters:**

| Name           | Type           | Required | Description                                                          |
| -------------- | -------------- | -------- | -------------------------------------------------------------------- |
| `question`     | string         | Yes      | Natural language question (3–1000 characters)                        |
| `verbose`      | boolean        | No       | Include node timings and metadata in the response (default: `false`) |
| `max_rewrites` | integer        | No       | Maximum query rewrites for this request, 0–5 (default: `2`)          |
| `library`      | string \| null | No       | Restrict retrieval to documents tagged with this library name        |

**Request Example:**

```text
POST /query
Content-Type: application/json

{
  "question": "What is the maximum number of carriers in NR carrier aggregation?",
  "verbose": false,
  "max_rewrites": 2,
  "library": null
}
```

**Response Example:**

```json
{
  "answer": "NR carrier aggregation supports a maximum of 16 component carriers per UE [TS 38.101 §5.4.1]. Each component carrier can be independently configured with different numerologies and bandwidths [TS 38.331 §6.3.2].",
  "citations": [
    {
      "spec_id": "38.101",
      "section": "5.4.1",
      "chunk_preview": "The maximum number of configured component carriers for a UE is 16..."
    },
    {
      "spec_id": "38.331",
      "section": "6.3.2",
      "chunk_preview": "Component carrier configuration includes numerology and bandwidth parameters..."
    }
  ],
  "confidence": 0.87,
  "metadata": {
    "rewrites": 0,
    "chunks_retrieved": 10,
    "chunks_used": 2,
    "latency_ms": 1843,
    "hallucination_check": "grounded",
    "rewritten_question": null,
    "node_timings": {
      "router": 143,
      "retriever": 421,
      "grader": 187,
      "generator": 892,
      "hallucination_check": 200
    }
  }
}
```

**Response Fields:**

| Field                          | Type           | Description                                                         |
| ------------------------------ | -------------- | ------------------------------------------------------------------- |
| `answer`                       | string         | Generated answer with inline citations in `[TS XX.XXX §Y.Z]` format |
| `citations`                    | array          | List of citation objects extracted from the answer                  |
| `citations[].spec_id`          | string         | 3GPP spec number, e.g. `"38.101"`                                   |
| `citations[].section`          | string         | Section reference, e.g. `"5.4.1"`                                   |
| `citations[].chunk_preview`    | string         | First 120 characters of the source chunk                            |
| `confidence`                   | float          | Composite confidence score, 0.0–1.0                                 |
| `metadata.rewrites`            | integer        | Number of query rewrites performed                                  |
| `metadata.chunks_retrieved`    | integer        | Number of chunks fetched from the index                             |
| `metadata.chunks_used`         | integer        | Number of chunks used for generation                                |
| `metadata.latency_ms`          | integer        | Total pipeline latency in milliseconds                              |
| `metadata.hallucination_check` | string         | Result: `"grounded"`, `"partial"`, or `"not_grounded"`              |
| `metadata.rewritten_question`  | string \| null | The rewritten question if rewriting occurred                        |
| `metadata.node_timings`        | object         | Per-node latency in milliseconds (only when `verbose: true`)        |

**Error Responses:**

| Code | Error            | Meaning                                               |
| ---- | ---------------- | ----------------------------------------------------- |
| 422  | `off_topic`      | The question is not about 3GPP specifications         |
| 500  | `pipeline_error` | The agent pipeline encountered an unrecoverable error |
| 500  | `internal_error` | Unexpected server error                               |

**Error Response Example:**

```json
{
  "detail": {
    "error": "off_topic",
    "message": "This question does not appear to be about 3GPP specifications."
  }
}
```

---

## CLI Reference

The `specagent` CLI is the primary way to run SpecAgent outside of a container.

### specagent query

Run a single question through the pipeline.

```bash
specagent query "Your question here" [--verbose]
```

| Option            | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `--verbose`, `-v` | Print retrieval details, node timings, and confidence breakdown |

### specagent serve

Start the FastAPI server.

```bash
specagent serve [--host HOST] [--port PORT] [--reload]
```

| Option     | Default   | Description                          |
| ---------- | --------- | ------------------------------------ |
| `--host`   | `0.0.0.0` | Bind address                         |
| `--port`   | `8000`    | Bind port                            |
| `--reload` | off       | Enable hot-reload (development only) |

### specagent index

Ingest documents into the LanceDB vector index.

```bash
specagent index [--docs-dir DIR] [--library NAME] [--force] [--max-concurrency N]
```

| Option              | Default        | Description                                                        |
| ------------------- | -------------- | ------------------------------------------------------------------ |
| `--docs-dir`        | `./data/specs` | Directory containing spec files to ingest                          |
| `--library`         | `default`      | Tag to assign to all ingested documents                            |
| `--force`           | off            | Re-ingest files even if they are already indexed (by content hash) |
| `--max-concurrency` | `4`            | Number of concurrent ingestion workers                             |

### specagent benchmark

Run the TSpec-LLM evaluation benchmark.

```bash
specagent benchmark [--dataset FILE] [--output-dir DIR] [--limit N]
```

| Option         | Default               | Description                                    |
| -------------- | --------------------- | ---------------------------------------------- |
| `--dataset`    | built-in test set     | Path to a JSON benchmark dataset file          |
| `--output-dir` | `./benchmark_results` | Directory for JSON and Markdown output reports |
| `--limit`      | all                   | Limit evaluation to the first N questions      |

### specagent download-model

Download the embedding model to local cache. Run once per machine before the first query.

```bash
specagent download-model
```

### specagent version

Print the installed version.

```bash
specagent version
```

---

## Key Environment Variables

| Variable              | Required                           | Default                                     | Description                                     |
| --------------------- | ---------------------------------- | ------------------------------------------- | ----------------------------------------------- |
| `GROQ_API_KEY`        | Yes (unless using custom endpoint) | —                                           | Groq cloud API key                              |
| `GROQ_MODEL`          | No                                 | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model ID                                   |
| `CUSTOM_ENDPOINT_URL` | No                                 | —                                           | OpenAI-compatible LLM endpoint (overrides Groq) |
| `LANCEDB_URI`         | No                                 | `data/lancedb`                              | Path or S3 URI for the vector index             |
| `EMBEDDING_MODEL`     | No                                 | `nomic-ai/nomic-embed-text-v1.5`            | FastEmbed model name                            |
| `RETRIEVAL_TOP_K`     | No                                 | `10`                                        | Number of chunks to retrieve per query          |
| `CORS_ALLOW_ORIGINS`  | No                                 | `http://localhost:3000`                     | Comma-separated list of allowed CORS origins    |
