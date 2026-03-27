# SpecAgent

SpecAgent answers technical questions about 3GPP telecommunications specifications, returning accurate answers with inline citations to the exact spec sections.

## How It Works

```mermaid
flowchart LR
    A[You ask a question] --> B[Agent checks relevance]
    B --> C[Agent retrieves spec sections]
    C --> D[Agent grades relevance]
    D -->|Low quality| E[Agent rewrites query]
    E --> C
    D -->|Good quality| F[Agent writes answer]
    F --> G[Agent checks for errors]
    G --> H[You receive cited answer]
```

The agent routes, retrieves, grades, and verifies — looping to improve if needed — before returning an answer with traceable citations.

## Getting Started

Install SpecAgent and build your index before querying. See [Installation](./installation.md).

## How to Use It

### Run a query from the terminal

1. Open a terminal in the directory where SpecAgent is installed.
2. Run: `specagent query "What is the maximum number of carriers in NR carrier aggregation?"`
3. SpecAgent prints the answer with citations like `[TS 38.101 §5.4.1]`.
4. Add `--verbose` to see retrieval details and timing.

### Start the REST API server

1. Run: `specagent serve`
2. The server starts on `http://localhost:8000`.
3. Send a POST request to `/query` with your question.
4. See [API Reference](./api-reference.md) for the full request and response format.

### Build or rebuild the index

1. Place your 3GPP spec files (PDF, DOCX, HTML, TXT) in a directory, e.g., `./specs/`.
2. Run: `specagent index --docs-dir ./specs`
3. Wait for ingestion to complete — progress is logged to the terminal.
4. Run a query to confirm the index is working.

### Filter queries to a specific library

1. Use `specagent index --docs-dir ./specs --library nr-radio` to tag documents.
2. Run: `specagent query "..." --library nr-radio` to restrict retrieval to that tag.
3. Useful when you have specs from multiple releases or technology areas.

### Download the embedding model

1. Run `specagent download-model` once before your first query on a new machine.
2. This downloads the ONNX embedding model to local cache — no internet needed after this step.

## Common Issues

| Symptom | Fix |
|---|---|
| `GROQ_API_KEY not set` error | Set the environment variable: `export GROQ_API_KEY=your_key` |
| Answer says "I cannot find information" | Run `specagent index` to build the index, or verify your docs directory contains spec files |
| Very slow first query | Run `specagent download-model` first; the embedding model downloads on first use if not cached |
| Port 8000 already in use | Run `specagent serve --port 8001` to use a different port |
| Citations missing from answer | The retrieved chunks may not contain spec section headers; ensure source files are well-formatted |

## Getting Help

Open an issue at the project repository or check [Contributing](./contributing.md) for developer contact details.
