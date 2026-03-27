# Use Cases

## Use Case 1: Looking up a specific 3GPP parameter value

**The situation:** A radio access network engineer needs to confirm the exact value of a physical layer parameter before configuring base station equipment.

**Input:** You run `specagent query "What is the subcarrier spacing for NR FR1 in numerology 1?"` in the terminal.

**What happens:** SpecAgent retrieves the relevant sections from TS 38.211, grades them for relevance, and generates a precise answer with the exact table reference.

**Output:** A terminal answer stating the subcarrier spacing value (30 kHz for numerology 1) with the citation `[TS 38.211 §4.2]`, printed within approximately 2 seconds.

---
### Try It

1. Run `specagent index --docs-dir ./specs` to index your 3GPP spec files.
2. Run `specagent query "What is the subcarrier spacing for NR FR1 in numerology 1?"`
3. Read the answer — the citation in brackets tells you exactly which spec and section to verify.

---

## Use Case 2: Querying specs via the REST API from an application

**The situation:** A developer is building an internal tool that lets their team ask specification questions from a web interface, and needs to integrate SpecAgent as a backend service.

**Input:** The application sends a POST request to `http://localhost:8000/query` with body `{"question": "What are the PDSCH mapping types defined in NR?", "verbose": false}`.

**What happens:** SpecAgent runs the full agentic pipeline — routing, retrieval, grading, and generation — and returns a structured JSON response with the answer, citations array, and a confidence score.

**Output:** A JSON response with `answer` containing the explanation of PDSCH mapping types A and B, a `citations` array listing `[{"spec_id": "38.211", "section": "7.4.1.1", "chunk_preview": "..."}]`, and `confidence: 0.91`.

---
### Try It

1. Start the server: `specagent serve`
2. Send the request:
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the PDSCH mapping types defined in NR?"}'
   ```
3. Read the JSON response — use the `citations` array to link answers back to source documents.
