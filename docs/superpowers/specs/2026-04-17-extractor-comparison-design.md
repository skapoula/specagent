# Design: Prose DAG Extractor vs Groq Vision — Head-to-Head Comparison

**Date:** 2026-04-17  
**Test document:** `data/raw/38413-i30.docx`  
**Output:** `scripts/compare_extractors.py` + `scripts/comparison_38413.md`

---

## Goal

Run a side-by-side comparison of two call-flow diagram extraction approaches on a real 3GPP spec file (`38413-i30.docx`), measuring accuracy (steps, actors, Mermaid validity) and practicality (offline/online, cost, speed).

---

## Approaches Being Compared

| Approach            | Input                        | Method                                            | LLM? |
| ------------------- | ---------------------------- | ------------------------------------------------- | ---- |
| **Prose extractor** | MarkItDown Markdown text     | Regex patterns on numbered step lines             | No   |
| **Groq vision**     | Embedded images in .docx ZIP | Groq vision API → Mermaid generation + validation | Yes  |

---

## Architecture & Data Flow

```
38413-i30.docx
     │
     ├─► convert() [MarkItDown pass 1]
     │        │
     │        ├─► extract_prose_call_flows()  ──► ProseCallFlow[]
     │        │         (regex, no LLM)
     │        │
     │        └─► [dry-run stops here]
     │
     └─► convert_docx_with_ocr()  ──► (enriched_md, ExtractedDiagram[])
              (Groq vision API)
                   │
                   └─► ExtractedDiagram[]  (mermaid_content, caption)

Both outputs → align_results() → ComparisonRow[]
ComparisonRow[] → print scorecard to stdout + write comparison_38413.md
```

**Alignment strategy:** Match prose flows to vision diagrams by figure caption text (fuzzy substring match on title vs caption). Unmatched entries appear as prose-only or vision-only rows.

---

## Script Interface

```bash
# Prose extractor only — no API calls, no GROQ_API_KEY needed
python scripts/compare_extractors.py --dry-run

# Full comparison — requires GROQ_API_KEY
python scripts/compare_extractors.py
```

---

## Scorecard Schema

One `ComparisonRow` per figure/diagram. Columns:

| Column          | Source                                             | Notes                            |
| --------------- | -------------------------------------------------- | -------------------------------- |
| `figure_id`     | Prose extractor (or caption)                       | e.g. `4.2.2.2.2-1`               |
| `caption`       | Vision caption or prose title                      |                                  |
| `prose_steps`   | `len(ProseCallFlow.steps)`                         | 0 if not found                   |
| `vision_steps`  | `->>`/`-->>` line count in vision Mermaid          | 0 if not found or dry-run        |
| `prose_actors`  | `len(ProseCallFlow.participants)`                  |                                  |
| `vision_actors` | Unique `participant` lines in vision Mermaid block |                                  |
| `prose_valid`   | `validate_mermaid()` result                        | ✓ / ✗                            |
| `vision_valid`  | `validate_mermaid()` result                        | ✓ / ✗ / — (dry-run)              |
| `winner`        | Heuristic (see below)                              | `prose` / `vision` / `tie` / `—` |

**Winner heuristic:**

- Both valid → more steps wins; equal steps → `tie`
- Only one valid → that one wins
- Neither valid → `—`
- Dry-run → `—`

**Summary row:** total figures, prose wins, vision wins, ties, prose-only, vision-only.

---

## Error Handling

| Scenario                                    | Behaviour                                                                                |
| ------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Missing `GROQ_API_KEY` (non-dry-run)        | Exit early with clear message                                                            |
| Vision API failure per image                | `convert_docx_with_ocr()` handles internally; row shows `vision_steps=0, vision_valid=✗` |
| Prose extractor finds no steps for a figure | Row shows `prose_steps=0, prose_valid=✗`                                                 |
| No figures found at all                     | Print warning and exit 0                                                                 |

---

## Output Files

- **stdout:** formatted Markdown table (scorecard)
- **`scripts/comparison_38413.md`:** same content written to disk; overwrites on re-run

---

## Practicality Notes

- Prose extractor: offline, ~instant, zero cost
- Groq vision: requires API key + network; ~2–5 s/image; 38413 likely has 10–30 images → 30–150 s total
- Both paths reuse existing project code — no new dependencies
