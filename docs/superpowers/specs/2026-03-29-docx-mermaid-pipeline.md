# Design Spec: docx-to-Markdown Image & Mermaid Pipeline Improvements

**Date:** 2026-03-29
**Scope:** `src/specagent/retrieval/` — docx OCR conversion pipeline
**Status:** Approved for implementation

---

## Overview

Three improvements to the two-pass docx-to-Markdown OCR pipeline:

1. **Caption-based image labeling** — extract MS Word caption text from docx XML and attach it to each image in the Markdown output.
2. **Structured Mermaid generation** — enforce JSON schema output from Groq VLM, expand diagram classification from 1 type to 5, use per-type few-shot examples.
3. **Mermaid validation + correction loop** — validate generated Mermaid structurally before insertion; on failure, re-send to Groq with the original image and error details for one correction attempt before falling back to prose.

---

## Current Architecture (Baseline)

```
extract_images(docx_path)
  → _parse_image_relationships(zf)     rel_id → media_filename list
  → _read_image_bytes(zf, rels)        → List[ExtractedImage]

convert_docx_with_ocr(docx_path)
  → MarkItDown pass                    → markdown with ![image](imageN.png) placeholders
  → for each image: analyze_image()    → ImageAnalysisResult
  → _stitch(markdown, results)         → final markdown
```

**Gaps addressed by this spec:**
- `ExtractedImage` carries no caption metadata.
- Vision prompt is a single free-form string; only `call_flow_diagram` triggers Mermaid.
- No validation of generated Mermaid before insertion.

---

## Phase 1: Caption-Based Image Labeling

### Approach

Parse `word/document.xml` from the docx ZIP to build a `rel_id → caption_text` map. Word stores captions as `w:p` paragraphs with `w:pStyle w:val="Caption"` immediately following the `w:drawing` element that references the image via `r:embed`. Use Clark-notation XML namespace queries throughout for cross-version robustness.

Caption extraction is **best-effort**: any XML parse failure silently returns an empty map. Caption absence is not an error.

### Data Model Change

`ExtractedImage` (`docx_image_extractor.py`) gains one field:

```python
caption: str = ""   # populated from Word Caption-style paragraph; empty if absent
```

Default of `""` makes this non-breaking for all existing callsites.

### New Function: `_extract_caption_map`

**File:** `docx_image_extractor.py`
**Signature:** `_extract_caption_map(zf: zipfile.ZipFile) -> dict[str, str]`

- Returns `{}` if `word/document.xml` is absent from the ZIP.
- Walks all `w:p` elements in document order.
- For each `w:drawing` element, reads the `r:embed` rel ID from the nested `a:blip` or `pic:blipFill` element.
- Looks for the next sibling `w:p` whose `w:pPr/w:pStyle/@w:val` equals `"Caption"`.
- Builds caption text by concatenating all `w:r/w:t` runs in that paragraph.
- On `ET.ParseError` or any exception: log WARNING, return `{}`.

**New constants:**

```python
_WORDML_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL_NS    = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_DOCUMENT_XML_PATH = "word/document.xml"
```

### Pipeline Changes

**`docx_image_extractor.py`:**
- `_read_image_bytes(zf, image_rels)` → `_read_image_bytes(zf, image_rels, caption_map: dict[str, str])` — populates `caption` from `caption_map.get(rel_id, "")`.
- `extract_images(docx_path)` calls `_extract_caption_map(zf)` inside the `zipfile.ZipFile` context block and passes the result to `_read_image_bytes`.

**`docx_ocr_converter.py`:**
- `_stitch(markdown, results)` → `_stitch(markdown, results, captions: dict[int, str])`.
- `_replace()` inner function prepends `\n**Figure: {caption}**\n\n` before image content when `captions.get(idx)` is non-empty.
- `convert_docx_with_ocr` builds `captions = {i: images[i].caption for i in range(len(images)) if images[i].caption}` and passes it to `_stitch`.

### Edge Cases

| Case | Behaviour |
|---|---|
| Image with no following Caption paragraph | `caption = ""`, no label in output |
| Caption paragraph not preceded by a drawing | Ignored — no rel_id to key on |
| Multiple images with interleaved captions | Each resolved independently by rel_id |
| Malformed `word/document.xml` | WARNING logged, `caption = ""` for all images |

### Tests

**New helper in `tests/conftest.py`:** `make_docx_zip_with_caption(image_filename, image_bytes, caption_text) -> bytes` — builds a minimal docx ZIP with `word/document.xml` containing a `w:drawing` + following Caption paragraph and the corresponding relationship entry.

**`tests/unit/test_docx_image_extractor.py` additions:**
- `test_caption_extracted_for_single_image`
- `test_no_caption_returns_empty_string`
- `test_malformed_document_xml_returns_empty_caption`
- `test_caption_map_ignores_non_caption_paragraphs`

**`tests/integration/test_docx_ocr_converter.py` additions:**
- `test_caption_appears_in_stitched_output`
- `test_no_caption_stitches_without_label`

---

## Phase 2: Structured Mermaid Generation with Expanded Classification

### Approach

Replace the free-form `_VISION_PROMPT` string with:
1. A structured system prompt with per-type few-shot examples.
2. Groq `response_format: {"type": "json_schema"}` enforcement for deterministic JSON output.
3. Expanded classification taxonomy covering 5 diagram types, each mapped to a specific Mermaid subtype.

The LLM remains in the loop for classification and content generation; the schema constrains its output format.

### Expanded Diagram Taxonomy

| `image_type` value | Mermaid subtype |
|---|---|
| `call_flow` | `sequenceDiagram` |
| `state_machine` | `stateDiagram-v2` |
| `block_diagram` | `graph LR` |
| `flowchart` | `flowchart TD` |
| `network_topology` | `graph LR` |
| `table` | *(Markdown table — no Mermaid)* |
| `screenshot_text` | *(Extracted text — no Mermaid)* |
| `other` | *(Prose description — no Mermaid)* |

**Breaking change:** `call_flow_diagram` renamed to `call_flow`. All test assertions on `image_type` must be updated. Grep `src/` and `tests/` for `"call_flow_diagram"` before implementation.

### New Constants (`groq_vision_client.py`)

```python
_KNOWN_DIAGRAM_TYPES: frozenset[str]   # all 8 type values above
_MERMAID_SUBTYPE: dict[str, str]       # diagram types → Mermaid header keyword
_SYSTEM_PROMPT: str                    # system message with taxonomy + few-shot examples
_USER_MESSAGE_TEXT: str                # user message asking for image analysis
_RESPONSE_JSON_SCHEMA: dict            # Groq json_schema enforcement object
```

`_RESPONSE_JSON_SCHEMA` structure:

```json
{
  "name": "image_analysis",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "type":           {"type": "string", "enum": ["call_flow", "state_machine", ...]},
      "content":        {"type": "string"},
      "prose_fallback": {"type": "string"},
      "mermaid_diagram_type": {"type": "string"}
    },
    "required": ["type", "content", "prose_fallback"],
    "additionalProperties": false
  }
}
```

`prose_fallback` is **always required** — the prompt instructs the model to provide a one-sentence plain-English description of the image alongside any Mermaid content. This field is consumed by Phase 3.

### API Request Change

```python
messages = [
    {"role": "system", "content": _SYSTEM_PROMPT},
    {"role": "user",   "content": [_USER_MESSAGE_TEXT, image_url_block]},
]
response_format = {"type": "json_schema", "json_schema": _RESPONSE_JSON_SCHEMA}
```

**Fallback:** If Groq returns HTTP 400 with `"response_format"` in the error body, retry once without `response_format` and rely on existing JSON-parsing fallback in `_parse_response`.

### `ImageAnalysisResult` Change

Add field:

```python
prose_fallback: str = ""   # one-sentence plain-English description, always populated
```

### `_parse_response` Change

- Accept expanded `_KNOWN_DIAGRAM_TYPES`.
- For diagram types in `_MERMAID_SUBTYPE`: verify `content` starts with `` ```mermaid `` and contains the expected subtype keyword on line 1. If not, inject the correct header by replacing line 1 of the fenced block. This is deterministic fixup, not validation (validation is Phase 3).
- Populate `result.prose_fallback` from the JSON `prose_fallback` key.

### File Length Guard

If `groq_vision_client.py` would exceed 300 lines after changes, extract `_SYSTEM_PROMPT`, `_USER_MESSAGE_TEXT`, `_RESPONSE_JSON_SCHEMA`, `_MERMAID_SUBTYPE`, and `_KNOWN_DIAGRAM_TYPES` into a new `_vision_prompts.py` module and import from there.

### Config Change (`config.py`)

```python
vision_diagram_types: list[str] = Field(
    default=["call_flow", "state_machine", "block_diagram", "flowchart", "network_topology"],
    description="Diagram types for which Mermaid output is requested from the vision model.",
)
```

### Tests

**`tests/unit/test_groq_vision_client.py`:**
- Update all assertions: `"call_flow_diagram"` → `"call_flow"`.
- `test_parse_response_state_machine_returns_statediagram`
- `test_parse_response_network_topology_returns_graph_lr`
- `test_parse_response_unknown_type_falls_back_to_other`
- `test_api_request_includes_response_format`
- `test_api_request_uses_system_message`
- `test_parse_response_populates_prose_fallback`
- `test_parse_response_prose_fallback_defaults_to_empty`

**`tests/integration/test_docx_ocr_converter.py`:**
- Update `test_placeholder_replaced_with_mermaid` → `image_type="call_flow"`.

---

## Phase 3: Mermaid Validation + Correction Loop

> **Dependency:** Phase 2 must be complete. Phase 3 relies on `prose_fallback` in `ImageAnalysisResult` and the renamed diagram type constants.

### Validation Flow

```
analyze_image(image) → ImageAnalysisResult
  → validate_mermaid(result.markdown_content)
      ✓ valid   → insert into .md  (no second Groq call)
      ✗ invalid → correct_mermaid_diagram(image, prior_attempt, errors, diagram_type)
                    → validate_mermaid(corrected.markdown_content)
                        ✓ valid   → insert into .md
                        ✗ invalid → use result.prose_fallback
                                    (fallback: "_[Diagram: imageN.png — validation failed]_")
```

The second Groq call (`correct_mermaid_diagram`) is **conditional** — it is only made when `validate_mermaid` returns invalid on the first attempt. Valid first attempts skip directly to insertion with no additional API call.

### New File: `mermaid_validator.py`

**Public interface:**

```python
def validate_mermaid(content: str) -> tuple[bool, str]:
    """Validate a fenced Mermaid code block.

    Returns:
        (True, "") if valid.
        (False, reason) if invalid, where reason describes the failure.
    """
```

**Tier 1 — Python structural checks (always runs):**

- `_extract_inner(content)` — extracts body between `` ```mermaid `` fences; returns `None` if absent.
- `_check_header(inner)` — first non-empty line starts with a keyword from `_VALID_DIAGRAM_HEADERS`.
- `_check_has_content(inner)` — at least 2 non-empty, non-comment lines (comments start with `%%`).
- `_check_bracket_balance(inner)` — counts `[`, `]`, `(`, `)`, `{`, `}` and verifies balance.

**`_VALID_DIAGRAM_HEADERS`:**

```python
frozenset([
    "sequenceDiagram", "stateDiagram-v2", "graph", "flowchart",
    "classDiagram", "erDiagram", "gantt", "pie", "gitGraph",
    "mindmap", "timeline", "xychart-beta",
])
```

**Tier 2 — `mmdc` subprocess (opt-in):**

- Only called when `settings.mermaid_validate_with_mmdc` is `True`.
- Writes diagram inner content to a temp `.mmd` file, runs `mmdc -i tmp.mmd -o /dev/null`.
- Non-zero exit or stderr → `(False, stderr)`.
- `FileNotFoundError` (mmdc not on PATH) → `(True, "")` with DEBUG log. Never fails validation when tool is absent.

### New Function: `correct_mermaid_diagram` (`groq_vision_client.py`)

**Signature:**

```python
async def correct_mermaid_diagram(
    image: ExtractedImage,
    prior_attempt: str,
    validation_errors: str,
    diagram_type: str,
    api_key: str,
) -> ImageAnalysisResult:
```

**Request structure:** Two-message conversation:

```python
messages = [
    {"role": "system", "content": _SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "text", "text": (
            f"This image was previously classified as '{diagram_type}'. "
            f"A Mermaid diagram was generated but failed validation with these errors:\n\n"
            f"{validation_errors}\n\n"
            f"Previous attempt:\n{prior_attempt}\n\n"
            "Re-analyze the image and return a corrected Mermaid diagram of type "
            f"'{diagram_type}'. Return the same JSON schema as before."
        )},
        image_url_block,
    ]},
]
```

- Reuses `_RESPONSE_JSON_SCHEMA` and `_parse_response()` — same output contract.
- Does **not** re-classify the image type — `diagram_type` from the first attempt is passed in and locked.

### Wiring in `docx_ocr_converter.py`

New async function `_apply_mermaid_validation`:

```python
async def _apply_mermaid_validation(
    result: ImageAnalysisResult,
    image: ExtractedImage,
    api_key: str,
) -> ImageAnalysisResult:
```

- If `result.image_type not in _DIAGRAM_TYPES_REQUIRING_VALIDATION` → return unchanged.
- Call `validate_mermaid(result.markdown_content)`.
- If valid → return unchanged.
- Log WARNING with validation reason and `image.placeholder_name`.
- Call `await correct_mermaid_diagram(image, result.markdown_content, reason, result.image_type, api_key)`.
- Call `validate_mermaid(corrected.markdown_content)` again.
- If valid → return corrected result.
- Log WARNING (second failure) with both error sets.
- Return `result.model_copy(update={"markdown_content": result.prose_fallback or f"_[Diagram: {image.placeholder_name} — validation failed]_"})`.

**`_DIAGRAM_TYPES_REQUIRING_VALIDATION`:**

```python
frozenset(["call_flow", "state_machine", "block_diagram", "flowchart", "network_topology"])
```

Called in `convert_docx_with_ocr` after each `analyze_image()`:

```python
result = await analyze_image(image, api_key)
result = await _apply_mermaid_validation(result, image, api_key)
results[idx] = result
```

### Config Additions (`config.py`)

```python
mermaid_validate_with_mmdc: bool = Field(
    default=False,
    description=(
        "If True and mmdc is on PATH, validate Mermaid diagrams via mmdc subprocess. "
        "Requires @mermaid-js/mermaid-cli installed globally."
    ),
)
mermaid_mmdc_timeout: int = Field(
    default=10, ge=1, le=60,
    description="Timeout in seconds for mmdc validation subprocess calls.",
)
```

### Tests

**`tests/unit/test_mermaid_validator.py` (new file):**

All tests marked `@pytest.mark.unit`.

- `test_valid_sequence_diagram_passes`
- `test_valid_state_diagram_passes`
- `test_valid_flowchart_passes`
- `test_missing_mermaid_fence_fails`
- `test_unknown_header_fails`
- `test_empty_content_fails` — only header line, no content
- `test_comment_lines_not_counted_as_content`
- `test_unbalanced_brackets_fails`
- `test_mmdc_not_called_when_disabled` — assert `subprocess.run` never called

**`tests/unit/test_groq_vision_client.py` additions:**
- `test_correct_mermaid_diagram_sends_two_messages` — assert messages array structure
- `test_correct_mermaid_diagram_locks_diagram_type` — assert type not reclassified

**`tests/integration/test_docx_ocr_converter.py` additions:**
- `test_invalid_mermaid_triggers_correction_call`
- `test_correction_produces_valid_mermaid_inserts_it`
- `test_correction_failure_falls_back_to_prose`
- `test_valid_mermaid_skips_correction` — assert `correct_mermaid_diagram` never called
- `test_non_diagram_type_skips_validation` — tables/screenshots pass through unvalidated

---

## Implementation Sequence

### Phase 1 (independent — can start immediately)

- [ ] Add `caption: str = ""` to `ExtractedImage`
- [ ] Add namespace constants and `_DOCUMENT_XML_PATH`
- [ ] Implement `_extract_caption_map(zf)`
- [ ] Update `_read_image_bytes` signature and `extract_images`
- [ ] Add `make_docx_zip_with_caption()` to `tests/conftest.py`
- [ ] Update `_stitch` signature; add caption heading in `_replace`
- [ ] Build caption dict in `convert_docx_with_ocr`; pass to `_stitch`
- [ ] Write caption extractor unit tests
- [ ] Write caption stitching integration tests
- [ ] `pytest tests/unit/test_docx_image_extractor.py tests/integration/test_docx_ocr_converter.py -v`
- [ ] `ruff check src/ tests/ && ruff format src/ tests/`

### Phase 2 (independent — can start immediately)

- [ ] Grep `src/` and `tests/` for `"call_flow_diagram"` — list all files to update
- [ ] Expand `_KNOWN_DIAGRAM_TYPES`; rename `call_flow_diagram` → `call_flow`
- [ ] Add `_MERMAID_SUBTYPE`, `_RESPONSE_JSON_SCHEMA`, `_SYSTEM_PROMPT`, `_USER_MESSAGE_TEXT`
- [ ] Add `prose_fallback: str = ""` to `ImageAnalysisResult`
- [ ] Restructure `analyze_image` request body (system message + `response_format`)
- [ ] Update `_parse_response` for expanded types, header fixup, `prose_fallback`
- [ ] Add `response_format` fallback retry on HTTP 400
- [ ] Add `vision_diagram_types` to `config.py`
- [ ] If file > 300 lines: extract prompt constants to `_vision_prompts.py`
- [ ] Update all affected test assertions
- [ ] Write new unit tests for Phase 2
- [ ] `pytest tests/unit/test_groq_vision_client.py -v`

### Phase 3 (requires Phase 2 complete)

- [ ] Create `mermaid_validator.py` with all private helpers and `validate_mermaid`
- [ ] Write all `test_mermaid_validator.py` unit tests; run them
- [ ] Add `correct_mermaid_diagram` to `groq_vision_client.py`
- [ ] Add `_DIAGRAM_TYPES_REQUIRING_VALIDATION` to `docx_ocr_converter.py`
- [ ] Implement `_apply_mermaid_validation` (async)
- [ ] Wire into `convert_docx_with_ocr` loop
- [ ] Add `mermaid_validate_with_mmdc` and `mermaid_mmdc_timeout` to `config.py`
- [ ] Write Phase 3 integration tests
- [ ] `pytest -v` — full suite
- [ ] `ruff check src/ tests/ && ruff format src/ tests/`
- [ ] `pytest --cov=src/specagent --cov-fail-under=80`

---

## Files Summary

### Created

| File | Purpose |
|---|---|
| `src/specagent/retrieval/mermaid_validator.py` | Tier 1/2 Mermaid structural validation |
| `src/specagent/retrieval/_vision_prompts.py` | Prompt/schema constants (if needed to stay under 300 lines) |
| `tests/unit/test_mermaid_validator.py` | Unit tests for validator |

### Modified

| File | Changes |
|---|---|
| `src/specagent/retrieval/docx_image_extractor.py` | `caption` field; `_extract_caption_map`; updated `_read_image_bytes` and `extract_images` |
| `src/specagent/retrieval/groq_vision_client.py` | Expanded types; structured prompt/schema; `prose_fallback`; `correct_mermaid_diagram` |
| `src/specagent/retrieval/docx_ocr_converter.py` | `_stitch` captions param; `_apply_mermaid_validation`; validation wiring |
| `src/specagent/config.py` | 3 new fields: `vision_diagram_types`, `mermaid_validate_with_mmdc`, `mermaid_mmdc_timeout` |
| `tests/conftest.py` | `make_docx_zip_with_caption()` helper |
| `tests/unit/test_docx_image_extractor.py` | Caption extraction tests |
| `tests/unit/test_groq_vision_client.py` | Updated type assertions; new Phase 2 + 3 tests |
| `tests/integration/test_docx_ocr_converter.py` | Caption, validation, and correction loop tests |
