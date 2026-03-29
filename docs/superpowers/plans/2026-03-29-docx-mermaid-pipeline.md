# docx Image Caption + Mermaid Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the two-pass docx OCR pipeline to label images with MS Word caption text, produce deterministic structured Mermaid output for 5 diagram types via a Groq JSON-schema-enforced prompt, and validate every Mermaid block before insertion with a one-shot LLM correction loop on failure.

**Architecture:** Three independent pipeline layers wired in sequence: (1) caption metadata flows from `docx_image_extractor` → `docx_ocr_converter._stitch`; (2) a structured prompt + JSON schema in `_vision_prompts.py` drives `groq_vision_client.analyze_image`; (3) `mermaid_validator.validate_mermaid` gates every diagram result, triggering `correct_mermaid_diagram` on failure before falling back to prose. Phase 3 depends on Phase 2 (`prose_fallback` field + renamed type constants). Phase 1 is independent.

**Tech Stack:** Python 3.11, Pydantic v2, httpx, tenacity, pytest-asyncio (asyncio_mode=auto), stdlib `xml.etree.ElementTree`, stdlib `re`, optional `mmdc` subprocess.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `tests/conftest.py` | Add `make_docx_zip_with_caption()` helper |
| Modify | `src/specagent/retrieval/docx_image_extractor.py` | `caption` field, `_extract_caption_map`, updated `_read_image_bytes` / `extract_images` |
| Modify | `src/specagent/retrieval/docx_ocr_converter.py` | `_stitch` caption param, `_apply_mermaid_validation`, validation wiring |
| Create | `src/specagent/retrieval/_vision_prompts.py` | All prompt/schema constants (keeps `groq_vision_client.py` under 300 lines) |
| Modify | `src/specagent/retrieval/groq_vision_client.py` | `prose_fallback` field, restructured API call, expanded `_parse_response`, `correct_mermaid_diagram` |
| Create | `src/specagent/retrieval/mermaid_validator.py` | `validate_mermaid` + private helpers |
| Modify | `src/specagent/config.py` | 3 new fields: `vision_diagram_types`, `mermaid_validate_with_mmdc`, `mermaid_mmdc_timeout` |
| Modify | `tests/unit/test_docx_image_extractor.py` | Caption extraction tests |
| Modify | `tests/unit/test_groq_vision_client.py` | Updated type assertions, structured output tests, correction tests |
| Create | `tests/unit/test_mermaid_validator.py` | Mermaid validator unit tests |
| Modify | `tests/integration/test_docx_ocr_converter.py` | Caption stitch tests, validation + correction loop tests |

---

## Phase 1: Caption-Based Image Labeling

---

### Task 1: Add `make_docx_zip_with_caption` helper and write failing caption tests

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/unit/test_docx_image_extractor.py`

- [ ] **Step 1: Add `make_docx_zip_with_caption` to `tests/conftest.py`**

Add after the existing `make_docx_zip` function:

```python
def make_docx_zip_with_caption(
    image_filename: str,
    image_bytes: bytes,
    caption_text: str,
) -> bytes:
    """Build a minimal .docx ZIP with one image followed by a Caption paragraph.

    word/document.xml contains a paragraph with a DrawingML a:blip (r:embed="rId1")
    followed immediately by a Caption-style paragraph containing caption_text.
    """
    _IMAGE_REL_TYPE = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    )
    _PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
    _W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    _R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    _A = "http://schemas.openxmlformats.org/drawingml/2006/main"

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{_W}" xmlns:r="{_R}">'
        "<w:body>"
        "<w:p>"
        "<w:r>"
        f'<w:drawing><a:blip xmlns:a="{_A}" r:embed="rId1"/></w:drawing>'
        "</w:r>"
        "</w:p>"
        "<w:p>"
        "<w:pPr>"
        '<w:pStyle w:val="Caption"/>'
        "</w:pPr>"
        "<w:r>"
        f"<w:t>{caption_text}</w:t>"
        "</w:r>"
        "</w:p>"
        "</w:body>"
        "</w:document>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{_PKG_REL_NS}">'
        f'<Relationship Id="rId1" Type="{_IMAGE_REL_TYPE}"'
        f' Target="media/{image_filename}"/>'
        "</Relationships>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", rels_xml)
        zf.writestr(f"word/media/{image_filename}", image_bytes)
    return buf.getvalue()
```

- [ ] **Step 2: Write failing caption tests in `tests/unit/test_docx_image_extractor.py`**

Append after the existing `TestExtractImages` class:

```python
@pytest.mark.unit
class TestCaptionExtraction:
    """Tests for caption metadata populated on ExtractedImage."""

    def test_caption_extracted_for_single_image(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """Caption text from a Caption-style paragraph populates ExtractedImage.caption."""
        from tests.conftest import make_docx_zip_with_caption
        from specagent.retrieval.docx_image_extractor import extract_images

        p = tmp_path / "captioned.docx"
        p.write_bytes(
            make_docx_zip_with_caption(
                image_filename="image1.png",
                image_bytes=large_png,
                caption_text="Figure 3: Network Architecture",
            )
        )
        result = extract_images(p)
        assert len(result) == 1
        assert result[0].caption == "Figure 3: Network Architecture"

    def test_no_caption_returns_empty_string(
        self, docx_one_image: Path
    ) -> None:
        """Image in a docx with no Caption paragraph gets caption=''."""
        from specagent.retrieval.docx_image_extractor import extract_images

        result = extract_images(docx_one_image)
        assert len(result) == 1
        assert result[0].caption == ""

    def test_malformed_document_xml_returns_empty_caption(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """Unparseable word/document.xml logs WARNING and leaves caption=''."""
        import logging

        buf = io.BytesIO()
        _IMAGE_NS = (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        )
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", "<<<not xml>>>")
            zf.writestr(
                "word/_rels/document.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'<Relationship Id="rId1" Type="{_IMAGE_NS}" Target="media/image1.png"/>'
                "</Relationships>",
            )
            zf.writestr("word/media/image1.png", large_png)
        p = tmp_path / "bad_xml.docx"
        p.write_bytes(buf.getvalue())

        from specagent.retrieval.docx_image_extractor import extract_images

        with caplog.at_level(logging.WARNING, logger="specagent.retrieval.docx_image_extractor"):
            result = extract_images(p)

        assert result[0].caption == ""
        assert any("captions unavailable" in r.message for r in caplog.records)

    def test_caption_map_ignores_non_caption_paragraphs(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """Paragraphs without Caption style are not mistaken for captions."""
        from tests.conftest import make_docx_zip_with_caption
        from specagent.retrieval.docx_image_extractor import extract_images

        # Normal paragraph text (not Caption style) should not be captured
        p = tmp_path / "no_caption_style.docx"
        # Build a docx where the following paragraph has Normal style, not Caption
        _W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        _R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        _A = "http://schemas.openxmlformats.org/drawingml/2006/main"
        _PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
        _IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        document_xml = (
            f'<w:document xmlns:w="{_W}" xmlns:r="{_R}"><w:body>'
            f'<w:p><w:r><w:drawing><a:blip xmlns:a="{_A}" r:embed="rId1"/></w:drawing></w:r></w:p>'
            # Normal paragraph — NOT a Caption
            "<w:p><w:pPr><w:pStyle w:val=\"Normal\"/></w:pPr>"
            "<w:r><w:t>Not a caption</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", document_xml)
            zf.writestr(
                "word/_rels/document.xml.rels",
                f'<Relationships xmlns="{_PKG}">'
                f'<Relationship Id="rId1" Type="{_IMAGE_REL}" Target="media/image1.png"/>'
                "</Relationships>",
            )
            zf.writestr("word/media/image1.png", large_png)
        p.write_bytes(buf.getvalue())

        result = extract_images(p)
        assert result[0].caption == ""
```

Note: `test_malformed_document_xml_returns_empty_caption` uses `caplog` — add it to the signature:
```python
    def test_malformed_document_xml_returns_empty_caption(
        self, tmp_path: Path, large_png: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd /workspace/specagent && pytest tests/unit/test_docx_image_extractor.py::TestCaptionExtraction -v
```

Expected: `FAILED` — `ExtractedImage` has no `caption` attribute yet.

---

### Task 2: Add `caption` field and `_extract_caption_map` to `docx_image_extractor.py`

**Files:**
- Modify: `src/specagent/retrieval/docx_image_extractor.py`

- [ ] **Step 1: Add new constants after existing constants (line 26)**

```python
_WORDML_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_DRAWINGML_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_DRAWING_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_DOCUMENT_XML_PATH = "word/document.xml"
```

- [ ] **Step 2: Add `caption` field to `ExtractedImage` (after `mime_type` field)**

```python
    caption: str = ""
    """Caption text from the MS Word Caption-style paragraph following this image.
    Empty string when no caption is present."""
```

- [ ] **Step 3: Add `_extract_caption_map` function (after `_parse_image_relationships`)**

```python
def _extract_caption_map(zf: zipfile.ZipFile) -> dict[str, str]:
    """Return {rel_id: caption_text} for images with a following Caption paragraph.

    Parses word/document.xml and walks body paragraphs in document order.
    For each paragraph containing a DrawingML a:blip element, looks at the
    next sibling paragraph for Caption style and extracts its text runs.

    Returns {} when word/document.xml is absent or unparseable.
    Caption extraction is best-effort and never blocks conversion.
    """
    if _DOCUMENT_XML_PATH not in zf.namelist():
        return {}
    try:
        root = ET.fromstring(zf.read(_DOCUMENT_XML_PATH))
    except Exception as exc:
        logger.warning(
            "Could not parse %s: %s — image captions unavailable",
            _DOCUMENT_XML_PATH,
            exc,
        )
        return {}

    body = root.find(f"{{{_WORDML_NS}}}body")
    if body is None:
        return {}

    caption_map: dict[str, str] = {}
    children = list(body)

    for i, elem in enumerate(children):
        if elem.tag != f"{{{_WORDML_NS}}}p":
            continue
        # Recursively find all a:blip elements in this paragraph
        blips = elem.findall(f".//{{{_DRAWINGML_NS}}}blip")
        if not blips:
            continue
        # Look at the immediate next sibling paragraph for Caption style
        for j in range(i + 1, len(children)):
            sibling = children[j]
            if sibling.tag != f"{{{_WORDML_NS}}}p":
                continue
            pstyle = sibling.find(
                f"{{{_WORDML_NS}}}pPr/{{{_WORDML_NS}}}pStyle"
            )
            if pstyle is not None and pstyle.get(f"{{{_WORDML_NS}}}val") == "Caption":
                texts = [
                    t.text or ""
                    for t in sibling.findall(f".//{{{_WORDML_NS}}}t")
                ]
                caption_text = "".join(texts).strip()
                for blip in blips:
                    rel_id = blip.get(f"{{{_DRAWING_REL_NS}}}embed")
                    if rel_id:
                        caption_map[rel_id] = caption_text
            break  # Only check the immediately following paragraph

    return caption_map
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /workspace/specagent && pytest tests/unit/test_docx_image_extractor.py::TestCaptionExtraction -v
```

Expected: All 4 tests `PASSED`.

---

### Task 3: Wire caption map through `_read_image_bytes` and `extract_images`

**Files:**
- Modify: `src/specagent/retrieval/docx_image_extractor.py`

- [ ] **Step 1: Update `_read_image_bytes` signature and body**

Replace the existing `_read_image_bytes` function (lines 120–152) with:

```python
def _read_image_bytes(
    zf: zipfile.ZipFile,
    image_rels: list[tuple[str, str]],
    caption_map: dict[str, str] | None = None,
) -> list[ExtractedImage]:
    """Build an ExtractedImage list from sorted relationship pairs and ZIP contents."""
    if caption_map is None:
        caption_map = {}
    zip_names = set(zf.namelist())
    results: list[ExtractedImage] = []

    for index, (rel_id, media_filename) in enumerate(image_rels):
        zip_entry = f"{_MEDIA_PREFIX}{media_filename}"
        if zip_entry not in zip_names:
            logger.warning(
                "Relationship references %r but entry %r is absent from ZIP — skipping",
                media_filename,
                zip_entry,
            )
            continue

        img_bytes = zf.read(zip_entry)
        mime_type, _ = mimetypes.guess_type(media_filename)
        if mime_type is None:
            mime_type = "application/octet-stream"

        results.append(
            ExtractedImage(
                placeholder_name=f"image{index}.png",
                media_filename=media_filename,
                image_bytes=img_bytes,
                mime_type=mime_type,
                caption=caption_map.get(rel_id, ""),
            )
        )

    return results
```

- [ ] **Step 2: Update `extract_images` to call `_extract_caption_map`**

Replace the `with zipfile.ZipFile(...)` block in `extract_images` (lines 67–69):

```python
        with zipfile.ZipFile(docx_path, "r") as zf:
            image_rels = _parse_image_relationships(zf)
            caption_map = _extract_caption_map(zf)
            return _read_image_bytes(zf, image_rels, caption_map)
```

- [ ] **Step 3: Run the full extractor test suite**

```bash
cd /workspace/specagent && pytest tests/unit/test_docx_image_extractor.py -v
```

Expected: All tests `PASSED` (including the pre-existing `TestExtractImages` suite — caption defaults to `""` so no breakage).

- [ ] **Step 4: Commit Phase 1a**

```bash
cd /workspace/specagent && git add src/specagent/retrieval/docx_image_extractor.py tests/conftest.py tests/unit/test_docx_image_extractor.py
git commit -m "feat(retrieval): extract MS Word caption text into ExtractedImage.caption

Parses word/document.xml to build a rel_id→caption_text map. Images
that have a following Caption-style paragraph in the docx body now
carry that text in ExtractedImage.caption (empty string otherwise).
Caption extraction is best-effort and never blocks conversion.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Update `_stitch` and `convert_docx_with_ocr` to emit caption headings

**Files:**
- Modify: `src/specagent/retrieval/docx_ocr_converter.py`
- Modify: `tests/integration/test_docx_ocr_converter.py`

- [ ] **Step 1: Write failing integration tests**

Add to `tests/integration/test_docx_ocr_converter.py` inside `TestConvertDocxWithOcr`:

```python
    async def test_caption_appears_in_stitched_output(
        self, tmp_path: Path, large_png: bytes
    ) -> None:
        """When ExtractedImage.caption is non-empty, **Figure: ...** heading is prepended."""
        from tests.conftest import make_docx_zip_with_caption
        from specagent.retrieval.docx_image_extractor import ExtractedImage
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        captioned_image = ExtractedImage(
            placeholder_name="image0.png",
            media_filename="image1.png",
            image_bytes=large_png,
            mime_type="image/png",
            caption="Figure 3: Network Architecture",
        )
        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="Before\n\n![image](image0.png)\n\nAfter",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.extract_images",
                return_value=[captioned_image],
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=_make_result("image0.png", "diagram content")),
            ),
        ):
            p = tmp_path / "captioned.docx"
            p.write_bytes(b"placeholder")
            result = await convert_docx_with_ocr(p, api_key="key")

        assert "**Figure: Figure 3: Network Architecture**" in result
        assert "diagram content" in result

    async def test_no_caption_stitches_without_label(
        self, docx_one_image: Path
    ) -> None:
        """When ExtractedImage.caption is empty, no Figure: heading is emitted."""
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=_make_result("image0.png", "content")),
            ),
        ):
            result = await convert_docx_with_ocr(docx_one_image, api_key="key")

        assert "**Figure:" not in result
        assert "content" in result
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /workspace/specagent && pytest tests/integration/test_docx_ocr_converter.py::TestConvertDocxWithOcr::test_caption_appears_in_stitched_output tests/integration/test_docx_ocr_converter.py::TestConvertDocxWithOcr::test_no_caption_stitches_without_label -v
```

Expected: `FAILED` — `_stitch` doesn't accept `captions` yet.

- [ ] **Step 3: Update `_stitch` in `docx_ocr_converter.py`**

Replace `_stitch` (lines 201–232):

```python
def _stitch(
    markdown: str,
    results: dict[int, ImageAnalysisResult],
    captions: dict[int, str] | None = None,
) -> str:
    """Replace image placeholders with analysed Markdown content.

    Matching is done by sequential counter. Optionally prepends a
    ``**Figure: <caption>**`` heading when a caption is available.

    Args:
        markdown: Raw Markdown with ``![alt](url)`` placeholders.
        results: Mapping of sequential image index → analysis result.
        captions: Optional mapping of sequential image index → caption text.

    Returns:
        Markdown with analysed placeholders replaced by their content.
    """
    if captions is None:
        captions = {}
    counter = 0

    def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
        nonlocal counter
        idx = counter
        counter += 1
        if idx in results and not results[idx].skipped:
            content = results[idx].markdown_content
            caption = captions.get(idx, "")
            if caption:
                return f"\n**Figure: {caption}**\n\n{content}"
            return content
        return match.group(0)

    return _IMAGE_PLACEHOLDER_RE.sub(_replace, markdown)
```

- [ ] **Step 4: Update `convert_docx_with_ocr` to build and pass the captions dict**

Replace the final `return _stitch(raw_markdown, results)` line (line 115) with:

```python
    captions = {i: images[i].caption for i in range(len(images)) if images[i].caption}
    return _stitch(raw_markdown, results, captions)
```

- [ ] **Step 5: Run integration tests**

```bash
cd /workspace/specagent && pytest tests/integration/test_docx_ocr_converter.py -v
```

Expected: All tests `PASSED`.

- [ ] **Step 6: Commit Phase 1b**

```bash
cd /workspace/specagent && git add src/specagent/retrieval/docx_ocr_converter.py tests/integration/test_docx_ocr_converter.py
git commit -m "feat(retrieval): prepend Figure caption heading in stitched Markdown output

Passes a captions dict from extracted image metadata into _stitch().
Images with a Word Caption-style paragraph get a **Figure: <text>**
heading prepended to their vision-analysed content.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Phase 2: Structured Mermaid Generation

---

### Task 5: Rename `call_flow_diagram` → `call_flow`, expand types, add `_MERMAID_SUBTYPE`

**Files:**
- Modify: `src/specagent/retrieval/groq_vision_client.py`
- Modify: `tests/unit/test_groq_vision_client.py`
- Modify: `tests/integration/test_docx_ocr_converter.py`

- [ ] **Step 1: Grep for all occurrences of `call_flow_diagram` in source and tests**

```bash
cd /workspace/specagent && grep -rn "call_flow_diagram" src/ tests/
```

Note every file and line. Expected locations:
- `src/specagent/retrieval/groq_vision_client.py:23` — `_KNOWN_IMAGE_TYPES`
- `tests/unit/test_groq_vision_client.py:56,66` — response mock + assertion
- `tests/integration/test_docx_ocr_converter.py:74,133,408` — result mocks

- [ ] **Step 2: Replace `_KNOWN_IMAGE_TYPES` in `groq_vision_client.py` (line 23)**

Replace:
```python
_KNOWN_IMAGE_TYPES = frozenset(["call_flow_diagram", "table", "screenshot_text", "other"])
```

With:
```python
_KNOWN_DIAGRAM_TYPES = frozenset([
    "call_flow",
    "state_machine",
    "block_diagram",
    "flowchart",
    "network_topology",
    "table",
    "screenshot_text",
    "other",
])

_MERMAID_SUBTYPE: dict[str, str] = {
    "call_flow": "sequenceDiagram",
    "state_machine": "stateDiagram-v2",
    "block_diagram": "graph LR",
    "flowchart": "flowchart TD",
    "network_topology": "graph LR",
}
```

- [ ] **Step 3: Update `_parse_response` to use `_KNOWN_DIAGRAM_TYPES`**

Replace `_KNOWN_IMAGE_TYPES` with `_KNOWN_DIAGRAM_TYPES` on line 150:
```python
        if image_type not in _KNOWN_DIAGRAM_TYPES:
```

- [ ] **Step 4: Update `ImageAnalysisResult.image_type` docstring**

Replace the docstring:
```python
    image_type: str
    """One of: ``call_flow``, ``state_machine``, ``block_diagram``, ``flowchart``,
    ``network_topology``, ``table``, ``screenshot_text``, ``other``."""
```

- [ ] **Step 5: Update all `"call_flow_diagram"` occurrences in test files**

In `tests/unit/test_groq_vision_client.py`, replace every `"call_flow_diagram"` with `"call_flow"`:
- Line 56: `httpx_mock.add_response(json=_groq_response("call_flow", mermaid_content))`
- Line 66: `assert result.image_type == "call_flow"`

In `tests/integration/test_docx_ocr_converter.py`, replace every `"call_flow_diagram"` with `"call_flow"`:
- Line 74: `_make_result("image0.png", mermaid, "call_flow")`
- Line 133: `_make_result("image0.png", "EMF diagram content", "call_flow")`
- Line 408: `_make_result("image0.png", "IANA EMF content", "call_flow")`

- [ ] **Step 6: Run unit and integration tests**

```bash
cd /workspace/specagent && pytest tests/unit/test_groq_vision_client.py tests/integration/test_docx_ocr_converter.py -v
```

Expected: All previously passing tests still `PASSED`.

- [ ] **Step 7: Commit**

```bash
cd /workspace/specagent && git add src/specagent/retrieval/groq_vision_client.py tests/unit/test_groq_vision_client.py tests/integration/test_docx_ocr_converter.py
git commit -m "feat(vision): expand diagram taxonomy and rename call_flow_diagram to call_flow

Adds state_machine, block_diagram, flowchart, network_topology to the
known diagram types. Adds _MERMAID_SUBTYPE mapping for each diagram
type to its canonical Mermaid header keyword.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Create `_vision_prompts.py` with structured prompt and JSON schema

**Files:**
- Create: `src/specagent/retrieval/_vision_prompts.py`

- [ ] **Step 1: Create `src/specagent/retrieval/_vision_prompts.py`**

```python
"""Prompt constants and JSON schema for Groq vision API calls.

Extracted here to keep groq_vision_client.py under the 300-line limit.
"""

from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a technical diagram analyst for 3GPP telecommunications specifications.\n"
    "Analyze images and respond with JSON only — no markdown wrapper, no explanation.\n\n"
    "## Classification Types\n\n"
    "Classify each image as exactly one of:\n"
    "- call_flow: Sequence diagram or call flow showing numbered message exchanges "
    "between network entities (UE, gNB, AMF, SMF, etc.).\n"
    "- state_machine: State diagram showing states, transitions, and guard conditions.\n"
    "- block_diagram: Block/component architecture diagram showing modules and connections.\n"
    "- flowchart: Process flow diagram with decision nodes and process steps.\n"
    "- network_topology: Network diagram showing physical or logical network layout.\n"
    "- table: Table of data, parameters, or values.\n"
    "- screenshot_text: Screenshot or image containing readable text.\n"
    "- other: Anything that does not fit the above categories.\n\n"
    "## Response Format\n\n"
    'Respond as: {"type": "<type>", "content": "<content>", "prose_fallback": "<one sentence>"}\n\n'
    "prose_fallback is ALWAYS required: one plain-English sentence describing what the image shows.\n\n"
    "## Content Format by Type\n\n"
    "call_flow      → ```mermaid\\nsequenceDiagram\\n  <entities and messages>\\n```\n"
    "state_machine  → ```mermaid\\nstateDiagram-v2\\n  <states and transitions>\\n```\n"
    "block_diagram  → ```mermaid\\ngraph LR\\n  <components and edges>\\n```\n"
    "flowchart      → ```mermaid\\nflowchart TD\\n  <nodes and edges>\\n```\n"
    "network_topology → ```mermaid\\ngraph LR\\n  <network nodes and links>\\n```\n"
    "table          → Markdown table (| Col | Col |\\n|---|---|\\n| val | val |)\n"
    "screenshot_text → Extracted text as Markdown.\n"
    "other          → One-sentence plain-English description.\n\n"
    "## Examples\n\n"
    'call_flow: {"type": "call_flow", '
    '"content": "```mermaid\\nsequenceDiagram\\n  UE->>gNB: RRC Setup Request\\n'
    '  gNB-->>UE: RRC Setup\\n  UE->>gNB: RRC Setup Complete\\n```", '
    '"prose_fallback": "RRC connection setup procedure between UE and gNB."}\n\n'
    'state_machine: {"type": "state_machine", '
    '"content": "```mermaid\\nstateDiagram-v2\\n  [*] --> Idle\\n'
    '  Idle --> Connected: RRC Setup\\n  Connected --> Idle: RRC Release\\n```", '
    '"prose_fallback": "UE RRC state machine showing Idle and Connected states."}\n\n'
    'block_diagram: {"type": "block_diagram", '
    '"content": "```mermaid\\ngraph LR\\n  UE[UE] --> gNB[gNB]\\n'
    '  gNB --> AMF[AMF]\\n  gNB --> UPF[UPF]\\n```", '
    '"prose_fallback": "5G network architecture block diagram showing UE, gNB, AMF, UPF."}\n\n'
    'flowchart: {"type": "flowchart", '
    '"content": "```mermaid\\nflowchart TD\\n  A[Start] --> B{Condition?}\\n'
    '  B -->|Yes| C[Action]\\n  B -->|No| D[End]\\n```", '
    '"prose_fallback": "Process flowchart with a conditional decision branch."}\n\n'
    'network_topology: {"type": "network_topology", '
    '"content": "```mermaid\\ngraph LR\\n  RAN[RAN] --> CN[5G Core]\\n'
    '  CN --> Internet[Internet]\\n```", '
    '"prose_fallback": "Network topology showing RAN connected to 5G Core and Internet."}'
)

_USER_MESSAGE_TEXT = (
    "Analyze this image and return the JSON response as specified in the system prompt."
)

_RESPONSE_JSON_SCHEMA: dict = {
    "name": "image_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": [
                    "call_flow",
                    "state_machine",
                    "block_diagram",
                    "flowchart",
                    "network_topology",
                    "table",
                    "screenshot_text",
                    "other",
                ],
            },
            "content": {"type": "string"},
            "prose_fallback": {"type": "string"},
        },
        "required": ["type", "content", "prose_fallback"],
        "additionalProperties": False,
    },
}
```

- [ ] **Step 2: Verify the file is importable**

```bash
cd /workspace/specagent && python -c "from specagent.retrieval._vision_prompts import _SYSTEM_PROMPT, _RESPONSE_JSON_SCHEMA; print('OK', len(_SYSTEM_PROMPT))"
```

Expected: `OK` followed by a number > 500.

---

### Task 7: Add `prose_fallback` field, restructure `analyze_image`, update `_parse_response`

**Files:**
- Modify: `src/specagent/retrieval/groq_vision_client.py`
- Modify: `tests/unit/test_groq_vision_client.py`

- [ ] **Step 1: Write failing tests for new behavior**

Add to `tests/unit/test_groq_vision_client.py` inside `TestAnalyzeImage`:

```python
    async def test_api_request_uses_system_message(self, httpx_mock) -> None:
        """analyze_image sends a system role message as the first messages entry."""
        import json as _json

        captured_body: list[dict] = []

        def capture(request):
            captured_body.append(_json.loads(request.content))
            return httpx_mock.add_response(
                json=_groq_response("other", "A diagram.", "A diagram.")
            )

        httpx_mock.add_callback(capture)

        from specagent.retrieval.groq_vision_client import analyze_image

        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=AsyncMock()),
        ):
            await analyze_image(_make_image(), api_key="test-key")

        assert captured_body[0]["messages"][0]["role"] == "system"

    async def test_api_request_includes_response_format(self, httpx_mock) -> None:
        """analyze_image includes response_format json_schema in request body."""
        import json as _json

        captured_body: list[dict] = []

        def capture(request):
            captured_body.append(_json.loads(request.content))
            return httpx_mock.add_response(
                json=_groq_response("other", "A diagram.", "A diagram.")
            )

        httpx_mock.add_callback(capture)

        from specagent.retrieval.groq_vision_client import analyze_image

        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=AsyncMock()),
        ):
            await analyze_image(_make_image(), api_key="test-key")

        assert captured_body[0]["response_format"]["type"] == "json_schema"

    async def test_parse_response_populates_prose_fallback(self) -> None:
        """prose_fallback field is extracted from JSON response."""
        from specagent.retrieval.groq_vision_client import _parse_response

        raw = json.dumps({
            "type": "call_flow",
            "content": "```mermaid\nsequenceDiagram\n  A->>B: msg\n```",
            "prose_fallback": "A call flow between A and B.",
        })
        result = _parse_response("image0.png", raw)
        assert result.prose_fallback == "A call flow between A and B."

    async def test_parse_response_prose_fallback_defaults_to_empty(self) -> None:
        """prose_fallback is empty string when key absent from JSON."""
        from specagent.retrieval.groq_vision_client import _parse_response

        raw = json.dumps({"type": "other", "content": "A logo."})
        result = _parse_response("image0.png", raw)
        assert result.prose_fallback == ""

    async def test_parse_response_state_machine_returns_statediagram(self) -> None:
        """state_machine type is recognised and returned as-is."""
        from specagent.retrieval.groq_vision_client import _parse_response

        content = "```mermaid\nstateDiagram-v2\n  [*] --> Idle\n```"
        raw = json.dumps({"type": "state_machine", "content": content, "prose_fallback": "A state machine."})
        result = _parse_response("image0.png", raw)
        assert result.image_type == "state_machine"
        assert "stateDiagram-v2" in result.markdown_content

    async def test_parse_response_unknown_type_falls_back_to_other(self) -> None:
        """Unrecognised type value is normalised to 'other'."""
        from specagent.retrieval.groq_vision_client import _parse_response

        raw = json.dumps({"type": "banana", "content": "weird", "prose_fallback": ""})
        result = _parse_response("image0.png", raw)
        assert result.image_type == "other"
```

Also update the existing `_groq_response` helper to include `prose_fallback`:

```python
def _groq_response(image_type: str, content: str, prose_fallback: str = "") -> dict:
    """Build a minimal Groq chat completion response dict."""
    payload = {"type": image_type, "content": content, "prose_fallback": prose_fallback}
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload),
                }
            }
        ]
    }
```

Update all existing `_groq_response(...)` calls to pass `prose_fallback` (add `""` as third arg to all that don't have it).

- [ ] **Step 2: Run to confirm new tests fail**

```bash
cd /workspace/specagent && pytest tests/unit/test_groq_vision_client.py::TestAnalyzeImage::test_api_request_uses_system_message tests/unit/test_groq_vision_client.py::TestAnalyzeImage::test_parse_response_populates_prose_fallback -v
```

Expected: `FAILED`.

- [ ] **Step 3: Add `prose_fallback` field to `ImageAnalysisResult` in `groq_vision_client.py`**

After `skip_reason` field (line 58), add:

```python
    prose_fallback: str = ""
    """One-sentence plain-English description of the image, always populated
    for diagram types. Used as fallback when Mermaid validation fails."""
```

- [ ] **Step 4: Add imports for prompt constants at top of `groq_vision_client.py`**

After the existing imports, add:

```python
from specagent.retrieval._vision_prompts import (
    _RESPONSE_JSON_SCHEMA,
    _SYSTEM_PROMPT,
    _USER_MESSAGE_TEXT,
)
```

Remove the old `_VISION_PROMPT` constant (line 25–39).

- [ ] **Step 5: Restructure the API request body in `_call()` inside `analyze_image`**

Replace the `json={...}` dict in `_call()`:

```python
            body: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _USER_MESSAGE_TEXT},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": _RESPONSE_JSON_SCHEMA,
                },
                "max_tokens": 1024,
                "temperature": 0.0,
            }
            _headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(_GROQ_CHAT_URL, headers=_headers, json=body)
                # Fallback: if Groq rejects response_format, retry without it
                if (
                    response.status_code == 400
                    and "response_format" in response.text
                ):
                    logger.warning(
                        "response_format not supported by model; retrying without it"
                    )
                    body = {k: v for k, v in body.items() if k != "response_format"}
                    response = await client.post(
                        _GROQ_CHAT_URL, headers=_headers, json=body
                    )
                response.raise_for_status()

            raw_content = response.json()["choices"][0]["message"]["content"]
            return _parse_response(image.placeholder_name, raw_content)
```

- [ ] **Step 6: Update `_parse_response` to handle expanded types and `prose_fallback`**

Replace the entire `_parse_response` function:

```python
def _parse_response(placeholder_name: str, raw_content: str) -> ImageAnalysisResult:
    """Parse the model's text response into an ImageAnalysisResult.

    Falls back to ``image_type='other'`` with the raw text as content if JSON
    parsing fails or the type field is unrecognised.
    """
    try:
        data = json.loads(raw_content)
        image_type = data.get("type", "other")
        if image_type not in _KNOWN_DIAGRAM_TYPES:
            image_type = "other"
        content = str(data.get("content", raw_content))
        prose_fallback = str(data.get("prose_fallback", ""))

        # For diagram types: ensure the Mermaid block uses the correct header keyword
        if image_type in _MERMAID_SUBTYPE:
            content = _fix_mermaid_header(content, _MERMAID_SUBTYPE[image_type])

    except (json.JSONDecodeError, AttributeError):
        logger.warning(
            "Non-JSON response from vision API for %r; treating as 'other'",
            placeholder_name,
        )
        image_type = "other"
        content = raw_content
        prose_fallback = ""

    return ImageAnalysisResult(
        placeholder_name=placeholder_name,
        markdown_content=content,
        image_type=image_type,
        prose_fallback=prose_fallback,
    )


def _fix_mermaid_header(content: str, expected_header: str) -> str:
    """Ensure a Mermaid fenced block starts with the expected diagram header keyword.

    If content is already correct, returns it unchanged.
    If the first non-empty line inside the fence is wrong, replaces it.
    """
    fence_match = re.match(r"```mermaid\n([\s\S]*?)```\s*$", content.strip())
    if not fence_match:
        return content  # Not a fenced block — leave for validator to catch
    inner = fence_match.group(1)
    lines = inner.split("\n")
    expected_keyword = expected_header.split()[0]  # e.g. "graph" from "graph LR"
    for i, line in enumerate(lines):
        if line.strip():
            if not line.strip().startswith(expected_keyword):
                lines[i] = expected_header
                return f"```mermaid\n{chr(10).join(lines)}```"
            break
    return content
```

- [ ] **Step 7: Run Phase 2 tests**

```bash
cd /workspace/specagent && pytest tests/unit/test_groq_vision_client.py -v
```

Expected: All tests `PASSED`.

- [ ] **Step 8: Commit**

```bash
cd /workspace/specagent && git add src/specagent/retrieval/groq_vision_client.py src/specagent/retrieval/_vision_prompts.py tests/unit/test_groq_vision_client.py
git commit -m "feat(vision): structured Groq prompt with JSON schema enforcement

Replaces free-form _VISION_PROMPT with a system-message-based
structured prompt (in _vision_prompts.py) that uses Groq
response_format json_schema enforcement. Expands classification to
5 diagram types with per-type Mermaid subtype headers. Adds
prose_fallback field to ImageAnalysisResult. Falls back to
unstructured parsing on HTTP 400 response_format rejection.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Add Phase 2 config field

**Files:**
- Modify: `src/specagent/config.py`

- [ ] **Step 1: Add `vision_diagram_types` field after `vision_max_retries` (line 374)**

```python
    vision_diagram_types: list[str] = Field(
        default=[
            "call_flow",
            "state_machine",
            "block_diagram",
            "flowchart",
            "network_topology",
        ],
        description=(
            "Diagram types for which Mermaid output is requested from the vision model. "
            "Env var: VISION_DIAGRAM_TYPES (comma-separated)."
        ),
    )
```

- [ ] **Step 2: Run config tests to confirm no breakage**

```bash
cd /workspace/specagent && pytest tests/unit/test_config.py -v
```

Expected: All `PASSED`.

- [ ] **Step 3: Commit**

```bash
cd /workspace/specagent && git add src/specagent/config.py
git commit -m "feat(config): add vision_diagram_types config field

Allows operators to restrict which diagram types trigger Mermaid
generation without code changes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Phase 3: Mermaid Validation + Correction Loop

---

### Task 9: Create `mermaid_validator.py` with unit tests

**Files:**
- Create: `src/specagent/retrieval/mermaid_validator.py`
- Create: `tests/unit/test_mermaid_validator.py`

- [ ] **Step 1: Write failing unit tests in `tests/unit/test_mermaid_validator.py`**

```python
"""Unit tests for mermaid_validator."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestValidateMermaid:
    """Tests for validate_mermaid()."""

    def test_valid_sequence_diagram_passes(self) -> None:
        """A well-formed sequenceDiagram block returns (True, '')."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nsequenceDiagram\n  UE->>gNB: msg\n  gNB-->>UE: ack\n```"
        valid, reason = validate_mermaid(content)
        assert valid is True
        assert reason == ""

    def test_valid_state_diagram_passes(self) -> None:
        """A well-formed stateDiagram-v2 block returns (True, '')."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nstateDiagram-v2\n  [*] --> Idle\n  Idle --> Active: start\n```"
        valid, reason = validate_mermaid(content)
        assert valid is True
        assert reason == ""

    def test_valid_flowchart_passes(self) -> None:
        """A well-formed flowchart TD block returns (True, '')."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nflowchart TD\n  A[Start] --> B[End]\n```"
        valid, reason = validate_mermaid(content)
        assert valid is True
        assert reason == ""

    def test_missing_mermaid_fence_fails(self) -> None:
        """Raw content without ```mermaid fence returns (False, reason)."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        valid, reason = validate_mermaid("sequenceDiagram\n  A->>B: msg\n")
        assert valid is False
        assert "fenced" in reason.lower() or "mermaid" in reason.lower()

    def test_unknown_header_fails(self) -> None:
        """Unrecognised diagram type keyword returns (False, reason)."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nunknownDiagramType\n  A --> B\n```"
        valid, reason = validate_mermaid(content)
        assert valid is False
        assert "unknownDiagramType" in reason or "Unknown" in reason

    def test_empty_content_fails(self) -> None:
        """A block with only the header line and no content returns (False, reason)."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nsequenceDiagram\n```"
        valid, reason = validate_mermaid(content)
        assert valid is False
        assert "content" in reason.lower() or "header" in reason.lower()

    def test_comment_lines_not_counted_as_content(self) -> None:
        """Lines starting with %% are comments and don't count as content."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nsequenceDiagram\n  %% This is a comment\n```"
        valid, reason = validate_mermaid(content)
        assert valid is False

    def test_unbalanced_brackets_fails(self) -> None:
        """A block with unbalanced [ returns (False, reason)."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\ngraph TD\n  A[Node --> B[Another\n```"
        valid, reason = validate_mermaid(content)
        assert valid is False
        assert "bracket" in reason.lower() or "unbalanced" in reason.lower()

    def test_mmdc_not_called_when_disabled(self, monkeypatch) -> None:
        """subprocess.run is never called when mermaid_validate_with_mmdc is False."""
        import subprocess
        from unittest.mock import patch

        content = "```mermaid\nsequenceDiagram\n  A->>B: msg\n  B-->>A: ack\n```"
        with patch("subprocess.run") as mock_run:
            monkeypatch.setattr(
                "specagent.retrieval.mermaid_validator.settings",
                type("S", (), {"mermaid_validate_with_mmdc": False, "mermaid_mmdc_timeout": 10})(),
            )
            from specagent.retrieval.mermaid_validator import validate_mermaid
            validate_mermaid(content)
        mock_run.assert_not_called()

    def test_valid_graph_lr_passes(self) -> None:
        """graph LR with two nodes passes validation."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\ngraph LR\n  A[UE] --> B[gNB]\n  B --> C[AMF]\n```"
        valid, reason = validate_mermaid(content)
        assert valid is True
        assert reason == ""
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /workspace/specagent && pytest tests/unit/test_mermaid_validator.py -v
```

Expected: `ERROR` — module not found.

- [ ] **Step 3: Create `src/specagent/retrieval/mermaid_validator.py`**

```python
"""Mermaid diagram structural validator.

Tier 1 (always): Pure Python regex checks for fenced block, header,
content, and bracket balance.

Tier 2 (opt-in): mmdc subprocess validation when
settings.mermaid_validate_with_mmdc is True.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from specagent.config import settings

logger = logging.getLogger(__name__)

_FENCED_MERMAID_RE = re.compile(r"```mermaid\n([\s\S]*?)```", re.MULTILINE)

_VALID_DIAGRAM_HEADERS = frozenset([
    "sequenceDiagram",
    "stateDiagram-v2",
    "graph",
    "flowchart",
    "classDiagram",
    "erDiagram",
    "gantt",
    "pie",
    "gitGraph",
    "mindmap",
    "timeline",
    "xychart-beta",
])


def validate_mermaid(content: str) -> tuple[bool, str]:
    """Validate a fenced Mermaid code block.

    Runs Tier 1 structural checks unconditionally. Optionally runs
    Tier 2 mmdc subprocess validation when settings.mermaid_validate_with_mmdc
    is True and mmdc is available on PATH.

    Args:
        content: A string containing a ```mermaid ... ``` fenced code block.

    Returns:
        (True, "") if valid.
        (False, reason) if invalid, where reason describes the failure.
    """
    inner = _extract_inner(content)
    if inner is None:
        return False, "Content does not contain a ```mermaid ... ``` fenced block."

    if not _check_header(inner):
        first = next((l.strip() for l in inner.split("\n") if l.strip()), "")
        return False, f"Unknown Mermaid diagram type on first line: {first!r}."

    if not _check_has_content(inner):
        return False, "Diagram has no content lines beyond the header (ignoring %% comments)."

    if not _check_bracket_balance(inner):
        return False, "Unbalanced brackets, parentheses, or braces in diagram."

    if settings.mermaid_validate_with_mmdc:
        return _check_with_mmdc(inner)

    return True, ""


def _extract_inner(content: str) -> str | None:
    """Return the body between ```mermaid and ``` fences, or None."""
    match = _FENCED_MERMAID_RE.search(content)
    return match.group(1) if match else None


def _check_header(inner: str) -> bool:
    """Return True if first non-empty line starts with a known Mermaid diagram keyword."""
    for line in inner.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        return any(
            stripped == kw or stripped.startswith(kw + " ") or stripped.startswith(kw + "\t")
            for kw in _VALID_DIAGRAM_HEADERS
        )
    return False


def _check_has_content(inner: str) -> bool:
    """Return True if at least 2 non-empty, non-comment lines exist."""
    count = 0
    for line in inner.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("%%"):
            count += 1
        if count >= 2:
            return True
    return False


def _check_bracket_balance(inner: str) -> bool:
    """Return True if brackets, parentheses, and braces balance."""
    pairs = {"[": "]", "(": ")", "{": "}"}
    counts: dict[str, int] = {c: 0 for c in "[](){}"}
    in_string = False
    for char in inner:
        if char == '"':
            in_string = not in_string
        if not in_string and char in counts:
            counts[char] += 1
    return (
        counts["["] == counts["]"]
        and counts["("] == counts[")"]
        and counts["{"] == counts["}"]
    )


def _check_with_mmdc(inner: str) -> tuple[bool, str]:
    """Validate using the mermaid-cli mmdc subprocess.

    Returns (True, "") when mmdc is absent (FileNotFoundError) or times out —
    missing tooling must not fail validation.
    """
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mmd", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(inner)
            tmp_path = Path(tmp.name)
        result = subprocess.run(
            ["mmdc", "-i", str(tmp_path), "-o", "/dev/null"],
            capture_output=True,
            text=True,
            timeout=settings.mermaid_mmdc_timeout,
        )
        tmp_path.unlink(missing_ok=True)
        if result.returncode != 0:
            return False, result.stderr.strip() or "mmdc validation failed."
        return True, ""
    except FileNotFoundError:
        logger.debug("mmdc not found on PATH — skipping Tier 2 Mermaid validation.")
        return True, ""
    except subprocess.TimeoutExpired:
        logger.warning(
            "mmdc validation timed out after %ds — treating as valid.",
            settings.mermaid_mmdc_timeout,
        )
        return True, ""
```

- [ ] **Step 4: Run validator tests**

```bash
cd /workspace/specagent && pytest tests/unit/test_mermaid_validator.py -v
```

Expected: All 10 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
cd /workspace/specagent && git add src/specagent/retrieval/mermaid_validator.py tests/unit/test_mermaid_validator.py
git commit -m "feat(retrieval): add two-tier Mermaid structural validator

Tier 1: pure Python checks (fenced block, known header, content lines,
bracket balance). Tier 2: optional mmdc subprocess, off by default.
FileNotFoundError and timeout both pass through silently.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 10: Add `correct_mermaid_diagram` to `groq_vision_client.py`

**Files:**
- Modify: `src/specagent/retrieval/groq_vision_client.py`
- Modify: `tests/unit/test_groq_vision_client.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_groq_vision_client.py` as a new class:

```python
@pytest.mark.unit
class TestCorrectMermaidDiagram:
    """Tests for correct_mermaid_diagram()."""

    async def test_correction_sends_system_message(self, httpx_mock) -> None:
        """correct_mermaid_diagram sends a system role message."""
        import json as _json

        captured: list[dict] = []

        def capture(request):
            captured.append(_json.loads(request.content))
            return httpx_mock.add_response(
                json=_groq_response(
                    "call_flow",
                    "```mermaid\nsequenceDiagram\n  A->>B: msg\n  B-->>A: ack\n```",
                    "A corrected call flow.",
                )
            )

        httpx_mock.add_callback(capture)

        from specagent.retrieval.groq_vision_client import correct_mermaid_diagram

        image = _make_image()
        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=AsyncMock()),
        ):
            await correct_mermaid_diagram(
                image=image,
                prior_attempt="```mermaid\nbadContent\n```",
                validation_errors="Unknown diagram type.",
                diagram_type="call_flow",
                api_key="test-key",
            )

        assert captured[0]["messages"][0]["role"] == "system"

    async def test_correction_locks_diagram_type(self, httpx_mock) -> None:
        """correct_mermaid_diagram does not reclassify — diagram_type is locked."""
        # Even if model returns a different type, the locked type is used
        httpx_mock.add_response(
            json=_groq_response(
                "other",  # model says "other"
                "```mermaid\nsequenceDiagram\n  A->>B: msg\n  B-->>A: ack\n```",
                "A call flow.",
            )
        )

        from specagent.retrieval.groq_vision_client import correct_mermaid_diagram

        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=AsyncMock()),
        ):
            result = await correct_mermaid_diagram(
                image=_make_image(),
                prior_attempt="bad mermaid",
                validation_errors="Missing fence.",
                diagram_type="call_flow",
                api_key="test-key",
            )

        # Locked to call_flow regardless of what model returned
        assert result.image_type == "call_flow"

    async def test_correction_raises_configuration_error_for_empty_key(self) -> None:
        """correct_mermaid_diagram raises ConfigurationError for empty api_key."""
        from specagent.retrieval.groq_vision_client import correct_mermaid_diagram

        with pytest.raises(ConfigurationError, match="api_key"):
            await correct_mermaid_diagram(
                image=_make_image(),
                prior_attempt="bad",
                validation_errors="error",
                diagram_type="call_flow",
                api_key="",
            )

    async def test_correction_includes_prior_attempt_in_user_message(
        self, httpx_mock
    ) -> None:
        """The user message contains the prior_attempt text."""
        import json as _json

        captured: list[dict] = []

        def capture(request):
            captured.append(_json.loads(request.content))
            return httpx_mock.add_response(
                json=_groq_response(
                    "call_flow",
                    "```mermaid\nsequenceDiagram\n  A->>B: ok\n  B-->>A: done\n```",
                    "A fixed call flow.",
                )
            )

        httpx_mock.add_callback(capture)

        from specagent.retrieval.groq_vision_client import correct_mermaid_diagram

        with patch(
            "specagent.retrieval.groq_vision_client._get_rate_limiter",
            return_value=MagicMock(acquire=AsyncMock()),
        ):
            await correct_mermaid_diagram(
                image=_make_image(),
                prior_attempt="```mermaid\nBAD_DIAGRAM\n```",
                validation_errors="Unknown type.",
                diagram_type="call_flow",
                api_key="test-key",
            )

        user_content = captured[0]["messages"][1]["content"]
        user_text = next(
            (c["text"] for c in user_content if c.get("type") == "text"), ""
        )
        assert "BAD_DIAGRAM" in user_text
        assert "Unknown type." in user_text
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /workspace/specagent && pytest tests/unit/test_groq_vision_client.py::TestCorrectMermaidDiagram -v
```

Expected: `ERROR` — `correct_mermaid_diagram` not defined.

- [ ] **Step 3: Add `correct_mermaid_diagram` to `groq_vision_client.py`**

Add after `analyze_image` and before `_parse_response`:

```python
async def correct_mermaid_diagram(
    image: "ExtractedImage",
    prior_attempt: str,
    validation_errors: str,
    diagram_type: str,
    api_key: str,
    model: str = _DEFAULT_MODEL,
) -> ImageAnalysisResult:
    """Re-submit an image with validation errors to obtain a corrected Mermaid diagram.

    Sends a two-message conversation: system prompt + user message containing
    the original image, the prior failed attempt, and the validation errors.
    The diagram_type is locked — the model cannot reclassify the image.

    Args:
        image: The original ExtractedImage (re-sent for visual context).
        prior_attempt: The Mermaid block that failed validation.
        validation_errors: Human-readable description of the failures.
        diagram_type: Locked image_type from the first analysis attempt.
        api_key: Groq API key. Never logged.
        model: Groq vision model identifier.

    Returns:
        ImageAnalysisResult with image_type locked to diagram_type.

    Raises:
        ConfigurationError: If api_key is empty.
        VisionError: If the API call fails after retries.
    """
    if not api_key:
        raise ConfigurationError(
            "api_key must be non-empty to call the Groq vision API."
        )

    await _get_rate_limiter().acquire()

    encoded = base64.b64encode(image.image_bytes).decode("ascii")
    data_url = f"data:{image.mime_type};base64,{encoded}"

    correction_text = (
        f"This image was previously classified as '{diagram_type}'. "
        f"A Mermaid diagram was generated but failed validation with these errors:\n\n"
        f"{validation_errors}\n\n"
        f"Previous attempt:\n{prior_attempt}\n\n"
        f"Re-analyze the image and return a corrected Mermaid diagram of type "
        f"'{diagram_type}'. Return the same JSON schema as before."
    )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    async def _call() -> ImageAnalysisResult:
        _headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": correction_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": _RESPONSE_JSON_SCHEMA,
            },
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(_GROQ_CHAT_URL, headers=_headers, json=body)
            if response.status_code == 400 and "response_format" in response.text:
                body = {k: v for k, v in body.items() if k != "response_format"}
                response = await client.post(_GROQ_CHAT_URL, headers=_headers, json=body)
            response.raise_for_status()

        raw_content = response.json()["choices"][0]["message"]["content"]
        result = _parse_response(image.placeholder_name, raw_content)
        # Lock diagram_type — correction cannot reclassify
        return result.model_copy(update={"image_type": diagram_type})

    try:
        return await _call()
    except (ConfigurationError, VisionError):
        raise
    except Exception as exc:
        raise VisionError(
            f"Correction API failed for {image.placeholder_name!r} after retries: {exc}"
        ) from exc
```

- [ ] **Step 4: Run correction tests**

```bash
cd /workspace/specagent && pytest tests/unit/test_groq_vision_client.py::TestCorrectMermaidDiagram -v
```

Expected: All 4 tests `PASSED`.

- [ ] **Step 5: Run full vision client test suite**

```bash
cd /workspace/specagent && pytest tests/unit/test_groq_vision_client.py -v
```

Expected: All tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
cd /workspace/specagent && git add src/specagent/retrieval/groq_vision_client.py tests/unit/test_groq_vision_client.py
git commit -m "feat(vision): add correct_mermaid_diagram for validation-triggered correction

Sends the original image plus prior Mermaid attempt and validation
errors back to Groq VLM. Diagram type is locked to prevent
reclassification. Uses the same JSON schema contract as analyze_image.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 11: Add `_apply_mermaid_validation` and wire it into `convert_docx_with_ocr`

**Files:**
- Modify: `src/specagent/retrieval/docx_ocr_converter.py`
- Modify: `tests/integration/test_docx_ocr_converter.py`

- [ ] **Step 1: Write failing integration tests**

Add to `tests/integration/test_docx_ocr_converter.py` inside `TestConvertDocxWithOcr`:

```python
    async def test_invalid_mermaid_triggers_correction_call(
        self, docx_one_image: Path
    ) -> None:
        """When analyze_image returns invalid Mermaid, correct_mermaid_diagram is called once."""
        from unittest.mock import AsyncMock, patch
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        bad_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="```mermaid\nBADCONTENT\n```",
            image_type="call_flow",
            prose_fallback="A call flow diagram.",
        )
        good_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="```mermaid\nsequenceDiagram\n  A->>B: msg\n  B-->>A: ack\n```",
            image_type="call_flow",
            prose_fallback="A call flow diagram.",
        )
        correction_mock = AsyncMock(return_value=good_result)

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=bad_result),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                correction_mock,
            ),
        ):
            result = await convert_docx_with_ocr(docx_one_image, api_key="key")

        correction_mock.assert_called_once()
        assert "sequenceDiagram" in result

    async def test_valid_mermaid_skips_correction(
        self, docx_one_image: Path
    ) -> None:
        """When first Mermaid result is valid, correct_mermaid_diagram is never called."""
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        good_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content=(
                "```mermaid\nsequenceDiagram\n  UE->>gNB: attach\n  gNB-->>UE: ok\n```"
            ),
            image_type="call_flow",
            prose_fallback="Attach procedure.",
        )
        correction_mock = AsyncMock()

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=good_result),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                correction_mock,
            ),
        ):
            await convert_docx_with_ocr(docx_one_image, api_key="key")

        correction_mock.assert_not_called()

    async def test_correction_failure_falls_back_to_prose(
        self, docx_one_image: Path
    ) -> None:
        """When corrected Mermaid is still invalid, prose_fallback is used."""
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        bad_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="```mermaid\nBADCONTENT\n```",
            image_type="call_flow",
            prose_fallback="A network call flow showing UE and gNB.",
        )
        still_bad = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="```mermaid\nSTILLBAD\n```",
            image_type="call_flow",
            prose_fallback="A network call flow showing UE and gNB.",
        )

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=bad_result),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                AsyncMock(return_value=still_bad),
            ),
        ):
            result = await convert_docx_with_ocr(docx_one_image, api_key="key")

        assert "A network call flow showing UE and gNB." in result
        assert "STILLBAD" not in result

    async def test_non_diagram_type_skips_validation(
        self, docx_one_image: Path
    ) -> None:
        """table and screenshot_text results pass through without validation."""
        from specagent.retrieval.groq_vision_client import ImageAnalysisResult
        from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

        table_result = ImageAnalysisResult(
            placeholder_name="image0.png",
            markdown_content="| Col A | Col B |\n|---|---|\n| 1 | 2 |",
            image_type="table",
            prose_fallback="A parameter table.",
        )
        correction_mock = AsyncMock()

        with (
            patch(
                "specagent.retrieval.docx_ocr_converter.convert",
                return_value="![image](image0.png)",
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.analyze_image",
                AsyncMock(return_value=table_result),
            ),
            patch(
                "specagent.retrieval.docx_ocr_converter.correct_mermaid_diagram",
                correction_mock,
            ),
        ):
            result = await convert_docx_with_ocr(docx_one_image, api_key="key")

        correction_mock.assert_not_called()
        assert "| Col A |" in result
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /workspace/specagent && pytest tests/integration/test_docx_ocr_converter.py::TestConvertDocxWithOcr::test_invalid_mermaid_triggers_correction_call tests/integration/test_docx_ocr_converter.py::TestConvertDocxWithOcr::test_valid_mermaid_skips_correction -v
```

Expected: `FAILED` — `_apply_mermaid_validation` not wired yet.

- [ ] **Step 3: Add imports and constants to `docx_ocr_converter.py`**

Update the imports block at the top of `docx_ocr_converter.py`:

```python
from specagent.retrieval.groq_vision_client import (
    ImageAnalysisResult,
    analyze_image,
    correct_mermaid_diagram,
)
from specagent.retrieval.mermaid_validator import validate_mermaid
```

Add constant after `_VISION_SUPPORTED_MIME_TYPES`:

```python
_DIAGRAM_TYPES_REQUIRING_VALIDATION = frozenset([
    "call_flow",
    "state_machine",
    "block_diagram",
    "flowchart",
    "network_topology",
])
```

- [ ] **Step 4: Add `_apply_mermaid_validation` function to `docx_ocr_converter.py`**

Add before `_prepare_image`:

```python
async def _apply_mermaid_validation(
    result: ImageAnalysisResult,
    image: "ExtractedImage",
    api_key: str,
) -> ImageAnalysisResult:
    """Validate Mermaid content and request a one-shot correction on failure.

    Only applies to diagram types in _DIAGRAM_TYPES_REQUIRING_VALIDATION.
    Non-diagram types (table, screenshot_text, other) pass through unchanged.

    On first-attempt validation failure: calls correct_mermaid_diagram once.
    On second-attempt validation failure: uses prose_fallback or a marker string.

    Args:
        result: The ImageAnalysisResult from analyze_image.
        image: The original ExtractedImage (passed to correction call).
        api_key: Groq API key.

    Returns:
        The original result, a corrected result, or a prose-fallback result.
    """
    if result.image_type not in _DIAGRAM_TYPES_REQUIRING_VALIDATION:
        return result

    valid, reason = validate_mermaid(result.markdown_content)
    if valid:
        return result

    logger.warning(
        "Mermaid validation failed for %s: %s — requesting correction from VLM",
        image.placeholder_name,
        reason,
    )

    try:
        corrected = await correct_mermaid_diagram(
            image=image,
            prior_attempt=result.markdown_content,
            validation_errors=reason,
            diagram_type=result.image_type,
            api_key=api_key,
        )
    except VisionError as exc:
        logger.warning(
            "Correction API call failed for %s: %s — using prose fallback",
            image.placeholder_name,
            exc,
        )
        fallback = (
            result.prose_fallback
            or f"_[Diagram: {image.placeholder_name} — validation failed]_"
        )
        return result.model_copy(update={"markdown_content": fallback})

    valid2, reason2 = validate_mermaid(corrected.markdown_content)
    if valid2:
        return corrected

    logger.warning(
        "Corrected Mermaid still invalid for %s: %s — using prose fallback",
        image.placeholder_name,
        reason2,
    )
    fallback = (
        result.prose_fallback
        or f"_[Diagram: {image.placeholder_name} — validation failed]_"
    )
    return result.model_copy(update={"markdown_content": fallback})
```

- [ ] **Step 5: Wire `_apply_mermaid_validation` into `convert_docx_with_ocr`**

Replace the `try:` block in the image analysis loop (lines 96–113):

```python
        try:
            result = await analyze_image(image, api_key=api_key)
            result = await _apply_mermaid_validation(result, image, api_key)
            results[idx] = result
            logger.info(
                "Analysed %s (index %d) in %s: type=%s",
                image.placeholder_name,
                idx,
                docx_path.name,
                result.image_type,
            )
        except VisionError as exc:
            logger.warning(
                "Vision analysis failed for %s (index %d) in %s: %s — keeping placeholder",
                image.placeholder_name,
                idx,
                docx_path.name,
                exc,
            )
```

- [ ] **Step 6: Run integration tests**

```bash
cd /workspace/specagent && pytest tests/integration/test_docx_ocr_converter.py -v
```

Expected: All tests `PASSED`.

- [ ] **Step 7: Run full test suite**

```bash
cd /workspace/specagent && pytest -v --tb=short 2>&1 | tail -30
```

Expected: All previously passing tests still `PASSED`, new tests `PASSED`.

- [ ] **Step 8: Commit**

```bash
cd /workspace/specagent && git add src/specagent/retrieval/docx_ocr_converter.py tests/integration/test_docx_ocr_converter.py
git commit -m "feat(retrieval): add Mermaid validation + one-shot correction loop

Validates every Mermaid diagram before insertion. On failure, re-sends
the original image + error context to Groq VLM for one correction
attempt. Second failure falls back to prose_fallback. Non-diagram
types (table, screenshot_text, other) bypass validation entirely.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 12: Add mmdc config fields and verify full suite + coverage

**Files:**
- Modify: `src/specagent/config.py`

- [ ] **Step 1: Add mmdc config fields after `vision_diagram_types`**

```python
    mermaid_validate_with_mmdc: bool = Field(
        default=False,
        description=(
            "If True and mmdc (@mermaid-js/mermaid-cli) is on PATH, validate generated "
            "Mermaid diagrams using an mmdc subprocess before insertion. "
            "Env var: MERMAID_VALIDATE_WITH_MMDC=true."
        ),
    )
    mermaid_mmdc_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description=(
            "Timeout in seconds for mmdc subprocess validation calls. "
            "Env var: MERMAID_MMdc_TIMEOUT. Default: 10."
        ),
    )
```

- [ ] **Step 2: Run config tests**

```bash
cd /workspace/specagent && pytest tests/unit/test_config.py -v
```

Expected: All `PASSED`.

- [ ] **Step 3: Run the complete test suite**

```bash
cd /workspace/specagent && pytest -v --tb=short
```

Expected: All tests `PASSED`. No regressions.

- [ ] **Step 4: Check coverage**

```bash
cd /workspace/specagent && pytest --cov=src/specagent --cov-report=term-missing --cov-fail-under=70
```

Expected: Coverage ≥ 70% (project target). If below, identify untested branches in the new files and add targeted tests.

- [ ] **Step 5: Run linter**

```bash
cd /workspace/specagent && ruff check src/ tests/ && ruff format src/ tests/
```

Expected: No errors. If format changes files, stage and amend to the previous commit or add to the next commit.

- [ ] **Step 6: Commit Phase 3 completion**

```bash
cd /workspace/specagent && git add src/specagent/config.py
git commit -m "feat(config): add mermaid_validate_with_mmdc and mermaid_mmdc_timeout fields

Enables optional Tier 2 mmdc subprocess validation. Off by default
(no Puppeteer dependency required in standard environments).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Against Spec

| Spec Requirement | Covered by Task |
|---|---|
| Caption extracted from docx XML `w:pStyle Caption` | Task 1–3 |
| Caption prepended as `**Figure: ...**` in stitched output | Task 4 |
| `call_flow_diagram` → `call_flow` rename | Task 5 |
| 5 diagram types with `_MERMAID_SUBTYPE` mapping | Task 5 |
| JSON schema enforcement via `response_format` | Task 7 |
| System prompt with per-type few-shot examples | Task 6 |
| 400 `response_format` fallback retry | Task 7 |
| `prose_fallback` field on `ImageAnalysisResult` | Task 7 |
| `_fix_mermaid_header` deterministic fixup | Task 7 |
| `_vision_prompts.py` extraction for line-length guard | Task 6 |
| `vision_diagram_types` config field | Task 8 |
| `validate_mermaid` Tier 1 Python checks | Task 9 |
| `validate_mermaid` Tier 2 mmdc opt-in | Task 9 |
| `correct_mermaid_diagram` locked type, system message, image re-sent | Task 10 |
| Correction triggered only when first validation fails | Task 11 |
| One retry max; second failure → prose_fallback | Task 11 |
| `mermaid_validate_with_mmdc` + `mermaid_mmdc_timeout` config | Task 12 |
| Non-diagram types bypass validation | Task 11 |
