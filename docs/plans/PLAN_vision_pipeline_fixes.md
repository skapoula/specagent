# PLAN: Vision Pipeline Bug Fixes

> **Scope:** `retrieval/` OCR sub-pipeline + `ingestor.py` + `kuzu/`
> **Source issues:** 13 bugs identified via full-pipeline critique (2026-04-27)
> **Status:** DRAFT — awaiting human approval before implementation

---

## Overview

The 13 issues are grouped into four implementation phases ordered by severity.
Each phase can be reviewed and merged independently. Later phases do not depend
on earlier ones unless stated.

| Phase | Issues      | Severity | Estimated risk                                           |
| ----- | ----------- | -------- | -------------------------------------------------------- |
| 1     | 1, 2        | Critical | Low — targeted logic fixes                               |
| 2     | 3, 4, 5, 6  | High     | Medium — async threading, rate-limiter internals, config |
| 3     | 7, 8, 9, 10 | Medium   | Low-Medium                                               |
| 4     | 11, 12, 13  | Low      | Low — config + docs                                      |

---

## Phase 1 — Critical: Data Integrity Fixes

### Issue 1 — `doc_id=""` permanently breaks the Kuzu→LanceDB link

**Root cause:** `ingestor.py:165` assigns `doc_id = str(uuid.uuid4())` at step 6,
but `_store_diagrams_as_dags` and `_store_prose_dags` are called at steps 3b/3c
with a hardcoded `doc_id=""`. Every Kuzu node for call-flow DAGs carries an empty
doc_id, making Kuzu→LanceDB joins impossible.

**Fix — `src/specagent/retrieval/ingestor.py`:**

1. Move UUID generation to the top of `ingest()`, immediately after step 2 (dedup
   check). Insert before line 110:

   ```python
   # Assign doc_id early so DAG storage can reference the same ID written to LanceDB.
   doc_id = str(uuid.uuid4())
   ```

2. Remove the duplicate `doc_id = str(uuid.uuid4())` on the current line 165.

3. Update the call at step 3b:

   ```python
   _store_diagrams_as_dags(diagrams, doc_name=path.stem, source=source_str, doc_id=doc_id)
   ```

4. Update the call at step 3c:

   ```python
   _store_prose_dags(text, doc_name=path.stem, source=source_str, doc_id=doc_id)
   ```

5. Update `_store_diagrams_as_dags` signature and body
   (`ingestor.py:362`): add `doc_id: str` parameter; pass it to
   `dag_store.store_call_flow_dag(doc_id=doc_id, ...)`.

6. Update `_store_prose_dags` signature and body (`ingestor.py:331`):
   same — add `doc_id: str` parameter, pass through.

**Tests to write (TDD — before implementing):**

- `tests/unit/test_ingestor.py`: assert that `_store_diagrams_as_dags` is called
  with a non-empty `doc_id` string matching the one returned in `IngestResult`.
- `tests/unit/test_ingestor.py`: assert `_store_prose_dags` receives the same non-empty
  `doc_id`.
- Use `unittest.mock.patch` on `dag_store.store_call_flow_dag` to capture the
  `doc_id` kwarg.

---

### Issue 2 — Prose-fallback diagrams reach `parse_sequence_diagram` as plain text

**Root cause:** `docx_ocr_converter.py:_prose_fallback_result()` copies the result
with a new `markdown_content` (plain English sentence or marker) but does **not** set
`skipped=True`. The DAG collection filter at line 215 checks `not result.skipped`,
so a call-flow with prose fallback passes the filter. `parse_sequence_diagram` then
receives a one-sentence English string, silently produces empty participants and steps,
and Kuzu stores a structurally hollow node with invalid `mermaid_content`.

**Fix — `src/specagent/retrieval/docx_ocr_converter.py`:**

1. In `_prose_fallback_result()` (line 77), set `skipped=True` and populate
   `skip_reason`:

   ```python
   def _prose_fallback_result(
       result: ImageAnalysisResult, placeholder_name: str
   ) -> ImageAnalysisResult:
       fallback = (
           result.prose_fallback
           or f"_[Diagram: {placeholder_name} — validation failed]_"
       )
       return result.model_copy(update={
           "markdown_content": fallback,
           "skipped": True,
           "skip_reason": "mermaid_validation_failed_prose_fallback",
       })
   ```

2. The existing DAG collection filter (`result.image_type in _DAG_DIAGRAM_TYPES and not result.skipped`)
   already handles `skipped=True` correctly — no further change needed there.

3. Verify that `_stitch()` also skips results where `result.skipped=True` — it
   already does (`if idx in results and not results[idx].skipped`). No change needed.

**Tests to write:**

- `tests/unit/test_docx_ocr_converter.py`: call `_prose_fallback_result()` with a
  `call_flow` ImageAnalysisResult; assert `result.skipped is True` and
  `result.skip_reason == "mermaid_validation_failed_prose_fallback"`.
- Assert that a prose-fallback result does not appear in the `diagrams` list returned
  by `convert_docx_with_ocr` (mock the VLM to return a bad Mermaid block that
  triggers two validation failures).

---

## Phase 2 — High: Async Correctness, Rate Limiting, and Token Budget

### Issue 3 — Kuzu writes block the event loop

**Root cause:** `_store_diagrams_as_dags` and `_store_prose_dags` are called
synchronously inside the `async ingest()` function. Kuzu is an embedded,
synchronous database. Writes for a spec with 30+ call flows hold the event loop
for their full duration, stalling all concurrent `ingest_folder` coroutines.

**Fix — `src/specagent/retrieval/ingestor.py`:**

1. Both helpers are `def` (sync). Keep them sync — they are pure sync logic.

2. Wrap each call site in `asyncio.to_thread()`:

   ```python
   # Step 3b
   if settings.enable_dag_storage and diagrams:
       await asyncio.to_thread(
           _store_diagrams_as_dags, diagrams, path.stem, source_str, doc_id
       )

   # Step 3c
   if settings.enable_dag_storage and not diagrams:
       await asyncio.to_thread(
           _store_prose_dags, text, path.stem, source_str, doc_id
       )
   ```

**Tests to write:**

- `tests/unit/test_ingestor.py`: mock `_store_diagrams_as_dags`; confirm it is
  invoked via `asyncio.to_thread` (i.e. not called directly on the main thread).
  A straightforward way: patch `asyncio.to_thread` and assert it was called with
  `_store_diagrams_as_dags` as the first argument.

---

### Issue 4 — Rate limiter is acquired once before the tenacity retry loop

**Root cause:** `groq_vision_client.py:analyze_image` (line 117) calls
`await _get_rate_limiter().acquire()` once before the `@retry`-decorated `_call()`.
Each tenacity retry re-invokes `_call()` without re-acquiring a limiter slot, so
up to 6 calls (1 + 5 retries) consume only 1 slot. Under 429 pressure this causes
actual quota overrun.

Same problem in `correct_mermaid_diagram` (line 220).

**Fix — `src/specagent/retrieval/groq_vision_client.py`:**

1. Remove the top-level `await _get_rate_limiter().acquire()` from both
   `analyze_image` and `correct_mermaid_diagram`.

2. Add `await _get_rate_limiter().acquire()` as the **first statement inside
   `_call()`** in both functions:

   ```python
   @retry(...)
   async def _call() -> ImageAnalysisResult:
       await _get_rate_limiter().acquire()   # ← moved here
       async with httpx.AsyncClient(timeout=30.0) as client:
           ...
   ```

   This ensures every tenacity attempt (including retries) re-acquires a limiter
   slot before touching the API.

**Tests to write:**

- `tests/unit/test_groq_vision_client.py`: simulate a 429 → 200 sequence (two
  httpx responses); assert `acquire()` is called exactly twice (once per attempt).
  Use `unittest.mock.AsyncMock` for the rate limiter.

---

### Issue 5 — Inkscape is a silent hard dependency

**Root cause:** `emf_converter.py:convert_emf_to_jpeg` raises `IngestionError`
when Inkscape is absent, which `_prepare_image` in `docx_ocr_converter.py` catches
as a warning and returns `None`. Every EMF diagram is silently skipped with no
visible error to the operator. Inkscape is undocumented in `pyproject.toml`,
`installation.md`, and the CLAUDE.md.

**Fix — three sub-tasks:**

**5a. Startup validation — `src/specagent/retrieval/docx_ocr_converter.py`:**

Add a module-level helper and call it in `convert_docx_with_ocr` before the image
extraction loop:

```python
import shutil

def _warn_if_inkscape_missing() -> None:
    """Log a prominent warning if Inkscape is not installed."""
    if shutil.which("inkscape") is None:
        logger.warning(
            "Inkscape not found on PATH. EMF/WMF diagrams (common in 3GPP .docx files) "
            "will be silently skipped. Install Inkscape to enable diagram extraction."
        )
```

Call `_warn_if_inkscape_missing()` once at the start of `convert_docx_with_ocr`.

**5b. pyproject.toml extras:**

Add an `[inkscape]` optional dependency note and document the system-package
requirement in the `[tool.uv]` or `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
ocr = ["pillow", "pymupdf"]  # existing
# Inkscape must be installed as a system package for EMF/WMF support:
# apt-get install inkscape   (Debian/Ubuntu)
# brew install inkscape      (macOS)
```

**5c. `docs/installation.md`:**

Add Inkscape to the "Optional: .docx OCR" requirements section with the exact
install command for each supported platform.

**Tests to write:**

- `tests/unit/test_emf_converter.py`: assert `convert_emf_to_jpeg` raises
  `IngestionError` with a message containing "Inkscape" when `shutil.which` returns
  `None` (patch `shutil.which`).
- `tests/unit/test_docx_ocr_converter.py`: assert `_warn_if_inkscape_missing`
  is called at the top of `convert_docx_with_ocr` (patch `shutil.which` to return
  `None`; assert `logger.warning` is called with "Inkscape").

---

### Issue 6 — `max_tokens=1024` truncates complex call-flow diagrams

**Root cause:** Both `analyze_image` and `correct_mermaid_diagram` in
`groq_vision_client.py` hardcode `"max_tokens": 1024`. A 3GPP registration
sequence with 35 steps requires ~1 500–2 000 tokens. Truncation causes Mermaid
validation failures → correction call (also 1024 tokens → also truncated) →
prose fallback. Two API calls burned per complex diagram, zero useful output.

**Fix:**

**6a. Add config field — `src/specagent/config.py`:**

```python
vision_max_tokens: int = Field(
    default=4096,
    ge=256,
    le=32768,
    description=(
        "Max output tokens for Groq vision API calls. "
        "3GPP call-flow diagrams with 30–50 steps require ~2000 tokens. "
        "Env var: VISION_MAX_TOKENS."
    ),
)
```

**6b. Thread config through — `src/specagent/retrieval/groq_vision_client.py`:**

Replace `"max_tokens": 1024` in both `analyze_image._call()` and
`correct_mermaid_diagram._call()` with:

```python
from specagent.config import settings
...
"max_tokens": settings.vision_max_tokens,
```

**Tests to write:**

- `tests/unit/test_groq_vision_client.py`: assert `body["max_tokens"]` equals
  `settings.vision_max_tokens`, not a hardcoded integer. Use `pytest-httpx` to
  intercept the request body.

---

## Phase 3 — Medium: Reliability and Correctness

### Issue 7 — `_stitch()` relies on MarkItDown placeholder ordering

**Root cause:** `_stitch()` replaces placeholders by a sequential counter that
must match the order of relationship IDs in `word/_rels/document.xml.rels`. The
comment in the code ("This matches MarkItDown's placeholder order regardless of
URL format") is an assumption, not a guarantee. A MarkItDown version change that
reorders output would silently swap diagram content.

**Fix — `src/specagent/retrieval/docx_ocr_converter.py`:**

The `ExtractedImage.placeholder_name` field already stores the filename portion
(e.g., `"image0.png"`). MarkItDown renders placeholders as `![image](image0.png)`.
The fix is to key `results` by `placeholder_name` rather than array index, and
match inside `_stitch()` by the URL in each `![alt](url)` match group.

1. In `convert_docx_with_ocr`, change the results dict key:

   ```python
   results: dict[str, ImageAnalysisResult] = {}  # key: placeholder_name
   ...
   results[image.placeholder_name] = result
   ```

2. Update captions dict similarly: `captions: dict[str, str]`, keyed by
   `images[i].placeholder_name`.

3. Update `_stitch()` signature and body:

   ```python
   def _stitch(
       markdown: str,
       results: dict[str, ImageAnalysisResult],
       captions: dict[str, str] | None = None,
   ) -> str:
       ...
       def _replace(match: re.Match[str]) -> str:
           url = match.group(1)          # e.g. "image0.png" or a data URI
           # Normalise: extract filename from data URIs if needed
           key = Path(url).name if not url.startswith("data:") else url
           if key in results and not results[key].skipped:
               ...
   ```

4. The `ExtractedDiagram` collector also needs updating to use the
   `placeholder_name` key.

**Tests to write:**

- `tests/unit/test_docx_ocr_converter.py`: feed `_stitch()` a Markdown string with
  two placeholders in reversed order relative to `results`; verify each gets the
  correct content based on the URL name, not insertion order.

---

### Issue 8 — No per-image vision call deduplication on re-ingest

**Root cause:** The document-level SHA-256 dedup (step 2 in `ingest()`) catches
unchanged documents. But if a doc changes for a non-visual reason, or if ingest
is interrupted mid-run, all images are re-submitted to the VLM. On the free tier
(1000 RPD), re-processing one 20-image 3GPP spec wastes 2–4% of the daily quota.

**Fix:**

Introduce a lightweight image-level result cache in
`src/specagent/retrieval/vision_cache.py`. The cache maps `image_content_sha256`
→ `ImageAnalysisResult`, stored as a JSON file in `data/vision_cache/`.

1. New file `vision_cache.py`:

   ```python
   """Disk-backed cache for Groq vision API results, keyed by image content hash."""
   import hashlib, json, logging
   from pathlib import Path
   from specagent.retrieval.groq_vision_client import ImageAnalysisResult

   logger = logging.getLogger(__name__)

   class VisionCache:
       def __init__(self, cache_path: Path) -> None:
           self._path = cache_path
           self._data: dict[str, dict] = {}
           self._load()

       def _load(self) -> None:
           if self._path.exists():
               try:
                   self._data = json.loads(self._path.read_text())
               except Exception:
                   logger.warning("Vision cache corrupted at %s; starting fresh", self._path)

       def get(self, image_bytes: bytes) -> ImageAnalysisResult | None:
           key = hashlib.sha256(image_bytes).hexdigest()
           raw = self._data.get(key)
           return ImageAnalysisResult.model_validate(raw) if raw else None

       def put(self, image_bytes: bytes, result: ImageAnalysisResult) -> None:
           key = hashlib.sha256(image_bytes).hexdigest()
           self._data[key] = result.model_dump()
           try:
               self._path.parent.mkdir(parents=True, exist_ok=True)
               self._path.write_text(json.dumps(self._data, indent=2))
           except Exception as exc:
               logger.warning("Failed to write vision cache: %s", exc)
   ```

2. Add `vision_cache_path: Path = Path("data/vision_cache/results.json")` to
   `Settings` in `config.py`.

3. In `convert_docx_with_ocr`, construct a `VisionCache` instance once per
   converter call (or use a module-level singleton gated by
   `settings.enable_docx_ocr`). Before `analyze_image`, check the cache; on hit,
   skip the API call. On miss, call `analyze_image` and write to cache.

**Tests to write:**

- `tests/unit/test_vision_cache.py`: round-trip put/get with a small byte string;
  assert the returned result matches the stored one.
- `tests/unit/test_vision_cache.py`: assert `get()` returns `None` for an unseen
  image hash.
- `tests/unit/test_docx_ocr_converter.py`: mock `analyze_image`; pre-populate
  cache with one image hash; assert `analyze_image` is not called for that image.

---

### Issue 9 — `_check_with_mmdc` leaks a temp file on unexpected subprocess error

**Root cause:** `mermaid_validator.py:_check_with_mmdc` (line 133) writes a temp
file and then calls `subprocess.run`. If `subprocess.run` raises an unexpected
exception (e.g., `PermissionError`, `OSError`), the `tmp_path.unlink()` at line
140 is never reached.

**Fix — `src/specagent/retrieval/mermaid_validator.py`:**

Wrap the subprocess call and unlink in a `try/finally`:

```python
def _check_with_mmdc(inner: str) -> tuple[bool, str]:
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mmd", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(inner)
            tmp_path = Path(tmp.name)
        try:
            result = subprocess.run(
                ["mmdc", "-i", str(tmp_path), "-o", "/dev/null"],
                capture_output=True,
                text=True,
                timeout=settings.mermaid_mmdc_timeout,
                check=False,
            )
        finally:
            tmp_path.unlink(missing_ok=True)   # ← always cleans up
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

**Tests to write:**

- `tests/unit/test_mermaid_validator.py`: patch `subprocess.run` to raise
  `OSError`; assert the temp file no longer exists after `_check_with_mmdc`
  returns (check `tmp_path.exists()` via monkeypatching `Path.unlink`).

---

### Issue 10 — `_check_bracket_balance` mis-handles Mermaid `%%` comments and apostrophes

**Root cause:** `mermaid_validator.py:_check_bracket_balance` (line 105) toggles
`in_string` on every `"`. A label containing an apostrophe (e.g., `UE's response`)
will flip `in_string` on the apostrophe (treated as a double-quote in the naive
loop because they are both `"`—this is only if the code uses `'` too, but the current
code only checks `"`). More critically, `%%` comment lines in Mermaid can contain
any characters including unbalanced brackets, causing false negatives.

**Fix — `src/specagent/retrieval/mermaid_validator.py`:**

Rewrite `_check_bracket_balance` to strip `%%` comment lines first, then process
character-by-character with correct quote handling:

```python
def _check_bracket_balance(inner: str) -> bool:
    """Return True if brackets, parentheses, and braces balance.

    Skips %% comment lines. Handles both single-quoted and double-quoted strings.
    """
    opens  = {"[": "]", "(": ")", "{": "}"}
    closes = {v: k for k, v in opens.items()}
    stack: list[str] = []
    in_string: str | None = None   # None, '"', or "'"

    for line in inner.splitlines():
        stripped = line.strip()
        if stripped.startswith("%%"):
            continue                # skip Mermaid comment lines
        for char in line:
            if in_string:
                if char == in_string:
                    in_string = None
            elif char in ('"', "'"):
                in_string = char
            elif char in opens:
                stack.append(opens[char])
            elif char in closes:
                if not stack or stack[-1] != char:
                    return False
                stack.pop()
    return len(stack) == 0
```

Note: This is a stricter check. The existing Tier 1 test suite may need updating
to handle legitimate Mermaid syntax (e.g., `note over A,B: text [optional]` with
intentionally unquoted brackets). Add a config flag
`mermaid_strict_bracket_check: bool = False` to gate the stricter mode if needed.

**Tests to write:**

- `tests/unit/test_mermaid_validator.py`:
  - `%%` comment with unbalanced `[` → should return `True` (brackets in comments ignored)
  - Label `"UE's response"` inside a sequenceDiagram → should return `True`
  - `note over UE,gNB: message [optional]` (unquoted bracket) → confirm behavior
    and document expected return value
  - Genuinely unbalanced `{` outside string/comment → should return `False`

---

## Phase 4 — Low: Cost Guard, Multi-Process Safety, and Model Separation

### Issue 11 — No bulk cost guard for vision calls

**Root cause:** `ingest_folder` with 50 .docx files at ~20 images each submits
~1 000 VLM calls. This exhausts the entire Groq free-tier daily quota (1 000 RPD)
in one run. There is no pre-flight estimate, no opt-in confirmation, and no cap.

**Fix:**

**11a. Pre-flight image count estimate — `src/specagent/retrieval/ingestor.py`:**

Add an `estimate_vision_calls(folder: Path) -> int` helper that counts images
across all .docx files without converting them:

```python
def estimate_vision_calls(folder: Path) -> int:
    """Count images in all .docx files under folder (no conversion)."""
    from specagent.retrieval.docx_image_extractor import extract_images
    total = 0
    for p in folder.rglob("*.docx"):
        try:
            total += len(extract_images(p))
        except Exception:
            pass
    return total
```

**11b. CLI `--max-vision-calls` flag — `src/specagent/cli.py`:**

Add a `max_vision_calls: int = typer.Option(0, help="...")` parameter to the
`index` command. When set, abort with a clear error if the pre-flight estimate
exceeds the limit. When `max_vision_calls=0`, proceed without limit (current
behavior) but log a warning with the estimate.

**11c. Settings field — `src/specagent/config.py`:**

```python
vision_max_calls_per_run: int = Field(
    default=0,
    ge=0,
    description=(
        "Maximum vision API calls allowed per ingest_folder run. "
        "0 = unlimited. Env var: VISION_MAX_CALLS_PER_RUN."
    ),
)
```

**Tests to write:**

- `tests/unit/test_ingestor.py`: `estimate_vision_calls` returns correct count
  when pointed at a temp directory with mock .docx content.
- CLI integration test (optional, medium effort): assert `specagent index` prints
  a warning when estimated calls exceed `vision_rpd_limit`.

---

### Issue 12 — Rate limiter is asyncio-only; unsafe under multi-process deployments

**Root cause:** `GroqVisionRateLimiter` uses `asyncio.Lock`, which only protects
within a single event loop. Two concurrent OS processes each maintain an independent
singleton, effectively doubling the real call rate against the API.

**Fix:**

This issue is architectural: a process-safe rate limiter requires either a shared
file lock, a Redis counter, or a sidecar service — all disproportionate for the
current use case (the default is `api_workers=1`).

**Pragmatic approach:**

1. Add a startup check in `api/main.py` lifespan:

   ```python
   if settings.api_workers > 1 and settings.enable_docx_ocr:
       logger.warning(
           "api_workers=%d with enable_docx_ocr=True: the vision rate limiter "
           "is not safe across multiple workers. Set api_workers=1 or use "
           "VISION_MAX_CALLS_PER_RUN to cap per-worker quota.",
           settings.api_workers,
       )
   ```

2. Add a note to `docs/developer-guide.md` under "Rate Limiting" documenting this
   constraint and the recommended single-worker configuration for OCR workloads.

**Tests to write:**

- `tests/unit/test_api_main.py`: assert the warning is logged when
  `api_workers > 1` and `enable_docx_ocr=True` (use `caplog` fixture).

---

### Issue 13 — Vision model shares the text LLM's Groq rate-limit bucket

**Root cause:** `groq_vision_client.py:_DEFAULT_MODEL` is hardcoded as
`"meta-llama/llama-4-scout-17b-16e-instruct"` — the same string used by
`llm/factory.py` for query/answer generation. Groq applies rate limits per model.
Heavy indexing runs eat into the query-time TPM budget, degrading live performance.

`settings.vision_model` already exists in `config.py` with the correct default,
but `analyze_image` and `correct_mermaid_diagram` default to `_DEFAULT_MODEL`,
and neither call site in `docx_ocr_converter.py` passes an explicit model.

**Fix:**

**13a. `src/specagent/retrieval/groq_vision_client.py`:**

Remove the `_DEFAULT_MODEL` sentinel and change the function signatures to read
from settings:

```python
async def analyze_image(
    image: ExtractedImage,
    api_key: str,
    model: str | None = None,
) -> ImageAnalysisResult:
    from specagent.config import settings
    _model = model or settings.vision_model
    ...
    body = {"model": _model, ...}
```

Apply the same change to `correct_mermaid_diagram`.

**13b. `docs/developer-guide.md`:**

Add a note explaining that `VISION_MODEL` and `GROQ_MODEL` should be set to
different models when both OCR indexing and live query serving run concurrently,
to isolate their respective Groq rate-limit buckets.

**Tests to write:**

- `tests/unit/test_groq_vision_client.py`: assert `body["model"]` equals
  `settings.vision_model` when `model=None` is passed (use `Settings(vision_model="test-model")`
  override).

---

## Implementation Order Within Each Phase

Within each phase, implement in this order to follow TDD:

1. Write failing test(s) in `tests/unit/test_<module>.py`
2. Run `uv run pytest tests/unit/test_<module>.py -v` — confirm failure
3. Implement the fix
4. Run the full test suite: `uv run pytest && uv run ruff check . && uv run mypy src/`
5. Commit: `fix(<scope>): <description>`

Each issue = one atomic commit on its own feature branch, e.g.:

- `fix/ingestor-doc-id-dag-link` (Issue 1)
- `fix/prose-fallback-skipped-flag` (Issue 2)
- `fix/kuzu-writes-to-thread` (Issue 3)
- etc.

---

## Files Modified per Phase

### Phase 1

| File                                            | Change                                                    |
| ----------------------------------------------- | --------------------------------------------------------- |
| `src/specagent/retrieval/ingestor.py`           | Move UUID generation; thread `doc_id` through DAG helpers |
| `src/specagent/retrieval/docx_ocr_converter.py` | Set `skipped=True` in `_prose_fallback_result`            |
| `tests/unit/test_ingestor.py`                   | New tests for `doc_id` propagation                        |
| `tests/unit/test_docx_ocr_converter.py`         | New tests for prose fallback skipped flag                 |

### Phase 2

| File                                            | Change                                                              |
| ----------------------------------------------- | ------------------------------------------------------------------- |
| `src/specagent/retrieval/ingestor.py`           | `asyncio.to_thread` for DAG store calls                             |
| `src/specagent/retrieval/groq_vision_client.py` | Move `acquire()` inside `_call()`; use `settings.vision_max_tokens` |
| `src/specagent/retrieval/docx_ocr_converter.py` | `_warn_if_inkscape_missing()`                                       |
| `src/specagent/config.py`                       | Add `vision_max_tokens` field                                       |
| `pyproject.toml`                                | Document Inkscape as system dependency in optional extras           |
| `docs/installation.md`                          | Inkscape install instructions                                       |
| `tests/unit/test_groq_vision_client.py`         | Rate-limiter and max_tokens tests                                   |
| `tests/unit/test_emf_converter.py`              | Inkscape-absent error test                                          |

### Phase 3

| File                                            | Change                                                    |
| ----------------------------------------------- | --------------------------------------------------------- |
| `src/specagent/retrieval/docx_ocr_converter.py` | Name-keyed results/captions dicts; update `_stitch()`     |
| `src/specagent/retrieval/mermaid_validator.py`  | `finally` block; rewrite `_check_bracket_balance`         |
| `src/specagent/retrieval/vision_cache.py`       | New file: disk-backed vision cache                        |
| `src/specagent/config.py`                       | Add `vision_cache_path` field                             |
| `tests/unit/test_docx_ocr_converter.py`         | Stitch ordering tests; cache integration test             |
| `tests/unit/test_mermaid_validator.py`          | Comment and apostrophe bracket tests; temp-file leak test |
| `tests/unit/test_vision_cache.py`               | New test file                                             |

### Phase 4

| File                                            | Change                                                 |
| ----------------------------------------------- | ------------------------------------------------------ |
| `src/specagent/retrieval/ingestor.py`           | `estimate_vision_calls()` helper                       |
| `src/specagent/cli.py`                          | `--max-vision-calls` flag on `index` command           |
| `src/specagent/config.py`                       | `vision_max_calls_per_run` field                       |
| `src/specagent/api/main.py`                     | Multi-worker + OCR warning                             |
| `src/specagent/retrieval/groq_vision_client.py` | Use `settings.vision_model` as default                 |
| `docs/developer-guide.md`                       | Rate-limiter multi-process note; model separation note |
| `tests/unit/test_ingestor.py`                   | `estimate_vision_calls` test                           |
| `tests/unit/test_groq_vision_client.py`         | `vision_model` default test                            |
