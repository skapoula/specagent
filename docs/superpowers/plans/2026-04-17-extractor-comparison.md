# Extractor Comparison Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/compare_extractors.py` — a standalone script that runs `prose_dag_extractor` and Groq vision side-by-side on `38413-i30.docx`, producing a Mermaid-validity + step-count scorecard saved to `scripts/comparison_38413.md`.

**Architecture:** The script runs MarkItDown pass-1 conversion once (shared input for both paths), then runs the prose regex extractor and — unless `--dry-run` — the full `convert_docx_with_ocr()` vision pipeline. A pure-function `align_results()` matches outputs by caption, builds `ComparisonRow` dataclasses, and renders the Markdown table.

**Tech Stack:** Python 3.11, existing `specagent` package (`prose_dag_extractor`, `docx_ocr_converter`, `mermaid_validator`, `mermaid_parser`), `argparse`, `asyncio`, `dataclasses`.

---

## File Map

| File                                    | Action                   | Responsibility                                                                              |
| --------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------- |
| `scripts/compare_extractors.py`         | **Create**               | CLI entry point: arg parsing, orchestration, output                                         |
| `scripts/comparison_38413.md`           | **Generated at runtime** | Scorecard output (not committed)                                                            |
| `tests/unit/test_compare_extractors.py` | **Create**               | Unit tests for `align_results`, `render_table`, `count_vision_steps`, `count_vision_actors` |

No existing files are modified.

---

## Task 1: Unit-test scaffolding for pure helper functions

**Files:**

- Create: `tests/unit/test_compare_extractors.py`

The script will expose four pure functions testable without any file I/O or API calls:

- `count_vision_steps(mermaid: str) -> int` — counts `->>` / `-->>` lines in a fenced block
- `count_vision_actors(mermaid: str) -> int` — counts `participant` declaration lines
- `align_results(prose_flows, vision_diagrams) -> list[ComparisonRow]` — matches by caption substring
- `render_table(rows: list[ComparisonRow]) -> str` — returns Markdown table string

- [ ] **Step 1: Create the test file with tests for `count_vision_steps`**

````python
# tests/unit/test_compare_extractors.py
"""Unit tests for compare_extractors helper functions."""

import pytest

# Import lazily inside each test so the module can be imported before the script exists.


@pytest.mark.unit
def test_count_vision_steps_counts_sync_arrows():
    from scripts.compare_extractors import count_vision_steps

    mermaid = "```mermaid\nsequenceDiagram\n    UE->>AMF: Msg1\n    AMF->>SMF: Msg2\n```"
    assert count_vision_steps(mermaid) == 2


@pytest.mark.unit
def test_count_vision_steps_counts_async_arrows():
    from scripts.compare_extractors import count_vision_steps

    mermaid = "```mermaid\nsequenceDiagram\n    AMF-->>UE: Response\n```"
    assert count_vision_steps(mermaid) == 1


@pytest.mark.unit
def test_count_vision_steps_empty_returns_zero():
    from scripts.compare_extractors import count_vision_steps

    assert count_vision_steps("") == 0


@pytest.mark.unit
def test_count_vision_steps_no_arrows_returns_zero():
    from scripts.compare_extractors import count_vision_steps

    mermaid = "```mermaid\nsequenceDiagram\n    participant UE\n```"
    assert count_vision_steps(mermaid) == 0
````

- [ ] **Step 2: Run test to confirm it fails (module not yet created)**

```bash
cd /workspace/specagent && uv run pytest tests/unit/test_compare_extractors.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or `ImportError` — the script does not exist yet.

- [ ] **Step 3: Add tests for `count_vision_actors`**

Append to `tests/unit/test_compare_extractors.py`:

````python
@pytest.mark.unit
def test_count_vision_actors_counts_participant_lines():
    from scripts.compare_extractors import count_vision_actors

    mermaid = "```mermaid\nsequenceDiagram\n    participant UE\n    participant AMF\n    UE->>AMF: Msg\n```"
    assert count_vision_actors(mermaid) == 2


@pytest.mark.unit
def test_count_vision_actors_empty_returns_zero():
    from scripts.compare_extractors import count_vision_actors

    assert count_vision_actors("") == 0
````

- [ ] **Step 4: Add tests for `align_results`**

Append to `tests/unit/test_compare_extractors.py`:

````python
@pytest.mark.unit
def test_align_results_matches_by_caption_substring():
    """Prose flow title substring matches vision diagram caption."""
    from unittest.mock import MagicMock

    from scripts.compare_extractors import align_results

    prose_flow = MagicMock()
    prose_flow.figure_id = "8.1-1"
    prose_flow.title = "NG Setup procedure"
    prose_flow.steps = [MagicMock(), MagicMock()]
    prose_flow.participants = ["gNB", "AMF"]
    prose_flow.mermaid_content = "```mermaid\nsequenceDiagram\n    gNB->>AMF: NG Setup Request\n```"

    vision_diag = MagicMock()
    vision_diag.caption = "NG Setup procedure"
    vision_diag.mermaid_content = "```mermaid\nsequenceDiagram\n    gNB->>AMF: NG Setup Request\n    AMF->>gNB: NG Setup Response\n```"

    rows = align_results([prose_flow], [vision_diag])
    assert len(rows) == 1
    assert rows[0].figure_id == "8.1-1"
    assert rows[0].prose_steps == 2
    assert rows[0].vision_steps == 2


@pytest.mark.unit
def test_align_results_prose_only_when_no_vision_match():
    from unittest.mock import MagicMock

    from scripts.compare_extractors import align_results

    prose_flow = MagicMock()
    prose_flow.figure_id = "8.1-1"
    prose_flow.title = "Unique Procedure"
    prose_flow.steps = [MagicMock()]
    prose_flow.participants = ["UE"]
    prose_flow.mermaid_content = "```mermaid\nsequenceDiagram\n    UE->>AMF: Msg\n```"

    rows = align_results([prose_flow], [])
    assert len(rows) == 1
    assert rows[0].vision_steps == 0
    assert rows[0].vision_valid == "—"


@pytest.mark.unit
def test_align_results_vision_only_when_no_prose_match():
    from unittest.mock import MagicMock

    from scripts.compare_extractors import align_results

    vision_diag = MagicMock()
    vision_diag.caption = "Handover Procedure"
    vision_diag.mermaid_content = "```mermaid\nsequenceDiagram\n    gNB->>AMF: Handover Required\n```"

    rows = align_results([], [vision_diag])
    assert len(rows) == 1
    assert rows[0].figure_id == "—"
    assert rows[0].prose_steps == 0
````

- [ ] **Step 5: Add tests for `render_table`**

Append to `tests/unit/test_compare_extractors.py`:

```python
@pytest.mark.unit
def test_render_table_contains_header_and_summary():
    from scripts.compare_extractors import ComparisonRow, render_table

    rows = [
        ComparisonRow(
            figure_id="8.1-1",
            caption="NG Setup",
            prose_steps=3,
            vision_steps=4,
            prose_actors=2,
            vision_actors=2,
            prose_valid="✓",
            vision_valid="✓",
            winner="vision",
        )
    ]
    table = render_table(rows)
    assert "| figure_id |" in table
    assert "8.1-1" in table
    assert "**Summary**" in table


@pytest.mark.unit
def test_render_table_dry_run_shows_dashes_for_vision():
    from scripts.compare_extractors import ComparisonRow, render_table

    rows = [
        ComparisonRow(
            figure_id="8.1-1",
            caption="NG Setup",
            prose_steps=3,
            vision_steps=0,
            prose_actors=2,
            vision_actors=0,
            prose_valid="✓",
            vision_valid="—",
            winner="—",
        )
    ]
    table = render_table(rows)
    assert "—" in table
```

---

## Task 2: Create `scripts/compare_extractors.py` — data types and pure helpers

**Files:**

- Create: `scripts/compare_extractors.py`

- [ ] **Step 1: Write the failing tests from Task 1 one more time to confirm they all fail**

```bash
cd /workspace/specagent && uv run pytest tests/unit/test_compare_extractors.py -v 2>&1 | tail -5
```

Expected: all tests fail with `ModuleNotFoundError`.

- [ ] **Step 2: Create the script with `ComparisonRow` and the four pure helpers**

```python
#!/usr/bin/env python3
"""Head-to-head comparison: prose_dag_extractor vs Groq vision call-flow extraction.

Usage:
    python scripts/compare_extractors.py           # full comparison (requires GROQ_API_KEY)
    python scripts/compare_extractors.py --dry-run # prose extractor only, no API calls
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure the specagent package is importable when run as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

DOCX_PATH = _REPO_ROOT / "data" / "raw" / "38413-i30.docx"
OUTPUT_PATH = Path(__file__).resolve().parent / "comparison_38413.md"

_ARROW_LINE_RE = re.compile(r"^\s+\S.*?(?:-->>|->>).*?:", re.MULTILINE)
_PARTICIPANT_LINE_RE = re.compile(r"^\s+participant\s+\S", re.MULTILINE | re.IGNORECASE)


@dataclass
class ComparisonRow:
    """One row in the comparison scorecard."""

    figure_id: str
    caption: str
    prose_steps: int
    vision_steps: int
    prose_actors: int
    vision_actors: int
    prose_valid: str   # "✓", "✗"
    vision_valid: str  # "✓", "✗", "—"
    winner: str        # "prose", "vision", "tie", "—"


def count_vision_steps(mermaid: str) -> int:
    """Count message arrow lines (->> or ->>) in a fenced Mermaid block."""
    return len(_ARROW_LINE_RE.findall(mermaid))


def count_vision_actors(mermaid: str) -> int:
    """Count ``participant`` declaration lines in a fenced Mermaid block."""
    return len(_PARTICIPANT_LINE_RE.findall(mermaid))


def _winner(row_prose_steps: int, row_vision_steps: int,
            prose_valid: str, vision_valid: str) -> str:
    """Determine winner from step counts and validity."""
    if vision_valid == "—":
        return "—"
    both_valid = prose_valid == "✓" and vision_valid == "✓"
    if both_valid:
        if row_prose_steps > row_vision_steps:
            return "prose"
        if row_vision_steps > row_prose_steps:
            return "vision"
        return "tie"
    if prose_valid == "✓":
        return "prose"
    if vision_valid == "✓":
        return "vision"
    return "—"


def align_results(prose_flows: list, vision_diagrams: list) -> list[ComparisonRow]:
    """Match prose flows to vision diagrams by caption substring; build ComparisonRow list.

    Matching strategy: a prose flow's title is matched to a vision diagram's caption
    if either string is a case-insensitive substring of the other. Unmatched entries
    appear as prose-only (vision_steps=0, vision_valid="—") or vision-only
    (figure_id="—", prose_steps=0) rows.
    """
    from specagent.retrieval.mermaid_validator import validate_mermaid

    matched_vision_indices: set[int] = set()
    rows: list[ComparisonRow] = []

    for flow in prose_flows:
        # Find best vision match by caption substring
        matched_idx: int | None = None
        for i, diag in enumerate(vision_diagrams):
            if i in matched_vision_indices:
                continue
            t_lower = flow.title.lower()
            c_lower = (diag.caption or "").lower()
            if c_lower and (t_lower in c_lower or c_lower in t_lower):
                matched_idx = i
                break

        prose_valid = "✓" if validate_mermaid(flow.mermaid_content)[0] else "✗"

        if matched_idx is not None:
            matched_vision_indices.add(matched_idx)
            diag = vision_diagrams[matched_idx]
            v_steps = count_vision_steps(diag.mermaid_content)
            v_actors = count_vision_actors(diag.mermaid_content)
            vision_valid = "✓" if validate_mermaid(diag.mermaid_content)[0] else "✗"
            rows.append(ComparisonRow(
                figure_id=flow.figure_id,
                caption=flow.title,
                prose_steps=len(flow.steps),
                vision_steps=v_steps,
                prose_actors=len(flow.participants),
                vision_actors=v_actors,
                prose_valid=prose_valid,
                vision_valid=vision_valid,
                winner=_winner(len(flow.steps), v_steps, prose_valid, vision_valid),
            ))
        else:
            # prose-only
            rows.append(ComparisonRow(
                figure_id=flow.figure_id,
                caption=flow.title,
                prose_steps=len(flow.steps),
                vision_steps=0,
                prose_actors=len(flow.participants),
                vision_actors=0,
                prose_valid=prose_valid,
                vision_valid="—",
                winner="—",
            ))

    # vision-only rows
    for i, diag in enumerate(vision_diagrams):
        if i in matched_vision_indices:
            continue
        v_steps = count_vision_steps(diag.mermaid_content)
        v_actors = count_vision_actors(diag.mermaid_content)
        vision_valid = "✓" if validate_mermaid(diag.mermaid_content)[0] else "✗"
        rows.append(ComparisonRow(
            figure_id="—",
            caption=diag.caption or "(no caption)",
            prose_steps=0,
            vision_steps=v_steps,
            prose_actors=0,
            vision_actors=v_actors,
            prose_valid="✗",
            vision_valid=vision_valid,
            winner=_winner(0, v_steps, "✗", vision_valid),
        ))

    return rows


def render_table(rows: list[ComparisonRow]) -> str:
    """Render comparison rows as a Markdown table with a summary row."""
    header = (
        "| figure_id | caption | prose_steps | vision_steps "
        "| prose_actors | vision_actors | prose_valid | vision_valid | winner |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    body_lines: list[str] = []
    for r in rows:
        caption_short = r.caption[:50] + "…" if len(r.caption) > 50 else r.caption
        body_lines.append(
            f"| {r.figure_id} | {caption_short} | {r.prose_steps} | {r.vision_steps} "
            f"| {r.prose_actors} | {r.vision_actors} | {r.prose_valid} "
            f"| {r.vision_valid} | {r.winner} |"
        )

    prose_wins = sum(1 for r in rows if r.winner == "prose")
    vision_wins = sum(1 for r in rows if r.winner == "vision")
    ties = sum(1 for r in rows if r.winner == "tie")
    prose_only = sum(1 for r in rows if r.vision_valid == "—" and r.prose_steps > 0)
    vision_only = sum(1 for r in rows if r.figure_id == "—")

    summary = (
        f"\n**Summary:** {len(rows)} figures | "
        f"prose wins: {prose_wins} | vision wins: {vision_wins} | "
        f"ties: {ties} | prose-only: {prose_only} | vision-only: {vision_only}"
    )

    return header + "\n".join(body_lines) + "\n" + summary
```

- [ ] **Step 3: Run the tests to confirm they pass**

```bash
cd /workspace/specagent && uv run pytest tests/unit/test_compare_extractors.py -v
```

Expected: all tests pass (`PASSED`).

- [ ] **Step 4: Commit**

```bash
cd /workspace/specagent
git checkout -b feat/extractor-comparison
git add scripts/compare_extractors.py tests/unit/test_compare_extractors.py
git commit -m "feat(comparison): add ComparisonRow, pure helpers, and unit tests"
```

---

## Task 3: Add `main()` — orchestration, `--dry-run`, and file output

**Files:**

- Modify: `scripts/compare_extractors.py` — append `main()` and `if __name__ == "__main__"` block

- [ ] **Step 1: Append `main()` to `scripts/compare_extractors.py`**

Add this after the `render_table` function:

```python
async def _run_vision(api_key: str) -> tuple[str, list]:
    """Run the two-pass OCR pipeline and return (enriched_markdown, diagrams)."""
    from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr

    return await convert_docx_with_ocr(DOCX_PATH, api_key=api_key)


def main() -> None:
    """Entry point: parse args, run extractors, print and save scorecard."""
    parser = argparse.ArgumentParser(
        description="Compare prose_dag_extractor vs Groq vision on 38413-i30.docx"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run prose extractor only — skip Groq vision API calls",
    )
    args = parser.parse_args()

    if not DOCX_PATH.exists():
        print(f"ERROR: spec file not found: {DOCX_PATH}", file=sys.stderr)
        sys.exit(1)

    api_key = ""
    if not args.dry_run:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print(
                "ERROR: GROQ_API_KEY environment variable is not set.\n"
                "Set it or use --dry-run to run the prose extractor only.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── Pass 1: MarkItDown → Markdown ─────────────────────────────────────
    print(f"Converting {DOCX_PATH.name} with MarkItDown…", flush=True)
    from specagent.retrieval.converter import convert

    markdown = convert(DOCX_PATH)

    # ── Prose extractor ────────────────────────────────────────────────────
    print("Running prose DAG extractor…", flush=True)
    from specagent.retrieval.prose_dag_extractor import extract_prose_call_flows

    prose_flows = extract_prose_call_flows(markdown)
    print(f"  → {len(prose_flows)} figures with parseable steps found.", flush=True)

    # ── Groq vision (skipped in dry-run) ───────────────────────────────────
    vision_diagrams: list = []
    if args.dry_run:
        print("Dry-run: skipping Groq vision pass.", flush=True)
    else:
        print(f"Running Groq vision pipeline on {DOCX_PATH.name}…", flush=True)
        _enriched_md, vision_diagrams = asyncio.run(_run_vision(api_key))
        print(f"  → {len(vision_diagrams)} call-flow diagrams extracted.", flush=True)

    # ── Align and render ───────────────────────────────────────────────────
    rows = align_results(prose_flows, vision_diagrams)

    if not rows:
        print("WARNING: no figures found by either extractor.", flush=True)
        sys.exit(0)

    table = render_table(rows)

    mode = "DRY-RUN (prose only)" if args.dry_run else "FULL COMPARISON"
    header = f"# Extractor Comparison: 38413-i30.docx — {mode}\n\n"
    output = header + table + "\n"

    print("\n" + output)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"\nSaved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports cleanly (no syntax errors)**

```bash
cd /workspace/specagent && uv run python -c "import scripts.compare_extractors"
```

Expected: no output (clean import).

- [ ] **Step 3: Run `--dry-run` against the real spec file**

```bash
cd /workspace/specagent && uv run python scripts/compare_extractors.py --dry-run
```

Expected output (sample):

```
Converting 38413-i30.docx with MarkItDown…
Running prose DAG extractor…
  → N figures with parseable steps found.
Dry-run: skipping Groq vision pass.

# Extractor Comparison: 38413-i30.docx — DRY-RUN (prose only)
...
| figure_id | caption | prose_steps | vision_steps | ...
...
**Summary:** N figures | prose wins: 0 | vision wins: 0 | ties: 0 | ...

Saved to scripts/comparison_38413.md
```

Confirm `scripts/comparison_38413.md` is created and non-empty:

```bash
wc -l /workspace/specagent/scripts/comparison_38413.md
```

- [ ] **Step 4: Run full test suite to confirm no regressions**

```bash
cd /workspace/specagent && uv run pytest tests/unit/ -v --tb=short 2>&1 | tail -20
```

Expected: all previously passing tests still pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/specagent
git add scripts/compare_extractors.py
git commit -m "feat(comparison): add main() with --dry-run, orchestration, and file output"
```

---

## Task 4: Live run (full comparison with Groq vision)

**Files:** No code changes — runtime execution only.

- [ ] **Step 1: Confirm `GROQ_API_KEY` is set**

```bash
echo "GROQ_API_KEY is set: $([ -n "$GROQ_API_KEY" ] && echo YES || echo NO)"
```

If NO: `export GROQ_API_KEY=<your-key>` before proceeding.

- [ ] **Step 2: Run the full comparison**

```bash
cd /workspace/specagent && uv run python scripts/compare_extractors.py
```

Expected: conversion → vision API calls (30–150 s) → scorecard printed → saved to `scripts/comparison_38413.md`.

- [ ] **Step 3: Inspect the scorecard**

```bash
cat /workspace/specagent/scripts/comparison_38413.md
```

Check:

- Summary row shows non-zero totals for both prose wins and vision wins
- Rows with `vision_valid=✗` identify where the vision pipeline struggled
- Rows with `prose_steps=0` identify diagrams the regex couldn't parse from prose

- [ ] **Step 4: Commit the scorecard**

```bash
cd /workspace/specagent
git add scripts/comparison_38413.md
git commit -m "chore(comparison): add 38413 extractor comparison results"
```
