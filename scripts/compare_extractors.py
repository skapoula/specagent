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

_ARROW_LINE_RE = re.compile(
    r"^\s+\S.*?(?:-->>|--x|-->|->>|->|-x|--\)|--|-\)).*?:",
    re.MULTILINE,
)
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
    prose_valid: str  # "✓", "✗"
    vision_valid: str  # "✓", "✗", "—"
    winner: str  # "prose", "vision", "tie", "—"


def count_vision_steps(mermaid: str) -> int:
    """Count message arrow lines (->> or ->>) in a fenced Mermaid block."""
    return len(_ARROW_LINE_RE.findall(mermaid))


def count_vision_actors(mermaid: str) -> int:
    """Count ``participant`` declaration lines in a fenced Mermaid block."""
    return len(_PARTICIPANT_LINE_RE.findall(mermaid))


def _winner(  # noqa: PLR0911 — each branch returns a distinct sentinel value; extracting further would obscure intent
    row_prose_steps: int, row_vision_steps: int, prose_valid: str, vision_valid: str
) -> str:
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


def align_results(
    prose_flows: list[ProseCallFlow],  # noqa: F821 — TYPE_CHECKING-only import; safe with from __future__ import annotations
    vision_diagrams: list[ExtractedDiagram],  # noqa: F821 — TYPE_CHECKING-only import; safe with from __future__ import annotations
) -> list[ComparisonRow]:
    """Match prose flows to vision diagrams by caption substring; build ComparisonRow list.

    Matching strategy: a prose flow's title is matched to a vision diagram's caption
    if either string is a case-insensitive substring of the other. Unmatched entries
    appear as prose-only (vision_steps=0, vision_valid="—") or vision-only
    (figure_id="—", prose_steps=0) rows.
    """
    from specagent.retrieval.mermaid_validator import (  # noqa: PLC0415 — lazy import avoids importing specagent at module load time when running tests
        validate_mermaid,
    )

    matched_vision_indices: set[int] = set()
    rows: list[ComparisonRow] = []

    for flow in prose_flows:
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
            rows.append(
                ComparisonRow(
                    figure_id=flow.figure_id,
                    caption=flow.title,
                    prose_steps=len(flow.steps),
                    vision_steps=v_steps,
                    prose_actors=len(flow.participants),
                    vision_actors=v_actors,
                    prose_valid=prose_valid,
                    vision_valid=vision_valid,
                    winner=_winner(len(flow.steps), v_steps, prose_valid, vision_valid),
                )
            )
        else:
            rows.append(
                ComparisonRow(
                    figure_id=flow.figure_id,
                    caption=flow.title,
                    prose_steps=len(flow.steps),
                    vision_steps=0,
                    prose_actors=len(flow.participants),
                    vision_actors=0,
                    prose_valid=prose_valid,
                    vision_valid="—",
                    winner="—",
                )
            )

    for i, diag in enumerate(vision_diagrams):
        if i in matched_vision_indices:
            continue
        v_steps = count_vision_steps(diag.mermaid_content)
        v_actors = count_vision_actors(diag.mermaid_content)
        vision_valid = "✓" if validate_mermaid(diag.mermaid_content)[0] else "✗"
        rows.append(
            ComparisonRow(
                figure_id="—",
                caption=diag.caption or "(no caption)",
                prose_steps=0,
                vision_steps=v_steps,
                prose_actors=0,
                vision_actors=v_actors,
                prose_valid="✗",
                vision_valid=vision_valid,
                winner=_winner(0, v_steps, "✗", vision_valid),
            )
        )

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
        f"\n**Summary**: {len(rows)} figures | "
        f"prose wins: {prose_wins} | vision wins: {vision_wins} | "
        f"ties: {ties} | prose-only: {prose_only} | vision-only: {vision_only}"
    )

    return header + "\n".join(body_lines) + "\n" + summary


async def _run_vision(api_key: str) -> tuple[str, list]:
    """Run the two-pass OCR pipeline and return (enriched_markdown, diagrams)."""
    from specagent.retrieval.docx_ocr_converter import convert_docx_with_ocr  # noqa: PLC0415

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

    # Pass 1: MarkItDown → Markdown
    print(f"Converting {DOCX_PATH.name} with MarkItDown…", flush=True)
    from specagent.retrieval.converter import convert  # noqa: PLC0415

    markdown = convert(DOCX_PATH)

    # Prose extractor
    print("Running prose DAG extractor…", flush=True)
    from specagent.retrieval.prose_dag_extractor import extract_prose_call_flows  # noqa: PLC0415

    prose_flows = extract_prose_call_flows(markdown)
    print(f"  → {len(prose_flows)} figures with parseable steps found.", flush=True)

    # Groq vision (skipped in dry-run)
    vision_diagrams: list = []
    if args.dry_run:
        print("Dry-run: skipping Groq vision pass.", flush=True)
    else:
        print(f"Running Groq vision pipeline on {DOCX_PATH.name}…", flush=True)
        _enriched_md, vision_diagrams = asyncio.run(_run_vision(api_key))
        print(f"  → {len(vision_diagrams)} call-flow diagrams extracted.", flush=True)

    # Align and render
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
