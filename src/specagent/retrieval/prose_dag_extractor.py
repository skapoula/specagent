"""Extract call-flow DAGs from 3GPP spec Markdown prose.

3GPP specifications describe call flows as:
  - A ``Figure X.Y-Z: Title`` label marking the start of a procedure
  - Numbered steps of the form ``N. Actor to Actor: message``
  - Sub-steps labelled ``Na.``, ``Nb.``, etc.

This module scans postprocessed Markdown for these patterns and builds
:class:`ProseCallFlow` objects with structured steps and a generated
Mermaid ``sequenceDiagram`` block — usable by the Kuzu DAG store without
any LLM involvement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from specagent.memgraph.mermaid_parser import StepRecord

# Figure label:  "Figure 4.2.2.2.2-1: Registration procedure"
_FIGURE_RE = re.compile(
    r"^Figure\s+([\d.]+(?:-\d+)?)\s*:\s*(.+)$",
    re.MULTILINE,
)

# Numbered step: "1. X to Y: message"  or  "3a. X to Y: message"
_STEP_RE = re.compile(
    r"^\d+[a-z]?\.\s+"  # step label
    r"(.+?)"  # from_actor (lazy, stops before " to ")
    r"\s+to\s+"  # separator
    r"(.+?)"  # to_actor (lazy, stops before ":")
    r"\s*:\s*"
    r"(.+)$",  # message
    re.IGNORECASE,
)

# Detect response / return steps: from_actor contains a service name
# and to_actor is upstream (heuristic: message starts with known patterns)
_RESPONSE_KEYWORDS = re.compile(
    r"(?:response|ack|accept|reject|confirm|complete|result|answer)\b",
    re.IGNORECASE,
)

# Actors to normalise: strip parenthetical "(R)AN" → "RAN" and bracket "[Conditional] AMF" → "AMF"
_PAREN_RE = re.compile(r"\([^)]*\)")
_BRACKET_PREFIX_RE = re.compile(r"^\[[^\]]*\]\s*")

# Max lines to scan after a Figure label before giving up on finding steps
_LOOKAHEAD_LINES = 200


@dataclass
class ProseCallFlow:
    """A call-flow procedure extracted from 3GPP Markdown prose."""

    figure_id: str
    """Numeric figure ID, e.g. ``"4.2.2.2.2-1"``."""

    title: str
    """Human-readable title from the Figure label."""

    steps: list[StepRecord]
    """Ordered message steps parsed from the numbered step list."""

    participants: list[str]
    """Deduplicated participant names in first-appearance order."""

    mermaid_content: str
    """Generated Mermaid ``sequenceDiagram`` fenced block."""


def extract_prose_call_flows(markdown: str) -> list[ProseCallFlow]:
    """Extract all call-flow procedures from postprocessed 3GPP Markdown.

    Scans for ``Figure X: Title`` labels followed by numbered step lines
    of the form ``N. Actor to Actor: message`` within a lookahead window.
    Figures with no parseable steps are skipped.

    Args:
        markdown: Postprocessed Markdown string (output of ``postprocess()``).

    Returns:
        List of :class:`ProseCallFlow`, one per Figure with at least one step.
    """
    lines = markdown.splitlines()
    figure_positions = _find_figures(lines)
    results: list[ProseCallFlow] = []

    for idx, (line_no, figure_id, title) in enumerate(figure_positions):
        end = figure_positions[idx + 1][0] if idx + 1 < len(figure_positions) else len(lines)
        window_end = min(line_no + _LOOKAHEAD_LINES, end)
        steps = _parse_steps(lines[line_no + 1 : window_end])
        if not steps:
            continue
        participants = _collect_participants(steps)
        mermaid = _build_mermaid(participants, steps)
        results.append(
            ProseCallFlow(
                figure_id=figure_id,
                title=title,
                steps=steps,
                participants=participants,
                mermaid_content=mermaid,
            )
        )

    return results


def _find_figures(lines: list[str]) -> list[tuple[int, str, str]]:
    """Return (line_index, figure_id, title) for every Figure label line."""
    figures: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        m = _FIGURE_RE.match(line.strip())
        if m:
            figures.append((i, m.group(1), m.group(2).strip()))
    return figures


def _parse_steps(lines: list[str]) -> list[StepRecord]:
    """Parse numbered step lines into :class:`StepRecord` objects."""
    steps: list[StepRecord] = []
    for line in lines:
        m = _STEP_RE.match(line.strip())
        if not m:
            continue
        from_actor = _normalise_actor(m.group(1))
        to_actor = _normalise_actor(m.group(2))
        message = m.group(3).strip()[:200]
        is_async = bool(_RESPONSE_KEYWORDS.search(message))
        steps.append(
            StepRecord(
                step_index=len(steps),
                from_actor=from_actor,
                to_actor=to_actor,
                message=message,
                is_async=is_async,
            )
        )
    return steps


def _normalise_actor(raw: str) -> str:
    """Strip bracket/paren qualifiers and trim whitespace from an actor name."""
    cleaned = _BRACKET_PREFIX_RE.sub("", raw.strip())
    cleaned = _PAREN_RE.sub("", cleaned).strip()
    return cleaned[:60] if cleaned else raw.strip()[:60]


def _collect_participants(steps: list[StepRecord]) -> list[str]:
    """Return deduplicated participant names in first-appearance order."""
    seen: set[str] = set()
    participants: list[str] = []
    for step in steps:
        for actor in (step.from_actor, step.to_actor):
            if actor and actor not in seen:
                seen.add(actor)
                participants.append(actor)
    return participants


def _build_mermaid(participants: list[str], steps: list[StepRecord]) -> str:
    """Build a fenced Mermaid sequenceDiagram block from participants and steps."""
    lines: list[str] = ["```mermaid", "sequenceDiagram"]
    for p in participants:
        lines.append(f"    participant {p}")
    for step in steps:
        arrow = "-->>" if step.is_async else "->>"
        lines.append(f"    {step.from_actor}{arrow}{step.to_actor}: {step.message}")
    lines.append("```")
    return "\n".join(lines)


__all__ = ["ProseCallFlow", "extract_prose_call_flows"]
