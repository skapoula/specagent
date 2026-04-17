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

# Numbered step: "1. X to Y: message", "3a. X to Y: msg", "7(A). X to Y: msg"
_STEP_RE = re.compile(
    r"^\d+[a-z]?(?:\([A-Z]\))?\.\s+"  # step label: 1. / 3a. / 7(A).
    r"(.+?)"  # from_actor (lazy)
    r"\s+to\s+"
    r"(.+?)"  # to_actor (lazy)
    r"\s*:\s*"
    r"(.+)$",  # message
    re.IGNORECASE,
)

# Known 3GPP NF/entity names for strict to_actor matching in sends-form steps.
# These are the actors that legitimately appear as message recipients.
_NF_NAMES = (
    r"UE|AMF|SMF|UPF|UDM|AUSF|PCF|NSSF|NRF|NEF|AF|SMSF|LMF|GMLC|"
    r"N3IWF|TNGF|W-AGF|EIR|CHF|SEPP|UDR|NWDAF|NSACF|NSSAAF|AAA-S|AAA-P|"
    r"NG-RAN|RAN|gNB|ng-eNB|eNB|5G-AN|AN|HSS|EPC|MME|SGW|PGW|"
    r"initial AMF|target AMF|old AMF|new AMF|serving AMF|"
    r"V-SMF|H-SMF|V-NEF|hNRF|vNRF|I-SMF"
)

# "The X sends [NAS message] Y to [the] Z" — both X and Z must be known NF names.
# Actor slots allow an optional qualifier word (e.g. "new AMF", "V-SMF") but not
# arbitrary prose — capped at 3 words total to prevent greedy over-matching.
_SENDS_RE = re.compile(
    r"^\d+[a-z]?(?:\([A-Z]\))?\.\s+"  # step label
    r"(?:\[.*?\]\s*)?"  # optional [Conditional]
    r"(?:The\s+)?"  # optional "The"
    r"((?:(?:new|old|initial|target|serving|v|h|i)-?\s*)?(?:" + _NF_NAMES + r"))"  # from_actor
    r"\s+sends?\s+"
    r"(?:NAS message\s+|N\d+\s+message\s+)?"  # optional message type qualifier
    r"(.+?)"  # message payload (lazy)
    r"\s+to\s+(?:the\s+)?"  # "to [the]"
    r"((?:(?:new|old|initial|target|serving|v|h|i)-?\s*)?(?:" + _NF_NAMES + r"))"  # to_actor
    r"(?:[.,\s(/].*)?$",
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
        stripped = line.strip()
        m = _STEP_RE.match(stripped)
        if m:
            from_actor = _normalise_actor(m.group(1))
            to_actor = _normalise_actor(m.group(2))
            message = m.group(3).strip()[:200]
        else:
            m2 = _SENDS_RE.match(stripped)
            if not m2:
                continue
            from_actor = _normalise_actor(m2.group(1))
            message = m2.group(2).strip()[:200]
            to_actor = _normalise_actor(m2.group(3))
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
