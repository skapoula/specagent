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

from specagent.kuzu.mermaid_parser import StepRecord

# Figure label:  "Figure 4.2.2.2.2-1: Registration procedure"
_FIGURE_RE = re.compile(
    r"^Figure\s+([\d.]+(?:-\d+)?)\s*:\s*(.+)$",
    re.MULTILINE,
)

# _STEP_RE is defined after _NF_NAMES (below) because it references the NF whitelist.

# Known 3GPP NF/entity names for strict to_actor matching in sends-form steps.
# These are the actors that legitimately appear as message recipients.
_NF_NAMES = (
    r"UE|AMF|SMF|UPF|UDM|AUSF|PCF|NSSF|NRF|NEF|AF|SMSF|LMF|GMLC|"
    r"N3IWF|TNGF|W-AGF|EIR|CHF|SEPP|UDR|NWDAF|NSACF|NSSAAF|AAA-S|AAA-P|"
    r"5GC|DN|SCEF|MBSF|MBSTF|5G-RG|W-5GAN|FN-RG|"
    r"NG-RAN|RAN|gNB|ng-eNB|eNB|5G-AN|AN|HSS|EPC|MME|SGW|PGW|"
    r"initial AMF|target AMF|old AMF|new AMF|serving AMF|"
    r"source gNB|target gNB|serving gNB|"
    r"V-SMF|H-SMF|V-NEF|hNRF|vNRF|I-SMF"
)

# "N. Actor to Actor: message" — both actor slots anchored to NF whitelist.
# Allows optional qualifier prefix (e.g. "new AMF", "[Conditional] old AMF")
# and optional paren suffix (e.g. "(R)AN"). Full actor captured including qualifier.
_STEP_RE = re.compile(
    r"^\d+[a-z]?(?:\([A-Z]\))?\.\s+"  # step label: 1. / 3a. / 7(A).
    r"(?:\[.*?\]\s*)?"  # optional [Conditional]
    r"(?:From\s+)?"  # optional "From" prefix (e.g. "From UE")
    r"((?:(?:new|old|initial|target|serving|source|intermediate)[\s-])*(?:\([^)]*\))?(?:"
    + _NF_NAMES
    + r")\s*(?:\([^)\s][^)]*\))?)"  # from_actor: qualifiers* + paren? + NF + space-paren?
    r"\s+to\s+(?:the\s+)?"  # "to [the]"
    r"((?:(?:new|old|initial|target|serving|source|intermediate)[\s-])*(?:\([^)]*\))?(?:"
    + _NF_NAMES
    + r")\s*(?:\([^)\s][^)]*\))?)"  # to_actor: qualifiers* + paren? + NF + space-paren?
    r"\s*:\s*"
    r"(.+)$",  # message
    re.IGNORECASE,
)

# "The X sends [NAS message] Y to [the] Z" — both X and Z must be known NF names.
# Actor slots allow an optional qualifier word (e.g. "new AMF", "source gNB") but
# not arbitrary prose — anchored to the NF whitelist to prevent greedy over-matching.
_SENDS_RE = re.compile(
    r"^\d+[a-z]?(?:\([A-Z]\))?\.\s+"  # step label
    r"(?:\[.*?\]\s*)?"  # optional [Conditional]
    r"(?:The\s+)?"  # optional "The"
    r"((?:new|old|initial|target|serving|source|v|h|i)[\s-])?(?P<from>(?:"
    + _NF_NAMES
    + r"))"  # from_actor
    r"\s+sends?\s+"
    r"(?:NAS message\s+|N\d+\s+message\s+)?"  # optional message type qualifier
    r"(.+?)"  # message payload (lazy)
    r"\s+to\s+(?:the\s+)?"  # "to [the]"
    r"((?:new|old|initial|target|serving|source|v|h|i)[\s-])?(?P<to>(?:"
    + _NF_NAMES
    + r"))"  # to_actor
    r"(?:[.,\s(/].*)?$",
    re.IGNORECASE,
)

# Verb-form steps: handles both "ACTOR verb MSG to ACTOR" and "ACTOR verb ACTOR with MSG".
_VERB_ACTIONS = r"(?:contacts?|invokes?|forwards?|relays?|triggers?|notifies?)"

_VERB_RE = re.compile(
    r"^\d+[a-z]?(?:\([A-Z]\))?\.\s+"  # step label
    r"(?:\[.*?\]\s*)?"  # optional [Conditional]
    r"(?:The\s+)?"  # optional "The"
    r"(?P<from>(?:(?:new|old|initial|target|serving|source|intermediate)[\s-])*(?:"
    + _NF_NAMES
    + r"))"  # from_actor
    r"\s+(?:then\s+)?"  # optional "then"
     + _VERB_ACTIONS + r"\s+"
    r"(?:"
    # ordering A: MSG {to|on|at} [the] ACTOR
    r"(?P<msgA>.+?)\s+(?:to|on|at)\s+(?:the\s+)?(?P<toA>(?:(?:new|old|initial|target|serving|source|intermediate)[\s-])*(?:"
    + _NF_NAMES
    + r"))"
    r"|"
    # ordering B: ACTOR {with|on} MSG
    r"(?P<toB>(?:(?:new|old|initial|target|serving|source|intermediate)[\s-])*(?:"
    + _NF_NAMES
    + r"))\s+(?:with|on)\s+(?P<msgB>.+?)"
    r")"
    r"(?:[.,\s(/].*)?$",
    re.IGNORECASE,
)

# "N. [Conditional] sends PAYLOAD to [the] ACTOR" — from_actor inferred from last_to_actor.
_IMPLICIT_STEP_RE = re.compile(
    r"^\d+[a-z]?(?:\([A-Z]\))?\.\s+"  # step label
    r"(?:\[.*?\]\s*)?"  # optional [Conditional]
    r"sends?\s+"  # verb with no subject
    r"(?:NAS message\s+|N\d+\s+message\s+)?"  # optional message type qualifier
    r"(.+?)"  # message payload (lazy)
    r"\s+to\s+(?:the\s+)?"  # "to [the]"
    r"((?:new|old|initial|target|serving|source|v|h|i)[\s-])?(?P<to>(?:"
    + _NF_NAMES
    + r"))"  # to_actor
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

    This is a **synchronous, CPU-bound** function (regex-heavy parsing).
    Async callers must not ``await`` it — dispatch via ``asyncio.to_thread``::

        flows = await asyncio.to_thread(extract_prose_call_flows, markdown)

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


def _match_step_line(stripped: str, last_to_actor: str | None) -> tuple[str, str, str] | None:
    """Try each pattern in priority order; return (from_actor, to_actor, message) or None."""
    m = _STEP_RE.match(stripped)
    if m:
        return _normalise_actor(m.group(1)), _normalise_actor(m.group(2)), m.group(3).strip()[:200]

    m2 = _SENDS_RE.match(stripped)
    if m2:
        from_actor = _normalise_actor(((m2.group(1) or "") + m2.group("from")).strip())
        to_actor = _normalise_actor(((m2.group(4) or "") + m2.group("to")).strip())
        return from_actor, to_actor, m2.group(3).strip()[:200]

    m3 = _VERB_RE.match(stripped)
    if m3:
        from_actor = _normalise_actor(m3.group("from"))
        if m3.group("toA") is not None:
            return (
                from_actor,
                _normalise_actor(m3.group("toA")),
                (m3.group("msgA") or "").strip()[:200],
            )
        return from_actor, _normalise_actor(m3.group("toB")), (m3.group("msgB") or "").strip()[:200]

    if last_to_actor:
        m4 = _IMPLICIT_STEP_RE.match(stripped)
        if m4:
            to_actor = _normalise_actor(((m4.group(2) or "") + m4.group("to")).strip())
            return last_to_actor, to_actor, m4.group(1).strip()[:200]

    return None


def _parse_steps(lines: list[str]) -> list[StepRecord]:
    """Parse numbered step lines into :class:`StepRecord` objects."""
    steps: list[StepRecord] = []
    last_to_actor: str | None = None

    for line in lines:
        result = _match_step_line(line.strip(), last_to_actor)
        if result is None:
            continue
        from_actor, to_actor, message = result
        is_async = bool(_RESPONSE_KEYWORDS.search(message))
        last_to_actor = to_actor
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
