"""Parse validated Mermaid sequenceDiagram blocks into structured data.

Extracts participants and ordered message steps from a ``sequenceDiagram``
Mermaid block, whether provided as raw inner content or wrapped in a fenced
code block (```mermaid ... ```).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class StepRecord:
    """A single message step in a sequence diagram."""

    step_index: int
    """Zero-based sequential position of this step in the diagram."""

    from_actor: str
    """Name of the actor sending the message."""

    to_actor: str
    """Name of the actor receiving the message."""

    message: str
    """Message label as written in the diagram."""

    is_async: bool
    """True for dashed arrows (-->>, --x, --)) which denote responses / async messages."""


# Matches fenced ```mermaid ... ``` blocks; captures inner content.
_FENCE_RE = re.compile(r"```mermaid\s*\n([\s\S]*?)```", re.IGNORECASE)

# Matches participant / actor declarations, with optional alias.
# Examples:
#   participant UE
#   participant UE as User Equipment
#   actor AMF
_PARTICIPANT_RE = re.compile(
    r"^\s*(?:participant|actor)\s+(\S+)(?:\s+as\s+.+)?$",
    re.IGNORECASE,
)

# Matches message arrows between actors.
# Captures: from_actor, arrow, to_actor, message
# Arrow types:
#   -->> ->> --> -> --x -x --) -)
_ARROW_RE = re.compile(
    r"^\s*"
    r"([A-Za-z0-9_][A-Za-z0-9_ \-]*?)"   # from_actor (no leading spaces after strip)
    r"\s*"
    r"(-->>|--x|-->|->>|->|-x|--\)|--|-\))"  # arrow type
    r"\s*"
    r"([A-Za-z0-9_][A-Za-z0-9_ \-]*?)"   # to_actor
    r"\s*:\s*"
    r"(.+)$",                              # message
)

# Dashed arrow prefixes → is_async=True
_ASYNC_ARROWS = frozenset(["-->>", "--x", "-->", "--)", "--"])


def parse_sequence_diagram(
    mermaid_content: str,
) -> tuple[list[str], list[StepRecord]]:
    """Parse a Mermaid sequenceDiagram block into participants and steps.

    Handles both raw inner content and fenced (```mermaid ... ```) blocks.
    Ignores comment lines (``%%``), ``Note`` lines, and block keywords
    (``loop``, ``alt``, ``else``, ``end``, ``activate``, ``deactivate``).

    Args:
        mermaid_content: Mermaid sequenceDiagram string, fenced or unfenced.

    Returns:
        Tuple of ``(participants, steps)`` where:
        - ``participants`` is a deduplicated list of actor/participant names
          in declaration order (implicit actors from arrows appended at the end).
        - ``steps`` is an ordered list of :class:`StepRecord` instances.
    """
    inner = _extract_inner(mermaid_content)
    lines = inner.splitlines()

    declared: list[str] = []          # participants in declaration order
    declared_set: set[str] = set()
    steps: list[StepRecord] = []
    step_index = 0

    for line in lines:
        stripped = line.strip()

        # Skip empty lines, comments, the header keyword, and structural keywords
        if not stripped:
            continue
        if stripped.startswith("%%"):
            continue
        if stripped.lower().startswith("sequencediagram"):
            continue
        if re.match(r"^(note|loop|alt|else|opt|par|critical|break|end|activate|deactivate)\b",
                    stripped, re.IGNORECASE):
            continue

        # Participant / actor declaration
        m = _PARTICIPANT_RE.match(stripped)
        if m:
            name = m.group(1)
            if name not in declared_set:
                declared.append(name)
                declared_set.add(name)
            continue

        # Message arrow
        m = _ARROW_RE.match(stripped)
        if m:
            from_actor = m.group(1).strip()
            arrow = m.group(2)
            to_actor = m.group(3).strip()
            message = m.group(4).strip()
            is_async = arrow in _ASYNC_ARROWS

            steps.append(StepRecord(
                step_index=step_index,
                from_actor=from_actor,
                to_actor=to_actor,
                message=message,
                is_async=is_async,
            ))
            step_index += 1

            # Collect implicit participants from arrows
            for actor in (from_actor, to_actor):
                if actor not in declared_set:
                    declared.append(actor)
                    declared_set.add(actor)

    return declared, steps


def _extract_inner(content: str) -> str:
    """Strip a ```mermaid ... ``` fence if present; otherwise return as-is."""
    m = _FENCE_RE.search(content)
    if m:
        return m.group(1)
    return content
