"""Unit tests for specagent.kuzu.mermaid_parser.

Tests are written FIRST (TDD). Implementation does not exist yet — all tests
must fail with ImportError / ModuleNotFoundError before the module is created.

Covers:
- Participant / actor extraction (explicit declarations + implicit from arrows)
- Step extraction: from_actor, to_actor, message, step_index, is_async
- Edge cases: empty diagrams, comment lines, fenced Mermaid blocks, aliases
"""

from __future__ import annotations

import pytest

from specagent.kuzu.mermaid_parser import StepRecord, parse_sequence_diagram

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_DIAGRAM = """\
sequenceDiagram
    participant UE
    participant AMF
    UE->>AMF: Registration Request
    AMF-->>UE: Registration Accept
"""

_FENCED_DIAGRAM = """\
```mermaid
sequenceDiagram
    participant UE
    participant AMF
    UE->>AMF: Registration Request
    AMF-->>UE: Registration Accept
```"""

_MULTI_STEP_DIAGRAM = """\
sequenceDiagram
    participant UE
    participant gNB
    participant AMF
    participant AUSF
    UE->>gNB: RRC Setup Request
    gNB->>AMF: Initial UE Message
    AMF->>AUSF: Nausf Authentication Request
    AUSF-->>AMF: Nausf Authentication Response
    AMF-->>gNB: Initial Context Setup Request
    gNB-->>UE: RRC Setup Complete
"""


# ---------------------------------------------------------------------------
# Participant extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_explicit_participant_declarations() -> None:
    """Explicit 'participant X' lines are returned in participants list."""
    participants, _ = parse_sequence_diagram(_SIMPLE_DIAGRAM)

    assert "UE" in participants
    assert "AMF" in participants


@pytest.mark.unit
def test_parse_actor_keyword() -> None:
    """'actor X' keyword is treated identically to 'participant X'."""
    diagram = """\
sequenceDiagram
    actor UE
    actor AMF
    UE->>AMF: Registration Request
"""
    participants, _ = parse_sequence_diagram(diagram)

    assert "UE" in participants
    assert "AMF" in participants


@pytest.mark.unit
def test_participant_alias_uses_short_name() -> None:
    """'participant X as Long Name' → short name X is extracted."""
    diagram = """\
sequenceDiagram
    participant UE as User Equipment
    participant AMF as Access and Mobility Function
    UE->>AMF: Registration Request
"""
    participants, _ = parse_sequence_diagram(diagram)

    assert "UE" in participants
    assert "AMF" in participants
    assert "User Equipment" not in participants
    assert "Access and Mobility Function" not in participants


@pytest.mark.unit
def test_implicit_participants_extracted_from_arrows() -> None:
    """Actors referenced only in arrows (no explicit declaration) are collected."""
    diagram = """\
sequenceDiagram
    UE->>AMF: Registration Request
    AMF-->>UE: Registration Accept
"""
    participants, _ = parse_sequence_diagram(diagram)

    assert "UE" in participants
    assert "AMF" in participants


@pytest.mark.unit
def test_participants_deduplicated() -> None:
    """Each participant name appears exactly once even if referenced many times."""
    participants, _ = parse_sequence_diagram(_MULTI_STEP_DIAGRAM)

    assert participants.count("UE") == 1
    assert participants.count("AMF") == 1


@pytest.mark.unit
def test_returns_all_declared_participants() -> None:
    """All four declared participants are present in the result."""
    participants, _ = parse_sequence_diagram(_MULTI_STEP_DIAGRAM)

    for name in ("UE", "gNB", "AMF", "AUSF"):
        assert name in participants


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step_count_matches_arrow_lines() -> None:
    """Number of steps equals the number of arrow lines in the diagram."""
    _, steps = parse_sequence_diagram(_SIMPLE_DIAGRAM)

    assert len(steps) == 2


@pytest.mark.unit
def test_step_fields_populated_correctly() -> None:
    """First step has correct from_actor, to_actor, and message."""
    _, steps = parse_sequence_diagram(_SIMPLE_DIAGRAM)

    first = steps[0]
    assert first.from_actor == "UE"
    assert first.to_actor == "AMF"
    assert first.message == "Registration Request"


@pytest.mark.unit
def test_step_index_is_sequential() -> None:
    """step_index is zero-based and matches the order of arrows in the diagram."""
    _, steps = parse_sequence_diagram(_MULTI_STEP_DIAGRAM)

    for expected_idx, step in enumerate(steps):
        assert step.step_index == expected_idx


@pytest.mark.unit
def test_solid_arrow_is_not_async() -> None:
    """->> (solid arrowhead) → is_async=False."""
    diagram = """\
sequenceDiagram
    UE->>AMF: Registration Request
"""
    _, steps = parse_sequence_diagram(diagram)

    assert steps[0].is_async is False


@pytest.mark.unit
def test_dashed_arrow_is_async() -> None:
    """-->> (dashed arrowhead) → is_async=True."""
    diagram = """\
sequenceDiagram
    AMF-->>UE: Registration Accept
"""
    _, steps = parse_sequence_diagram(diagram)

    assert steps[0].is_async is True


@pytest.mark.unit
def test_all_dashed_variants_are_async() -> None:
    """All dashed arrow types (-->>, --x, --)) produce is_async=True."""
    diagram = """\
sequenceDiagram
    AMF-->>UE: Response
    UE--xAMF: Lost message
    gNB--)UE: Async notify
"""
    _, steps = parse_sequence_diagram(diagram)

    # At least the -->> step must be async
    async_steps = [s for s in steps if s.is_async]
    assert len(async_steps) >= 1
    # The -->> step specifically
    response_step = next(s for s in steps if s.message == "Response")
    assert response_step.is_async is True


@pytest.mark.unit
def test_multi_step_returns_correct_count() -> None:
    """Six arrow lines produce six StepRecords."""
    _, steps = parse_sequence_diagram(_MULTI_STEP_DIAGRAM)

    assert len(steps) == 6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_diagram_returns_empty_steps() -> None:
    """A diagram with only the header line produces no steps."""
    diagram = "sequenceDiagram\n"
    _participants, steps = parse_sequence_diagram(diagram)

    assert steps == []


@pytest.mark.unit
def test_comment_lines_are_skipped() -> None:
    """Lines starting with %% are not counted as steps or participants."""
    diagram = """\
sequenceDiagram
    %% This is a comment about the call flow
    participant UE
    UE->>AMF: Registration Request
    %% Another comment
"""
    _, steps = parse_sequence_diagram(diagram)

    assert len(steps) == 1
    assert steps[0].message == "Registration Request"


@pytest.mark.unit
def test_fenced_mermaid_block_is_handled() -> None:
    """Input wrapped in ```mermaid ... ``` fences is parsed identically."""
    participants_fenced, steps_fenced = parse_sequence_diagram(_FENCED_DIAGRAM)
    participants_plain, steps_plain = parse_sequence_diagram(_SIMPLE_DIAGRAM)

    assert set(participants_fenced) == set(participants_plain)
    assert len(steps_fenced) == len(steps_plain)


@pytest.mark.unit
def test_note_lines_do_not_produce_steps() -> None:
    """'Note over ...' lines are not treated as message steps."""
    diagram = """\
sequenceDiagram
    participant UE
    participant AMF
    UE->>AMF: Registration Request
    Note over UE,AMF: 3GPP TS 23.502
    AMF-->>UE: Registration Accept
"""
    _, steps = parse_sequence_diagram(diagram)

    assert len(steps) == 2
    messages = [s.message for s in steps]
    assert "Registration Request" in messages
    assert "Registration Accept" in messages


@pytest.mark.unit
def test_returns_steprecord_instances() -> None:
    """parse_sequence_diagram returns StepRecord dataclass instances."""
    _, steps = parse_sequence_diagram(_SIMPLE_DIAGRAM)

    assert all(isinstance(s, StepRecord) for s in steps)


@pytest.mark.unit
def test_returns_list_of_strings_for_participants() -> None:
    """parse_sequence_diagram returns a list[str] for participants."""
    participants, _ = parse_sequence_diagram(_SIMPLE_DIAGRAM)

    assert isinstance(participants, list)
    assert all(isinstance(p, str) for p in participants)
