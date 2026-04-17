"""Unit tests for the prose call-flow DAG extractor.

Covers extraction of Figure-labelled call-flow procedures from 3GPP spec
Markdown (prose form: ``Figure X: Title`` followed by numbered steps
``N. Actor to Actor: message``).

All tests use synthetic Markdown snippets — no file I/O, no network.
"""

from __future__ import annotations

import pytest

from specagent.retrieval.prose_dag_extractor import (
    ProseCallFlow,
    extract_prose_call_flows,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SIMPLE_FLOW = """\
## 4.2.2.2.2 Registration

Some introductory text about the registration procedure.

Figure 4.2.2.2.2-1: Registration procedure

1. UE to (R)AN: Registration Request (SUCI, Registration type)

2. (R)AN to AMF: N2 message (Registration Request)

3. AMF to UDM: Nudm_UECM_Registration (SUPI)

4. UDM to AMF: Nudm_UECM_Registration Response

## 4.2.2.2.3 Next section
"""

_TWO_FLOWS = """\
Figure 4.2.2.2.2-1: Registration procedure

1. UE to AMF: Registration Request

2. AMF to UDM: Nudm_UECM_Registration

Figure 4.2.2.3.2-1: UE-initiated Deregistration

1. UE to AMF: Deregistration Request

2. AMF to SMF: Nsmf_PDUSession_ReleaseSMContext
"""

_ALPHA_STEPS = """\
Figure 4.2.2.2.3-1: Registration with AMF re-allocation

1. UE to (R)AN: Registration Request

2. (R)AN to initial AMF: N2 message

3a. initial AMF to UDM: Nudm_Authentication_Get

3b. UDM to initial AMF: Nudm_Authentication_Get Response

4. initial AMF to target AMF: Namf_Communication_UEContextTransfer
"""

_NO_FIGURES = """\
## 4.1 Scope

This clause describes the general architecture.

No procedure diagrams here.
"""

_FIGURE_NO_STEPS = """\
Figure 5.1.1-1: Architecture overview

This figure shows the general architecture.
The next figure shows procedures.

Figure 5.1.2-1: Registration procedure

1. UE to AMF: Registration Request

2. AMF to UDM: Nudm_UECM_Registration
"""


# ---------------------------------------------------------------------------
# ProseCallFlow dataclass
# ---------------------------------------------------------------------------


class TestProseCallFlow:
    """ProseCallFlow structure and field types."""

    @pytest.mark.unit
    def test_has_required_fields(self) -> None:
        """ProseCallFlow exposes title, figure_id, steps, participants, mermaid_content."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert len(flows) >= 1
        flow = flows[0]
        assert isinstance(flow, ProseCallFlow)
        assert hasattr(flow, "title")
        assert hasattr(flow, "figure_id")
        assert hasattr(flow, "steps")
        assert hasattr(flow, "participants")
        assert hasattr(flow, "mermaid_content")

    @pytest.mark.unit
    def test_title_extracted_correctly(self) -> None:
        """title matches the text after the colon in the Figure label."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert flows[0].title == "Registration procedure"

    @pytest.mark.unit
    def test_figure_id_extracted_correctly(self) -> None:
        """figure_id matches the numeric ID before the colon."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert flows[0].figure_id == "4.2.2.2.2-1"


# ---------------------------------------------------------------------------
# extract_prose_call_flows — basic extraction
# ---------------------------------------------------------------------------


class TestExtractProseCallFlows:
    """Core extraction behaviour."""

    @pytest.mark.unit
    def test_returns_empty_list_for_no_figures(self) -> None:
        """Returns [] when markdown has no Figure labels."""
        assert extract_prose_call_flows(_NO_FIGURES) == []

    @pytest.mark.unit
    def test_extracts_single_flow(self) -> None:
        """Returns one ProseCallFlow for a single Figure with steps."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert len(flows) == 1

    @pytest.mark.unit
    def test_extracts_two_flows(self) -> None:
        """Returns two ProseCallFlows when two Figures are present."""
        flows = extract_prose_call_flows(_TWO_FLOWS)
        assert len(flows) == 2

    @pytest.mark.unit
    def test_figure_without_steps_is_skipped(self) -> None:
        """A Figure label followed by no numbered steps is not returned."""
        flows = extract_prose_call_flows(_FIGURE_NO_STEPS)
        # Only the second figure has steps
        assert len(flows) == 1
        assert flows[0].title == "Registration procedure"

    @pytest.mark.unit
    def test_alpha_sub_steps_are_extracted(self) -> None:
        """Steps labelled 3a, 3b are extracted as separate steps."""
        flows = extract_prose_call_flows(_ALPHA_STEPS)
        assert len(flows) == 1
        assert len(flows[0].steps) == 5  # 1, 2, 3a, 3b, 4


# ---------------------------------------------------------------------------
# Step parsing
# ---------------------------------------------------------------------------


class TestStepParsing:
    """Individual step field extraction."""

    @pytest.mark.unit
    def test_step_count_matches_numbered_lines(self) -> None:
        """steps list length equals number of numbered step lines."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert len(flows[0].steps) == 4

    @pytest.mark.unit
    def test_step_from_actor_extracted(self) -> None:
        """from_actor of first step is 'UE'."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert flows[0].steps[0].from_actor == "UE"

    @pytest.mark.unit
    def test_step_to_actor_extracted(self) -> None:
        """to_actor of first step contains 'AN'."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert "AN" in flows[0].steps[0].to_actor

    @pytest.mark.unit
    def test_step_message_extracted(self) -> None:
        """message contains the content after the colon."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert "Registration Request" in flows[0].steps[0].message

    @pytest.mark.unit
    def test_step_index_is_zero_based(self) -> None:
        """step_index starts at 0 and increments sequentially."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        for i, step in enumerate(flows[0].steps):
            assert step.step_index == i

    @pytest.mark.unit
    def test_response_step_is_async(self) -> None:
        """A step from UDM back to AMF (response direction) is marked is_async=True."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        # Step index 3: UDM to AMF (response)
        response_step = flows[0].steps[3]
        assert response_step.from_actor == "UDM"
        assert response_step.to_actor == "AMF"
        assert response_step.is_async is True


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------


class TestParticipants:
    """Participant extraction from steps."""

    @pytest.mark.unit
    def test_participants_include_ue_and_amf(self) -> None:
        """Participants list includes actors extracted from steps."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        participants = flows[0].participants
        assert "UE" in participants
        assert "AMF" in participants

    @pytest.mark.unit
    def test_participants_deduplicated(self) -> None:
        """Each participant name appears at most once."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        participants = flows[0].participants
        assert len(participants) == len(set(participants))


# ---------------------------------------------------------------------------
# Mermaid output
# ---------------------------------------------------------------------------


class TestMermaidOutput:
    """Generated Mermaid sequenceDiagram content."""

    @pytest.mark.unit
    def test_mermaid_content_starts_with_fence(self) -> None:
        """mermaid_content starts with ```mermaid."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert flows[0].mermaid_content.startswith("```mermaid")

    @pytest.mark.unit
    def test_mermaid_content_ends_with_fence(self) -> None:
        """mermaid_content ends with closing ```."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert flows[0].mermaid_content.strip().endswith("```")

    @pytest.mark.unit
    def test_mermaid_contains_sequencediagram(self) -> None:
        """mermaid_content contains the sequenceDiagram keyword."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        assert "sequenceDiagram" in flows[0].mermaid_content

    @pytest.mark.unit
    def test_mermaid_contains_actors(self) -> None:
        """mermaid_content contains participant declarations."""
        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        content = flows[0].mermaid_content
        assert "UE" in content
        assert "AMF" in content

    @pytest.mark.unit
    def test_mermaid_parseable_by_parse_sequence_diagram(self) -> None:
        """Generated Mermaid content can be parsed by the existing parser."""
        from specagent.memgraph.mermaid_parser import parse_sequence_diagram

        flows = extract_prose_call_flows(_SIMPLE_FLOW)
        participants, steps = parse_sequence_diagram(flows[0].mermaid_content)
        assert len(participants) > 0
        assert len(steps) > 0


# ---------------------------------------------------------------------------
# 3GPP-specific step patterns
# ---------------------------------------------------------------------------

# Pattern: [Conditional] prefix before Actor to Actor
_CONDITIONAL_FLOW = """\
Figure 4.2.2.2.2-1: Registration procedure

1. UE to AMF: Registration Request

2. [Conditional] new AMF to old AMF: Namf_Communication_UEContextTransfer (complete Registration Request)

3. [Conditional] old AMF to new AMF: Response to Namf_Communication_UEContextTransfer (SUPI, UE Context)
"""

# Pattern: "The X sends Y to the Z" narrative form
_SENDS_FLOW = """\
Figure 4.2.2.3.2-1: UE-initiated Deregistration

1. The UE sends NAS message Deregistration Request to the AMF

2. [Conditional] AMF to SMF: Nsmf_PDUSession_ReleaseSMContext (SM Context ID)

3. The SMF sends N4 Session Release Request to the UPF
"""

# Pattern: parenthetical step labels 7(A), 7(B)
_PAREN_LABEL_FLOW = """\
Figure 4.2.2.2.3-1: Registration with AMF re-allocation procedure

1. UE to initial AMF: Registration Request

7(A). initial AMF to target AMF: Namf_Communication_CreateUEContext Request

7(B). target AMF to initial AMF: Namf_Communication_CreateUEContext Response
"""

# Pattern: "X responds with Y" / "X responds to Y with Z"
_RESPONDS_FLOW = """\
Figure 4.3.2.2.1-1: PDU Session Establishment

1. UE to AMF: PDU Session Establishment Request

2. AMF to SMF: Nsmf_PDUSession_CreateSMContext Request

3. [Conditional] The SMF responds with Nsmf_PDUSession_CreateSMContext Response to the AMF
"""

# Pattern: range steps "6-7. Skipped" should be ignored
_RANGE_SKIPPED_FLOW = """\
Figure 4.2.2.2.4-1: UE Registration with ON-SNPN

1. UE to AMF: Registration Request

6-7. Skipped.

8. AMF to UDM: Nudm_UECM_Registration
"""

# Pattern: "X invokes Nxxx_Service on Y" — implicit message
_INVOKES_FLOW = """\
Figure 4.2.2.2.2-1: Registration procedure

1. UE to AMF: Registration Request

12. new AMF invokes N5g-eir_EquipmentIdentityCheck_Get on EIR
"""


class TestConditionalPrefix:
    """[Conditional] prefix before Actor to Actor is handled."""

    @pytest.mark.unit
    def test_conditional_step_is_extracted(self) -> None:
        """[Conditional] Actor to Actor: msg is extracted as a normal step."""
        flows = extract_prose_call_flows(_CONDITIONAL_FLOW)
        assert len(flows) == 1
        assert len(flows[0].steps) == 3

    @pytest.mark.unit
    def test_conditional_actors_correct(self) -> None:
        """from_actor and to_actor are extracted without the [Conditional] prefix."""
        flows = extract_prose_call_flows(_CONDITIONAL_FLOW)
        step = flows[0].steps[1]  # step index 1: new AMF to old AMF
        assert "AMF" in step.from_actor
        assert "AMF" in step.to_actor

    @pytest.mark.unit
    def test_conditional_message_extracted(self) -> None:
        """Message content is preserved after [Conditional] prefix is stripped."""
        flows = extract_prose_call_flows(_CONDITIONAL_FLOW)
        step = flows[0].steps[1]
        assert "Namf_Communication_UEContextTransfer" in step.message


class TestSendsPattern:
    """'The X sends Y to the Z' narrative steps are extracted."""

    @pytest.mark.unit
    def test_sends_step_is_extracted(self) -> None:
        """'The UE sends ... to the AMF' is extracted as a step."""
        flows = extract_prose_call_flows(_SENDS_FLOW)
        assert len(flows) == 1
        # Steps 1 and 3 are sends-form; step 2 is actor-to-actor
        assert len(flows[0].steps) == 3

    @pytest.mark.unit
    def test_sends_from_actor_correct(self) -> None:
        """from_actor of a sends-form step is the subject of 'sends'."""
        flows = extract_prose_call_flows(_SENDS_FLOW)
        step = flows[0].steps[0]
        assert step.from_actor == "UE"

    @pytest.mark.unit
    def test_sends_to_actor_correct(self) -> None:
        """to_actor of a sends-form step is the object of 'to'."""
        flows = extract_prose_call_flows(_SENDS_FLOW)
        step = flows[0].steps[0]
        assert step.to_actor == "AMF"

    @pytest.mark.unit
    def test_sends_message_extracted(self) -> None:
        """Message is the payload of the sends-form step."""
        flows = extract_prose_call_flows(_SENDS_FLOW)
        step = flows[0].steps[0]
        assert "Deregistration Request" in step.message


class TestParenStepLabels:
    """Parenthetical step labels like 7(A), 7(B) are matched."""

    @pytest.mark.unit
    def test_paren_label_steps_extracted(self) -> None:
        """Steps labelled 7(A) and 7(B) are extracted."""
        flows = extract_prose_call_flows(_PAREN_LABEL_FLOW)
        assert len(flows) == 1
        assert len(flows[0].steps) == 3

    @pytest.mark.unit
    def test_paren_label_actors_correct(self) -> None:
        """Actors from 7(A) step are extracted correctly."""
        flows = extract_prose_call_flows(_PAREN_LABEL_FLOW)
        step_7a = flows[0].steps[1]
        assert "AMF" in step_7a.from_actor
        assert "AMF" in step_7a.to_actor


class TestRangeSkipped:
    """Range steps like '6-7. Skipped' are silently ignored."""

    @pytest.mark.unit
    def test_range_skipped_not_counted(self) -> None:
        """'6-7. Skipped.' produces no step."""
        flows = extract_prose_call_flows(_RANGE_SKIPPED_FLOW)
        assert len(flows) == 1
        assert len(flows[0].steps) == 2  # only steps 1 and 8


# ---------------------------------------------------------------------------
# Real document smoke test
# ---------------------------------------------------------------------------


class TestRealDocument:
    """Smoke tests against the actual ingested document."""

    @pytest.mark.integration
    def test_extracts_flows_from_real_doc(self, tmp_path) -> None:
        """Extracts at least 50 call-flow DAGs from the real 3GPP spec document."""
        from pathlib import Path

        from specagent.retrieval.converter import convert
        from specagent.retrieval.markdown_postprocessor import postprocess

        doc = Path("data/docs/23502-j70.docx")
        if not doc.exists():
            pytest.skip("Real document not present")

        text = postprocess(convert(doc))
        flows = extract_prose_call_flows(text)

        assert len(flows) >= 38, f"Expected ≥38 flows, got {len(flows)}"
        total_steps = sum(len(f.steps) for f in flows)
        assert total_steps >= 380, (
            f"Expected ≥380 total steps after improvements, got {total_steps}"
        )
        # Every flow must have at least one step
        for flow in flows:
            assert len(flow.steps) >= 1, f"Flow {flow.figure_id!r} has no steps"
        # Every flow must produce parseable Mermaid
        from specagent.memgraph.mermaid_parser import parse_sequence_diagram

        for flow in flows[:10]:
            participants, steps = parse_sequence_diagram(flow.mermaid_content)
            assert len(steps) >= 1, f"Flow {flow.figure_id!r} mermaid produced no steps"
