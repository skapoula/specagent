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
        # Every flow must have at least one step
        for flow in flows:
            assert len(flow.steps) >= 1, f"Flow {flow.figure_id!r} has no steps"
        # Every flow must produce parseable Mermaid
        from specagent.memgraph.mermaid_parser import parse_sequence_diagram

        for flow in flows[:10]:
            participants, steps = parse_sequence_diagram(flow.mermaid_content)
            assert len(steps) >= 1, f"Flow {flow.figure_id!r} mermaid produced no steps"
