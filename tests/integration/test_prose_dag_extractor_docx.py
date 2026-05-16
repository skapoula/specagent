"""Integration tests for prose DAG extractor against real .docx files in data/3gpp/docx/."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from specagent.retrieval.converter import convert
from specagent.retrieval.markdown_postprocessor import postprocess
from specagent.retrieval.prose_dag_extractor import extract_prose_call_flows

logger = logging.getLogger(__name__)

_RAW_DIR = Path(__file__).parents[2] / "data" / "3gpp" / "docx"


def _discover_docx() -> list[Path]:
    """Return all .docx files in data/3gpp/docx/, sorted by name."""
    if not _RAW_DIR.exists():
        return []
    return sorted(_RAW_DIR.glob("*.docx"))


def _docx_ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


_DOCX_FILES = _discover_docx()


@pytest.mark.integration
@pytest.mark.skipif(not _DOCX_FILES, reason="No .docx files found in data/3gpp/docx/")
class TestDocxFlowExtraction:
    """Per-file smoke tests: each docx must yield at least one parseable flow or skip."""

    @pytest.mark.parametrize("docx_path", _DOCX_FILES, ids=_docx_ids(_DOCX_FILES))
    def test_extracts_at_least_one_flow(self, docx_path: Path) -> None:
        """Each docx in data/3gpp/docx/ produces at least one call-flow DAG."""
        text = postprocess(convert(docx_path))
        flows = extract_prose_call_flows(text)
        if not flows:
            pytest.skip(f"{docx_path.name}: no prose call-flows found (not a procedure spec)")
        for flow in flows:
            assert len(flow.steps) >= 1, f"Flow {flow.figure_id!r} in {docx_path.name} has no steps"

    @pytest.mark.parametrize("docx_path", _DOCX_FILES, ids=_docx_ids(_DOCX_FILES))
    def test_mermaid_parseable(self, docx_path: Path) -> None:
        """Mermaid output from sampled flows is parseable by parse_sequence_diagram."""
        from specagent.kuzu.mermaid_parser import parse_sequence_diagram

        text = postprocess(convert(docx_path))
        flows = extract_prose_call_flows(text)
        if not flows:
            pytest.skip(f"{docx_path.name}: no prose call-flows found")
        for flow in flows[:5]:
            participants, steps = parse_sequence_diagram(flow.mermaid_content)
            assert len(steps) >= 1, (
                f"Flow {flow.figure_id!r} in {docx_path.name} produced no Mermaid steps"
            )


@pytest.mark.integration
@pytest.mark.skipif(not _DOCX_FILES, reason="No .docx files found in data/3gpp/docx/")
class TestImprovementsVisible:
    """Baselines for prose call-flow extraction against 38300-i30.docx (TS 38.300 NR Overview).

    Calibrated empirically: 10 flows, 20 steps, actors include AMF/UE/gNB.
    """

    @pytest.fixture(scope="class")
    def flows_38300(self) -> list:
        target = _RAW_DIR / "38300-i30_rel18.docx"
        if not target.exists():
            pytest.skip("38300-i30_rel18.docx not in data/3gpp/docx/")
        text = postprocess(convert(target))
        return extract_prose_call_flows(text)

    def test_flow_count_meets_baseline(self, flows_38300: list) -> None:
        """Flow count from 38300 meets the empirical baseline of 10."""
        assert len(flows_38300) >= 10, f"Flow count regressed: got {len(flows_38300)}, expected ≥10"

    def test_step_count_meets_baseline(self, flows_38300: list) -> None:
        """Total step count from 38300 meets the empirical baseline of 20."""
        total = sum(len(f.steps) for f in flows_38300)
        assert total >= 20, f"Step count regressed: got {total}, expected ≥20"

    def test_known_nf_names_appear(self, flows_38300: list) -> None:
        """At least one of the known NR actors (AMF, UE, gNB) appears in extracted flows."""
        all_actors = {
            actor
            for flow in flows_38300
            for step in flow.steps
            for actor in (step.from_actor, step.to_actor)
        }
        known_nf_set = {"AMF", "UE", "gNB"}
        found = known_nf_set & all_actors
        assert found, (
            f"None of the expected NF names appeared in 38300-i30.docx. "
            f"Actor sample: {sorted(all_actors)[:20]}"
        )
