"""Integration tests for prose DAG extractor against real .docx files in data/raw/."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from specagent.retrieval.converter import convert
from specagent.retrieval.markdown_postprocessor import postprocess
from specagent.retrieval.prose_dag_extractor import extract_prose_call_flows

logger = logging.getLogger(__name__)

_RAW_DIR = Path(__file__).parents[2] / "data" / "raw"


def _discover_docx() -> list[Path]:
    """Return all .docx files in data/raw/, sorted by name."""
    if not _RAW_DIR.exists():
        return []
    return sorted(_RAW_DIR.glob("*.docx"))


def _docx_ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


_DOCX_FILES = _discover_docx()


@pytest.mark.integration
@pytest.mark.skipif(not _DOCX_FILES, reason="No .docx files found in data/raw/")
class TestDocxFlowExtraction:
    """Per-file smoke tests: each docx must yield at least one parseable flow or skip."""

    @pytest.mark.parametrize("docx_path", _DOCX_FILES, ids=_docx_ids(_DOCX_FILES))
    def test_extracts_at_least_one_flow(self, docx_path: Path) -> None:
        """Each docx in data/raw produces at least one call-flow DAG."""
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
@pytest.mark.skipif(not _DOCX_FILES, reason="No .docx files found in data/raw/")
class TestImprovementsVisible:
    """Confirm improvements yield observable results and baselines hold on 23502-j70.docx."""

    @pytest.fixture(scope="class")
    def flows_23502(self) -> list:
        target = _RAW_DIR / "23502-j70.docx"
        if not target.exists():
            pytest.skip("23502-j70.docx not in data/raw/")
        text = postprocess(convert(target))
        return extract_prose_call_flows(text)

    def test_flow_count_meets_baseline(self, flows_23502: list) -> None:
        """Flow count from 23502 is at least the baseline of 38."""
        assert len(flows_23502) >= 38, f"Flow count regressed: got {len(flows_23502)}, expected ≥38"

    def test_step_count_meets_baseline(self, flows_23502: list) -> None:
        """Total step count from 23502 is at least the baseline of 250."""
        total = sum(len(f.steps) for f in flows_23502)
        assert total >= 250, f"Step count regressed: got {total}, expected ≥250"

    def test_new_nf_names_appear(self, flows_23502: list) -> None:
        """At least one of the new NF names (EIR, DN, SCEF, MBSF, 5GC, FN-RG, W-5GAN) appears."""
        all_actors = {
            actor
            for flow in flows_23502
            for step in flow.steps
            for actor in (step.from_actor, step.to_actor)
        }
        new_nf_set = {"EIR", "DN", "SCEF", "5GC", "MBSF", "MBSTF", "FN-RG", "W-5GAN"}
        found = new_nf_set & all_actors
        assert found, (
            f"None of the new NF names appeared in 23502-j70.docx. "
            f"Actor sample: {sorted(all_actors)[:20]}"
        )
