"""Unit tests for compare_extractors helper functions."""

import pytest

# Import lazily inside each test so the module can be imported before the script exists.


@pytest.mark.unit
def test_count_vision_steps_counts_sync_arrows():
    from scripts.compare_extractors import count_vision_steps

    mermaid = "```mermaid\nsequenceDiagram\n    UE->>AMF: Msg1\n    AMF->>SMF: Msg2\n```"
    assert count_vision_steps(mermaid) == 2


@pytest.mark.unit
def test_count_vision_steps_counts_async_arrows():
    from scripts.compare_extractors import count_vision_steps

    mermaid = "```mermaid\nsequenceDiagram\n    AMF-->>UE: Response\n```"
    assert count_vision_steps(mermaid) == 1


@pytest.mark.unit
def test_count_vision_steps_empty_returns_zero():
    from scripts.compare_extractors import count_vision_steps

    assert count_vision_steps("") == 0


@pytest.mark.unit
def test_count_vision_steps_no_arrows_returns_zero():
    from scripts.compare_extractors import count_vision_steps

    mermaid = "```mermaid\nsequenceDiagram\n    participant UE\n```"
    assert count_vision_steps(mermaid) == 0


@pytest.mark.unit
def test_count_vision_actors_counts_participant_lines():
    from scripts.compare_extractors import count_vision_actors

    mermaid = "```mermaid\nsequenceDiagram\n    participant UE\n    participant AMF\n    UE->>AMF: Msg\n```"
    assert count_vision_actors(mermaid) == 2


@pytest.mark.unit
def test_count_vision_actors_empty_returns_zero():
    from scripts.compare_extractors import count_vision_actors

    assert count_vision_actors("") == 0


@pytest.mark.unit
def test_align_results_matches_by_caption_substring():
    """Prose flow title substring matches vision diagram caption."""
    from unittest.mock import MagicMock

    from scripts.compare_extractors import align_results

    prose_flow = MagicMock()
    prose_flow.figure_id = "8.1-1"
    prose_flow.title = "NG Setup procedure"
    prose_flow.steps = [MagicMock(), MagicMock()]
    prose_flow.participants = ["gNB", "AMF"]
    prose_flow.mermaid_content = "```mermaid\nsequenceDiagram\n    gNB->>AMF: NG Setup Request\n```"

    vision_diag = MagicMock()
    vision_diag.caption = "NG Setup procedure"
    vision_diag.mermaid_content = "```mermaid\nsequenceDiagram\n    gNB->>AMF: NG Setup Request\n    AMF->>gNB: NG Setup Response\n```"

    rows = align_results([prose_flow], [vision_diag])
    assert len(rows) == 1
    assert rows[0].figure_id == "8.1-1"
    # prose_steps comes from len(flow.steps), not mermaid arrow count
    assert rows[0].prose_steps == 2
    assert rows[0].vision_steps == 2


@pytest.mark.unit
def test_align_results_prose_only_when_no_vision_match():
    from unittest.mock import MagicMock

    from scripts.compare_extractors import align_results

    prose_flow = MagicMock()
    prose_flow.figure_id = "8.1-1"
    prose_flow.title = "Unique Procedure"
    prose_flow.steps = [MagicMock()]
    prose_flow.participants = ["UE"]
    prose_flow.mermaid_content = "```mermaid\nsequenceDiagram\n    UE->>AMF: Msg\n```"

    rows = align_results([prose_flow], [])
    assert len(rows) == 1
    assert rows[0].vision_steps == 0
    assert rows[0].vision_valid == "—"


@pytest.mark.unit
def test_align_results_vision_only_when_no_prose_match():
    from unittest.mock import MagicMock

    from scripts.compare_extractors import align_results

    vision_diag = MagicMock()
    vision_diag.caption = "Handover Procedure"
    vision_diag.mermaid_content = (
        "```mermaid\nsequenceDiagram\n    gNB->>AMF: Handover Required\n```"
    )

    rows = align_results([], [vision_diag])
    assert len(rows) == 1
    assert rows[0].figure_id == "—"
    assert rows[0].prose_steps == 0


@pytest.mark.unit
def test_render_table_contains_header_and_summary():
    from scripts.compare_extractors import ComparisonRow, render_table

    rows = [
        ComparisonRow(
            figure_id="8.1-1",
            caption="NG Setup",
            prose_steps=3,
            vision_steps=4,
            prose_actors=2,
            vision_actors=2,
            prose_valid="✓",
            vision_valid="✓",
            winner="vision",
        )
    ]
    table = render_table(rows)
    assert "| figure_id |" in table
    assert "8.1-1" in table
    assert "**Summary**" in table


@pytest.mark.unit
def test_render_table_dry_run_shows_dashes_for_vision():
    from scripts.compare_extractors import ComparisonRow, render_table

    rows = [
        ComparisonRow(
            figure_id="8.1-1",
            caption="NG Setup",
            prose_steps=3,
            vision_steps=0,
            prose_actors=2,
            vision_actors=0,
            prose_valid="✓",
            vision_valid="—",
            winner="—",
        )
    ]
    table = render_table(rows)
    assert table.count("—") >= 2  # vision_valid and winner both show "—"
