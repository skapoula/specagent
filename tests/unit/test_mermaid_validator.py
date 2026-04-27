"""Unit tests for mermaid_validator."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestValidateMermaid:
    """Tests for validate_mermaid()."""

    def test_valid_sequence_diagram_passes(self) -> None:
        """A well-formed sequenceDiagram block returns (True, '')."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nsequenceDiagram\n  UE->>gNB: msg\n  gNB-->>UE: ack\n```"
        valid, reason = validate_mermaid(content)
        assert valid is True
        assert reason == ""

    def test_valid_state_diagram_passes(self) -> None:
        """A well-formed stateDiagram-v2 block returns (True, '')."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nstateDiagram-v2\n  [*] --> Idle\n  Idle --> Active: start\n```"
        valid, reason = validate_mermaid(content)
        assert valid is True
        assert reason == ""

    def test_valid_flowchart_passes(self) -> None:
        """A well-formed flowchart TD block returns (True, '')."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nflowchart TD\n  A[Start] --> B[End]\n```"
        valid, reason = validate_mermaid(content)
        assert valid is True
        assert reason == ""

    def test_missing_mermaid_fence_fails(self) -> None:
        """Raw content without ```mermaid fence returns (False, reason)."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        valid, reason = validate_mermaid("sequenceDiagram\n  A->>B: msg\n")
        assert valid is False
        assert "fenced" in reason.lower() or "mermaid" in reason.lower()

    def test_unknown_header_fails(self) -> None:
        """Unrecognised diagram type keyword returns (False, reason)."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nunknownDiagramType\n  A --> B\n```"
        valid, reason = validate_mermaid(content)
        assert valid is False
        assert "unknownDiagramType" in reason or "Unknown" in reason

    def test_empty_content_fails(self) -> None:
        """A block with only the header line and no content returns (False, reason)."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nsequenceDiagram\n```"
        valid, reason = validate_mermaid(content)
        assert valid is False
        assert "content" in reason.lower() or "header" in reason.lower()

    def test_comment_lines_not_counted_as_content(self) -> None:
        """Lines starting with %% are comments and don't count as content."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\nsequenceDiagram\n  %% This is a comment\n```"
        valid, reason = validate_mermaid(content)
        assert valid is False

    def test_unbalanced_brackets_fails(self) -> None:
        """A block with unbalanced [ returns (False, reason)."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\ngraph TD\n  A[Node --> B[Another\n```"
        valid, reason = validate_mermaid(content)
        assert valid is False
        assert "bracket" in reason.lower() or "unbalanced" in reason.lower()

    def test_mmdc_not_called_when_disabled(self, monkeypatch) -> None:
        """subprocess.run is never called when mermaid_validate_with_mmdc is False."""
        from unittest.mock import patch

        content = "```mermaid\nsequenceDiagram\n  A->>B: msg\n  B-->>A: ack\n```"
        with patch("subprocess.run") as mock_run:
            monkeypatch.setattr(
                "specagent.retrieval.mermaid_validator.settings",
                type("S", (), {"mermaid_validate_with_mmdc": False, "mermaid_mmdc_timeout": 10})(),
            )
            from specagent.retrieval.mermaid_validator import validate_mermaid

            validate_mermaid(content)
        mock_run.assert_not_called()

    def test_valid_graph_lr_passes(self) -> None:
        """graph LR with two nodes passes validation."""
        from specagent.retrieval.mermaid_validator import validate_mermaid

        content = "```mermaid\ngraph LR\n  A[UE] --> B[gNB]\n  B --> C[AMF]\n```"
        valid, reason = validate_mermaid(content)
        assert valid is True
        assert reason == ""


# ---------------------------------------------------------------------------
# Issue 9: _check_with_mmdc cleans up temp file even on OSError (TDD)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckWithMmdcTempFileCleanup:
    def test_temp_file_deleted_on_oserror(self, tmp_path, monkeypatch) -> None:
        """Temp file must be deleted even when subprocess.run raises OSError."""
        from pathlib import Path
        from unittest.mock import patch

        created_paths: list[Path] = []
        original_NamedTemporaryFile = __import__("tempfile").NamedTemporaryFile

        def tracking_ntf(**kwargs):
            f = original_NamedTemporaryFile(**kwargs)
            created_paths.append(Path(f.name))
            return f

        with (
            patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf),
            patch("subprocess.run", side_effect=OSError("permission denied")),
            patch(
                "specagent.retrieval.mermaid_validator.settings",
                mermaid_validate_with_mmdc=True,
                mermaid_mmdc_timeout=10,
            ),
        ):
            from specagent.retrieval.mermaid_validator import _check_with_mmdc

            _check_with_mmdc("sequenceDiagram\n  A->>B: msg\n")

        # All created temp files must have been cleaned up
        for p in created_paths:
            assert not p.exists(), f"Temp file {p} was not cleaned up"


# ---------------------------------------------------------------------------
# Issue 10: _check_bracket_balance handles %% comments and apostrophes (TDD)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckBracketBalance:
    def test_comment_line_with_unbalanced_bracket_passes(self) -> None:
        """%% comment lines with unbalanced [ must not cause false negatives."""
        from specagent.retrieval.mermaid_validator import _check_bracket_balance

        inner = "sequenceDiagram\n  A->>B: msg\n  %% unmatched [ bracket in comment\n"
        assert _check_bracket_balance(inner) is True

    def test_label_with_apostrophe_passes(self) -> None:
        """Labels like \"UE's response\" must not be treated as unbalanced."""
        from specagent.retrieval.mermaid_validator import _check_bracket_balance

        inner = 'sequenceDiagram\n  A->>B: "UE\'s response"\n  B-->>A: ack\n'
        assert _check_bracket_balance(inner) is True

    def test_genuinely_unbalanced_brace_fails(self) -> None:
        """An unbalanced { outside strings/comments must return False."""
        from specagent.retrieval.mermaid_validator import _check_bracket_balance

        inner = "graph TD\n  A{Open brace without close\n  B --> A\n"
        assert _check_bracket_balance(inner) is False

    def test_balanced_brackets_pass(self) -> None:
        """All paired brackets return True."""
        from specagent.retrieval.mermaid_validator import _check_bracket_balance

        inner = "graph TD\n  A[Node] --> B(Circle)\n  B --> C{Diamond}\n"
        assert _check_bracket_balance(inner) is True
