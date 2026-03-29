"""Unit tests for markdown_postprocessor — pure function tests, no I/O."""

import pytest


# ---------------------------------------------------------------------------
# _normalize_nbsp
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeNbsp:
    """Tests for _normalize_nbsp()."""

    def test_replaces_nbsp_in_reference(self):
        """Non-breaking spaces in TS references are replaced with regular spaces."""
        from specagent.retrieval.markdown_postprocessor import _normalize_nbsp

        result = _normalize_nbsp("see TS\xa023.501\xa0[2]")
        assert result == "see TS 23.501 [2]"

    def test_replaces_nbsp_in_note_prefix(self):
        """Non-breaking space after NOTE is replaced."""
        from specagent.retrieval.markdown_postprocessor import _normalize_nbsp

        result = _normalize_nbsp("NOTE\xa01: Some note text.")
        assert result == "NOTE 1: Some note text."

    def test_no_nbsp_unchanged(self):
        """Text without non-breaking spaces is returned unchanged."""
        from specagent.retrieval.markdown_postprocessor import _normalize_nbsp

        text = "Normal text with spaces and\nnewlines."
        assert _normalize_nbsp(text) == text

    def test_empty_string(self):
        """Empty string input returns empty string."""
        from specagent.retrieval.markdown_postprocessor import _normalize_nbsp

        assert _normalize_nbsp("") == ""

    def test_multiple_nbsp_in_line(self):
        """All non-breaking spaces in a single line are replaced."""
        from specagent.retrieval.markdown_postprocessor import _normalize_nbsp

        result = _normalize_nbsp("a\xa0b\xa0c\xa0d")
        assert result == "a b c d"
        assert "\xa0" not in result


# ---------------------------------------------------------------------------
# _strip_toc
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStripToc:
    """Tests for _strip_toc()."""

    def test_strips_lines_before_first_h1(self):
        """Content before the first # heading is removed."""
        from specagent.retrieval.markdown_postprocessor import _strip_toc

        text = "Contents\n\n1 Scope 5\n\n2 References 6\n\n# Foreword\n\nBody."
        result = _strip_toc(text)
        assert result.startswith("# Foreword")
        assert "1 Scope 5" not in result

    def test_no_toc_unchanged(self):
        """Text that starts with a # heading is returned unchanged."""
        from specagent.retrieval.markdown_postprocessor import _strip_toc

        text = "# Title\n\nContent."
        assert _strip_toc(text) == text

    def test_no_headings_returns_unchanged(self):
        """Text with no # headings is returned unchanged."""
        from specagent.retrieval.markdown_postprocessor import _strip_toc

        text = "Just plain text\nNo headings here."
        assert _strip_toc(text) == text

    def test_empty_string(self):
        """Empty input returns empty string."""
        from specagent.retrieval.markdown_postprocessor import _strip_toc

        assert _strip_toc("") == ""

    def test_preserves_all_content_from_heading_onwards(self):
        """All content from the first heading onward is preserved."""
        from specagent.retrieval.markdown_postprocessor import _strip_toc

        body = "# Foreword\n\nParagraph one.\n\n## Section\n\nParagraph two."
        text = "TOC line 1\nTOC line 2\n\n" + body
        result = _strip_toc(text)
        assert result == body

    def test_h2_heading_does_not_trigger_strip(self):
        """Content before a ## heading is NOT stripped (only # triggers)."""
        from specagent.retrieval.markdown_postprocessor import _strip_toc

        text = "TOC content\n\n## Section\n\nBody."
        # No H1 found, so the full text is returned unchanged
        result = _strip_toc(text)
        assert result == text


# ---------------------------------------------------------------------------
# _strip_change_history
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStripChangeHistory:
    """Tests for _strip_change_history()."""

    def test_strips_annex_l_bare_heading(self):
        """Bare 'Annex L' line at start triggers stripping."""
        from specagent.retrieval.markdown_postprocessor import _strip_change_history

        text = "# K.1 Use of connect-udp\n\nSome text.\n\nAnnex L (informative):\n\n| Date | CR |\n| --- | --- |\n"
        result = _strip_change_history(text)
        assert "Annex L" not in result
        assert "K.1 Use of connect-udp" in result

    def test_strips_annex_l_h1_heading(self):
        """'# Annex L' heading form triggers stripping."""
        from specagent.retrieval.markdown_postprocessor import _strip_change_history

        text = "# K.1 Content\n\nBody.\n\n# Annex L (informative): Change history\n\n| Date | Rev |\n"
        result = _strip_change_history(text)
        assert "# Annex L" not in result
        assert "K.1 Content" in result

    def test_strips_change_history_table_header(self):
        """'| Change history |' table header fallback triggers stripping."""
        from specagent.retrieval.markdown_postprocessor import _strip_change_history

        text = "Body content.\n\n| Change history | | | |\n| --- | --- | --- | --- |\n| 2024-03 | SP-109 | 1.0 | Initial |\n"
        result = _strip_change_history(text)
        assert "Change history" not in result
        assert "Body content" in result

    def test_no_change_history_unchanged(self):
        """Text without change history is returned unchanged."""
        from specagent.retrieval.markdown_postprocessor import _strip_change_history

        text = "# Scope\n\nContent.\n\n# Annex A\n\nNormative content."
        assert _strip_change_history(text) == text

    def test_mid_sentence_annex_l_not_stripped(self):
        """'Annex L' in the middle of a sentence is not stripped."""
        from specagent.retrieval.markdown_postprocessor import _strip_change_history

        text = "# Body\n\nSee also Annex L of TS 23.501.\n\n# Annex A\n\nContent."
        # "Annex L" here is not at the start of a line
        result = _strip_change_history(text)
        assert "See also Annex L" in result

    def test_result_is_rstripped(self):
        """Trailing whitespace is removed from the stripped result."""
        from specagent.retrieval.markdown_postprocessor import _strip_change_history

        text = "Body.\n\nAnnex L (informative):\n\n| table |\n"
        result = _strip_change_history(text)
        assert result == result.rstrip()

    def test_empty_string(self):
        """Empty input returns empty string."""
        from specagent.retrieval.markdown_postprocessor import _strip_change_history

        assert _strip_change_history("") == ""


# ---------------------------------------------------------------------------
# _fix_annex_headings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFixAnnexHeadings:
    """Tests for _fix_annex_headings()."""

    def test_bare_annex_a_gets_heading_marker(self):
        """'Annex A (informative):' at start of line gets '# ' prepended."""
        from specagent.retrieval.markdown_postprocessor import _fix_annex_headings

        text = "Previous content.\n\nAnnex A (informative):\n\nAnnex body.\n"
        result = _fix_annex_headings(text)
        assert "# Annex A (informative):" in result

    def test_already_headed_annex_not_doubled(self):
        """'# Annex B' already has a heading — must not become '# # Annex B'."""
        from specagent.retrieval.markdown_postprocessor import _fix_annex_headings

        text = "# Annex B (normative):\n\nContent.\n"
        result = _fix_annex_headings(text)
        assert result == text
        assert "# # Annex" not in result

    def test_sub_annex_heading_unchanged(self):
        """'# K.1 Use of connect-udp' sub-heading is not modified."""
        from specagent.retrieval.markdown_postprocessor import _fix_annex_headings

        text = "# K.1 Use of connect-udp\n\nContent.\n"
        result = _fix_annex_headings(text)
        assert result == text

    def test_mid_sentence_annex_not_headed(self):
        """'Annex A' in body text (not at line start) is not modified."""
        from specagent.retrieval.markdown_postprocessor import _fix_annex_headings

        text = "# Body\n\nSee Annex A for details.\n"
        result = _fix_annex_headings(text)
        assert "# See Annex A" not in result
        assert "See Annex A for details." in result

    def test_multiple_annexes_all_fixed(self):
        """All bare Annex headings in a document are promoted."""
        from specagent.retrieval.markdown_postprocessor import _fix_annex_headings

        text = (
            "Body.\n\n"
            "Annex A (informative):\n\nA content.\n\n"
            "Annex B (normative):\n\nB content.\n"
        )
        result = _fix_annex_headings(text)
        assert "# Annex A (informative):" in result
        assert "# Annex B (normative):" in result

    def test_lowercase_annex_not_matched(self):
        """'annex a' (lowercase) is not promoted — only 'Annex A' form."""
        from specagent.retrieval.markdown_postprocessor import _fix_annex_headings

        text = "annex a content\n"
        result = _fix_annex_headings(text)
        # The line starts with 'annex ' (lowercase) — stripped starts with 'annex '
        # not 'Annex ' so it should be unchanged
        assert result == text

    def test_empty_string(self):
        """Empty input returns empty string."""
        from specagent.retrieval.markdown_postprocessor import _fix_annex_headings

        assert _fix_annex_headings("") == ""


# ---------------------------------------------------------------------------
# postprocess (integration of all transforms)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPostprocess:
    """Tests for the top-level postprocess() orchestrator."""

    def test_all_transforms_applied(self):
        """postprocess() applies all four transforms in the correct order."""
        from specagent.retrieval.markdown_postprocessor import postprocess

        text = (
            "TOC line\n"
            "4.1 Scope 5\n\n"
            "# Foreword\n\n"
            "See TS\xa023.501\xa0[2] for details.\n\n"
            "Annex A (informative):\n\nNormative text.\n\n"
            "Annex L (informative):\n\n| Change history | |\n"
        )
        result = postprocess(text)

        # TOC stripped
        assert "4.1 Scope 5" not in result
        assert result.startswith("# Foreword")

        # NBSP normalised
        assert "\xa0" not in result
        assert "TS 23.501 [2]" in result

        # Change history stripped
        assert "Annex L" not in result

        # Annex heading promoted
        assert "# Annex A (informative):" in result

    def test_empty_string_safe(self):
        """postprocess() on empty string returns empty string without error."""
        from specagent.retrieval.markdown_postprocessor import postprocess

        assert postprocess("") == ""

    def test_plain_document_unchanged_in_structure(self):
        """A document with no TOC, no change history, no annexes is structurally unchanged."""
        from specagent.retrieval.markdown_postprocessor import postprocess

        text = "# Title\n\nContent with regular spaces.\n"
        result = postprocess(text)
        assert result == text
