"""Unit tests for specagent.retrieval.spec_filename."""

from pathlib import Path

import pytest

from specagent.retrieval.spec_filename import parse_3gpp_release, release_paths

# ---------------------------------------------------------------------------
# parse_3gpp_release
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "stem, expected",
    [
        # Alpha major — a=10 (Rel-10) through z=35
        ("38108-i40", 18),  # i = 18
        ("38104-ic0", 18),  # i = 18 (multi-char minor ignored)
        ("23502-j70", 19),  # j = 19
        ("38300-a00", 10),  # a = 10
        ("38300-z99", 35),  # z = 35
        # Uppercase treated as lowercase
        ("38300-I40", 18),
        # Digit major — older releases (0-9)
        ("24301-900", 9),
        ("24301-800", 8),
        # Spec number without hyphened version
        ("38108", None),
        # Only a hyphen but no version chars
        ("38108-", None),
        # Empty stem
        ("", None),
        # Non-standard stem (no digit spec number prefix)
        ("somespec-i40", 18),
    ],
)
def test_parse_3gpp_release(stem: str, expected: int | None) -> None:
    assert parse_3gpp_release(stem) == expected


@pytest.mark.unit
def test_parse_3gpp_release_returns_none_for_hyphen_only() -> None:
    assert parse_3gpp_release("-") is None


# ---------------------------------------------------------------------------
# release_paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_release_paths_returns_correct_structure(tmp_path: Path) -> None:
    # Use the real DOCX_SMALL stem (38413-i30.docx, TS 38.413 NG-AP, Rel-18).
    # release_paths() only reads source.stem — no file I/O needed.
    source = Path("38413-i30.docx")

    docx_dest, md_dest = release_paths(source, tmp_path)

    assert docx_dest == tmp_path / "3gpp_rel_18" / "docx" / "38413-i30_rel18.docx"
    assert md_dest == tmp_path / "3gpp_rel_18" / "md" / "38413-i30_rel18.md"


@pytest.mark.unit
def test_release_paths_rel19(tmp_path: Path) -> None:
    source = Path("23502-j70.docx")

    docx_dest, md_dest = release_paths(source, tmp_path)

    assert docx_dest == tmp_path / "3gpp_rel_19" / "docx" / "23502-j70_rel19.docx"
    assert md_dest == tmp_path / "3gpp_rel_19" / "md" / "23502-j70_rel19.md"


@pytest.mark.unit
def test_release_paths_returns_none_when_release_unknown(tmp_path: Path) -> None:
    source = Path("unknownspec.docx")

    result = release_paths(source, tmp_path)

    assert result is None


@pytest.mark.unit
def test_release_paths_stem_suffix_correct(tmp_path: Path) -> None:
    """The release suffix appended to the stem must use two-digit zero-padded release."""
    source = Path("24301-900.docx")

    docx_dest, md_dest = release_paths(source, tmp_path)

    assert docx_dest == tmp_path / "3gpp_rel_09" / "docx" / "24301-900_rel09.docx"
    assert md_dest == tmp_path / "3gpp_rel_09" / "md" / "24301-900_rel09.md"
