"""Tests for the MarkItDown-based document converter."""
import pytest
from pathlib import Path


@pytest.mark.unit
def test_convert_markdown_file(tmp_path):
    """convert() on a .md file returns its content as a string."""
    from specagent.retrieval.converter import convert

    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n\nSome content.")
    result = convert(md_file)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.unit
def test_convert_unsupported_extension_raises(tmp_path):
    """convert() raises UnsupportedFormatError for unknown extensions."""
    from specagent.retrieval.converter import convert
    from specagent.retrieval.exceptions import UnsupportedFormatError

    bad_file = tmp_path / "file.xyz"
    bad_file.write_text("content")
    with pytest.raises(UnsupportedFormatError, match=r"\.xyz"):
        convert(bad_file)
