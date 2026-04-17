"""Tests for the MarkItDown-based document converter."""

from unittest.mock import patch

import pytest

from tests.conftest import DOCX_SMALL


@pytest.mark.unit
def test_convert_docx_file():
    """convert() on a real 3GPP .docx returns non-empty markdown string."""
    from specagent.retrieval.converter import convert

    result = convert(DOCX_SMALL)
    assert isinstance(result, str)
    assert len(result) > 0
    # Real 3GPP spec content expected
    assert "3GPP" in result


@pytest.mark.unit
def test_convert_unsupported_extension_raises(tmp_path):
    """convert() raises UnsupportedFormatError for unknown extensions."""
    from specagent.retrieval.converter import convert
    from specagent.retrieval.exceptions import UnsupportedFormatError

    bad_file = tmp_path / "file.xyz"
    bad_file.write_text("content")
    with pytest.raises(UnsupportedFormatError, match=r"\.xyz"):
        convert(bad_file)


@pytest.mark.unit
def test_convert_no_extension_raises(tmp_path):
    """convert() raises UnsupportedFormatError when file has no extension."""
    from specagent.retrieval.converter import convert
    from specagent.retrieval.exceptions import UnsupportedFormatError

    no_ext = tmp_path / "noext"
    no_ext.write_text("content")
    with pytest.raises(UnsupportedFormatError, match="No file extension"):
        convert(no_ext)


@pytest.mark.unit
def test_convert_markitdown_error_raises_ingestion_error(tmp_path):
    """convert() wraps MarkItDown exceptions as IngestionError."""
    from specagent.retrieval.converter import convert
    from specagent.retrieval.exceptions import IngestionError

    md = tmp_path / "fail.md"
    md.write_text("content")
    with patch("specagent.retrieval.converter._get_markitdown") as mock_md:
        mock_md.return_value.convert.side_effect = RuntimeError("parse error")
        with pytest.raises(IngestionError, match="Failed to convert"):
            convert(md)


@pytest.mark.unit
def test_convert_empty_result_returns_empty_string(tmp_path):
    """convert() returns empty string and logs warning when text is empty."""
    from specagent.retrieval.converter import convert

    md = tmp_path / "empty.md"
    md.write_text("content")
    with patch("specagent.retrieval.converter._get_markitdown") as mock_md:
        mock_result = mock_md.return_value.convert.return_value
        mock_result.text_content = ""
        result = convert(md)
    assert result == ""
