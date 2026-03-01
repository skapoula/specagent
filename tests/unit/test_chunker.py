"""Tests for token-aware chunker with section header extraction."""
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
def test_chunk_empty_text_returns_empty_list():
    """chunk() on empty string returns []."""
    mock_tok = MagicMock()
    mock_tok.encode.side_effect = lambda t, **kw: list(range(len(t.split())))
    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        chunker._tokenizer = None
        result = chunker.chunk("")
    assert result == []


@pytest.mark.unit
def test_chunk_with_metadata_returns_list_of_tuples():
    """chunk_with_metadata() returns (text, section_header) tuples."""
    with patch("specagent.retrieval.chunker.chunk") as mock_chunk:
        mock_chunk.return_value = ["Some content."]
        from specagent.retrieval import chunker

        results = chunker.chunk_with_metadata("Some content.")
    assert isinstance(results, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in results)


@pytest.mark.unit
def test_chunk_with_metadata_extracts_leading_header():
    """chunk_with_metadata() finds the nearest preceding heading for each chunk."""
    with patch("specagent.retrieval.chunker.chunk") as mock_chunk:
        mock_chunk.return_value = [
            "# Section Alpha\n\nFirst chunk content.",
            "Continuation without a header.",
        ]
        from specagent.retrieval import chunker

        results = chunker.chunk_with_metadata("irrelevant")

    _, section0 = results[0]
    _, section1 = results[1]
    assert "Alpha" in section0
    # Second chunk inherits header from first
    assert section1 == section0


@pytest.mark.unit
def test_chunk_with_metadata_updates_header_when_new_heading_found():
    """A new heading in a later chunk updates the running section header."""
    with patch("specagent.retrieval.chunker.chunk") as mock_chunk:
        mock_chunk.return_value = [
            "# Section One\n\nContent one.",
            "# Section Two\n\nContent two.",
        ]
        from specagent.retrieval import chunker

        results = chunker.chunk_with_metadata("irrelevant")

    _, section0 = results[0]
    _, section1 = results[1]
    assert "One" in section0
    assert "Two" in section1
    assert section0 != section1
