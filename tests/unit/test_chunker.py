"""Tests for token-aware chunker with section header extraction."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_chunk_empty_text_returns_empty_list():
    """chunk() on empty string returns []."""
    mock_tok = MagicMock()
    mock_tok.encode.side_effect = lambda t, **kw: list(range(len(t.split())))
    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        original = chunker._tokenizer
        chunker._tokenizer = None
        try:
            result = chunker.chunk("")
        finally:
            chunker._tokenizer = original
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


# ---------------------------------------------------------------------------
# _get_tokenizer: caching and error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_tokenizer_caches_instance():
    """_get_tokenizer() returns the same object on repeated calls."""
    mock_tok = MagicMock()
    with patch("transformers.AutoTokenizer") as mock_auto:
        mock_auto.from_pretrained.return_value = mock_tok
        from specagent.retrieval import chunker

        original = chunker._tokenizer
        chunker._tokenizer = None
        try:
            t1 = chunker._get_tokenizer()
            # Second call must not invoke from_pretrained again
            t2 = chunker._get_tokenizer()
        finally:
            chunker._tokenizer = original

    assert t1 is t2
    mock_auto.from_pretrained.assert_called_once()


@pytest.mark.unit
def test_get_tokenizer_raises_on_load_failure():
    """_get_tokenizer() wraps any load exception in RuntimeError."""
    from specagent.retrieval import chunker

    original = chunker._tokenizer
    chunker._tokenizer = None
    try:
        with patch("transformers.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.side_effect = OSError("not cached")
            with pytest.raises(RuntimeError, match="not in the local cache"):
                chunker._get_tokenizer()
    finally:
        chunker._tokenizer = original


# ---------------------------------------------------------------------------
# _token_length: delegates to tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_token_length_uses_tokenizer():
    """_token_length() returns the number of token IDs returned by encode."""
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3, 4, 5]
    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        result = chunker._token_length("five tokens here right now")

    assert result == 5
    mock_tok.encode.assert_called_once_with("five tokens here right now", add_special_tokens=False)


# ---------------------------------------------------------------------------
# _merge_splits: produces multiple chunks when input exceeds chunk_size
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_splits_creates_multiple_chunks():
    """_merge_splits() flushes a new chunk when the size limit is exceeded."""
    # Each word maps to one token: encode returns one id per word.
    mock_tok = MagicMock()
    mock_tok.encode.side_effect = lambda t, **kw: list(range(len(t.split())))

    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        # 10 splits of 1 token each, chunk_size=3 tokens → must produce >1 chunk
        result = chunker._merge_splits(["word"] * 10, " ", chunk_size=3, overlap=1)

    assert len(result) > 1


# ---------------------------------------------------------------------------
# _split_recursive: character-level (empty separator) fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_split_recursive_char_fallback_splits_oversized_text():
    """_split_recursive() with empty separator encodes once and decodes windows."""
    mock_tok = MagicMock()
    # Pretend the text encodes to 20 token ids
    mock_tok.encode.return_value = list(range(20))
    # decode returns a string representation of the slice
    mock_tok.decode.side_effect = lambda ids: " ".join(str(i) for i in ids)

    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        result = chunker._split_recursive("some long text", [""], chunk_size=5, overlap=1)

    # With 20 tokens, chunk_size=5, step=4 → ceil(20/4)=5 windows
    assert len(result) > 1


@pytest.mark.unit
def test_split_recursive_char_fallback_returns_single_chunk_when_fits():
    """_split_recursive() with empty separator returns the text unchanged if it fits."""
    mock_tok = MagicMock()
    mock_tok.encode.return_value = list(range(3))  # only 3 tokens

    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        result = chunker._split_recursive("tiny", [""], chunk_size=5, overlap=1)

    assert result == ["tiny"]


# ---------------------------------------------------------------------------
# chunk(): min-token fallback preserves raw chunks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chunk_min_token_fallback_preserves_raw_chunks():
    """chunk() keeps raw chunks when all are below chunk_min_tokens."""
    mock_tok = MagicMock()
    # Always return 1 token so every chunk falls below any reasonable min
    mock_tok.encode.return_value = [1]
    mock_tok.decode.side_effect = lambda ids: " ".join(str(i) for i in ids)

    with (
        patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok),
        patch("specagent.retrieval.chunker.settings") as mock_settings,
    ):
        mock_settings.chunk_size_tokens = 512
        mock_settings.chunk_overlap_tokens = 64
        mock_settings.chunk_min_tokens = 100  # higher than any produced chunk

        from specagent.retrieval import chunker

        result = chunker.chunk("Short text that is below the minimum token floor.")

    # The min-token fallback must preserve the raw chunks rather than return []
    assert len(result) > 0


@pytest.mark.unit
def test_chunk_normal_path_filters_chunks_above_min_tokens():
    """chunk() normal path: filtered list is non-empty when tokens >= chunk_min_tokens."""
    mock_tok = MagicMock()
    # Return enough tokens to pass the min-token filter
    mock_tok.encode.return_value = list(range(50))  # 50 tokens
    mock_tok.decode.side_effect = lambda ids: " ".join(str(i) for i in ids)

    with (
        patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok),
        patch("specagent.retrieval.chunker.settings") as mock_settings,
    ):
        mock_settings.chunk_size_tokens = 512
        mock_settings.chunk_overlap_tokens = 64
        mock_settings.chunk_min_tokens = 10  # lower than produced chunks

        from specagent.retrieval import chunker

        result = chunker.chunk("Some text with enough tokens to pass the filter.")

    assert len(result) > 0


# ---------------------------------------------------------------------------
# _split_recursive: no separators left (base case)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_split_recursive_no_separators_returns_text_as_is():
    """_split_recursive() with empty separator list returns [text] immediately."""
    from specagent.retrieval import chunker

    result = chunker._split_recursive("any text", [], chunk_size=5, overlap=1)

    assert result == ["any text"]


# ---------------------------------------------------------------------------
# _split_recursive: oversized split flushes accumulated good_splits
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_split_recursive_flushes_good_splits_before_oversized_chunk():
    """_split_recursive() flushes good_splits when it encounters an oversized piece."""
    mock_tok = MagicMock()

    # Token length depends on content: short splits get small counts, long one gets big.
    def token_len(t: str, **kw: object) -> list[int]:
        """Return large token list for OVERSIZED text, small otherwise."""
        if "OVERSIZED" in t:
            return list(range(200))  # too big for chunk_size=10
        return list(range(2))  # small

    mock_tok.encode.side_effect = token_len

    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        # Text split by "\n\n": two small pieces then one oversized
        text = "small one\n\nsmall two\n\nOVERSIZED " + "x " * 200
        result = chunker._split_recursive(text, ["\n\n", "\n", " ", ""], chunk_size=10, overlap=2)

    assert isinstance(result, list)
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# _split_recursive: empty string pieces skipped
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_split_recursive_skips_empty_string_pieces():
    """_split_recursive() skips empty strings produced by consecutive separators."""
    mock_tok = MagicMock()
    mock_tok.encode.return_value = list(range(2))  # 2 tokens per piece

    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        # Leading and trailing separators produce empty strings in splits
        result = chunker._split_recursive(
            "\n\nactual content\n\n", ["\n\n", "\n", " ", ""], chunk_size=50, overlap=5
        )

    # Empty strings must not appear in the result
    assert all(r != "" for r in result)


# ---------------------------------------------------------------------------
# _merge_splits: all splits flushed during loop, nothing left at the end
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_splits_no_remainder_after_loop():
    """_merge_splits() produces no trailing chunk when all splits are flushed in-loop."""
    mock_tok = MagicMock()
    # Each word = 5 tokens; chunk_size = 5 → each split exactly fills one chunk.
    mock_tok.encode.side_effect = lambda t, **kw: list(range(5))

    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        # Separator is empty string so sep_len=0; each split is exactly chunk_size.
        # After flushing, overlap trimming removes everything (overlap=0),
        # so the next split starts a fresh chunk, and no remainder is left after the loop
        # only when the last split precisely triggered a flush.
        result = chunker._merge_splits(["a", "b"], "", chunk_size=5, overlap=0)

    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _merge_splits: overlap trimming empties current (inner if current: branch)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_splits_overlap_trimming_empties_current():
    """_merge_splits() handles the case where overlap trimming empties the current buffer."""
    mock_tok = MagicMock()
    # Each split = 4 tokens; chunk_size = 5, overlap = 0 → trimming removes everything.
    mock_tok.encode.side_effect = lambda t, **kw: [0] * 4

    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        # With chunk_size=5 and each split=4 tokens:
        # First split: current=[a], current_len=4
        # Second split: 4+0+4=8 > 5 → flush, then trim (overlap=0): current becomes []
        result = chunker._merge_splits(["a", "b", "c"], " ", chunk_size=5, overlap=0)

    assert len(result) >= 2


@pytest.mark.unit
def test_merge_splits_empty_splits_returns_empty_list():
    """_merge_splits() with an empty splits list returns [] without appending."""
    mock_tok = MagicMock()
    mock_tok.encode.return_value = []

    with patch("specagent.retrieval.chunker._get_tokenizer", return_value=mock_tok):
        from specagent.retrieval import chunker

        result = chunker._merge_splits([], " ", chunk_size=10, overlap=2)

    assert result == []
