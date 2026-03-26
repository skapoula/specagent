"""Token-aware recursive text chunking using a shared tokenizer singleton."""

import logging
import re as _re
from typing import TYPE_CHECKING

from specagent.config import settings

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_tokenizer: "PreTrainedTokenizerBase | None" = None

# Separator hierarchy for recursive splitting
_SEPARATORS = ["\n\n", "\n", " ", ""]

# Compiled regex for markdown headings (used by chunk_with_metadata)
_HEADER_RE = _re.compile(r"^#{1,6}\s+(.+)$", _re.MULTILINE)


def _get_tokenizer() -> "PreTrainedTokenizerBase":
    """Return the tokenizer singleton, loading it on first call.

    Uses settings.embedding_model as the model ID so the tokenizer always
    matches the configured embedding model.

    Raises:
        RuntimeError: If the tokenizer is not cached locally. Run
            'python -m specagent download-model' to download it.
    """
    global _tokenizer  # noqa: PLW0603
    if _tokenizer is None:
        from transformers import AutoTokenizer

        model_id = settings.embedding_model
        logger.info("Loading tokenizer %s", model_id)
        try:
            _tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                model_id, local_files_only=True, trust_remote_code=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Tokenizer '{model_id}' is not in the local cache. "
                "Run 'specagent download-model' to download it, "
                "then restart the server."
            ) from exc
    assert _tokenizer is not None  # guaranteed by the if-block above
    return _tokenizer


def _token_length(text: str) -> int:
    """Return the number of tokens in *text* using the embedding tokenizer."""
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def _merge_splits(splits: list[str], separator: str, chunk_size: int, overlap: int) -> list[str]:
    """Merge small splits into chunks respecting chunk_size and overlap.

    Token lengths for each split are cached to avoid redundant tokenizer calls
    during overlap trimming. Separator tokens are counted in the assembled
    chunk length so the configured limit is never silently exceeded.
    """
    sep_len = _token_length(separator) if separator else 0
    chunks: list[str] = []
    current: list[str] = []
    lengths: list[int] = []  # cached token lengths, parallel to current
    current_len = 0

    for split in splits:
        split_len = _token_length(split)
        # Separator tokens added before this split when current is non-empty
        sep_addition = sep_len if current else 0
        # If adding this split would exceed chunk_size, flush current
        if current_len + sep_addition + split_len > chunk_size and current:
            chunks.append(separator.join(current))
            # Keep overlap: drop splits from the front until under overlap budget
            while current and current_len > overlap:
                removed_len = lengths.pop(0)
                current.pop(0)
                current_len -= removed_len
                if current:
                    # The separator that preceded this element is also gone
                    current_len -= sep_len
            # Recalculate sep_addition after overlap trimming
            sep_addition = sep_len if current else 0
        current.append(split)
        lengths.append(split_len)
        current_len += sep_addition + split_len

    if current:
        chunks.append(separator.join(current))

    return chunks


def _split_recursive(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Recursively split text using the first separator that produces usable pieces.

    Sub-pieces that required recursion are not re-joined with the parent separator;
    they are flushed as independent chunks to preserve the separator that was
    actually used at each recursion level.

    For the character-level fallback (empty separator), the text is encoded once
    and decoded as token windows to avoid O(n) per-character tokenizer calls.
    """
    if not separators:
        # No more separators — return text as-is (may be oversized, caller filters)
        return [text]

    sep = separators[0]
    remaining = separators[1:]

    # Character-level last resort: encode once and decode sliding windows.
    if sep == "":
        tok = _get_tokenizer()
        token_ids = tok.encode(text, add_special_tokens=False)
        if len(token_ids) <= chunk_size:
            return [text]
        step = max(1, chunk_size - overlap)
        return [tok.decode(token_ids[i : i + chunk_size]) for i in range(0, len(token_ids), step)]

    splits = text.split(sep)

    # Separate direct (small) splits from oversized ones that need recursion.
    # Flush good_splits before appending recursed output so each group is merged
    # with the separator that was actually used to produce its pieces.
    final_chunks: list[str] = []
    good_splits: list[str] = []

    for s in splits:
        if not s:
            continue
        if _token_length(s) > chunk_size:
            if good_splits:
                final_chunks.extend(_merge_splits(good_splits, sep, chunk_size, overlap))
                good_splits = []
            final_chunks.extend(_split_recursive(s, remaining, chunk_size, overlap))
        else:
            good_splits.append(s)

    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, sep, chunk_size, overlap))

    return final_chunks


def chunk(text: str) -> list[str]:
    """Split *text* into token-bounded chunks suitable for embedding.

    Args:
        text: The Markdown text to split.

    Returns:
        List of chunk strings, each between chunk_min_tokens and chunk_size_tokens.
    """
    if not text.strip():
        return []

    raw_chunks = _split_recursive(
        text,
        _SEPARATORS,
        settings.chunk_size_tokens,
        settings.chunk_overlap_tokens,
    )
    filtered = [c for c in raw_chunks if _token_length(c) >= settings.chunk_min_tokens]

    if not filtered and raw_chunks:
        # Document is shorter than chunk_min_tokens — preserve raw_chunks rather
        # than returning text.strip() which is not size-validated.
        logger.debug(
            "All chunks below min-token floor (%d); indexing document as-is",
            settings.chunk_min_tokens,
        )
        filtered = raw_chunks

    logger.debug(
        "Chunked text: %d raw → %d after min-token filter",
        len(raw_chunks),
        len(filtered),
    )
    return filtered


def chunk_with_metadata(text: str) -> list[tuple[str, str]]:
    """Split text into chunks, attaching the nearest section heading to each.

    Each chunk inherits the most recently seen markdown heading. When a chunk
    itself starts a new heading, that heading propagates to subsequent chunks.

    Args:
        text: The Markdown text to split.

    Returns:
        List of (chunk_text, section_header) tuples.
        section_header is "" if no heading has appeared yet.
    """
    chunks = chunk(text)
    result: list[tuple[str, str]] = []
    last_header = ""

    for chunk_text in chunks:
        match = _HEADER_RE.search(chunk_text)
        if match:
            last_header = match.group(1).strip()
        result.append((chunk_text, last_header))

    return result
