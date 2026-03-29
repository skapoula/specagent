"""Post-processing transforms applied to converted Markdown before chunking.

All transforms are applied in a fixed order by :func:`postprocess`:

1. :func:`_normalize_nbsp` — replace non-breaking spaces with regular spaces
2. :func:`_strip_toc` — remove Table of Contents before the first body heading
3. :func:`_strip_change_history` — remove change history annex from the end
4. :func:`_fix_annex_headings` — prepend ``#`` to bare "Annex X" heading lines
"""

import logging
import re

logger = logging.getLogger(__name__)

# Matches a top-level Markdown heading ("# " followed by a non-space char) at
# start of a line.  Used to detect where the document body begins after the TOC.
_FIRST_H1_RE = re.compile(r"^# \S", re.MULTILINE)

# Matches the Annex L heading line rendered as a top-level Markdown heading
# (present after _fix_annex_headings runs, but _strip_change_history runs
# *before* that transform — so we need both patterns as fallbacks).
# Primary: bare "Annex L" at start of line (pre-heading-fix form).
_CHANGE_HISTORY_ANNEX_RE = re.compile(
    r"^Annex\s+L\b",
    re.MULTILINE,
)
# Secondary: "# Annex L" heading form (in case order changes in the future).
_CHANGE_HISTORY_H1_RE = re.compile(
    r"^# Annex\s+L\b",
    re.MULTILINE,
)
# Tertiary fallback: the rendered table header row "| Change history |".
_CHANGE_HISTORY_TABLE_RE = re.compile(
    r"^\|[^\n]*Change history[^\n]*\|",
    re.MULTILINE | re.IGNORECASE,
)


def postprocess(text: str) -> str:
    """Apply all quality-improvement transforms to converted Markdown.

    Transforms are applied in a fixed order designed to maximise correctness:
    non-breaking space normalisation runs first so that subsequent regex
    patterns can rely on regular spaces; TOC and change-history stripping run
    before heading fixes so the heading fixer only processes body content.

    Args:
        text: Raw Markdown string produced by a file converter.

    Returns:
        Cleaned Markdown string with noise removed and headings normalised.
    """
    text = _normalize_nbsp(text)
    text = _strip_toc(text)
    text = _strip_change_history(text)
    text = _fix_annex_headings(text)
    return text


def _normalize_nbsp(text: str) -> str:
    """Replace non-breaking spaces (U+00A0) with regular spaces."""
    return text.replace("\xa0", " ")


def _strip_toc(text: str) -> str:
    """Remove Table of Contents content before the first top-level heading.

    3GPP documents typically begin with a multi-page TOC followed by
    "# Foreword" or "# 1 Scope".  The TOC lines add noise to retrieval
    because they duplicate every section title with a trailing page number.

    If no top-level heading is found, the text is returned unchanged.

    Args:
        text: Converted Markdown text, possibly with a leading TOC.

    Returns:
        Text starting at the first ``# `` heading, or the original text if
        no such heading exists.
    """
    match = _FIRST_H1_RE.search(text)
    if match is None:
        logger.debug("_strip_toc: no top-level heading found; returning unchanged")
        return text
    removed = match.start()
    if removed > 0:
        logger.debug("_strip_toc: removed %d chars before first heading", removed)
    return text[match.start():]


def _strip_change_history(text: str) -> str:
    """Remove the change history annex from the end of the document.

    3GPP specifications end with Annex L (change history), a dense table of
    version-tracking rows that carry no value for question-answering.

    Three patterns are tried in order of specificity:
    1. Bare "Annex L" at start of line (pre-heading-fix plain-text form).
    2. "# Annex L" heading (if heading fix ran first).
    3. A table row containing "Change history" (fallback for non-3GPP docs).

    If no pattern matches, the text is returned unchanged.

    Args:
        text: Converted Markdown text, possibly with a trailing change history.

    Returns:
        Text with the change history section removed and trailing whitespace
        stripped, or the original text if no change history section is found.
    """
    for pattern in (
        _CHANGE_HISTORY_ANNEX_RE,
        _CHANGE_HISTORY_H1_RE,
        _CHANGE_HISTORY_TABLE_RE,
    ):
        match = pattern.search(text)
        if match:
            stripped = text[: match.start()].rstrip()
            logger.debug(
                "_strip_change_history: removed %d chars from position %d",
                len(text) - len(stripped),
                match.start(),
            )
            return stripped
    logger.debug("_strip_change_history: no change history section found")
    return text


def _fix_annex_headings(text: str) -> str:
    """Prepend ``#`` to Annex heading lines that lack a Markdown heading marker.

    MarkItDown does not map the Word "Annex Heading" paragraph style to a
    Markdown heading.  This transform detects lines that begin with
    ``Annex <UppercaseLetter>`` and are not already prefixed with ``#``,
    and promotes them to top-level headings.

    Only lines at the very start of a line are affected; occurrences of
    "Annex" in the middle of a sentence are left untouched.

    Args:
        text: Converted Markdown text.

    Returns:
        Text with Annex heading lines promoted to ``# Annex ...`` headings.
    """
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if (
            stripped.startswith("Annex ")
            and len(stripped) > 7
            and stripped[6].isupper()
            and not stripped.startswith("#")
        ):
            line = "# " + stripped
        result.append(line)
    return "".join(result)
