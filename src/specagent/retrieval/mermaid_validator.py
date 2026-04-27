"""Mermaid diagram structural validator.

Tier 1 (always): Pure Python regex checks for fenced block, header,
content, and bracket balance.

Tier 2 (opt-in): mmdc subprocess validation when
settings.mermaid_validate_with_mmdc is True.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from specagent.config import settings

logger = logging.getLogger(__name__)

_FENCED_MERMAID_RE = re.compile(r"```mermaid\n([\s\S]*?)```", re.MULTILINE)

_VALID_DIAGRAM_HEADERS = frozenset(
    [
        "sequenceDiagram",
        "stateDiagram-v2",
        "graph",
        "flowchart",
        "classDiagram",
        "erDiagram",
        "gantt",
        "pie",
        "gitGraph",
        "mindmap",
        "timeline",
        "xychart-beta",
    ]
)


def validate_mermaid(content: str) -> tuple[bool, str]:
    """Validate a fenced Mermaid code block.

    Runs Tier 1 structural checks unconditionally. Optionally runs
    Tier 2 mmdc subprocess validation when settings.mermaid_validate_with_mmdc
    is True and mmdc is available on PATH.

    Args:
        content: A string containing a ```mermaid ... ``` fenced code block.

    Returns:
        (True, "") if valid.
        (False, reason) if invalid, where reason describes the failure.
    """
    inner = _extract_inner(content)
    if inner is None:
        return False, "Content does not contain a ```mermaid ... ``` fenced block."

    if not _check_header(inner):
        first = next((line.strip() for line in inner.split("\n") if line.strip()), "")
        return False, f"Unknown Mermaid diagram type on first line: {first!r}."

    if not _check_has_content(inner):
        return False, "Diagram has no content lines beyond the header (ignoring %% comments)."

    if not _check_bracket_balance(inner):
        return False, "Unbalanced brackets, parentheses, or braces in diagram."

    if settings.mermaid_validate_with_mmdc:
        return _check_with_mmdc(inner)

    return True, ""


def _extract_inner(content: str) -> str | None:
    """Return the body between ```mermaid and ``` fences, or None."""
    match = _FENCED_MERMAID_RE.search(content)
    return match.group(1) if match else None


def _check_header(inner: str) -> bool:
    """Return True if first non-empty line starts with a known Mermaid diagram keyword."""
    for line in inner.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        return any(
            stripped == kw or stripped.startswith(kw + " ") or stripped.startswith(kw + "\t")
            for kw in _VALID_DIAGRAM_HEADERS
        )
    return False


def _check_has_content(inner: str) -> bool:
    """Return True if at least 2 non-empty, non-comment lines exist."""
    count = 0
    for line in inner.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("%%"):
            count += 1
        if count >= 2:
            return True
    return False


def _check_bracket_balance(inner: str) -> bool:
    """Return True if brackets, parentheses, and braces balance.

    Skips %% comment lines. Handles both single-quoted and double-quoted strings.
    """
    opens = {"[": "]", "(": ")", "{": "}"}
    closes = {v: k for k, v in opens.items()}
    stack: list[str] = []
    in_string: str | None = None  # None, '"', or "'"

    for line in inner.splitlines():
        if line.strip().startswith("%%"):
            continue
        for char in line:
            if in_string:
                if char == in_string:
                    in_string = None
            elif char in ('"', "'"):
                in_string = char
            elif char in opens:
                stack.append(opens[char])
            elif char in closes:
                if not stack or stack[-1] != char:
                    return False
                stack.pop()
    return len(stack) == 0


def _check_with_mmdc(inner: str) -> tuple[bool, str]:
    """Validate using the mermaid-cli mmdc subprocess.

    Returns (True, "") when mmdc is absent (FileNotFoundError) or times out —
    missing tooling must not fail validation.
    """
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mmd", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(inner)
            tmp_path = Path(tmp.name)
        try:
            result = subprocess.run(
                ["mmdc", "-i", str(tmp_path), "-o", "/dev/null"],
                capture_output=True,
                text=True,
                timeout=settings.mermaid_mmdc_timeout,
                check=False,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        if result.returncode != 0:
            return False, result.stderr.strip() or "mmdc validation failed."
        return True, ""
    except FileNotFoundError:
        logger.debug("mmdc not found on PATH — skipping Tier 2 Mermaid validation.")
        return True, ""
    except subprocess.TimeoutExpired:
        logger.warning(
            "mmdc validation timed out after %ds — treating as valid.",
            settings.mermaid_mmdc_timeout,
        )
        return True, ""
    except OSError:
        logger.debug("mmdc subprocess OS error — skipping Tier 2 Mermaid validation.")
        return True, ""
