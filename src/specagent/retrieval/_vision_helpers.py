"""Helper functions and constants for Groq vision API response processing."""

from __future__ import annotations

import re

import httpx

_KNOWN_DIAGRAM_TYPES = frozenset([
    "call_flow",
    "state_machine",
    "block_diagram",
    "flowchart",
    "network_topology",
    "table",
    "screenshot_text",
    "other",
])

_MERMAID_SUBTYPE: dict[str, str] = {
    "call_flow": "sequenceDiagram",
    "state_machine": "stateDiagram-v2",
    "block_diagram": "graph LR",
    "flowchart": "flowchart TD",
    "network_topology": "graph LR",
}


def _is_retryable(exc: BaseException) -> bool:
    """Return ``True`` for transient HTTP errors that warrant a retry."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


def _fix_mermaid_header(content: str, expected_header: str) -> str:
    """Ensure a Mermaid fenced block starts with the expected diagram header keyword.

    If content is already correct, returns it unchanged.
    If the first non-empty line inside the fence is wrong, replaces it.
    """
    fence_match = re.match(r"```mermaid\n([\s\S]*?)```\s*$", content.strip())
    if not fence_match:
        return content  # Not a fenced block — leave for validator to catch
    inner = fence_match.group(1)
    lines = inner.split("\n")
    expected_keyword = expected_header.split()[0]  # e.g. "graph" from "graph LR"
    for i, line in enumerate(lines):
        if line.strip():
            if not line.strip().startswith(expected_keyword):
                lines[i] = expected_header
                return f"```mermaid\n{chr(10).join(lines)}```"
            break
    return content


__all__ = [
    "_KNOWN_DIAGRAM_TYPES",
    "_MERMAID_SUBTYPE",
    "_fix_mermaid_header",
    "_is_retryable",
]
