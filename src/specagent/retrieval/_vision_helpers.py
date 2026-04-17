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

# 429 responses with Retry-After strictly beyond this threshold signal daily
# quota exhaustion (not a per-minute TPM spike). Retrying would be futile
# and would waste the remaining quota budget, so _is_retryable returns False.
_MAX_RETRY_AFTER: float = 3600.0  # 1 hour


def _is_retryable(exc: BaseException) -> bool:
    """Return ``True`` for transient HTTP errors that warrant a retry.

    For 429 responses, inspects the ``Retry-After`` header:

    * Missing or ≤ 3600 s → transient TPM/RPM rate limit → retry.
    * > 3600 s → daily quota exhausted → return ``False`` so tenacity stops
      immediately rather than burning more API calls and quota.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 429:
            retry_after_raw = exc.response.headers.get("retry-after", "")
            try:
                if float(retry_after_raw) > _MAX_RETRY_AFTER:
                    return False
            except (ValueError, TypeError):
                pass
            return True
        return exc.response.status_code in (503, 504)
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
    "_MAX_RETRY_AFTER",
    "_MERMAID_SUBTYPE",
    "_fix_mermaid_header",
    "_is_retryable",
]
