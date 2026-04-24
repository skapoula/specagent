"""Shared helpers for LangGraph nodes.

Small utilities that multiple nodes would otherwise duplicate:

- ``record_llm_call`` captures the last LLM call and appends it to ``state["llm_calls"]``.
- ``format_spec_ref`` formats a ``[TS XX.XXX §Y.Z]`` citation string from a spec_id + section.
- ``parse_json_object`` extracts the first JSON object from an LLM response string.
"""

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from specagent.graph.state import GraphState
    from specagent.llm.factory import LLMProtocol


# Matches the first JSON object in a string (non-greedy across newlines).
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def record_llm_call(state: "GraphState", llm: "LLMProtocol", node_name: str) -> None:
    """Capture the LLM's last call record and append it to ``state["llm_calls"]``.

    Args:
        state: The current graph state; mutated in place.
        llm: The LLM client whose most-recent call should be recorded.
        node_name: Node identifier to attach to the record (e.g. ``"router"``).
    """
    call = llm.get_last_call()
    if call is None:
        return
    call.node = node_name
    call.trace_id = state.get("trace_id", "")
    state["llm_calls"] = [*list(state.get("llm_calls", [])), call]


def format_spec_ref(spec_id: str, section: str) -> str:
    """Format a ``[TS XX.XXX §Y.Z]`` / ``[TR XX.XXX §Y.Z]`` citation reference.

    Strips the leading series prefix (``TS`` or ``TR``) from ``spec_id`` so the
    series label and numeric part can be joined with a single space — matching
    the format used throughout generator prompts and hallucination sources.
    """
    prefix = "TS" if spec_id.startswith("TS") else "TR"
    spec_num = spec_id[len(prefix) :]
    return f"[{prefix} {spec_num} §{section}]"


def parse_json_object(response: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from an LLM response string.

    Falls back to parsing the entire response when no ``{...}`` substring is
    found — mirroring the tolerant parsing the router and grader previously
    duplicated inline.
    """
    match = _JSON_OBJECT_RE.search(response)
    raw = match.group(0) if match else response
    return json.loads(raw)
