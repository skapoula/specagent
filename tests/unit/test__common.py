"""Unit tests for specagent.nodes._common helpers."""

import json
from unittest.mock import MagicMock

import pytest

from specagent.graph.state import create_initial_state
from specagent.nodes._common import format_spec_ref, parse_json_object, record_llm_call


@pytest.mark.unit
class TestRecordLLMCall:
    """Tests for the ``record_llm_call`` helper."""

    def test_appends_call_with_node_and_trace_id(self):
        """Helper stamps node name + trace_id and appends to llm_calls."""
        call = MagicMock()
        call.node = ""
        call.trace_id = ""

        llm = MagicMock()
        llm.get_last_call.return_value = call

        state = create_initial_state("q")
        state["trace_id"] = "trace-123"

        record_llm_call(state, llm, "router")

        assert call.node == "router"
        assert call.trace_id == "trace-123"
        assert state["llm_calls"] == [call]

    def test_noop_when_no_last_call(self):
        """Helper is a no-op when the LLM has no recorded call."""
        llm = MagicMock()
        llm.get_last_call.return_value = None

        state = create_initial_state("q")

        record_llm_call(state, llm, "router")

        assert state.get("llm_calls", []) == []

    def test_appends_to_existing_calls(self):
        """Helper preserves existing llm_calls entries."""
        existing = MagicMock()
        new_call = MagicMock()
        new_call.node = ""
        new_call.trace_id = ""

        llm = MagicMock()
        llm.get_last_call.return_value = new_call

        state = create_initial_state("q")
        state["llm_calls"] = [existing]

        record_llm_call(state, llm, "grader")

        assert state["llm_calls"] == [existing, new_call]
        assert new_call.node == "grader"


@pytest.mark.unit
class TestFormatSpecRef:
    """Tests for the ``format_spec_ref`` helper."""

    def test_ts_spec(self):
        """Formats a TS-series spec reference."""
        assert format_spec_ref("TS38.321", "5.4") == "[TS 38.321 §5.4]"

    def test_tr_spec(self):
        """Formats a TR-series spec reference."""
        assert format_spec_ref("TR38.821", "6.1.2") == "[TR 38.821 §6.1.2]"

    def test_non_ts_prefix_defaults_to_tr(self):
        """Non-TS prefixes default to TR label and strip the first two chars.

        The original nodes both used ``startswith("TS")`` and otherwise
        assumed ``TR`` — the helper must preserve that exact behaviour.
        """
        assert format_spec_ref("XX38.999", "1.0") == "[TR 38.999 §1.0]"


@pytest.mark.unit
class TestParseJsonObject:
    """Tests for the ``parse_json_object`` helper."""

    def test_extracts_first_object_from_mixed_text(self):
        """Extracts the JSON object embedded in surrounding text."""
        response = 'prefix {"route": "retrieve", "reasoning": "ok"} suffix'
        assert parse_json_object(response) == {"route": "retrieve", "reasoning": "ok"}

    def test_parses_full_response_when_no_braces_match(self):
        """Falls back to parsing the full response when the regex misses."""
        response = json.dumps({"grounded": "yes", "ungrounded_claims": []})
        assert parse_json_object(response) == {"grounded": "yes", "ungrounded_claims": []}

    def test_raises_on_invalid_json(self):
        """Propagates JSONDecodeError for malformed content."""
        with pytest.raises(json.JSONDecodeError):
            parse_json_object("not json at all")
