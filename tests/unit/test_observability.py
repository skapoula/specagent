"""Unit tests for the observability package (models and journal)."""

import datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from specagent.observability.journal import QueryJournal, get_journal
from specagent.observability.models import LLMCallRecord, QueryEvent, RetrievalRecord


def _make_llm_record(
    node: str = "router",
    trace_id: str = "trace-001",
    model: str = "llama-4-scout",
    provider: str = "groq",
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 5,
    total_tokens: int | None = 15,
    inference_ms: float = 120.5,
) -> LLMCallRecord:
    return LLMCallRecord(
        node=node,
        trace_id=trace_id,
        model=model,
        provider=provider,  # type: ignore[arg-type]
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        inference_ms=inference_ms,
    )


def _make_retrieval_record(
    trace_id: str = "trace-001",
    query: str = "What is HARQ?",
    embed_ms: float = 5.0,
    search_ms: float = 12.0,
    num_results: int = 3,
    top_similarity: float | None = 0.88,
    mean_similarity: float | None = 0.78,
    rewrite_index: int = 0,
) -> RetrievalRecord:
    return RetrievalRecord(
        trace_id=trace_id,
        query=query,
        embed_ms=embed_ms,
        search_ms=search_ms,
        num_results=num_results,
        top_similarity=top_similarity,
        mean_similarity=mean_similarity,
        rewrite_index=rewrite_index,
    )


@pytest.mark.unit
class TestLLMCallRecord:
    def test_fields_present(self):
        rec = _make_llm_record()
        assert rec.node == "router"
        assert rec.trace_id == "trace-001"
        assert rec.model == "llama-4-scout"
        assert rec.provider == "groq"
        assert rec.prompt_tokens == 10
        assert rec.completion_tokens == 5
        assert rec.total_tokens == 15
        assert rec.inference_ms == 120.5
        assert isinstance(rec.timestamp, datetime.datetime)

    def test_nullable_tokens(self):
        rec = _make_llm_record(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        assert rec.prompt_tokens is None
        assert rec.completion_tokens is None
        assert rec.total_tokens is None

    def test_json_roundtrip(self):
        rec = _make_llm_record()
        restored = LLMCallRecord.model_validate_json(rec.model_dump_json())
        assert restored.node == rec.node
        assert restored.inference_ms == rec.inference_ms
        assert restored.provider == rec.provider

    def test_node_mutable(self):
        rec = _make_llm_record(node="")
        rec.node = "generator"
        assert rec.node == "generator"

    def test_trace_id_mutable(self):
        rec = _make_llm_record(trace_id="")
        rec.trace_id = "new-trace"
        assert rec.trace_id == "new-trace"


@pytest.mark.unit
class TestRetrievalRecord:
    def test_fields_present(self):
        rec = _make_retrieval_record()
        assert rec.trace_id == "trace-001"
        assert rec.query == "What is HARQ?"
        assert rec.embed_ms == 5.0
        assert rec.search_ms == 12.0
        assert rec.num_results == 3
        assert rec.top_similarity == 0.88
        assert rec.mean_similarity == 0.78
        assert rec.rewrite_index == 0
        assert isinstance(rec.timestamp, datetime.datetime)

    def test_nullable_similarity(self):
        rec = _make_retrieval_record(top_similarity=None, mean_similarity=None)
        assert rec.top_similarity is None
        assert rec.mean_similarity is None

    def test_json_roundtrip(self):
        rec = _make_retrieval_record()
        restored = RetrievalRecord.model_validate_json(rec.model_dump_json())
        assert restored.query == rec.query
        assert restored.num_results == rec.num_results


@pytest.mark.unit
class TestQueryEvent:
    def test_empty_lists(self):
        event = QueryEvent(
            trace_id="t1",
            question="Test?",
            route_decision="retrieve",
            rewrite_count=0,
            num_retrieved=5,
            num_relevant=3,
            hallucination_check="grounded",
            total_ms=450.0,
            llm_calls=[],
            retrievals=[],
        )
        assert event.llm_calls == []
        assert event.retrievals == []
        assert event.trace_id == "t1"

    def test_nullable_hallucination_check(self):
        event = QueryEvent(
            trace_id="t2",
            question="?",
            route_decision="reject",
            rewrite_count=0,
            num_retrieved=0,
            num_relevant=0,
            hallucination_check=None,
            total_ms=50.0,
            llm_calls=[],
            retrievals=[],
        )
        assert event.hallucination_check is None


@pytest.mark.unit
class TestQueryJournal:
    def test_write_creates_file(self, tmp_path: Path):
        journal = QueryJournal(tmp_path / "j")
        journal.write(_make_llm_record())
        files = list((tmp_path / "j").glob("journal-*.jsonl"))
        assert len(files) == 1
        line = files[0].read_text().strip()
        parsed = json.loads(line)
        assert parsed["node"] == "router"

    def test_write_appends_lines(self, tmp_path: Path):
        journal = QueryJournal(tmp_path / "j")
        journal.write(_make_llm_record())
        journal.write(_make_retrieval_record())
        files = list((tmp_path / "j").glob("journal-*.jsonl"))
        assert len(files) == 1
        lines = [ln for ln in files[0].read_text().splitlines() if ln.strip()]
        assert len(lines) == 2
        for ln in lines:
            json.loads(ln)

    def test_write_swallows_os_error(self, tmp_path: Path):
        journal = QueryJournal(tmp_path / "j")
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            journal.write(_make_llm_record())  # must not raise

    def test_get_journal_singleton(self, tmp_path: Path):
        get_journal.cache_clear()
        try:
            with patch("specagent.config.settings") as ms:
                ms.journal_dir = tmp_path / "journal"
                j1 = get_journal()
                j2 = get_journal()
                assert j1 is j2
        finally:
            get_journal.cache_clear()
