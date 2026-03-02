"""Tests for API request/response Pydantic models."""

import pytest
from pydantic import ValidationError

from specagent.api.models import (
    CitationSchema,
    ErrorResponse,
    HealthResponse,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
)


def _make_metadata(**overrides: object) -> QueryMetadata:
    """Return a valid QueryMetadata instance with sensible defaults."""
    defaults = {
        "rewrites": 0,
        "chunks_retrieved": 5,
        "chunks_used": 3,
        "latency_ms": 1200.0,
        "hallucination_check": "grounded",
    }
    defaults.update(overrides)
    return QueryMetadata(**defaults)


@pytest.mark.unit
class TestQueryRequest:
    def test_valid_question_with_defaults(self):
        req = QueryRequest(question="What is HARQ?")
        assert req.question == "What is HARQ?"
        assert req.verbose is False
        assert req.max_rewrites == 2

    def test_verbose_can_be_set_true(self):
        req = QueryRequest(question="What is HARQ?", verbose=True)
        assert req.verbose is True

    def test_max_rewrites_can_be_overridden(self):
        req = QueryRequest(question="What is HARQ?", max_rewrites=4)
        assert req.max_rewrites == 4

    def test_max_rewrites_zero_allowed(self):
        req = QueryRequest(question="What is HARQ?", max_rewrites=0)
        assert req.max_rewrites == 0

    def test_max_rewrites_five_allowed(self):
        req = QueryRequest(question="What is HARQ?", max_rewrites=5)
        assert req.max_rewrites == 5

    def test_min_length_validation_rejects_two_chars(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")

    def test_min_length_validation_accepts_three_chars(self):
        req = QueryRequest(question="Why")
        assert req.question == "Why"

    def test_max_length_validation_rejects_1001_chars(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="x" * 1001)

    def test_max_length_validation_accepts_1000_chars(self):
        req = QueryRequest(question="x" * 1000)
        assert len(req.question) == 1000

    def test_max_rewrites_upper_bound_rejects_six(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question here", max_rewrites=6)

    def test_max_rewrites_lower_bound_rejects_negative(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question here", max_rewrites=-1)

    def test_missing_question_raises_validation_error(self):
        with pytest.raises(ValidationError):
            QueryRequest()  # type: ignore[call-arg]


@pytest.mark.unit
class TestCitationSchema:
    def test_valid_citation_with_defaults(self):
        c = CitationSchema(spec_id="TS38.331", section="5.3.3")
        assert c.spec_id == "TS38.331"
        assert c.section == "5.3.3"
        assert c.chunk_preview == ""

    def test_with_chunk_preview(self):
        c = CitationSchema(spec_id="TS38.331", section="5.3", chunk_preview="text here")
        assert c.chunk_preview == "text here"

    def test_missing_spec_id_raises_validation_error(self):
        with pytest.raises(ValidationError):
            CitationSchema(section="5.3.3")  # type: ignore[call-arg]

    def test_missing_section_raises_validation_error(self):
        with pytest.raises(ValidationError):
            CitationSchema(spec_id="TS38.331")  # type: ignore[call-arg]


@pytest.mark.unit
class TestQueryMetadata:
    def test_valid_metadata(self):
        m = _make_metadata()
        assert m.rewrites == 0
        assert m.chunks_retrieved == 5
        assert m.chunks_used == 3
        assert m.latency_ms == 1200.0
        assert m.hallucination_check == "grounded"

    def test_all_fields_required(self):
        with pytest.raises(ValidationError):
            QueryMetadata(rewrites=0)  # type: ignore[call-arg]

    def test_float_latency_ms(self):
        m = _make_metadata(latency_ms=2340.5)
        assert m.latency_ms == 2340.5

    def test_hallucination_check_arbitrary_string(self):
        m = _make_metadata(hallucination_check="ungrounded")
        assert m.hallucination_check == "ungrounded"


@pytest.mark.unit
class TestQueryResponse:
    def test_valid_response_with_defaults(self):
        resp = QueryResponse(
            answer="The maximum is 16.",
            confidence=0.9,
            metadata=_make_metadata(),
        )
        assert resp.answer == "The maximum is 16."
        assert resp.citations == []
        assert resp.confidence == 0.9

    def test_citations_default_factory_creates_independent_instances(self):
        r1 = QueryResponse(answer="A.", confidence=0.5, metadata=_make_metadata())
        r2 = QueryResponse(answer="B.", confidence=0.5, metadata=_make_metadata())
        assert r1.citations is not r2.citations

    def test_citations_can_be_populated(self):
        citation = CitationSchema(spec_id="TS38.321", section="5.4.1")
        resp = QueryResponse(
            answer="Answer.",
            confidence=0.8,
            citations=[citation],
            metadata=_make_metadata(),
        )
        assert len(resp.citations) == 1
        assert resp.citations[0].spec_id == "TS38.321"

    def test_confidence_zero_allowed(self):
        resp = QueryResponse(
            answer="Answer.",
            confidence=0.0,
            metadata=_make_metadata(),
        )
        assert resp.confidence == 0.0

    def test_confidence_one_allowed(self):
        resp = QueryResponse(
            answer="Answer.",
            confidence=1.0,
            metadata=_make_metadata(),
        )
        assert resp.confidence == 1.0

    def test_confidence_upper_bound_rejects_above_one(self):
        with pytest.raises(ValidationError):
            QueryResponse(
                answer="ok",
                confidence=1.5,
                metadata=_make_metadata(),
            )

    def test_confidence_lower_bound_rejects_below_zero(self):
        with pytest.raises(ValidationError):
            QueryResponse(
                answer="ok",
                confidence=-0.1,
                metadata=_make_metadata(),
            )

    def test_missing_answer_raises_validation_error(self):
        with pytest.raises(ValidationError):
            QueryResponse(confidence=0.5, metadata=_make_metadata())  # type: ignore[call-arg]

    def test_missing_confidence_raises_validation_error(self):
        with pytest.raises(ValidationError):
            QueryResponse(answer="ok", metadata=_make_metadata())  # type: ignore[call-arg]

    def test_missing_metadata_raises_validation_error(self):
        with pytest.raises(ValidationError):
            QueryResponse(answer="ok", confidence=0.5)  # type: ignore[call-arg]


@pytest.mark.unit
class TestHealthResponse:
    def test_all_fields_set(self):
        h = HealthResponse(status="healthy", version="0.1.0", index_loaded=True)
        assert h.status == "healthy"
        assert h.version == "0.1.0"
        assert h.index_loaded is True

    def test_index_loaded_false(self):
        h = HealthResponse(status="degraded", version="0.1.0", index_loaded=False)
        assert h.index_loaded is False

    def test_missing_status_raises_validation_error(self):
        with pytest.raises(ValidationError):
            HealthResponse(version="0.1.0", index_loaded=True)  # type: ignore[call-arg]

    def test_missing_version_raises_validation_error(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy", index_loaded=True)  # type: ignore[call-arg]

    def test_missing_index_loaded_raises_validation_error(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy", version="0.1.0")  # type: ignore[call-arg]


@pytest.mark.unit
class TestErrorResponse:
    def test_without_reasoning_defaults_to_none(self):
        e = ErrorResponse(error="off_topic", message="Not a 3GPP question.")
        assert e.error == "off_topic"
        assert e.message == "Not a 3GPP question."
        assert e.reasoning is None

    def test_with_reasoning(self):
        e = ErrorResponse(error="off_topic", message="msg", reasoning="because X")
        assert e.reasoning == "because X"

    def test_reasoning_explicit_none(self):
        e = ErrorResponse(error="pipeline_error", message="msg", reasoning=None)
        assert e.reasoning is None

    def test_missing_error_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ErrorResponse(message="msg")  # type: ignore[call-arg]

    def test_missing_message_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ErrorResponse(error="off_topic")  # type: ignore[call-arg]
