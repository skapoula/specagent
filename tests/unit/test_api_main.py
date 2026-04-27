"""Tests for FastAPI application endpoints."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def client():
    """Provide a TestClient with mocked resource initialization."""
    with (
        patch(
            "specagent.api.main.initialize_resources",
            return_value={"store": True, "embedder": True},
        ),
        patch("specagent.api.main.settings") as mock_settings,
    ):
        mock_settings.enable_tracing = False
        from fastapi.testclient import TestClient

        from specagent.api.main import app

        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_with_tracing():
    """Provide a TestClient with tracing enabled."""
    with (
        patch(
            "specagent.api.main.initialize_resources",
            return_value={"store": True, "embedder": True},
        ),
        patch("specagent.api.main.settings") as mock_settings,
    ):
        mock_settings.enable_tracing = True
        from fastapi.testclient import TestClient

        from specagent.api.main import app

        with TestClient(app) as c:
            yield c


@pytest.mark.unit
class TestHealthEndpoint:
    def test_healthy_with_store(self, client):
        with patch("specagent.retrieval.resources.get_store", return_value=MagicMock()):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["index_loaded"] is True

    def test_store_exception_shows_not_loaded(self, client):
        with patch(
            "specagent.retrieval.resources.get_store",
            side_effect=Exception("store down"),
        ):
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["index_loaded"] is False

    def test_store_returns_none_shows_not_loaded(self, client):
        with patch("specagent.retrieval.resources.get_store", return_value=None):
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["index_loaded"] is False

    def test_health_response_fields(self, client):
        with patch("specagent.retrieval.resources.get_store", return_value=MagicMock()):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "index_loaded" in data


@pytest.mark.unit
class TestQueryEndpoint:
    """Tests for the POST /query endpoint."""

    def _make_result(
        self,
        route: str = "retrieve",
        generation: str | None = "The answer is 16.",
        error: str | None = None,
        citations: list | None = None,
        graded_chunks: list | None = None,
        retrieved_chunks: list | None = None,
        rewrite_count: int = 0,
        processing_time_ms: float = 1000.0,
        hallucination_check: str = "grounded",
        average_confidence: float = 0.9,
    ) -> dict[str, object]:
        return {
            "route_decision": route,
            "route_reasoning": "This is a 3GPP question.",
            "generation": generation,
            "citations": citations or [],
            "graded_chunks": graded_chunks or [],
            "retrieved_chunks": retrieved_chunks or [],
            "rewrite_count": rewrite_count,
            "processing_time_ms": processing_time_ms,
            "hallucination_check": hallucination_check,
            "average_confidence": average_confidence,
            "error": error,
        }

    def test_success_returns_200(self, client):
        result = self._make_result()
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 200

    def test_success_response_has_answer(self, client):
        result = self._make_result(generation="The answer is 16.")
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.json()["answer"] == "The answer is 16."

    def test_success_response_has_citations_list(self, client):
        result = self._make_result()
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert isinstance(resp.json()["citations"], list)

    def test_success_response_has_confidence(self, client):
        result = self._make_result(average_confidence=0.9, rewrite_count=0)
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        data = resp.json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_success_response_has_metadata(self, client):
        result = self._make_result()
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        data = resp.json()
        assert "metadata" in data
        meta = data["metadata"]
        assert "rewrites" in meta
        assert "chunks_retrieved" in meta
        assert "chunks_used" in meta
        assert "latency_ms" in meta
        assert "hallucination_check" in meta

    def test_off_topic_route_returns_422(self, client):
        result = self._make_result(route="reject", generation=None)
        result["route_reasoning"] = "Not a 3GPP question."
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is Python?"})
        assert resp.status_code == 422

    def test_off_topic_error_detail(self, client):
        result = self._make_result(route="reject", generation=None)
        result["route_reasoning"] = "Unrelated to 3GPP."
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is Python?"})
        detail = resp.json()["detail"]
        assert detail["error"] == "off_topic"
        assert "3GPP" in detail["message"]
        assert detail["reasoning"] == "Unrelated to 3GPP."

    def test_pipeline_error_returns_500(self, client):
        result = self._make_result(error="LLM call failed")
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 500

    def test_pipeline_error_detail(self, client):
        result = self._make_result(error="LLM call failed")
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        detail = resp.json()["detail"]
        assert detail["error"] == "pipeline_error"
        assert detail["message"] == "An internal error occurred."

    def test_unexpected_exception_returns_500(self, client):
        with patch(
            "specagent.api.main.run_query",
            side_effect=RuntimeError("unexpected boom"),
        ):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 500

    def test_unexpected_exception_detail(self, client):
        with patch(
            "specagent.api.main.run_query",
            side_effect=RuntimeError("unexpected boom"),
        ):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        detail = resp.json()["detail"]
        assert detail["error"] == "internal_error"
        assert detail["message"] == "An internal error occurred."

    def test_pipeline_error_does_not_leak_exception_text(self, client):
        result = self._make_result(error="DB password is hunter2")
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 500
        body = str(resp.json())
        assert "hunter2" not in body
        assert "An internal error occurred." in body

    def test_unhandled_exception_does_not_leak_exception_text(self, client):
        with patch(
            "specagent.api.main.run_query",
            side_effect=RuntimeError("secret_api_key_xyz"),
        ):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 500
        body = str(resp.json())
        assert "secret_api_key_xyz" not in body
        assert "An internal error occurred." in body

    def test_short_question_rejected_by_pydantic(self, client):
        resp = client.post("/query", json={"question": "Hi"})
        assert resp.status_code == 422

    def test_missing_question_rejected(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_citations_in_result_serialized(self, client):
        from specagent.graph.state import Citation

        citation = Citation(
            spec_id="TS38.321",
            section="5.4.1",
            raw_citation="[TS 38.321 §5.4.1]",
            chunk_preview="The UE shall support...",
        )
        result = self._make_result(citations=[citation])
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 200
        citations = resp.json()["citations"]
        assert len(citations) == 1
        assert citations[0]["spec_id"] == "TS38.321"
        assert citations[0]["section"] == "5.4.1"
        assert citations[0]["chunk_preview"] == "The UE shall support..."

    def test_graded_chunks_count_in_metadata(self, client):
        from specagent.graph.state import GradedChunk, RetrievedChunk

        chunk = RetrievedChunk(
            content="HARQ text",
            chunk_id="c1",
            doc_id="d1",
            source="TS38.321.docx",
            title="TS38.321",
            chunk_index=0,
            file_type="docx",
            spec_id="TS38.321",
            section="5.4",
            similarity_score=0.9,
        )
        graded = [
            GradedChunk(chunk=chunk, relevant="yes", confidence=0.9),
            GradedChunk(chunk=chunk, relevant="no", confidence=0.3),
        ]
        result = self._make_result(
            graded_chunks=graded,
            retrieved_chunks=[chunk, chunk],
        )
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert meta["chunks_retrieved"] == 2
        assert meta["chunks_used"] == 1

    def test_ungrounded_claims_surfaced_in_response(self, client):
        result = self._make_result()
        result["ungrounded_claims"] = ["Claim A", "Claim B"]
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 200
        assert resp.json()["ungrounded_claims"] == ["Claim A", "Claim B"]

    def test_hallucination_status_surfaced_in_response(self, client):
        result = self._make_result(hallucination_check="partial")
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 200
        assert resp.json()["hallucination_status"] == "partial"

    def test_ungrounded_claims_defaults_to_empty_when_absent(self, client):
        result = self._make_result()
        # Remove ungrounded_claims from result entirely
        result.pop("ungrounded_claims", None)
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 200
        assert resp.json()["ungrounded_claims"] == []

    def test_hallucination_status_defaults_to_unknown_when_absent(self, client):
        result = self._make_result()
        # Remove hallucination_check from result
        result.pop("hallucination_check", None)
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 200
        assert resp.json()["hallucination_status"] == "unknown"

    def test_query_passes_max_rewrites_to_run_query(self, client):
        result = self._make_result()
        with patch("specagent.api.main.run_query", return_value=result) as mock_run_query:
            client.post("/query", json={"question": "What is HARQ?", "max_rewrites": 0})
        mock_run_query.assert_called_once()
        _, kwargs = mock_run_query.call_args
        assert kwargs.get("max_rewrites") == 0

    def test_generation_none_triggers_response_error(self, client):
        # generation=None is present in the dict so dict.get() returns None,
        # not the fallback string.  Pydantic rejects None for a required str field,
        # which FastAPI surfaces as a 500 response-validation error.
        result = self._make_result(generation=None)
        with patch("specagent.api.main.run_query", return_value=result):
            resp = client.post("/query", json={"question": "What is HARQ?"})
        assert resp.status_code == 500


@pytest.mark.unit
class TestCalculateConfidence:
    """Tests for the _calculate_confidence module-level function."""

    def test_grounded_no_rewrites_returns_base(self):
        from specagent.api.main import _calculate_confidence

        r = {
            "average_confidence": 0.8,
            "hallucination_check": "grounded",
            "rewrite_count": 0,
        }
        assert _calculate_confidence(r) == pytest.approx(0.8)

    def test_not_grounded_halves_confidence(self):
        from specagent.api.main import _calculate_confidence

        r = {
            "average_confidence": 1.0,
            "hallucination_check": "not_grounded",
            "rewrite_count": 0,
        }
        assert _calculate_confidence(r) == pytest.approx(0.5)

    def test_partial_multiplies_by_0_8(self):
        from specagent.api.main import _calculate_confidence

        r = {
            "average_confidence": 1.0,
            "hallucination_check": "partial",
            "rewrite_count": 0,
        }
        assert _calculate_confidence(r) == pytest.approx(0.8)

    def test_rewrite_penalty_applied(self):
        from specagent.api.main import _calculate_confidence

        r = {
            "average_confidence": 1.0,
            "hallucination_check": "grounded",
            "rewrite_count": 2,
        }
        # 1.0 * (1 - 2*0.05) = 1.0 * 0.9 = 0.9
        assert _calculate_confidence(r) == pytest.approx(0.9)

    def test_not_grounded_and_rewrites(self):
        from specagent.api.main import _calculate_confidence

        r = {
            "average_confidence": 1.0,
            "hallucination_check": "not_grounded",
            "rewrite_count": 2,
        }
        # 1.0 * 0.5 * 0.9 = 0.45
        assert _calculate_confidence(r) == pytest.approx(0.45)

    def test_clamp_min_zero(self):
        from specagent.api.main import _calculate_confidence

        r = {
            "average_confidence": 0.0,
            "hallucination_check": "not_grounded",
            "rewrite_count": 20,
        }
        assert _calculate_confidence(r) == 0.0

    def test_clamp_max_one(self):
        from specagent.api.main import _calculate_confidence

        r = {
            "average_confidence": 2.0,
            "hallucination_check": "grounded",
            "rewrite_count": 0,
        }
        assert _calculate_confidence(r) == 1.0

    def test_defaults_when_keys_missing(self):
        from specagent.api.main import _calculate_confidence

        # All keys missing — uses defaults: average_confidence=0.5, grounded, 0 rewrites
        r = {}
        assert _calculate_confidence(r) == pytest.approx(0.5)


@pytest.mark.unit
class TestLifespan:
    """Tests for the lifespan startup/shutdown behaviour."""

    def test_startup_calls_initialize_resources(self):
        call_count = {"n": 0}

        def _fake_init():
            call_count["n"] += 1
            return {"store": True, "embedder": True}

        with (
            patch("specagent.api.main.initialize_resources", side_effect=_fake_init),
            patch("specagent.api.main.settings") as ms,
        ):
            ms.enable_tracing = False
            from fastapi.testclient import TestClient

            from specagent.api.main import app

            with TestClient(app):
                pass

        assert call_count["n"] == 1

    def test_startup_with_tracing_enabled(self):
        with (
            patch(
                "specagent.api.main.initialize_resources",
                return_value={"store": True, "embedder": True},
            ),
            patch("specagent.api.main.settings") as ms,
        ):
            ms.enable_tracing = True
            from fastapi.testclient import TestClient

            from specagent.api.main import app

            # Should not raise even with tracing enabled
            with TestClient(app) as c:
                resp = c.get("/health")
            assert resp.status_code == 200

    def test_startup_failure_raises_runtime_error(self):
        """Exercises lines 52-54: the except branch in lifespan."""
        with patch(
            "specagent.api.main.initialize_resources",
            side_effect=RuntimeError("DB unavailable"),
        ):
            from fastapi.testclient import TestClient

            from specagent.api.main import app

            with pytest.raises(RuntimeError, match="Startup failed"):
                with TestClient(app, raise_server_exceptions=True):
                    pass


# ---------------------------------------------------------------------------
# Issue 12: multi-worker + OCR warning logged at startup (TDD)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiWorkerOcrWarning:
    def test_warning_logged_when_workers_gt_1_and_ocr_enabled(self, caplog) -> None:
        """lifespan logs a WARNING when api_workers > 1 and enable_docx_ocr=True."""
        import logging

        with (
            patch(
                "specagent.api.main.initialize_resources",
                return_value={"store": True, "embedder": True},
            ),
            patch("specagent.api.main.settings") as ms,
            caplog.at_level(logging.WARNING, logger="specagent.api.main"),
        ):
            ms.enable_tracing = False
            ms.enable_langsmith = False
            ms.api_workers = 2
            ms.enable_docx_ocr = True

            from fastapi.testclient import TestClient

            from specagent.api.main import app

            with TestClient(app):
                pass

        assert any(
            "rate limiter" in r.message.lower() or "worker" in r.message.lower()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_no_warning_when_single_worker(self, caplog) -> None:
        """No multi-worker warning when api_workers == 1."""
        import logging

        with (
            patch(
                "specagent.api.main.initialize_resources",
                return_value={"store": True, "embedder": True},
            ),
            patch("specagent.api.main.settings") as ms,
            caplog.at_level(logging.WARNING, logger="specagent.api.main"),
        ):
            ms.enable_tracing = False
            ms.enable_langsmith = False
            ms.api_workers = 1
            ms.enable_docx_ocr = True

            from fastapi.testclient import TestClient

            from specagent.api.main import app

            with TestClient(app):
                pass

        rate_limiter_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "rate limiter" in r.message.lower()
        ]
        assert len(rate_limiter_warnings) == 0
