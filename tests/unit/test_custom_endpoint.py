"""Tests for CustomEndpointLLM with mocked HTTP."""
import pytest
import requests
from unittest.mock import MagicMock, patch

from specagent.llm.custom_endpoint import CustomEndpointLLM


def _llm(retries: int = 1, delay: float = 0.0) -> CustomEndpointLLM:
    return CustomEndpointLLM(
        "http://test/v1/chat/completions", max_retries=retries, retry_delay=delay
    )


def _ok(text: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.unit
class TestInvoke:
    def test_success(self):
        with patch(
            "specagent.llm.custom_endpoint.requests.post", return_value=_ok("hi")
        ):
            assert _llm().invoke("p") == "hi"

    def test_timing(self):
        with patch(
            "specagent.llm.custom_endpoint.requests.post", return_value=_ok()
        ):
            text, ms = _llm().invoke_with_timing("p")
        assert text == "ok"
        assert isinstance(ms, float)

    def test_retry_on_502(self):
        err = MagicMock()
        err.status_code = 502
        http_err = requests.HTTPError(response=err)
        with patch(
            "specagent.llm.custom_endpoint.requests.post",
            side_effect=[http_err, _ok("retried")],
        ), patch("specagent.llm.custom_endpoint.time.sleep"):
            assert _llm(retries=2).invoke("p") == "retried"

    def test_retry_on_503(self):
        err = MagicMock()
        err.status_code = 503
        http_err = requests.HTTPError(response=err)
        with patch(
            "specagent.llm.custom_endpoint.requests.post",
            side_effect=[http_err, _ok("retried-503")],
        ), patch("specagent.llm.custom_endpoint.time.sleep"):
            assert _llm(retries=2).invoke("p") == "retried-503"

    def test_retry_on_504(self):
        err = MagicMock()
        err.status_code = 504
        http_err = requests.HTTPError(response=err)
        with patch(
            "specagent.llm.custom_endpoint.requests.post",
            side_effect=[http_err, _ok("retried-504")],
        ), patch("specagent.llm.custom_endpoint.time.sleep"):
            assert _llm(retries=2).invoke("p") == "retried-504"

    def test_raises_400(self):
        err = MagicMock()
        err.status_code = 400
        http_err = requests.HTTPError(response=err)
        # 400 is not in retryable codes; raise_for_status raises on the response
        ok_resp = MagicMock()
        ok_resp.raise_for_status.side_effect = http_err
        with patch(
            "specagent.llm.custom_endpoint.requests.post", return_value=ok_resp
        ):
            with pytest.raises(requests.HTTPError):
                _llm().invoke("p")

    def test_retry_on_timeout(self):
        with patch(
            "specagent.llm.custom_endpoint.requests.post",
            side_effect=[requests.Timeout(), _ok()],
        ), patch("specagent.llm.custom_endpoint.time.sleep"):
            assert _llm(retries=2).invoke("p") == "ok"

    def test_all_503_retries_exhausted(self):
        err = MagicMock()
        err.status_code = 503
        http_err = requests.HTTPError(response=err)
        ok_resp = MagicMock()
        ok_resp.raise_for_status.side_effect = http_err
        with patch(
            "specagent.llm.custom_endpoint.requests.post", return_value=ok_resp
        ), patch("specagent.llm.custom_endpoint.time.sleep"):
            with pytest.raises(requests.HTTPError):
                _llm(retries=2).invoke("p")

    def test_connection_error_after_retries(self):
        with patch(
            "specagent.llm.custom_endpoint.requests.post",
            side_effect=requests.ConnectionError("refused"),
        ), patch("specagent.llm.custom_endpoint.time.sleep"):
            with pytest.raises(requests.ConnectionError):
                _llm(retries=2).invoke("p")

    def test_timeout_all_retries_exhausted(self):
        with patch(
            "specagent.llm.custom_endpoint.requests.post",
            side_effect=requests.Timeout(),
        ), patch("specagent.llm.custom_endpoint.time.sleep"):
            with pytest.raises(requests.Timeout):
                _llm(retries=2).invoke("p")

    def test_zero_retries_raises_runtime_error(self):
        """With max_retries=0 the loop never runs and RuntimeError is raised."""
        with patch("specagent.llm.custom_endpoint.requests.post") as mp:
            with pytest.raises(RuntimeError, match="All retry attempts failed"):
                _llm(retries=0).invoke("p")
        mp.assert_not_called()


@pytest.mark.unit
class TestHealthCheck:
    def test_healthy(self):
        resp = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        resp.raise_for_status = MagicMock()
        resp.elapsed.total_seconds.return_value = 0.1
        with patch(
            "specagent.llm.custom_endpoint.requests.post", return_value=resp
        ):
            ok, msg = _llm().health_check()
        assert ok is True
        assert "healthy" in msg.lower()

    def test_timeout(self):
        with patch(
            "specagent.llm.custom_endpoint.requests.post",
            side_effect=requests.Timeout(),
        ):
            ok, msg = _llm().health_check()
        assert ok is False and "timed out" in msg.lower()

    def test_connection_error(self):
        with patch(
            "specagent.llm.custom_endpoint.requests.post",
            side_effect=requests.ConnectionError("refused"),
        ):
            ok, msg = _llm().health_check()
        assert ok is False
        assert "connection failed" in msg.lower()

    def test_http_error(self):
        err = MagicMock()
        err.status_code = 502
        err.reason = "Bad Gateway"
        http_err = requests.HTTPError(response=err)
        ok_resp = MagicMock()
        ok_resp.raise_for_status.side_effect = http_err
        with patch(
            "specagent.llm.custom_endpoint.requests.post", return_value=ok_resp
        ):
            ok, msg = _llm().health_check()
        assert ok is False and "502" in msg

    def test_invalid_structure(self):
        resp = MagicMock()
        resp.json.return_value = {"no_choices": True}
        resp.raise_for_status = MagicMock()
        with patch(
            "specagent.llm.custom_endpoint.requests.post", return_value=resp
        ):
            ok, _ = _llm().health_check()
        assert ok is False

    def test_unexpected_exception(self):
        with patch(
            "specagent.llm.custom_endpoint.requests.post",
            side_effect=ValueError("oops"),
        ):
            ok, msg = _llm().health_check()
        assert ok is False
        assert "oops" in msg.lower()

    def test_empty_choices(self):
        resp = MagicMock()
        resp.json.return_value = {"choices": []}
        resp.raise_for_status = MagicMock()
        with patch(
            "specagent.llm.custom_endpoint.requests.post", return_value=resp
        ):
            ok, _ = _llm().health_check()
        assert ok is False


@pytest.mark.unit
def test_create_custom_llm_explicit_url():
    from specagent.llm.custom_endpoint import create_custom_llm

    llm = create_custom_llm(endpoint_url="http://explicit/v1")
    assert "explicit" in llm.endpoint_url


@pytest.mark.unit
def test_create_custom_llm_custom_temperature_and_tokens():
    from specagent.llm.custom_endpoint import create_custom_llm

    llm = create_custom_llm(
        endpoint_url="http://test/v1", temperature=0.7, max_tokens=512
    )
    assert llm.temperature == 0.7
    assert llm.max_tokens == 512


@pytest.mark.unit
def test_create_custom_llm_from_settings():
    from specagent.llm.custom_endpoint import create_custom_llm

    with patch("specagent.config.settings") as ms:
        ms.custom_endpoint_url = "http://from-settings/v1"
        llm = create_custom_llm()
    assert "from-settings" in llm.endpoint_url


@pytest.mark.unit
def test_create_custom_llm_settings_fallback():
    """When settings has no custom_endpoint_url, getattr fallback is used."""
    from specagent.llm.custom_endpoint import create_custom_llm

    with patch("specagent.config.settings") as ms:
        # Simulate missing attribute so getattr returns the fallback default
        del ms.custom_endpoint_url
        llm = create_custom_llm()
    # The fallback URL contains qwen3 (hardcoded default)
    assert "qwen3" in llm.endpoint_url


@pytest.mark.unit
def test_check_llm_endpoint_health():
    from specagent.llm.custom_endpoint import check_llm_endpoint_health

    with patch("specagent.llm.custom_endpoint.CustomEndpointLLM") as mock_cls, patch(
        "specagent.config.settings"
    ) as ms:
        ms.custom_endpoint_url = "http://test"
        mock_cls.return_value.health_check.return_value = (True, "ok")
        ok, msg = check_llm_endpoint_health()
    assert ok is True


@pytest.mark.unit
def test_check_llm_endpoint_health_with_timeout():
    from specagent.llm.custom_endpoint import check_llm_endpoint_health

    with patch("specagent.llm.custom_endpoint.CustomEndpointLLM") as mock_cls, patch(
        "specagent.config.settings"
    ) as ms:
        ms.custom_endpoint_url = "http://test"
        mock_cls.return_value.health_check.return_value = (False, "timed out after 10s")
        ok, msg = check_llm_endpoint_health(timeout=10)
    assert ok is False
    mock_cls.return_value.health_check.assert_called_once_with(timeout=10)
