"""Tests for Prometheus metrics."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from model_serving_api.backends.protocol import (
    BackendError,
    BackendHealth,
    BackendOverloadedError,
    InferenceResponse,
    ModelNotFoundError,
    TokenDelta,
)
from model_serving_api.main import app
from model_serving_api.queue.batcher import BatchDispatcher
from model_serving_api.queue.manager import QueueManager


@pytest.fixture
def mock_backend():
    return AsyncMock()


@pytest.fixture
def client(mock_backend):
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.backend = mock_backend
        app.state.queue_manager = QueueManager(max_size=50)
        app.state.batch_dispatcher = BatchDispatcher(
            backend=mock_backend, max_batch_size=1, wait_ms=0
        )
        yield c


VALID_REQUEST = {
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "hello"}],
}


def _success_response(**overrides):
    defaults = {
        "model": "llama3.2",
        "content": "Hi there!",
        "finish_reason": "stop",
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_duration_ns": 1_000_000_000,
    }
    defaults.update(overrides)
    return InferenceResponse(**defaults)


def _get_metric_value(metrics_text: str, metric_name: str, labels: dict | None = None) -> str:
    """Extract a metric value from Prometheus text format."""
    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        search = f"{metric_name}{{{label_str}}}"
    else:
        search = metric_name

    for line in metrics_text.split("\n"):
        if line.startswith("#"):
            continue
        if search in line:
            return line.split()[-1]
    return ""


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------


def test_metrics_endpoint_exists(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")


def test_metrics_contains_custom_metrics(client):
    response = client.get("/metrics")
    body = response.text
    assert "request_queue_depth" in body
    assert "request_latency_seconds" in body
    assert "time_to_first_token_seconds" in body
    assert "tokens_per_second" in body
    assert "tokens_generated_total" in body
    assert "active_requests" in body
    assert "model_loaded" in body
    assert "inference_errors_total" in body
    assert "queue_wait_seconds" in body
    assert "backend_latency_seconds" in body
    assert "requests_total" in body


# ---------------------------------------------------------------------------
# Non-streaming metrics
# ---------------------------------------------------------------------------


def test_request_latency_recorded(client, mock_backend):
    mock_backend.generate.return_value = _success_response()

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    count = _get_metric_value(metrics, "request_latency_seconds_count", {"model": "llama3.2"})
    assert float(count) >= 1.0


def test_tokens_generated_incremented(client, mock_backend):
    mock_backend.generate.return_value = _success_response(completion_tokens=10)

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    total = _get_metric_value(metrics, "tokens_generated_total", {"model": "llama3.2"})
    assert float(total) >= 10.0


def test_tokens_per_second_recorded(client, mock_backend):
    mock_backend.generate.return_value = _success_response(completion_tokens=10)

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    count = _get_metric_value(metrics, "tokens_per_second_count", {"model": "llama3.2"})
    assert float(count) >= 1.0


def test_active_requests_returns_to_zero(client, mock_backend):
    mock_backend.generate.return_value = _success_response()

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    value = _get_metric_value(metrics, "active_requests", {"model": "llama3.2"})
    assert float(value) == 0.0


def test_latency_recorded_on_error(client, mock_backend):
    from model_serving_api.backends.protocol import BackendError

    mock_backend.generate.side_effect = BackendError("fail")

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    count = _get_metric_value(metrics, "request_latency_seconds_count", {"model": "llama3.2"})
    assert float(count) >= 1.0


# ---------------------------------------------------------------------------
# Streaming metrics
# ---------------------------------------------------------------------------


def test_ttft_recorded_on_streaming(client, mock_backend):
    async def mock_stream(request):
        yield TokenDelta(content="Hello", model="llama3.2")
        yield TokenDelta(
            content="",
            finish_reason="stop",
            model="llama3.2",
            completion_tokens=1,
        )

    mock_backend.stream = mock_stream

    client.post("/v1/chat/completions", json={**VALID_REQUEST, "stream": True})
    metrics = client.get("/metrics").text

    count = _get_metric_value(metrics, "time_to_first_token_seconds_count", {"model": "llama3.2"})
    assert float(count) >= 1.0


def test_stream_tokens_generated(client, mock_backend):
    async def mock_stream(request):
        yield TokenDelta(content="Hello", model="llama3.2")
        yield TokenDelta(content=" world", model="llama3.2")
        yield TokenDelta(
            content="",
            finish_reason="stop",
            model="llama3.2",
            completion_tokens=2,
        )

    mock_backend.stream = mock_stream

    client.post("/v1/chat/completions", json={**VALID_REQUEST, "stream": True})
    metrics = client.get("/metrics").text

    total = _get_metric_value(metrics, "tokens_generated_total", {"model": "llama3.2"})
    assert float(total) >= 2.0


# ---------------------------------------------------------------------------
# Queue depth & model loaded
# ---------------------------------------------------------------------------


def test_queue_depth_reflects_idle(client):
    metrics = client.get("/metrics").text
    value = _get_metric_value(metrics, "request_queue_depth")
    assert float(value) == 0.0


def test_model_loaded_updated_by_ready(client, mock_backend):
    mock_backend.health.return_value = BackendHealth(
        healthy=True,
        model_loaded=True,
        models_available=["llama3.2"],
    )

    client.get("/ready")
    metrics = client.get("/metrics").text

    value = _get_metric_value(metrics, "model_loaded")
    assert float(value) == 1.0


def test_model_not_loaded_gauge(client, mock_backend):
    mock_backend.health.return_value = BackendHealth(
        healthy=True,
        model_loaded=False,
        models_available=[],
    )

    client.get("/ready")
    metrics = client.get("/metrics").text

    value = _get_metric_value(metrics, "model_loaded")
    assert float(value) == 0.0


# ---------------------------------------------------------------------------
# Error counter (inference_errors_total)
# ---------------------------------------------------------------------------


def test_error_counter_backend_error(client, mock_backend):
    mock_backend.generate.side_effect = BackendError("fail")

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    value = _get_metric_value(metrics, "inference_errors_total", {"error_type": "backend_error"})
    assert float(value) >= 1.0


def test_error_counter_model_not_found(client, mock_backend):
    mock_backend.generate.side_effect = ModelNotFoundError("not found")

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    value = _get_metric_value(metrics, "inference_errors_total", {"error_type": "model_not_found"})
    assert float(value) >= 1.0


def test_error_counter_overload(client, mock_backend):
    mock_backend.generate.side_effect = BackendOverloadedError("overloaded")

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    value = _get_metric_value(metrics, "inference_errors_total", {"error_type": "overload"})
    assert float(value) >= 1.0


def test_error_counter_queue_full(mock_backend):
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.backend = mock_backend
        app.state.queue_manager = QueueManager(max_size=0)
        app.state.batch_dispatcher = BatchDispatcher(
            backend=mock_backend, max_batch_size=1, wait_ms=0
        )

        c.post("/v1/chat/completions", json=VALID_REQUEST)
        metrics = c.get("/metrics").text

        value = _get_metric_value(metrics, "inference_errors_total", {"error_type": "queue_full"})
        assert float(value) >= 1.0


# ---------------------------------------------------------------------------
# Queue wait time
# ---------------------------------------------------------------------------


def test_queue_wait_recorded(client, mock_backend):
    mock_backend.generate.return_value = _success_response()

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    count = _get_metric_value(metrics, "queue_wait_seconds_count", {"model": "llama3.2"})
    assert float(count) >= 1.0


def test_queue_wait_recorded_streaming(client, mock_backend):
    async def mock_stream(request):
        yield TokenDelta(content="Hi", model="llama3.2")
        yield TokenDelta(content="", finish_reason="stop", model="llama3.2", completion_tokens=1)

    mock_backend.stream = mock_stream

    client.post("/v1/chat/completions", json={**VALID_REQUEST, "stream": True})
    metrics = client.get("/metrics").text

    count = _get_metric_value(metrics, "queue_wait_seconds_count", {"model": "llama3.2"})
    assert float(count) >= 1.0


# ---------------------------------------------------------------------------
# Backend latency
# ---------------------------------------------------------------------------


def test_backend_latency_recorded(client, mock_backend):
    mock_backend.generate.return_value = _success_response(total_duration_ns=2_000_000_000)

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    count = _get_metric_value(metrics, "backend_latency_seconds_count", {"model": "llama3.2"})
    assert float(count) >= 1.0


def test_backend_latency_not_recorded_when_none(client, mock_backend):
    # Get count before this request
    before_metrics = client.get("/metrics").text
    before = _get_metric_value(
        before_metrics, "backend_latency_seconds_count", {"model": "llama3.2"}
    )
    before_count = float(before) if before else 0.0

    mock_backend.generate.return_value = _success_response(total_duration_ns=None)
    client.post("/v1/chat/completions", json=VALID_REQUEST)

    after_metrics = client.get("/metrics").text
    after = _get_metric_value(after_metrics, "backend_latency_seconds_count", {"model": "llama3.2"})
    after_count = float(after) if after else 0.0

    assert after_count == before_count


def test_backend_latency_recorded_streaming(client, mock_backend):
    async def mock_stream(request):
        yield TokenDelta(content="Hi", model="llama3.2")
        yield TokenDelta(
            content="",
            finish_reason="stop",
            model="llama3.2",
            completion_tokens=1,
            total_duration_ns=1_500_000_000,
        )

    mock_backend.stream = mock_stream

    client.post("/v1/chat/completions", json={**VALID_REQUEST, "stream": True})
    metrics = client.get("/metrics").text

    count = _get_metric_value(metrics, "backend_latency_seconds_count", {"model": "llama3.2"})
    assert float(count) >= 1.0


# ---------------------------------------------------------------------------
# Requests total counter
# ---------------------------------------------------------------------------


def test_requests_total_success(client, mock_backend):
    mock_backend.generate.return_value = _success_response()

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    value = _get_metric_value(metrics, "requests_total", {"model": "llama3.2", "status": "200"})
    assert float(value) >= 1.0


def test_requests_total_error(client, mock_backend):
    mock_backend.generate.side_effect = BackendError("fail")

    client.post("/v1/chat/completions", json=VALID_REQUEST)
    metrics = client.get("/metrics").text

    value = _get_metric_value(metrics, "requests_total", {"model": "llama3.2", "status": "502"})
    assert float(value) >= 1.0
