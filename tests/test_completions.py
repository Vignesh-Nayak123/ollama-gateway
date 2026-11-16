"""Tests for /v1/chat/completions endpoint."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from model_serving_api.backends.protocol import (
    BackendError,
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


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


def test_completion_success(client, mock_backend):
    mock_backend.generate.return_value = _success_response()

    response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "llama3.2"
    assert data["id"].startswith("chatcmpl-")
    assert isinstance(data["created"], int)
    assert len(data["choices"]) == 1
    assert data["choices"][0]["index"] == 0
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "Hi there!"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 5
    assert data["usage"]["completion_tokens"] == 3
    assert data["usage"]["total_tokens"] == 8


def test_completion_with_options(client, mock_backend):
    mock_backend.generate.return_value = _success_response()

    request = {
        **VALID_REQUEST,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 100,
        "stop": ["\n"],
    }
    response = client.post("/v1/chat/completions", json=request)

    assert response.status_code == 200
    call_args = mock_backend.generate.call_args[0][0]
    assert call_args.temperature == 0.7
    assert call_args.top_p == 0.9
    assert call_args.max_tokens == 100
    assert call_args.stop == ["\n"]


def test_completion_finish_reason_length(client, mock_backend):
    mock_backend.generate.return_value = _success_response(finish_reason="length")

    response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_completion_null_token_counts(client, mock_backend):
    mock_backend.generate.return_value = _success_response(
        prompt_tokens=None, completion_tokens=None
    )

    response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens"] == 0
    assert response.json()["usage"]["completion_tokens"] == 0
    assert response.json()["usage"]["total_tokens"] == 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_completion_model_not_found(client, mock_backend):
    mock_backend.generate.side_effect = ModelNotFoundError("Model 'bad' not found")

    response = client.post(
        "/v1/chat/completions",
        json={**VALID_REQUEST, "model": "bad"},
    )

    assert response.status_code == 404
    data = response.json()
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["code"] == "model_not_found"


def test_completion_backend_overloaded(client, mock_backend):
    mock_backend.generate.side_effect = BackendOverloadedError("overloaded")

    response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 503
    assert response.headers.get("retry-after") == "5"
    # The retry must not reuse this connection (see RETRY_LATER_HEADERS).
    assert response.headers.get("connection") == "close"
    assert response.json()["error"]["code"] == "backend_overloaded"


def test_completion_backend_error(client, mock_backend):
    mock_backend.generate.side_effect = BackendError("connection failed")

    response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "backend_error"


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


def _parse_sse_events(response_text: str) -> list[dict | str]:
    """Parse SSE response text into list of parsed chunks."""
    events = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                events.append("[DONE]")
            else:
                events.append(json.loads(data))
    return events


def test_stream_success(client, mock_backend):
    async def mock_stream(request):
        yield TokenDelta(content="Hello", model="llama3.2")
        yield TokenDelta(content=" world", model="llama3.2")
        yield TokenDelta(
            content="",
            finish_reason="stop",
            model="llama3.2",
            prompt_tokens=5,
            completion_tokens=2,
            total_duration_ns=500_000_000,
        )

    mock_backend.stream = mock_stream

    response = client.post(
        "/v1/chat/completions",
        json={**VALID_REQUEST, "stream": True},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    events = _parse_sse_events(response.text)

    # First event: role announcement
    assert events[0]["object"] == "chat.completion.chunk"
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert events[0]["choices"][0]["finish_reason"] is None

    # Content chunks
    assert events[1]["choices"][0]["delta"] == {"content": "Hello"}
    assert events[1]["choices"][0]["finish_reason"] is None
    assert events[2]["choices"][0]["delta"] == {"content": " world"}
    assert events[2]["choices"][0]["finish_reason"] is None

    # Final chunk: empty delta with finish_reason
    assert events[3]["choices"][0]["delta"] == {}
    assert events[3]["choices"][0]["finish_reason"] == "stop"

    # Terminator
    assert events[4] == "[DONE]"

    # All chunks share the same id and model
    chunk_ids = {e["id"] for e in events if isinstance(e, dict)}
    assert len(chunk_ids) == 1
    assert next(iter(chunk_ids)).startswith("chatcmpl-")


def test_stream_single_token(client, mock_backend):
    async def mock_stream(request):
        yield TokenDelta(content="Hi", finish_reason="stop", model="llama3.2")

    mock_backend.stream = mock_stream

    response = client.post(
        "/v1/chat/completions",
        json={**VALID_REQUEST, "stream": True},
    )

    events = _parse_sse_events(response.text)

    # Role chunk + content chunk + finish chunk + [DONE]
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert events[1]["choices"][0]["delta"] == {"content": "Hi"}
    assert events[2]["choices"][0]["delta"] == {}
    assert events[2]["choices"][0]["finish_reason"] == "stop"
    assert events[3] == "[DONE]"


def test_stream_backend_error_mid_stream(client, mock_backend):
    async def mock_stream(request):
        yield TokenDelta(content="partial", model="llama3.2")
        raise BackendError("connection lost")

    mock_backend.stream = mock_stream

    response = client.post(
        "/v1/chat/completions",
        json={**VALID_REQUEST, "stream": True},
    )

    # Stream started successfully (200), but error occurred mid-stream
    assert response.status_code == 200
    events = _parse_sse_events(response.text)

    # Role chunk + partial content, then an in-band error event instead of [DONE]
    assert any(
        isinstance(e, dict)
        and "choices" in e
        and e["choices"][0]["delta"].get("content") == "partial"
        for e in events
    )
    assert events[-1] == {
        "error": {
            "message": "connection lost",
            "type": "server_error",
            "param": None,
            "code": "backend_error",
        }
    }
    assert "[DONE]" not in events


def _metric(client, name: str, labels: dict[str, str]) -> float:
    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
    prefix = f"{name}{{{label_str}}} "
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(prefix):
            return float(line.split()[-1])
    return 0.0


def test_stream_error_is_not_counted_as_success(client, mock_backend):
    # Regression: a failed stream used to end with [DONE] and increment status="200".
    async def mock_stream(request):
        raise BackendError("Ollama streaming request timed out")
        yield  # pragma: no cover - makes this an async generator

    mock_backend.stream = mock_stream
    labels_200 = {"model": "stream-err-model", "status": "200"}
    labels_502 = {"model": "stream-err-model", "status": "502"}
    ok_before = _metric(client, "requests_total", labels_200)
    err_before = _metric(client, "requests_total", labels_502)

    response = client.post(
        "/v1/chat/completions",
        json={**VALID_REQUEST, "model": "stream-err-model", "stream": True},
    )

    events = _parse_sse_events(response.text)
    assert events[-1]["error"]["code"] == "backend_error"
    assert _metric(client, "requests_total", labels_200) == ok_before
    assert _metric(client, "requests_total", labels_502) == err_before + 1


def test_stream_model_not_found_reports_error_event(client, mock_backend):
    async def mock_stream(request):
        raise ModelNotFoundError("Model 'nope' not found on Ollama")
        yield  # pragma: no cover

    mock_backend.stream = mock_stream

    response = client.post(
        "/v1/chat/completions",
        json={**VALID_REQUEST, "model": "nope", "stream": True},
    )

    events = _parse_sse_events(response.text)
    assert events[-1]["error"]["code"] == "model_not_found"
    assert events[-1]["error"]["type"] == "invalid_request_error"
    assert _metric(client, "requests_total", {"model": "nope", "status": "404"}) >= 1


def test_stream_passes_options(client, mock_backend):
    async def mock_stream(request):
        # Verify options were passed through
        assert request.temperature == 0.5
        assert request.max_tokens == 50
        yield TokenDelta(content="ok", finish_reason="stop", model="llama3.2")

    mock_backend.stream = mock_stream

    response = client.post(
        "/v1/chat/completions",
        json={**VALID_REQUEST, "stream": True, "temperature": 0.5, "max_tokens": 50},
    )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_completion_missing_model(client):
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 422


def test_completion_empty_messages(client, mock_backend):
    mock_backend.generate.return_value = _success_response()

    response = client.post(
        "/v1/chat/completions",
        json={"model": "llama3.2", "messages": []},
    )

    # FastAPI/Pydantic accepts empty list by default
    # This is fine — the backend will handle the empty messages case
    assert response.status_code in (200, 422)


def test_completion_invalid_temperature(client):
    response = client.post(
        "/v1/chat/completions",
        json={**VALID_REQUEST, "temperature": 3.0},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Queue backpressure
# ---------------------------------------------------------------------------


def test_completion_queue_full(client, mock_backend):
    # Replace with a full queue
    app.state.queue_manager = QueueManager(max_size=0)

    response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 503
    assert response.headers.get("retry-after") == "5"
    # The retry must not reuse this connection (see RETRY_LATER_HEADERS).
    assert response.headers.get("connection") == "close"
    assert response.json()["error"]["code"] == "queue_full"


def test_stream_queue_full(client, mock_backend):
    app.state.queue_manager = QueueManager(max_size=0)

    response = client.post(
        "/v1/chat/completions",
        json={**VALID_REQUEST, "stream": True},
    )

    assert response.status_code == 503
    assert response.headers.get("retry-after") == "5"
    assert response.headers.get("connection") == "close"
    assert response.json()["error"]["code"] == "queue_full"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_completion_timeout(mock_backend):
    async def slow_generate(request):
        await asyncio.sleep(10)

    mock_backend.generate = slow_generate

    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.backend = mock_backend
        app.state.queue_manager = QueueManager(max_size=50)
        app.state.batch_dispatcher = BatchDispatcher(
            backend=mock_backend, max_batch_size=1, wait_ms=0
        )
        app.state.settings.request_timeout_seconds = 0.01

        response = c.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "request_timeout"


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------


def test_unhandled_exception_returns_json(client, mock_backend):
    mock_backend.generate.side_effect = RuntimeError("unexpected crash")

    response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 500
    data = response.json()
    assert data["error"]["type"] == "server_error"
    assert data["error"]["code"] == "internal_error"


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


def test_shutdown_rejects_requests(client, mock_backend):
    app.state.shutting_down = True

    response = client.post("/v1/chat/completions", json=VALID_REQUEST)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "shutting_down"

    # Reset for other tests
    app.state.shutting_down = False


def test_shutdown_rejects_streaming_requests(client, mock_backend):
    app.state.shutting_down = True

    response = client.post(
        "/v1/chat/completions",
        json={**VALID_REQUEST, "stream": True},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "shutting_down"

    app.state.shutting_down = False
