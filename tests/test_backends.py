"""Tests for OllamaBackend — mock httpx, test translation logic."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from model_serving_api.backends.ollama import OllamaBackend
from model_serving_api.backends.protocol import (
    BackendError,
    BackendOverloadedError,
    InferenceBackend,
    InferenceRequest,
    Message,
    ModelNotFoundError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    return OllamaBackend(base_url="http://test:11434", timeout=10.0)


@pytest.fixture
def request_minimal():
    return InferenceRequest(
        model="llama3.2",
        messages=[Message(role="user", content="hello")],
        request_id="test-123",
    )


@pytest.fixture
def request_with_options():
    return InferenceRequest(
        model="llama3.2",
        messages=[Message(role="user", content="hello")],
        request_id="test-456",
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        stop=["\n"],
    )


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockStreamResponse:
    """Mock for httpx streaming response context manager."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def make_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "http://test:11434/api/chat"),
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_ollama_satisfies_protocol(backend):
    assert isinstance(backend, InferenceBackend)


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def test_build_payload_minimal(backend, request_minimal):
    payload = backend._build_payload(request_minimal, stream=False)
    assert payload == {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }


def test_build_payload_with_options(backend, request_with_options):
    payload = backend._build_payload(request_with_options, stream=True)
    assert payload["stream"] is True
    assert payload["options"] == {
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 100,
    }
    assert payload["stop"] == ["\n"]
    assert "stop" not in payload.get("options", {})


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_success(backend, request_minimal):
    response_data = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": "Hi there!"},
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 3,
        "total_duration": 1_000_000_000,
    }
    mock_response = make_response(response_data)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await backend.generate(request_minimal)

    assert result.content == "Hi there!"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 3
    assert result.total_duration_ns == 1_000_000_000


@pytest.mark.asyncio
async def test_generate_finish_reason_length(backend, request_minimal):
    response_data = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": "truncated"},
        "done": False,
    }
    mock_response = make_response(response_data)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await backend.generate(request_minimal)

    assert result.finish_reason == "length"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None


@pytest.mark.asyncio
async def test_generate_model_not_found(backend, request_minimal):
    mock_response = make_response({}, status_code=404)

    with (
        patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response),
        pytest.raises(ModelNotFoundError, match=r"llama3\.2"),
    ):
        await backend.generate(request_minimal)


@pytest.mark.asyncio
async def test_generate_overloaded(backend, request_minimal):
    mock_response = make_response({}, status_code=503)

    with (
        patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_response),
        pytest.raises(BackendOverloadedError),
    ):
        await backend.generate(request_minimal)


@pytest.mark.asyncio
async def test_generate_connection_error(backend, request_minimal):
    with (
        patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ),
        pytest.raises(BackendError, match="Cannot connect"),
    ):
        await backend.generate(request_minimal)


@pytest.mark.asyncio
async def test_generate_timeout(backend, request_minimal):
    with (
        patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ReadTimeout("timed out"),
        ),
        pytest.raises(BackendError, match="timed out"),
    ):
        await backend.generate(request_minimal)


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_success(backend, request_minimal):
    lines = [
        json.dumps(
            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "Hi"},
                "done": False,
            }
        ),
        json.dumps(
            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "!"},
                "done": True,
                "eval_count": 2,
                "prompt_eval_count": 5,
                "total_duration": 500_000_000,
            }
        ),
    ]
    mock_stream = MockStreamResponse(lines)

    with patch.object(backend._client, "stream", return_value=mock_stream):
        deltas = [delta async for delta in backend.stream(request_minimal)]

    assert len(deltas) == 2
    assert deltas[0].content == "Hi"
    assert deltas[0].finish_reason is None
    assert deltas[0].completion_tokens is None
    assert deltas[1].content == "!"
    assert deltas[1].finish_reason == "stop"
    assert deltas[1].completion_tokens == 2
    assert deltas[1].prompt_tokens == 5


@pytest.mark.asyncio
async def test_stream_skips_empty_lines(backend, request_minimal):
    lines = [
        "",
        json.dumps(
            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "eval_count": 1,
            }
        ),
        "   ",
    ]
    mock_stream = MockStreamResponse(lines)

    with patch.object(backend._client, "stream", return_value=mock_stream):
        deltas = [delta async for delta in backend.stream(request_minimal)]

    assert len(deltas) == 1
    assert deltas[0].content == "ok"


@pytest.mark.asyncio
async def test_stream_error_in_chunk(backend, request_minimal):
    lines = [
        json.dumps({"error": "model not loaded"}),
    ]
    mock_stream = MockStreamResponse(lines)

    with (
        patch.object(backend._client, "stream", return_value=mock_stream),
        pytest.raises(BackendError, match="model not loaded"),
    ):
        async for _ in backend.stream(request_minimal):
            pass


@pytest.mark.asyncio
async def test_stream_model_not_found(backend, request_minimal):
    mock_stream = MockStreamResponse([], status_code=404)

    with (
        patch.object(backend._client, "stream", return_value=mock_stream),
        pytest.raises(ModelNotFoundError, match=r"llama3\.2"),
    ):
        async for _ in backend.stream(request_minimal):
            pass


@pytest.mark.asyncio
async def test_stream_overloaded(backend, request_minimal):
    mock_stream = MockStreamResponse([], status_code=503)

    with (
        patch.object(backend._client, "stream", return_value=mock_stream),
        pytest.raises(BackendOverloadedError),
    ):
        async for _ in backend.stream(request_minimal):
            pass


@pytest.mark.asyncio
async def test_stream_connection_error(backend, request_minimal):
    with (
        patch.object(
            backend._client,
            "stream",
            side_effect=httpx.ConnectError("refused"),
        ),
        pytest.raises(BackendError, match="Cannot connect"),
    ):
        async for _ in backend.stream(request_minimal):
            pass


# ---------------------------------------------------------------------------
# health() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_with_models(backend):
    response_data = {"models": [{"name": "llama3.2"}, {"name": "mistral"}]}
    mock_response = make_response(response_data)

    with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_response):
        health = await backend.health()

    assert health.healthy is True
    assert health.model_loaded is True
    assert health.models_available == ["llama3.2", "mistral"]


@pytest.mark.asyncio
async def test_health_no_models(backend):
    response_data = {"models": []}
    mock_response = make_response(response_data)

    with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_response):
        health = await backend.health()

    assert health.healthy is True
    assert health.model_loaded is False


@pytest.mark.asyncio
async def test_health_unhealthy(backend):
    mock_response = make_response({}, status_code=500)

    with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_response):
        health = await backend.health()

    assert health.healthy is False
    assert health.error is not None


@pytest.mark.asyncio
async def test_health_connection_refused(backend):
    with patch.object(
        backend._client,
        "get",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("refused"),
    ):
        health = await backend.health()

    assert health.healthy is False
    assert health.error is not None
