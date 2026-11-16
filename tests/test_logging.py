"""Tests for request logging middleware."""

import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from model_serving_api.backends.protocol import InferenceResponse, TokenDelta
from model_serving_api.main import app
from model_serving_api.queue.batcher import BatchDispatcher
from model_serving_api.queue.manager import QueueManager


@pytest.fixture
def mock_backend():
    return AsyncMock()


@pytest.fixture
def client(mock_backend):
    app.state.backend = mock_backend
    app.state.queue_manager = QueueManager(max_size=50)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_request_id_header(client):
    response = client.get("/health")
    assert "x-request-id" in response.headers
    request_id = response.headers["x-request-id"]
    # Should be a valid UUID format (8-4-4-4-12 hex chars)
    assert len(request_id) == 36
    assert request_id.count("-") == 4


def test_request_id_unique_per_request(client):
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


def test_single_request_id_flows_to_backend_and_completion_id(client, mock_backend):
    # Regression: the middleware logged one UUID while the endpoint generated a separate
    # chatcmpl-... id for backend/batcher logs, so a request's log lines didn't correlate.
    mock_backend.generate.return_value = InferenceResponse(
        model="llama3.2", content="hi", finish_reason="stop", completion_tokens=1
    )
    app.state.backend = mock_backend
    app.state.batch_dispatcher = BatchDispatcher(backend=mock_backend, max_batch_size=1, wait_ms=0)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    header_id = response.headers["x-request-id"]
    backend_request = mock_backend.generate.await_args.args[0]
    assert backend_request.request_id == header_id
    assert response.json()["id"] == "chatcmpl-" + header_id.replace("-", "")


def test_single_request_id_in_stream_chunks(client, mock_backend):
    seen: list[str] = []

    async def mock_stream(request):
        seen.append(request.request_id)
        yield TokenDelta(content="hi", finish_reason="stop", model="llama3.2")

    mock_backend.stream = mock_stream
    app.state.backend = mock_backend

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    header_id = response.headers["x-request-id"]
    assert seen == [header_id]
    chunk = json.loads(response.text.split("data: ", 2)[1].split("\n")[0])
    assert chunk["id"] == "chatcmpl-" + header_id.replace("-", "")
