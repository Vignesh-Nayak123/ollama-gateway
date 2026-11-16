"""Tests for BatchDispatcher."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from model_serving_api.backends.protocol import (
    BackendError,
    InferenceRequest,
    InferenceResponse,
    Message,
)
from model_serving_api.queue.batcher import BatchDispatcher


def _make_request(model: str = "llama3.2", content: str = "hello") -> InferenceRequest:
    return InferenceRequest(
        model=model,
        messages=[Message(role="user", content=content)],
        request_id=f"test-{content}",
    )


def _make_response(content: str = "Hi!", **overrides) -> InferenceResponse:
    defaults = {
        "model": "llama3.2",
        "content": content,
        "finish_reason": "stop",
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_duration_ns": 100_000_000,
    }
    defaults.update(overrides)
    return InferenceResponse(**defaults)


@pytest.fixture
def mock_backend():
    return AsyncMock()


# ---------------------------------------------------------------------------
# Basic dispatch
# ---------------------------------------------------------------------------


async def test_single_request_dispatched(mock_backend):
    mock_backend.generate.return_value = _make_response()
    batcher = BatchDispatcher(mock_backend, max_batch_size=8, wait_ms=10)

    result = await batcher.submit(_make_request())

    assert result.content == "Hi!"
    mock_backend.generate.assert_called_once()


async def test_immediate_dispatch_when_batch_full(mock_backend):
    responses = [_make_response(content=f"resp-{i}") for i in range(3)]
    mock_backend.generate.side_effect = responses

    batcher = BatchDispatcher(mock_backend, max_batch_size=3, wait_ms=1000)

    # Submit 3 requests concurrently — should trigger immediate dispatch
    results = await asyncio.gather(
        batcher.submit(_make_request(content="a")),
        batcher.submit(_make_request(content="b")),
        batcher.submit(_make_request(content="c")),
    )

    assert len(results) == 3
    assert mock_backend.generate.call_count == 3


async def test_timer_dispatch_before_batch_full(mock_backend):
    mock_backend.generate.return_value = _make_response()
    batcher = BatchDispatcher(mock_backend, max_batch_size=10, wait_ms=10)

    # Submit 2 requests (batch_size=10 won't trigger), timer will
    results = await asyncio.gather(
        batcher.submit(_make_request(content="a")),
        batcher.submit(_make_request(content="b")),
    )

    assert len(results) == 2
    assert mock_backend.generate.call_count == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_error_in_one_request_does_not_affect_others(mock_backend):
    mock_backend.generate.side_effect = [
        _make_response(content="ok"),
        BackendError("fail"),
        _make_response(content="also ok"),
    ]

    batcher = BatchDispatcher(mock_backend, max_batch_size=3, wait_ms=1000)

    tasks = [
        asyncio.create_task(batcher.submit(_make_request(content="a"))),
        asyncio.create_task(batcher.submit(_make_request(content="b"))),
        asyncio.create_task(batcher.submit(_make_request(content="c"))),
    ]

    # Wait for all tasks
    done, _ = await asyncio.wait(tasks, timeout=5)
    assert len(done) == 3

    # First and third succeed, second raises
    assert tasks[0].result().content == "ok"
    with pytest.raises(BackendError):
        tasks[1].result()
    assert tasks[2].result().content == "also ok"


async def test_each_request_gets_correct_result(mock_backend):
    async def generate_with_echo(request):
        return _make_response(content=f"echo-{request.messages[0].content}")

    mock_backend.generate = generate_with_echo

    batcher = BatchDispatcher(mock_backend, max_batch_size=3, wait_ms=1000)

    results = await asyncio.gather(
        batcher.submit(_make_request(content="alpha")),
        batcher.submit(_make_request(content="beta")),
        batcher.submit(_make_request(content="gamma")),
    )

    contents = {r.content for r in results}
    assert contents == {"echo-alpha", "echo-beta", "echo-gamma"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_batch_size_one_dispatches_immediately(mock_backend):
    mock_backend.generate.return_value = _make_response()
    batcher = BatchDispatcher(mock_backend, max_batch_size=1, wait_ms=1000)

    result = await batcher.submit(_make_request())

    assert result.content == "Hi!"
    mock_backend.generate.assert_called_once()
