"""Tests for QueueManager."""

import asyncio

import pytest

from model_serving_api.queue.manager import QueueFullError, QueueManager


@pytest.mark.asyncio
async def test_acquire_and_release():
    qm = QueueManager(max_size=2)
    assert qm.active_requests == 0

    async with qm.acquire():
        assert qm.active_requests == 1

    assert qm.active_requests == 0


@pytest.mark.asyncio
async def test_properties():
    qm = QueueManager(max_size=5)
    assert qm.max_size == 5
    assert qm.active_requests == 0
    assert qm.is_full is False


@pytest.mark.asyncio
async def test_queue_full_raises():
    qm = QueueManager(max_size=1)

    async with qm.acquire():
        assert qm.is_full is True
        with pytest.raises(QueueFullError) as exc_info:
            async with qm.acquire():
                pass
        assert exc_info.value.max_size == 1


@pytest.mark.asyncio
async def test_release_on_exception():
    qm = QueueManager(max_size=2)

    with pytest.raises(RuntimeError):
        async with qm.acquire():
            raise RuntimeError("boom")

    assert qm.active_requests == 0
    assert qm.is_full is False


@pytest.mark.asyncio
async def test_concurrent_acquires():
    qm = QueueManager(max_size=3)
    results = []

    async def worker(worker_id: int):
        async with qm.acquire():
            results.append(worker_id)
            await asyncio.sleep(0.01)

    await asyncio.gather(worker(1), worker(2), worker(3))
    assert len(results) == 3
    assert qm.active_requests == 0


@pytest.mark.asyncio
async def test_queue_full_error_message():
    error = QueueFullError(10)
    assert "10" in str(error)
    assert error.max_size == 10
