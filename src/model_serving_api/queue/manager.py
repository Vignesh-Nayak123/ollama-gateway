"""Bounded request queue with backpressure."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class QueueFullError(Exception):
    """Raised when the request queue is at capacity."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        super().__init__(f"Queue full ({max_size} active requests)")


class QueueManager:
    """Bounded concurrency limiter for inference requests.

    Tracks active requests and rejects new ones when at capacity.
    asyncio is single-threaded, so the counter is safe without locks.
    """

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._active = 0

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Acquire a slot. Raises QueueFullError if at capacity."""
        if self._active >= self._max_size:
            raise QueueFullError(self._max_size)
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1

    @property
    def active_requests(self) -> int:
        return self._active

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def is_full(self) -> bool:
        return self._active >= self._max_size
