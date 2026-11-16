"""Naive request batching — collect requests and dispatch concurrently."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from model_serving_api.backends.protocol import (
        InferenceBackend,
        InferenceRequest,
        InferenceResponse,
    )

logger = structlog.get_logger(__name__)


class BatchDispatcher:
    """Collect non-streaming requests and dispatch in batches.

    Requests are accumulated until either ``max_batch_size`` requests are
    pending or ``wait_ms`` milliseconds have elapsed, whichever comes first.
    The batch is then dispatched concurrently via ``asyncio.gather``.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        max_batch_size: int = 8,
        wait_ms: int = 100,
    ) -> None:
        self._backend = backend
        self._max_batch_size = max_batch_size
        self._wait_ms = wait_ms
        self._pending: list[tuple[InferenceRequest, asyncio.Future[InferenceResponse]]] = []
        self._timer_task: asyncio.Task[None] | None = None
        self._batch_task: asyncio.Task[None] | None = None

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """Submit a request for batched execution.

        The caller awaits the returned future, which resolves when the batch
        containing this request has been dispatched and executed.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResponse] = loop.create_future()
        self._pending.append((request, future))

        if len(self._pending) >= self._max_batch_size:
            self._dispatch_batch(trigger="size")
        elif self._timer_task is None:
            self._timer_task = asyncio.create_task(self._wait_and_dispatch())

        return await future

    async def _wait_and_dispatch(self) -> None:
        """Wait for the batch window, then dispatch whatever has accumulated."""
        await asyncio.sleep(self._wait_ms / 1000)
        self._dispatch_batch(trigger="timeout")

    def _dispatch_batch(self, trigger: str = "size") -> None:
        """Take the pending list, cancel the timer, and fire off execution."""
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None

        if not self._pending:
            return

        batch = self._pending[:]
        self._pending.clear()

        self._batch_task = asyncio.create_task(self._execute_batch(batch, trigger))

    async def _execute_batch(
        self,
        batch: list[tuple[InferenceRequest, asyncio.Future[InferenceResponse]]],
        trigger: str = "size",
    ) -> None:
        """Execute all requests in the batch concurrently."""
        logger.info(
            "batch_dispatched",
            batch_size=len(batch),
            models=[req.model for req, _ in batch],
            trigger=trigger,
        )

        results = await asyncio.gather(
            *(self._backend.generate(req) for req, _ in batch),
            return_exceptions=True,
        )

        for (request, future), result in zip(batch, results, strict=True):
            if future.cancelled():
                continue
            if isinstance(result, BaseException):
                logger.warning(
                    "batch_item_failed",
                    request_id=request.request_id,
                    model=request.model,
                    error_type=type(result).__name__,
                    error=str(result),
                )
                future.set_exception(result)
            else:
                future.set_result(result)
