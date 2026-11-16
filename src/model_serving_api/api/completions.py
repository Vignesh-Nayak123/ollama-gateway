"""OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import asyncio
import time
import uuid

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from model_serving_api.backends.protocol import (
    BackendError,
    BackendOverloadedError,
    InferenceRequest,
    Message,
    ModelNotFoundError,
)
from model_serving_api.metrics.prometheus import (
    active_requests,
    backend_latency_seconds,
    inference_errors_total,
    queue_wait_seconds,
    request_latency_seconds,
    requests_total,
    time_to_first_token_seconds,
    tokens_generated_total,
    tokens_per_second,
)
from model_serving_api.queue.manager import QueueFullError
from model_serving_api.streaming.sse import StreamStats, stream_chat_completion

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["completions"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1)
    stop: list[str] | None = None
    stream: bool = False


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Headers for "retry later" 503s. Connection: close stops clients from reusing this
# connection for the retry: with Retry-After equal to uvicorn's 5 s idle keep-alive
# timeout, a client that waits exactly Retry-After sends its retry on a pooled
# connection just as the server closes it, and the retry fails with a connection reset
# before it reaches the app (~13% of requests in the backpressure load test).
RETRY_LATER_HEADERS = {"Retry-After": "5", "Connection": "close"}


def _completion_id(request_id: str) -> str:
    """Derive the OpenAI-style completion id from the request id, so they can be matched."""
    return f"chatcmpl-{request_id.replace('-', '')}"


def _error_response(
    status_code: int,
    message: str,
    error_type: str = "server_error",
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ErrorDetail(message=message, type=error_type, code=code)
        ).model_dump(),
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/chat/completions")
async def create_chat_completion(
    body: ChatCompletionRequest,
    http_request: Request,
):
    if getattr(http_request.app.state, "shutting_down", False):
        return _error_response(
            status_code=503,
            message="Server is shutting down",
            error_type="server_error",
            code="shutting_down",
        )

    backend = http_request.app.state.backend
    queue_manager = http_request.app.state.queue_manager
    # One id per request: the middleware's X-Request-ID is used in every log line
    # (including backend/batcher logs) and the completion id is derived from it.
    request_id = getattr(http_request.state, "request_id", None) or str(uuid.uuid4())
    completion_id = _completion_id(request_id)
    created = int(time.time())

    structlog.contextvars.bind_contextvars(
        model=body.model,
        stream=body.stream,
        message_count=len(body.messages),
    )

    inference_request = InferenceRequest(
        model=body.model,
        messages=[Message(role=m.role, content=m.content) for m in body.messages],
        request_id=request_id,
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
        stop=body.stop,
    )

    timeout_seconds = http_request.app.state.settings.request_timeout_seconds

    if body.stream:
        # Check queue capacity before streaming. The generator holds the
        # slot for the entire duration of the stream — releasing only when
        # the response is fully sent or the client disconnects.
        if queue_manager.is_full:
            inference_errors_total.labels(error_type="queue_full").inc()
            requests_total.labels(model=body.model, status="503").inc()
            logger.warning(
                "queue_rejected",
                model=body.model,
                queue_depth=queue_manager.active_requests,
                queue_max=queue_manager.max_size,
                stream=True,
            )
            return _error_response(
                status_code=503,
                message="Server is at capacity. Please retry later.",
                error_type="server_error",
                code="queue_full",
                headers=RETRY_LATER_HEADERS,
            )

        async def _guarded_stream():
            start_time = time.perf_counter()
            chunk_index = 0
            stats = StreamStats()
            async with queue_manager.acquire():
                queue_wait_seconds.labels(model=body.model).observe(
                    time.perf_counter() - start_time
                )
                active_requests.labels(model=body.model).inc()
                try:
                    async for chunk in stream_chat_completion(
                        backend=backend,
                        inference_request=inference_request,
                        request_id=completion_id,
                        created=created,
                        stats=stats,
                    ):
                        if chunk_index == 1 and chunk != "[DONE]":
                            time_to_first_token_seconds.labels(model=body.model).observe(
                                time.perf_counter() - start_time
                            )
                        chunk_index += 1
                        yield chunk
                finally:
                    duration = time.perf_counter() - start_time
                    active_requests.labels(model=body.model).dec()
                    request_latency_seconds.labels(model=body.model).observe(duration)
                    if stats.error_status is not None:
                        inference_errors_total.labels(error_type=stats.error_type).inc()
                    requests_total.labels(
                        model=body.model, status=stats.error_status or "200"
                    ).inc()
                    if stats.total_duration_ns is not None:
                        backend_latency_seconds.labels(model=body.model).observe(
                            stats.total_duration_ns / 1e9
                        )
                    if stats.completion_tokens > 0:
                        tokens_generated_total.labels(model=body.model).inc(stats.completion_tokens)
                        if duration > 0:
                            tokens_per_second.labels(model=body.model).observe(
                                stats.completion_tokens / duration
                            )

        return EventSourceResponse(_guarded_stream(), media_type="text/event-stream")

    start_time = time.perf_counter()
    try:
        async with queue_manager.acquire():
            queue_wait_seconds.labels(model=body.model).observe(time.perf_counter() - start_time)
            active_requests.labels(model=body.model).inc()
            try:
                try:
                    batch_dispatcher = http_request.app.state.batch_dispatcher
                    async with asyncio.timeout(timeout_seconds):
                        result = await batch_dispatcher.submit(inference_request)
                except TimeoutError:
                    inference_errors_total.labels(error_type="timeout").inc()
                    requests_total.labels(model=body.model, status="504").inc()
                    return _error_response(
                        status_code=504,
                        message=f"Request timed out after {timeout_seconds}s",
                        error_type="server_error",
                        code="request_timeout",
                    )
                except ModelNotFoundError as exc:
                    inference_errors_total.labels(error_type="model_not_found").inc()
                    requests_total.labels(model=body.model, status="404").inc()
                    return _error_response(
                        status_code=404,
                        message=str(exc),
                        error_type="invalid_request_error",
                        code="model_not_found",
                    )
                except BackendOverloadedError as exc:
                    inference_errors_total.labels(error_type="overload").inc()
                    requests_total.labels(model=body.model, status="503").inc()
                    return _error_response(
                        status_code=503,
                        message=str(exc),
                        error_type="server_error",
                        code="backend_overloaded",
                        headers=RETRY_LATER_HEADERS,
                    )
                except BackendError as exc:
                    inference_errors_total.labels(error_type="backend_error").inc()
                    requests_total.labels(model=body.model, status="502").inc()
                    return _error_response(
                        status_code=502,
                        message=str(exc),
                        error_type="server_error",
                        code="backend_error",
                    )
            finally:
                active_requests.labels(model=body.model).dec()
                request_latency_seconds.labels(model=body.model).observe(
                    time.perf_counter() - start_time
                )

            completion_tokens = result.completion_tokens or 0
            duration = time.perf_counter() - start_time
            tokens_generated_total.labels(model=result.model).inc(completion_tokens)
            requests_total.labels(model=body.model, status="200").inc()
            if result.total_duration_ns is not None:
                backend_latency_seconds.labels(model=result.model).observe(
                    result.total_duration_ns / 1e9
                )
            if completion_tokens > 0 and duration > 0:
                tokens_per_second.labels(model=result.model).observe(completion_tokens / duration)

            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=result.model,
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(role="assistant", content=result.content),
                        finish_reason=result.finish_reason,
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=result.prompt_tokens or 0,
                    completion_tokens=completion_tokens,
                    total_tokens=(result.prompt_tokens or 0) + completion_tokens,
                ),
            )
    except QueueFullError:
        inference_errors_total.labels(error_type="queue_full").inc()
        requests_total.labels(model=body.model, status="503").inc()
        logger.warning(
            "queue_rejected",
            model=body.model,
            queue_depth=queue_manager.active_requests,
            queue_max=queue_manager.max_size,
            stream=False,
        )
        return _error_response(
            status_code=503,
            message="Server is at capacity. Please retry later.",
            error_type="server_error",
            code="queue_full",
            headers=RETRY_LATER_HEADERS,
        )
