"""SSE streaming for chat completions."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import structlog

from model_serving_api.backends.protocol import (
    BackendError,
    BackendOverloadedError,
    InferenceBackend,
    InferenceRequest,
    ModelNotFoundError,
)

logger = structlog.get_logger(__name__)


@dataclass
class StreamStats:
    """Mutable container populated during streaming for metrics collection."""

    completion_tokens: int = 0
    total_duration_ns: int | None = None
    # Set when the stream ends with a backend error. HTTP 200 has already been sent by
    # then, so these record the status/error the non-streaming path would have returned.
    error_status: str | None = None
    error_type: str | None = None


def classify_backend_error(exc: BackendError) -> tuple[str, str, str]:
    """Map a backend exception to (http_status, error_code, metrics error_type)."""
    if isinstance(exc, ModelNotFoundError):
        return "404", "model_not_found", "model_not_found"
    if isinstance(exc, BackendOverloadedError):
        return "503", "backend_overloaded", "overload"
    return "502", "backend_error", "backend_error"


def build_error_event(message: str, code: str) -> str:
    """Build an OpenAI-style error payload sent as an SSE data event."""
    error_type = "invalid_request_error" if code == "model_not_found" else "server_error"
    return json.dumps(
        {"error": {"message": message, "type": error_type, "param": None, "code": code}}
    )


def _build_chunk(
    request_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> str:
    """Build a single OpenAI-format streaming chunk as JSON string."""
    return json.dumps(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


async def stream_chat_completion(
    backend: InferenceBackend,
    inference_request: InferenceRequest,
    request_id: str,
    created: int | None = None,
    stats: StreamStats | None = None,
) -> AsyncIterator[str]:
    """Transform backend token stream into OpenAI SSE chunks.

    Yields JSON strings. EventSourceResponse wraps each in ``data: ...\\n\\n``.
    """
    if created is None:
        created = int(time.time())

    model = inference_request.model
    # request_id here is the completion (chunk) id. The per-request log id comes from the
    # structlog context bound by the logging middleware, so don't overwrite it.
    log = logger.bind(completion_id=request_id, model=model)

    # First chunk: role announcement
    yield _build_chunk(request_id, created, model, {"role": "assistant", "content": ""}, None)

    log.info("stream_started")
    stream_start_time = time.perf_counter()

    try:
        async for delta in backend.stream(inference_request):
            if delta.finish_reason:
                if stats is not None:
                    stats.completion_tokens = delta.completion_tokens or 0
                    stats.total_duration_ns = delta.total_duration_ns
                # Emit any remaining content before the finish signal
                if delta.content:
                    yield _build_chunk(
                        request_id, created, delta.model or model, {"content": delta.content}, None
                    )
                # Final chunk: empty delta with finish_reason
                yield _build_chunk(
                    request_id, created, delta.model or model, {}, delta.finish_reason
                )
                break
            else:
                yield _build_chunk(
                    request_id, created, delta.model or model, {"content": delta.content}, None
                )
    except BackendError as exc:
        # Headers (HTTP 200) are already sent, so report the failure in-band instead of
        # ending with [DONE], which clients read as a successful (possibly empty) answer.
        status, code, error_type = classify_backend_error(exc)
        if stats is not None:
            stats.error_status = status
            stats.error_type = error_type
        log.error(
            "stream_error",
            error_code=code,
            error=str(exc),
            duration_ms=round((time.perf_counter() - stream_start_time) * 1000, 2),
        )
        yield build_error_event(str(exc), code)
        return

    stream_duration = time.perf_counter() - stream_start_time
    token_count = stats.completion_tokens if stats else 0
    log.info(
        "stream_completed",
        token_count=token_count,
        duration_ms=round(stream_duration * 1000, 2),
    )

    # Terminator
    yield "[DONE]"
