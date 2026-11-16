"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from model_serving_api import __version__
from model_serving_api.api.completions import router as completions_router
from model_serving_api.api.health import router as health_router
from model_serving_api.backends.ollama import OllamaBackend
from model_serving_api.backends.protocol import InferenceRequest, Message
from model_serving_api.config import Settings
from model_serving_api.metrics.prometheus import inference_errors_total, request_queue_depth
from model_serving_api.middleware.logging import RequestLoggingMiddleware
from model_serving_api.queue.batcher import BatchDispatcher
from model_serving_api.queue.manager import QueueManager


async def _warm_up_model(settings: Settings, log) -> None:
    """Send one tiny request so the model is loaded before real traffic arrives.

    Uses a dedicated client with ``warmup_timeout_seconds``: loading a model from disk
    can exceed the per-request timeout, and a timed-out warm-up leaves the first real
    requests to pay the load cost and hit 504s.
    """
    log.info(
        "model_warmup_started",
        model=settings.model_name,
        timeout_seconds=settings.warmup_timeout_seconds,
    )
    warmup_start = time.perf_counter()
    backend = OllamaBackend(
        base_url=settings.ollama_url,
        timeout=float(settings.warmup_timeout_seconds),
    )
    try:
        warmup_request = InferenceRequest(
            model=settings.model_name,
            messages=[Message(role="user", content="hi")],
            request_id="warmup",
            max_tokens=1,
        )
        await backend.generate(warmup_request)
        log.info(
            "model_warmup_complete",
            duration_seconds=round(time.perf_counter() - warmup_start, 2),
        )
    except Exception as exc:
        log.warning(
            "model_warmup_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            duration_seconds=round(time.perf_counter() - warmup_start, 2),
        )
    finally:
        await backend.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage application lifecycle."""
    log = structlog.get_logger(__name__)
    settings = Settings()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    backend = OllamaBackend(
        base_url=settings.ollama_url,
        timeout=float(settings.request_timeout_seconds),
    )
    queue_manager = QueueManager(max_size=settings.max_queue_size)
    batch_dispatcher = BatchDispatcher(
        backend=backend,
        max_batch_size=settings.batch_max_size,
        wait_ms=settings.batch_wait_ms,
    )
    app.state.settings = settings
    app.state.backend = backend
    app.state.queue_manager = queue_manager
    app.state.batch_dispatcher = batch_dispatcher
    app.state.shutting_down = False
    request_queue_depth.set_function(lambda: queue_manager.active_requests)

    if settings.warmup_enabled:
        await _warm_up_model(settings, log)
    else:
        log.info("model_warmup_skipped", model=settings.model_name)

    yield

    # --- Graceful shutdown ---
    log.info("shutdown_started", active_requests=queue_manager.active_requests)
    app.state.shutting_down = True

    # Wait for in-flight requests to drain
    poll_interval = 0.25
    elapsed = 0.0
    while queue_manager.active_requests > 0 and elapsed < settings.shutdown_grace_seconds:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    if queue_manager.active_requests > 0:
        log.warning("shutdown_forced", remaining=queue_manager.active_requests)
    else:
        log.info("shutdown_drained", elapsed_seconds=round(elapsed, 1))

    await backend.close()


app = FastAPI(
    title="Model Serving API",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from starlette.responses import Response

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Paths that don't require authentication
_PUBLIC_PATHS = {"/", "/health", "/ready", "/metrics"}


logger = structlog.get_logger(__name__)


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """Reject requests to protected endpoints without a valid API key."""
    settings = request.app.state.settings
    if settings.api_key and request.url.path not in _PUBLIC_PATHS:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {settings.api_key}":
            logger.warning(
                "auth_failure",
                path=request.url.path,
                client_ip=request.client.host if request.client else "unknown",
            )
            inference_errors_total.labels(error_type="auth_failure").inc()
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid or missing API key",
                        "type": "auth_error",
                        "code": "invalid_api_key",
                    }
                },
            )
    return await call_next(request)


app.add_middleware(RequestLoggingMiddleware)
app.include_router(completions_router)
app.include_router(health_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return structured JSON for unhandled exceptions instead of HTML 500."""
    logger.exception("unhandled_exception", path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": "internal_error",
            }
        },
    )


@app.get("/")
async def root():
    return {"service": "model-serving-api", "version": __version__}
