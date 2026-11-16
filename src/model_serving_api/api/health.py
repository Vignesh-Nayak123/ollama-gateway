"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from model_serving_api.metrics.prometheus import model_loaded as model_loaded_gauge

router = APIRouter(tags=["health"])


def is_model_available(model_name: str, models_available: list[str]) -> bool:
    """Return True if the configured model is on the backend.

    Ollama lists models with explicit tags, so an untagged name like ``llama3.2``
    matches ``llama3.2:latest`` (the tag Ollama resolves it to), but not ``llama3.2:3b``.
    """
    if model_name in models_available:
        return True
    return ":" not in model_name and f"{model_name}:latest" in models_available


@router.get("/health")
async def health():
    """Liveness probe — always returns 200."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    """Readiness probe — checks backend health and queue capacity."""
    backend = request.app.state.backend
    queue_manager = request.app.state.queue_manager

    backend_health = await backend.health()
    model_loaded_gauge.set(1.0 if backend_health.model_loaded else 0.0)

    queue_info = {
        "active": queue_manager.active_requests,
        "max": queue_manager.max_size,
    }

    if not backend_health.healthy:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": backend_health.error or "Backend unhealthy",
                "queue": queue_info,
            },
        )

    model_name = request.app.state.settings.model_name
    if not is_model_available(model_name, backend_health.models_available):
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": f"Model '{model_name}' is not available on the backend",
                "models_available": backend_health.models_available,
                "queue": queue_info,
            },
        )

    if queue_manager.is_full:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "Queue is full",
                "queue": queue_info,
            },
        )

    return {
        "status": "ready",
        "model_loaded": backend_health.model_loaded,
        "models_available": backend_health.models_available,
        "queue": queue_info,
    }
