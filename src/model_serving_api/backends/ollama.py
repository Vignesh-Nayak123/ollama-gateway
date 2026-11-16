"""Ollama backend implementation."""

from __future__ import annotations

import functools
import json
import ssl
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from model_serving_api.backends.protocol import (
    BackendError,
    BackendHealth,
    BackendOverloadedError,
    InferenceRequest,
    InferenceResponse,
    ModelNotFoundError,
    TokenDelta,
)

logger = structlog.get_logger(__name__)


@functools.cache
def _shared_ssl_context() -> ssl.SSLContext:
    """Build the default (certificate-verifying) SSL context once per process.

    httpx otherwise loads the CA bundle for every new client, which costs ~1 s on some
    machines and dominated test-suite runtime.
    """
    return httpx.create_ssl_context()


class OllamaBackend:
    """InferenceBackend implementation that delegates to an Ollama server."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            verify=_shared_ssl_context(),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Send a non-streaming chat completion request to Ollama."""
        payload = self._build_payload(request, stream=False)
        log = logger.bind(request_id=request.request_id, model=request.model)
        log.debug("ollama_generate")

        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise BackendError(f"Cannot connect to Ollama at {self._base_url}") from exc
        except httpx.TimeoutException as exc:
            raise BackendError("Ollama request timed out") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"Ollama request failed: {exc}") from exc

        self._check_response_status(response, request.model)
        data = response.json()

        return InferenceResponse(
            model=data.get("model", request.model),
            content=data["message"]["content"],
            finish_reason="stop" if data.get("done") else "length",
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_duration_ns=data.get("total_duration"),
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[TokenDelta]:
        """Stream token deltas from Ollama."""
        payload = self._build_payload(request, stream=True)
        log = logger.bind(request_id=request.request_id, model=request.model)
        log.debug("ollama_stream")

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                self._check_response_status(response, request.model)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise BackendError(f"Ollama streaming error: {chunk['error']}")
                    done = chunk.get("done", False)
                    yield TokenDelta(
                        content=chunk.get("message", {}).get("content", ""),
                        finish_reason="stop" if done else None,
                        model=chunk.get("model", request.model),
                        prompt_tokens=chunk.get("prompt_eval_count") if done else None,
                        completion_tokens=chunk.get("eval_count") if done else None,
                        total_duration_ns=chunk.get("total_duration") if done else None,
                    )
                    if done:
                        break
        except BackendError:
            raise
        except httpx.ConnectError as exc:
            raise BackendError(f"Cannot connect to Ollama at {self._base_url}") from exc
        except httpx.TimeoutException as exc:
            raise BackendError("Ollama streaming request timed out") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"Ollama streaming request failed: {exc}") from exc

    async def health(self) -> BackendHealth:
        """Check Ollama health by querying /api/tags."""
        try:
            response = await self._client.get("/api/tags")
            if response.status_code != 200:
                return BackendHealth(
                    healthy=False,
                    model_loaded=False,
                    error=f"Ollama returned status {response.status_code}",
                )
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return BackendHealth(
                healthy=True,
                model_loaded=len(models) > 0,
                models_available=models,
            )
        except httpx.HTTPError as exc:
            return BackendHealth(healthy=False, model_loaded=False, error=str(exc))

    def _build_payload(self, request: InferenceRequest, *, stream: bool) -> dict[str, Any]:
        """Build the Ollama /api/chat request payload."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": stream,
        }
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop
        if options:
            payload["options"] = options
        return payload

    def _check_response_status(self, response: httpx.Response, model: str) -> None:
        """Raise appropriate BackendError for non-2xx responses."""
        if response.status_code == 200:
            return
        if response.status_code == 404:
            raise ModelNotFoundError(f"Model '{model}' not found on Ollama")
        if response.status_code in (503, 429):
            raise BackendOverloadedError(f"Ollama is overloaded (HTTP {response.status_code})")
        raise BackendError(f"Ollama returned HTTP {response.status_code}")
