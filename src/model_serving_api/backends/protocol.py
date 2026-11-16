"""Inference backend protocol and shared data types."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Message:
    """A single message in a chat conversation."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Request to generate a completion."""

    model: str
    messages: list[Message]
    request_id: str = ""
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    """Complete (non-streaming) response from the backend."""

    model: str
    content: str
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ns: int | None = None


@dataclass(frozen=True, slots=True)
class TokenDelta:
    """A single token chunk from a streaming response."""

    content: str
    finish_reason: str | None = None
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ns: int | None = None


@dataclass(frozen=True, slots=True)
class BackendHealth:
    """Health status reported by a backend."""

    healthy: bool
    model_loaded: bool
    models_available: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BackendError(Exception):
    """Base exception for backend failures."""


class ModelNotFoundError(BackendError):
    """The requested model is not available on the backend."""


class BackendOverloadedError(BackendError):
    """The backend is overloaded or unresponsive."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol that all inference backends must satisfy."""

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate a complete response (non-streaming)."""
        ...

    async def stream(self, request: InferenceRequest) -> AsyncIterator[TokenDelta]:
        """Stream token deltas as they are generated."""
        ...

    async def health(self) -> BackendHealth:
        """Check backend health and model availability."""
        ...
