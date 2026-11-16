# CLAUDE.md — Model Serving API

> A production-grade model serving layer wrapping Ollama with request batching, SSE streaming, backpressure, graceful degradation, and Prometheus metrics. OpenAI-compatible API.

## Stack

- **Language:** Python 3.12+
- **Framework:** FastAPI + asyncio
- **Model Backend:** Ollama (v1)
- **Streaming:** SSE via `sse-starlette`
- **Metrics:** `prometheus-client` (custom inference metrics)
- **Logging:** `structlog` (JSON output)
- **Deployment:** Docker Compose (API + Ollama + Caddy + Grafana Alloy)

## Project Structure

```
src/model_serving_api/
├── main.py              # FastAPI app, startup/shutdown
├── config.py            # Environment variable configuration
├── api/
│   ├── completions.py   # /v1/chat/completions endpoint
│   └── health.py        # /health, /ready, /metrics
├── backends/
│   ├── protocol.py      # InferenceBackend Protocol
│   └── ollama.py        # OllamaBackend implementation
├── queue/
│   ├── manager.py       # Bounded request queue + backpressure
│   └── batcher.py       # Naive batch dispatcher
├── streaming/
│   └── sse.py           # SSE response formatting
├── metrics/
│   └── prometheus.py    # Custom inference metrics
└── middleware/
    └── logging.py       # Request ID + structured logging
```

## Conventions

- **Tests:** pytest. Test the serving logic (queue, batching, streaming), not the Ollama client.
- **Code style:** ruff for linting and formatting.
- **Commits:** Conventional commits. One logical change per commit.
- **Config:** All settings via environment variables. No config files.
- **Error handling:** Return proper HTTP status codes with structured error responses. Never crash the server.
- **Logging:** structlog JSON. Every log line includes `request_id`, `model`, `event`, `duration_ms`.

## Key Design Decisions

- **Backend as Protocol:** `InferenceBackend` with `generate()`, `stream()`, `health()`. Enables swapping Ollama for other backends without touching serving logic.
- **Bounded queue:** Counter-based with 503 + Retry-After. This is the core production concept.
- **OpenAI-compatible API:** `/v1/chat/completions` with `stream: true/false`. Industry standard format.
- **Custom Prometheus metrics:** TTFT, tokens/sec, queue depth — not generic HTTP metrics.
