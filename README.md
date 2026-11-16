# Model Serving API

An OpenAI-compatible gateway in front of [Ollama](https://ollama.com), with request batching, SSE streaming, bounded-queue backpressure (503 + `Retry-After`), Prometheus metrics, and structured JSON logs.

## Architecture

```
Client (curl / OpenAI SDK)
    │
    ▼
┌─────────────────────────────────────┐
│  Caddy (optional, auto-TLS)        │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  FastAPI Server                     │
│  ├─ /v1/chat/completions           │  ← OpenAI-compatible endpoint
│  ├─ /health, /ready                │  ← Health probes
│  ├─ /metrics                       │  ← Prometheus scrape
│  │                                  │
│  Bounded Request Queue              │
│  ┌────────────────────────┐        │
│  │ max_size=50, 503 when  │        │  ← Backpressure via Retry-After
│  │ full                    │        │
│  └─────────┬──────────────┘        │
│            │                        │
│  Batch Dispatcher                   │
│  ┌─────────▼──────────────┐        │
│  │ Collect for 100ms or   │        │  ← Naive batching (time/size)
│  │ 8 requests              │        │
│  └─────────┬──────────────┘        │
│            │                        │
│  InferenceBackend Protocol          │
│  ┌─────────▼──────────────┐        │
│  │ OllamaBackend          │        │  ← HTTP client to Ollama
│  └────────────────────────┘        │
└─────────────────────────────────────┘
    │
    ▼
┌──────────────────┐     ┌────────────────────────┐
│  Ollama (host)   │     │  Prometheus / Grafana  │
│  llama3.2:3b     │     │  scrape /metrics       │
└──────────────────┘     └────────────────────────┘
```

## Features

- **OpenAI-compatible API** — `/v1/chat/completions` with streaming and non-streaming modes
- **SSE streaming** — Token-by-token delivery in OpenAI chunk format
- **Request batching** — Naive batching with configurable time/size thresholds, concurrent dispatch
- **Backpressure** — Bounded request queue; returns 503 + `Retry-After` when full
- **Prometheus metrics** — 11 custom inference metrics: TTFT, tokens/sec, queue depth, error rates, latency histograms
- **Observability** — Pre-built Grafana dashboard (`deploy/grafana/dashboard.json`) and an optional Grafana Alloy config for shipping metrics and logs to Grafana Cloud
- **Graceful shutdown** — SIGTERM → reject new requests → drain in-flight → close backend
- **API key auth** — Optional Bearer token authentication
- **Model warm-up** — Dummy request on startup to pre-load the model, with its own timeout for slow cold loads
- **Structured logging** — JSON logs via structlog with per-request `request_id`, model, and stream context
- **Health checks** — `/health` (liveness) and `/ready` (readiness: backend reachable, configured model available, queue status)
- **Error handling** — Proper HTTP status codes (401, 404, 502, 503, 504) with structured error responses
- **Backend abstraction** — `InferenceBackend` Protocol enables swapping Ollama for other backends

## Quickstart

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Ollama](https://ollama.com/download) running on the host, with the model pulled:
  `ollama pull llama3.2:3b` (about 2 GB)

### Run

```bash
# Start the API (talks to the host's Ollama at host.docker.internal:11434)
docker compose up -d

# Wait for readiness (503 until Ollama is reachable and the model is pulled)
curl http://localhost:8000/ready
```

To run Ollama in a container instead, use `docker compose --profile bundled-ollama up -d`
with `OLLAMA_URL=http://ollama:11434` (CPU-only unless you add a GPU reservation). Add
`--profile proxy` for Caddy with auto-TLS.

### Try it

**Non-streaming:**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }' | python -m json.tool
```

**Streaming:**

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Benchmarks

Measured on a laptop: **NVIDIA RTX 3050 6 GB Laptop GPU**, Ollama 0.40.16, `llama3.2:3b`
fully on the GPU, gateway running natively on Windows. Every request is non-streaming with the
same prompt, `max_tokens: 64` and `temperature: 0`, so each completion is 64 tokens. Each
row is a 60 s [Locust](https://locust.io) run where every simulated user sends requests
back-to-back with no think time. Single runs, so expect a few percent of noise. Raw summaries
are in [`benchmarks/rtx3050/`](benchmarks/rtx3050/); reproduce with
[`loadtest/sweep.py`](loadtest/sweep.py).

### Ollama parallelism is what scales throughput

| Users | p50, 1 slot | p95, 1 slot | tok/s, 1 slot | p50, 4 slots | p95, 4 slots | tok/s, 4 slots |
|------:|------:|------:|------:|------:|------:|------:|
| 1  | 1.2 s | 1.3 s  | 49 | 1.2 s | 1.3 s | 55  |
| 4  | 6.0 s | 9.1 s  | 41 | 1.8 s | 2.5 s | 128 |
| 8  | 9.2 s | 13 s   | 54 | 3.5 s | 3.8 s | 150 |
| 16 | 17 s  | 17 s   | 60 | 6.5 s | 7.1 s | 157 |

"Slots" is `OLLAMA_NUM_PARALLEL`. With one slot (the default on this machine) Ollama runs
one request at a time: throughput stays flat at about 50-60 tok/s and latency grows in
proportion to the number of users. With four slots llama.cpp batches the decoding of
concurrent requests on the GPU: **at 16 users, 2.6x the throughput and 2.4x lower p95**,
for 1.4 GB more VRAM (4.0 GB vs 2.6 GB).

### Gateway-level batching adds nothing on top

| Users | Gateway batching | p50 | p95 | tok/s |
|------:|------------------|----:|----:|------:|
| 1 | on (8 requests / 100 ms) | 1.2 s | 1.3 s | 55 |
| 1 | off (`BATCH_MAX_SIZE=1`) | 1.1 s | 1.5 s | 58 |
| 8 | on (8 requests / 100 ms) | 3.5 s | 3.8 s | 150 |
| 8 | off (`BATCH_MAX_SIZE=1`) | 3.1 s | 3.6 s | 165 |

Both with four Ollama slots. The gateway's batcher holds requests for up to 100 ms and then
sends them to Ollama concurrently, but Ollama already batches whatever is in flight, so the
wait buys nothing. Real throughput gains need batching inside the inference engine.

### Backpressure: queue of 4, 16 users

| Setup | OK | 503 | Connection errors | p95 (OK) | tok/s |
|-------|---:|----:|------------------:|---------:|------:|
| Clients honour `Retry-After`, before fix | 114 | 144 | **38** | 3.2 s | 114 |
| Same, uvicorn keep-alive raised to 30 s | 116 | 144 | 0 | 2.7 s | 122 |
| Clients ignore `Retry-After`, before fix | 88 | 2,542 | 0 | 4.7 s | 90 |
| Clients honour `Retry-After`, **with `Connection: close` fix** | 140 | 144 | **0** | 1.9 s | 147 |

With 16 users and room for 4, the gateway rejects the overflow with 503 in milliseconds
instead of letting latency grow: p95 for accepted requests is 1.9 s, against 7.1 s when all
16 are queued (previous table).

The first row is how the keep-alive bug showed up: clients that waited the 5 s
`Retry-After` reused a connection that uvicorn closes after 5 s idle, so about 13% of
requests failed before reaching the app. Raising the keep-alive timeout made the errors
disappear, which confirmed the cause; the fix sends `Connection: close` on retry-later
503s so it works however the server is launched. The third row shows why `Retry-After`
matters: clients that retry immediately produce 17x more rejections and lower throughput.

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `llama3.2` (`llama3.2:3b` in Compose) | Model that `/ready` checks for and warm-up loads |
| `OLLAMA_URL` | `http://ollama:11434` (`http://host.docker.internal:11434` in Compose) | Ollama backend URL |
| `MAX_QUEUE_SIZE` | `50` | Max concurrent requests before 503 |
| `BATCH_WAIT_MS` | `100` | Max time to wait for batch to fill |
| `BATCH_MAX_SIZE` | `8` | Max requests per batch |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `SHUTDOWN_GRACE_SECONDS` | `30` | Drain timeout on shutdown |
| `WARMUP_ENABLED` | `true` | Send a warm-up request on startup to load the model |
| `WARMUP_TIMEOUT_SECONDS` | `300` | Timeout for the warm-up request (cold loads are slow) |
| `API_KEY` | *(none)* | Bearer token for auth (disabled if unset) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `info` | Log level |

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completion. Supports `stream: true` for SSE.

**Request body:**

```json
{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 100,
  "stop": ["\n"],
  "stream": false
}
```

**Error responses:**

| Status | Code | Cause |
|--------|------|-------|
| 401 | `invalid_api_key` | Missing or invalid API key |
| 404 | `model_not_found` | Model not available in Ollama |
| 422 | — | Invalid request body |
| 502 | `backend_error` | Ollama connection failure |
| 503 | `backend_overloaded` | Ollama overloaded |
| 503 | `queue_full` | Request queue at capacity |
| 503 | `shutting_down` | Server is draining for shutdown |
| 504 | `request_timeout` | Backend response timeout |

### `GET /health`

Liveness probe. Always returns 200.

### `GET /ready`

Readiness probe. Returns 200 if backend is healthy and queue has capacity, 503 otherwise.

### `GET /metrics`

Prometheus metrics endpoint. Custom inference metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `request_latency_seconds` | Histogram | Total request latency |
| `time_to_first_token_seconds` | Histogram | Time to first content token (streaming) |
| `tokens_per_second` | Histogram | Token generation speed |
| `tokens_generated_total` | Counter | Total tokens generated |
| `active_requests` | Gauge | Currently processing requests |
| `request_queue_depth` | Gauge | Current queue depth |
| `model_loaded` | Gauge | Whether a model is loaded (0/1) |
| `inference_errors_total` | Counter | Errors by type (backend_error, model_not_found, overload, queue_full, auth_failure) |
| `requests_total` | Counter | Requests by model and HTTP status |
| `queue_wait_seconds` | Histogram | Time from arrival to queue slot |
| `backend_latency_seconds` | Histogram | Backend-reported processing time |

## Deployment

`docker-compose.yml` services:

| Service | Profile | Purpose |
|---------|---------|---------|
| `api` | *(default)* | FastAPI gateway, talks to host Ollama |
| `ollama` | `bundled-ollama` | Containerised Ollama (CPU-only unless given a GPU) |
| `caddy` | `proxy` | Reverse proxy with auto-TLS via Let's Encrypt |

`deploy/` also contains a server deploy script, systemd unit, and Grafana Alloy config,
for running the stack on a VPS and shipping telemetry to Grafana Cloud.

## Development

```bash
# Install dependencies
uv sync --extra dev

# Or use the justfile shortcuts
just setup      # = uv sync --extra dev
just dev        # run locally with --reload
just check      # lint + format check + tests, same as CI

# Run tests (101 tests, ~9 s)
uv run pytest tests/ -v

# Lint and format check (same as CI)
uv run ruff check . && uv run ruff format --check .

# Run locally (requires Ollama running on localhost:11434)
OLLAMA_URL=http://localhost:11434 uv run uvicorn model_serving_api.main:app --reload
```

### Load testing

`loadtest/locustfile.py` drives `/v1/chat/completions` (non-streaming) with a fixed prompt and
`temperature: 0`. Each simulated user sends requests back-to-back, so `-u` is the number of
concurrent clients. A 503 with `Retry-After` counts as expected backpressure: it is reported
separately as `chat_completion [503 backpressure]`, and the user then waits `Retry-After` seconds.

```bash
# One run
uv run --extra loadtest locust -f loadtest/locustfile.py --headless \
    --host http://localhost:8000 -u 5 -r 5 -t 60s --csv results/example

# Sweep several concurrency levels; writes summary.md / summary.json to --out
uv run --extra loadtest python loadtest/sweep.py --host http://localhost:8000 \
    --users 1,4,8,16 --duration 60s --out results/my-run
```

Optional env vars: `MODEL`, `MAX_TOKENS`, `LOADTEST_PROMPT` (not `PROMPT`, which Windows already sets), `HONOR_RETRY_AFTER=0` (retry immediately),
`TOKENS_FILE=path.json` (write completion-token totals and tokens/sec at test stop).

## Project Structure

```
src/model_serving_api/
├── main.py              # FastAPI app, lifespan, warm-up, graceful shutdown
├── config.py            # Environment variable settings
├── api/
│   ├── completions.py   # /v1/chat/completions endpoint
│   └── health.py        # /health, /ready endpoints
├── backends/
│   ├── protocol.py      # InferenceBackend Protocol + data types
│   └── ollama.py        # OllamaBackend implementation
├── queue/
│   ├── manager.py       # Bounded request queue + backpressure
│   └── batcher.py       # Naive batch dispatcher
├── metrics/
│   └── prometheus.py    # Custom inference metrics
├── streaming/
│   └── sse.py           # SSE chunk formatting
└── middleware/
    └── logging.py       # Request ID + structured logging

deploy/
├── Caddyfile            # Caddy reverse proxy config
├── alloy/
│   └── config.alloy     # Grafana Alloy telemetry config
└── grafana/
    └── dashboard.json   # Pre-built Grafana dashboard (14 panels)

loadtest/
├── locustfile.py        # Locust load test (backpressure-aware)
└── sweep.py             # Runs the load test at several concurrency levels

benchmarks/rtx3050/      # Benchmark summaries behind the README tables

.github/workflows/
└── ci.yml               # Lint + format + tests on every push and PR

scripts/
├── load_test.py         # Quick load test script
└── experiments.py       # Comprehensive experiment suite (8 experiments)
```

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Backend as Protocol | `InferenceBackend` with `generate()`, `stream()`, `health()` | Enables swapping Ollama for llama-cpp-python or other backends without touching serving logic |
| Bounded queue | Counter-based with 503 + Retry-After | Backpressure prevents cascading failures; simpler than asyncio.Queue since asyncio is single-threaded |
| Naive batching | Time/size threshold dispatch via asyncio.gather | Demonstrates the pattern; continuous batching requires direct model access |
| Custom Prometheus metrics | TTFT, tokens/sec, queue depth, error counters | Inference-specific metrics are more valuable than generic HTTP metrics |
| SSE streaming | sse-starlette with OpenAI chunk format | Industry standard; every client library works with it |
| Graceful shutdown | Lifespan teardown with drain polling | Uvicorn handles SIGTERM; app drains in-flight requests before closing backend |
| Caddy for TLS | Auto-certificate via Let's Encrypt | Zero-config HTTPS with automatic renewal |
| Grafana Alloy (optional) | Single agent for metrics + logs | Replaces both Prometheus server and log shipper; Grafana Cloud free tier for storage. Config in `deploy/alloy/`, not in the default Compose stack |
