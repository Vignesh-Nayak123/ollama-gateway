"""Custom Prometheus metrics for inference serving."""

from prometheus_client import Counter, Gauge, Histogram

MODEL_LABEL = ["model"]

# --- Gauges (current state) ---

request_queue_depth = Gauge(
    "request_queue_depth",
    "Current number of requests in the serving queue",
)

active_requests = Gauge(
    "active_requests",
    "Number of requests currently being processed",
    MODEL_LABEL,
)

model_loaded = Gauge(
    "model_loaded",
    "Whether a model is currently loaded (1=yes, 0=no)",
)

# --- Histograms (distributions) ---

request_latency_seconds = Histogram(
    "request_latency_seconds",
    "Total request latency in seconds",
    MODEL_LABEL,
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

time_to_first_token_seconds = Histogram(
    "time_to_first_token_seconds",
    "Time to first content token in seconds (streaming only)",
    MODEL_LABEL,
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

tokens_per_second = Histogram(
    "tokens_per_second",
    "Token generation speed (tokens/sec)",
    MODEL_LABEL,
    buckets=(1, 5, 10, 20, 50, 100, 200, 500),
)

# --- Counters ---

tokens_generated_total = Counter(
    "tokens_generated_total",
    "Total number of tokens generated",
    MODEL_LABEL,
)

inference_errors_total = Counter(
    "inference_errors_total",
    "Total inference errors by type",
    ["error_type"],
)

requests_total = Counter(
    "requests_total",
    "Total requests by model and HTTP status",
    ["model", "status"],
)

queue_wait_seconds = Histogram(
    "queue_wait_seconds",
    "Time from request arrival to queue slot acquisition (seconds)",
    MODEL_LABEL,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

backend_latency_seconds = Histogram(
    "backend_latency_seconds",
    "Backend-reported processing time in seconds",
    MODEL_LABEL,
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)
