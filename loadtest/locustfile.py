"""Locust load test for /v1/chat/completions (non-streaming).

Every simulated user sends the same request (fixed prompt, max_tokens, temperature 0)
back-to-back with no think time, so the number of users equals the number of
concurrent clients.

503 handling: the gateway rejects requests when MAX_QUEUE_SIZE requests are already in
flight. A 503 with a Retry-After header is *expected* backpressure, not a failure, so it
is recorded under its own name ("chat_completion [503 backpressure]") and marked as a
success. The user then waits Retry-After seconds, as a well-behaved client would
(disable with HONOR_RETRY_AFTER=0). A 503 *without* Retry-After is a failure.

Environment variables (all optional):
    MODEL              model name sent in the request      (default: llama3.2:3b)
    MAX_TOKENS         max_tokens for every request         (default: 64)
    LOADTEST_PROMPT    user prompt                          (default: see DEFAULT_PROMPT)
    HONOR_RETRY_AFTER  1 = sleep Retry-After after a 503    (default: 1)
    TOKENS_FILE        if set, write completion-token totals here as JSON at test stop

Example:
    uv run --extra loadtest locust -f loadtest/locustfile.py --headless \
        --host http://localhost:8000 -u 5 -r 5 -t 60s --csv results/example
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path

from locust import HttpUser, constant, events, task

DEFAULT_PROMPT = "Explain in three sentences why the sky is blue."

MODEL = os.environ.get("MODEL", "llama3.2:3b")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
PROMPT = os.environ.get("LOADTEST_PROMPT", DEFAULT_PROMPT)
HONOR_RETRY_AFTER = os.environ.get("HONOR_RETRY_AFTER", "1") == "1"
TOKENS_FILE = os.environ.get("TOKENS_FILE")

REQUEST_NAME = "chat_completion"
BACKPRESSURE_NAME = "chat_completion [503 backpressure]"

PAYLOAD = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": MAX_TOKENS,
    "temperature": 0,
    "stream": False,
}

# Totals across all users (Locust users are greenlets in one process, so no locking).
_totals = {
    "ok_requests": 0,
    "completion_tokens": 0,
    "prompt_tokens": 0,
    "backpressure_503": 0,
    "retry_after_values": {},
    "started_at": None,
    "stopped_at": None,
}


class ChatCompletionUser(HttpUser):
    wait_time = constant(0)

    @task
    def chat_completion(self) -> None:
        with self.client.post(
            "/v1/chat/completions",
            json=PAYLOAD,
            name=REQUEST_NAME,
            catch_response=True,
            timeout=120,
        ) as response:
            if response.status_code == 200:
                try:
                    usage = response.json()["usage"]
                    completion_tokens = int(usage["completion_tokens"])
                    prompt_tokens = int(usage["prompt_tokens"])
                except (ValueError, KeyError, TypeError):
                    response.failure("200 without a valid usage block")
                    return
                _totals["ok_requests"] += 1
                _totals["completion_tokens"] += completion_tokens
                _totals["prompt_tokens"] += prompt_tokens
                response.success()
                return

            if response.status_code == 503:
                retry_after = response.headers.get("Retry-After")
                # Report 503s under their own name so they are counted separately.
                response.request_meta["name"] = BACKPRESSURE_NAME
                if retry_after is None:
                    response.failure("503 without Retry-After header")
                    return
                _totals["backpressure_503"] += 1
                values = _totals["retry_after_values"]
                values[retry_after] = values.get(retry_after, 0) + 1
                response.success()
                if HONOR_RETRY_AFTER:
                    with contextlib.suppress(ValueError):
                        time.sleep(float(retry_after))  # gevent-patched: yields
                return

            if response.status_code == 0:
                # No HTTP response at all: report the underlying connection error.
                response.failure(f"connection error: {response.error!r}")
                return

            response.failure(f"HTTP {response.status_code}: {response.text[:200]}")


@events.test_start.add_listener
def _on_test_start(**_kwargs) -> None:
    _totals["started_at"] = time.time()


@events.test_stop.add_listener
def _on_test_stop(**_kwargs) -> None:
    _totals["stopped_at"] = time.time()
    if not TOKENS_FILE:
        return
    elapsed = _totals["stopped_at"] - (_totals["started_at"] or _totals["stopped_at"])
    summary = {
        **_totals,
        "elapsed_seconds": round(elapsed, 2),
        "completion_tokens_per_second": (
            round(_totals["completion_tokens"] / elapsed, 2) if elapsed > 0 else None
        ),
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "prompt": PROMPT,
        "honor_retry_after": HONOR_RETRY_AFTER,
    }
    path = Path(TOKENS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
