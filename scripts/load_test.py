#!/usr/bin/env python3
"""Load test script for model-serving-api.

Demonstrates concurrent requests, queue saturation, and backpressure behavior.

Usage:
    python scripts/load_test.py                          # defaults: 10 concurrent, non-streaming
    python scripts/load_test.py --concurrent 50          # exceed queue capacity (default max=50)
    python scripts/load_test.py --stream                 # streaming requests
    python scripts/load_test.py --concurrent 20 --stream # concurrent streaming

Requires: httpx (already a project dependency)
    pip install httpx
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field

import httpx

BASE_URL = "http://localhost:8000"
API_KEY: str | None = None
MODEL = "llama3.2"
PROMPT = "Count from 1 to 5."


@dataclass
class RequestResult:
    status: int
    duration: float
    tokens: int = 0
    error: str | None = None


@dataclass
class LoadTestSummary:
    results: list[RequestResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def successful(self) -> int:
        return sum(1 for r in self.results if 200 <= r.status < 300)

    @property
    def rejected(self) -> int:
        return sum(1 for r in self.results if r.status == 503)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status >= 400 and r.status != 503)

    @property
    def avg_latency(self) -> float:
        successful = [r.duration for r in self.results if 200 <= r.status < 300]
        return sum(successful) / len(successful) if successful else 0.0

    @property
    def p50_latency(self) -> float:
        return self._percentile(50)

    @property
    def p99_latency(self) -> float:
        return self._percentile(99)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.total_duration if self.total_duration > 0 else 0.0

    def _percentile(self, p: int) -> float:
        successful = sorted(r.duration for r in self.results if 200 <= r.status < 300)
        if not successful:
            return 0.0
        idx = int(len(successful) * p / 100)
        return successful[min(idx, len(successful) - 1)]


async def send_completion(
    client: httpx.AsyncClient,
    stream: bool = False,
) -> RequestResult:
    """Send a single chat completion request."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 50,
        "stream": stream,
    }

    start = time.perf_counter()
    try:
        if stream:
            token_count = 0
            async with client.stream(
                "POST", f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    return RequestResult(
                        status=response.status_code,
                        duration=time.perf_counter() - start,
                        error=response.text[:200],
                    )
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        token_count += 1
            return RequestResult(
                status=200,
                duration=time.perf_counter() - start,
                tokens=max(token_count - 2, 0),  # subtract role + finish chunks
            )
        else:
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120
            )
            tokens = 0
            if response.status_code == 200:
                data = response.json()
                tokens = data.get("usage", {}).get("completion_tokens", 0)
            return RequestResult(
                status=response.status_code,
                duration=time.perf_counter() - start,
                tokens=tokens,
                error=response.text[:200] if response.status_code >= 400 else None,
            )
    except httpx.ConnectError:
        return RequestResult(
            status=0,
            duration=time.perf_counter() - start,
            error="Connection refused — is the server running?",
        )
    except httpx.ReadTimeout:
        return RequestResult(
            status=0,
            duration=time.perf_counter() - start,
            error="Request timed out",
        )


async def run_load_test(concurrent: int, stream: bool) -> LoadTestSummary:
    """Run concurrent requests and collect results."""
    mode = "streaming" if stream else "non-streaming"
    print(f"\n{'=' * 60}")
    print(f"  Load Test: {concurrent} concurrent {mode} requests")
    print(f"  Target: {BASE_URL}")
    print(f"  Model: {MODEL}")
    print(f"{'=' * 60}\n")

    # Check server is reachable
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    async with httpx.AsyncClient(headers=headers) as client:
        try:
            health = await client.get(f"{BASE_URL}/health", timeout=5)
            if health.status_code != 200:
                print("Server health check failed!")
                sys.exit(1)
        except httpx.ConnectError:
            print(f"Cannot connect to {BASE_URL} — is the server running?")
            sys.exit(1)

    # Run concurrent requests
    summary = LoadTestSummary()
    start = time.perf_counter()

    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [send_completion(client, stream=stream) for _ in range(concurrent)]
        results = await asyncio.gather(*tasks)

    summary.total_duration = time.perf_counter() - start
    summary.results = list(results)

    return summary


def print_summary(summary: LoadTestSummary) -> None:
    """Print a formatted summary of the load test results."""
    print(f"\n{'─' * 60}")
    print("  Results")
    print(f"{'─' * 60}")
    print(f"  Total requests:     {summary.total}")
    print(f"  Successful (2xx):   {summary.successful}")
    print(f"  Rejected (503):     {summary.rejected}")
    print(f"  Errors:             {summary.errors}")
    print(f"  Total duration:     {summary.total_duration:.2f}s")
    print()
    print(f"  Avg latency:        {summary.avg_latency:.2f}s")
    print(f"  P50 latency:        {summary.p50_latency:.2f}s")
    print(f"  P99 latency:        {summary.p99_latency:.2f}s")

    if summary.total_tokens > 0:
        print()
        print(f"  Total tokens:       {summary.total_tokens}")
        print(f"  Tokens/sec:         {summary.tokens_per_second:.1f}")

    if summary.rejected > 0:
        print()
        print(f"  Backpressure: {summary.rejected}/{summary.total} requests rejected (queue full)")

    # Show first error if any
    errors = [r for r in summary.results if r.error]
    if errors:
        print()
        print(f"  First error: {errors[0].error}")

    print(f"{'─' * 60}\n")


def main():
    global BASE_URL, API_KEY

    parser = argparse.ArgumentParser(description="Load test model-serving-api")
    parser.add_argument(
        "--concurrent", type=int, default=10, help="Number of concurrent requests (default: 10)"
    )
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--url", default=BASE_URL, help=f"Server URL (default: {BASE_URL})")
    parser.add_argument("--api-key", default=None, help="Bearer token for API authentication")
    args = parser.parse_args()

    BASE_URL = args.url
    API_KEY = args.api_key

    summary = asyncio.run(run_load_test(args.concurrent, args.stream))
    print_summary(summary)


if __name__ == "__main__":
    main()
