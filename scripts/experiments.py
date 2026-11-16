#!/usr/bin/env python3
"""Experiment suite for model-serving-api behavioral analysis.

Runs structured experiments against the deployed server and saves raw data
as JSON for later analysis and write-up.

Usage:
    uv run python scripts/experiments.py --list
    uv run python scripts/experiments.py --experiment concurrency_sweep --api-key "..."
    uv run python scripts/experiments.py --all --url http://localhost:8000 --api-key "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SERVER = "http://localhost:8000"
DEFAULT_MODEL = "llama3.2:3b"
RESULTS_DIR = Path("experiments/results")

# Prompts
PROMPT_SHORT = "What is 2+2?"
PROMPT_MEDIUM = "Count from 1 to 5."
PROMPT_LONG = (
    "Explain the difference between TCP and UDP protocols. "
    "Cover reliability, ordering, flow control, and use cases."
)


def messages_short() -> list[dict]:
    return [{"role": "user", "content": PROMPT_SHORT}]


def messages_medium() -> list[dict]:
    return [{"role": "user", "content": PROMPT_MEDIUM}]


def messages_long() -> list[dict]:
    return [{"role": "user", "content": PROMPT_LONG}]


def messages_5_turn() -> list[dict]:
    return [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."},
        {"role": "user", "content": "What are its main features?"},
        {
            "role": "assistant",
            "content": "Dynamic typing, readability, and a large standard library.",
        },
        {"role": "user", "content": "Give me an example of a list comprehension."},
    ]


def messages_10_turn() -> list[dict]:
    return [
        {"role": "user", "content": "What is machine learning?"},
        {
            "role": "assistant",
            "content": "Machine learning is a subset of AI that learns from data.",
        },
        {"role": "user", "content": "What types of ML are there?"},
        {"role": "assistant", "content": "Supervised, unsupervised, and reinforcement learning."},
        {"role": "user", "content": "Explain supervised learning."},
        {
            "role": "assistant",
            "content": "It uses labeled data to train models that predict outcomes.",
        },
        {"role": "user", "content": "What is a neural network?"},
        {
            "role": "assistant",
            "content": "A computing system inspired by biological neural networks.",
        },
        {"role": "user", "content": "What is deep learning?"},
        {"role": "assistant", "content": "Deep learning uses many-layered neural networks."},
        {"role": "user", "content": "What are transformers?"},
    ]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RequestRecord:
    request_index: int
    status: int
    latency_s: float
    tokens: int
    ttft_s: float | None = None
    chunk_timestamps: list[float] | None = None
    error: str | None = None
    prompt_desc: str = ""
    stream: bool = False
    started_at: float = 0.0


@dataclass
class ExperimentResult:
    experiment: str
    timestamp: str
    server: str
    parameters: dict
    results: list | dict
    duration_s: float
    metadata: dict


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def _send_streaming(
    client: httpx.AsyncClient,
    server: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    start: float,
    started_at: float,
    request_index: int,
    prompt_desc: str,
) -> RequestRecord:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    chunk_timestamps: list[float] = []
    ttft_s: float | None = None
    token_count = 0

    try:
        async with client.stream(
            "POST", f"{server}/v1/chat/completions", json=payload, timeout=120.0
        ) as response:
            if response.status_code != 200:
                await response.aread()
                return RequestRecord(
                    request_index=request_index,
                    status=response.status_code,
                    latency_s=time.perf_counter() - start,
                    tokens=0,
                    error=response.text[:500],
                    prompt_desc=prompt_desc,
                    stream=True,
                    started_at=started_at,
                )
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                ts = time.perf_counter() - start
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [{}])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    chunk_timestamps.append(ts)
                    token_count += 1
                    if ttft_s is None:
                        ttft_s = ts
    except httpx.RemoteProtocolError:
        return RequestRecord(
            request_index=request_index,
            status=0,
            latency_s=time.perf_counter() - start,
            tokens=token_count,
            ttft_s=ttft_s,
            chunk_timestamps=chunk_timestamps if chunk_timestamps else None,
            error="Connection closed by server",
            prompt_desc=prompt_desc,
            stream=True,
            started_at=started_at,
        )

    return RequestRecord(
        request_index=request_index,
        status=200,
        latency_s=time.perf_counter() - start,
        tokens=token_count,
        ttft_s=ttft_s,
        chunk_timestamps=chunk_timestamps,
        prompt_desc=prompt_desc,
        stream=True,
        started_at=started_at,
    )


async def _send_non_streaming(
    client: httpx.AsyncClient,
    server: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    start: float,
    started_at: float,
    request_index: int,
    prompt_desc: str,
) -> RequestRecord:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = await client.post(f"{server}/v1/chat/completions", json=payload, timeout=120.0)
    latency = time.perf_counter() - start
    tokens = 0
    error = None
    if response.status_code == 200:
        data = response.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
    else:
        error = response.text[:500]
    return RequestRecord(
        request_index=request_index,
        status=response.status_code,
        latency_s=latency,
        tokens=tokens,
        error=error,
        prompt_desc=prompt_desc,
        stream=False,
        started_at=started_at,
    )


async def send_request(
    client: httpx.AsyncClient,
    server: str,
    model: str,
    messages: list[dict],
    *,
    stream: bool = False,
    max_tokens: int = 50,
    request_index: int = 0,
    prompt_desc: str = "",
) -> RequestRecord:
    start = time.perf_counter()
    started_at = time.time()
    try:
        if stream:
            return await _send_streaming(
                client,
                server,
                model,
                messages,
                max_tokens,
                start,
                started_at,
                request_index,
                prompt_desc,
            )
        else:
            return await _send_non_streaming(
                client,
                server,
                model,
                messages,
                max_tokens,
                start,
                started_at,
                request_index,
                prompt_desc,
            )
    except httpx.ConnectError:
        return RequestRecord(
            request_index=request_index,
            status=0,
            latency_s=time.perf_counter() - start,
            tokens=0,
            error="Connection refused",
            prompt_desc=prompt_desc,
            stream=stream,
            started_at=started_at,
        )
    except httpx.ReadTimeout:
        return RequestRecord(
            request_index=request_index,
            status=0,
            latency_s=time.perf_counter() - start,
            tokens=0,
            error="Read timeout",
            prompt_desc=prompt_desc,
            stream=stream,
            started_at=started_at,
        )
    except httpx.RemoteProtocolError:
        return RequestRecord(
            request_index=request_index,
            status=0,
            latency_s=time.perf_counter() - start,
            tokens=0,
            error="Connection closed by server",
            prompt_desc=prompt_desc,
            stream=stream,
            started_at=started_at,
        )


async def run_sequential(
    client: httpx.AsyncClient,
    server: str,
    model: str,
    messages: list[dict],
    *,
    n: int,
    stream: bool = False,
    max_tokens: int = 50,
    prompt_desc: str = "",
) -> list[RequestRecord]:
    results = []
    for i in range(n):
        r = await send_request(
            client,
            server,
            model,
            messages,
            stream=stream,
            max_tokens=max_tokens,
            request_index=i,
            prompt_desc=prompt_desc,
        )
        results.append(r)
    return results


async def run_concurrent(
    client: httpx.AsyncClient,
    server: str,
    model: str,
    messages: list[dict],
    *,
    n: int,
    concurrency: int,
    stream: bool = False,
    max_tokens: int = 50,
    prompt_desc: str = "",
) -> list[RequestRecord]:
    sem = asyncio.Semaphore(concurrency)

    async def _limited(idx: int) -> RequestRecord:
        async with sem:
            return await send_request(
                client,
                server,
                model,
                messages,
                stream=stream,
                max_tokens=max_tokens,
                request_index=idx,
                prompt_desc=prompt_desc,
            )

    tasks = [_limited(i) for i in range(n)]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def build_metadata(server: str, model: str) -> dict:
    return {
        "server": server,
        "model": model,
        "client_platform": platform.platform(),
        "client_machine": platform.machine(),
        "queue_max_size": 50,
        "batch_max_size": 8,
        "batch_wait_ms": 100,
        "request_timeout_s": 60,
    }


def save_result(result: ExperimentResult) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result.experiment}.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Experiment 1: Concurrency sweep
# ---------------------------------------------------------------------------


async def concurrency_sweep(
    client: httpx.AsyncClient,
    server: str,
    model: str,
) -> ExperimentResult:
    levels = [1, 2, 5, 10, 15, 20, 30, 50, 60]
    max_tokens = 20
    all_level_results = []

    for level in levels:
        print(f"  Concurrency {level}: sending {level} requests...")
        tasks = [
            send_request(
                client,
                server,
                model,
                messages_medium(),
                max_tokens=max_tokens,
                request_index=i,
                prompt_desc="medium",
            )
            for i in range(level)
        ]
        records = list(await asyncio.gather(*tasks))

        ok = [r for r in records if r.status == 200]
        rejected = [r for r in records if r.status == 503]
        avg_lat = sum(r.latency_s for r in ok) / max(len(ok), 1)
        print(f"    -> {len(ok)} OK, {len(rejected)} rejected, avg latency {avg_lat:.2f}s")

        all_level_results.append(
            {
                "concurrency": level,
                "requests": [asdict(r) for r in records],
            }
        )

        # Brief cooldown between levels
        if level != levels[-1]:
            await asyncio.sleep(3)

    return ExperimentResult(
        experiment="concurrency_sweep",
        timestamp=datetime.now(UTC).isoformat(),
        server=server,
        parameters={"concurrency_levels": levels, "max_tokens": max_tokens},
        results=all_level_results,
        duration_s=0,
        metadata=build_metadata(server, model),
    )


# ---------------------------------------------------------------------------
# Experiment 2: Backpressure flood
# ---------------------------------------------------------------------------


async def backpressure_flood(
    client: httpx.AsyncClient,
    server: str,
    model: str,
) -> ExperimentResult:
    n = 100
    max_tokens = 20
    print(f"  Sending {n} simultaneous requests (queue max=50)...")

    tasks = [
        send_request(
            client,
            server,
            model,
            messages_medium(),
            max_tokens=max_tokens,
            request_index=i,
            prompt_desc="medium",
        )
        for i in range(n)
    ]
    records = list(await asyncio.gather(*tasks))

    ok = [r for r in records if r.status == 200]
    rejected = [r for r in records if r.status == 503]
    timeout = [r for r in records if r.status == 504]
    other = [r for r in records if r.status not in (200, 503, 504)]
    print(f"    -> 200: {len(ok)}, 503: {len(rejected)}, 504: {len(timeout)}, other: {len(other)}")
    if rejected:
        avg_reject_latency = sum(r.latency_s for r in rejected) / len(rejected)
        print(f"    -> 503 avg latency: {avg_reject_latency:.4f}s (should be near-zero)")
    if ok:
        latencies = sorted(r.latency_s for r in ok)
        print(
            f"    -> 200 latency: min={latencies[0]:.2f}s, "
            f"median={latencies[len(latencies) // 2]:.2f}s, "
            f"max={latencies[-1]:.2f}s"
        )

    return ExperimentResult(
        experiment="backpressure_flood",
        timestamp=datetime.now(UTC).isoformat(),
        server=server,
        parameters={"total_requests": n, "max_tokens": max_tokens},
        results={
            "summary": {
                "status_200": len(ok),
                "status_503": len(rejected),
                "status_504": len(timeout),
                "other_errors": len(other),
            },
            "requests": [asdict(r) for r in records],
        },
        duration_s=0,
        metadata=build_metadata(server, model),
    )


# ---------------------------------------------------------------------------
# Experiment 3: Streaming vs non-streaming
# ---------------------------------------------------------------------------


async def streaming_vs_nonstreaming(
    client: httpx.AsyncClient,
    server: str,
    model: str,
) -> ExperimentResult:
    n = 5
    max_tokens = 50
    configs = [
        ("non_streaming_sequential", False, True),
        ("streaming_sequential", True, True),
        ("non_streaming_concurrent_5", False, False),
        ("streaming_concurrent_5", True, False),
    ]
    all_results = {}

    for label, stream_flag, sequential in configs:
        print(f"  {label}: {n} requests...")
        if sequential:
            records = await run_sequential(
                client,
                server,
                model,
                messages_medium(),
                n=n,
                stream=stream_flag,
                max_tokens=max_tokens,
                prompt_desc="medium",
            )
        else:
            records = await run_concurrent(
                client,
                server,
                model,
                messages_medium(),
                n=n,
                concurrency=5,
                stream=stream_flag,
                max_tokens=max_tokens,
                prompt_desc="medium",
            )

        ok = [r for r in records if r.status == 200]
        avg_lat = sum(r.latency_s for r in ok) / max(len(ok), 1)
        if stream_flag:
            ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
            avg_ttft = sum(ttfts) / max(len(ttfts), 1) if ttfts else 0
            print(f"    -> avg latency {avg_lat:.2f}s, avg TTFT {avg_ttft:.3f}s")
        else:
            print(f"    -> avg latency {avg_lat:.2f}s")

        all_results[label] = [asdict(r) for r in records]

    return ExperimentResult(
        experiment="streaming_vs_nonstreaming",
        timestamp=datetime.now(UTC).isoformat(),
        server=server,
        parameters={"requests_per_config": n, "max_tokens": max_tokens},
        results=all_results,
        duration_s=0,
        metadata=build_metadata(server, model),
    )


# ---------------------------------------------------------------------------
# Experiment 4: Prompt length impact
# ---------------------------------------------------------------------------


async def prompt_length_impact(
    client: httpx.AsyncClient,
    server: str,
    model: str,
) -> ExperimentResult:
    n = 3
    max_tokens = 30
    prompts = [
        ("short", messages_short()),
        ("long", messages_long()),
        ("5_turn", messages_5_turn()),
        ("10_turn", messages_10_turn()),
    ]
    all_results = []

    for label, messages in prompts:
        print(f"  Prompt '{label}' ({len(messages)} messages): {n} sequential requests...")
        records = await run_sequential(
            client,
            server,
            model,
            messages,
            n=n,
            stream=False,
            max_tokens=max_tokens,
            prompt_desc=label,
        )
        ok = [r for r in records if r.status == 200]
        avg_lat = sum(r.latency_s for r in ok) / max(len(ok), 1)
        avg_tokens = sum(r.tokens for r in ok) / max(len(ok), 1)
        print(f"    -> avg latency {avg_lat:.2f}s, avg tokens {avg_tokens:.0f}")
        all_results.append(
            {
                "prompt_type": label,
                "message_count": len(messages),
                "requests": [asdict(r) for r in records],
            }
        )

    return ExperimentResult(
        experiment="prompt_length_impact",
        timestamp=datetime.now(UTC).isoformat(),
        server=server,
        parameters={"requests_per_prompt": n, "max_tokens": max_tokens},
        results=all_results,
        duration_s=0,
        metadata=build_metadata(server, model),
    )


# ---------------------------------------------------------------------------
# Experiment 5: TTFT deep dive
# ---------------------------------------------------------------------------


async def ttft_deep_dive(
    client: httpx.AsyncClient,
    server: str,
    model: str,
) -> ExperimentResult:
    max_tokens = 50
    results = {}

    # Phase 1: Sequential (no contention)
    n_seq = 20
    print(f"  Phase 1: {n_seq} sequential streaming requests...")
    records = await run_sequential(
        client,
        server,
        model,
        messages_medium(),
        n=n_seq,
        stream=True,
        max_tokens=max_tokens,
        prompt_desc="medium",
    )
    ttfts = [r.ttft_s for r in records if r.ttft_s is not None]
    if ttfts:
        print(
            f"    -> TTFT: mean={sum(ttfts) / len(ttfts):.3f}s, "
            f"min={min(ttfts):.3f}s, max={max(ttfts):.3f}s"
        )
    results["sequential"] = [asdict(r) for r in records]

    await asyncio.sleep(5)

    # Phase 2: Concurrent (contention)
    n_conc = 20
    concurrency = 5
    print(f"  Phase 2: {n_conc} streaming requests at concurrency={concurrency}...")
    records = await run_concurrent(
        client,
        server,
        model,
        messages_medium(),
        n=n_conc,
        concurrency=concurrency,
        stream=True,
        max_tokens=max_tokens,
        prompt_desc="medium",
    )
    ttfts = [r.ttft_s for r in records if r.ttft_s is not None]
    if ttfts:
        print(
            f"    -> TTFT: mean={sum(ttfts) / len(ttfts):.3f}s, "
            f"min={min(ttfts):.3f}s, max={max(ttfts):.3f}s"
        )
    results["concurrent_5"] = [asdict(r) for r in records]

    return ExperimentResult(
        experiment="ttft_deep_dive",
        timestamp=datetime.now(UTC).isoformat(),
        server=server,
        parameters={
            "sequential_count": n_seq,
            "concurrent_count": n_conc,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
        },
        results=results,
        duration_s=0,
        metadata=build_metadata(server, model),
    )


# ---------------------------------------------------------------------------
# Experiment 6: Token generation rate
# ---------------------------------------------------------------------------


async def token_generation_rate(
    client: httpx.AsyncClient,
    server: str,
    model: str,
) -> ExperimentResult:
    n = 5
    max_tokens = 100
    print(f"  Sending {n} sequential streaming requests (max_tokens={max_tokens})...")

    records = await run_sequential(
        client,
        server,
        model,
        messages_long(),
        n=n,
        stream=True,
        max_tokens=max_tokens,
        prompt_desc="long",
    )

    enriched = []
    for r in records:
        intervals: list[float] = []
        if r.chunk_timestamps and len(r.chunk_timestamps) > 1:
            for i in range(1, len(r.chunk_timestamps)):
                intervals.append(r.chunk_timestamps[i] - r.chunk_timestamps[i - 1])

        mean_interval = sum(intervals) / len(intervals) if intervals else None
        tps = len(intervals) / sum(intervals) if intervals and sum(intervals) > 0 else None
        entry = asdict(r)
        entry["inter_token_intervals_s"] = intervals
        entry["mean_interval_s"] = mean_interval
        entry["tokens_per_sec"] = tps
        enriched.append(entry)

        if tps:
            print(f"    Request {r.request_index}: {r.tokens} tokens, {tps:.1f} tok/s")

    return ExperimentResult(
        experiment="token_generation_rate",
        timestamp=datetime.now(UTC).isoformat(),
        server=server,
        parameters={"request_count": n, "max_tokens": max_tokens},
        results=enriched,
        duration_s=0,
        metadata=build_metadata(server, model),
    )


# ---------------------------------------------------------------------------
# Experiment 7: Sustained throughput
# ---------------------------------------------------------------------------


async def sustained_throughput(
    client: httpx.AsyncClient,
    server: str,
    model: str,
) -> ExperimentResult:
    duration_s = 120
    concurrency = 5
    max_tokens = 30
    all_records: list[RequestRecord] = []
    experiment_start = time.time()
    idx = 0

    sem = asyncio.Semaphore(concurrency)

    print(f"  Running sustained load for {duration_s}s at concurrency={concurrency}...")

    # Worker loop: each worker sends requests sequentially until time expires
    async def worker() -> None:
        nonlocal idx
        while time.time() - experiment_start < duration_s:
            async with sem:
                i = idx
                idx += 1
                r = await send_request(
                    client,
                    server,
                    model,
                    messages_medium(),
                    max_tokens=max_tokens,
                    request_index=i,
                    prompt_desc="medium",
                )
                all_records.append(r)
                status = "OK" if r.status == 200 else f"ERR:{r.status}"
                elapsed = time.time() - experiment_start
                print(f"    [{elapsed:5.0f}s] #{i:3d} {status} {r.latency_s:.2f}s {r.tokens}tok")

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers, return_exceptions=True)

    # Bucket into 10-second windows
    window_size = 10
    base_time = min(r.started_at for r in all_records) if all_records else experiment_start
    windows = []
    for ws in range(0, duration_s + window_size, window_size):
        we = ws + window_size
        bucket = [r for r in all_records if ws <= (r.started_at - base_time) < we]
        if not bucket:
            continue
        ok = [r for r in bucket if r.status == 200]
        windows.append(
            {
                "window_start_s": ws,
                "window_end_s": we,
                "requests_started": len(bucket),
                "requests_successful": len(ok),
                "requests_failed": len(bucket) - len(ok),
                "total_tokens": sum(r.tokens for r in ok),
                "avg_latency_s": (sum(r.latency_s for r in ok) / len(ok) if ok else 0),
            }
        )

    total_ok = [r for r in all_records if r.status == 200]
    total_tokens = sum(r.tokens for r in total_ok)
    wall_time = time.time() - experiment_start
    print(
        f"  Completed: {len(all_records)} requests, "
        f"{len(total_ok)} OK, {total_tokens} tokens in {wall_time:.0f}s "
        f"({total_tokens / wall_time:.1f} tok/s)"
    )

    return ExperimentResult(
        experiment="sustained_throughput",
        timestamp=datetime.now(UTC).isoformat(),
        server=server,
        parameters={
            "duration_s": duration_s,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
        },
        results={
            "summary": {
                "total_requests": len(all_records),
                "successful": len(total_ok),
                "failed": len(all_records) - len(total_ok),
                "total_tokens": total_tokens,
                "wall_time_s": wall_time,
                "requests_per_sec": len(total_ok) / wall_time if wall_time > 0 else 0,
                "tokens_per_sec": total_tokens / wall_time if wall_time > 0 else 0,
            },
            "windows": windows,
            "requests": [asdict(r) for r in all_records],
        },
        duration_s=0,
        metadata=build_metadata(server, model),
    )


# ---------------------------------------------------------------------------
# Experiment 8: Graceful degradation
# ---------------------------------------------------------------------------


async def graceful_degradation(
    client: httpx.AsyncClient,
    server: str,
    model: str,
) -> ExperimentResult:
    n = 50
    max_tokens = 30
    print(f"  Sending {n} simultaneous requests (queue capacity={n})...")

    tasks = [
        send_request(
            client,
            server,
            model,
            messages_medium(),
            max_tokens=max_tokens,
            request_index=i,
            prompt_desc="medium",
        )
        for i in range(n)
    ]
    records = list(await asyncio.gather(*tasks))

    ok = [r for r in records if r.status == 200]
    rejected = [r for r in records if r.status == 503]
    timeout = [r for r in records if r.status == 504]
    other = [r for r in records if r.status not in (200, 503, 504)]

    print(f"    -> 200: {len(ok)}, 503: {len(rejected)}, 504: {len(timeout)}, other: {len(other)}")

    latencies = sorted(r.latency_s for r in ok) if ok else []
    if latencies:
        p50_idx = len(latencies) // 2
        p95_idx = int(len(latencies) * 0.95)
        print(
            f"    -> Latency: min={latencies[0]:.2f}s, "
            f"p50={latencies[p50_idx]:.2f}s, "
            f"p95={latencies[min(p95_idx, len(latencies) - 1)]:.2f}s, "
            f"max={latencies[-1]:.2f}s"
        )

    summary = {
        "total_requests": n,
        "status_200": len(ok),
        "status_503": len(rejected),
        "status_504": len(timeout),
        "other_errors": len(other),
    }
    if latencies:
        summary["latency_distribution"] = {
            "min": latencies[0],
            "p25": latencies[len(latencies) // 4],
            "p50": latencies[len(latencies) // 2],
            "p75": latencies[3 * len(latencies) // 4],
            "p95": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)],
            "max": latencies[-1],
        }

    return ExperimentResult(
        experiment="graceful_degradation",
        timestamp=datetime.now(UTC).isoformat(),
        server=server,
        parameters={"total_requests": n, "max_tokens": max_tokens},
        results={
            "summary": summary,
            "requests": [asdict(r) for r in records],
        },
        duration_s=0,
        metadata=build_metadata(server, model),
    )


# ---------------------------------------------------------------------------
# Registry and runner
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "concurrency_sweep": concurrency_sweep,
    "backpressure_flood": backpressure_flood,
    "streaming_vs_nonstreaming": streaming_vs_nonstreaming,
    "prompt_length_impact": prompt_length_impact,
    "ttft_deep_dive": ttft_deep_dive,
    "token_generation_rate": token_generation_rate,
    "sustained_throughput": sustained_throughput,
    "graceful_degradation": graceful_degradation,
}


async def run_experiments(
    experiments: list[str],
    server: str,
    api_key: str | None,
    model: str,
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async with httpx.AsyncClient(headers=headers) as client:
        # Pre-flight health check
        try:
            r = await client.get(f"{server}/health", timeout=10)
            if r.status_code != 200:
                print(f"Health check failed: {r.status_code}")
                return
        except httpx.ConnectError:
            print(f"Cannot connect to {server}")
            return

        print(f"Server: {server}")
        print(f"Model: {model}")
        print(f"Experiments: {', '.join(experiments)}")
        print()

        for i, name in enumerate(experiments):
            func = EXPERIMENTS[name]
            print(f"{'=' * 60}")
            print(f"  EXPERIMENT: {name}")
            print(f"{'=' * 60}")

            exp_start = time.perf_counter()
            result = await func(client, server, model)
            result.duration_s = time.perf_counter() - exp_start

            path = save_result(result)
            print(f"  Completed in {result.duration_s:.1f}s -> {path}")
            print()

            # Cooldown between experiments
            if i < len(experiments) - 1:
                print("  Cooling down 10s...")
                await asyncio.sleep(10)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-serving-api experiment suite")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=f"Run a specific experiment: {', '.join(EXPERIMENTS.keys())}",
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--url", default=DEFAULT_SERVER, help="Server URL")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY"),
        help="Bearer token",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    args = parser.parse_args()

    if args.list:
        for name in EXPERIMENTS:
            print(f"  {name}")
        return

    if args.all:
        experiments = list(EXPERIMENTS.keys())
    elif args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return
        experiments = [args.experiment]
    else:
        parser.print_help()
        return

    asyncio.run(run_experiments(experiments, args.url, args.api_key, args.model))


if __name__ == "__main__":
    main()
