"""Run the Locust load test at several concurrency levels and summarise the results.

For each user count this runs ``locustfile.py`` headless for a fixed duration, then reads
Locust's stats CSV and the token totals the locustfile writes, and records one row:
successful requests, failures, 503 backpressure responses, p50/p95/p99 latency,
requests/s and completion tokens/s.

Output (under --out):
    u<N>_stats.csv, u<N>_tokens.json   raw Locust output for each user count
    summary.json                       all rows plus run metadata
    summary.md                         the same rows as a Markdown table

Example:
    uv run --extra loadtest python loadtest/sweep.py --host http://localhost:8000 \
        --users 1,4,8,16 --duration 60s --out results/baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
REQUEST_NAME = "chat_completion"
BACKPRESSURE_NAME = "chat_completion [503 backpressure]"


def _row(stats_csv: Path, name: str) -> dict[str, str] | None:
    with stats_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["Name"] == name:
                return row
    return None


def run_point(host: str, users: int, duration: str, out: Path) -> dict:
    prefix = out / f"u{users}"
    tokens_file = out / f"u{users}_tokens.json"
    env = {**os.environ, "TOKENS_FILE": str(tokens_file)}
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "--host",
        host,
        "-u",
        str(users),
        "-r",
        str(users),
        "-t",
        duration,
        "--csv",
        str(prefix),
        "--only-summary",
        "--stop-timeout",
        "120",
    ]
    print(f"[sweep] users={users} duration={duration}", flush=True)
    subprocess.run(cmd, env=env, check=False, stdout=subprocess.DEVNULL)

    ok = _row(Path(f"{prefix}_stats.csv"), REQUEST_NAME) or {}
    bp = _row(Path(f"{prefix}_stats.csv"), BACKPRESSURE_NAME) or {}
    tokens = json.loads(tokens_file.read_text()) if tokens_file.exists() else {}

    def num(row: dict[str, str], key: str) -> float:
        try:
            return float(row.get(key, 0) or 0)
        except ValueError:
            return 0.0

    return {
        "users": users,
        "requests_ok": tokens.get("ok_requests", 0),
        "failures": int(num(ok, "Failure Count") + num(bp, "Failure Count")),
        "backpressure_503": tokens.get("backpressure_503", 0),
        "p50_s": round(num(ok, "50%") / 1000, 2),
        "p95_s": round(num(ok, "95%") / 1000, 2),
        "p99_s": round(num(ok, "99%") / 1000, 2),
        "avg_s": round(num(ok, "Average Response Time") / 1000, 2),
        "req_per_s": round(tokens.get("ok_requests", 0) / tokens["elapsed_seconds"], 3)
        if tokens.get("elapsed_seconds")
        else 0.0,
        "completion_tok_per_s": tokens.get("completion_tokens_per_second"),
    }


def to_markdown(rows: list[dict]) -> str:
    header = (
        "| Users | OK | Failed | 503 | p50 (s) | p95 (s) | p99 (s) | req/s | tok/s |\n"
        "|------:|---:|-------:|----:|--------:|--------:|--------:|------:|------:|\n"
    )
    lines = [
        f"| {r['users']} | {r['requests_ok']} | {r['failures']} | {r['backpressure_503']} "
        f"| {r['p50_s']} | {r['p95_s']} | {r['p99_s']} | {r['req_per_s']} "
        f"| {r['completion_tok_per_s']} |"
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--users", default="1,4,8,16", help="comma-separated user counts")
    parser.add_argument("--duration", default="60s", help="Locust run time per point")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="", help="free-text note stored in summary.json")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [run_point(args.host, int(u), args.duration, args.out) for u in args.users.split(",")]

    summary = {
        "label": args.label,
        "host": args.host,
        "duration": args.duration,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": {
            k: os.environ[k]
            for k in ("MODEL", "MAX_TOKENS", "LOADTEST_PROMPT", "HONOR_RETRY_AFTER")
            if k in os.environ
        },
        "rows": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    table = to_markdown(rows)
    (args.out / "summary.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
