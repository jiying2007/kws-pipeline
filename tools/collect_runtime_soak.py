#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import time

from runtime_soak_contract import (
    CPU_MEASUREMENT_CONTRACT_ID, CPU_PERCENT_SEMANTICS,
    RUNTIME_SOAK_SCHEMA_VERSION, cpu_percent,
)


def read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def process_rss_kib(pid: int) -> float | None:
    text = read_text(pathlib.Path(f"/proc/{pid}/status"))
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def process_cpu_seconds(pid: int) -> float | None:
    text = read_text(pathlib.Path(f"/proc/{pid}/stat"))
    if text is None:
        return None
    end = text.rfind(")")
    if end < 0 or end + 2 >= len(text):
        return None
    fields = text[end + 2 :].split()
    # After pid/comm, fields[0] is proc-stat field 3 (state).
    # utime/stime are fields 14/15, therefore indexes 11/12 here.
    if len(fields) <= 12:
        return None
    try:
        ticks = int(fields[11]) + int(fields[12])
        hz = int(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError):
        return None
    return ticks / float(hz) if hz > 0 else None


def process_thread_count(pid: int) -> int | None:
    text = read_text(pathlib.Path(f"/proc/{pid}/status"))
    for line in (text or "").splitlines():
        if line.startswith("Threads:"):
            try:
                count = int(line.split()[1])
                return count if count > 0 else None
            except (IndexError, ValueError):
                return None
    return None


def online_cpu_count() -> int:
    value = os.cpu_count()
    return value if isinstance(value, int) and value > 0 else 1


def thermal_max_c() -> float | None:
    values: list[float] = []
    for path in pathlib.Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        text = read_text(path)
        try:
            value = float(text) if text is not None else None
        except ValueError:
            value = None
        if value is not None:
            values.append(value / 1000.0 if value > 1000.0 else value)
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", required=True, type=float)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--sample-seconds", type=float, default=60.0)
    parser.add_argument(
        "--command",
        required=True,
        nargs=argparse.REMAINDER,
        help="child command and all remaining arguments; place this option last",
    )
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0.0 for value in (args.hours, args.sample_seconds)):
        raise ValueError("hours and sample-seconds must be > 0")
    if not args.command:
        parser.error("--command requires a child executable")

    capacity_cpus = online_cpu_count()
    started = time.monotonic()
    process = subprocess.Popen(args.command)
    initial_cpu = process_cpu_seconds(process.pid)
    samples: list[dict] = []
    completed_requested_duration = False
    try:
        deadline = started + args.hours * 3600.0
        while True:
            now = time.monotonic()
            if process.poll() is not None:
                raise RuntimeError(f"soak command exited early: {process.returncode}")
            if now >= deadline:
                completed_requested_duration = True
                break
            samples.append(
                {
                    "elapsed_s": now - started,
                    "rss_kib": process_rss_kib(process.pid),
                    "cpu_seconds": process_cpu_seconds(process.pid),
                    "thread_count": process_thread_count(process.pid),
                    "temp_c": thermal_max_c(),
                }
            )
            time.sleep(min(args.sample_seconds, max(0.0, deadline - time.monotonic())))

        final_now = time.monotonic()
        samples.append(
            {
                "elapsed_s": final_now - started,
                "rss_kib": process_rss_kib(process.pid),
                "cpu_seconds": process_cpu_seconds(process.pid),
                "thread_count": process_thread_count(process.pid),
                "temp_c": thermal_max_c(),
            }
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    # Exclude process termination/wait time from the measured CPU interval.
    elapsed_s = samples[-1]["elapsed_s"]
    last_cpu = samples[-1]["cpu_seconds"]
    process_cpu_time = None
    average_cpu_percent = None
    if initial_cpu is not None and last_cpu is not None and elapsed_s > 0.0:
        process_cpu_time = last_cpu - initial_cpu
        average_cpu_percent = cpu_percent(process_cpu_time, elapsed_s)
    thread_values = [row["thread_count"] for row in samples if row["thread_count"] is not None]

    rss_values = [
        row["rss_kib"] for row in samples if row["rss_kib"] is not None
    ]
    temp_values = [
        row["temp_c"] for row in samples if row["temp_c"] is not None
    ]
    result = {
        "schema_version": RUNTIME_SOAK_SCHEMA_VERSION,
        "command": args.command,
        "pid": process.pid,
        "cpu_capacity_count": capacity_cpus,
        "measurement_contract_id": CPU_MEASUREMENT_CONTRACT_ID,
        "cpu_percent_semantics": CPU_PERCENT_SEMANTICS,
        "process_cpu_seconds": process_cpu_time,
        "wall_seconds": elapsed_s,
        "max_thread_count": max(thread_values) if thread_values else None,
        "audio_seconds": None,
        "cpu_seconds_per_audio_second": None,
        "requested_hours": args.hours,
        "elapsed_seconds": elapsed_s,
        "elapsed_hours": elapsed_s / 3600.0,
        "completed_requested_duration": completed_requested_duration,
        "termination_returncode": process.returncode,
        "sample_seconds": args.sample_seconds,
        "initial_cpu_seconds": initial_cpu,
        "samples": samples,
        "max_rss_kib": max(rss_values) if rss_values else None,
        "average_cpu_percent": average_cpu_percent,
        "max_temp_c": max(temp_values) if temp_values else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote soak evidence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
