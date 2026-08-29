#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import resource
import subprocess
import time


def read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_existing(paths: list[pathlib.Path]) -> str | None:
    for path in paths:
        value = read_text(path)
        if value:
            return value
    return None


def thermal_max_c() -> float | None:
    values = []
    for path in pathlib.Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        text = read_text(path)
        try:
            value = float(text) if text is not None else None
        except ValueError:
            value = None
        if value is not None:
            values.append(value / 1000.0 if value > 1000.0 else value)
    return max(values) if values else None


def process_rss_kib() -> float:
    status = read_text(pathlib.Path("/proc/self/status")) or ""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1])
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def cpu_model() -> str:
    info = read_text(pathlib.Path("/proc/cpuinfo")) or ""
    for key in ("model name", "Processor", "Hardware"):
        for line in info.splitlines():
            if line.lower().startswith(key.lower() + ":"):
                return line.split(":", 1)[1].strip()
    return platform.machine()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--board-revision", required=True)
    parser.add_argument("--soc")
    parser.add_argument("--toolchain", required=True)
    parser.add_argument("--compiler-flags", required=True)
    parser.add_argument("--audio-frontend", required=True)
    parser.add_argument("--audio-frontend-sha256")
    parser.add_argument("--soak-hours", type=float, default=0.0)
    parser.add_argument("--cpu-percent", type=float, required=True)
    parser.add_argument("--stack-high-water-bytes", type=float, required=True)
    parser.add_argument("--average-power-mw", type=float, required=True)
    parser.add_argument("--power-raw", type=pathlib.Path)
    parser.add_argument("--instrument-id")
    parser.add_argument("--calibration-id")
    args = parser.parse_args()

    governor = first_existing(
        list(pathlib.Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
    ) or "unknown"
    now = time.time()
    evidence = {
        "schema_version": 2,
        "collector": "collect_target_evidence.py",
        "collector_source_sha": git_sha(),
        "collected_unix_s": now,
        "target": args.target,
        "board_revision": args.board_revision,
        "soc": args.soc or cpu_model(),
        "toolchain": args.toolchain,
        "compiler_flags": args.compiler_flags,
        "governor": governor,
        "audio_frontend": args.audio_frontend,
        "audio_frontend_sha256": args.audio_frontend_sha256,
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_online": read_text(pathlib.Path("/sys/devices/system/cpu/online")),
        "uptime_s": float((read_text(pathlib.Path("/proc/uptime")) or "0").split()[0]),
        "soak_hours": args.soak_hours,
        "cpu_percent": args.cpu_percent,
        "rss_kib": process_rss_kib(),
        "stack_high_water_bytes": args.stack_high_water_bytes,
        "max_temp_c": thermal_max_c() if thermal_max_c() is not None else -273.15,
        "average_power_mw": args.average_power_mw,
        "instrument_id": args.instrument_id,
        "calibration_id": args.calibration_id,
    }
    if args.power_raw:
        evidence["power_raw_sha256"] = sha256_file(args.power_raw.resolve(strict=True))
        evidence["power_raw_name"] = args.power_raw.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote target evidence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
