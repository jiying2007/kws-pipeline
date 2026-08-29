#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True, nargs="+")
    parser.add_argument("--hours", required=True, type=float)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--sample-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.hours <= 0.0 or args.sample_seconds <= 0.0:
        raise ValueError("hours and sample-seconds must be > 0")
    started = time.monotonic()
    process = subprocess.Popen(args.command)
    samples: list[dict] = []
    try:
        deadline = started + args.hours * 3600.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"soak command exited early: {process.returncode}")
            status = pathlib.Path(f"/proc/{process.pid}/status")
            rss_kib = None
            if status.is_file():
                for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("VmRSS:"):
                        rss_kib = float(line.split()[1])
                        break
            samples.append(
                {
                    "elapsed_s": time.monotonic() - started,
                    "rss_kib": rss_kib,
                }
            )
            time.sleep(min(args.sample_seconds, max(0.0, deadline - time.monotonic())))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    result = {
        "schema_version": 1,
        "command": args.command,
        "requested_hours": args.hours,
        "elapsed_hours": (time.monotonic() - started) / 3600.0,
        "exit_code": process.returncode,
        "samples": samples,
        "max_rss_kib": max(
            (row["rss_kib"] for row in samples if row["rss_kib"] is not None),
            default=None,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote soak evidence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
