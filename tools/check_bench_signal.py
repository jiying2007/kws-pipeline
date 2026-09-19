#!/usr/bin/env python3
"""Fail-closed contract for the hosted benchmark signal.

CI runs ``kws_bench`` and asserts nothing: as long as it exits zero, whatever it
printed is accepted. That is a smoke test, not a gate. A change that made the
bench print zeros, silently drop its worst case, or drift by an order of
magnitude would stay green.

The checker separates two things the raw output mixes together:

* **structural invariants** -- geometry, model/engine size, MACs per frame,
  audio length. These are properties of the model and the build, not of the
  machine, so they are asserted strictly. They are checked by **agreement
  between cases** rather than against hard-coded numbers, so a legitimate model
  update (which moves every case together) still passes, while a case that
  degrades on its own does not.
* **timing** -- ``cpu_s`` / ``rtf`` / ``us_per_audio_s``. These depend on the
  runner, so they get a deliberately loose ceiling: enough headroom that machine
  noise never trips it, tight enough that an order-of-magnitude regression
  cannot hide behind "it's just a slow runner".
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

REQUIRED_FIELDS = (
    "case",
    "frontend",
    "keywords",
    "trie_nodes",
    "geometry",
    "gate",
    "model_bytes",
    "engine_bytes",
    "estimated_macs_per_frame",
    "audio_s",
    "cpu_s",
    "rtf",
    "us_per_audio_s",
)

# Describe the build rather than the run, so every case must agree on them.
STRUCTURAL_FIELDS = (
    "geometry",
    "model_bytes",
    "engine_bytes",
    "estimated_macs_per_frame",
    "audio_s",
)

# Depend on the machine. They must be measured and finite, and stay under a
# deliberately generous ceiling.
TIMING_FIELDS = ("cpu_s", "rtf", "us_per_audio_s")


def parse_rows(text: str) -> list[tuple[int, dict[str, str], list[str]]]:
    rows: list[tuple[int, dict[str, str], list[str]]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        row: dict[str, str] = {}
        malformed: list[str] = []
        for item in line.split():
            key, sep, value = item.partition("=")
            if not sep:
                malformed.append(item)
                continue
            row[key] = value
        rows.append((lineno, row, malformed))
    return rows


def as_number(row: dict[str, str], field: str, label: str, errors: list[str]) -> float | None:
    try:
        return float(row[field])
    except ValueError:
        errors.append(f"{label}: {field} is not a number: {row[field]!r}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check kws_bench output for structural and order-of-magnitude regressions."
    )
    parser.add_argument("--input", type=pathlib.Path, default=None,
                        help="read a saved bench output instead of stdin")
    parser.add_argument("--max-rtf", type=float, default=0.1,
                        help="ceiling on real-time factor (default 0.1)")
    parser.add_argument("--min-cases", type=int, default=2,
                        help="minimum number of bench cases (default 2)")
    args = parser.parse_args()

    if args.max_rtf <= 0.0:
        raise ValueError("--max-rtf must be positive")
    if args.min_cases < 1:
        raise ValueError("--min-cases must be at least 1")

    text = args.input.read_text(encoding="utf-8") if args.input is not None else sys.stdin.read()
    rows = parse_rows(text)
    if not rows:
        raise ValueError("no bench rows parsed: refusing to pass vacuously")

    errors: list[str] = []

    names = [row.get("case", "") for _, row, _ in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append("duplicate case names: " + ", ".join(duplicates))
    if len(rows) < args.min_cases:
        errors.append(f"expected at least {args.min_cases} cases, got {len(rows)}")

    for lineno, row, malformed in rows:
        label = row.get("case") or f"line {lineno}"
        if malformed:
            errors.append(f"{label}: not key=value: {', '.join(malformed)}")
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"{label}: missing fields: {', '.join(missing)}")
            continue
        for field in TIMING_FIELDS:
            value = as_number(row, field, label, errors)
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0.0:
                errors.append(f"{label}: {field} is not a positive finite number: {row[field]}")
        rtf = as_number(row, "rtf", label, errors)
        if rtf is not None and rtf > args.max_rtf:
            errors.append(f"{label}: rtf {rtf} exceeds --max-rtf {args.max_rtf}")

    # Structural invariants: the build, not the machine. Agreement between cases
    # catches a single case drifting without pinning values a model update is
    # allowed to move.
    for field in STRUCTURAL_FIELDS:
        values = sorted({row[field] for _, row, _ in rows if field in row})
        if len(values) > 1:
            errors.append(f"structural field {field} differs between cases: {values}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    measured = [float(row["rtf"]) for _, row, _ in rows]
    print(
        f"check_bench_signal: ok cases={len(rows)} "
        f"max_rtf={max(measured):.6f} ceiling={args.max_rtf}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
