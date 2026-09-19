#!/usr/bin/env python3
"""Turn a frozen-candidate qualification gate failure into a readable diagnosis.

The gate in the frozen qualification workflows is three bare shell tests::

    test "${{ steps.fresh.outputs.exit_code }}" = '0'
    test "${{ steps.shadow.outputs.exit_code }}" = '0'
    test "${{ steps.formal.outputs.exit_code }}" = '0'

When it fails the job log says nothing except a non-zero status, and every number
that would explain the failure is inside an uploaded zip. A frozen candidate has
failed this gate four times in a row with no diagnosis beyond "exit 1".

This tool replaces those three tests. It reports, per stage, whether the stage
ran, whether its summary exists, whether the summary claims a pass, and -- when
the gate limits are available -- exactly which metric on which split exceeded
which limit, with the measured value next to it.

Two properties keep it honest:

* **Fail-closed.** A stage that reported success but left no readable summary is
an error, not a pass. So is a summary that claims ``qualified`` while its own
metrics violate the limits it was measured against: the verdict is re-derived
from the metrics rather than trusted, because a stale or hand-edited summary is
exactly the kind of thing that makes a gate meaningless.
* **No inference of missing stages.** A stage that never ran (because an earlier
stage failed) is reported as "did not run", never as "failed" -- the distinction
decides where to look.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

STAGE_SEP = ":"

# Metric -> (where to read it, gate field). The gate is the conjunction of all
# four; reporting them separately is the point, because a model that satisfies
# three and misses one is a very different problem from one that misses all four.
BASE_METRICS = (
    ("frr", "max_frr"),
    ("far_per_hour", "max_far_per_hour"),
    ("p95_post_end_latency_ms", "max_p95_latency_ms"),
)
FAR_DOMAIN_METRIC = ("distance:far", "frr", "max_far_frr")


def load_json(path: pathlib.Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{label}: summary is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: summary is unreadable: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label}: summary is not a JSON object")
    return value


def as_float(value: object, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{label} is not a number: {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def load_gates(path: pathlib.Path) -> dict[str, float]:
    raw = load_json(path, "gates")
    gates = raw.get("domain_gates")
    if not isinstance(gates, dict):
        raise ValueError("gates file has no 'domain_gates' object")
    fields = ("max_frr", "max_far_per_hour", "max_p95_latency_ms", "max_far_frr")
    result: dict[str, float] = {}
    for field in fields:
        if field not in gates:
            raise ValueError(f"domain_gates is missing {field}")
        result[field] = as_float(gates[field], f"domain_gates.{field}")
    return result


def violations_for(split: str, row: dict, gates: dict[str, float] | None) -> list[str]:
    """Re-derive the verdict from the metrics rather than trusting the flag."""
    if gates is None:
        return []
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return []
    found: list[str] = []
    for metric, limit_key in BASE_METRICS:
        if metric not in metrics:
            continue
        measured = as_float(metrics[metric], f"{split}.{metric}")
        if measured > gates[limit_key]:
            found.append(
                f"{split}: {metric}={measured:g} exceeds {limit_key}={gates[limit_key]:g}"
            )
    domains = row.get("domains")
    domain_key, metric, limit_key = FAR_DOMAIN_METRIC
    if isinstance(domains, dict):
        table = domains.get("domains")
        if isinstance(table, dict) and isinstance(table.get(domain_key), dict):
            far = table[domain_key]
            if metric in far:
                measured = as_float(far[metric], f"{split}.{domain_key}.{metric}")
                if measured > gates[limit_key]:
                    found.append(
                        f"{split}: {domain_key}.{metric}={measured:g} "
                        f"exceeds {limit_key}={gates[limit_key]:g}"
                    )
    return found


def classify(name: str, summary: pathlib.Path | None, exit_code: str,
             gates: dict[str, float] | None) -> dict:
    """Return a stage record: status, and the findings that explain it."""
    record: dict = {"stage": name, "exit_code": exit_code, "status": "", "findings": []}
    ran = exit_code == "0"
    started = exit_code != ""

    if not started:
        record["status"] = "did-not-run"
        record["findings"].append("stage did not run: an earlier stage failed")
        return record

    if not ran:
        record["status"] = "failed"
        record["findings"].append(f"stage exited {exit_code}")
    else:
        record["status"] = "ran"

    if summary is None:
        if ran:
            record["status"] = "failed"
            record["findings"].append("no summary path supplied, so the pass cannot be confirmed")
        return record

    try:
        payload = load_json(summary, name)
    except ValueError as exc:
        record["status"] = "failed"
        record["findings"].append(str(exc))
        return record

    claimed = payload.get("qualified")
    record["claimed_qualified"] = claimed

    violations: list[str] = []
    splits = payload.get("splits")
    if isinstance(splits, dict):
        for split in sorted(splits):
            row = splits[split]
            if not isinstance(row, dict):
                continue
            violations.extend(violations_for(split, row, gates))
            if gates is None and row.get("qualified") is False:
                violations.append(f"{split}: summary reports qualified=false")

    if violations:
        record["status"] = "failed"
        record["findings"].extend(violations)
        if ran and claimed is not False:
            record["findings"].append(
                "summary claims a pass but its own metrics violate the gate"
            )
    elif ran and claimed is False:
        record["status"] = "failed"
        record["findings"].append("summary reports qualified=false")

    return record


def render(records: list[dict]) -> str:
    lines = ["frozen qualification gate:"]
    for record in records:
        lines.append(f"  {record['stage']}: {record['status']} (exit={record['exit_code']!r})")
        for finding in record["findings"]:
            lines.append(f"      - {finding}")
    return "\n".join(lines)


def render_markdown(records: list[dict]) -> str:
    lines = ["## Frozen candidate qualification gate", ""]
    lines.append("| Stage | Status | Exit | Findings |")
    lines.append("| --- | --- | --- | --- |")
    for record in records:
        findings = "<br>".join(record["findings"]) or "&mdash;"
        lines.append(
            f"| {record['stage']} | {record['status']} | `{record['exit_code']}` | {findings} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_stage(spec: str) -> tuple[str, pathlib.Path | None, str]:
    parts = spec.split(STAGE_SEP, 2)
    if len(parts) != 3:
        raise ValueError(f"--stage expects name:summary:exit, got {spec!r}")
    name, summary, exit_code = parts
    if not name:
        raise ValueError(f"--stage has an empty stage name: {spec!r}")
    return name, (pathlib.Path(summary) if summary else None), exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        required=True,
        metavar="NAME:SUMMARY:EXIT",
        help="one stage, in run order; SUMMARY may be empty, EXIT is the step's exit code",
    )
    parser.add_argument("--gates", type=pathlib.Path, default=None,
                        help="config carrying domain_gates, so metrics can be re-checked")
    parser.add_argument("--step-summary", type=pathlib.Path, default=None,
                        help="write a markdown table here (GITHUB_STEP_SUMMARY)")
    args = parser.parse_args()

    gates = load_gates(args.gates) if args.gates is not None else None
    specs = [parse_stage(spec) for spec in args.stage]
    names = [name for name, _, _ in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("duplicate stage names: " + ", ".join(duplicates))

    records = [classify(name, summary, exit_code, gates) for name, summary, exit_code in specs]
    text = render(records)
    if args.step_summary is not None:
        args.step_summary.write_text(render_markdown(records) + "\n", encoding="utf-8")

    failed = [record for record in records if record["status"] != "ran"]
    if failed:
        print(text, file=sys.stderr)
        return 1
    print(text)
    print(f"report_frozen_qualification_gate: ok stages={len(records)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
