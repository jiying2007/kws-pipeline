from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_json(path: pathlib.Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def run_gate(*, runner: pathlib.Path, work: pathlib.Path) -> dict:
    replay = work / "best/hard-negatives.tsv"
    stream_root = work / "hard-negative-stream"
    plan_path = stream_root / "plan.json"
    if not replay.is_file() or replay.stat().st_size == 0:
        raise ValueError("finalized active FAR replay is missing")
    stream_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/plan_far_stream.py"),
            "--negative-manifest",
            str(replay),
            "--injections-per-clip",
            "1",
            "--baseline-seconds",
            "900",
            "--minimum-payload-gap-seconds",
            "2.0",
            "--minimum-payload-rate-per-minute",
            "8.0",
            "--output",
            str(plan_path),
        ],
        check=True,
    )
    plan = read_json(plan_path, "continuous FAR capacity plan")
    stream_seconds = int(plan["planned_seconds"])
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval/long_far_stream.py"),
            "--runner",
            str(runner),
            "--model",
            str(work / "best/model.kwm"),
            "--keywords",
            str(work / "best/keywords.kwk"),
            "--negative-manifest",
            str(replay),
            "--hard-negative-rate-per-minute",
            "0.0",
            "--seconds",
            str(stream_seconds),
            "--seed",
            "5501",
            "--output-dir",
            str(stream_root),
            "--max-far-per-hour",
            "0",
        ],
        check=True,
    )
    training = read_json(work / "training-run-summary.json", "training summary")
    stream = read_json(stream_root / "summary.json", "continuous FAR summary")
    injections = [
        json.loads(line)
        for line in (stream_root / "hard-negative-injections.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_clips = int(training["candidate_selection"]["far_holdout_clip_count"])
    if int(plan.get("schema_version", 0)) != 1 or str(plan.get("policy", "")) != "coverage-capacity-v1":
        raise ValueError("continuous FAR capacity plan contract drifted")
    if int(plan.get("negative_manifest_clips", -1)) != expected_clips:
        raise ValueError("continuous FAR capacity plan clip count differs from finalized holdout")
    if int(plan.get("injections_per_clip", -1)) != 1:
        raise ValueError("continuous FAR capacity plan must schedule exactly one injection per clip")
    if int(plan.get("coverage_injections", -1)) != expected_clips:
        raise ValueError("continuous FAR planned coverage count differs from finalized holdout")
    if float(plan.get("minimum_payload_gap_seconds", -1.0)) != 2.0:
        raise ValueError("continuous FAR planned payload gap drifted")
    if float(plan.get("minimum_payload_rate_per_minute", -1.0)) != 8.0:
        raise ValueError("continuous FAR planned payload rate floor drifted")
    if int(plan.get("planned_seconds", -1)) != int(stream.get("seconds", -2)):
        raise ValueError("continuous FAR executed duration differs from capacity plan")
    if float(plan.get("planned_payload_rate_per_minute", -1.0)) < 8.0:
        raise ValueError("continuous FAR capacity plan fell below 8 payloads/min")
    if int(stream.get("schema_version", 0)) != 2:
        raise ValueError("continuous FAR summary is not coverage-aware schema v2")
    if int(stream.get("false_accepts", -1)) != 0 or float(stream.get("far_per_hour", -1.0)) != 0.0:
        raise ValueError("continuous FAR gate is not zero-error")
    if not bool(stream.get("full_negative_manifest_coverage")):
        raise ValueError("continuous FAR gate did not cover the full active holdout")
    if int(stream.get("negative_manifest_clips", -1)) != expected_clips:
        raise ValueError("continuous FAR manifest clip count differs from finalized holdout")
    if int(stream.get("unique_hard_negative_clips_injected", -1)) != expected_clips:
        raise ValueError("continuous FAR did not inject every active holdout clip")
    if int(stream.get("min_observed_hard_negative_injections_per_clip", 0)) != 1:
        raise ValueError("continuous FAR must inject every active holdout clip exactly once")
    if int(stream.get("coverage_forced_injections", -1)) != expected_clips:
        raise ValueError("continuous FAR forced coverage count differs from active holdout size")
    if int(stream.get("hard_negative_injections", -1)) != expected_clips or len(injections) != expected_clips:
        raise ValueError("continuous FAR injection count differs from exact forced coverage")
    if float(stream.get("hard_negative_rate_per_minute", -1.0)) != 0.0:
        raise ValueError("continuous FAR random hard-negative injection is not disabled")
    if float(stream.get("coverage_hard_negative_gain", 0.0)) != 0.90:
        raise ValueError("continuous FAR coverage gain drifted from 0.90")
    if any(not bool(row.get("coverage_required")) for row in injections):
        raise ValueError("continuous FAR contains a non-coverage random injection")

    seconds = float(stream["seconds"])
    actual_rate = len(injections) * 60.0 / seconds
    observed_gaps = [
        float(current["start_second"])
        - (float(previous["start_second"]) + float(previous["source_seconds"]))
        for previous, current in zip(injections, injections[1:])
    ]
    observed_min_gap = min(observed_gaps) if observed_gaps else seconds
    if actual_rate < 8.0:
        raise ValueError("continuous FAR actual payload rate fell below 8/min")
    if observed_min_gap + 1.0e-9 < 2.0:
        raise ValueError("continuous FAR payload gap is inside decoder retention window")
    contract = {
        "schema_version": 1,
        "policy": "semantic-negative-boundary-v1",
        "all_injections_coverage_required": True,
        "payload_injections": len(injections),
        "actual_injection_rate_per_minute": actual_rate,
        "minimum_payload_gap_seconds": 2.0,
        "observed_min_payload_gap_seconds": observed_min_gap,
        "decoder_retention_proof_window_seconds": 1.6,
        "full_negative_manifest_coverage": True,
        "negative_manifest_sha256": stream["negative_manifest_sha256"],
        "model_sha256": stream["model_sha256"],
        "capacity_plan_policy": plan["policy"],
        "capacity_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "planned_seconds": int(plan["planned_seconds"]),
        "planned_payload_rate_per_minute": float(plan["planned_payload_rate_per_minute"]),
        "max_clip_span_seconds": int(plan["max_clip_span_seconds"]),
        "required_start_stride_seconds": int(plan["required_start_stride_seconds"]),
    }
    (stream_root / "stream-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = run_gate(runner=args.runner.resolve(), work=args.work_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
