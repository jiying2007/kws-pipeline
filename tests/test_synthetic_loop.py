#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def metric_line(label: str, metrics: dict) -> str:
    return (
        f"{label}: frr={float(metrics['frr']):.6f} "
        f"far_per_hour={float(metrics['far_per_hour']):.6f} "
        f"p95_ms={float(metrics['p95_post_end_latency_ms']):.3f} "
        f"expected={int(metrics['expected'])} matched={int(metrics['matched'])} "
        f"fa={int(metrics['false_accepts'])} fr={int(metrics['false_rejects'])}"
    )


def failure_diagnostic(work: pathlib.Path, completed: subprocess.CompletedProcess[str]) -> str:
    lines = [f"iterate rc={completed.returncode}"]
    manifest_path = work / "synthetic-loop-manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest.get("rounds", []):
                round_index = int(item.get("round", -1))
                lines.append(metric_line(f"round{round_index}.calibration", item["calibration"]))
                lines.append(metric_line(f"round{round_index}.test", item["test"]))
                lines.append(
                    f"round{round_index}.replay: hard_negatives="
                    f"{int(item.get('cumulative_hard_negatives', 0))} "
                    f"missed_positives={int(item.get('cumulative_missed_positives', 0))}"
                )
            if isinstance(manifest.get("synthetic_qualification"), dict):
                lines.append(
                    metric_line(
                        "qualification", manifest["synthetic_qualification"]
                    )
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            lines.append(f"manifest diagnostic failed: {exc}")
    else:
        generated = sorted(
            str(path.relative_to(work)) for path in work.rglob("*") if path.is_file()
        )
        lines.append("manifest missing; generated files: " + ", ".join(generated[-20:]))
    stderr_lines = [line for line in completed.stderr.splitlines() if line.strip()]
    if stderr_lines:
        lines.append("stderr tail: " + " | ".join(stderr_lines[-12:]))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    args = parser.parse_args()
    runner = args.runner.resolve()
    assert runner.is_file(), runner

    with tempfile.TemporaryDirectory() as td:
        work = pathlib.Path(td) / "loop"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "training" / "iterate.py"),
                "--config",
                str(ROOT / "configs" / "training" / "xiaowo.synthetic.json"),
                "--runner",
                str(runner),
                "--work-dir",
                str(work),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert completed.returncode == 0, failure_diagnostic(work, completed)
        manifest_path = work / "synthetic-loop-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["evidence_class"] == "synthetic-only"
        assert manifest["qualified"] is True
        assert len(manifest["rounds"]) >= 2
        assert manifest["synthetic_qualification"]["frr"] <= 0.05
        assert manifest["synthetic_qualification"]["far_per_hour"] == 0.0
        assert "replay" in manifest
        assert len(manifest["replay"]["hard_negatives_sha256"]) == 64
        assert len(manifest["replay"]["missed_positives_sha256"]) == 64
        assert (work / "best" / "model.kwm").stat().st_size > 72
        assert (work / "best" / "keywords.kwk").stat().st_size > 24
        assert (work / "dataset" / "train.tsv").stat().st_size > 0
        assert (work / "dataset" / "calibration.tsv").stat().st_size > 0
        assert (work / "dataset" / "test.tsv").stat().st_size > 0
        assert (work / "dataset" / "qualification.tsv").stat().st_size > 0
        audit = json.loads((work / "dataset-audit.json").read_text(encoding="utf-8"))
        assert audit["clean"] is True

    print("test_synthetic_loop: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
