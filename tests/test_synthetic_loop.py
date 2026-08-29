#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


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
        assert completed.returncode == 0, (
            f"iterate rc={completed.returncode}\nSTDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
        manifest_path = work / "synthetic-loop-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["evidence_class"] == "synthetic-only"
        assert manifest["qualified"] is True
        assert len(manifest["rounds"]) >= 2
        assert manifest["synthetic_qualification"]["frr"] <= 0.05
        assert manifest["synthetic_qualification"]["far_per_hour"] == 0.0
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
