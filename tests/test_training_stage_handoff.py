#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from base_stage_receipt import verify_receipt, write_receipt  # noqa: E402


def expect_value_error(fn, label: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError: {label}")


def main() -> int:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    old = os.environ.get("GITHUB_SHA")
    os.environ["GITHUB_SHA"] = head
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = root / "config.json"
            work = root / "work"
            work.mkdir()
            manifest = work / "domain-loop-manifest.json"
            receipt = work / "base-stage-receipt.json"
            output = root / "github-output.txt"
            config.write_text(
                json.dumps(
                    {
                        "qualification_holdout_seed": 271843,
                        "retired_qualification_holdout_seeds": [271842],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps({"records": [{"round": 0, "calibration_gate": True, "test_gate": True}]})
                + "\n",
                encoding="utf-8",
            )
            written = write_receipt(
                config_path=config,
                work=work,
                output=receipt,
                iteration_exit_code=0,
            )
            if written["formal_seed_consumed"] or written["qualification_holdout_seed"] != 271843:
                raise AssertionError("base receipt seed contract drifted")
            verified = verify_receipt(
                config_path=config,
                work=work,
                receipt_path=receipt,
                github_output=output,
            )
            if verified != written:
                raise AssertionError("verified staged receipt differs from written receipt")
            if output.read_text(encoding="utf-8") != "iteration_exit_code=0\n":
                raise AssertionError("staged receipt did not publish iteration exit code")

            original_manifest = manifest.read_text(encoding="utf-8")
            manifest.write_text(json.dumps({"records": [{"round": 99}]}) + "\n", encoding="utf-8")
            expect_value_error(
                lambda: verify_receipt(
                    config_path=config,
                    work=work,
                    receipt_path=receipt,
                    github_output=None,
                ),
                "tampered domain manifest",
            )
            manifest.write_text(original_manifest, encoding="utf-8")

            config.write_text(
                json.dumps(
                    {
                        "qualification_holdout_seed": 271844,
                        "retired_qualification_holdout_seeds": [271842],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expect_value_error(
                lambda: verify_receipt(
                    config_path=config,
                    work=work,
                    receipt_path=receipt,
                    github_output=None,
                ),
                "tampered config/formal seed",
            )

            expect_value_error(
                lambda: write_receipt(
                    config_path=config,
                    work=work,
                    output=receipt,
                    iteration_exit_code=2,
                ),
                "infrastructure exit code",
            )
    finally:
        if old is None:
            os.environ.pop("GITHUB_SHA", None)
        else:
            os.environ["GITHUB_SHA"] = old
    print("staged model-training handoff receipt: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
