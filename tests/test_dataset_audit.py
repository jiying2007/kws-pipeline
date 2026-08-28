#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

from qualification_fixture import write_wav

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "training" / "audit_dataset.py"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDITOR), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        train_dir = root / "train"
        eval_dir = root / "eval"
        train_dir.mkdir()
        eval_dir.mkdir()

        write_wav(train_dir / "a.wav", seconds=1)
        write_wav(train_dir / "b.wav", seconds=1)
        # Make b byte-distinct while preserving valid PCM16 WAV format.
        data = bytearray((train_dir / "b.wav").read_bytes())
        data[-2:] = (123).to_bytes(2, "little", signed=True)
        (train_dir / "b.wav").write_bytes(data)
        write_wav(eval_dir / "c.wav", seconds=1)
        data = bytearray((eval_dir / "c.wav").read_bytes())
        data[-2:] = (-321).to_bytes(2, "little", signed=True)
        (eval_dir / "c.wav").write_bytes(data)

        train_manifest = root / "train.tsv"
        train_manifest.write_text("train/a.wav\t1 2\ntrain/b.wav\t\n", encoding="utf-8")
        eval_manifest = root / "eval.jsonl"
        eval_manifest.write_text(
            json.dumps(
                {
                    "recording": "eval-c",
                    "path": "eval/c.wav",
                    "duration_s": 1.0,
                    "expected": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = root / "audit.json"

        clean = run(
            "--split",
            f"train={train_manifest}",
            "--split",
            f"eval={eval_manifest}",
            "--report",
            str(report),
        )
        assert clean.returncode == 0, clean.stderr
        result = json.loads(report.read_text(encoding="utf-8"))
        assert result["clean"] is True
        assert result["cross_split_leaks"] == []
        assert result["splits"]["train"]["examples"] == 2
        assert result["splits"]["eval"]["examples"] == 1

        shutil.copyfile(train_dir / "a.wav", eval_dir / "leaked.wav")
        eval_manifest.write_text(
            json.dumps(
                {
                    "recording": "leaked",
                    "path": "eval/leaked.wav",
                    "duration_s": 1.0,
                    "expected": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        leaked = run(
            "--split",
            f"train={train_manifest}",
            "--split",
            f"eval={eval_manifest}",
        )
        assert leaked.returncode == 1
        leaked_result = json.loads(leaked.stdout)
        assert leaked_result["clean"] is False
        assert len(leaked_result["cross_split_leaks"]) == 1
        assert leaked_result["cross_split_leaks"][0]["splits"] == ["eval", "train"]

        duplicate_manifest = root / "dup.tsv"
        duplicate_manifest.write_text(
            "train/a.wav\t1 2\ntrain/a.wav\t1 2\n", encoding="utf-8"
        )
        duplicate = run(
            "--split",
            f"dup={duplicate_manifest}",
            "--fail-within-split",
        )
        assert duplicate.returncode == 1
        duplicate_result = json.loads(duplicate.stdout)
        assert len(duplicate_result["within_split_duplicates"]) == 1

    print("test_dataset_audit: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
