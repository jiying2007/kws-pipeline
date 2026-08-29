#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

from qualification_fixture import write_model, write_pack, write_tokens

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        tokens = root / "tokens.txt"
        _, fingerprint = write_tokens(tokens)
        model = root / "model.kwm"
        pack = root / "keywords.kwk"
        write_model(model, fingerprint)
        write_pack(pack, fingerprint)
        output = root / "far"
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "eval" / "long_far_stream.py"),
                "--runner",
                str(args.runner.resolve()),
                "--model",
                str(model),
                "--keywords",
                str(pack),
                "--seconds",
                "3",
                "--seed",
                "404",
                "--output-dir",
                str(output),
                "--max-far-per-hour",
                "0",
            ]
        )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["qualified"] is True
        assert summary["seconds"] == 3
        assert summary["false_accepts"] == 0
        assert summary["far_per_hour"] == 0.0
        assert sum(summary["profile_seconds"].values()) == 3
        assert (output / "detections.jsonl").read_text(encoding="utf-8") == ""

    print("test_long_far: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
