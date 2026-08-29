#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("tests"))
    parser.add_argument("--runner", type=pathlib.Path)
    args = parser.parse_args()
    tests = sorted(
        path for path in args.root.glob("test_*.py") if path.name != "test_inventory.py"
    )
    if not tests:
        raise ValueError("no Python tests discovered")
    for path in tests:
        command = [sys.executable, str(path)]
        if "--runner" in path.read_text(encoding="utf-8"):
            if args.runner is None:
                print(f"inventory: {path} requires explicit runner; syntax-only discovery")
                continue
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    print(f"inventory discovered {len(tests)} Python test files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
