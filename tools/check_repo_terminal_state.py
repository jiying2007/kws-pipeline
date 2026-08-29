#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


def git_lines(*args: str) -> list[str]:
    value = subprocess.check_output(["git", *args], text=True).strip()
    return value.splitlines() if value else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-branch", default="main")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = git_lines("status", "--porcelain")
    result = {
        "schema_version": 1,
        "branch": branch,
        "sha": sha,
        "clean": not status,
        "expected_branch": args.expect_branch,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status or branch != args.expect_branch:
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
