#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("tests"))
    parser.add_argument("--workflow", action="append", required=True, type=pathlib.Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="test filename intentionally executed only outside Actions",
    )
    args = parser.parse_args()
    tests = sorted(path.name for path in args.root.glob("test_*.py"))
    if not tests:
        raise ValueError("no Python tests discovered")
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8") for path in args.workflow
    )
    excluded = set(args.exclude)
    missing = [name for name in tests if name not in workflow_text and name not in excluded]
    unknown_exclusions = sorted(excluded - set(tests))
    if unknown_exclusions:
        raise ValueError(
            "inventory exclusions do not exist: " + ", ".join(unknown_exclusions)
        )
    if missing:
        print(
            "Python tests missing from declared workflows: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    print(
        f"inventory covered {len(tests)} Python test file(s) across "
        f"{len(args.workflow)} workflow(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
