#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import sys

LINE_RE = re.compile(r"^\s*([^:]+):\s*\d+:")


def coverage(paths: list[pathlib.Path]) -> tuple[int, int]:
    covered = 0
    total = 0
    for path in paths:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = LINE_RE.match(raw)
            if match is None:
                continue
            count = match.group(1).strip()
            if count == "-":
                continue
            total += 1
            if count not in {"#####", "====="}:
                try:
                    if int(count.rstrip("*")) > 0:
                        covered += 1
                except ValueError:
                    pass
    return covered, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--minimum", type=float, default=55.0)
    args = parser.parse_args()
    files = sorted(args.root.glob("*.gcov"))
    if not files:
        raise ValueError("no .gcov files found")
    covered, total = coverage(files)
    if total == 0:
        raise ValueError("gcov files contain no executable lines")
    percent = 100.0 * covered / total
    print(f"C line coverage: {covered}/{total} = {percent:.2f}%")
    if percent + 1.0e-9 < args.minimum:
        print(
            f"coverage {percent:.2f}% is below required {args.minimum:.2f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
