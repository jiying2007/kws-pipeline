#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: pathlib.Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    result = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=pathlib.Path)
    parser.add_argument("second", type=pathlib.Path)
    args = parser.parse_args()
    first = inventory(args.first)
    second = inventory(args.second)
    if first != second:
        keys = sorted(set(first) | set(second))
        for key in keys:
            if first.get(key) != second.get(key):
                print(f"mismatch {key}: {first.get(key)} != {second.get(key)}", file=sys.stderr)
        return 1
    print(f"reproducible SDK inventory matched: {len(first)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
