#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from corpus_identity import corpus_digest, inspect_pcm16_wav


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    args = parser.parse_args()
    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("corpus identity manifest schema_version must be 1")
    rows = value.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("corpus identity manifest recordings must be non-empty")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"recordings[{index}] must be an object")
        path = pathlib.Path(str(row.get("path", ""))).resolve(strict=True)
        measured = inspect_pcm16_wav(path)
        for key in ("file_sha256", "pcm_sha256", "frames"):
            if row.get(key) != measured[key]:
                raise ValueError(f"recordings[{index}].{key} does not match {path}")
        normalized.append(row)
    digest = corpus_digest(normalized)
    if value.get("corpus_sha256") != digest:
        raise ValueError("corpus_sha256 does not match canonical recording identity")
    print(f"verified corpus identity: {digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
