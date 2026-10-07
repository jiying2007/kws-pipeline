#!/usr/bin/env python3
"""Explicitly prepare three pinned public logs; offline tests never fetch data."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re
import ssl
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/fixtures/eval_context/sources.json"
DEFAULT_OUTPUT = ROOT / "build/eval-context-fixtures"


def verify(raw, row):
    if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError(row["name"] + ": pinned size/SHA-256 mismatch")
    return raw


def fetch(source, row):
    # Fixed HTTPS origin, ordinary public GET, no redirects or credentials.
    connection = http.client.HTTPSConnection(
        "raw.githubusercontent.com", timeout=30, context=ssl.create_default_context())
    try:
        path = "/" + source["repository"] + "/" + source["commit"] + "/" + row["path"]
        connection.request("GET", path, headers={"Accept-Encoding": "identity"})
        response = connection.getresponse()
        if response.status != 200 or response.getheader("Content-Encoding") not in (None, "identity"):
            raise ValueError(row["name"] + ": fixture GET failed (HTTP " + str(response.status) + ")")
        return verify(response.read(row["bytes"] + 1), row)
    finally:
        connection.close()


def prepare(output, source_dir=None):
    source = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (source.get("schema") != "eval-context-fixtures-v1"
            or source.get("repository") != "jiying2007/kws-data"
            or re.fullmatch(r"[0-9a-f]{40}", source.get("commit", "")) is None):
        raise ValueError("invalid immutable fixture source")
    rows = source["files"]
    if [r["name"] for r in rows] != ["original.raw.jsonl", "positive300.raw.jsonl", "negative300.raw.jsonl"]:
        raise ValueError("unexpected fixture set")
    bodies = []
    for row in rows:
        if (type(row["bytes"]) is not int or not 0 < row["bytes"] <= 65536
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                or not row["path"].startswith("research/") or ".." in row["path"].split("/")):
            raise ValueError("invalid fixture pin")
        destination = output / row["name"]
        if destination.exists():
            raw = verify(destination.read_bytes(), row)
        elif source_dir is not None:
            raw = verify((source_dir / row["name"]).read_bytes(), row)
        else:
            raw = fetch(source, row)
        bodies.append((destination, raw))
    # Verify every body before creating files; no failed GET becomes an empty fixture.
    output.mkdir(parents=True, exist_ok=True)
    for destination, raw in bodies:
        if not destination.exists():
            destination.write_bytes(raw)
    return dict(commit=source["commit"], files=len(rows), bytes=sum(len(raw) for _, raw in bodies),
                output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-dir", type=Path,
                        help="verify and copy already downloaded named fixtures without network")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.source_dir), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as exc:
        print("fixture preparation failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
