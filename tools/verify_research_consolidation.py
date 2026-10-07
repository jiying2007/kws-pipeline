#!/usr/bin/env python3
"""Verify retained public source bytes offline; this is not qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


def verify(root: Path, manifest: dict) -> int:
    if manifest.get("schema") != "research-consolidation-retention-v1":
        raise ValueError("unsupported retention schema")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source inventory must be nonempty")
    declared = set()
    for source in sources:
        pr, commit = source.get("pr"), source.get("commit")
        if (type(pr) is not int or pr <= 0 or not isinstance(commit, str)
                or not re.fullmatch("[0-9a-f]{40}", commit)
                or (pr, commit) in declared):
            raise ValueError("invalid or duplicate source identity")
        declared.add((pr, commit))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("retention file inventory must be nonempty")
    seen = set()
    for entry in files:
        name = entry.get("path")
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError("invalid retained path")
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or str(rel) != name:
            raise ValueError("noncanonical retained path")
        if name in seen:
            raise ValueError("duplicate retained path")
        seen.add(name)
        if entry.get("mode") not in ("100644", "100755"):
            raise ValueError("unsupported retained file mode")
        if (entry.get("source_pr"), entry.get("source_commit")) not in declared:
            raise ValueError(f"undeclared retained source: {name}")
        path = root / name
        if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise ValueError("symlink in retained source path")
        if not path.is_file():
            raise ValueError(f"missing retained source: {name}")
        if os.name == "posix":
            actual_mode = "100755" if path.stat().st_mode & 0o111 else "100644"
            if actual_mode != entry["mode"]:
                raise ValueError(f"retained executable mode changed: {name}")
        raw = path.read_bytes()
        expected_size = entry.get("bytes")
        if type(expected_size) is not int or len(raw) != expected_size:
            raise ValueError(f"retained byte count changed: {name}")
        for field, length in (("sha256", 64), ("git_blob_sha1", 40), ("source_commit", 40)):
            value = entry.get(field)
            if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{" + str(length) + "}", value):
                raise ValueError(f"invalid {field}: {name}")
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"] or blob != entry["git_blob_sha1"]:
            raise ValueError(f"retained source bytes changed: {name}")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", default="research/consolidation/core-2026-10-07.json")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / args.manifest).read_text(encoding="utf-8"))
    count = verify(root, manifest)
    print(f"PASS: {count} retained source files; bytes and source provenance only")


if __name__ == "__main__":
    main()
