#!/usr/bin/env python3
"""Prepare exact public N1 saved inputs from an explicit local cache, offline.

The cache has the allowlisted destination layout data/, source/, scorer/.
This tool has no network, acquisition, model, audio-playback, or native path.
The immutable source URLs are provenance for separate explicit preparation.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "source_fixtures/n1-saved-inputs.json"
MANIFEST_SHA256 = "b6b48f4d365422fa29e3523692c0b9e341e35a315bbd0ad953ee272306bebf69"
MAX_FILE_BYTES = 65536
TOTAL_BYTES = 137742
FILE_COUNT = 19


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_path(root, relative):
    parts = PurePosixPath(relative).parts
    if (not relative or "\\" in relative or relative.startswith("/")
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("Unsafe fixture path")
    path = root
    if root.is_symlink():
        raise ValueError("Symlink root prohibited")
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink fixture path prohibited")
    return path


def load_manifest():
    raw = MANIFEST.read_bytes()
    if sha(raw) != MANIFEST_SHA256:
        raise ValueError("Fixture allowlist identity changed")
    value = json.loads(raw)
    rows = value["files"]
    if (len(rows) != FILE_COUNT or sum(row["bytes"] for row in rows) != TOTAL_BYTES
            or len({row["destination"] for row in rows}) != FILE_COUNT):
        raise ValueError("Fixture count/aggregate size mismatch")
    for row in rows:
        if (row["repository"] not in ("jiying2007/kws-data", "jiying2007/kws-pipeline")
                or not re.fullmatch("[0-9a-f]{40}", row["commit"])
                or not 0 < row["bytes"] <= MAX_FILE_BYTES
                or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
            raise ValueError("Invalid immutable fixture identity")
        expected = "https://raw.githubusercontent.com/" + row["repository"] + "/" + row["commit"] + "/" + row["path"]
        if row["raw_url"] != expected:
            raise ValueError("Fixture provenance URL mismatch")
    return rows


def read_verified(root, row):
    path = checked_path(root, row["destination"])
    if not path.is_file() or path.stat().st_size != row["bytes"]:
        raise ValueError("Missing or wrong-sized fixture: " + row["destination"])
    with path.open("rb") as stream:
        raw = stream.read(row["bytes"] + 1)
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if len(raw) != row["bytes"] or sha(raw) != row["sha256"] or blob != row["git_blob_sha1"]:
        raise ValueError("Fixture byte identity mismatch: " + row["destination"])
    return raw


def verify_output(root, rows):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Prepared output must be a regular directory")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Prepared output contains symlink")
    actual = {path.relative_to(root).as_posix() for path in paths if path.is_file()}
    if actual != {row["destination"] for row in rows}:
        raise ValueError("Prepared output is not the closed 19-file set")
    for row in rows:
        read_verified(root, row)


def prepare(source, output):
    rows = load_manifest()
    source, output = Path(source), Path(output)
    if output.exists() or output.is_symlink():
        verify_output(output, rows)
        return {"status": "PASS_EXISTING_N1_SAVED_FIXTURES", "files": FILE_COUNT,
                "bytes": TOTAL_BYTES, "network_requests": 0}
    # Verify every input before creating an output directory.
    inputs = [(row, read_verified(source, row)) for row in rows]
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("Output parent must already be a regular directory")
    with tempfile.TemporaryDirectory(prefix=".n1-saved-stage-", dir=output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        for row, raw in inputs:
            path = checked_path(stage, row["destination"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        verify_output(stage, rows)
        if output.exists():
            raise FileExistsError("Refusing to replace prepared output")
        stage.rename(output)
    return {"status": "PASS_N1_SAVED_FIXTURES", "files": FILE_COUNT,
            "bytes": TOTAL_BYTES, "network_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
