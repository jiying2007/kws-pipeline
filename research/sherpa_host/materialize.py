#!/usr/bin/env python3
"""Materialize the locked research source from explicit local inputs, offline.

Apache-2.0; see the repository LICENSE. No configure, compiler or model is run.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or p.as_posix() != value or any(x in (".", "..") for x in p.parts):
        raise ValueError("noncanonical relative path: " + value)
    if any(ord(c) < 32 for c in value):
        raise ValueError("control character in path")
    return p


def regular(root, name):
    root = Path(root).absolute()
    path = root / relative(name)
    # Resolve no links, including the root and any of its ancestors.
    for part in reversed((path, *path.parents)):
        mode = part.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("symlink input: " + str(part))
        if part != path and not stat.S_ISDIR(mode):
            raise ValueError("non-directory path component")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("input is not a regular file: " + str(path))
    return path


def checked_bytes(root, name, sha256, size=None):
    data = regular(root, name).read_bytes()
    if digest(data) != sha256 or (size is not None and len(data) != size):
        raise ValueError("input identity differs: " + name)
    return data


def exact_patch(original, patch):
    """Apply one UTF-8 unified file diff, with exact offsets/context and no fuzz."""
    lines = patch.decode("utf-8").splitlines(keepends=True)
    if len(lines) < 3 or not lines[0].startswith("--- a/") or not lines[1].startswith("+++ b/"):
        raise ValueError("expected single-file unified patch")
    old_name, new_name = lines[0][6:].rstrip("\n"), lines[1][6:].rstrip("\n")
    if relative(old_name) != relative(new_name):
        raise ValueError("patch changes file name")
    source = original.decode("utf-8").splitlines(keepends=True)
    output, cursor, at = [], 0, 2
    while at < len(lines):
        match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n", lines[at])
        if not match:
            raise ValueError("invalid hunk header")
        old_line, old_count, new_line, new_count = (int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1))
        start = old_line - (old_count != 0)
        if not cursor <= start <= len(source):
            raise ValueError("invalid hunk offset")
        output.extend(source[cursor:start]); cursor = start
        if len(output) != new_line - (new_count != 0):
            raise ValueError("new hunk offset differs")
        at += 1; used_old = used_new = 0
        while at < len(lines) and not lines[at].startswith("@@ "):
            line = lines[at]; at += 1
            if not line or line[0] not in " +-":
                raise ValueError("unsupported patch record")
            if line[0] in " -":
                if cursor >= len(source) or source[cursor] != line[1:]:
                    raise ValueError("patch context differs")
                cursor += 1; used_old += 1
            if line[0] in " +":
                output.append(line[1:]); used_new += 1
        if (used_old, used_new) != (old_count, new_count):
            raise ValueError("hunk counts differ")
    output.extend(source[cursor:])
    return "".join(output).encode("utf-8")


def validate_manifest(manifest):
    if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("unsupported source manifest")
    paths = set()
    for row in manifest["files"]:
        path = str(relative(row["path"])); relative(row["input_path"])
        if path in paths or any(path.startswith(p + "/") or p.startswith(path + "/") for p in paths):
            raise ValueError("duplicate/conflicting output path")
        paths.add(path)
        if not isinstance(row["bytes"], int) or row["bytes"] < 0:
            raise ValueError("invalid byte count")
        for field in ("sha256", "input_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row[field]):
                raise ValueError("invalid SHA256")
        if "patch" in row:
            relative(row["patch"])
            if not re.fullmatch(r"[0-9a-f]{64}", row["patch_sha256"]):
                raise ValueError("invalid patch SHA256")
    return paths


def source_receipt(manifest):
    return {"schema": 1, "source_manifest_canonical_sha256": digest(json.dumps(manifest, sort_keys=True).encode()),
            "files": len(manifest["files"]), "compiled": False, "model_loaded": False}


def verify_tree(output, manifest):
    expected = validate_manifest(manifest)
    actual = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink output")
        if path.is_file():
            actual.add(path.relative_to(output).as_posix())
        elif not path.is_dir():
            raise ValueError("special output")
    if actual != expected | {"MATERIALIZED.json"}:
        raise ValueError("output file set differs")
    for row in manifest["files"]:
        checked_bytes(output, row["path"], row["sha256"], row["bytes"])
    if load_json(regular(output, "MATERIALIZED.json")) != source_receipt(manifest):
        raise ValueError("materialization receipt differs")


def materialize(inputs, output, manifest=None, repo=ROOT):
    manifest = manifest or load_json(repo / "sources.lock.json")
    validate_manifest(manifest)
    required = {row["component"] for row in manifest["files"]} - {"repository"}
    if set(inputs) != required:
        raise ValueError("input component set differs: " + str(sorted(required)))
    roots = {name: Path(value).absolute() for name, value in inputs.items()}
    roots["repository"] = repo
    if any(row["path"] == "source/sherpa-onnx/c-api/kws-c-api.cc" for row in manifest["files"]):
        wrapper = load_json(repo / "wrapper-extraction.json")
        original = checked_bytes(roots["sherpa-onnx"], wrapper["original_path"], wrapper["original_sha256"])
        lines = original.splitlines(keepends=True)
        for span in wrapper["segments"]:
            data = b"".join(lines[span["original_start_line"] - 1:span["original_end_line"]])
            if digest(data) != span["sha256"] or len(data) != span["bytes"]:
                raise ValueError("original wrapper span differs: " + span["name"])
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output must not exist")
    parent = output.parent.resolve(strict=True)
    if parent != output.parent:
        raise ValueError("output parent must be canonical, without links")
    for root in [repo.parents[1], *roots.values()]:
        if output.is_relative_to(root.resolve()):
            raise ValueError("output must be outside repository and input trees")
    temporary = Path(tempfile.mkdtemp(prefix=".kws-materialize-", dir=parent))
    try:
        for row in manifest["files"]:
            data = checked_bytes(roots[row["component"]], row["input_path"], row["input_sha256"])
            if "patch" in row:
                patch = checked_bytes(repo, row["patch"], row["patch_sha256"])
                data = exact_patch(data, patch)
            if digest(data) != row["sha256"] or len(data) != row["bytes"]:
                raise ValueError("output identity differs: " + row["path"])
            target = temporary / row["path"]; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data); target.chmod(0o755 if row["path"] == "cli/run-kws" else 0o644)
        receipt = source_receipt(manifest)
        (temporary / "MATERIALIZED.json").write_text(json.dumps(receipt, indent=2) + "\n")
        verify_tree(temporary, manifest)
        # Reserve final name without replacing an existing destination.
        output.mkdir()
        for child in temporary.iterdir():
            child.rename(output / child.name)
        temporary.rmdir()
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="JSON mapping component name to extracted local root")
    parser.add_argument("--output", required=True, type=Path, help="new source directory outside the repository")
    args = parser.parse_args()
    print(json.dumps(materialize(load_json(args.inputs), args.output), sort_keys=True))
