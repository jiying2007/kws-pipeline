#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tarfile

REFERENCE_CLASS = "speech-like-provider-reference-v1"
RECEIPT_CLASS = "speech-like-sherpa-backend-bundle-v1"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_stream(stream) -> tuple[int, str]:
    h = hashlib.sha256()
    total = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        total += len(chunk)
        h.update(chunk)
    return total, h.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def backend_contract(reference_path: pathlib.Path, platform_key: str) -> dict:
    reference = load_object(reference_path)
    if int(reference.get("schema_version", 0)) != 1 or reference.get("evidence_class") != REFERENCE_CLASS:
        raise ValueError("provider reference identity mismatch")
    table = reference.get("backend_bootstrap")
    if not isinstance(table, dict) or platform_key not in table:
        raise ValueError(f"backend bootstrap is not declared for {platform_key}")
    value = table[platform_key]
    if not isinstance(value, dict):
        raise ValueError("backend bootstrap contract must be an object")
    return value


def safe_member_path(name: str, archive_root: str) -> pathlib.PurePosixPath:
    raw = pathlib.PurePosixPath(name)
    if raw.is_absolute() or not raw.parts or any(part in {"", ".", ".."} for part in raw.parts):
        raise ValueError(f"backend archive member path is unsafe: {name}")
    if raw.parts[0] != archive_root:
        raise ValueError(f"backend archive member is outside expected root {archive_root}: {name}")
    return raw


def validate_symlink(member_path: pathlib.PurePosixPath, linkname: str, archive_root: str) -> str:
    target = pathlib.PurePosixPath(linkname)
    if target.is_absolute() or not target.parts:
        raise ValueError(f"backend archive symlink target is unsafe: {member_path} -> {linkname}")
    resolved = member_path.parent.joinpath(target)
    parts: list[str] = []
    for part in resolved.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"backend archive symlink escaped root: {member_path} -> {linkname}")
            parts.pop()
        else:
            parts.append(part)
    if not parts or parts[0] != archive_root:
        raise ValueError(f"backend archive symlink escaped root: {member_path} -> {linkname}")
    return linkname


def inspect_verified_archive(
    *,
    reference_path: pathlib.Path,
    platform_key: str,
    archive_path: pathlib.Path,
) -> dict:
    reference_path = reference_path.resolve()
    archive_path = archive_path.resolve()
    contract = backend_contract(reference_path, platform_key)
    expected_name = require_text(contract.get("asset"), "backend asset")
    expected_size = int(contract.get("expected_size_bytes", 0))
    expected_sha = require_text(contract.get("expected_sha256"), "backend sha256").lower()
    source_url = require_text(contract.get("url"), "backend URL")
    archive_root = require_text(contract.get("archive_root"), "backend archive_root")
    executable_rel = pathlib.PurePosixPath(require_text(contract.get("backend_executable"), "backend executable"))
    lib_dir_rel = pathlib.PurePosixPath(require_text(contract.get("lib_dir"), "backend lib_dir"))
    if expected_size <= 0 or len(expected_sha) != 64:
        raise ValueError("backend size/sha contract is invalid")
    if archive_path.name != expected_name:
        raise ValueError(f"backend archive filename mismatch: expected {expected_name}, got {archive_path.name}")
    if not archive_path.is_file():
        raise ValueError(f"backend archive is missing: {archive_path}")
    if archive_path.stat().st_size != expected_size:
        raise ValueError(f"backend archive size mismatch: expected {expected_size}, got {archive_path.stat().st_size}")
    actual_sha = sha256_file(archive_path)
    if actual_sha != expected_sha:
        raise ValueError(f"backend archive sha256 mismatch: expected {expected_sha}, got {actual_sha}")

    expected_executable = pathlib.PurePosixPath(archive_root) / executable_rel
    expected_lib_dir = pathlib.PurePosixPath(archive_root) / lib_dir_rel
    seen: set[pathlib.PurePosixPath] = set()
    regular: dict[pathlib.PurePosixPath, tarfile.TarInfo] = {}
    symlinks: dict[pathlib.PurePosixPath, str] = {}

    with tarfile.open(archive_path, "r:bz2") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("backend archive is empty")
        for member in members:
            member_path = safe_member_path(member.name, archive_root)
            if member_path in seen:
                raise ValueError(f"duplicate backend archive member: {member.name}")
            seen.add(member_path)
            if member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"backend archive contains unsupported member type: {member.name}")
            if member.issym():
                symlinks[member_path] = validate_symlink(member_path, member.linkname, archive_root)
            elif member.isfile():
                regular[member_path] = member
            elif not member.isdir():
                raise ValueError(f"backend archive contains unsupported member type: {member.name}")

        if expected_executable not in regular:
            raise ValueError(f"backend archive is missing executable: {expected_executable}")
        lib_members = sorted(
            path for path in regular
            if len(path.parts) > len(expected_lib_dir.parts)
            and path.parts[: len(expected_lib_dir.parts)] == expected_lib_dir.parts
        )
        if not lib_members:
            raise ValueError("backend archive has no regular files under lib_dir")

        def digest_member(path: pathlib.PurePosixPath) -> dict:
            member = regular[path]
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read backend archive member: {path}")
            with stream:
                size, digest = sha256_stream(stream)
            if size <= 0 or size != int(member.size):
                raise ValueError(f"backend archive member size is invalid: {path}")
            return {"path": path.as_posix(), "size_bytes": size, "sha256": digest, "mode": int(member.mode & 0o777)}

        executable = digest_member(expected_executable)
        libraries = [digest_member(path) for path in lib_members]

    lib_symlinks = [
        {"path": path.as_posix(), "target": target}
        for path, target in sorted(symlinks.items(), key=lambda item: item[0].as_posix())
        if len(path.parts) > len(expected_lib_dir.parts)
        and path.parts[: len(expected_lib_dir.parts)] == expected_lib_dir.parts
    ]
    return {
        "schema_version": 1,
        "evidence_class": RECEIPT_CLASS,
        "platform": platform_key,
        "reference_sha256": sha256_file(reference_path),
        "archive": {
            "name": expected_name,
            "size_bytes": expected_size,
            "sha256": actual_sha,
            "source_url": source_url,
        },
        "archive_root": archive_root,
        "backend_executable": executable,
        "lib_dir": expected_lib_dir.as_posix(),
        "libraries": libraries,
        "library_symlinks": lib_symlinks,
        "safe_archive_verified": True,
    }


def validate_extracted_bundle(*, receipt: dict, output_dir: pathlib.Path) -> None:
    output_dir = output_dir.resolve()
    executable_item = receipt["backend_executable"]
    executable = output_dir / pathlib.PurePosixPath(str(executable_item["path"]))
    if not executable.is_file():
        raise ValueError("extracted backend executable is missing")
    if executable.stat().st_size != int(executable_item["size_bytes"]):
        raise ValueError("extracted backend executable size mismatch")
    if sha256_file(executable) != str(executable_item["sha256"]):
        raise ValueError("extracted backend executable sha256 mismatch")

    lib_dir = output_dir / pathlib.PurePosixPath(str(receipt["lib_dir"]))
    if not lib_dir.is_dir():
        raise ValueError("extracted backend lib dir is missing")
    expected_regular = {str(item["path"]): item for item in receipt["libraries"]}
    actual_regular: dict[str, pathlib.Path] = {}
    actual_symlinks: dict[str, str] = {}
    for path in sorted(lib_dir.rglob("*")):
        rel = path.relative_to(output_dir).as_posix()
        if path.is_symlink():
            actual_symlinks[rel] = os.readlink(path)
        elif path.is_file():
            actual_regular[rel] = path
        elif not path.is_dir():
            raise ValueError(f"backend lib dir contains unsupported entry: {path}")
    if set(actual_regular) != set(expected_regular):
        raise ValueError("extracted backend library file set mismatch")
    for rel, path in actual_regular.items():
        expected = expected_regular[rel]
        if path.stat().st_size != int(expected["size_bytes"]):
            raise ValueError(f"extracted backend library size mismatch: {rel}")
        if sha256_file(path) != str(expected["sha256"]):
            raise ValueError(f"extracted backend library sha256 mismatch: {rel}")
    expected_symlinks = {
        str(item["path"]): str(item["target"])
        for item in receipt["library_symlinks"]
    }
    if actual_symlinks != expected_symlinks:
        raise ValueError("extracted backend library symlink set/targets mismatch")


def extract_verified_bundle(
    *,
    reference_path: pathlib.Path,
    platform_key: str,
    archive_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict:
    receipt = inspect_verified_archive(
        reference_path=reference_path,
        platform_key=platform_key,
        archive_path=archive_path,
    )
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("backend output-dir must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_root = str(receipt["archive_root"])

    symlinks: list[tuple[pathlib.Path, str]] = []
    with tarfile.open(archive_path.resolve(), "r:bz2") as archive:
        for member in archive.getmembers():
            member_path = safe_member_path(member.name, archive_root)
            destination = output_dir.joinpath(*member_path.parts)
            if member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"backend archive contains unsupported member type: {member.name}")
            if member.issym():
                symlinks.append((destination, validate_symlink(member_path, member.linkname, archive_root)))
                continue
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"backend archive contains unsupported member type: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read backend member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as sink:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    sink.write(chunk)
            os.chmod(destination, int(member.mode & 0o777))

    for destination, target in symlinks:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(target)

    validate_extracted_bundle(receipt=receipt, output_dir=output_dir)
    return receipt


def main() -> int:
    p = argparse.ArgumentParser(description="Verify and safely extract pinned sherpa-onnx backend bundle.")
    p.add_argument("--reference", required=True, type=pathlib.Path)
    p.add_argument("--platform", required=True)
    p.add_argument("--archive", required=True, type=pathlib.Path)
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--receipt", required=True, type=pathlib.Path)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if args.verify_only:
        if not args.receipt.is_file():
            raise ValueError(f"backend receipt is missing: {args.receipt}")
        receipt = load_object(args.receipt.resolve())
        if (
            int(receipt.get("schema_version", 0)) != 1
            or receipt.get("evidence_class") != RECEIPT_CLASS
        ):
            raise ValueError("backend receipt identity mismatch")
        inspected = inspect_verified_archive(
            reference_path=args.reference,
            platform_key=args.platform,
            archive_path=args.archive,
        )
        if receipt != inspected:
            raise ValueError("backend receipt does not match verified archive")
        validate_extracted_bundle(receipt=receipt, output_dir=args.output_dir)
    else:
        receipt = extract_verified_bundle(
            reference_path=args.reference,
            platform_key=args.platform,
            archive_path=args.archive,
            output_dir=args.output_dir,
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"speech-like sherpa backend: platform={receipt['platform']} "
        f"archive={receipt['archive']['sha256']} verify_only={args.verify_only}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, tarfile.TarError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
