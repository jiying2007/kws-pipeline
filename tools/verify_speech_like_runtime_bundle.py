#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tarfile

REFERENCE_CLASS = "speech-like-provider-reference-v1"
RECEIPT_CLASS = "speech-like-runtime-asset-bundle-v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        total += len(chunk)
        digest.update(chunk)
    return total, digest.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def reference_candidate(reference_path: pathlib.Path, candidate_name: str) -> tuple[dict, dict]:
    reference = load_object(reference_path)
    if (
        int(reference.get("schema_version", 0)) != 1
        or reference.get("evidence_class") != REFERENCE_CLASS
    ):
        raise ValueError("provider reference identity mismatch")
    rows = reference.get("reference_candidates")
    if not isinstance(rows, list):
        raise ValueError("provider reference candidates must be a list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("name", "")).strip() == candidate_name
    ]
    if len(matches) != 1:
        raise ValueError(f"reference candidate must match exactly once: {candidate_name}")
    row = matches[0]
    bundle = row.get("runtime_asset_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("reference candidate is missing runtime_asset_bundle")
    return row, bundle


def safe_member_path(name: str, archive_root: str) -> pathlib.PurePosixPath:
    raw = pathlib.PurePosixPath(name)
    if raw.is_absolute() or not raw.parts:
        raise ValueError(f"archive member path is unsafe: {name}")
    if any(part in {"", ".", ".."} for part in raw.parts):
        raise ValueError(f"archive member path is unsafe: {name}")
    if raw.parts[0] != archive_root:
        raise ValueError(
            f"archive member is outside expected root {archive_root}: {name}"
        )
    return raw


def inspect_verified_archive(
    *,
    reference_path: pathlib.Path,
    candidate_name: str,
    archive_path: pathlib.Path,
) -> dict:
    reference_path = reference_path.resolve()
    archive_path = archive_path.resolve()
    _, bundle = reference_candidate(reference_path, candidate_name)

    expected_name = require_text(bundle.get("asset"), "runtime asset name")
    expected_size = int(bundle.get("expected_size_bytes", 0))
    expected_sha = require_text(bundle.get("expected_sha256"), "runtime asset sha256").lower()
    source_url = require_text(bundle.get("url"), "runtime asset url")
    archive_root = require_text(bundle.get("archive_root"), "runtime archive root")
    required = bundle.get("required_files")
    if not isinstance(required, dict) or not required:
        raise ValueError("runtime required_files must be a non-empty object")
    minimum_roles = {"model", "tokens", "lexicon"}
    if not minimum_roles.issubset(set(required)):
        raise ValueError("runtime required_files must contain model/tokens/lexicon")
    for role, rel in required.items():
        require_text(role, "runtime required_files role")
        require_text(rel, f"required_files.{role}")
    if expected_size <= 0 or len(expected_sha) != 64:
        raise ValueError("runtime archive size/sha256 contract is invalid")
    if archive_path.name != expected_name:
        raise ValueError(
            f"runtime archive filename mismatch: expected {expected_name}, got {archive_path.name}"
        )
    if not archive_path.is_file():
        raise ValueError(f"runtime archive is missing: {archive_path}")
    actual_size = archive_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"runtime archive size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_sha = sha256_file(archive_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"runtime archive sha256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

    required_paths = {
        role: pathlib.PurePosixPath(archive_root) / require_text(rel, f"required_files.{role}")
        for role, rel in required.items()
    }
    seen_members: set[pathlib.PurePosixPath] = set()
    required_members: dict[str, tarfile.TarInfo] = {}

    with tarfile.open(archive_path, mode="r:bz2") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("runtime archive is empty")
        for member in members:
            member_path = safe_member_path(member.name, archive_root)
            if member_path in seen_members:
                raise ValueError(f"duplicate archive member: {member.name}")
            seen_members.add(member_path)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"runtime archive contains unsupported member type: {member.name}")
            if not member.isdir() and not member.isfile():
                raise ValueError(f"runtime archive contains unsupported member type: {member.name}")

        for role, required_path in required_paths.items():
            matches = [
                member
                for member in members
                if pathlib.PurePosixPath(member.name) == required_path and member.isfile()
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"runtime archive must contain exactly one {role}: {required_path}"
                )
            required_members[role] = matches[0]

        files: dict[str, dict] = {}
        for role in sorted(required_members):
            member = required_members[role]
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read runtime archive member: {member.name}")
            with stream:
                size_bytes, digest = sha256_stream(stream)
            if size_bytes <= 0 or size_bytes != int(member.size):
                raise ValueError(f"runtime archive member size is invalid: {member.name}")
            files[role] = {
                "path": pathlib.PurePosixPath(member.name).as_posix(),
                "size_bytes": size_bytes,
                "sha256": digest,
            }

    return {
        "schema_version": 1,
        "evidence_class": RECEIPT_CLASS,
        "candidate": candidate_name,
        "reference_sha256": sha256_file(reference_path),
        "archive": {
            "name": expected_name,
            "size_bytes": actual_size,
            "sha256": actual_sha,
            "source_url": source_url,
        },
        "archive_root": archive_root,
        "files": files,
        "safe_archive_verified": True,
    }


def extract_verified_bundle(
    *,
    reference_path: pathlib.Path,
    candidate_name: str,
    archive_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict:
    reference_path = reference_path.resolve()
    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    receipt = inspect_verified_archive(
        reference_path=reference_path,
        candidate_name=candidate_name,
        archive_path=archive_path,
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("runtime bundle output-dir must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_root = str(receipt["archive_root"])

    with tarfile.open(archive_path, mode="r:bz2") as archive:
        for member in archive.getmembers():
            member_path = safe_member_path(member.name, archive_root)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"runtime archive contains unsupported member type: {member.name}")
            if not member.isdir() and not member.isfile():
                raise ValueError(f"runtime archive contains unsupported member type: {member.name}")
            destination = output_dir.joinpath(*member_path.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read runtime archive member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as sink:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    sink.write(chunk)

    for role, item in receipt["files"].items():
        relative = pathlib.PurePosixPath(str(item["path"]))
        path = output_dir.joinpath(*relative.parts)
        if not path.is_file():
            raise ValueError(f"extracted runtime {role} is missing: {path}")
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"extracted runtime {role} size mismatch")
        if sha256_file(path) != str(item["sha256"]):
            raise ValueError(f"extracted runtime {role} sha256 mismatch")

    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and safely extract a pinned speech-like TTS runtime asset archive."
    )
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--archive", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--receipt", required=True, type=pathlib.Path)
    args = parser.parse_args()

    receipt = extract_verified_bundle(
        reference_path=args.reference,
        candidate_name=args.candidate.strip(),
        archive_path=args.archive,
        output_dir=args.output_dir,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"speech-like runtime bundle: candidate={receipt['candidate']} "
        f"archive={receipt['archive']['sha256']} root={receipt['archive_root']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, tarfile.TarError, TypeError, ValueError) as exc:
        # stderr, like every other verifier in tools/ and like the backend
        # twin of this file: a caller capturing stdout must not get a
        # failure message mixed into what it captured.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
