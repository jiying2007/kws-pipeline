#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import re
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = "product-development-preflight-job-handoff-v1"
METADATA_PATH = pathlib.PurePosixPath("build/product-preflight-handoff/manifest.json")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: str, label: str) -> str:
    if not HEX40_RE.fullmatch(value):
        raise ValueError(f"{label} must be canonical 40-hex")
    return value


def _repo_relative(path: pathlib.Path, repo_root: pathlib.Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"handoff path escapes repository root: {resolved}") from exc
    return relative.as_posix()


def _add_if_file(
    paths: dict[str, pathlib.Path],
    path: pathlib.Path,
    repo_root: pathlib.Path,
    *,
    required: bool = False,
) -> None:
    if path.is_file():
        paths[_repo_relative(path, repo_root)] = path.resolve()
    elif required:
        raise ValueError(f"required handoff file is missing: {path}")


def _collect_handoff_files(
    *,
    work_dir: pathlib.Path,
    config_path: pathlib.Path,
    manifest: dict,
    repo_root: pathlib.Path,
) -> dict[str, pathlib.Path]:
    files: dict[str, pathlib.Path] = {}
    _add_if_file(files, config_path, repo_root, required=True)
    _add_if_file(files, work_dir / "domain-loop-manifest.json", repo_root, required=True)
    _add_if_file(files, work_dir / "domain-loop-progress.jsonl", repo_root)
    _add_if_file(files, work_dir / "base-refinement-eligibility.json", repo_root)
    _add_if_file(files, work_dir / "acoustic-alignment.json", repo_root)
    _add_if_file(files, work_dir / "compact-base-diagnostics.json", repo_root)
    _add_if_file(files, work_dir / "preflight-experiment-receipt.json", repo_root)

    rounds: set[int] = set()
    for index, record in enumerate(manifest.get("records", [])):
        if not isinstance(record, dict):
            raise ValueError(f"domain manifest record {index} must be an object")
        round_index = int(record.get("round", -1))
        if round_index < 0:
            raise ValueError(f"domain manifest record {index} has invalid round")
        rounds.add(round_index)
        for field in ("model", "checkpoint", "provenance", "keywords", "pack"):
            raw = record.get(field)
            if isinstance(raw, str) and raw:
                _add_if_file(files, pathlib.Path(raw), repo_root, required=field == "checkpoint")
        for split in ("calibration", "test"):
            metrics = record.get(split)
            if not isinstance(metrics, dict):
                raise ValueError(f"domain manifest record {index} {split} metrics are missing")
            for field in ("false_positives_path", "false_rejects_path"):
                raw = metrics.get(field)
                if isinstance(raw, str) and raw:
                    _add_if_file(files, pathlib.Path(raw), repo_root, required=True)
            if split == "calibration":
                curve = metrics.get("calibration_operating_curve_path")
                if isinstance(curve, str) and curve:
                    _add_if_file(files, pathlib.Path(curve), repo_root, required=True)

    for round_index in sorted(rounds):
        dataset = work_dir / "datasets" / f"round-{round_index:02d}"
        _add_if_file(files, dataset / "domain-index.jsonl", repo_root, required=True)
        _add_if_file(files, dataset / "audit.json", repo_root)

    curriculum = work_dir / "curriculum"
    if curriculum.is_dir():
        for path in sorted(curriculum.glob("*.json")):
            _add_if_file(files, path, repo_root)

    cache = work_dir / "hard-negative-replay" / ".clean-command-tts-cache"
    if cache.is_dir():
        for path in sorted(cache.rglob("*")):
            if path.is_file():
                _add_if_file(files, path, repo_root)

    return dict(sorted(files.items()))


def _tar_add_bytes(
    archive: tarfile.TarFile,
    name: str,
    payload: bytes,
) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    archive.addfile(info, io.BytesIO(payload))


def pack_handoff(
    *,
    work_dir: pathlib.Path,
    config_path: pathlib.Path,
    effective_config_path: pathlib.Path,
    request_path: pathlib.Path,
    output_path: pathlib.Path,
    head_sha: str,
    base_sha: str,
    repo_root: pathlib.Path = ROOT,
) -> dict:
    head_sha = _require_sha(head_sha, "head SHA")
    base_sha = _require_sha(base_sha, "base SHA")
    repo_root = repo_root.resolve()
    work_dir = work_dir.resolve()
    manifest_path = work_dir / "domain-loop-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("base development manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("base development manifest must be an object")
    if manifest.get("qualification_deferred") is not True:
        raise ValueError("base handoff requires deferred qualification")
    if manifest.get("qualification_qualified") is not None:
        raise ValueError("base handoff unexpectedly contains qualification verdict")

    files = _collect_handoff_files(
        work_dir=work_dir,
        config_path=config_path.resolve(),
        manifest=manifest,
        repo_root=repo_root,
    )
    entries = [
        {
            "path": relative,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for relative, path in files.items()
    ]
    metadata = {
        "schema_version": 1,
        "policy": POLICY,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "workspace_root": str(repo_root),
        "request_path": _repo_relative(request_path.resolve(), repo_root),
        "request_sha256": sha256_file(request_path.resolve()),
        "effective_config_path": _repo_relative(effective_config_path.resolve(), repo_root),
        "effective_config_sha256": sha256_file(effective_config_path.resolve()),
        "preflight_config_path": _repo_relative(config_path.resolve(), repo_root),
        "preflight_config_sha256": sha256_file(config_path.resolve()),
        "development_manifest_path": _repo_relative(manifest_path, repo_root),
        "development_manifest_sha256": sha256_file(manifest_path),
        "files": entries,
    }
    payload = (
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w") as archive:
        _tar_add_bytes(archive, METADATA_PATH.as_posix(), payload)
        for relative, path in files.items():
            _tar_add_bytes(archive, relative, path.read_bytes())
    return metadata


def _safe_member_name(name: str) -> pathlib.PurePosixPath:
    value = pathlib.PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"unsafe handoff archive member: {name}")
    return value


def restore_handoff(
    *,
    archive_path: pathlib.Path,
    head_sha: str,
    base_sha: str,
    repo_root: pathlib.Path = ROOT,
) -> dict:
    head_sha = _require_sha(head_sha, "head SHA")
    base_sha = _require_sha(base_sha, "base SHA")
    repo_root = repo_root.resolve()

    with tarfile.open(archive_path.resolve(), "r") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("handoff archive contains duplicate paths")
        for member in members:
            _safe_member_name(member.name)
            if not member.isfile():
                raise ValueError(f"handoff archive member must be a regular file: {member.name}")
        try:
            metadata_member = archive.getmember(METADATA_PATH.as_posix())
        except KeyError as exc:
            raise ValueError("handoff archive metadata is missing") from exc
        stream = archive.extractfile(metadata_member)
        if stream is None:
            raise ValueError("handoff archive metadata cannot be read")
        metadata = json.loads(stream.read().decode("utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("handoff metadata must be an object")
        if metadata.get("schema_version") != 1 or metadata.get("policy") != POLICY:
            raise ValueError("unsupported preflight handoff contract")
        if metadata.get("head_sha") != head_sha or metadata.get("base_sha") != base_sha:
            raise ValueError("preflight handoff head/base SHA mismatch")
        if pathlib.Path(str(metadata.get("workspace_root", ""))).resolve() != repo_root:
            raise ValueError(
                "preflight handoff workspace changed; absolute evidence paths would be invalid"
            )

        expected = {
            str(row["path"]): row
            for row in metadata.get("files", [])
            if isinstance(row, dict) and isinstance(row.get("path"), str)
        }
        if len(expected) != len(metadata.get("files", [])):
            raise ValueError("handoff metadata file list is invalid")
        archive_payload_names = set(names) - {METADATA_PATH.as_posix()}
        if archive_payload_names != set(expected):
            raise ValueError("handoff archive file set does not match metadata")

        for member in members:
            if member.name == METADATA_PATH.as_posix():
                continue
            row = expected[member.name]
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read handoff member: {member.name}")
            payload = stream.read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != str(row.get("sha256")) or len(payload) != int(row.get("size", -1)):
                raise ValueError(f"handoff payload integrity mismatch: {member.name}")
            target = repo_root / pathlib.PurePosixPath(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

    metadata_target = repo_root / METADATA_PATH
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata_target.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    verify_restored_handoff(metadata=metadata, repo_root=repo_root)
    return metadata


def verify_restored_handoff(*, metadata: dict, repo_root: pathlib.Path = ROOT) -> None:
    repo_root = repo_root.resolve()
    config_path = repo_root / str(metadata["preflight_config_path"])
    manifest_path = repo_root / str(metadata["development_manifest_path"])
    if sha256_file(config_path) != str(metadata["preflight_config_sha256"]):
        raise ValueError("restored preflight config SHA mismatch")
    if sha256_file(manifest_path) != str(metadata["development_manifest_sha256"]):
        raise ValueError("restored development manifest SHA mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("restored development manifest must be an object")
    work_dir = manifest_path.parent

    for index, record in enumerate(manifest.get("records", [])):
        if not isinstance(record, dict):
            raise ValueError(f"restored record {index} must be an object")
        checkpoint = pathlib.Path(str(record.get("checkpoint", "")))
        if not checkpoint.is_file():
            raise ValueError(f"restored checkpoint is missing for record {index}")
        model = record.get("model")
        if isinstance(model, str) and model:
            model_path = pathlib.Path(model)
            if model_path.is_file() and record.get("model_sha256"):
                if sha256_file(model_path) != str(record["model_sha256"]):
                    raise ValueError(f"restored model SHA mismatch for record {index}")
        provenance = record.get("provenance")
        if isinstance(provenance, str) and provenance:
            provenance_path = pathlib.Path(provenance)
            if provenance_path.is_file() and record.get("provenance_sha256"):
                if sha256_file(provenance_path) != str(record["provenance_sha256"]):
                    raise ValueError(f"restored provenance SHA mismatch for record {index}")
        round_index = int(record["round"])
        if not (work_dir / "datasets" / f"round-{round_index:02d}" / "domain-index.jsonl").is_file():
            raise ValueError(f"restored domain index is missing for round {round_index}")
        for split in ("calibration", "test"):
            metrics = record.get(split)
            if not isinstance(metrics, dict):
                raise ValueError(f"restored {split} metrics are missing for record {index}")
            for field in ("false_positives_path", "false_rejects_path"):
                raw = metrics.get(field)
                if isinstance(raw, str) and raw and not pathlib.Path(raw).is_file():
                    raise ValueError(
                        f"restored {split} failure evidence is missing for record {index}: {field}"
                    )
            if split == "calibration":
                curve = metrics.get("calibration_operating_curve_path")
                if isinstance(curve, str) and curve:
                    curve_path = pathlib.Path(curve)
                    if not curve_path.is_file():
                        raise ValueError(
                            f"restored calibration operating curve is missing for record {index}"
                        )
                    expected_curve_sha = metrics.get("calibration_operating_curve_sha256")
                    if expected_curve_sha and sha256_file(curve_path) != str(expected_curve_sha):
                        raise ValueError(
                            f"restored calibration operating curve SHA mismatch for record {index}"
                        )


def verify_materialization(
    *,
    metadata_path: pathlib.Path,
    effective_config_path: pathlib.Path,
    request_path: pathlib.Path,
    head_sha: str,
    base_sha: str,
    repo_root: pathlib.Path = ROOT,
) -> dict:
    head_sha = _require_sha(head_sha, "head SHA")
    base_sha = _require_sha(base_sha, "base SHA")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("handoff metadata must be an object")
    if metadata.get("head_sha") != head_sha or metadata.get("base_sha") != base_sha:
        raise ValueError("handoff materialization head/base SHA mismatch")
    if sha256_file(request_path.resolve()) != str(metadata.get("request_sha256")):
        raise ValueError("training request changed across preflight jobs")
    if sha256_file(effective_config_path.resolve()) != str(metadata.get("effective_config_sha256")):
        raise ValueError("governed product materialization changed across preflight jobs")
    config_path = repo_root.resolve() / str(metadata["preflight_config_path"])
    if sha256_file(config_path) != str(metadata.get("preflight_config_sha256")):
        raise ValueError("preflight config changed across preflight jobs")
    verify_restored_handoff(metadata=metadata, repo_root=repo_root)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack")
    pack.add_argument("--work-dir", required=True, type=pathlib.Path)
    pack.add_argument("--config", required=True, type=pathlib.Path)
    pack.add_argument("--effective-config", required=True, type=pathlib.Path)
    pack.add_argument("--request", required=True, type=pathlib.Path)
    pack.add_argument("--output", required=True, type=pathlib.Path)
    pack.add_argument("--head-sha", required=True)
    pack.add_argument("--base-sha", required=True)

    restore = sub.add_parser("restore")
    restore.add_argument("--archive", required=True, type=pathlib.Path)
    restore.add_argument("--head-sha", required=True)
    restore.add_argument("--base-sha", required=True)

    verify = sub.add_parser("verify-materialization")
    verify.add_argument("--metadata", required=True, type=pathlib.Path)
    verify.add_argument("--effective-config", required=True, type=pathlib.Path)
    verify.add_argument("--request", required=True, type=pathlib.Path)
    verify.add_argument("--head-sha", required=True)
    verify.add_argument("--base-sha", required=True)

    args = parser.parse_args()
    if args.command == "pack":
        result = pack_handoff(
            work_dir=args.work_dir,
            config_path=args.config,
            effective_config_path=args.effective_config,
            request_path=args.request,
            output_path=args.output,
            head_sha=args.head_sha,
            base_sha=args.base_sha,
        )
    elif args.command == "restore":
        result = restore_handoff(
            archive_path=args.archive,
            head_sha=args.head_sha,
            base_sha=args.base_sha,
        )
    else:
        result = verify_materialization(
            metadata_path=args.metadata,
            effective_config_path=args.effective_config,
            request_path=args.request,
            head_sha=args.head_sha,
            base_sha=args.base_sha,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, tarfile.TarError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
