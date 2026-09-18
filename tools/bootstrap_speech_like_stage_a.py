#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
REFERENCE_CLASS = "speech-like-provider-reference-v1"
SUMMARY_CLASS = "speech-like-stage-a-bootstrap-v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def require_sha256(value: object, label: str) -> str:
    result = require_text(value, label).lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{label} must be lowercase sha256")
    return result


def require_executable(raw: pathlib.Path | None, fallback: str, label: str) -> pathlib.Path:
    if raw is None:
        resolved = shutil.which(fallback)
        if resolved is None:
            raise ValueError(f"{label} not found in PATH; pass an explicit path")
        path = pathlib.Path(resolved).resolve()
    else:
        path = raw.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{label} is missing or not executable: {path}")
    return path


def auto_backend_platform_key() -> str | None:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    return None


def select_candidate(reference_path: pathlib.Path, requested: str | None) -> tuple[dict, dict]:
    reference = load_object(reference_path)
    if (
        int(reference.get("schema_version", 0)) != 1
        or reference.get("evidence_class") != REFERENCE_CLASS
    ):
        raise ValueError("provider reference identity mismatch")
    candidate_name = (
        requested.strip()
        if requested is not None and requested.strip()
        else str(reference.get("preferred_stage_a_candidate", "")).strip()
    )
    if not candidate_name:
        raise ValueError("provider reference has no preferred Stage A candidate")
    rows = reference.get("reference_candidates")
    if not isinstance(rows, list):
        raise ValueError("provider reference candidates must be a list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("name", "")).strip() == candidate_name
    ]
    if len(matches) != 1:
        raise ValueError(f"provider candidate must match exactly once: {candidate_name}")
    row = matches[0]
    if row.get("suitable_for_24_voice_baseline") is not True:
        raise ValueError("selected provider candidate is not Stage-A approved")
    if not str(row.get("license_status", "")).startswith("verified-"):
        raise ValueError("selected provider candidate does not have verified license evidence")
    if not isinstance(row.get("runtime_asset_bundle"), dict):
        raise ValueError("selected provider candidate has no runtime asset bundle")
    if not isinstance(row.get("license_evidence"), dict):
        raise ValueError("selected provider candidate has no license evidence contract")
    return reference, row


def validate_source_url(url: str, allow_file_urls: bool) -> str:
    parsed = urllib.parse.urlparse(url)
    allowed = {"https"}
    if allow_file_urls:
        allowed.add("file")
    if parsed.scheme not in allowed:
        raise ValueError(f"unsupported download URL scheme {parsed.scheme!r}: {url}")
    if parsed.scheme == "https" and not parsed.netloc:
        raise ValueError(f"HTTPS URL has no host: {url}")
    return url


def validate_cached_file(
    path: pathlib.Path,
    *,
    expected_sha256: str,
    expected_size: int | None,
    label: str,
) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(
            f"{label} size mismatch: expected {expected_size}, got {path.stat().st_size}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError(
            f"{label} sha256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )


def download_verified(
    *,
    url: str,
    target: pathlib.Path,
    expected_sha256: str,
    expected_size: int | None,
    label: str,
    allow_file_urls: bool,
    refresh: bool,
) -> str:
    url = validate_source_url(url, allow_file_urls)
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            validate_cached_file(
                target,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                label=f"cached {label}",
            )
            if not refresh:
                return "cache-hit"
        except ValueError:
            if not refresh:
                raise
        target.unlink(missing_ok=True)

    part = target.with_name(target.name + ".part")
    part.unlink(missing_ok=True)
    digest = hashlib.sha256()
    total = 0
    try:
        request: str | urllib.request.Request
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "https":
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "kws-pipeline-speech-like-bootstrap/1"},
            )
        else:
            request = url
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as sink:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if expected_size is not None and total > expected_size:
                    raise ValueError(
                        f"{label} download exceeded expected size {expected_size}"
                    )
                digest.update(chunk)
                sink.write(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if expected_size is not None and total != expected_size:
            raise ValueError(
                f"{label} download size mismatch: expected {expected_size}, got {total}"
            )
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha256:
            raise ValueError(
                f"{label} download sha256 mismatch: expected {expected_sha256}, got {actual_sha}"
            )
        part.replace(target)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    validate_cached_file(
        target,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        label=label,
    )
    return "downloaded"


def run_checked(command: list[str], log: pathlib.Path) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}; see {log}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap pinned AISHELL3 Stage A speech-like assets and optionally run the "
            "complete governed 384-recording corpus generation chain."
        )
    )
    parser.add_argument(
        "--provider-reference",
        type=pathlib.Path,
        default=ROOT / "configs" / "training" / "speech-like-provider-reference-v1.json",
    )
    parser.add_argument("--reference-candidate")
    parser.add_argument("--backend-executable", type=pathlib.Path)
    parser.add_argument("--resampler-executable", type=pathlib.Path)
    parser.add_argument("--cache-dir", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--allow-file-urls", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    reference_path = args.provider_reference.resolve()
    if not reference_path.is_file():
        raise ValueError(f"provider reference is missing: {reference_path}")
    reference, candidate = select_candidate(reference_path, args.reference_candidate)
    candidate_name = str(candidate["name"])
    profile = str(candidate.get("provider_profile", ""))
    runtime = candidate["runtime_asset_bundle"]
    license_evidence = candidate["license_evidence"]

    archive_name = require_text(runtime.get("asset"), "runtime asset name")
    archive_url = require_text(runtime.get("url"), "runtime asset URL")
    archive_sha = require_sha256(runtime.get("expected_sha256"), "runtime asset sha256")
    archive_size = int(runtime.get("expected_size_bytes", 0))
    if archive_size <= 0:
        raise ValueError("runtime asset expected_size_bytes must be positive")

    license_url = require_text(license_evidence.get("url"), "license evidence URL")
    license_sha = require_sha256(
        license_evidence.get("expected_sha256"), "license evidence sha256"
    )

    cache = args.cache_dir.resolve()
    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError("bootstrap work-dir must be empty")
    work.mkdir(parents=True, exist_ok=True)

    archive_path = cache / archive_name
    license_suffix = pathlib.Path(
        urllib.parse.urlparse(license_url).path
    ).suffix or ".txt"
    license_path = cache / f"license-evidence-{license_sha[:12]}{license_suffix}"

    archive_state = download_verified(
        url=archive_url,
        target=archive_path,
        expected_sha256=archive_sha,
        expected_size=archive_size,
        label="AISHELL3 runtime archive",
        allow_file_urls=args.allow_file_urls,
        refresh=args.refresh,
    )
    license_state = download_verified(
        url=license_url,
        target=license_path,
        expected_sha256=license_sha,
        expected_size=None,
        label="AISHELL3 license evidence",
        allow_file_urls=args.allow_file_urls,
        refresh=args.refresh,
    )

    backend_platform = None
    backend_archive = None
    backend_receipt = None
    backend_root = None
    backend_lib_dir = None
    backend_archive_state = None
    backend_source_url = None

    if args.backend_executable is not None:
        backend = require_executable(
            args.backend_executable,
            "sherpa-onnx-offline-tts",
            "sherpa-onnx-offline-tts",
        )
        backend_mode = "explicit"
    else:
        backend_platform = auto_backend_platform_key()
        if backend_platform is None:
            backend = require_executable(
                None,
                "sherpa-onnx-offline-tts",
                "sherpa-onnx-offline-tts",
            )
            backend_mode = "path"
        else:
            table = reference.get("backend_bootstrap")
            if not isinstance(table, dict) or backend_platform not in table:
                raise ValueError(
                    f"provider reference has no backend bootstrap for {backend_platform}"
                )
            backend_contract = table[backend_platform]
            if not isinstance(backend_contract, dict):
                raise ValueError("backend bootstrap contract must be an object")
            backend_name = require_text(
                backend_contract.get("asset"), "backend asset name"
            )
            backend_source_url = require_text(
                backend_contract.get("url"), "backend asset URL"
            )
            backend_sha = require_sha256(
                backend_contract.get("expected_sha256"), "backend asset sha256"
            )
            backend_size = int(backend_contract.get("expected_size_bytes", 0))
            if backend_size <= 0:
                raise ValueError("backend expected_size_bytes must be positive")
            backend_archive = cache / backend_name
            backend_archive_state = download_verified(
                url=backend_source_url,
                target=backend_archive,
                expected_sha256=backend_sha,
                expected_size=backend_size,
                label="sherpa backend archive",
                allow_file_urls=args.allow_file_urls,
                refresh=args.refresh,
            )
            backend_root = cache / (
                "backend-" + backend_platform + "-" + backend_sha[:12]
            )
            backend_receipt = cache / (
                "backend-receipt-" + backend_platform + "-" + backend_sha[:12] + ".json"
            )
            if args.refresh and backend_root.exists():
                shutil.rmtree(backend_root)
            if args.refresh:
                backend_receipt.unlink(missing_ok=True)
            if not backend_receipt.is_file():
                if backend_root.exists() and any(backend_root.iterdir()):
                    raise ValueError(
                        "backend extraction cache exists without a verified receipt; "
                        "use --refresh to rebuild it"
                    )
                run_checked(
                    [
                        sys.executable,
                        str(TOOLS / "verify_speech_like_backend_bundle.py"),
                        "--reference",
                        str(reference_path),
                        "--platform",
                        backend_platform,
                        "--archive",
                        str(backend_archive),
                        "--output-dir",
                        str(backend_root),
                        "--receipt",
                        str(backend_receipt),
                    ],
                    work / "verify-backend-bundle.log",
                )
            run_checked(
                [
                    sys.executable,
                    str(TOOLS / "verify_speech_like_backend_bundle.py"),
                    "--reference",
                    str(reference_path),
                    "--platform",
                    backend_platform,
                    "--archive",
                    str(backend_archive),
                    "--output-dir",
                    str(backend_root),
                    "--receipt",
                    str(backend_receipt),
                    "--verify-only",
                ],
                work / "verify-backend-cache.log",
            )
            receipt_value = load_object(backend_receipt)
            expected_root = pathlib.PurePosixPath(
                str(receipt_value["backend_executable"]["path"])
            )
            backend = require_executable(
                backend_root.joinpath(*expected_root.parts),
                "sherpa-onnx-offline-tts",
                "sherpa-onnx-offline-tts",
            )
            lib_rel = pathlib.PurePosixPath(str(receipt_value["lib_dir"]))
            backend_lib_dir = backend_root.joinpath(*lib_rel.parts).resolve()
            if not backend_lib_dir.is_dir():
                raise ValueError(f"verified backend lib dir is missing: {backend_lib_dir}")
            backend_mode = "bootstrapped"

    resampler: pathlib.Path | None = None
    if profile == "sherpa-vits-resampled-to-16k-v1":
        if args.resampler_executable is not None:
            resampler = require_executable(
                args.resampler_executable, "resampler executable", "resampler executable"
            )
    elif profile != "sherpa-vits-native-16k-v1":
        raise ValueError(f"unsupported provider profile: {profile}")

    summary = {
        "schema_version": 1,
        "evidence_class": SUMMARY_CLASS,
        "provider_candidate": candidate_name,
        "provider_reference_sha256": sha256_file(reference_path),
        "profile": profile,
        "archive": {
            "path": str(archive_path),
            "sha256": archive_sha,
            "size_bytes": archive_size,
            "state": archive_state,
            "source_url": archive_url,
        },
        "license_evidence": {
            "path": str(license_path),
            "sha256": license_sha,
            "state": license_state,
            "source_url": license_url,
        },
        "backend": {
            "mode": backend_mode,
            "path": str(backend),
            "sha256": sha256_file(backend),
            "platform": backend_platform,
            "archive_path": str(backend_archive) if backend_archive is not None else None,
            "archive_state": backend_archive_state,
            "archive_source_url": backend_source_url,
            "receipt_path": str(backend_receipt) if backend_receipt is not None else None,
            "receipt_sha256": (
                sha256_file(backend_receipt)
                if backend_receipt is not None and backend_receipt.is_file()
                else None
            ),
            "bundle_root": str(backend_root) if backend_root is not None else None,
            "lib_dir": str(backend_lib_dir) if backend_lib_dir is not None else None,
        },
        "resampler": (
            {
                "mode": "external-executable",
                "path": str(resampler),
                "sha256": sha256_file(resampler),
            }
            if resampler is not None
            else {
                "mode": "builtin-lanczos-2x-v1",
                "path": None,
                "sha256": None,
            }
        ),
        "status": "assets-ready" if args.assets_only else "running-corpus-generation",
    }

    summary_path = work / "bootstrap-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.assets_only:
        print(
            f"speech-like Stage A bootstrap: assets-ready candidate={candidate_name} "
            f"archive={archive_sha}"
        )
        return 0

    corpus_work = work / "corpus"
    command = [
        sys.executable,
        str(TOOLS / "run_speech_like_corpus_generation.py"),
        "--provider-reference",
        str(reference_path),
        "--reference-candidate",
        candidate_name,
        "--runtime-archive",
        str(archive_path),
        "--license-evidence",
        str(license_path),
        "--backend-executable",
        str(backend),
        "--work-dir",
        str(corpus_work),
    ]
    if backend_platform is not None:
        command.extend(
            [
                "--backend-platform",
                backend_platform,
                "--backend-bundle-archive",
                str(backend_archive),
                "--backend-bundle-receipt",
                str(backend_receipt),
                "--backend-bundle-root",
                str(backend_root),
                "--backend-lib-dir",
                str(backend_lib_dir),
            ]
        )
    if resampler is not None:
        command.extend(["--resampler-executable", str(resampler)])
    run_checked(command, work / "corpus-generation.log")

    corpus_manifest = corpus_work / "stage-a-base-bundle.json"
    if not corpus_manifest.is_file():
        raise ValueError("corpus generator did not produce stage-a-base-bundle.json")
    corpus = load_object(corpus_manifest)
    if int(corpus.get("recordings", 0)) != 384 or int(corpus.get("voice_slots", 0)) != 24:
        raise ValueError("generated corpus does not match fixed Stage A shape")
    if corpus.get("tone_backend_used") is not False:
        raise ValueError("generated corpus unexpectedly used tone backend")
    summary["status"] = "complete"
    summary["corpus"] = {
        "work_dir": str(corpus_work),
        "bundle_manifest": str(corpus_manifest),
        "bundle_manifest_sha256": sha256_file(corpus_manifest),
        "external_base_bundle_sha256": str(corpus["external_base_bundle_sha256"]),
        "recordings": int(corpus["recordings"]),
        "voice_slots": int(corpus["voice_slots"]),
        "portable_bundle_manifest": str(
            corpus_work / str(corpus["portable_bundle_manifest"])
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"speech-like Stage A bootstrap: complete candidate={candidate_name} "
        f"bundle={summary['corpus']['external_base_bundle_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        urllib.error.URLError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
