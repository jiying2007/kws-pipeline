#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import platform
import re
import subprocess
import time

from runtime_soak_contract import (
    TARGET_EVIDENCE_SCHEMA_VERSION, require_cpu_contract, validate_runtime_soak,
)
SOURCE_SHA_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_existing(paths: list[pathlib.Path]) -> str | None:
    for path in paths:
        value = read_text(path)
        if value:
            return value
    return None


def cpu_model() -> str:
    info = read_text(pathlib.Path("/proc/cpuinfo")) or ""
    for key in ("model name", "Processor", "Hardware"):
        for line in info.splitlines():
            if line.lower().startswith(key.lower() + ":"):
                return line.split(":", 1)[1].strip()
    return platform.machine()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def load_runtime_soak(path: pathlib.Path) -> dict:
    return validate_runtime_soak(load_json(path))


def load_jsonl(path: pathlib.Path, label: str) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA256 hex")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--board-revision", required=True)
    parser.add_argument("--soc")
    parser.add_argument("--toolchain", required=True)
    parser.add_argument("--compiler-flags", required=True)
    parser.add_argument("--audio-frontend", required=True)
    parser.add_argument("--audio-frontend-sha256")
    parser.add_argument("--audio-frontend-identity-sha256")
    parser.add_argument("--resource-budget", type=pathlib.Path)
    parser.add_argument("--runtime-soak", required=True, type=pathlib.Path)
    parser.add_argument("--stack-high-water-bytes", type=float, required=True)
    parser.add_argument("--average-power-mw", type=float, required=True)
    parser.add_argument("--raw-evidence", action="append", type=pathlib.Path, default=[])
    parser.add_argument("--power-raw", required=True, type=pathlib.Path)
    parser.add_argument("--evidence-raw", required=True, type=pathlib.Path)
    parser.add_argument("--attestation-verification", required=True, type=pathlib.Path)
    parser.add_argument("--board-runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keyword-pack", required=True, type=pathlib.Path)
    parser.add_argument("--board-audio", required=True, type=pathlib.Path)
    parser.add_argument("--sku", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--builder-id", required=True)
    parser.add_argument("--dut-id", required=True)
    parser.add_argument("--collector-id", required=True)
    parser.add_argument("--instrument-id", required=True)
    parser.add_argument("--calibration-id", required=True)
    args = parser.parse_args()

    source_sha = args.source_sha.strip().lower()
    if SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError("--source-sha must be lowercase 40- or 64-character hex")
    audio_frontend_sha256 = (
        require_sha256(args.audio_frontend_sha256, "--audio-frontend-sha256")
        if args.audio_frontend_sha256 is not None
        else None
    )
    audio_frontend_identity_sha256 = (
        require_sha256(
            args.audio_frontend_identity_sha256,
            "--audio-frontend-identity-sha256",
        )
        if args.audio_frontend_identity_sha256 is not None
        else None
    )
    resource_budget_sha256 = None
    if args.resource_budget is not None:
        budget = load_json(args.resource_budget)
        if budget.get("schema_version") != 2:
            raise ValueError("resource budget schema_version must be 2")
        require_cpu_contract(budget, "resource budget")
        resource_budget_sha256 = sha256_file(args.resource_budget.resolve(strict=True))
    sku = require_text(args.sku, "--sku")
    builder_id = require_text(args.builder_id, "--builder-id")
    dut_id = require_text(args.dut_id, "--dut-id")
    collector_id = require_text(args.collector_id, "--collector-id")
    if builder_id == dut_id:
        raise ValueError("builder-id and dut-id must be distinct")
    if args.stack_high_water_bytes < 0.0 or args.average_power_mw < 0.0:
        raise ValueError("stack/power measurements must be non-negative")

    runtime_soak_path = args.runtime_soak.resolve(strict=True)
    runtime_soak_raw = runtime_soak_path.read_text(encoding="utf-8")
    runtime = load_runtime_soak(runtime_soak_path)
    power_path = args.power_raw.resolve(strict=True)
    raw_paths = [
        runtime_soak_path,
        *[path.resolve(strict=True) for path in args.raw_evidence],
        power_path,
    ]
    names = [path.name for path in raw_paths]
    if len(names) != len(set(names)):
        raise ValueError("raw evidence file names must be unique")
    raw_artifacts = [
        {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in raw_paths
    ]

    evidence_raw_path = args.evidence_raw.resolve(strict=True)
    rows = load_jsonl(evidence_raw_path, "evidence-raw")
    canonical: set[tuple[str, str, int]] = set()
    for index, row in enumerate(rows):
        name = require_text(row.get("name"), f"evidence-raw[{index}].name")
        digest = row.get("sha256")
        size = row.get("bytes")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"evidence-raw[{index}].sha256 must be lowercase SHA256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"evidence-raw[{index}].bytes must be a positive integer")
        canonical.add((name, digest, size))
    actual = {(item["name"], item["sha256"], item["bytes"]) for item in raw_artifacts}
    if len(rows) != len(canonical) or canonical != actual:
        raise ValueError("canonical evidence-raw manifest does not match selected raw artifacts")

    collector = pathlib.Path(__file__).resolve()
    collector_sha256 = sha256_file(collector)
    board_runner_sha256 = sha256_file(args.board_runner.resolve(strict=True))
    model_sha256 = sha256_file(args.model.resolve(strict=True))
    keyword_pack_sha256 = sha256_file(args.keyword_pack.resolve(strict=True))
    board_audio_sha256 = sha256_file(args.board_audio.resolve(strict=True))
    evidence_raw_sha256 = sha256_file(evidence_raw_path)
    attestation_path = args.attestation_verification.resolve(strict=True)
    attestation = load_json(attestation_path)
    if attestation.get("schema_version") != 1:
        raise ValueError("attestation verification schema_version must be 1")
    if attestation.get("verified") is not True:
        raise ValueError("attestation verification must report verified=true")
    if attestation.get("subject_kind") != "kws-target-evidence":
        raise ValueError("attestation subject_kind must be kws-target-evidence")
    for field in ("issuer", "trust_policy", "verified_at_utc"):
        require_text(attestation.get(field), f"attestation.{field}")
    if not require_text(attestation.get("verified_at_utc"), "attestation.verified_at_utc").endswith("Z"):
        raise ValueError("attestation verified_at_utc must be UTC")
    expected_attestation = {
        "subject_sha256": evidence_raw_sha256,
        "collector_sha256": collector_sha256,
        "board_runner_sha256": board_runner_sha256,
        "model_sha256": model_sha256,
        "keyword_pack_sha256": keyword_pack_sha256,
    }
    if audio_frontend_identity_sha256 is not None:
        expected_attestation["audio_frontend_identity_sha256"] = (
            audio_frontend_identity_sha256
        )
    if resource_budget_sha256 is not None:
        expected_attestation["resource_budget_sha256"] = resource_budget_sha256
    for key, expected in expected_attestation.items():
        if attestation.get(key) != expected:
            raise ValueError(f"attestation {key} does not match selected artifact")

    governor = first_existing(
        list(pathlib.Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
    ) or "unknown"
    evidence = {
        "schema_version": TARGET_EVIDENCE_SCHEMA_VERSION,
        "evidence_class": "product-board",
        "sku": sku,
        "source_sha": source_sha,
        "collected_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder_id": builder_id,
        "dut_id": dut_id,
        "collector_id": collector_id,
        "collector": collector.name,
        "collector_sha256": collector_sha256,
        "collector_source_sha": git_sha(),
        "collected_unix_s": time.time(),
        "raw_evidence_sha256": evidence_raw_sha256,
        "attestation_verification_sha256": sha256_file(attestation_path),
        "board_runner_sha256": board_runner_sha256,
        "model_sha256": model_sha256,
        "keyword_pack_sha256": keyword_pack_sha256,
        "board_audio_sha256": board_audio_sha256,
        "target": args.target,
        "board_revision": args.board_revision,
        "soc": args.soc or cpu_model(),
        "toolchain": args.toolchain,
        "compiler_flags": args.compiler_flags,
        "governor": governor,
        "audio_frontend": args.audio_frontend,
        "audio_frontend_sha256": audio_frontend_sha256,
        "audio_frontend_identity_sha256": audio_frontend_identity_sha256,
        "resource_budget_sha256": resource_budget_sha256,
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_online": read_text(pathlib.Path("/sys/devices/system/cpu/online")) or "unknown",
        "uptime_s": float((read_text(pathlib.Path("/proc/uptime")) or "0").split()[0]),
        **runtime,
        "stack_high_water_bytes": args.stack_high_water_bytes,
        "average_power_mw": args.average_power_mw,
        "runtime_soak_name": runtime_soak_path.name,
        "runtime_soak_sha256": sha256_file(runtime_soak_path),
        "runtime_soak_raw": runtime_soak_raw,
        "power_raw_name": power_path.name,
        "power_raw_sha256": sha256_file(power_path),
        "raw_evidence": raw_artifacts,
        "instrument_id": require_text(args.instrument_id, "--instrument-id"),
        "calibration_id": require_text(args.calibration_id, "--calibration-id"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote target evidence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
