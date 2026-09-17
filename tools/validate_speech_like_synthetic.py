#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import wave

POLICY = "speech-like-synthetic-v1"
EVIDENCE_CLASS = "speech-like-synthetic-recording-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("speech-like policy must be schema_version 1")
    if value.get("policy") != POLICY:
        raise ValueError("speech-like policy identity mismatch")
    groups = value.get("groups")
    if not isinstance(groups, list) or not groups or len(set(map(str, groups))) != len(groups):
        raise ValueError("policy groups must be unique and non-empty")
    allowed = value.get("allowed_provider_kinds")
    forbidden = value.get("forbidden_provider_kinds")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("allowed_provider_kinds must be non-empty")
    if not isinstance(forbidden, list) or "tone" not in {str(item) for item in forbidden}:
        raise ValueError("tone must remain forbidden for speech-like v1")
    required_isolation = value.get("required_cross_group_isolation_fields")
    if required_isolation != ["voice_id", "source_id"]:
        raise ValueError("v1 requires cross-group voice_id/source_id isolation")
    return value


def parse_manifest_arg(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise ValueError("--manifest must use GROUP=PATH")
    group, raw_path = value.split("=", 1)
    group = group.strip()
    path = pathlib.Path(raw_path).resolve()
    if not group or not path.is_file():
        raise ValueError(f"invalid manifest binding: {value}")
    return group, path


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    return rows


def inspect_wav(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise ValueError(f"audio file missing: {path}")
    with wave.open(str(path), "rb") as reader:
        channels = int(reader.getnchannels())
        sample_width = int(reader.getsampwidth())
        sample_rate = int(reader.getframerate())
        frames = int(reader.getnframes())
        compression = reader.getcomptype()
        pcm = reader.readframes(frames)
    if channels != 1 or sample_width != 2 or sample_rate != 16000 or compression != "NONE":
        raise ValueError(f"{path}: speech-like v1 requires mono PCM16 16-kHz WAV")
    if frames <= 0:
        raise ValueError(f"{path}: WAV must contain frames")
    return {
        "frames": frames,
        "duration_s": frames / 16000.0,
        "pcm_sha256": sha256_bytes(pcm),
        "file_sha256": sha256_file(path),
    }


def require_text(value: object, label: str) -> str:
    result = str(value) if isinstance(value, str) else ""
    if not result.strip():
        raise ValueError(f"{label} must be non-empty text")
    return result.strip()


def validate_row(*, policy: dict, manifest: pathlib.Path, group: str, index: int, row: dict) -> dict:
    label = f"{manifest}:{index + 1}"
    if int(row.get("schema_version", 0)) != 1 or row.get("evidence_class") != EVIDENCE_CLASS:
        raise ValueError(f"{label}: recording identity mismatch")
    if row.get("synthetic") is not True:
        raise ValueError(f"{label}: v1 requires synthetic=true")
    provider_kind = require_text(row.get("provider_kind"), f"{label}.provider_kind")
    allowed = {str(item) for item in policy["allowed_provider_kinds"]}
    forbidden = {str(item) for item in policy["forbidden_provider_kinds"]}
    if provider_kind in forbidden or provider_kind not in allowed:
        raise ValueError(f"{label}: provider_kind {provider_kind!r} is not allowed")
    provider_name = require_text(row.get("provider_name"), f"{label}.provider_name")
    provider_version = require_text(row.get("provider_version"), f"{label}.provider_version")
    license_id = require_text(row.get("license_id"), f"{label}.license_id")
    voice_id = require_text(row.get("voice_id"), f"{label}.voice_id")
    source_id = require_text(row.get("source_id"), f"{label}.source_id")
    locale = require_text(row.get("locale"), f"{label}.locale")
    if locale.lower() not in {"zh-cn", "zh_cn", "cmn-hans-cn"}:
        raise ValueError(f"{label}: v1 corpus is scoped to Mandarin zh-CN")
    text = require_text(row.get("text"), f"{label}.text")
    tokens = row.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(not isinstance(item, str) or not item for item in tokens):
        raise ValueError(f"{label}: tokens must be a non-empty string list")
    generation_sha = require_text(row.get("generation_config_sha256"), f"{label}.generation_config_sha256")
    if SHA256_RE.fullmatch(generation_sha) is None:
        raise ValueError(f"{label}: generation_config_sha256 is invalid")
    audio_raw = require_text(row.get("audio"), f"{label}.audio")
    audio = pathlib.Path(audio_raw)
    audio = audio.resolve() if audio.is_absolute() else (manifest.parent / audio).resolve()
    inspected = inspect_wav(audio)
    expected_file = require_text(row.get("file_sha256"), f"{label}.file_sha256")
    expected_pcm = require_text(row.get("pcm_sha256"), f"{label}.pcm_sha256")
    if expected_file != inspected["file_sha256"] or expected_pcm != inspected["pcm_sha256"]:
        raise ValueError(f"{label}: retained WAV/PCM hashes do not match manifest")
    return {
        "group": group,
        "audio": audio.as_posix(),
        "provider_kind": provider_kind,
        "provider_name": provider_name,
        "provider_version": provider_version,
        "license_id": license_id,
        "voice_id": voice_id,
        "source_id": source_id,
        "locale": locale,
        "text": text,
        "tokens": list(tokens),
        "generation_config_sha256": generation_sha,
        **inspected,
    }


def audit(policy: dict, bindings: list[tuple[str, pathlib.Path]]) -> dict:
    allowed_groups = {str(item) for item in policy["groups"]}
    groups = [group for group, _ in bindings]
    if len(groups) != len(set(groups)):
        raise ValueError("manifest groups must be unique")
    if any(group not in allowed_groups for group in groups):
        raise ValueError("manifest group is not declared by policy")
    records: list[dict] = []
    for group, manifest in bindings:
        for index, row in enumerate(load_jsonl(manifest)):
            records.append(validate_row(policy=policy, manifest=manifest, group=group, index=index, row=row))

    isolation_fields = list(policy["required_cross_group_isolation_fields"])
    overlaps: dict[str, list[str]] = {}
    for field in isolation_fields:
        owners: dict[str, set[str]] = {}
        for record in records:
            owners.setdefault(str(record[field]), set()).add(str(record["group"]))
        bad = sorted(value for value, owner_groups in owners.items() if len(owner_groups) > 1)
        overlaps[field] = bad
        if bad:
            raise ValueError(f"cross-group {field} overlap: {', '.join(bad)}")

    grouped: dict[str, dict] = {}
    for group, _ in bindings:
        subset = [row for row in records if row["group"] == group]
        grouped[group] = {
            "recordings": len(subset),
            "audio_hours": sum(float(row["duration_s"]) for row in subset) / 3600.0,
            "voices": sorted({str(row["voice_id"]) for row in subset}),
            "sources": sorted({str(row["source_id"]) for row in subset}),
            "providers": sorted({str(row["provider_name"]) for row in subset}),
            "licenses": sorted({str(row["license_id"]) for row in subset}),
        }
    return {
        "schema_version": 1,
        "policy": POLICY,
        "evidence_class": "speech-like-synthetic-audit-v1",
        "tone_backend_allowed": False,
        "cross_group_isolation": overlaps,
        "groups": grouped,
        "recordings": len(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate speech-like synthetic corpus identity and isolation.")
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, action="append")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    policy = load_policy(args.policy.resolve())
    bindings = [parse_manifest_arg(value) for value in args.manifest]
    report = audit(policy, bindings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"speech-like-synthetic audit: groups={len(report['groups'])} recordings={report['recordings']} tone_allowed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
