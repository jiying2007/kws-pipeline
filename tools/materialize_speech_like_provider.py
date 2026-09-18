#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from generate_speech_like_command_provider import load_policy, normalize_provider  # noqa: E402
from speech_like_corpus_plan import VOICE_CLASS, normalize_plan  # noqa: E402

SPEAKER_MAP_CLASS = "speech-like-provider-speaker-map-v1"
BANNED_LICENSES = {"", "unknown", "todo", "tbd", "required", "required-before-generation", "n/a", "na"}


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


def require_file(path: pathlib.Path, label: str, executable: bool = False) -> pathlib.Path:
    result = path.resolve()
    if not result.is_file():
        raise ValueError(f"{label} is missing: {result}")
    if executable and not os.access(result, os.X_OK):
        raise ValueError(f"{label} is not executable: {result}")
    return result


def require_text(value: str, label: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def load_speaker_map(path: pathlib.Path, expected_slots: list[str]) -> dict[str, dict]:
    value = load_object(path)
    if int(value.get("schema_version", 0)) != 1 or value.get("evidence_class") != SPEAKER_MAP_CLASS:
        raise ValueError("speaker map identity mismatch")
    rows = value.get("speakers")
    if not isinstance(rows, list):
        raise ValueError("speaker map speakers must be a list")
    result: dict[str, dict] = {}
    seen_sid: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"speaker map row {index} must be an object")
        slot = require_text(row.get("slot", ""), f"speaker row {index}.slot")
        if slot in result:
            raise ValueError(f"duplicate speaker slot: {slot}")
        sid = row.get("speaker_id")
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 0:
            raise ValueError(f"speaker row {index}.speaker_id must be a non-negative integer")
        if sid in seen_sid:
            raise ValueError(f"speaker_id reused across slots: {sid}")
        length_scale = float(row.get("length_scale", 1.0))
        if not 0.5 <= length_scale <= 2.0:
            raise ValueError(f"speaker row {index}.length_scale must be in [0.5,2.0]")
        result[slot] = {"speaker_id": sid, "length_scale": length_scale}
        seen_sid.add(sid)
    if set(result) != set(expected_slots):
        missing = sorted(set(expected_slots) - set(result))
        extra = sorted(set(result) - set(expected_slots))
        raise ValueError(f"speaker map must cover exactly corpus-plan slots; missing={missing} extra={extra}")
    return result


def materialize(
    *,
    corpus_plan: pathlib.Path,
    command_policy: pathlib.Path,
    speaker_map: pathlib.Path,
    provider_name: str,
    provider_version: str,
    license_id: str,
    license_file: pathlib.Path,
    executable: pathlib.Path,
    model: pathlib.Path,
    tokens: pathlib.Path,
    lexicon: pathlib.Path,
) -> tuple[dict, list[dict], dict]:
    plan = normalize_plan(corpus_plan)
    slots = [slot for split in ("train", "calibration", "test", "qualification") for slot in plan["roles"][split]["voice_slots"]]
    speakers = load_speaker_map(speaker_map, slots)

    provider_name = require_text(provider_name, "provider_name")
    provider_version = require_text(provider_version, "provider_version")
    license_id = require_text(license_id, "license_id")
    if license_id.strip().lower() in BANNED_LICENSES:
        raise ValueError("license_id must be verified before generation")

    executable = require_file(executable, "provider executable", executable=True)
    model = require_file(model, "VITS model")
    tokens = require_file(tokens, "VITS tokens")
    lexicon = require_file(lexicon, "VITS lexicon")
    license_file = require_file(license_file, "license evidence")

    provider = {
        "schema_version": 1,
        "provider_kind": "offline-tts",
        "provider_name": provider_name,
        "provider_version": provider_version,
        "license_id": license_id,
        "locale": "zh-CN",
        "executable": {"path": str(executable), "sha256": sha256_file(executable)},
        "assets": [
            {"role": "model", "path": str(model), "sha256": sha256_file(model)},
            {"role": "tokens", "path": str(tokens), "sha256": sha256_file(tokens)},
            {"role": "lexicon", "path": str(lexicon), "sha256": sha256_file(lexicon)},
            {"role": "license_evidence", "path": str(license_file), "sha256": sha256_file(license_file)},
        ],
        "argv_template": [
            "{executable}",
            "--vits-model={asset:model}",
            "--vits-tokens={asset:tokens}",
            "--vits-lexicon={asset:lexicon}",
            "--sid={speaker_id}",
            "--vits-length-scale={length_scale}",
            "--output-filename={output}",
            "{text}",
        ],
        "timeout_seconds": 120,
    }

    # Reuse the canonical command-provider validator before emitting anything.
    policy = load_policy(command_policy)
    tmp_provider_path = speaker_map.parent / ".provider-materialize-validation.json"
    tmp_provider_path.write_text(json.dumps(provider, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        normalized = normalize_provider(tmp_provider_path, provider, policy)
    finally:
        tmp_provider_path.unlink(missing_ok=True)

    inventory: list[dict] = []
    for split in ("train", "calibration", "test", "qualification"):
        for slot in plan["roles"][split]["voice_slots"]:
            speaker = speakers[slot]
            sid = int(speaker["speaker_id"])
            inventory.append(
                {
                    "schema_version": 1,
                    "evidence_class": VOICE_CLASS,
                    "slot": slot,
                    "voice_id": f"{provider_name}:sid-{sid:04d}",
                    "parameters": {
                        "speaker_id": sid,
                        "length_scale": speaker["length_scale"],
                    },
                }
            )

    summary = {
        "schema_version": 1,
        "evidence_class": "speech-like-provider-materialization-v1",
        "provider_name": provider_name,
        "provider_version": provider_version,
        "license_id": license_id,
        "license_evidence_sha256": sha256_file(license_file),
        "corpus_plan_sha256": sha256_file(corpus_plan),
        "speaker_map_sha256": sha256_file(speaker_map),
        "command_policy_sha256": sha256_file(command_policy),
        "provider_identity_sha256": hashlib.sha256(
            json.dumps(normalized["identity"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "voice_slots": len(inventory),
        "speaker_ids": [row["parameters"]["speaker_id"] for row in inventory],
        "protected_evidence_used": False,
        "model_asset_sha256": sha256_file(model),
        "executable_sha256": sha256_file(executable),
    }
    return provider, inventory, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a hash-bound offline-TTS provider and exact corpus voice inventory.")
    parser.add_argument("--corpus-plan", required=True, type=pathlib.Path)
    parser.add_argument("--command-policy", required=True, type=pathlib.Path)
    parser.add_argument("--speaker-map", required=True, type=pathlib.Path)
    parser.add_argument("--provider-name", required=True)
    parser.add_argument("--provider-version", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--license-file", required=True, type=pathlib.Path)
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--lexicon", required=True, type=pathlib.Path)
    parser.add_argument("--output-provider", required=True, type=pathlib.Path)
    parser.add_argument("--output-inventory", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    args = parser.parse_args()

    provider, inventory, summary = materialize(
        corpus_plan=args.corpus_plan.resolve(),
        command_policy=args.command_policy.resolve(),
        speaker_map=args.speaker_map.resolve(),
        provider_name=args.provider_name,
        provider_version=args.provider_version,
        license_id=args.license_id,
        license_file=args.license_file,
        executable=args.executable,
        model=args.model,
        tokens=args.tokens,
        lexicon=args.lexicon,
    )
    args.output_provider.parent.mkdir(parents=True, exist_ok=True)
    args.output_provider.write_text(json.dumps(provider, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_inventory.parent.mkdir(parents=True, exist_ok=True)
    args.output_inventory.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in inventory),
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"speech-like provider materialized: provider={summary['provider_name']} "
        f"voices={summary['voice_slots']} model={summary['model_asset_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
