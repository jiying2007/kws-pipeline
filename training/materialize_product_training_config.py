#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))

from external_base_dataset import load_external_base_bundle  # noqa: E402
from generate_speech_like_command_provider import load_policy, normalize_provider  # noqa: E402

SPLITS = ("train", "calibration", "test", "qualification")
HEX = set("0123456789abcdef")
PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: object, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(ch not in HEX for ch in text):
        raise ValueError(f"{label} must be lowercase SHA256")
    return text


def relpath(path: pathlib.Path, root: pathlib.Path) -> str:
    return pathlib.Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", required=True, type=pathlib.Path)
    parser.add_argument("--base-contract", required=True, type=pathlib.Path)
    parser.add_argument("--bundle-root", required=True, type=pathlib.Path)
    parser.add_argument("--replay-provider", required=True, type=pathlib.Path)
    parser.add_argument("--voice-inventory", required=True, type=pathlib.Path)
    parser.add_argument(
        "--command-policy",
        type=pathlib.Path,
        default=ROOT / "configs/training/speech-like-command-provider-v1.json",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    source_path = args.source_config.resolve()
    contract_path = args.base_contract.resolve()
    bundle_root = args.bundle_root.resolve()
    replay_provider_path = args.replay_provider.resolve()
    voice_inventory_path = args.voice_inventory.resolve()
    command_policy_path = args.command_policy.resolve()
    output = args.output.resolve()
    source = load_json(source_path)
    contract = load_json(contract_path)
    if contract.get("policy") != "product-speech-like-base-v1":
        raise ValueError("product speech-like base contract identity mismatch")

    manifest_path = bundle_root / "stage-a-base-bundle.json"
    manifest = load_json(manifest_path)
    expected_bundle = require_sha(
        contract.get("external_base_bundle_sha256"),
        "contract external_base_bundle_sha256",
    )
    expected_provider = require_sha(
        contract.get("provider_identity_sha256"),
        "contract provider_identity_sha256",
    )
    if str(manifest.get("external_base_bundle_sha256") or "") != expected_bundle:
        raise ValueError("speech-like external-base bundle identity mismatch")
    if str(manifest.get("provider_identity_sha256") or "") != expected_provider:
        raise ValueError("speech-like provider identity mismatch")
    if int(manifest.get("recordings", -1)) != int(contract["recordings"]):
        raise ValueError("speech-like recording count mismatch")
    if int(manifest.get("voice_slots", -1)) != int(contract["voice_slots"]):
        raise ValueError("speech-like voice-slot count mismatch")
    if manifest.get("tone_backend_used") is not False:
        raise ValueError("tone-backed product training base is forbidden")
    if manifest.get("protected_evidence_used") is not False:
        raise ValueError("protected evidence may not feed product candidate training")
    if int(manifest.get("source_sample_rate_hz", -1)) != int(contract["source_sample_rate_hz"]):
        raise ValueError("source sample-rate contract drift")
    if int(manifest.get("normalized_sample_rate_hz", -1)) != int(contract["normalized_sample_rate_hz"]):
        raise ValueError("normalized sample-rate contract drift")
    if bool(manifest.get("source_bandwidth_limitation_retained")) is not bool(
        contract["source_bandwidth_limitation_retained"]
    ):
        raise ValueError("source bandwidth-limitation contract drift")

    provider_value = load_json(replay_provider_path)
    command_policy = load_policy(command_policy_path)
    normalized_provider = normalize_provider(
        replay_provider_path,
        provider_value,
        command_policy,
    )
    provider_identity = hashlib.sha256(
        json.dumps(
            normalized_provider["identity"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if provider_identity != expected_provider:
        raise ValueError("replay provider identity differs from product speech-like base")

    inventory_rows = []
    for line_no, raw in enumerate(
        voice_inventory_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"voice inventory line {line_no} must be an object")
        inventory_rows.append(row)

    train_profiles = []
    for row in inventory_rows:
        slot = str(row.get("slot") or "")
        if not slot.startswith("train-"):
            continue
        params = row.get("parameters")
        if not isinstance(params, dict):
            raise ValueError(f"train voice {slot}: parameters are missing")
        speaker_id = params.get("speaker_id")
        length_scale = params.get("length_scale")
        if isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or speaker_id < 0:
            raise ValueError(f"train voice {slot}: speaker_id is invalid")
        length_scale = float(length_scale)
        if not 0.5 <= length_scale <= 2.0:
            raise ValueError(f"train voice {slot}: length_scale is invalid")
        train_profiles.append(
            {"slot": slot, "speaker_id": speaker_id, "length_scale": length_scale}
        )
    train_profiles.sort(key=lambda row: row["slot"])
    expected_train_voices = int(contract.get("replay_train_voice_slots", 0))
    if len(train_profiles) != expected_train_voices or expected_train_voices <= 0:
        raise ValueError(
            f"replay train voice count mismatch: {len(train_profiles)} != {expected_train_voices}"
        )
    if len({row["speaker_id"] for row in train_profiles}) != len(train_profiles):
        raise ValueError("replay train speaker IDs must be unique")

    static_mapping = {"executable": normalized_provider["executable"].as_posix()}
    for role, asset in normalized_provider["assets"].items():
        static_mapping[f"asset:{role}"] = asset["path"].as_posix()

    resolved_command = []
    allowed_dynamic = {"text", "output", "speaker_id", "length_scale"}
    for token in normalized_provider["argv_template"]:
        value = str(token)
        for placeholder in PLACEHOLDER_RE.findall(value):
            if placeholder in static_mapping:
                value = value.replace("{" + placeholder + "}", static_mapping[placeholder])
            elif placeholder not in allowed_dynamic:
                raise ValueError(
                    f"replay provider argv contains unsupported dynamic placeholder: {placeholder}"
                )
        resolved_command.append(value)
    if (
        not resolved_command
        or pathlib.Path(resolved_command[0]).resolve()
        != normalized_provider["executable"]
    ):
        raise ValueError("resolved replay provider executable drifted")

    output.parent.mkdir(parents=True, exist_ok=True)
    effective = copy.deepcopy(source)
    generator = effective.setdefault("generator", {})
    if not isinstance(generator, dict):
        raise ValueError("source training generator must be an object")

    split_spec: dict[str, dict] = {}
    manifest_splits = manifest.get("splits")
    if not isinstance(manifest_splits, dict):
        raise ValueError("speech-like bundle manifest splits are missing")
    expected_splits = contract.get("splits")
    if not isinstance(expected_splits, dict) or set(expected_splits) != set(SPLITS):
        raise ValueError("product speech-like split contract drift")

    for split in SPLITS:
        split_root = bundle_root / "bundle" / split
        index_path = split_root / "dataset-index.jsonl"
        summary_path = split_root / "dataset-summary.json"
        if not index_path.is_file() or not summary_path.is_file():
            raise ValueError(f"{split}: external-base index/summary missing")
        summary = load_json(summary_path)
        if int(summary.get("recordings", -1)) != int(expected_splits[split]):
            raise ValueError(f"{split}: recording count mismatch")
        if summary.get("tone_backend_used") is not False:
            raise ValueError(f"{split}: tone-backed split is forbidden")
        if summary.get("audio_path_contract") != "index-relative-v1":
            raise ValueError(f"{split}: product base must be portable index-relative-v1")
        index_sha = sha256_file(index_path)
        summary_sha = sha256_file(summary_path)
        manifest_row = manifest_splits.get(split)
        if not isinstance(manifest_row, dict):
            raise ValueError(f"{split}: bundle manifest split entry missing")
        if str(manifest_row.get("index_sha256") or "") != index_sha:
            raise ValueError(f"{split}: index SHA differs from bundle manifest")
        if str(manifest_row.get("summary_sha256") or "") != summary_sha:
            raise ValueError(f"{split}: summary SHA differs from bundle manifest")
        split_spec[split] = {
            "index": relpath(index_path, output.parent),
            "summary": relpath(summary_path, output.parent),
            "index_sha256": index_sha,
            "summary_sha256": summary_sha,
        }

    generator["external_base_dataset"] = split_spec
    generator["tts"] = {
        "backend": "command",
        "command": resolved_command,
        "speaker_profiles": [
            {
                "speaker_id": int(row["speaker_id"]),
                "length_scale": float(row["length_scale"]),
            }
            for row in train_profiles
        ],
        "provider_identity_sha256": provider_identity,
        "provider_name": normalized_provider["provider_name"],
        "provider_version": normalized_provider["provider_version"],
        "license_id": normalized_provider["license_id"],
        "command_policy_sha256": sha256_file(command_policy_path),
        "provider_spec_sha256": sha256_file(replay_provider_path),
        "voice_inventory_sha256": sha256_file(voice_inventory_path),
        "replay_voice_scope": "train-only",
    }
    effective["product_candidate_data"] = {
        "schema_version": 1,
        "policy": "external-speech-like-product-base-v1",
        "release_tag": str(contract["release_tag"]),
        "source_repro_run_id": int(contract["source_repro_run_id"]),
        "source_repro_head_sha": str(contract["source_repro_head_sha"]),
        "external_base_bundle_sha256": expected_bundle,
        "provider_identity_sha256": expected_provider,
        "replay_provider_identity_sha256": provider_identity,
        "replay_backend": "command",
        "replay_train_voice_slots": len(train_profiles),
        "replay_tone_allowed": False,
        "base_contract_path": contract_path.relative_to(ROOT).as_posix(),
        "base_contract_sha256": sha256_file(contract_path),
        "source_template_config_path": source_path.relative_to(ROOT).as_posix(),
        "source_template_config_sha256": sha256_file(source_path),
        "recordings": int(contract["recordings"]),
        "voice_slots": int(contract["voice_slots"]),
        "tone_fallback_allowed": False,
        "protected_evidence_used": False,
    }

    # Reuse the production external-base validator against the effective paths,
    # hashes, source/voice split isolation and WAV bytes before training starts.
    loaded = load_external_base_bundle(output, effective)
    if loaded is None:
        raise ValueError("effective config did not bind external speech-like base")
    _, external_summary = loaded
    if str(external_summary.get("bundle_sha256") or "") != expected_bundle:
        raise ValueError("effective external-base summary identity mismatch")

    output.write_text(
        json.dumps(effective, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "product training config: "
        f"release={contract['release_tag']} bundle={expected_bundle} "
        f"provider={expected_provider}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
