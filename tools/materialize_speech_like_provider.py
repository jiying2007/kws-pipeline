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
from verify_speech_like_runtime_bundle import RECEIPT_CLASS, inspect_verified_archive  # noqa: E402
from verify_speech_like_backend_bundle import (  # noqa: E402
    RECEIPT_CLASS as BACKEND_RECEIPT_CLASS,
    inspect_verified_archive as inspect_verified_backend_archive,
)

SPEAKER_MAP_CLASS = "speech-like-provider-speaker-map-v1"
NATIVE_PROFILE = "sherpa-vits-native-16k-v1"
RESAMPLED_PROFILE = "sherpa-vits-resampled-to-16k-v1"
PROFILES = (NATIVE_PROFILE, RESAMPLED_PROFILE)
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


def require_file(path: pathlib.Path | None, label: str, executable: bool = False) -> pathlib.Path:
    if path is None:
        raise ValueError(f"{label} is required")
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


def validate_provider_reference(
    path: pathlib.Path,
    candidate_name: str,
    *,
    provider_profile: str,
    license_id: str,
    license_file: pathlib.Path,
    source_sample_rate: int | None,
    speakers: dict[str, dict],
) -> dict:
    reference = load_object(path.resolve())
    if (
        int(reference.get("schema_version", 0)) != 1
        or reference.get("evidence_class") != "speech-like-provider-reference-v1"
    ):
        raise ValueError("provider reference identity mismatch")
    rows = reference.get("reference_candidates")
    if not isinstance(rows, list):
        raise ValueError("provider reference candidates must be a list")
    matches = [
        row for row in rows
        if isinstance(row, dict) and str(row.get("name", "")).strip() == candidate_name
    ]
    if len(matches) != 1:
        raise ValueError(f"reference candidate must match exactly once: {candidate_name}")
    row = matches[0]
    if row.get("suitable_for_24_voice_baseline") is not True:
        raise ValueError("reference candidate is not approved for the 24-voice Stage A baseline")
    status = str(row.get("license_status", ""))
    if not status.startswith("verified-"):
        raise ValueError("reference candidate license is not verified")
    if str(row.get("provider_profile", "")) != provider_profile:
        raise ValueError("provider profile does not match pinned reference")
    expected_license = str(row.get("license_id", ""))
    if expected_license != license_id:
        raise ValueError("license_id does not match pinned reference")

    evidence = row.get("license_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("verified reference candidate must declare license_evidence")
    expected_license_sha = str(evidence.get("expected_sha256", "")).lower()
    if len(expected_license_sha) != 64 or sha256_file(license_file) != expected_license_sha:
        raise ValueError("license evidence sha256 does not match pinned reference")

    reported_speakers = int(row.get("reported_speakers", 0))
    if reported_speakers < len(speakers):
        raise ValueError("reference candidate does not have enough speakers for the planned slots")
    speaker_ids = [int(item["speaker_id"]) for item in speakers.values()]
    if any(sid >= reported_speakers for sid in speaker_ids):
        raise ValueError("speaker map uses an id outside the pinned reference speaker range")

    reported_rate = int(row.get("reported_sample_rate_hz", 0))
    normalized_rate = int(row.get("normalized_sample_rate_hz", 0))
    if normalized_rate != 16000:
        raise ValueError("reference candidate normalized sample rate must remain 16000 Hz")
    if provider_profile == RESAMPLED_PROFILE:
        if source_sample_rate != reported_rate:
            raise ValueError("source_sample_rate does not match pinned reference")
        if reported_rate <= 0 or reported_rate >= 16000:
            raise ValueError("resampled reference source rate must be below 16000 Hz")
    elif provider_profile == NATIVE_PROFILE:
        if reported_rate != 16000 or source_sample_rate is not None:
            raise ValueError("native reference candidate must be native 16000 Hz")

    return {
        "reference_sha256": sha256_file(path.resolve()),
        "candidate": candidate_name,
        "provider_profile": provider_profile,
        "license_id": expected_license,
        "license_evidence_sha256": expected_license_sha,
        "license_source": {
            "kind": str(evidence.get("kind", "")),
            "repository": str(evidence.get("repository", "")),
            "revision": str(evidence.get("revision", "")),
            "path": str(evidence.get("path", "")),
        },
        "reported_speakers": reported_speakers,
        "reported_sample_rate_hz": reported_rate,
        "normalized_sample_rate_hz": normalized_rate,
        "runtime_asset_bundle_required": isinstance(row.get("runtime_asset_bundle"), dict),
    }


def validate_runtime_asset_binding(
    *,
    reference_path: pathlib.Path,
    candidate_name: str,
    archive_path: pathlib.Path,
    receipt_path: pathlib.Path,
    model: pathlib.Path,
    tokens: pathlib.Path,
    lexicon: pathlib.Path,
    phone_fst: pathlib.Path | None,
    date_fst: pathlib.Path | None,
    number_fst: pathlib.Path | None,
    rule_far: pathlib.Path | None,
) -> dict:
    archive_path = require_file(archive_path, "runtime asset archive")
    receipt_path = require_file(receipt_path, "runtime asset receipt")
    receipt = load_object(receipt_path)
    if (
        int(receipt.get("schema_version", 0)) != 1
        or receipt.get("evidence_class") != RECEIPT_CLASS
    ):
        raise ValueError("runtime asset receipt identity mismatch")

    inspected = inspect_verified_archive(
        reference_path=reference_path.resolve(),
        candidate_name=candidate_name,
        archive_path=archive_path,
    )
    if receipt != inspected:
        raise ValueError("runtime asset receipt does not match the verified archive")

    supplied = {
        "model": require_file(model, "VITS model"),
        "tokens": require_file(tokens, "VITS tokens"),
        "lexicon": require_file(lexicon, "VITS lexicon"),
    }
    rule_values = {
        "phone_fst": phone_fst,
        "date_fst": date_fst,
        "number_fst": number_fst,
        "rule_far": rule_far,
    }
    if any(value is not None for value in rule_values.values()):
        if any(value is None for value in rule_values.values()):
            raise ValueError("phone/date/number FSTs and rule.far must be supplied together")
        supplied.update(
            {
                role: require_file(value, role.replace("_", " "))
                for role, value in rule_values.items()
            }
        )
    if set(supplied) != set(inspected["files"]):
        missing = sorted(set(inspected["files"]) - set(supplied))
        extra = sorted(set(supplied) - set(inspected["files"]))
        raise ValueError(
            f"supplied runtime asset roles do not match verified archive; missing={missing} extra={extra}"
        )
    for role, path in supplied.items():
        expected = inspected["files"][role]
        if path.stat().st_size != int(expected["size_bytes"]):
            raise ValueError(f"runtime {role} size does not match verified archive member")
        if sha256_file(path) != str(expected["sha256"]):
            raise ValueError(f"runtime {role} sha256 does not match verified archive member")

    return {
        "receipt_sha256": sha256_file(receipt_path),
        "archive_sha256": str(inspected["archive"]["sha256"]),
        "archive_size_bytes": int(inspected["archive"]["size_bytes"]),
        "archive_source_url": str(inspected["archive"]["source_url"]),
        "files": {
            role: {
                "archive_path": str(inspected["files"][role]["path"]),
                "size_bytes": int(inspected["files"][role]["size_bytes"]),
                "sha256": str(inspected["files"][role]["sha256"]),
            }
            for role in sorted(inspected["files"])
        },
        "safe_archive_verified": True,
    }


def validate_backend_bundle_binding(
    *,
    reference_path: pathlib.Path,
    platform_key: str,
    archive_path: pathlib.Path,
    receipt_path: pathlib.Path,
    bundle_root: pathlib.Path,
    backend_executable: pathlib.Path,
    backend_lib_dir: pathlib.Path,
) -> dict:
    archive_path = require_file(archive_path, "backend bundle archive")
    receipt_path = require_file(receipt_path, "backend bundle receipt")
    bundle_root = bundle_root.resolve()
    backend_executable = require_file(
        backend_executable, "backend executable", executable=True
    )
    backend_lib_dir = backend_lib_dir.resolve()
    if not backend_lib_dir.is_dir():
        raise ValueError(f"backend lib dir is missing: {backend_lib_dir}")

    receipt = load_object(receipt_path)
    if (
        int(receipt.get("schema_version", 0)) != 1
        or receipt.get("evidence_class") != BACKEND_RECEIPT_CLASS
    ):
        raise ValueError("backend bundle receipt identity mismatch")
    inspected = inspect_verified_backend_archive(
        reference_path=reference_path.resolve(),
        platform_key=platform_key,
        archive_path=archive_path,
    )
    if receipt != inspected:
        raise ValueError("backend bundle receipt does not match the verified archive")

    expected_executable = (
        bundle_root
        / pathlib.PurePosixPath(str(inspected["backend_executable"]["path"]))
    ).resolve()
    if backend_executable != expected_executable:
        raise ValueError("backend executable path does not match verified bundle")
    if backend_executable.stat().st_size != int(
        inspected["backend_executable"]["size_bytes"]
    ):
        raise ValueError("backend executable size does not match verified bundle")
    if sha256_file(backend_executable) != str(
        inspected["backend_executable"]["sha256"]
    ):
        raise ValueError("backend executable sha256 does not match verified bundle")

    expected_lib_dir = (
        bundle_root / pathlib.PurePosixPath(str(inspected["lib_dir"]))
    ).resolve()
    if backend_lib_dir != expected_lib_dir:
        raise ValueError("backend lib dir does not match verified bundle")

    expected_regular = {
        str(item["path"]): item for item in inspected["libraries"]
    }
    actual_regular: dict[str, pathlib.Path] = {}
    actual_symlinks: dict[str, str] = {}
    archive_root = pathlib.PurePosixPath(str(inspected["archive_root"]))
    for path in sorted(backend_lib_dir.rglob("*")):
        rel_from_bundle = path.relative_to(bundle_root).as_posix()
        if path.is_symlink():
            actual_symlinks[rel_from_bundle] = os.readlink(path)
        elif path.is_file():
            actual_regular[rel_from_bundle] = path
        elif not path.is_dir():
            raise ValueError(f"backend lib dir contains unsupported entry: {path}")

    if set(actual_regular) != set(expected_regular):
        missing = sorted(set(expected_regular) - set(actual_regular))
        extra = sorted(set(actual_regular) - set(expected_regular))
        raise ValueError(
            f"backend library file set drifted; missing={missing} extra={extra}"
        )
    for rel, path in actual_regular.items():
        expected = expected_regular[rel]
        if path.stat().st_size != int(expected["size_bytes"]):
            raise ValueError(f"backend library size mismatch: {rel}")
        if sha256_file(path) != str(expected["sha256"]):
            raise ValueError(f"backend library sha256 mismatch: {rel}")

    expected_symlinks = {
        str(item["path"]): str(item["target"])
        for item in inspected["library_symlinks"]
    }
    if actual_symlinks != expected_symlinks:
        raise ValueError("backend library symlink set/targets drifted")

    assets = [
        {
            "role": "backend_bundle_archive",
            "path": str(archive_path),
            "sha256": sha256_file(archive_path),
        },
        {
            "role": "backend_bundle_receipt",
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
        },
        {
            "role": "backend_executable",
            "path": str(backend_executable),
            "sha256": sha256_file(backend_executable),
        },
    ]
    for index, rel in enumerate(sorted(actual_regular)):
        path = actual_regular[rel]
        assets.append(
            {
                "role": f"backend_lib_{index:04d}",
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )

    return {
        "platform": platform_key,
        "archive_sha256": str(inspected["archive"]["sha256"]),
        "receipt_sha256": sha256_file(receipt_path),
        "backend_executable_sha256": sha256_file(backend_executable),
        "backend_lib_dir": str(backend_lib_dir),
        "library_count": len(actual_regular),
        "library_symlink_count": len(actual_symlinks),
        "assets": assets,
        "safe_archive_verified": True,
    }


def build_provider(
    *,
    provider_profile: str,
    provider_name: str,
    provider_version: str,
    license_id: str,
    license_file: pathlib.Path,
    executable: pathlib.Path,
    model: pathlib.Path,
    tokens: pathlib.Path,
    lexicon: pathlib.Path,
    phone_fst: pathlib.Path | None,
    date_fst: pathlib.Path | None,
    number_fst: pathlib.Path | None,
    rule_far: pathlib.Path | None,
    adapter: pathlib.Path | None,
    backend_executable: pathlib.Path | None,
    backend_lib_dir: pathlib.Path | None,
    backend_bundle_assets: list[dict] | None,
    resampler_executable: pathlib.Path | None,
    source_sample_rate: int | None,
    runtime_asset_receipt: pathlib.Path | None,
) -> tuple[dict, dict]:
    executable = require_file(executable, "provider executable", executable=True)
    model = require_file(model, "VITS model")
    tokens = require_file(tokens, "VITS tokens")
    lexicon = require_file(lexicon, "VITS lexicon")
    license_file = require_file(license_file, "license evidence")

    assets = [
        {"role": "model", "path": str(model), "sha256": sha256_file(model)},
        {"role": "tokens", "path": str(tokens), "sha256": sha256_file(tokens)},
        {"role": "lexicon", "path": str(lexicon), "sha256": sha256_file(lexicon)},
        {"role": "license_evidence", "path": str(license_file), "sha256": sha256_file(license_file)},
    ]
    rule_values = {
        "phone_fst": phone_fst,
        "date_fst": date_fst,
        "number_fst": number_fst,
        "rule_far": rule_far,
    }
    if any(value is not None for value in rule_values.values()):
        if any(value is None for value in rule_values.values()):
            raise ValueError("phone/date/number FSTs and rule.far must be supplied together")
        for role, value in rule_values.items():
            path = require_file(value, role.replace("_", " "))
            assets.append({"role": role, "path": str(path), "sha256": sha256_file(path)})
    if runtime_asset_receipt is not None:
        runtime_asset_receipt = require_file(runtime_asset_receipt, "runtime asset receipt")
        assets.append(
            {
                "role": "runtime_asset_receipt",
                "path": str(runtime_asset_receipt),
                "sha256": sha256_file(runtime_asset_receipt),
            }
        )
    normalization: dict

    if provider_profile == NATIVE_PROFILE:
        if any(value is not None for value in (adapter, backend_executable, resampler_executable, source_sample_rate)):
            raise ValueError("native VITS profile does not accept adapter/backend/resampler/source-sample-rate")
        argv_template = [
            "{executable}",
            "--vits-model={asset:model}",
            "--vits-tokens={asset:tokens}",
            "--vits-lexicon={asset:lexicon}",
        ]
        if phone_fst is not None:
            argv_template.extend(
                [
                    "--tts-rule-fsts={asset:phone_fst},{asset:date_fst},{asset:number_fst}",
                    "--tts-rule-fars={asset:rule_far}",
                ]
            )
        argv_template.extend(
            [
                "--sid={speaker_id}",
                "--vits-length-scale={length_scale}",
                "--output-filename={output}",
                "{text}",
            ]
        )
        timeout_seconds = 120
        normalization = {
            "policy": "native-16k-v1",
            "source_sample_rate_hz": 16000,
            "output_sample_rate_hz": 16000,
            "resampled": False,
            "vits_noise_scale": 0.0,
            "vits_noise_scale_w": 0.0,
            "deterministic_vits_sampling": True,
        }
    elif provider_profile == RESAMPLED_PROFILE:
        adapter = require_file(adapter, "VITS resample adapter")
        backend_executable = require_file(backend_executable, "VITS backend executable", executable=True)
        resampler_executable = (
            require_file(
                resampler_executable, "resampler executable", executable=True
            )
            if resampler_executable is not None
            else None
        )
        source_sample_rate = int(source_sample_rate or 0)
        if source_sample_rate <= 0 or source_sample_rate >= 16000:
            raise ValueError("resampled VITS profile requires source_sample_rate in [1,15999]")
        if resampler_executable is None and source_sample_rate != 8000:
            raise ValueError(
                "builtin Lanczos resampler requires source_sample_rate=8000"
            )
        assets.append(
            {"role": "adapter", "path": str(adapter), "sha256": sha256_file(adapter)}
        )
        if (backend_lib_dir is None) != (backend_bundle_assets is None):
            raise ValueError(
                "backend_lib_dir and backend_bundle_assets must be supplied together"
            )
        if backend_bundle_assets is None:
            assets.append(
                {
                    "role": "backend_executable",
                    "path": str(backend_executable),
                    "sha256": sha256_file(backend_executable),
                }
            )
        else:
            seen_roles = {item["role"] for item in assets}
            for item in backend_bundle_assets:
                role = require_text(item.get("role"), "backend bundle asset role")
                if role in seen_roles:
                    raise ValueError(f"duplicate provider asset role: {role}")
                asset_path = require_file(
                    pathlib.Path(str(item.get("path", ""))),
                    f"backend bundle asset {role}",
                )
                expected = str(item.get("sha256", "")).lower()
                if len(expected) != 64 or sha256_file(asset_path) != expected:
                    raise ValueError(f"backend bundle asset sha256 mismatch: {role}")
                assets.append(
                    {"role": role, "path": str(asset_path), "sha256": expected}
                )
                seen_roles.add(role)
            if "backend_executable" not in seen_roles:
                raise ValueError("backend bundle assets must include backend_executable")
        if resampler_executable is not None:
            assets.append(
                {
                    "role": "resampler_executable",
                    "path": str(resampler_executable),
                    "sha256": sha256_file(resampler_executable),
                }
            )
        argv_template = [
            "{executable}",
            "{asset:adapter}",
            "--backend-executable={asset:backend_executable}",
        ]
        if resampler_executable is not None:
            argv_template.append(
                "--resampler-executable={asset:resampler_executable}"
            )
        argv_template.extend(
            [
                "--model={asset:model}",
                "--tokens={asset:tokens}",
                "--lexicon={asset:lexicon}",
            ]
        )
        if phone_fst is not None:
            argv_template.extend(
                [
                    "--phone-fst={asset:phone_fst}",
                    "--date-fst={asset:date_fst}",
                    "--number-fst={asset:number_fst}",
                    "--rule-far={asset:rule_far}",
                ]
            )
        argv_template.extend(
            [
                "--speaker-id={speaker_id}",
                "--length-scale={length_scale}",
                "--noise-scale=0.0",
                "--noise-scale-w=0.0",
                f"--source-sample-rate={source_sample_rate}",
                "--output={output}",
                "{text}",
            ]
        )
        timeout_seconds = 300
        normalization = {
            "policy": "verified-source-rate-to-pcm16-16k-v1",
            "source_sample_rate_hz": source_sample_rate,
            "output_sample_rate_hz": 16000,
            "resampled": True,
            "vits_noise_scale": 0.0,
            "vits_noise_scale_w": 0.0,
            "deterministic_vits_sampling": True,
            "backend_lib_resolution": (
                "sibling-lib-from-verified-backend"
                if backend_bundle_assets is not None
                else "process-environment"
            ),
            "adapter_sha256": sha256_file(adapter),
            "backend_executable_sha256": sha256_file(backend_executable),
            "backend_bundle_verified": backend_bundle_assets is not None,
            "backend_library_count": (
                sum(
                    1
                    for item in (backend_bundle_assets or [])
                    if str(item.get("role", "")).startswith("backend_lib_")
                )
            ),
            "resampler_kind": (
                "external-executable"
                if resampler_executable is not None
                else "builtin-lanczos-2x-v1"
            ),
            "resampler_executable_sha256": (
                sha256_file(resampler_executable)
                if resampler_executable is not None
                else None
            ),
        }
    else:
        raise ValueError(f"unsupported provider_profile: {provider_profile}")

    provider = {
        "schema_version": 1,
        "provider_kind": "offline-tts",
        "provider_name": provider_name,
        "provider_version": provider_version,
        "license_id": license_id,
        "locale": "zh-CN",
        "executable": {"path": str(executable), "sha256": sha256_file(executable)},
        "assets": assets,
        "argv_template": argv_template,
        "timeout_seconds": timeout_seconds,
    }
    return provider, normalization


def materialize(
    *,
    corpus_plan: pathlib.Path,
    command_policy: pathlib.Path,
    speaker_map: pathlib.Path,
    provider_profile: str,
    provider_name: str,
    provider_version: str,
    license_id: str,
    license_file: pathlib.Path,
    executable: pathlib.Path,
    model: pathlib.Path,
    tokens: pathlib.Path,
    lexicon: pathlib.Path,
    phone_fst: pathlib.Path | None = None,
    date_fst: pathlib.Path | None = None,
    number_fst: pathlib.Path | None = None,
    rule_far: pathlib.Path | None = None,
    adapter: pathlib.Path | None = None,
    backend_executable: pathlib.Path | None = None,
    resampler_executable: pathlib.Path | None = None,
    source_sample_rate: int | None = None,
    provider_reference: pathlib.Path | None = None,
    reference_candidate: str | None = None,
    runtime_asset_archive: pathlib.Path | None = None,
    runtime_asset_receipt: pathlib.Path | None = None,
    backend_platform: str | None = None,
    backend_bundle_archive: pathlib.Path | None = None,
    backend_bundle_receipt: pathlib.Path | None = None,
    backend_bundle_root: pathlib.Path | None = None,
    backend_lib_dir: pathlib.Path | None = None,
) -> tuple[dict, list[dict], dict]:
    plan = normalize_plan(corpus_plan)
    slots = [
        slot
        for split in ("train", "calibration", "test", "qualification")
        for slot in plan["roles"][split]["voice_slots"]
    ]
    speakers = load_speaker_map(speaker_map, slots)

    provider_name = require_text(provider_name, "provider_name")
    provider_version = require_text(provider_version, "provider_version")
    license_id = require_text(license_id, "license_id")
    if license_id.strip().lower() in BANNED_LICENSES:
        raise ValueError("license_id must be verified before generation")
    license_file = require_file(license_file, "license evidence")
    if (provider_reference is None) != (reference_candidate is None):
        raise ValueError("provider_reference and reference_candidate must be supplied together")
    reference_binding = None
    runtime_asset_binding = None
    if provider_reference is not None and reference_candidate is not None:
        candidate_name = require_text(reference_candidate, "reference_candidate")
        reference_binding = validate_provider_reference(
            provider_reference,
            candidate_name,
            provider_profile=provider_profile,
            license_id=license_id,
            license_file=license_file,
            source_sample_rate=source_sample_rate,
            speakers=speakers,
        )
        if reference_binding["runtime_asset_bundle_required"]:
            if runtime_asset_archive is None or runtime_asset_receipt is None:
                raise ValueError(
                    "pinned reference candidate requires runtime asset archive and receipt"
                )
            runtime_asset_binding = validate_runtime_asset_binding(
                reference_path=provider_reference,
                candidate_name=candidate_name,
                archive_path=runtime_asset_archive,
                receipt_path=runtime_asset_receipt,
                model=model,
                tokens=tokens,
                lexicon=lexicon,
                phone_fst=phone_fst,
                date_fst=date_fst,
                number_fst=number_fst,
                rule_far=rule_far,
            )
        elif runtime_asset_archive is not None or runtime_asset_receipt is not None:
            raise ValueError("reference candidate does not declare a runtime asset bundle")
    elif runtime_asset_archive is not None or runtime_asset_receipt is not None:
        raise ValueError("runtime asset archive/receipt require a pinned provider reference")

    backend_bundle_binding = None
    backend_bundle_values = (
        backend_platform,
        backend_bundle_archive,
        backend_bundle_receipt,
        backend_bundle_root,
        backend_lib_dir,
    )
    if any(value is not None for value in backend_bundle_values):
        if any(value is None for value in backend_bundle_values):
            raise ValueError(
                "backend platform/archive/receipt/root/lib-dir must be supplied together"
            )
        if provider_reference is None:
            raise ValueError("backend bundle verification requires provider_reference")
        if backend_executable is None:
            raise ValueError("backend bundle verification requires backend_executable")
        backend_bundle_binding = validate_backend_bundle_binding(
            reference_path=provider_reference,
            platform_key=require_text(backend_platform, "backend_platform"),
            archive_path=backend_bundle_archive,
            receipt_path=backend_bundle_receipt,
            bundle_root=backend_bundle_root,
            backend_executable=backend_executable,
            backend_lib_dir=backend_lib_dir,
        )

    provider, normalization = build_provider(
        provider_profile=provider_profile,
        provider_name=provider_name,
        provider_version=provider_version,
        license_id=license_id,
        license_file=license_file,
        executable=executable,
        model=model,
        tokens=tokens,
        lexicon=lexicon,
        phone_fst=phone_fst,
        date_fst=date_fst,
        number_fst=number_fst,
        rule_far=rule_far,
        adapter=adapter,
        backend_executable=backend_executable,
        backend_lib_dir=backend_lib_dir,
        backend_bundle_assets=(
            backend_bundle_binding["assets"]
            if backend_bundle_binding is not None
            else None
        ),
        resampler_executable=resampler_executable,
        source_sample_rate=source_sample_rate,
        runtime_asset_receipt=runtime_asset_receipt,
    )

    # Reuse the canonical command-provider validator before emitting anything.
    policy = load_policy(command_policy)
    tmp_provider_path = speaker_map.parent / ".provider-materialize-validation.json"
    tmp_provider_path.write_text(
        json.dumps(provider, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
        "provider_profile": provider_profile,
        "provider_name": provider_name,
        "provider_version": provider_version,
        "license_id": license_id,
        "license_evidence_sha256": sha256_file(license_file),
        "corpus_plan_sha256": sha256_file(corpus_plan),
        "speaker_map_sha256": sha256_file(speaker_map),
        "command_policy_sha256": sha256_file(command_policy),
        "provider_identity_sha256": hashlib.sha256(
            json.dumps(
                normalized["identity"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "voice_slots": len(inventory),
        "speaker_ids": [row["parameters"]["speaker_id"] for row in inventory],
        "protected_evidence_used": False,
        "model_asset_sha256": sha256_file(model),
        "executable_sha256": sha256_file(executable),
        "audio_normalization": normalization,
    }
    if reference_binding is not None:
        summary["provider_reference"] = reference_binding
        summary["license_reference_verified"] = True
    if runtime_asset_binding is not None:
        summary["runtime_asset_binding"] = runtime_asset_binding
        summary["runtime_asset_receipt_verified"] = True
    if backend_bundle_binding is not None:
        summary["backend_bundle_binding"] = {
            key: value
            for key, value in backend_bundle_binding.items()
            if key != "assets"
        }
        summary["backend_bundle_verified"] = True
    return provider, inventory, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a hash-bound offline-TTS provider and exact corpus voice inventory."
    )
    parser.add_argument("--corpus-plan", required=True, type=pathlib.Path)
    parser.add_argument("--command-policy", required=True, type=pathlib.Path)
    parser.add_argument("--speaker-map", required=True, type=pathlib.Path)
    parser.add_argument("--provider-profile", choices=PROFILES, default=NATIVE_PROFILE)
    parser.add_argument("--provider-name", required=True)
    parser.add_argument("--provider-version", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--license-file", required=True, type=pathlib.Path)
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--lexicon", required=True, type=pathlib.Path)
    parser.add_argument("--phone-fst", type=pathlib.Path)
    parser.add_argument("--date-fst", type=pathlib.Path)
    parser.add_argument("--number-fst", type=pathlib.Path)
    parser.add_argument("--rule-far", type=pathlib.Path)
    parser.add_argument("--adapter", type=pathlib.Path)
    parser.add_argument("--backend-executable", type=pathlib.Path)
    parser.add_argument("--resampler-executable", type=pathlib.Path)
    parser.add_argument("--source-sample-rate", type=int)
    parser.add_argument("--provider-reference", type=pathlib.Path)
    parser.add_argument("--reference-candidate")
    parser.add_argument("--runtime-asset-archive", type=pathlib.Path)
    parser.add_argument("--runtime-asset-receipt", type=pathlib.Path)
    parser.add_argument("--backend-platform")
    parser.add_argument("--backend-bundle-archive", type=pathlib.Path)
    parser.add_argument("--backend-bundle-receipt", type=pathlib.Path)
    parser.add_argument("--backend-bundle-root", type=pathlib.Path)
    parser.add_argument("--backend-lib-dir", type=pathlib.Path)
    parser.add_argument("--output-provider", required=True, type=pathlib.Path)
    parser.add_argument("--output-inventory", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    args = parser.parse_args()

    provider, inventory, summary = materialize(
        corpus_plan=args.corpus_plan.resolve(),
        command_policy=args.command_policy.resolve(),
        speaker_map=args.speaker_map.resolve(),
        provider_profile=args.provider_profile,
        provider_name=args.provider_name,
        provider_version=args.provider_version,
        license_id=args.license_id,
        license_file=args.license_file,
        executable=args.executable,
        model=args.model,
        tokens=args.tokens,
        lexicon=args.lexicon,
        phone_fst=args.phone_fst,
        date_fst=args.date_fst,
        number_fst=args.number_fst,
        rule_far=args.rule_far,
        adapter=args.adapter,
        backend_executable=args.backend_executable,
        resampler_executable=args.resampler_executable,
        source_sample_rate=args.source_sample_rate,
        provider_reference=args.provider_reference,
        reference_candidate=args.reference_candidate,
        runtime_asset_archive=args.runtime_asset_archive,
        runtime_asset_receipt=args.runtime_asset_receipt,
        backend_platform=args.backend_platform,
        backend_bundle_archive=args.backend_bundle_archive,
        backend_bundle_receipt=args.backend_bundle_receipt,
        backend_bundle_root=args.backend_bundle_root,
        backend_lib_dir=args.backend_lib_dir,
    )
    args.output_provider.parent.mkdir(parents=True, exist_ok=True)
    args.output_provider.write_text(
        json.dumps(provider, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_inventory.parent.mkdir(parents=True, exist_ok=True)
    args.output_inventory.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in inventory),
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"speech-like provider materialized: profile={summary['provider_profile']} "
        f"provider={summary['provider_name']} voices={summary['voice_slots']} "
        f"model={summary['model_asset_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
