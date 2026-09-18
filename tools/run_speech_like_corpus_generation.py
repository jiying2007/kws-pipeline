#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
TRAINING = ROOT / "training"
SPLITS = ("train", "calibration", "test", "qualification")
REFERENCE_CLASS = "speech-like-provider-reference-v1"
BUNDLE_CLASS = "speech-like-stage-a-base-bundle-v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def require_file(path: pathlib.Path | None, label: str, *, executable: bool = False) -> pathlib.Path:
    if path is None:
        raise ValueError(f"{label} is required")
    result = path.resolve()
    if not result.is_file():
        raise ValueError(f"{label} is missing: {result}")
    if executable and not os.access(result, os.X_OK):
        raise ValueError(f"{label} is not executable: {result}")
    return result


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


def select_reference_candidate(reference_path: pathlib.Path, requested: str | None) -> tuple[dict, dict]:
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
        raise ValueError("selected provider candidate is not approved for Stage A")
    if not str(row.get("license_status", "")).startswith("verified-"):
        raise ValueError("selected provider candidate license is not verified")
    if int(row.get("reported_speakers", 0)) < 24:
        raise ValueError("selected provider candidate has fewer than 24 reported speakers")
    bundle = row.get("runtime_asset_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("selected provider candidate lacks runtime_asset_bundle")
    return reference, row


def corpus_plan_slots(plan_path: pathlib.Path) -> tuple[list[str], dict]:
    plan = load_object(plan_path)
    if int(plan.get("schema_version", 0)) != 1 or plan.get("policy") != "speech-like-corpus-plan-v1":
        raise ValueError("speech-like corpus plan identity mismatch")
    roles = plan.get("split_roles")
    utterances = plan.get("utterances")
    if not isinstance(roles, dict) or set(roles) != set(SPLITS):
        raise ValueError("speech-like corpus plan split roles drifted")
    if not isinstance(utterances, list) or len(utterances) != 16:
        raise ValueError("Stage A speech-like corpus plan must contain 16 utterances")
    slots: list[str] = []
    split_counts: dict[str, int] = {}
    for split in SPLITS:
        role = roles[split]
        if not isinstance(role, dict) or not isinstance(role.get("voice_slots"), list):
            raise ValueError(f"speech-like corpus plan {split} role is invalid")
        voice_slots = [str(value) for value in role["voice_slots"]]
        if any(not value for value in voice_slots):
            raise ValueError(f"speech-like corpus plan {split} has empty voice slot")
        slots.extend(voice_slots)
        split_counts[split] = len(voice_slots) * len(utterances)
    if len(slots) != 24 or len(set(slots)) != 24:
        raise ValueError("Stage A speech-like corpus plan must contain exactly 24 unique voice slots")
    if split_counts != {"train": 128, "calibration": 64, "test": 64, "qualification": 128}:
        raise ValueError(f"Stage A speech-like split counts drifted: {split_counts}")
    return slots, split_counts


def build_speaker_map(slots: list[str], output: pathlib.Path) -> None:
    rows = [
        {
            "slot": slot,
            "speaker_id": index,
            "length_scale": round(0.90 + (index % 5) * 0.05, 2),
        }
        for index, slot in enumerate(slots)
    ]
    value = {
        "schema_version": 1,
        "evidence_class": "speech-like-provider-speaker-map-v1",
        "mapping_policy": "sequential-first-24-speakers-v1",
        "speakers": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_stage_inputs(stage_config_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    config = load_object(stage_config_path)
    tokens_raw = config.get("tokens")
    keywords_raw = config.get("keywords")
    if not isinstance(tokens_raw, str) or not tokens_raw:
        raise ValueError("Stage A base config tokens path is missing")
    if not isinstance(keywords_raw, str) or not keywords_raw:
        raise ValueError("Stage A base config keywords path is missing")
    tokens = pathlib.Path(tokens_raw)
    keywords = pathlib.Path(keywords_raw)
    if tokens.is_absolute() or keywords.is_absolute():
        raise ValueError("Stage A canonical tokens/keywords paths must remain repository-relative")
    return require_file(ROOT / tokens, "Stage A tokens"), require_file(
        ROOT / keywords, "Stage A keywords"
    )


def relative_ref(target: pathlib.Path, root: pathlib.Path) -> str:
    return pathlib.Path(os.path.relpath(target.resolve(), root.resolve())).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the governed 24-voice/384-recording speech-like Stage A base corpus "
            "from a pinned offline TTS provider."
        )
    )
    parser.add_argument(
        "--provider-reference",
        type=pathlib.Path,
        default=ROOT / "configs" / "training" / "speech-like-provider-reference-v1.json",
    )
    parser.add_argument("--reference-candidate")
    parser.add_argument(
        "--corpus-plan",
        type=pathlib.Path,
        default=ROOT / "configs" / "training" / "speech-like-corpus-plan-v1.json",
    )
    parser.add_argument(
        "--command-policy",
        type=pathlib.Path,
        default=ROOT / "configs" / "training" / "speech-like-command-provider-v1.json",
    )
    parser.add_argument(
        "--stage-base-config",
        type=pathlib.Path,
        default=ROOT / "configs" / "training" / "xiaowo.v2-speech-like-stage-base.json",
    )
    parser.add_argument("--runtime-archive", required=True, type=pathlib.Path)
    parser.add_argument("--license-evidence", required=True, type=pathlib.Path)
    parser.add_argument("--backend-executable", required=True, type=pathlib.Path)
    parser.add_argument("--resampler-executable", type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    reference_path = require_file(args.provider_reference, "provider reference")
    corpus_plan = require_file(args.corpus_plan, "corpus plan")
    command_policy = require_file(args.command_policy, "command-provider policy")
    stage_base_config = require_file(args.stage_base_config, "Stage A base config")
    runtime_archive = require_file(args.runtime_archive, "runtime archive")
    license_evidence = require_file(args.license_evidence, "license evidence")
    backend = require_file(args.backend_executable, "TTS backend executable", executable=True)

    reference, candidate = select_reference_candidate(reference_path, args.reference_candidate)
    candidate_name = str(candidate["name"])
    profile = str(candidate.get("provider_profile", ""))
    source_rate = int(candidate.get("reported_sample_rate_hz", 0))
    license_id = str(candidate.get("license_id", "")).strip()
    bundle_contract = candidate["runtime_asset_bundle"]
    if not license_id:
        raise ValueError("selected provider candidate has no license_id")

    resampler: pathlib.Path | None = None
    if profile == "sherpa-vits-resampled-to-16k-v1":
        resampler = require_file(args.resampler_executable, "resampler executable", executable=True)
        if source_rate <= 0 or source_rate >= 16000:
            raise ValueError("resampled provider source rate must be below 16000 Hz")
    elif profile == "sherpa-vits-native-16k-v1":
        if source_rate != 16000:
            raise ValueError("native provider source rate must be 16000 Hz")
        if args.resampler_executable is not None:
            raise ValueError("native provider must not receive a resampler executable")
    else:
        raise ValueError(f"unsupported Stage A provider profile: {profile}")

    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError("speech-like corpus work-dir must be empty")
    work.mkdir(parents=True, exist_ok=True)
    logs = work / "logs"

    slots, expected_split_counts = corpus_plan_slots(corpus_plan)
    speaker_map = work / "speaker-map.json"
    build_speaker_map(slots, speaker_map)

    runtime_root = work / "runtime-assets"
    runtime_receipt = work / "runtime-asset-receipt.json"
    run_checked(
        [
            sys.executable,
            str(TOOLS / "verify_speech_like_runtime_bundle.py"),
            "--reference",
            str(reference_path),
            "--candidate",
            candidate_name,
            "--archive",
            str(runtime_archive),
            "--output-dir",
            str(runtime_root),
            "--receipt",
            str(runtime_receipt),
        ],
        logs / "verify-runtime-bundle.log",
    )
    receipt = load_object(runtime_receipt)
    asset_paths = {
        role: require_file(
            runtime_root / pathlib.Path(str(receipt["files"][role]["path"])),
            f"runtime {role}",
        )
        for role in ("model", "tokens", "lexicon")
    }

    provider_dir = work / "provider"
    provider_path = provider_dir / "provider.json"
    inventory_path = provider_dir / "voice-inventory.jsonl"
    provider_summary_path = provider_dir / "provider-summary.json"
    provider_version = (
        f"{str(bundle_contract.get('tag', 'runtime'))}-"
        f"{str(bundle_contract.get('expected_sha256', ''))[:12]}"
    )
    command = [
        sys.executable,
        str(TOOLS / "materialize_speech_like_provider.py"),
        "--corpus-plan",
        str(corpus_plan),
        "--command-policy",
        str(command_policy),
        "--speaker-map",
        str(speaker_map),
        "--provider-profile",
        profile,
        "--provider-name",
        candidate_name,
        "--provider-version",
        provider_version,
        "--license-id",
        license_id,
        "--license-file",
        str(license_evidence),
        "--provider-reference",
        str(reference_path),
        "--reference-candidate",
        candidate_name,
        "--runtime-asset-archive",
        str(runtime_archive),
        "--runtime-asset-receipt",
        str(runtime_receipt),
        "--model",
        str(asset_paths["model"]),
        "--tokens",
        str(asset_paths["tokens"]),
        "--lexicon",
        str(asset_paths["lexicon"]),
        "--output-provider",
        str(provider_path),
        "--output-inventory",
        str(inventory_path),
        "--summary",
        str(provider_summary_path),
    ]
    if profile == "sherpa-vits-resampled-to-16k-v1":
        command.extend(
            [
                "--executable",
                sys.executable,
                "--adapter",
                str(TOOLS / "speech_like_vits_resample_adapter.py"),
                "--backend-executable",
                str(backend),
                "--resampler-executable",
                str(resampler),
                "--source-sample-rate",
                str(source_rate),
            ]
        )
    else:
        command.extend(["--executable", str(backend)])
    run_checked(command, logs / "materialize-provider.log")

    provider_summary = load_object(provider_summary_path)
    if provider_summary.get("license_reference_verified") is not True:
        raise ValueError("provider materialization did not verify pinned license evidence")
    if provider_summary.get("runtime_asset_receipt_verified") is not True:
        raise ValueError("provider materialization did not verify runtime asset receipt")
    if int(provider_summary.get("voice_slots", 0)) != 24:
        raise ValueError("provider materialization did not produce exactly 24 voices")

    plan_dir = work / "request-plan"
    requests = plan_dir / "requests.jsonl"
    intents = plan_dir / "intents.jsonl"
    request_summary_path = plan_dir / "summary.json"
    run_checked(
        [
            sys.executable,
            str(TOOLS / "speech_like_corpus_plan.py"),
            "build",
            "--plan",
            str(corpus_plan),
            "--voice-inventory",
            str(inventory_path),
            "--requests",
            str(requests),
            "--intents",
            str(intents),
            "--summary",
            str(request_summary_path),
        ],
        logs / "build-request-plan.log",
    )
    request_summary = load_object(request_summary_path)
    if int(request_summary.get("requests", 0)) != 384:
        raise ValueError("Stage A request plan must contain exactly 384 requests")
    if int(request_summary.get("voice_slots", 0)) != 24 or int(request_summary.get("utterances", 0)) != 16:
        raise ValueError("Stage A request plan voice/utterance count drifted")
    if request_summary.get("split_counts") != expected_split_counts:
        raise ValueError("Stage A request split counts drifted")

    generated_root = work / "generated"
    generation_summary_path = work / "generation-summary.json"
    run_checked(
        [
            sys.executable,
            str(TOOLS / "generate_speech_like_command_provider.py"),
            "--policy",
            str(command_policy),
            "--provider",
            str(provider_path),
            "--requests",
            str(requests),
            "--output-root",
            str(generated_root),
            "--summary",
            str(generation_summary_path),
        ],
        logs / "generate-recordings.log",
    )
    generation_summary = load_object(generation_summary_path)
    if int(generation_summary.get("recordings", 0)) != 384:
        raise ValueError("speech-like provider did not generate all 384 recordings")

    labeled_root = work / "labeled"
    label_summary_path = work / "label-summary.json"
    run_checked(
        [
            sys.executable,
            str(TOOLS / "speech_like_corpus_plan.py"),
            "materialize",
            "--intents",
            str(intents),
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(labeled_root),
            "--summary",
            str(label_summary_path),
        ],
        logs / "materialize-labels.log",
    )
    label_summary = load_object(label_summary_path)
    if int(label_summary.get("recordings", 0)) != 384:
        raise ValueError("speech-like label materialization did not retain all 384 recordings")

    tokens, keywords = resolve_stage_inputs(stage_base_config)
    bundle_root = work / "bundle"
    split_spec: dict[str, dict] = {}
    observed_sources: set[str] = set()
    observed_voice_owner: dict[str, str] = {}
    for split in SPLITS:
        manifest = labeled_root / split / "labeled-manifest.jsonl"
        index = bundle_root / split / "dataset-index.jsonl"
        summary = bundle_root / split / "dataset-summary.json"
        run_checked(
            [
                sys.executable,
                str(TOOLS / "materialize_speech_like_base_index.py"),
                "--manifest",
                str(manifest),
                "--split",
                split,
                "--tokens",
                str(tokens),
                "--keywords",
                str(keywords),
                "--output-index",
                str(index),
                "--output-summary",
                str(summary),
            ],
            logs / f"materialize-base-{split}.log",
        )
        summary_value = load_object(summary)
        if int(summary_value.get("recordings", -1)) != expected_split_counts[split]:
            raise ValueError(f"{split}: canonical base recording count drifted")
        if summary_value.get("tone_backend_used") is not False:
            raise ValueError(f"{split}: tone backend unexpectedly present")
        for row in load_jsonl(index):
            provenance = row.get("speech_like_provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"{split}: speech-like provenance missing")
            source_id = str(provenance.get("source_id", ""))
            voice_id = str(provenance.get("voice_id", ""))
            if not source_id or source_id in observed_sources:
                raise ValueError(f"{split}: duplicate/empty source identity")
            observed_sources.add(source_id)
            previous = observed_voice_owner.get(voice_id)
            if previous is not None and previous != split:
                raise ValueError(f"cross-split voice overlap: {voice_id} in {previous}/{split}")
            observed_voice_owner[voice_id] = split
        split_spec[split] = {
            "index": index,
            "summary": summary,
            "index_sha256": sha256_file(index),
            "summary_sha256": sha256_file(summary),
            "corpus_sha256": str(summary_value["corpus_sha256"]),
            "recordings": int(summary_value["recordings"]),
        }

    if len(observed_sources) != 384 or len(observed_voice_owner) != 24:
        raise ValueError("final Stage A corpus source/voice identity count drifted")

    validation_dir = work / "validation"
    validation_config = validation_dir / "effective-stage-base.json"
    validation_dir.mkdir(parents=True, exist_ok=True)
    effective = json.loads(json.dumps(load_object(stage_base_config)))
    external = {
        split: {
            "index": relative_ref(split_spec[split]["index"], validation_dir),
            "summary": relative_ref(split_spec[split]["summary"], validation_dir),
            "index_sha256": split_spec[split]["index_sha256"],
            "summary_sha256": split_spec[split]["summary_sha256"],
        }
        for split in SPLITS
    }
    generator = effective.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("Stage A base config generator is missing")
    if "external_base_dataset" in generator:
        raise ValueError("Stage A base config already declares external_base_dataset")
    generator["external_base_dataset"] = external
    validation_config.write_text(
        json.dumps(effective, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    external_summary_path = validation_dir / "external-base-summary.json"
    run_checked(
        [
            sys.executable,
            str(TRAINING / "external_base_dataset.py"),
            "--config",
            str(validation_config),
            "--output",
            str(external_summary_path),
        ],
        logs / "validate-external-base.log",
    )
    external_summary = load_object(external_summary_path)
    if int(external_summary.get("recordings", 0)) != 384:
        raise ValueError("external base validator did not retain all 384 recordings")
    if external_summary.get("tone_backend_used") is not False:
        raise ValueError("external base validator reported tone backend")

    manifest_body = {
        "schema_version": 1,
        "evidence_class": BUNDLE_CLASS,
        "evidence_scope": "development-only",
        "provider_candidate": candidate_name,
        "provider_reference_sha256": sha256_file(reference_path),
        "provider_identity_sha256": str(provider_summary["provider_identity_sha256"]),
        "license_id": license_id,
        "license_evidence_sha256": sha256_file(license_evidence),
        "runtime_archive_sha256": sha256_file(runtime_archive),
        "runtime_receipt_sha256": sha256_file(runtime_receipt),
        "speaker_map_sha256": sha256_file(speaker_map),
        "corpus_plan_sha256": sha256_file(corpus_plan),
        "request_set_sha256": str(request_summary["request_set_sha256"]),
        "intent_set_sha256": str(request_summary["intent_set_sha256"]),
        "stage_base_config_sha256": sha256_file(stage_base_config),
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords),
        "recordings": 384,
        "voice_slots": 24,
        "utterances": 16,
        "external_base_bundle_sha256": str(external_summary["bundle_sha256"]),
        "source_sample_rate_hz": source_rate,
        "normalized_sample_rate_hz": int(candidate["normalized_sample_rate_hz"]),
        "source_bandwidth_limitation_retained": bool(
            candidate.get("audio_normalization", {}).get(
                "source_bandwidth_limitation_retained", False
            )
        ),
        "splits": {
            split: {
                "index": relative_ref(split_spec[split]["index"], work),
                "summary": relative_ref(split_spec[split]["summary"], work),
                "index_sha256": split_spec[split]["index_sha256"],
                "summary_sha256": split_spec[split]["summary_sha256"],
                "corpus_sha256": split_spec[split]["corpus_sha256"],
                "recordings": split_spec[split]["recordings"],
            }
            for split in SPLITS
        },
        "tone_backend_used": False,
        "protected_evidence_used": False,
    }
    manifest = dict(manifest_body)
    manifest["bundle_manifest_sha256"] = canonical_sha256(manifest_body)
    bundle_manifest = work / "stage-a-base-bundle.json"
    bundle_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"speech-like Stage A corpus: recordings=384 voices=24 "
        f"provider={candidate_name} bundle={manifest['external_base_bundle_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
