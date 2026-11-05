"""Measured input identity and saved-input evidence for system comparisons.

The common runner context is an explicit declaration, not execution attestation.
System comparison additionally rechecks the WAV/PCM bytes, references, saved
corpus provenance, detections and scores. Matching caller-chosen IDs is not enough.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import shutil
import sys
import tempfile
import wave
from collections.abc import Callable

from corpus_identity import (
    canonical_audio_path, canonical_hash, corpus_digest, inspect_pcm16_wav,
    sha256_file, verified_afe_metadata,
)
from dataset_iteration_policy import DATASET_IDENTITY_POLICY
from export_artifacts import paths_alias

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from domain_metrics import domain_keys  # noqa: E402
from score_events import (  # noqa: E402
    DEFAULT_POST_TOLERANCE_MS, DEFAULT_PRE_TOLERANCE_MS,
    load_jsonl as load_events, score, validate_detections, validate_recordings,
)

SYSTEM_INPUT_POLICY = "verified-shared-recording-inputs-v1"
CONTEXT_FIELDS = {
    "sample_rate_hz", "feed_chunk_samples", "reset", "appended_context_samples",
    "eof", "event_time", "keyword_map",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("system comparison: " + message)


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from None
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def checked_metadata(row: dict, audio_root: pathlib.Path,
                     relocate: Callable | None = None) -> tuple[pathlib.Path | None, str | None]:
    """Relocate only path lookup; reference bytes and identity remain unchanged."""
    if relocate is None:
        return verified_afe_metadata(row, audio_root)
    raw = row.get("metadata_path")
    if isinstance(raw, str) and raw:
        path = pathlib.Path(raw)
        row = dict(row, metadata_path=str(relocate(path if path.is_absolute() else audio_root / path)))
    return verified_afe_metadata(row, relocate(audio_root))


def dataset_input_paths(rows: list[dict], audio_root: pathlib.Path,
                        relocate: Callable | None = None) -> list[pathlib.Path]:
    """Resolve audio/AFE inputs before an output can overwrite any of them."""
    paths = []
    for row in rows:
        path = pathlib.Path(canonical_audio_path(row))
        audio = path if path.is_absolute() else audio_root / path
        if relocate is not None:
            audio = relocate(audio)
        if not audio.is_file():
            raise ValueError(f"{row.get('recording')}: audio file is missing: {audio}")
        paths.append(audio)
        metadata, _ = checked_metadata(row, audio_root, relocate)
        if metadata is not None:
            paths.append(metadata)
    return paths


def validate_output_paths(outputs: dict[str, pathlib.Path], inputs: list[pathlib.Path]) -> None:
    """Stable-filesystem alias preflight; callers retain concurrent-write ownership."""
    previous = []
    try:
        for name, output in outputs.items():
            if any(paths_alias(output, path) for path in inputs):
                raise ValueError(f"output {name} aliases an input")
            if any(paths_alias(output, path) for path in previous):
                raise ValueError(f"output {name} aliases another output")
            previous.append(output)
    except RuntimeError as exc:
        raise ValueError(f"cannot resolve iteration output path: {exc}") from exc


def evidence_relocation(evidence: dict, evidence_root: pathlib.Path | None = None) -> Callable:
    origin = evidence.get("origin_root")
    require(type(origin) is str and pathlib.Path(origin).is_absolute(), "evidence origin root is missing")
    origin_root = pathlib.Path(origin)
    target_root = origin_root if evidence_root is None else evidence_root.resolve(strict=True)
    require(target_root.is_dir(), "evidence root is unavailable")

    def relocate(path: pathlib.Path) -> pathlib.Path:
        require(path.is_absolute(), "evidence path must be absolute")
        # Normalize lexically, without resolving an old symlink after relocation.
        normalized = pathlib.Path(os.path.normpath(path))
        try:
            relative = normalized.relative_to(origin_root)
        except ValueError:
            raise ValueError("system comparison: evidence path is outside its origin root") from None
        return target_root / relative

    return relocate


def system_input_paths(card: dict, evidence_root: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Protect the evidence that has already passed verify_system_evidence."""
    evidence = card["system_evaluation"]
    relocate = evidence_relocation(evidence, evidence_root)
    files = {name: relocate(pathlib.Path(binding["path"])) for name, binding in evidence["files"].items()}
    return [*files.values(), *dataset_input_paths(
        load_jsonl(files["references"]), pathlib.Path(evidence["audio_root"]), relocate)]


def adjacent_evidence_root(card_path: pathlib.Path, card: dict) -> pathlib.Path | None:
    """A restored scorecard can find its sibling bundle without changing bytes."""
    run_id = card.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{16}", run_id) is None:
        return None
    path = card_path.parent / "evidence" / run_id
    return path if path.exists() or path.is_symlink() else None


def bundle_system_evidence(card: dict, destination: pathlib.Path) -> None:
    """Copy only replay inputs, publish after full readback, retain existing bytes.

    WAV/AFE aliases become regular files with the same content at their recorded
    relative paths. No model weights or runner binary are needed for replay.
    This is bounded pre/post integrity checking, not a concurrent-writer lock.
    """
    verify_system_evidence(card, "bundle source")
    if destination.exists() or destination.is_symlink():
        verify_system_evidence(card, "existing bundle", evidence_root=destination)
        return
    sources = system_input_paths(card)
    origin = pathlib.Path(card["system_evaluation"]["origin_root"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        staging = pathlib.Path(temporary)
        for source in dict.fromkeys(sources):
            relative = pathlib.Path(os.path.normpath(source)).relative_to(origin)
            output = staging / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            digest = sha256_file(source)
            shutil.copyfile(source, output)
            require(sha256_file(source) == digest == sha256_file(output),
                    "input changed while saving evidence: " + str(source))
        verify_system_evidence(card, "saved bundle", evidence_root=staging)
        staging.rename(destination)


def dataset_identity(references: pathlib.Path, audio_root: pathlib.Path, *,
                     relocate: Callable | None = None) -> tuple[str, dict]:
    """Keep the schema-4 effective-domain/verified-AFE identity unchanged."""
    entries = []
    for row in load_jsonl(references):
        name = str(row.get("recording", ""))
        if not name:
            raise ValueError("reference row is missing 'recording'")
        audio = pathlib.Path(canonical_audio_path(row, name))
        if not audio.is_absolute():
            audio = audio_root / audio
        if relocate is not None:
            audio = relocate(audio)
        if not audio.is_file():
            raise ValueError(f"{name}: audio file is missing: {audio}")
        metadata, digest = checked_metadata(row, audio_root, relocate)
        with wave.open(str(audio), "rb") as reader:
            duration = reader.getnframes() / float(reader.getframerate())
        entries.append({
            "recording": name, "wav_sha256": sha256_file(audio),
            "duration_s": round(duration, 6), "expected": row.get("expected", []),
            "domains": sorted(domain_keys(row)),
            "afe_metadata": ({"present": False} if metadata is None else
                             {"present": True, "sha256": digest}),
        })
    entries.sort(key=lambda item: item["recording"])
    return canonical_hash({"dataset_identity_policy": DATASET_IDENTITY_POLICY,
                           "entries": entries}), {"recordings": len(entries)}


def validate_runner_context(value: object) -> dict:
    require(type(value) is dict and set(value) == CONTEXT_FIELDS,
            "runner context fields are incomplete or unknown")
    require(type(value["sample_rate_hz"]) is int and value["sample_rate_hz"] == 16000,
            "runner context requires 16000 Hz")
    require(type(value["feed_chunk_samples"]) is int and
            1 <= value["feed_chunk_samples"] <= 16000, "invalid feed chunk size")
    require(type(value["appended_context_samples"]) is int and
            value["appended_context_samples"] == 0, "appended input is not supported")
    require(value["reset"] == "per-recording" and value["eof"] == "no-extra-audio" and
            value["event_time"] == "input-wav-sample-seconds", "unsupported runner context")
    keywords = value["keyword_map"]
    require(type(keywords) is dict and bool(keywords), "keyword map must be nonempty")
    require(all(type(key) is str and re.fullmatch(r"0|[1-9][0-9]*", key) and
                int(key) <= 0xFFFFFFFF and type(text) is str and text.strip()
                for key, text in keywords.items()), "invalid keyword map")
    return value


def scoring_protocol(context: dict) -> dict:
    return {
        "policy": "shared-input-event-scoring-v1",
        "runner_context": validate_runner_context(context),
        "event_match_pre_tolerance_ms": DEFAULT_PRE_TOLERANCE_MS,
        "event_match_post_tolerance_ms": DEFAULT_POST_TOLERANCE_MS,
        "latency_coordinate": "reference-end-to-input-wav-event-sample",
        "scoring_sources": {name: sha256_file(ROOT / name) for name in (
            "eval/score_events.py", "eval/context_policy.py", "eval/domain_metrics.py",
            "tools/statistical_bounds.py",
        )},
    }


def bind_file(path: pathlib.Path) -> dict:
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def checked_file(binding: object, label: str, relocate: Callable) -> pathlib.Path:
    require(type(binding) is dict and set(binding) == {"path", "sha256"},
            label + " binding is missing")
    require(type(binding["path"]) is str and pathlib.Path(binding["path"]).is_absolute(),
            label + " path must be absolute")
    require(type(binding["sha256"]) is str and HEX64.fullmatch(binding["sha256"]) is not None,
            label + " SHA-256 is invalid")
    path = relocate(pathlib.Path(binding["path"]))
    require(path.is_file() and sha256_file(path) == binding["sha256"],
            label + " bytes changed or are unavailable")
    return path


def system_evidence(*, references: pathlib.Path, source_references: pathlib.Path,
                    detections: pathlib.Path,
                    provenance: pathlib.Path, audio_root: pathlib.Path,
                    context: dict) -> dict:
    paths = {name: path for name, path in (
        ("references", references), ("source_references", source_references),
        ("detections", detections), ("provenance", provenance)
    )}
    files = {name: bind_file(path) for name, path in paths.items()}
    root = audio_root.resolve(strict=True)
    sources = [*(pathlib.Path(binding["path"]) for binding in files.values()),
               *dataset_input_paths(load_jsonl(references), root)]
    origin_root = os.path.commonpath([str(root),
                                     *(str(pathlib.Path(os.path.abspath(path)).parent) for path in sources)])
    return {
        "policy": SYSTEM_INPUT_POLICY, "protocol": scoring_protocol(context),
        "origin_root": origin_root,
        "audio_root": str(root),
        "files": files,
    }


def verify_system_evidence(card: dict, label: str, *, evidence_root: pathlib.Path | None = None) -> dict:
    """Re-read saved evidence; never run a model or accept an ID-only cohort."""
    require(type(card.get("schema_version")) is int and card["schema_version"] == 5,
            label + " requires a schema-5 scorecard")
    identity = card["identity"]
    require(identity.get("comparison_mode") == "system" and identity.get("variable") == "system",
            label + " was not produced for system comparison")
    require(type(identity.get("native_frontend")) is str and
            bool(identity["native_frontend"].strip()), label + " native frontend is undeclared")
    require(type(identity.get("code_sha")) is str and
            re.fullmatch(r"[0-9a-f]{40}", identity["code_sha"]) is not None,
            label + " code SHA must identify a source revision")
    evidence = card.get("system_evaluation")
    require(type(evidence) is dict and set(evidence) == {
                "policy", "protocol", "origin_root", "audio_root", "files"}
            and evidence["policy"] == SYSTEM_INPUT_POLICY, label + " input evidence is missing")
    relocate = evidence_relocation(evidence, evidence_root)
    protocol = evidence["protocol"]
    require(type(protocol) is dict and
            protocol == scoring_protocol(protocol.get("runner_context")),
            label + " scoring implementation/protocol changed; regenerate both cards")
    require(identity.get("evaluation_protocol_id") == canonical_hash(protocol),
            label + " evaluation protocol binding differs")
    files = evidence["files"]
    require(type(files) is dict and set(files) == {"references", "source_references", "detections", "provenance"},
            label + " requires references, detections and corpus provenance")
    paths = {name: checked_file(binding, label + "." + name, relocate) for name, binding in files.items()}
    root = evidence["audio_root"]
    require(type(root) is str and pathlib.Path(root).is_absolute(), label + " audio root is missing")
    audio_root = pathlib.Path(root)
    dataset_id, info = dataset_identity(paths["references"], audio_root, relocate=relocate)
    require(dataset_id == identity.get("dataset_id"), label + " measured input identity differs")
    refs = load_jsonl(paths["references"])
    original = load_jsonl(paths["source_references"])
    def without_execution_alias(rows):
        return [{key: value for key, value in row.items() if key != "_execution_path"} for row in rows]
    require(without_execution_alias(refs) == without_execution_alias(original),
            label + " effective references differ from original evaluation inputs")
    recordings = validate_recordings(refs)
    expected_ids = {str(event["keyword_id"]) for item in recordings.values() for event in item["expected"]}
    require(bool(expected_ids) and expected_ids <= set(protocol["runner_context"]["keyword_map"]),
            label + " positive references are missing or outside the keyword map")
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    require(type(provenance) is dict and provenance.get("schema_version") == 2 and
            provenance.get("evaluation_mode") == "direct-runner-v1", label + " unsupported corpus provenance")
    require(provenance.get("references_sha256") == files["references"]["sha256"] and
            provenance.get("detections_sha256") == files["detections"]["sha256"] and
            provenance.get("runner_sha256") == identity.get("runner_sha256") and
            provenance.get("recordings") == info["recordings"], label + " provenance binding differs")
    for field in ("runner_sha256", "model_sha256", "keyword_pack_sha256"):
        require(type(provenance.get(field)) is str and HEX64.fullmatch(provenance[field]) is not None,
                label + " invalid provenance " + field)
    require(identity.get("model_id") == canonical_hash({
        "model_sha256": provenance["model_sha256"], "pack_sha256": provenance["keyword_pack_sha256"]
    }), label + " model tuple differs from corpus execution")
    audio_files = provenance.get("audio_files")
    require(type(audio_files) is list and len(audio_files) == len(recordings) and
            all(type(item) is dict for item in audio_files), label + " incomplete corpus audio evidence")
    by_name = {item.get("recording"): item for item in audio_files}
    require(set(by_name) == set(recordings) and len(by_name) == len(audio_files),
            label + " corpus recording coverage differs")
    require(provenance.get("audio_corpus_sha256") == corpus_digest(audio_files),
            label + " corpus evidence digest differs")
    measured_afe = {}
    for row in refs:
        name = row["recording"]
        relative = canonical_audio_path(row, name)
        audio = pathlib.Path(relative)
        measured = inspect_pcm16_wav(relocate(audio if audio.is_absolute() else audio_root / audio))
        require(type(row.get("duration_s")) in (int, float) and math.isclose(
                    row["duration_s"], measured["duration_s"], rel_tol=0.0,
                    abs_tol=1.0 / protocol["runner_context"]["sample_rate_hz"]),
                label + ": " + name + " reference duration_s does not match WAV duration")
        recorded = by_name[name]
        require(recorded.get("path") == relative and
                all(recorded.get(key) == value for key, value in measured.items()),
                label + ": " + name + " consumed WAV/PCM binding differs")
        metadata, digest = checked_metadata(row, audio_root, relocate)
        if metadata is not None:
            measured_afe[name] = digest
    afe_files = provenance.get("afe_metadata_files", [])
    require(type(afe_files) is list and all(type(item) is dict for item in afe_files),
            label + " invalid AFE provenance")
    require({item.get("recording"): item.get("sha256") for item in afe_files} == measured_afe and
            len(afe_files) == len(measured_afe), label + " consumed AFE binding differs")
    rows = load_events(paths["detections"])
    require(provenance.get("detections") == len(rows), label + " detection coverage differs")
    summary, _, _ = score(recordings, validate_detections(rows, recordings),
                          DEFAULT_PRE_TOLERANCE_MS / 1000, DEFAULT_POST_TOLERANCE_MS / 1000)
    for field in ("expected", "matched", "false_rejects", "false_accepts", "frr",
                  "far_per_hour", "p95_post_end_latency_ms"):
        require(card["metrics"].get(field) == summary[field], label + " score differs from saved events: " + field)
    require(card["far_rate"].get("observed_false_accepts") == summary["negative_recording_false_accepts"] and
            card["exposure"].get("exposure_hours") == summary["negative_recording_audio_hours"],
            label + " negative population differs from saved events")
    artifacts = card.get("artifacts", {})
    require(artifacts.get("detections_sha256") == files["detections"]["sha256"] and
            artifacts.get("references_sha256") == files["source_references"]["sha256"] and
            artifacts.get("provenance_sha256") == files["provenance"]["sha256"],
            label + " artifact bindings differ")
    # Recheck after reading: these checks are not an atomic filesystem snapshot.
    for name, binding in files.items():
        checked_file(binding, label + "." + name, relocate)
    require(dataset_identity(paths["references"], audio_root, relocate=relocate)[0] == dataset_id,
            label + " input bytes changed during comparison")
    return {"protocol_id": identity["evaluation_protocol_id"], "dataset_id": dataset_id,
            "references_sha256": files["references"]["sha256"],
            "source_references_sha256": files["source_references"]["sha256"],
            "recordings": len(recordings), "expected": summary["expected"],
            "negative_hours": summary["negative_recording_audio_hours"]}
