#!/usr/bin/env python3
"""Observe frozen acoustic variants with the existing three native ASR engines.

The source WAVs and inherited labels remain unchanged. This separate view
compares full observed transcripts with retained parent text only after native
recognition. Deliberately attenuated audio has no RMS admission threshold.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from array import array
from collections import Counter
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "training")]
import active_asr_batch as batch
import active_asr_selective as selective
import research_acoustic_augment as augment
import research_d2_checkpoints as checkpoints
import research_queue as queue

SCHEMA = "kws-acoustic-asr-campaign-v2"
JOB = "kws-acoustic-asr-job-v1"
VIEW = "kws-acoustic-asr-observation-view-v1"
PREREQUISITES = "kws-acoustic-asr-prerequisites-v2"
HANDOFF = "kws-context-v3-metadata-handoff-v1"
INGESTION_PREP = "kws-context-v3-handoff-ingestion-preparation-v1"
TOTAL, CHUNK, JOBS = 124, 8, 16
MINUTES, WORKER_SECONDS, WALL_SECONDS = 160, 570, 10800
KINDS = {"onset-offset": 28, "digital-level": 28,
         "measured-rir-convolution": 28, "recorded-noise-mix": 40}
WORDS = ("你好小窝", "小窝小窝")
OWNER_ENV = "KWS_ACOUSTIC_ASR_OWNER"
INTENT_ENV = "KWS_ACOUSTIC_ASR_INTENT"


def require(ok, reason):
    queue.require(ok, reason)


def read(path):
    return queue.loads(queue.read_bytes(path, maximum=64 * 1024 ** 2))


def pin(path):
    return checkpoints.pin(path)


def checked(value):
    require(type(value) is dict and set(value) == {"path", "bytes", "sha256"} and
            pin(value["path"]) == value, "ACOUSTIC_ASR_PIN_CHANGED")
    return Path(value["path"])


def source_records():
    names = set(selective.source_identity()) | {
        "tools/research_acoustic_asr.py", "tools/research_d2_checkpoints.py",
        "tools/research_queue.py", "training/research_acoustic_augment.py",
        "configs/research/selective-asr-v2.json", "research/qwen6_asr/runtime-lock.json",
        "research/source-screen32-whisper-v1/runtime-lock.json"}
    return {name: queue.file_record(ROOT / name) for name in sorted(names)}


def decode_pcm16(body):
    """Exact original PCM16 / 32768 -> float32; no resample, trim or scaling."""
    with wave.open(io.BytesIO(body), "rb") as reader:
        require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                 reader.getcomptype()) == (1, 2, 16000, "NONE"),
                "ACOUSTIC_ASR_MONO_PCM16_16K_REQUIRED")
        frames = reader.getnframes()
        require(1 <= frames <= 60 * 16000, "ACOUSTIC_ASR_BOUNDED_AUDIO_REQUIRED")
        pcm = reader.readframes(frames)
        require(len(pcm) == 2 * frames, "ACOUSTIC_ASR_TRUNCATED_PCM")
    values = array("h")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    floats = array("f", (value / 32768.0 for value in values))
    require(floats.itemsize == 4, "ACOUSTIC_ASR_FLOAT32_REQUIRED")
    if sys.byteorder != "little":
        floats.byteswap()
    rms = math.sqrt(sum(value * value for value in values) / frames) / 32768.0
    return pcm, floats.tobytes(), {
        "samples": frames, "sample_rate_hz": 16000, "duration_seconds": frames / 16000,
        "peak_absolute": max(abs(value) for value in values) / 32768.0,
        "rms": rms, "rms_dbfs": 20 * math.log10(rms) if rms else None,
        "rms_used_as_admission_gate": False, "normalization": False, "trimming": False}


def verify_variants(root, expected_sha):
    """Authenticate the already rendered package and its original parents."""
    root = queue.safe_path(root)
    require(queue.file_record(root / "MANIFEST.json")["sha256"] == expected_sha,
            "ACOUSTIC_ASR_VARIANT_MANIFEST_CHANGED")
    manifest = read(root / "MANIFEST.json")
    require(manifest["schema"] == augment.SCHEMA and
            set(manifest["files"]) == {p.relative_to(root).as_posix() for p in
                checkpoints.regular_tree(root) if p != root / "MANIFEST.json"},
            "ACOUSTIC_ASR_VARIANT_FILE_ROSTER")
    for name, expected in manifest["files"].items():
        require(queue.file_record(root / queue.relative(name)) == expected,
                "ACOUSTIC_ASR_VARIANT_FILE_CHANGED:" + name)
    request = read(root / "request.json")
    acoustic_root, acoustic_source = augment.acoustic_sources(request["acoustics"])
    require(read(root / "acoustic-source-subset.json") == acoustic_source,
            "ACOUSTIC_ASR_ORIGINAL_ACOUSTIC_SOURCE_CHANGED")
    acoustic_assets = {item["sha256"]: item for item in acoustic_source["assets"]}
    require(len(acoustic_assets) == len(acoustic_source["assets"]), "ACOUSTIC_ASR_DUPLICATE_SOURCE_ASSET")
    originals = {tuple(row["global_source_key"]): (row, path, proof)
                 for row, path, proof in augment.checkpoint_rows(request["parents"])}
    rows = [queue.loads(line) for line in queue.read_bytes(root / "rows.jsonl").splitlines()]
    require(len(rows) == len({row["sample_id"] for row in rows}) and
            len(rows) == len({row["pcm_sha256"] for row in rows}), "ACOUSTIC_ASR_DUPLICATE_VARIANT")
    producer_sha = queue.file_record(root / "execution-source.py")["sha256"]
    require(producer_sha == queue.file_record(ROOT / "training/research_acoustic_augment.py")["sha256"],
            "ACOUSTIC_ASR_ORIGINAL_PRODUCER_CHANGED")
    references = {}
    for row in rows:
        require(queue.ID.fullmatch(row["sample_id"]) and row["kind"] in KINDS and
                row["producer_sha256"] == producer_sha and row["human_gold"] is False and
                row["new_tts_call"] is False and row["new_asr_call"] is False and
                row["label_inheritance_is_new_asr_verification"] is False and
                row["training_admitted"] is False, "ACOUSTIC_ASR_VARIANT_SCOPE")
        key = tuple(row["parent_key"])
        require(key in originals, "ACOUSTIC_ASR_PARENT_NOT_ARCHIVED")
        parent, _parent_audio, proof = originals[key]
        name = augment.digest(augment.canonical(row["parent_key"]))
        retained = read(root / "parents" / (name + ".json"))
        require(retained == {"original_row": parent, "proof": proof} and
                row["parent_proof"] == proof and row["parent_file_sha256"] == parent["file_sha256"] and
                queue.file_record(root / "parents" / (name + ".wav"))["sha256"] == parent["file_sha256"] and
                all(row[field] == parent[field] for field in
                    ("source_family_id", "speaker_id", "source_role", "split", "vocabulary_sha256")),
                "ACOUSTIC_ASR_PARENT_FAMILY_SPLIT_JOIN")
        inherited = augment.inherited_labels(parent, row["kind"] == "recorded-noise-mix")
        require(all(row[field] == value for field, value in inherited.items()),
                "ACOUSTIC_ASR_INHERITED_LABEL_CHANGED")
        identity = {field: row[field] for field in ("parent_key", "kind", "recipe", "producer_sha256")}
        require(row["sample_id"] == "aug-" + augment.digest(augment.canonical(identity))[:24],
                "ACOUSTIC_ASR_VARIANT_RECIPE_IDENTITY")
        wav = root / queue.relative(row["audio"])
        require(row["audio"] == "audio/" + row["sample_id"] + ".wav" and
                queue.file_record(wav)["sha256"] == row["file_sha256"], "ACOUSTIC_ASR_ORIGINAL_WAV_CHANGED")
        pcm, floating, measurements = decode_pcm16(queue.read_bytes(wav))
        require(queue.sha(pcm) == row["pcm_sha256"] and measurements["samples"] == row["samples"] and
                row["sample_rate_hz"] == 16000, "ACOUSTIC_ASR_ORIGINAL_PCM_CHANGED")
        asset = None
        if row["kind"] in ("measured-rir-convolution", "recorded-noise-mix"):
            asset = acoustic_assets.get(row["recipe"]["asset_sha256"])
            require(asset is not None and asset["split"] == row["split"] == row["recipe"]["asset_split"] and
                    asset["source_group"] == row["recipe"]["asset_group"] and
                    asset["kind"] == ("measured-rir" if row["kind"] == "measured-rir-convolution" else "recorded-noise"),
                    "ACOUSTIC_ASR_ACOUSTIC_ASSET_SPLIT_CHANGED")
            require(queue.file_record(root / "assets" / (asset["sha256"] + ".wav"))["sha256"] == asset["sha256"],
                    "ACOUSTIC_ASR_RETAINED_ACOUSTIC_ASSET_CHANGED")
        references[row["sample_id"]] = {
            "variant_row": row, "variant_row_sha256": queue.sha(queue.canonical(row)),
            "parent_row": parent, "parent_row_sha256": queue.sha(queue.canonical(parent)),
            "parent_retained_metadata": pin(root / "parents" / (name + ".json")),
            "parent_retained_wav": pin(root / "parents" / (name + ".wav")),
            "acoustic_source_manifest": pin(acoustic_root / "MANIFEST.json"),
            "acoustic_source_metadata": pin(root / "acoustic-source-subset.json"),
            "acoustic_asset": asset,
            "acoustic_asset_wav": pin(root / "assets" / (asset["sha256"] + ".wav")) if asset else None,
            "variant_wav": pin(wav), "original_pcm16_sha256": queue.sha(pcm),
            "decoded_float32_sha256": queue.sha(floating), "measurements": measurements}
    summary = read(root / "summary.json")
    require(summary["derived_rows"] == len(rows) and summary["by_kind"] == dict(Counter(r["kind"] for r in rows)) and
            summary["by_split"] == dict(Counter(r["split"] for r in rows)) and
            summary["additive_noise_ctc_masked_rows"] == sum(not r["ctc_mask"] for r in rows) and
            summary["model_calls"] == summary["training_steps"] == 0 and summary["human_gold"] is False,
            "ACOUSTIC_ASR_VARIANT_SUMMARY_CHANGED")
    return rows, references


def chunk_path(root, job_id):
    return Path(root) / "chunks" / job_id


def job_id(index):
    return f"observe-{index + 1:03d}"


def prepare_audio(root, selected, bindings, plan_sha):
    """Write the unchanged native worker's audio-only manifest without its
    TTS-specific 24-kHz/resampling preparer.
    """
    root.mkdir(parents=True, mode=0o700)
    inputs = root / "inputs"
    inputs.mkdir(mode=0o700)
    archive_path = root / "source-original.zip"
    clips, mapping = [], []
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED) as archive:
        for index, ident in enumerate(selected):
            binding = bindings[ident]
            row = binding["variant_row"]
            body = queue.read_bytes(checked(binding["variant_wav"]))
            pcm, floating, measured = decode_pcm16(body)
            require(queue.sha(pcm) == binding["original_pcm16_sha256"] and
                    queue.sha(floating) == binding["decoded_float32_sha256"] and
                    measured == binding["measurements"], "ACOUSTIC_ASR_COPY_PCM_CHANGED")
            clip_id = f"clip-{index:03d}"
            archive.writestr(zipfile.ZipInfo(row["audio"]), body)
            original = inputs / "original" / (clip_id + ".wav")
            derived = inputs / "pcm16k" / (clip_id + ".f32")
            original.parent.mkdir(exist_ok=True)
            derived.parent.mkdir(exist_ok=True)
            queue.write_new(original, body)
            queue.write_new(derived, floating)
            clips.append({"id": clip_id, "original_member": row["audio"],
                "original_wav": {"path": original.relative_to(inputs).as_posix(), "sha256": queue.sha(body), "bytes": len(body)},
                "original_pcm": {"sha256": queue.sha(floating), "sample_rate": 16000, "channels": 1,
                                 "frames": row["samples"], "dtype": "<f4"},
                "derived_pcm": {"path": derived.relative_to(inputs).as_posix(), "sha256": queue.sha(floating),
                    "sample_rate": 16000, "channels": 1, "frames": row["samples"], "dtype": "<f4"}})
            mapping.append({"clip_id": clip_id, "source_id": ident,
                "original_wav_sha256": queue.sha(body), "original_pcm_sha256": queue.sha(floating),
                "derived_pcm_sha256": queue.sha(floating), "quality_uncertain": True,
                "quality_reasons": ["derived-acoustics-not-yet-independently-observed", row["kind"]]})
    manifest = {"schema": "active-asr-inputs-v1",
        "source_archive_sha256": queue.file_record(archive_path)["sha256"],
        "created_at": queue.utc(), "preparation_source_sha256": queue.file_record(Path(__file__))["sha256"],
        "derivation": {"source": "original mono PCM16 16000Hz WAV, exact bytes retained",
            "operation": "signed-int16 / 32768 exactly represented as little-endian float32",
            "resampling": False, "clipping": False, "normalization": False, "trimming": False,
            "rms_used_as_admission_gate": False, "output": "mono16000Hz little-endian float32 PCM"},
        "clips": clips}
    queue.write_json(inputs / "inputs.json", manifest)
    queue.write_json(root / "source-map.json", {"schema": "active-asr-source-map-v2",
        "input_manifest_sha256": queue.file_record(inputs / "inputs.json")["sha256"],
        "source_archive_sha256": manifest["source_archive_sha256"], "plan_sha256": plan_sha, "rows": mapping})
    batch.verify_inputs(inputs / "inputs.json")
    selective.validate_map(inputs / "inputs.json", root / "source-map.json")
    return {"inputs": pin(inputs / "inputs.json"), "source_map": pin(root / "source-map.json"),
            "archive": pin(archive_path)}


def expected_recipe(stage_path, stage, index):
    return {"schema": JOB, "stage": pin(stage_path), "job_id": job_id(index), "chunk_index": index,
        "sample_ids": stage["chunks"][index], "asr_request": stage["asr_requests"][index],
        "maximum_native_calls": len(stage["chunks"][index]) * 3, "worker_seconds": WORKER_SECONDS,
        "automatic_retries": 0, "new_tts_calls": 0, "new_optimizer_steps": 0}


def queue_definition(stage_path, stage):
    root = Path(stage["root"])
    jobs = []
    for index, selected in enumerate(stage["chunks"]):
        ident = job_id(index)
        # The unchanged native controller authenticates the queue recipe hash.
        # Therefore the queue carries its exact request, while the separate
        # owner recipe binds that request to this stage and chunk.
        recipe = Path(stage["asr_requests"][index]["path"])
        jobs.append({"job_id": ident, "recipe": str(recipe), "recipe_sha256": queue.file_record(recipe)["sha256"],
            "argv": [stage["asr_python"], "-B", str(ROOT / "tools/research_acoustic_asr.py"),
                     "run-job", "--recipe", "{recipe}", "--stage", str(stage_path),
                     "--chunk-index", str(index)],
            "cwd": str(ROOT), "depends_on": [] if index == 0 else [job_id(index - 1)],
            "timeout_minutes": 10, "outputs": ["asr/audit/final-summary.json", "asr/audit/file-hashes.json"]})
    return {"schema": queue.SCHEMA, "queue_id": "acoustic-derived124-asr",
            "global_budget_minutes": MINUTES, "jobs": jobs}


def register_declaration(stage_path, stage, plan):
    registration = Path(stage["declaration_registration"])
    with queue.exclusive(Path(stage["runtime"]) / "acoustic-asr-declaration.lock"):
        require(not registration.exists(), "ACOUSTIC_ASR_ORIGINAL124_ALREADY_DECLARED")
        registration.parent.mkdir(exist_ok=True)
        queue.write_json(registration, {"schema": "kws-acoustic-asr-declaration-v1",
            "stage": pin(stage_path), "variant_manifest": plan["variant_manifest"],
            "maximum_asr_calls": TOTAL * 3, "automatic_retries": 0, "human_gold": False})


def handoff_binding(campaign_pin, original_validator_pin, handoff_pin, validator_pin, heavy_lock):
    campaign = read(checked(campaign_pin))
    value = read(checked(handoff_pin))
    validator = checked(validator_pin)
    checked(original_validator_pin)
    require(value["schema"] == HANDOFF and Path(handoff_pin["path"]) == Path(value["root"]) / "handoff.json" and
            value["parent_campaign"] == campaign_pin and value["original_validator"] == original_validator_pin and
            value["original_stage"] == campaign["stage"] and value["driver_runtime"] == campaign["driver_runtime"] and
            value["heavy_lock"] == heavy_lock == str(Path(campaign["asr_runtime"]) / "heavy-model.lock") and
            validator.name == "research_context_v3_metadata_handoff.py" and validator.parent.name == "tools" and
            value["pipeline_files"]["tools/research_context_v3_metadata_handoff.py"] == queue.file_record(validator) and
            value["original_completed_chunks"] == [f"generate-{n:03d}" for n in range(1, 5)] and
            value["continuation_chunks"] == [f"generate-{n:03d}" for n in range(5, 10)] and
            value["read_only_generation_jobs"] == ["generate-005"] and
            value["new_generation_jobs"] == [f"generate-{n:03d}" for n in range(6, 10)] and
            value["automatic_retries"] == 0 and value["model_concurrency"] == 1,
            "ACOUSTIC_ASR_EXACT_CONTEXT_HANDOFF_REQUIRED")
    return value


def prepare(args):
    root = queue.safe_path(args.output)
    require(not root.exists() and root.parent.is_dir(), "ACOUSTIC_ASR_CREATE_ONLY_PREPARATION")
    rows, bindings = verify_variants(args.variants, args.variants_manifest_sha256)
    require(len(rows) == TOTAL and Counter(r["kind"] for r in rows) == KINDS and
            Counter(r["split"] for r in rows) == {"train": 100, "development": 24} and
            sum(r["kind"] == "recorded-noise-mix" and not r["ctc_mask"] for r in rows) == 40,
            "ACOUSTIC_ASR_EXACT124_DECLARATION")
    runtime = queue.safe_path(args.runtime)
    require(read(runtime / "setup.json")["status"] == "complete", "ACOUSTIC_ASR_RUNTIME_INCOMPLETE")
    python = runtime / "venv/bin/python"
    require(python.is_file() and os.access(python, os.X_OK), "ACOUSTIC_ASR_PYTHON_MISSING")
    context, selection = read(args.context_campaign), read(args.ingestion_selection)
    require(context["schema"] == "kws-context-v3-campaign-v1" and
            context["plan_sha256"] == selection["plan_canonical_sha256"] and
            selection["schema"] == "kws-context-v3-ingestion-selection-v1",
            "ACOUSTIC_ASR_PRECEDING_PHASE_IDENTITIES")
    require(runtime == Path(context["asr_runtime"]), "ACOUSTIC_ASR_CANONICAL_SHARED_RUNTIME_REQUIRED")
    registration = runtime / "acoustic-asr-registrations" / (args.variants_manifest_sha256 + ".json")
    require(not registration.exists(), "ACOUSTIC_ASR_ORIGINAL124_ALREADY_DECLARED")
    context_validator = queue.safe_path(args.context_validator)
    require(context["pipeline_files"]["tools/research_context_v3_campaign.py"] ==
            queue.file_record(context_validator), "ACOUSTIC_ASR_ORIGINAL_CONTEXT_VALIDATOR")
    ingestion_validator = queue.safe_path(args.ingestion_validator)
    require(ingestion_validator.name == "research_context_v3_consume.py" and ingestion_validator.parent.name == "training",
            "ACOUSTIC_ASR_PREDECLARED_INGESTION_VALIDATOR_REQUIRED")
    handoff_binding(pin(args.context_campaign), pin(context_validator), pin(args.context_handoff),
                    pin(args.handoff_validator), str(runtime / "heavy-model.lock"))
    root.mkdir(mode=0o700)
    code = source_records()
    for name in code:
        target = root / "execution-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        queue.write_new(target, queue.read_bytes(ROOT / name))
    metadata = {}
    for source, name in ((Path(args.variants) / "MANIFEST.json", "variant-manifest.json"),
                         (Path(args.variants) / "rows.jsonl", "variant-rows.jsonl"),
                         (Path(args.variants) / "summary.json", "variant-summary.json"),
                         (Path(args.variants) / "request.json", "variant-request.json"),
                         (runtime / "setup.json", "asr-setup.json")):
        queue.write_new(root / name, queue.read_bytes(source))
        metadata[name] = pin(root / name)
    plan = {"schema": SCHEMA, "variant_manifest": pin(Path(args.variants) / "MANIFEST.json"),
        "variant_rows": pin(Path(args.variants) / "rows.jsonl"), "ordered_sample_ids": [r["sample_id"] for r in rows],
        "bindings": bindings, "all_requested_engines": list(batch.MODELS),
        "historical_labels_are_unchanged": True, "noise_ctc_masked_rows_unchanged": 40,
        "full_transcript_comparison_occurs_after_asr": True, "human_gold": False}
    queue.write_json(root / "plan.json", plan)
    plan_sha = queue.file_record(root / "plan.json")["sha256"]
    selected = plan["ordered_sample_ids"]
    chunks = [selected[n:n + CHUNK] for n in range(0, len(selected), CHUNK)]
    audio, requests = [], []
    for index, ids in enumerate(chunks):
        folder = chunk_path(root, job_id(index))
        audio.append(prepare_audio(folder, ids, bindings, plan_sha))
        request = folder / "asr-request.json"
        selective.declare(Namespace(runtime=runtime, inputs=Path(audio[-1]["inputs"]["path"]),
            source_map=Path(audio[-1]["source_map"]["path"]), output=root / "queue/jobs" / job_id(index) / "outputs/asr",
            request=request, policy=selective.POLICY_PATH, threads=4, budget_seconds=WORKER_SECONDS))
        requests.append(pin(request))
    stage = {"schema": SCHEMA, "root": str(root), "created_at": queue.utc(), "plan": pin(root / "plan.json"),
        "source_hashes": code, "runtime": str(runtime), "asr_python": str(python),
        "variant_metadata": metadata, "declaration_registration": str(registration),
        "runtime_setup": pin(root / "asr-setup.json"), "live_runtime_setup": pin(runtime / "setup.json"),
        "heavy_lock": str(runtime / "heavy-model.lock"), "chunks": chunks, "audio_inputs": audio,
        "asr_requests": requests, "queue_minutes": MINUTES, "job_minutes": 10,
        "worker_seconds": WORKER_SECONDS, "driver_wall_seconds": WALL_SECONDS,
        "maximum_asr_calls": TOTAL * 3, "maximum_tts_calls": 0, "maximum_optimizer_steps": 0,
        "automatic_retries": 0, "model_concurrency": 1, "human_gold": False,
        "context_campaign": pin(args.context_campaign), "context_validator": pin(args.context_validator),
        "context_handoff": pin(args.context_handoff), "handoff_validator": pin(args.handoff_validator),
        "ingestion_selection": pin(args.ingestion_selection),
        "ingestion_validator": pin(ingestion_validator),
        "prerequisites_required_before_any_claim": True, "rms_used_as_admission_gate": False,
        "frozen_variant_rows_changed": False, "noise_fulltext_ctc_masks_changed": False}
    queue.write_json(root / "stage.json", stage)
    for index in range(JOBS):
        path = root / "recipes" / (job_id(index) + ".json")
        path.parent.mkdir(exist_ok=True)
        queue.write_json(path, expected_recipe(root / "stage.json", stage, index))
    request = queue_definition(root / "stage.json", stage)
    queue.write_json(root / "queue-request.json", request)
    queue.create(root / "queue", root / "queue-request.json", args.active_plan, runtime / "heavy-model.lock")
    register_declaration(root / "stage.json", stage, plan)
    return {"stage": str(root / "stage.json"), "stage_sha256": queue.file_record(root / "stage.json")["sha256"],
            "chunks": JOBS, "variant_rows": TOTAL, "maximum_asr_calls": TOTAL * 3,
            "new_model_calls": 0, "prerequisite_binding_pending": True}


def load_stage(path):
    path = queue.safe_path(path)
    stage = read(path)
    root = Path(stage["root"])
    require(stage["schema"] == SCHEMA and path == root / "stage.json" and
            stage["queue_minutes"] == MINUTES and stage["job_minutes"] == 10 and
            stage["worker_seconds"] == WORKER_SECONDS and stage["driver_wall_seconds"] == WALL_SECONDS and
            stage["maximum_asr_calls"] == TOTAL * 3 and
            stage["maximum_tts_calls"] == stage["maximum_optimizer_steps"] == stage["automatic_retries"] == 0 and
            stage["model_concurrency"] == 1 and stage["human_gold"] is False and
            stage["prerequisites_required_before_any_claim"] is True and
            stage["rms_used_as_admission_gate"] is False and stage["frozen_variant_rows_changed"] is False and
            stage["noise_fulltext_ctc_masks_changed"] is False and stage["source_hashes"] == source_records() and
            stage["heavy_lock"] == str(Path(stage["runtime"]) / "heavy-model.lock"),
            "ACOUSTIC_ASR_FROZEN_STAGE_CHANGED")
    for key in ("plan", "runtime_setup", "live_runtime_setup", "context_campaign",
                "context_validator", "context_handoff", "handoff_validator", "ingestion_selection", "ingestion_validator"):
        checked(stage[key])
    handoff_binding(stage["context_campaign"], stage["context_validator"], stage["context_handoff"],
                    stage["handoff_validator"], stage["heavy_lock"])
    require(read(stage["runtime_setup"]["path"]) == read(stage["live_runtime_setup"]["path"]),
            "ACOUSTIC_ASR_RUNTIME_SETUP_CHANGED")
    require(stage["heavy_lock"] == str(Path(read(stage["context_campaign"]["path"])["asr_runtime"]) / "heavy-model.lock"),
            "ACOUSTIC_ASR_CANONICAL_SHARED_RUNTIME_REQUIRED")
    require(set(stage["variant_metadata"]) == {"variant-manifest.json", "variant-rows.jsonl",
            "variant-summary.json", "variant-request.json", "asr-setup.json"}, "ACOUSTIC_ASR_METADATA_ROSTER_CHANGED")
    for name, value in stage["variant_metadata"].items():
        require(checked(value) == root / name, "ACOUSTIC_ASR_METADATA_COPY_CHANGED")
    for name, expected in stage["source_hashes"].items():
        require(queue.file_record(root / "execution-source" / name) == expected, "ACOUSTIC_ASR_SOURCE_SNAPSHOT_CHANGED")
    plan = read(stage["plan"]["path"])
    require(queue.file_record(root / "variant-manifest.json") == {k: plan["variant_manifest"][k] for k in ("bytes", "sha256")} and
            queue.file_record(root / "variant-rows.jsonl") == {k: plan["variant_rows"][k] for k in ("bytes", "sha256")},
            "ACOUSTIC_ASR_ORIGINAL_METADATA_BINDING_CHANGED")
    registration = Path(stage["runtime"]) / "acoustic-asr-registrations" / (plan["variant_manifest"]["sha256"] + ".json")
    require(stage["declaration_registration"] == str(registration) and read(registration) == {
            "schema": "kws-acoustic-asr-declaration-v1", "stage": pin(path), "variant_manifest": plan["variant_manifest"],
            "maximum_asr_calls": TOTAL * 3, "automatic_retries": 0, "human_gold": False},
            "ACOUSTIC_ASR_UNIQUE_DECLARATION_CHANGED")
    ids = plan["ordered_sample_ids"]
    require(len(ids) == len(set(ids)) == TOTAL and set(ids) == set(plan["bindings"]) and
            stage["chunks"] == [ids[n:n + CHUNK] for n in range(0, TOTAL, CHUNK)] and
            len(stage["chunks"]) == JOBS and len(stage["audio_inputs"]) == len(stage["asr_requests"]) == JOBS,
            "ACOUSTIC_ASR_FULL124_SCOPE_CHANGED")
    for index in range(JOBS):
        for value in stage["audio_inputs"][index].values():
            checked(value)
        checked(stage["asr_requests"][index])
        inputs, mapping = selective.validate_map(stage["audio_inputs"][index]["inputs"]["path"],
                                                stage["audio_inputs"][index]["source_map"]["path"])
        require([r["source_id"] for r in mapping["rows"]] == stage["chunks"][index] and
                all(r["quality_uncertain"] is True for r in mapping["rows"]) and
                mapping["plan_sha256"] == stage["plan"]["sha256"] and
                all(set(row) == {"id", "original_member", "original_wav", "original_pcm", "derived_pcm"}
                    for row in inputs["clips"]), "ACOUSTIC_ASR_AUDIO_ONLY_SCOPE_CHANGED")
        for clip, ident in zip(inputs["clips"], stage["chunks"][index]):
            binding = plan["bindings"][ident]
            raw = queue.read_bytes(Path(stage["audio_inputs"][index]["inputs"]["path"]).parent / clip["original_wav"]["path"])
            pcm, floating, measured = decode_pcm16(raw)
            require(queue.sha(raw) == binding["variant_wav"]["sha256"] and
                    queue.sha(pcm) == binding["original_pcm16_sha256"] and
                    queue.sha(floating) == clip["derived_pcm"]["sha256"] == binding["decoded_float32_sha256"] and
                    measured == binding["measurements"], "ACOUSTIC_ASR_EXACT_PCM_REPLAY_CHANGED")
    qroot, metadata, request, qsha = queue.load_queue(root / "queue")
    require(request == queue_definition(path, stage) and metadata["heavy_lock"] == stage["heavy_lock"],
            "ACOUSTIC_ASR_QUEUE_CHANGED")
    for index in range(JOBS):
        require(read(root / "recipes" / (job_id(index) + ".json")) == expected_recipe(path, stage, index) and
                queue.file_record(qroot / "recipes" / (job_id(index) + ".json")) ==
                    {k: stage["asr_requests"][index][k] for k in ("bytes", "sha256")},
                "ACOUSTIC_ASR_JOB_RECIPE_CHANGED")
    return stage, plan, request, qsha


def phase_metadata(folder):
    checkpoints.verify_ack(folder)
    return {name: pin(Path(folder) / name) for name in ("capsule.json", "ack.json", "provider-save.json", "phase-marker.json")}


def prerequisite_readback(stage, ingestion_stage_path, ingestion_validator):
    context = read(stage["context_campaign"]["path"])
    handoff = handoff_binding(stage["context_campaign"], stage["context_validator"], stage["context_handoff"],
                              stage["handoff_validator"], stage["heavy_lock"])
    ingestion_stage_path, ingestion_validator = queue.safe_path(ingestion_stage_path), queue.safe_path(ingestion_validator)
    require(pin(ingestion_validator) == stage["ingestion_validator"], "ACOUSTIC_ASR_PREDECLARED_INGESTION_SOURCE_CHANGED")
    ingestion = read(ingestion_stage_path)
    require(ingestion["schema"] == "kws-context-v3-ingestion-stage-v1" and
            Path(ingestion["root"]) / "stage.json" == ingestion_stage_path and
            ingestion["heavy_lock"] == stage["heavy_lock"] and
            ingestion["source_hashes"]["training/research_context_v3_consume.py"] ==
                queue.file_record(ingestion_validator)["sha256"], "ACOUSTIC_ASR_INGESTION_OWNER_CHANGED")
    require(ingestion_validator.name == "research_context_v3_consume.py" and ingestion_validator.parent.name == "training" and
            ingestion["automatic_retries"] == 0 and ingestion["worker_seconds"] == 120 and
            ingestion["queue_minutes"] == 10 and ingestion["pending_intent_restart_allowed"] is False,
            "ACOUSTIC_ASR_INGESTION_SCOPE_CHANGED")
    for name, expected in ingestion["source_hashes"].items():
        require(queue.file_record(ingestion_validator.parents[1] / queue.relative(name))["sha256"] == expected,
                "ACOUSTIC_ASR_INGESTION_SOURCE_CHANGED")
    for name in ("queue", "request", "active_plan", "recipe", "original_recipe"):
        checked(ingestion[name])
    recipe = read(ingestion["recipe"]["path"])
    require(recipe["schema"] == "kws-context-v3-ingestion-job-v1" and
            recipe["selection_sha256"] == stage["ingestion_selection"]["sha256"] and
            recipe["optimizer_steps"] == 25, "ACOUSTIC_ASR_INGESTION_SELECTION_CHANGED")
    prepared = queue.safe_path(recipe["prepared"])
    prepared_manifest = pin(prepared / "MANIFEST.json")
    require(prepared_manifest["sha256"] == recipe["prepared_manifest_sha256"],
            "ACOUSTIC_ASR_INGESTION_PREPARATION_MANIFEST_CHANGED")
    members = read(prepared_manifest["path"])
    preparation_files = {name: pin(prepared / name) for name in ("request.json", "campaign-closure.json")}
    require(all(members[name] == {k: item[k] for k in ("bytes", "sha256")} for name, item in preparation_files.items()),
            "ACOUSTIC_ASR_INGESTION_PREPARATION_METADATA_CHANGED")
    preparation_request = read(preparation_files["request.json"]["path"])
    require(preparation_request["schema"] == INGESTION_PREP and
            all(preparation_request[source] == stage[target] for source, target in
                (("campaign", "context_campaign"), ("campaign_validator", "context_validator"),
                 ("handoff", "context_handoff"), ("handoff_validator", "handoff_validator"),
                 ("selection", "ingestion_selection"))), "ACOUSTIC_ASR_INGESTION_HANDOFF_CHANGED")
    # The exact frozen consumer authenticates its complete source/runtime/queue
    # and replays the original-four/new-five closure in a fresh process. This
    # status-only path constructs no model or optimizer and makes no new claim.
    verifier_code = ("import sys; from pathlib import Path; "
        "sys.path.insert(0, str(Path(sys.argv[1]).parent)); "
        "import research_context_v3_consume as owner; "
        "owner.load_stage(Path(sys.argv[2])); "
        "import json; request=json.loads(Path(sys.argv[3]).read_bytes()); "
        "owner.validate_request(request); print(json.dumps(owner.handoff_closure(request)))")
    verified = subprocess.run([ingestion["runtime"]["executable"], "-I", "-B", "-c", verifier_code,
        str(ingestion_validator), str(ingestion_stage_path), preparation_files["request.json"]["path"]],
        capture_output=True, timeout=180)
    require(verified.returncode == 0 and len(verified.stdout) <= 8 * 1024 ** 2,
            "ACOUSTIC_ASR_MIXED_INGESTION_CLOSURE_VERIFICATION_FAILED")
    for value in (prepared_manifest, *preparation_files.values()):
        checked(value)
    closure = queue.loads(verified.stdout)
    require(closure == read(preparation_files["campaign-closure.json"]["path"]),
            "ACOUSTIC_ASR_PREPARED_CONTEXT_CLOSURE_CHANGED")
    status, closures = closure["status"], closure["closures"]
    require(closure["campaign"] == stage["context_campaign"] and closure["handoff"] == stage["context_handoff"] and
            closure["handoff_validator"] == stage["handoff_validator"] and closure["new_model_calls"] == 0 and
            status["schema"] == HANDOFF and status["handoff"] == stage["context_handoff"] and
            status["completed_chunks"] == [f"generate-{n:03d}" for n in range(1, 10)] and
            status["driver_active"] is False and not status["budget"]["failed_jobs"] and
            not status["budget"]["unresolved_jobs"] and status["budget"]["actual_total_is_complete"] is True and
            status["budget"]["overrun_nanoseconds"] == status["budget"]["new_overrun_nanoseconds"] == 0 and
            [item["job_id"] for item in closures] == status["completed_chunks"] and
            [item["reader_revision"] for item in closures] == ["context-v3-r1"] * 4 + ["context-v3-metadata-r2"] * 5,
            "ACOUSTIC_ASR_CONTEXT_HANDOFF_NOT_CLOSED")
    qroot, queue_metadata, request, qsha = queue.load_queue(Path(ingestion["root"]) / "queue")
    require(len(request["jobs"]) == 1 and request["jobs"][0]["job_id"] == "context-v3-ingestion" and
            ingestion["queue"] == pin(qroot / "queue.json") and
            ingestion["request"] == pin(qroot / "request.json") and
            ingestion["active_plan"] == pin(qroot / "active-plan.json") and
            ingestion["recipe"] == pin(qroot / "recipes/context-v3-ingestion.json") and
            request["global_budget_minutes"] == request["jobs"][0]["timeout_minutes"] == 10 and
            queue_metadata["heavy_lock"] == stage["heavy_lock"],
            "ACOUSTIC_ASR_INGESTION_QUEUE_SCOPE")
    terminal = queue.result_for(qroot, request["jobs"][0], qsha)
    require(terminal["status"] in ("completed", "failed", "timed_out", "unknown"),
            "ACOUSTIC_ASR_INGESTION_STILL_ACTIVE")
    terminal_folder = Path(ingestion["root"]) / "checkpoints/training-terminal"
    terminal_metadata = phase_metadata(terminal_folder)
    scope = read(terminal_folder / "capsule.json")["context"]
    terminal_pin = pin(qroot / "jobs/context-v3-ingestion/result.json")
    require(scope["stage"] == pin(ingestion_stage_path) and scope["recipe"] == ingestion["recipe"] and
            scope["queue"] == ingestion["queue"] and scope["phase"] == "training-terminal" and
            scope["terminal_state"] == terminal["status"] and scope["queue_result"] == terminal_pin and
            scope["launch_intent"] is False, "ACOUSTIC_ASR_INGESTION_TERMINAL_ACK_SCOPE")
    readback, readback_files = None, {}
    if terminal["status"] == "completed":
        folder = Path(ingestion["root"]) / "checkpoints/readback-complete"
        readback = phase_metadata(folder)
        closed = read(folder / "capsule.json")["context"]
        require(closed["stage"] == pin(ingestion_stage_path) and closed["recipe"] == ingestion["recipe"] and
                closed["queue"] == ingestion["queue"] and closed["heavy_lock"] == stage["heavy_lock"] and
                closed["queue_result"] == terminal_pin and closed["phase"] == "readback-complete" and
                closed["launch_intent"] is False and (closed["readback"] is not None or closed["failure"] is not None),
                "ACOUSTIC_ASR_INGESTION_READBACK_CLOSURE")
        for key in ("readback", "failure"):
            if closed[key] is not None:
                checked(closed[key])
                readback_files[key] = closed[key]
    dependency_paths = [ingestion_validator.parents[1] / queue.relative(name) for name in ingestion["source_hashes"]]
    dependency_paths += [Path(stage["handoff_validator"]["path"]).parents[1] / queue.relative(name)
                         for name in handoff["pipeline_files"]]
    dependency_paths += [Path(stage["context_validator"]["path"]).parents[1] / queue.relative(name)
                         for name in context["pipeline_files"]]
    for reader in closure["readers"].values():
        dependency_paths += [Path(reader["root"]) / queue.relative(name) for name in reader["source_files_sha256"]]
    return {"schema": PREREQUISITES, "context_campaign": stage["context_campaign"],
        "context_handoff": stage["context_handoff"], "handoff_validator": stage["handoff_validator"],
        "context_closure": closure,
        "context_status": status, "context_closures": closures, "ingestion_stage": pin(ingestion_stage_path),
        "ingestion_validator": pin(ingestion_validator), "ingestion_selection": stage["ingestion_selection"],
        "ingestion_queue_metadata": {name: ingestion[name] for name in
            ("queue", "request", "active_plan", "recipe", "original_recipe")},
        "ingestion_terminal": terminal_pin, "ingestion_terminal_state": terminal["status"],
        "ingestion_terminal_checkpoint": terminal_metadata, "ingestion_readback_checkpoint": readback,
        "ingestion_readback_files": readback_files,
        "ingestion_preparation": {"manifest": prepared_manifest, **preparation_files},
        "prerequisite_source_files": [pin(path) for path in sorted(set(dependency_paths))],
        "handoff_history": {**{key: handoff[key] for key in ("original_stage", "initial_status", "failure_receipt",
                              "original_generation_results")},
                            "failure_checkpoint": phase_metadata(handoff["failure_checkpoint"])},
        "context_package_manifests": [pin(Path(p["path"]) / "MANIFEST.json") for p in closure["packages"]],
        "new_model_calls": 0}


def bind_prerequisites(stage_path, ingestion_stage, validator):
    stage, _plan, _request, _sha = load_stage(stage_path)
    root = Path(stage["root"])
    with queue.exclusive(root / "driver.lock"):
        require(all(s["status"] == "pending" for s in queue.inspect(root / "queue")["jobs"].values()),
                "ACOUSTIC_ASR_BIND_BEFORE_ANY_CLAIM")
        value = prerequisite_readback(stage, ingestion_stage, validator)
        queue.write_json(root / "prerequisites.json", value)
    return {"prerequisites": pin(root / "prerequisites.json"), "new_model_calls": 0}


def verify_prerequisites(stage):
    path = Path(stage["root"]) / "prerequisites.json"
    require(path.is_file(), "ACOUSTIC_ASR_PREREQUISITE_CLOSURES_REQUIRED")
    prior = read(path)
    require(prior == prerequisite_readback(stage, prior["ingestion_stage"]["path"], prior["ingestion_validator"]["path"]),
            "ACOUSTIC_ASR_PREREQUISITE_CLOSURE_CHANGED")
    return pin(path)


def checkpoint_scope(stage_path, stage, index, phase):
    root = Path(stage["root"])
    _qroot, _meta, _request, qsha = queue.load_queue(root / "queue")
    recipe = root / "queue/recipes" / (job_id(index) + ".json")
    return {"schema": SCHEMA, "phase": phase, "stage": pin(stage_path), "queue_sha256": qsha,
        "job_id": job_id(index), "recipe": pin(recipe), "asr_request": stage["asr_requests"][index],
        "owner_recipe": pin(root / "recipes" / (job_id(index) + ".json")),
        "sample_ids": stage["chunks"][index], "maximum_asr_calls": 3 * len(stage["chunks"][index]),
        "prerequisites": pin(root / "prerequisites.json"), "heavy_lock": stage["heavy_lock"],
        "automatic_retries": 0, "new_tts_calls": 0, "new_optimizer_steps": 0,
        "pending_intent_restart_allowed": False}


def pinned_metadata_paths(value):
    """Collect retained closure metadata, without pulling earlier audio/ZIPs."""
    if type(value) is dict:
        if set(value) == {"path", "bytes", "sha256"}:
            return [checked(value)]
        return [path for item in value.values() for path in pinned_metadata_paths(item)]
    if type(value) is list:
        return [path for item in value for path in pinned_metadata_paths(item)]
    return []


def checkpoint_sources(stage, index):
    root = Path(stage["root"])
    files = [root / name for name in ("stage.json", "plan.json", "variant-manifest.json", "variant-rows.jsonl",
        "variant-summary.json", "variant-request.json", "asr-setup.json", "queue-request.json", "prerequisites.json")]
    files += checkpoints.regular_tree(root / "execution-source") + checkpoints.regular_tree(root / "recipes")
    files += [ROOT / name for name in stage["source_hashes"]]
    files += [Path(stage[key]["path"]) for key in
              ("live_runtime_setup", "context_campaign", "context_validator", "context_handoff", "handoff_validator",
               "ingestion_selection", "ingestion_validator")]
    files += [Path(stage["declaration_registration"])]
    files += pinned_metadata_paths(read(root / "prerequisites.json"))
    plan = read(stage["plan"]["path"])
    for ident in stage["chunks"][index]:
        binding = plan["bindings"][ident]
        for name in ("parent_retained_metadata", "parent_retained_wav", "acoustic_source_manifest",
                     "acoustic_source_metadata", "acoustic_asset_wav"):
            if binding[name] is not None:
                files.append(checked(binding[name]))
    files += [root / "queue" / name for name in ("queue.json", "request.json", "active-plan.json")]
    files += checkpoints.regular_tree(root / "queue/recipes")
    files += checkpoints.regular_tree(root / "queue/jobs" / job_id(index))
    files += checkpoints.regular_tree(chunk_path(root, job_id(index)))
    for ack in sorted((root / "checkpoints").glob("*/*/ack.json")):
        files += [ack, ack.parent / "provider-save.json", ack.parent / "capsule.json", ack.parent / "phase-marker.json"]
    return files


def verify_intent(stage_path, stage, index):
    folder = Path(stage["root"]) / "checkpoints" / job_id(index) / "before-asr"
    require(os.environ.get(INTENT_ENV) == str(folder) and os.environ.get(OWNER_ENV),
            "ACOUSTIC_ASR_LIVE_INTENT_REQUIRED")
    descriptor = checkpoints.verify_capsule(folder, current_sources=True)
    checkpoints.verify_ack(folder)
    expected = {**checkpoint_scope(stage_path, stage, index, "before-asr"),
        "owner_invocation": os.environ[OWNER_ENV], "launch_intent": True}
    require(descriptor["context"] == expected, "ACOUSTIC_ASR_LIVE_INTENT_SCOPE_CHANGED")


def run_job(stage_path, index, recipe_path):
    require(os.environ.get("KWS_RESEARCH_RECIPE_SHA256") ==
            queue.file_record(recipe_path)["sha256"], "ACOUSTIC_ASR_QUEUE_RECIPE_BINDING")
    stage, _plan, _request, _sha = load_stage(stage_path)
    require(type(index) is int and 0 <= index < JOBS, "ACOUSTIC_ASR_WORKER_CHUNK_INDEX")
    root, ident = Path(stage["root"]), job_id(index)
    require(Path(recipe_path).absolute() == root / "queue/recipes" / (ident + ".json") and
            queue.file_record(recipe_path) == {k: stage["asr_requests"][index][k] for k in ("bytes", "sha256")} and
            os.environ.get("KWS_RESEARCH_JOB_ID") == ident and
            os.environ.get("KWS_RESEARCH_RECIPE") == str(Path(recipe_path).absolute()) and
            os.environ.get("KWS_RESEARCH_OUTPUT_DIR") == str(root / "queue/jobs" / ident / "outputs"),
            "ACOUSTIC_ASR_WORKER_SCOPE")
    verify_prerequisites(stage)
    verify_intent(stage_path, stage, index)
    with batch.heavy_model_slot(stage["heavy_lock"]) as lease:
        require(lease.supervised and lease.inherited, "ACOUSTIC_ASR_SUPERVISED_QUEUE_REQUIRED")
        receipt = selective.run(recipe_path)
        return 0 if receipt["state"]["status"] == "complete" else 1


def observe_variant(binding, results):
    """Only posthoc comparison; never return replacement training labels."""
    row, parent = binding["variant_row"], binding["parent_row"]
    require(set(results) == set(batch.MODELS), "ACOUSTIC_ASR_ALL_THREE_OBSERVATIONS_REQUIRED")
    observed = {}
    for model in batch.MODELS:
        value = results[model]
        require(value["model"] == model and value["status"] == "complete" and
                value["boundary"] == batch.BOUNDARY, "ACOUSTIC_ASR_NATIVE_RESULT_BOUNDARY")
        text = selective.normalize(value["text"])
        observed[model] = {"raw_text": value["text"], "normalized_full_text": text,
            "decoder_terminated": value["decoder_terminated"], "quality_flags": value["quality_flags"],
            "usable_complete_text": value["decoder_terminated"] is True and not value["quality_flags"] and bool(text),
            "keyword_occurrences": {word: text.count(word) for word in WORDS}}
    texts = [value["normalized_full_text"] for value in observed.values()]
    counts = Counter(texts)
    unanimous = texts[0] if len(counts) == 1 else None
    usable = all(value["usable_complete_text"] for value in observed.values())
    inherited = selective.normalize(parent["full_text"])
    require(inherited, "ACOUSTIC_ASR_COMPLETE_PARENT_TRANSCRIPT_REQUIRED")
    if all(not text for text in texts):
        conclusion = "all-empty-native-transcripts"
    elif not usable:
        conclusion = "native-quality-or-termination-uncertain"
    elif unanimous is None:
        conclusion = "full-transcript-disagreement"
    elif unanimous == inherited:
        conclusion = "unanimous-parent-fulltext-match"
    else:
        conclusion = "unanimous-different-fulltext"
    return {"sample_id": row["sample_id"], "variant_row_sha256": binding["variant_row_sha256"],
        "parent_key": row["parent_key"], "parent_row_sha256": binding["parent_row_sha256"],
        "parent_file_sha256": row["parent_file_sha256"], "variant_file_sha256": row["file_sha256"],
        "original_pcm16_sha256": row["pcm_sha256"], "decoder_float32_sha256": binding["decoded_float32_sha256"],
        "source_family_id": row["source_family_id"], "speaker_id": row["speaker_id"],
        "source_role": row["source_role"], "split": row["split"], "kind": row["kind"],
        "recipe": row["recipe"], "measurements": binding["measurements"],
        "inherited_parent_full_text": parent["full_text"], "normalized_parent_full_text": inherited,
        "observations": observed, "unanimous_normalized_text": unanimous,
        "text_agreement_counts": dict(counts), "conclusion": conclusion,
        "quarantine_suggested": conclusion != "unanimous-parent-fulltext-match",
        "quarantine_is_automatic_label_mutation": False,
        "inherited_variant_labels": {key: row[key] for key in
            ("ctc_mask", "ctc_weight", "full_text", "target_ids", "keyword_targets", "keyword_mask", "keyword_weight")},
        "source_row_changed": False, "new_loss_weights_created": False,
        "human_gold": False, "new_independent_speaker": False,
        "scope": "three-engine-observation-and-posthoc-parent-comparison"}


def terminal(stage, index):
    root = Path(stage["root"]) / "queue"
    queue.inspect(root, reconcile=True)
    qroot, _meta, request, qsha = queue.load_queue(root)
    result = queue.result_for(qroot, request["jobs"][index], qsha)
    require(result["status"] in ("completed", "failed", "timed_out", "unknown"),
            "ACOUSTIC_ASR_AUTHENTIC_TERMINAL_REQUIRED")
    return result, pin(qroot / "jobs" / job_id(index) / "result.json")


def build_view(stage, plan, index):
    result, result_pin = terminal(stage, index)
    require(result["status"] == "completed", "ACOUSTIC_ASR_COMPLETED_NATIVE_JOB_REQUIRED")
    request_path = Path(stage["asr_requests"][index]["path"])
    request = read(request_path)
    output = Path(request["output"])
    require((output / "audit/final-summary.json").is_file(), "ACOUSTIC_ASR_SEALED_NATIVE_AUDIT_REQUIRED")
    audit = selective.seal_or_verify(request_path)
    require(audit["state"]["status"] == "complete" and
            audit["state"]["original_recognition_calls"] == 3 * len(stage["chunks"][index]) and
            all(audit["state"]["models"][model]["not_requested"] == 0 for model in batch.MODELS),
            "ACOUSTIC_ASR_THREE_ENGINE_COVERAGE_REQUIRED")
    mapping = read(stage["audio_inputs"][index]["source_map"]["path"])
    rows = []
    for item in mapping["rows"]:
        ident, clip = item["source_id"], item["clip_id"]
        results = {model: read(output / "results" / model / (clip + ".json")) for model in batch.MODELS}
        row = observe_variant(plan["bindings"][ident], results)
        row["native_evidence"] = {model: {kind: pin(output / kind / model / (clip + ".json"))
            for kind in ("claims", "native", "results")} for model in batch.MODELS}
        rows.append(row)
    return {"schema": VIEW, "stage": pin(Path(stage["root"]) / "stage.json"), "job_id": job_id(index),
        "sample_ids": stage["chunks"][index], "native_job_result": result_pin,
        "native_audit": pin(output / "audit/final-summary.json"), "rows": rows,
        "observed_rows": len(rows), "asr_calls": audit["state"]["original_recognition_calls"],
        "by_conclusion": dict(Counter(r["conclusion"] for r in rows)),
        "quarantine_suggestions": sum(r["quarantine_suggested"] for r in rows),
        "new_tts_calls": 0, "new_optimizer_steps": 0, "source_rows_changed": 0, "human_gold": False}


def save_or_verify_view(stage, plan, index):
    path = chunk_path(stage["root"], job_id(index)) / "validation.json"
    value = build_view(stage, plan, index)
    if path.exists():
        require(read(path) == value, "ACOUSTIC_ASR_VALIDATION_VIEW_CHANGED")
    else:
        queue.write_json(path, value)
    return value, pin(path)


def checkpoint(gate, stage_path, stage, index, phase, *, result_pin=None, view_pin=None):
    scope = checkpoint_scope(stage_path, stage, index, phase)
    if result_pin is not None:
        scope["native_job_result"] = result_pin
        scope["terminal_state"] = read(result_pin["path"])["status"]
    if view_pin is not None:
        scope["validation_view"] = view_pin
    return gate.checkpoint(Path(stage["root"]) / "checkpoints" / job_id(index) / phase,
        scope, checkpoint_sources(stage, index), intent=phase == "before-asr")


def verify_chunk_closure(stage, index):
    root = Path(stage["root"])
    folder = root / "checkpoints" / job_id(index) / "chunk-complete"
    checkpoints.verify_ack(folder)
    qroot, _meta, request, qsha = queue.load_queue(root / "queue")
    result = queue.result_for(qroot, request["jobs"][index], qsha)
    require(result["status"] == "completed", "ACOUSTIC_ASR_CLOSURE_REQUIRES_COMPLETED_NATIVE_RESULT")
    expected = {**checkpoint_scope(root / "stage.json", stage, index, "chunk-complete"),
        "native_job_result": pin(qroot / "jobs" / job_id(index) / "result.json"),
        "terminal_state": "completed", "validation_view": pin(chunk_path(root, job_id(index)) / "validation.json"),
        "launch_intent": False}
    observed = read(folder / "capsule.json")["context"]
    require({k: v for k, v in observed.items() if k != "owner_invocation"} == expected,
            "ACOUSTIC_ASR_CHUNK_CLOSURE_SCOPE_CHANGED")
    return expected


def aggregate(stage, plan):
    rows = []
    for index in range(JOBS):
        view, _pin = save_or_verify_view(stage, plan, index)
        verify_chunk_closure(stage, index)
        rows.extend(view["rows"])
    require([r["sample_id"] for r in rows] == plan["ordered_sample_ids"] and len(rows) == TOTAL,
            "ACOUSTIC_ASR_AGGREGATE_ROSTER")
    noise = [row for row in rows if row["kind"] == "recorded-noise-mix"]
    require(len(noise) == 40 and all(r["inherited_variant_labels"]["ctc_mask"] is False and
            r["inherited_variant_labels"]["ctc_weight"] == 0 and
            r["inherited_variant_labels"]["target_ids"] is None for r in noise),
            "ACOUSTIC_ASR_NOISE_CTC_MASK_CHANGED")
    summary = {"schema": VIEW, "observed_rows": len(rows), "native_asr_calls": TOTAL * 3,
        "by_kind": dict(Counter(r["kind"] for r in rows)), "by_split": dict(Counter(r["split"] for r in rows)),
        "by_conclusion": dict(Counter(r["conclusion"] for r in rows)),
        "quarantine_suggestions": sum(r["quarantine_suggested"] for r in rows),
        "noise_ctc_masked_rows_unchanged": 40, "frozen_variant_rows_changed": 0,
        "new_loss_weights_created": False, "new_tts_calls": 0, "new_optimizer_steps": 0,
        "human_gold": False, "real_farfield_or_soft_speech_claim": False}
    target = Path(stage["root"]) / "validation"
    bodies = {"rows.jsonl": b"".join(queue.canonical(row) for row in rows), "summary.json": queue.canonical(summary)}
    expected = {name: {"bytes": len(body), "sha256": queue.sha(body)} for name, body in bodies.items()}
    if target.exists():
        require(read(target / "MANIFEST.json") == expected and
                set(p.name for p in target.iterdir()) == set(bodies) | {"MANIFEST.json"} and
                all((target / name).read_bytes() == body for name, body in bodies.items()),
                "ACOUSTIC_ASR_AGGREGATE_CHANGED_OR_PARTIAL")
    else:
        target.mkdir(mode=0o700)
        for name, body in bodies.items():
            queue.write_new(target / name, body)
        queue.write_json(target / "MANIFEST.json", expected)
    return summary, pin(target / "MANIFEST.json")


def run(stage_path, *, gate=None):
    stage_path = queue.safe_path(stage_path)
    stage, plan, request, _qsha = load_stage(stage_path)
    root = Path(stage["root"])
    verify_prerequisites(stage)
    if gate is None:
        announced = set()
        def on_wait(folder, descriptor):
            if str(folder) not in announced:
                print(json.dumps({"event": "awaiting_durable_save", "checkpoint": str(folder),
                                  "capsule": descriptor["capsule"]}, sort_keys=True), flush=True)
                announced.add(str(folder))
        gate = checkpoints.Gate(on_wait=on_wait)
    deadline = time.monotonic() + WALL_SECONDS
    with queue.exclusive(root / "driver.lock"):
        for index in range(JOBS):
            report = queue.inspect(root / "queue", reconcile=True)
            state = report["jobs"][job_id(index)]["status"]
            require(state != "running", "ACOUSTIC_ASR_OTHER_WORKER_ACTIVE")
            if state == "pending":
                require(time.monotonic() + 600 < deadline and not report["unresolved_jobs"] and
                        not report["budget_overrun_nanoseconds"], "ACOUSTIC_ASR_NO_NEW_MODEL_BUDGET")
                if index:
                    verify_chunk_closure(stage, index - 1)
                checkpoint(gate, stage_path, stage, index, "before-asr")
                require(OWNER_ENV not in os.environ and INTENT_ENV not in os.environ, "ACOUSTIC_ASR_OWNER_ENV_CONFLICT")
                os.environ.update({OWNER_ENV: gate.owner,
                    INTENT_ENV: str(root / "checkpoints" / job_id(index) / "before-asr")})
                try:
                    while True:
                        require(time.monotonic() + 600 < deadline, "ACOUSTIC_ASR_DRIVER_WALL_BUDGET")
                        current, code = queue.run(root / "queue", max_jobs=1)
                        if code != 75 or current["status"] != "waiting_for_heavy_model_slot":
                            break
                        time.sleep(1)
                finally:
                    os.environ.pop(OWNER_ENV, None); os.environ.pop(INTENT_ENV, None)
            result, result_pin = terminal(stage, index)
            phase = "asr-complete" if result["status"] == "completed" else "asr-terminal"
            checkpoint(gate, stage_path, stage, index, phase, result_pin=result_pin)
            if result["status"] != "completed":
                return {"status": result["status"], "job_id": job_id(index), "new_tts_calls": 0,
                        "new_optimizer_steps": 0, "automatic_retry": False}, 1
            view, view_pin = save_or_verify_view(stage, plan, index)
            checkpoint(gate, stage_path, stage, index, "chunk-complete", result_pin=result_pin, view_pin=view_pin)
            print(json.dumps({"event": "acoustic_chunk_durably_closed", "job_id": job_id(index),
                "observed_rows": view["observed_rows"], "asr_calls": view["asr_calls"],
                "by_conclusion": view["by_conclusion"]}, ensure_ascii=False), flush=True)
        summary, manifest = aggregate(stage, plan)
        sources = checkpoint_sources(stage, JOBS - 1) + checkpoints.regular_tree(root / "validation")
        gate.checkpoint(root / "checkpoints/final/campaign-complete",
            {"schema": SCHEMA, "phase": "campaign-complete", "stage": pin(stage_path),
             "aggregate": manifest, "maximum_asr_calls": TOTAL * 3,
             "source_rows_changed": 0, "new_tts_calls": 0, "new_optimizer_steps": 0},
            sources, intent=False)
        return {"status": "completed", **summary, "actual_queue_budget": queue.inspect(root / "queue")}, 0


def status(stage_path):
    stage, _plan, _request, _sha = load_stage(stage_path)
    closed = []
    for index in range(JOBS):
        if (Path(stage["root"]) / "checkpoints" / job_id(index) / "chunk-complete/ack.json").exists():
            verify_chunk_closure(stage, index)
            closed.append(job_id(index))
    return {"schema": SCHEMA, "queue": queue.inspect(Path(stage["root"]) / "queue"),
        "durably_closed_chunks": closed,
        "prerequisites_bound": (Path(stage["root"]) / "prerequisites.json").is_file(),
        "new_model_calls": 0}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("variants", "runtime", "output", "context-campaign", "context-validator", "context-handoff",
                 "handoff-validator", "ingestion-selection", "ingestion-validator"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--variants-manifest-sha256", required=True)
    prep.add_argument("--active-plan", type=Path, default=ROOT / "configs/research/active-plan-20261105.json")
    bind = commands.add_parser("bind-prerequisites")
    for name in ("stage", "ingestion-stage", "ingestion-validator"):
        bind.add_argument("--" + name, type=Path, required=True)
    for name in ("run", "status"):
        command = commands.add_parser(name)
        command.add_argument("--stage", type=Path, required=True)
    worker = commands.add_parser("run-job")
    worker.add_argument("--recipe", type=Path, required=True)
    worker.add_argument("--stage", type=Path, required=True)
    worker.add_argument("--chunk-index", type=int, required=True)
    args = cli.parse_args(argv)
    if args.command == "prepare":
        result, code = prepare(args), 0
    elif args.command == "bind-prerequisites":
        result, code = bind_prerequisites(args.stage, args.ingestion_stage, args.ingestion_validator), 0
    elif args.command == "run":
        result, code = run(args.stage)
    elif args.command == "run-job":
        code = run_job(args.stage, args.chunk_index, args.recipe)
        result = {"status": "completed" if code == 0 else "failed", "new_tts_calls": 0, "new_optimizer_steps": 0}
    else:
        result, code = status(args.stage), 0
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
