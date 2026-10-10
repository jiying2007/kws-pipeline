#!/usr/bin/env python3
"""Offline source-screen planning only. No inference, acquisition or execution API."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SHA = re.compile(r"[0-9a-f]{64}\Z")
TEXTS = ("你好小窝", "小窝小窝", "你好小屋", "小屋小屋", "小窝小屋", "小屋小窝", "你好你好", "小窝")
DESIGNS = (
    ("design-a", "成年女性，普通话，音色明亮自然，平静，中等语速，连续说完，不加额外词语。"),
    ("design-b", "成年男性，普通话，音色低沉自然，平静，中等语速，连续说完，不加额外词语。"),
)
MODELS = (
    ("qwen-voicedesign", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "5ecdb67327fd37bb2e042aab12ff7391903235d3"),
    ("firered-instruct", "FireRedTeam/FireRedTTS3", "dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610"),
)
ASR_IDS = ("qwen06", "sensevoice")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def decode(raw):
    require(type(raw) is bytes and len(raw) <= 1024 * 1024, "bounded JSON bytes required")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def expected_plan():
    cells = []
    for lane_index, (lane, model, revision) in enumerate(MODELS):
        for design_index, (design_id, prompt) in enumerate(DESIGNS):
            for text_index, text in enumerate(TEXTS):
                ordinal = lane_index * 16 + design_index * 8 + text_index + 1
                cells.append({"cell_id": f"screen32-{ordinal:03d}", "lane": lane,
                              "model_id": model, "model_revision": revision,
                              "design_id": design_id, "instruction": prompt,
                              "intended_text": text, "seed": 101000 + ordinal,
                              "attempts_max": 1, "role": "source_screen_only",
                              "human_gold": False, "training_admitted": False,
                              "lineage_group": lane + ":" + design_id})
    return {"schema": "bounded-source-screen32-plan-v1", "created_utc": "2026-10-10",
            "status": "PREPARED_NOT_EXECUTION_READY", "execution_enabled": False,
            "max_tts_calls": 32, "max_primary_asr_calls": 64,
            "max_dispute_asr_calls": 32, "max_attempts_per_cell": 1,
            "reference_audio": None, "voice_cloning": False, "paid_api": False,
            "regeneration": False, "cherry_picking": False, "cells": cells,
            "asr": {"primary": list(ASR_IDS), "context": "", "hotwords": [],
                    "language": "auto", "expected_text_visible": False,
                    "model_locks": "research/qwen6_asr/model-locks.json",
                    "dispute": "whisper_only_after_primary_raw_freeze",
                    "dispute_model_lock": None, "human_gold": False},
            "boundary": {"d20": "FAIL", "d90": "NOT_RUN", "shipping_approved": False,
                         "human_qualification": "DEFERRED", "device_qualification": "DEFERRED"},
            "stop_rule": "One attempt consumes the cell even on failure. Stop that source on exception, timeout or resource breach; retain all NOT_RUN rows. No replacement source, voice, seed, text, gain repair or regeneration."}


def validate_plan(plan):
    # Canonical-byte equality also rejects bool-as-int and unknown fields.
    require(canonical(plan) == canonical(expected_plan()), "frozen 32-cell planning contract changed")
    return plan["cells"]


def request_preview(plan, cell_id):
    """Return data, never a callable. Loading/generation remains outside this tool."""
    rows = validate_plan(plan)
    verify_local_pins()
    row = next((item for item in rows if item["cell_id"] == cell_id), None)
    require(row is not None, "unknown cell")
    recipe = decode((ROOT / "research/qwen6_tts/plan.json").read_bytes())["recipe"]
    if row["lane"] == "qwen-voicedesign":
        return {"method": "generate_voice_design", "seed": row["seed"],
                "kwargs": {"text": row["intended_text"], "instruct": row["instruction"], **recipe},
                "device": "cpu", "dtype": "float32", "attention": "eager",
                "execution_enabled": False}
    return {"method": "generate_voice_design", "seed": row["seed"],
            "kwargs": {"text": row["intended_text"], "instruction": row["instruction"],
                       "language": "Chinese", "do_clean": False, "do_tn": False,
                       "do_split": False, "n_timesteps": 10, "inference_cfg": 1.2,
                       "seed": row["seed"]},
            "execution_enabled": False, "blocked": "UPSTREAM_CUDA_ONLY_UNQUALIFIED_CPU_ROUTE"}


def validate_generation_ledger(plan, rows):
    expected = validate_plan(plan)
    require(type(rows) is list and len(rows) == 32, "retain all 32 generation outcomes")
    hashes, pcms = set(), set()
    stopped = set()
    for row, cell in zip(rows, expected):
        require(type(row) is dict and set(row) == {"cell_id", "status", "attempts", "audio"}, "ledger fields")
        require(row["cell_id"] == cell["cell_id"], "frozen cell order")
        require(type(row["attempts"]) is int, "attempts must be integer")
        status = row["status"]
        require(status in ("NOT_RUN", "GENERATED", "FAILED_NO_RETRY"), "generation state")
        require(row["attempts"] == (0 if status == "NOT_RUN" else 1), "single generation attempt")
        require(cell["lane"] not in stopped or status == "NOT_RUN", "source stopped; no later attempts")
        if status != "GENERATED":
            require(row["audio"] is None, "no audio claimed for unavailable output")
            stopped.add(cell["lane"])
            continue
        audio = row["audio"]
        require(type(audio) is dict and set(audio) == {"wav_sha256", "pcm_sha256", "frames", "sample_rate_hz", "channels", "sample_width_bytes"}, "audio fields")
        for name in ("wav_sha256", "pcm_sha256"):
            require(type(audio[name]) is str and SHA.fullmatch(audio[name]), "audio hash")
        require(type(audio["frames"]) is int and 0 < audio["frames"] <= 192000, "12-second duration ceiling")
        require(all(type(audio[k]) is int for k in ("sample_rate_hz", "channels", "sample_width_bytes")), "audio geometry types")
        require((audio["sample_rate_hz"], audio["channels"], audio["sample_width_bytes"]) == (16000, 1, 2), "PCM16 mono16k required")
        require(audio["wav_sha256"] not in hashes and audio["pcm_sha256"] not in pcms, "duplicate WAV or decoded PCM")
        hashes.add(audio["wav_sha256"]); pcms.add(audio["pcm_sha256"])
    return rows


def blind_job(plan, ledger):
    """Metadata projection; this does not validate or package actual WAV bytes."""
    validate_generation_ledger(plan, ledger)
    audio = sorted((row["audio"] for row in ledger if row["status"] == "GENERATED"), key=lambda a: a["wav_sha256"])
    return {"schema": "blind-source-screen32-job-v1", "clips": [
        {"audio_id": f"clip-{index:06d}", "audio_path": "audio/" + row["wav_sha256"] + ".wav",
         "wav_sha256": row["wav_sha256"]} for index, row in enumerate(audio, 1)]}


def validate_blind_job(job):
    require(type(job) is dict and set(job) == {"schema", "clips"} and job["schema"] == "blind-source-screen32-job-v1", "blind schema")
    require(type(job["clips"]) is list and 0 <= len(job["clips"]) <= 32, "blind denominator")
    seen = set()
    for index, row in enumerate(job["clips"], 1):
        require(type(row) is dict and set(row) == {"audio_id", "audio_path", "wav_sha256"}, "audio-only blind fields")
        sha = row["wav_sha256"]
        require(type(sha) is str and SHA.fullmatch(sha) and sha not in seen, "blind audio hash")
        require(row["audio_id"] == f"clip-{index:06d}" and row["audio_path"] == "audio/" + sha + ".wav", "opaque audio identity")
        seen.add(sha)
    return [row["audio_id"] for row in job["clips"]]


def freeze_disputes(job, primary_raw, expected_raw_sha256):
    """Consume two saved recognizers only. No target joins or inference allowed."""
    ids = validate_blind_job(job)
    require(type(primary_raw) is bytes and type(expected_raw_sha256) is str
            and SHA.fullmatch(expected_raw_sha256)
            and hashlib.sha256(primary_raw).hexdigest() == expected_raw_sha256, "raw primary freeze mismatch")
    primary = decode(primary_raw)
    require(type(primary) is dict and set(primary) == set(ASR_IDS), "exact two heterogeneous recognizers")
    spec = importlib.util.spec_from_file_location("screen_quality", ROOT / "research/experiment_quality_guards/quality_gates.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for rows in primary.values():
        require(type(rows) is list and len(rows) == len(ids), "all blind outcomes retained")
        for audio_id, row in zip(ids, rows):
            require(type(row) is dict and set(row) == {"audio_id", "status", "raw_text", "quality_flags"}, "raw ASR fields")
            require(row["audio_id"] == audio_id, "ASR order")
            require(row["status"] in ("complete", "failed", "not_run"), "ASR state")
            require(row["raw_text"] is None or type(row["raw_text"]) is str, "ASR text")
            require(type(row["quality_flags"]) is list and all(type(v) is str for v in row["quality_flags"]), "ASR quality flags")
            require(row["status"] == "complete" or row["raw_text"] is None, "unavailable ASR must not claim text")
    disputes, unresolved = [], []
    for index, audio_id in enumerate(ids):
        pair = [primary[name][index] for name in ASR_IDS]
        texts = [module.normalized_actual(row["raw_text"] or "") for row in pair]
        if any(row["status"] != "complete" or not text or row["quality_flags"] for row, text in zip(pair, texts)):
            unresolved.append(audio_id)
        elif texts[0] != texts[1]:
            disputes.append(audio_id)
    return {"schema": "screen32-frozen-disputes-v1", "job_sha256": digest(job),
            "primary_raw_sha256": expected_raw_sha256, "whisper_audio_ids": disputes,
            "unresolved_audio_ids": unresolved, "max_attempts_per_dispute": 1,
            "whisper_execution_enabled": False, "human_gold": False, "training_admitted": False}


def verify_local_pins():
    pins = decode((HERE / "reuse-pins.json").read_bytes())
    for name, sha in pins.items():
        require(name.startswith("research/") and ".." not in Path(name).parts and SHA.fullmatch(sha), "reuse pin")
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), "missing pinned source")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == sha, "reused source changed: " + name)
    return len(pins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-cell", choices=[row["cell_id"] for row in expected_plan()["cells"]])
    args = parser.parse_args()
    plan = decode((HERE / "plan.json").read_bytes())
    validate_plan(plan)
    count = verify_local_pins()
    output = request_preview(plan, args.preview_cell) if args.preview_cell else {
        "status": "OFFLINE_PLANNING_VALID", "execution_ready": False,
        "planned_tts_cells": 32, "generation_calls": 0, "asr_calls": 0,
        "local_reuse_pins_verified": count, "plan_sha256": digest(plan),
        "d20": "FAIL", "d90": "NOT_RUN", "shipping_approved": False}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
