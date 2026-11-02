#!/usr/bin/env python3
"""Bounded, offline, batch three-ASR *diagnostic* screening for clone WAVs.

One pinned local command per independent recognizer processes a WAV-only input
manifest, with no requested words or target hotwords. This does not authenticate
recognizer execution or provide any training, human-gold, or release authority.
No command runs unless --execute is supplied. No unknown call is retried.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from authorized_voice_clone_batch import (create_once, claim_invocation_once, identifier,
                                           need, pinned_file, read_object)
from corpus_identity import canonical_hash, sha256_file
from voice_clone_effect_audit import (audit as effect_audit, private_audio_path,
                                      read_jsonl)

POLICY = "offline-authorized-clone-three-asr-batch-screen-v1"
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
MAX_RECORDINGS = 256
DEFAULT_CONFIG = {"language": "zh", "task": "transcribe", "prompt": "", "hotwords": []}


def parse_plan(path: Path) -> dict:
    raw = read_object(path)
    need(type(raw.get("schema_version")) is int and raw["schema_version"] == 1
         and raw.get("policy") == POLICY, "three-ASR screen plan identity mismatch")
    maximum = raw.get("max_recordings")
    need(type(maximum) is int and 1 <= maximum <= MAX_RECORDINGS,
         "max_recordings must be an explicit bound in [1,256]")
    recognizers = raw.get("recognizers")
    need(isinstance(recognizers, list) and len(recognizers) == 3,
         "screen requires exactly three independently named recognizers")
    families, model_ids, weights = set(), set(), set()
    parsed = []
    for i, record in enumerate(recognizers):
        need(isinstance(record, dict), "recognizer must be an object")
        family = identifier(record.get("model_family"), f"recognizers[{i}].model_family")
        model_id = identifier(record.get("model_id"), f"recognizers[{i}].model_id")
        revision = identifier(record.get("model_revision"), f"recognizers[{i}].model_revision")
        need(family not in families and model_id not in model_ids,
             "repeated ASR family/model cannot make three votes")
        need(record.get("decode_config") == DEFAULT_CONFIG,
             "diagnostic ASR must use Mandarin transcription without prompt/hotwords")
        need(isinstance(record.get("decode_config"), dict) and
             set(record["decode_config"]) == set(DEFAULT_CONFIG),
             "ASR decode_config needs exactly the frozen fields")
        exe, exe_sha = pinned_file(path.parent, record.get("executable"), "recognizer.executable")
        entries = record.get("assets")
        need(isinstance(entries, list) and entries, "recognizer assets missing")
        assets, asset_identity = {}, {}
        for entry in entries:
            need(isinstance(entry, dict), "ASR asset entry must be an object")
            role = identifier(entry.get("role"), "ASR asset role")
            need(role not in assets, "duplicate ASR asset role")
            resolved, digest = pinned_file(path.parent, entry, "recognizer.asset")
            assets[role], asset_identity[role] = resolved, digest
        need("model" in assets and asset_identity["model"] not in weights,
             "each ASR requires a distinct pinned model asset")
        weights.add(asset_identity["model"])
        argv = record.get("argv_template")
        need(isinstance(argv, list) and argv and all(isinstance(v, str) and v for v in argv),
             "recognizer argv_template must be nonempty strings")
        need(argv[0] == "{executable}", "ASR command must start with its pinned executable")
        used = {x for word in argv for x in PLACEHOLDER.findall(word)}
        need({"executable", "input", "output", "asset:model"} <= used,
             "ASR argv_template must include WAV input, output and pinned model")
        need(used <= {"executable", "input", "output"} |
             {"asset:" + role for role in assets}, "ASR argv_template contains unknown placeholder")
        need(all("{" not in PLACEHOLDER.sub("", word) and
                 "}" not in PLACEHOLDER.sub("", word) for word in argv),
             "unmatched brace in recognizer argv")
        timeout = record.get("timeout_seconds")
        need(type(timeout) is int and 1 <= timeout <= 1200,
             "ASR timeout must be explicit and in [1,1200] seconds")
        identity = {
            "model_family": family, "model_id": model_id, "model_revision": revision,
            "executable_sha256": exe_sha, "asset_sha256": dict(sorted(asset_identity.items())),
            "argv_template": argv, "decode_config": DEFAULT_CONFIG, "timeout_seconds": timeout}
        parsed.append({"model_family": family, "assets": assets, "executable": exe,
                       "argv_template": argv, "timeout_seconds": timeout, "identity": identity})
        families.add(family)
        model_ids.add(model_id)
    return {"identity": {"schema_version": 1, "policy": POLICY, "max_recordings": maximum,
                         "recognizers": [row["identity"] for row in parsed]},
            "max_recordings": maximum, "recognizers": parsed}


def source_batch(campaign_root: Path, maximum: int) -> tuple[dict, list[dict], str]:
    report = effect_audit(campaign_root)
    rows = read_jsonl(campaign_root / "manifest.jsonl")
    need(0 < len(rows) <= maximum, "clone campaign exceeds ASR screening batch budget")
    source_rows = []
    for row in rows:
        wav = private_audio_path(campaign_root, row["audio"])
        source_rows.append({"item_id": row["item_id"], "file_sha256": row["file_sha256"],
                            "pcm_sha256": row["pcm_sha256"], "audio": str(wav)})
    source_rows.sort(key=lambda x: x["item_id"])
    encoded = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                      for row in source_rows)
    # Model input contains no requested text, intent, human label, or source permission notes.
    need(all(set(row) == {"item_id", "file_sha256", "pcm_sha256", "audio"}
             for row in source_rows), "ASR input must be audio-only identity")
    return report, source_rows, encoded


def expected_output(path: Path, audio_sha: dict[str, str]) -> dict[str, str]:
    rows = read_jsonl(path)
    need(len(rows) == len(audio_sha), "ASR output does not cover every WAV")
    outputs = {}
    for row in rows:
        need(set(row) == {"item_id", "file_sha256", "raw_text", "status"},
             "ASR output has incomplete or unexpected fields")
        item = row["item_id"]
        need(isinstance(item, str) and item in audio_sha and item not in outputs,
             "ASR output has duplicate or unrecognized item ID")
        need(row["file_sha256"] == audio_sha[item], "ASR output WAV SHA mismatch")
        need(row["status"] == "complete" and isinstance(row["raw_text"], str),
             "ASR output must carry actual complete text, possibly empty")
        outputs[item] = row["raw_text"]
    need(set(outputs) == set(audio_sha), "ASR output missing WAV entries")
    return outputs


def render_command(recognizer: dict, input_path: Path, output_path: Path) -> list[str]:
    mapping = {"executable": str(recognizer["executable"]),
               "input": str(input_path), "output": str(output_path)}
    mapping.update({"asset:" + k: str(v) for k, v in recognizer["assets"].items()})
    argv = [PLACEHOLDER.sub(lambda match: mapping[match.group(1)], word)
            for word in recognizer["argv_template"]]
    need(Path(argv[0]).resolve() == recognizer["executable"],
         "recognizer executable changed after hash preflight")
    return argv


def run_recognizer(recognizer: dict, *, output_root: Path,
                   input_path: Path, audio_sha: dict[str, str],
                   plan_sha: str) -> dict:
    family = recognizer["model_family"]
    raw_path = output_root / "raw" / (family + ".jsonl")
    marker = output_root / "attempts" / (family + ".intent.json")
    receipt_path = output_root / "attempts" / (family + ".receipt.json")
    intent = {"schema_version": 1, "policy": POLICY, "model_family": family,
              "plan_sha256": plan_sha, "batch_sha256": sha256_file(input_path),
              "recognizer_identity_sha256": canonical_hash(recognizer["identity"])}
    intent_text = json.dumps(intent, sort_keys=True) + "\n"
    if marker.exists():
        need(marker.read_text(encoding="utf-8") == intent_text,
             "ASR invocation marker identity drift")
        need(receipt_path.is_file() and raw_path.is_file(),
             "unreceipted ASR invocation is unknown and consumed; no retry")
        receipt = read_object(receipt_path)
        need(all(receipt.get(k) == v for k, v in intent.items()) and
             receipt.get("raw_sha256") == sha256_file(raw_path) and
             receipt.get("screen_only") is True and
             receipt.get("training_admitted") is False and
             receipt.get("qualification_allowed") is False,
             "ASR readback receipt mismatch")
    else:
        need(not raw_path.exists() and not receipt_path.exists(),
             "orphan ASR output cannot be overwritten")
        argv = render_command(recognizer, input_path, raw_path)
        claim_invocation_once(marker, intent_text)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        env = {k: os.environ[k] for k in
               ("PATH", "LANG", "LC_ALL", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "HOME")
               if k in os.environ}
        subprocess.run(argv, shell=False, check=True,
                       timeout=recognizer["timeout_seconds"], env=env)
        expected_output(raw_path, audio_sha)
        receipt = {**intent, "raw_sha256": sha256_file(raw_path), "screen_only": True,
                   "training_admitted": False, "qualification_allowed": False}
        create_once(receipt_path, json.dumps(receipt, sort_keys=True) + "\n")
    return {"model_family": family, "identity_sha256": intent["recognizer_identity_sha256"],
            "raw_sha256": receipt["raw_sha256"], "outputs": expected_output(raw_path, audio_sha)}


def execute(plan: dict, campaign_root: Path, output_root: Path) -> dict:
    campaign_root = campaign_root.resolve(strict=True)
    output_root = output_root.resolve()
    need(output_root != ROOT and ROOT not in output_root.parents,
         "private ASR output cannot be under public source repository")
    need(output_root != campaign_root and output_root not in campaign_root.parents and
         campaign_root not in output_root.parents, "ASR output must not alias source campaign")
    report, batch, input_text = source_batch(campaign_root, plan["max_recordings"])
    identity = {**plan["identity"], "campaign_sha256": report["campaign_sha256"],
                "clone_manifest_sha256": report["manifest_sha256"]}
    plan_sha = canonical_hash(identity)
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = output_root / "audio-only-batch.jsonl"
    create_once(input_path, input_text)
    audio_sha = {item["item_id"]: item["file_sha256"] for item in batch}
    results = [run_recognizer(recognizer, output_root=output_root,
                              input_path=input_path, audio_sha=audio_sha, plan_sha=plan_sha)
               for recognizer in plan["recognizers"]]
    # Run source WAV verification again after all provider processes.
    after = effect_audit(campaign_root)
    need(after["manifest_sha256"] == report["manifest_sha256"] and
         after["campaign_sha256"] == report["campaign_sha256"],
         "source campaign drift during screening")
    screen_rows = []
    for item in batch:
        screen_rows.append({"item_id": item["item_id"], "file_sha256": item["file_sha256"],
                            "observations": [{"model_family": row["model_family"],
                                              "raw_text": row["outputs"][item["item_id"]]}
                                             for row in results]})
    screen_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                          for row in screen_rows)
    screen_path = output_root / "screen.jsonl"
    create_once(screen_path, screen_text)
    effect = effect_audit(campaign_root, screen_path)
    summary = {"schema_version": 1, "policy": POLICY, "campaign_sha256": report["campaign_sha256"],
               "plan_sha256": plan_sha, "wav_only_batch_sha256": sha256_file(input_path),
               "screen_sha256": sha256_file(screen_path),
               "recognizers": [{key: row[key] for key in
                                ("model_family", "identity_sha256", "raw_sha256")} for row in results],
               "screened_recordings": len(batch), "screen_counts": effect["screen_counts"],
               "asr_source_authenticated": False, "human_gold": False,
               "training_admitted": False, "qualification_allowed": False,
               "shipping_approved": False}
    create_once(output_root / "summary.json", json.dumps(summary, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="explicitly run three pinned local ASR batch commands")
    args = parser.parse_args(argv)
    plan = parse_plan(args.plan.resolve(strict=True))
    campaign = args.campaign_root.resolve(strict=True)
    report, batch, _ = source_batch(campaign, plan["max_recordings"])
    if args.execute:
        result = execute(plan, campaign, args.output_root)
    else:
        result = {"policy": POLICY, "campaign_sha256": report["campaign_sha256"],
                  "recordings_planned": len(batch), "recognizers": 3, "dry_run": True,
                  "provider_calls": 0, "asr_source_authenticated": False,
                  "training_admitted": False, "qualification_allowed": False}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
