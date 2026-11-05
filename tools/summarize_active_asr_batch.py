#!/usr/bin/env python3
"""Audit a private completed ASR matrix and retain complete content inventories.

Reads evidence only. It does not load models, edit transcripts, or publish data.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

from active_asr_batch import (BOUNDARY, CONFIGS, MODELS, ROOT, canonical, digest,
                              model_identity, read_json, utc, verify_inputs, write_once)
from setup_asr_runtime import sha_file


def inventory(root, prefix):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if path.is_symlink():
                raise ValueError("evidence inventory must not contain symlinks")
            result[prefix + "/" + path.relative_to(root).as_posix()] = {
                "sha256": sha_file(path), "bytes": path.stat().st_size}
    return result


def native_projection(model, native):
    if model == "qwen06":
        sys.path.insert(0, str(ROOT / "research/qwen6_asr/core"))
        from asr_stage.decoding import qwen_transcript
        row, metadata = qwen_transcript(native["decoded_skip_special_tokens"], native["token_ids"])
        return row["raw_text"], metadata["eos_terminated"], row["quality_flags"]
    if model == "sensevoice":
        raw = native["result"][0]["text"]
        if native["token_evidence"]["decoded"] != raw:
            raise ValueError("SenseVoice captured tokens/text disagree")
        text = re.sub(r"<\|[^<>]*\|>", "", raw)
        return text, True, [] if text else ["empty_text"]
    trace = native["pre_finalize_greedy_trace"]
    if not trace or native["effective_options"] != CONFIGS["whisper"] or native["language"] != "zh":
        raise ValueError("Whisper native language/configuration/trace differs")
    terminated = trace[-1]["completed"]
    return native["text"], terminated, [] if terminated else ["incomplete"]


def validate_cell(output, model, clip, identity, worker_sha):
    key = digest({"model_identity": identity, "input": clip})
    paths = {kind: output / kind / model / (clip["id"] + ".json") for kind in ("claims", "native", "results")}
    rows = {kind: read_json(path) for kind, path in paths.items()}
    for kind, row in rows.items():
        if row["identity_sha256"] != key or row["model"] != model or row["clip_id"] != clip["id"]:
            raise ValueError(kind + " identity differs")
    result, claim, native = rows["results"], rows["claims"], rows["native"]
    if result["status"] != "complete" or result["input"] != clip or claim["input"] != clip:
        raise ValueError("input/status differs")
    if result["model_identity"] != identity or claim["model_identity"] != identity:
        raise ValueError("model/runtime/configuration differs")
    if result["native_output_sha256"] != sha_file(paths["native"]):
        raise ValueError("native output hash differs")
    if result["decoder_source_sha256"] != worker_sha or result["boundary"] != BOUNDARY:
        raise ValueError("decoder source or evidence boundary differs")
    if claim["attempt"] != result["attempt"]:
        raise ValueError("claim/result attempt differs")
    text, terminated, flags = native_projection(model, native["native"])
    if result["text"] != text or result["decoder_terminated"] != terminated or result["quality_flags"] != flags:
        raise ValueError("consumer projection differs from native evidence")
    return result


def input_boundary_audit(inputs, worker_source):
    expected = {"schema", "source_archive_sha256", "created_at", "preparation_source_sha256", "derivation", "clips"}
    if set(inputs) != expected:
        raise ValueError("unexpected top-level decoder input context")
    tree = ast.parse(worker_source)
    qwen = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_qwen")
    calls = [node for node in ast.walk(qwen) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "_build_text_prompt"]
    if len(calls) != 1 or {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords} != {"context": "", "force_language": None}:
        raise ValueError("Qwen context boundary differs")
    sense = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_sense")
    calls = [node for node in ast.walk(sense) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "inference"]
    if len(calls) != 1 or any(kw.arg in ("context", "hotwords", "prompt", "text") for kw in calls[0].keywords):
        raise ValueError("unexpected SenseVoice text conditioning")
    if CONFIGS["whisper"]["prompt"] is not None or CONFIGS["whisper"]["prefix"] is not None:
        raise ValueError("unexpected Whisper text conditioning")
    return {"passed": True, "input_manifest_contains_only_audio_identities_and_derivation": True,
            "qwen_context": "", "qwen_force_language": None, "sensevoice_language": "auto",
            "whisper_language": "zh", "whisper_prompt": None, "whisper_prefix": None,
            "intended_text_and_peer_results_passed_to_native_recognizers": False,
            "method": "Exact persisted input/configuration checks and static native-callsite audit; not OS file-access tracing"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-bindings", type=Path)
    args = parser.parse_args(argv)
    runtime, output = args.runtime.resolve(), args.output.resolve()
    inputs = verify_inputs(args.inputs)
    job = read_json(output / "job.json")
    worker = ROOT / "tools/active_asr_worker.py"
    worker_sha = sha_file(worker)
    errors, identities = [], set()
    expected = {(model, clip["id"]) for model in MODELS for clip in inputs["clips"]}
    actual_sets = {}
    claim_pairs = Counter()
    for kind in ("claims", "native", "results"):
        paths = list((output / kind).glob("*/*.json"))
        actual_sets[kind] = {(path.parent.name, path.stem) for path in paths}
        if kind == "claims":
            for path in paths:
                row = read_json(path)
                claim_pairs[(row.get("model"), row.get("clip_id"))] += 1
        if actual_sets[kind] != expected:
            errors.append({"scope": kind, "missing": sorted(expected - actual_sets[kind]), "extra": sorted(actual_sets[kind] - expected)})
    worker_receipts = [read_json(path) for path in sorted((output / "attempts").glob("*/*.json"))
                       if path.name in {name + ".json" for name in MODELS}]
    attempts = [read_json(path) for path in sorted((output / "attempts").glob("*/summary.json"))]
    models, by_model = {}, {}
    for model in MODELS:
        rows = []
        identity = model_identity(model, runtime, args.inputs, job["identity"]["threads"])
        for clip in inputs["clips"]:
            try:
                result = validate_cell(output, model, clip, identity, worker_sha)
                if result["identity_sha256"] in identities:
                    raise ValueError("duplicate recognition identity")
                identities.add(result["identity_sha256"])
                rows.append(result)
            except Exception as error:
                errors.append({"scope": "cell", "model": model, "clip_id": clip["id"], "error": str(error)})
        receipts = [row for row in worker_receipts if row["model"] == model]
        flags = Counter(flag for row in rows for flag in row["quality_flags"])
        times = [row["elapsed_seconds"] for row in rows]
        claimed = sum(1 for m, _ in actual_sets["claims"] if m == model)
        counted = sum(row["new_calls"] for row in receipts)
        if claimed != counted:
            errors.append({"scope": "call_count", "model": model, "claims": claimed, "worker_new_calls": counted})
        models[model] = {"completed": len(rows), "claims": claimed, "worker_new_calls": counted,
            "smoke_results_reused_in_full": sum(row["reused"] for row in receipts if row["attempt"].startswith("full-")),
            "failed_worker_attempts": [{"attempt": row["attempt"], "stage": row["stage"], "error_type": row.get("error_type"),
                                        "error": row.get("error")} for row in receipts if row["status"] != "complete"],
            "quality_flags": dict(flags), "nonempty_text": sum(bool(row["text"].strip()) for row in rows),
            "decoder_terminated": sum(row["decoder_terminated"] for row in rows),
            "recognition_seconds_total": round(sum(times), 6),
            "recognition_seconds_median": statistics.median(times) if times else None,
            "recognition_seconds_max": max(times) if times else None,
            "load_seconds_total": round(sum(row.get("load_seconds", 0) for row in receipts), 6),
            "max_process_rss_kib": max((row.get("max_rss_kib", 0) for row in receipts), default=0),
            "effective_threads": sorted({row["effective_threads"] for row in receipts if "effective_threads" in row}),
            "effective_interop_threads": sorted({row["effective_interop_threads"] for row in receipts if "effective_interop_threads" in row})}
        by_model[model] = {row["clip_id"]: row["text"] for row in rows}
    duplicates = sum(count - 1 for count in claim_pairs.values() if count > 1)
    if duplicates:
        errors.append({"scope": "duplicate_claims", "count": duplicates})
    if sum(row["claims"] for row in models.values()) > job["identity"]["max_calls"]:
        errors.append({"scope": "budget", "error": "call cap exceeded"})
    for clip in inputs["clips"]:
        original = args.inputs.parent / clip["original_wav"]["path"]
        if sha_file(original) != clip["original_wav"]["sha256"] or original.stat().st_size != clip["original_wav"]["bytes"]:
            errors.append({"scope": "original_audio", "clip_id": clip["id"]})
    boundary = input_boundary_audit(inputs, worker.read_text())
    files = inventory(args.inputs.parent, "inputs")
    for kind in ("claims", "native", "results", "attempts"):
        files.update(inventory(output / kind, kind))
    files["job.json"] = {"sha256": sha_file(output / "job.json"), "bytes": (output / "job.json").stat().st_size}
    bindings_summary = None
    if args.source_bindings:
        bindings = read_json(args.source_bindings)
        if bindings["inputs_manifest_sha256"] != sha_file(args.inputs) or bindings["target_text_included"] is not False or bindings["not_decoder_input"] is not True:
            raise ValueError("source bindings must remain separate from decoder inputs")
        if {row["clip_id"] for row in bindings["rows"]} != {row["id"] for row in inputs["clips"]}:
            raise ValueError("source bindings do not cover inputs")
        bindings_summary = {"sha256": sha_file(args.source_bindings), "rows": len(bindings["rows"]),
                            "speaker_presets": sorted({row["speaker_preset_name"] for row in bindings["rows"]}),
                            "sentence_ids_present": sum(row["sentence_id"] is not None for row in bindings["rows"])}
        files["metadata/source-bindings.json"] = {"sha256": sha_file(args.source_bindings), "bytes": args.source_bindings.stat().st_size}
    audit_dir = output / "audit"
    audit_dir.mkdir(exist_ok=False)
    hashes = {"schema": "active-asr-file-inventory-v1", "roots": {"inputs": str(args.inputs.parent.resolve()), "results": str(output)},
              "file_count": len(files), "files_sha256": digest(files), "files": files}
    write_once(audit_dir / "file-hashes.json", hashes)
    finish_times = [datetime.fromisoformat(row["finished_at"]) for row in attempts if "finished_at" in row]
    finished = max(finish_times) if finish_times else datetime.now(timezone.utc)
    receipt = {"schema": "active-asr-batch-final-audit-v1", "created_at": utc(),
        "status": "complete" if not errors and len(identities) == len(expected) else "incomplete",
        "source_archive_sha256": inputs["source_archive_sha256"], "input_manifest_sha256": sha_file(args.inputs),
        "inputs": len(inputs["clips"]), "models": models, "expected_matrix_cells": len(expected),
        "verified_matrix_cells": len(identities), "original_recognition_calls": len(actual_sets["claims"]),
        "call_budget": job["identity"]["max_calls"], "duplicate_model_clip_pairs": duplicates,
        "job_created_at": job["created_at"], "finished_at": finished.isoformat(),
        "job_wall_seconds": round((finished - datetime.fromisoformat(job["created_at"])).total_seconds(), 6),
        "persisted_deadline_unix": job["deadline_unix"], "finished_before_deadline": finished.timestamp() <= job["deadline_unix"],
        "heavy_model_concurrency_limit": job["identity"]["heavy_model_concurrency"],
        "max_model_process_rss_kib": max(row["max_process_rss_kib"] for row in models.values()),
        "resource_measurement_scope": "Per-model-process getrusage peak RSS; not aggregate container peak or CPU-seconds",
        "target_context_audit": boundary, "source_bindings": bindings_summary,
        "decoder_source_sha256": worker_sha, "auditor_source_sha256": sha_file(__file__),
        "inventory_sha256": sha_file(audit_dir / "file-hashes.json"), "inventory_file_count": len(files),
        "raw_text_agreement": {"method": "Exact native consumer strings, no normalization; agreement is not accuracy",
            "all_three": sum(len({by_model[model][clip["id"]] for model in MODELS}) == 1
                             for clip in inputs["clips"] if all(clip["id"] in by_model[model] for model in MODELS)),
            "pairs": {left + ":" + right: sum(by_model[left][key] == by_model[right][key]
                       for key in set(by_model[left]) & set(by_model[right]))
                      for left, right in (("qwen06", "sensevoice"), ("qwen06", "whisper"), ("sensevoice", "whisper"))}},
        "errors": errors, "boundary": BOUNDARY}
    write_once(audit_dir / "final-summary.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key not in ("models", "boundary", "target_context_audit")}, sort_keys=True))
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
