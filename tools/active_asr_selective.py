#!/usr/bin/env python3
"""Bounded v2 ASR: two audio-only primaries and predeclared selective Whisper.

The original native worker and v1 audio/result formats are unchanged. Routing
metadata never enters the native decoder command. Unrequested is not failed,
negative, or a third agreeing observation. An interrupted call is never retried.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_asr_batch as batch
import research_no_network
from research_job_lock import heavy_model_slot
from setup_asr_runtime import ROOT, MODEL_LOCK, WHISPER_SHA, WHISPER_SOURCE, safe_relative, save, sha_file
import summarize_active_asr_batch as native_audit

PRIMARY = ("qwen06", "sensevoice")
POLICY_PATH = ROOT / "configs/research/selective-asr-v2.json"
WORKER = ROOT / "tools/active_asr_worker.py"
NAMESPACE = "kws-d1-whisper-agreement-audit-v2"
POLICY = {
    "schema": "active-asr-selective-policy-v2",
    "primary_models": list(PRIMARY), "selective_model": "whisper",
    "normalization": "reference-text-NFKC-casefold-punctuation-CJK-spaces-v1",
    "mandatory_reasons": ["primary_incomplete", "primary_quality_uncertain",
                          "primary_disagreement", "source_quality_uncertain"],
    "agreement_audit": {"namespace": NAMESPACE, "seed": 20261105,
        "hash_input": "UTF8(namespace|seed|plan_sha256|source_id)",
        "hash_algorithm": "sha256", "selection": "int(hash,16)*5 < 2**256",
        "probability_numerator": 1, "probability_denominator": 5},
    "no_target_or_peer_text_to_native_worker": True,
    "missing_requested_result_blocks_complete_audit": True,
    "quality_retries": 0, "human_gold": False,
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def normalize(text):
    """Identical orthographic fold to kws_data_contract.reference_text.normalize."""
    value = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    value = " ".join("".join(
        " " if unicodedata.category(c).startswith("P") and c != "'" else c
        for c in value).split())
    return re.sub(r"(?<=[\u3400-\u4dbf\u4e00-\u9fff]) +"
                  r"(?=[\u3400-\u4dbf\u4e00-\u9fff])", "", value)


def is_sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def plain_file(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file(),
            "pinned file must be a regular non-symlink file")
    return path


def source_identity():
    names = ["tools/active_asr_selective.py", "tools/active_asr_batch.py",
             "tools/active_asr_worker.py", "tools/summarize_active_asr_batch.py",
             "tools/research_runtime.py", "tools/research_job_lock.py",
             "tools/research_no_network.py", "tools/setup_asr_runtime.py",
             "configs/runtime/research-runtime.json", "research/qwen6_asr/model-locks.json",
             "research/source-screen32-whisper-v1/source-lock.json"]
    names += [p.relative_to(ROOT).as_posix() for folder in ("asr_stage", "vendor") for p in
              sorted((ROOT / "research/qwen6_asr/core" / folder).rglob("*.py"))]
    return {name: sha_file(plain_file(ROOT / name)) for name in names}


def validate_map(inputs_path, map_path):
    inputs = batch.verify_inputs(inputs_path)
    native_audit.input_boundary_audit(inputs, WORKER.read_text(encoding="utf-8"))
    mapping = batch.read_json(map_path)
    require(type(mapping) is dict and set(mapping) == {
        "schema", "input_manifest_sha256", "source_archive_sha256", "plan_sha256", "rows"}
        and mapping["schema"] == "active-asr-source-map-v2", "source map v2 fields")
    require(mapping["input_manifest_sha256"] == sha_file(inputs_path)
            and mapping["source_archive_sha256"] == inputs["source_archive_sha256"]
            and is_sha(mapping["plan_sha256"]), "source map input/plan identity")
    rows = mapping["rows"]
    require(type(rows) is list and len(rows) == len(inputs["clips"]), "source map coverage")
    by_id, source_ids = {}, set()
    for row in rows:
        require(type(row) is dict and set(row) == {
            "clip_id", "source_id", "original_wav_sha256", "original_pcm_sha256",
            "derived_pcm_sha256", "quality_uncertain", "quality_reasons"}, "source map row fields")
        require(row["clip_id"] not in by_id and type(row["source_id"]) is str
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", row["source_id"])
                and row["source_id"] not in source_ids, "source map duplicate/invalid identity")
        reasons = row["quality_reasons"]
        require(type(row["quality_uncertain"]) is bool and type(reasons) is list
                and all(type(x) is str and 1 <= len(x) <= 160 for x in reasons)
                and len(reasons) == len(set(reasons))
                and row["quality_uncertain"] == bool(reasons), "source quality must be explicit")
        by_id[row["clip_id"]] = row
        source_ids.add(row["source_id"])
    require(set(by_id) == {c["id"] for c in inputs["clips"]}, "source map clip coverage")
    for clip in inputs["clips"]:
        row = by_id[clip["id"]]
        for key in ("original_wav", "original_pcm", "derived_pcm"):
            require(is_sha(row[key + "_sha256"]) and row[key + "_sha256"] == clip[key]["sha256"],
                    "source map audio identity")
        original = plain_file(Path(inputs_path).parent / safe_relative(clip["original_wav"]["path"]))
        require(sha_file(original) == clip["original_wav"]["sha256"]
                and original.stat().st_size == clip["original_wav"]["bytes"], "original WAV changed")
        plain_file(Path(inputs_path).parent / safe_relative(clip["derived_pcm"]["path"]))
    return inputs, mapping


def agreement_hash(plan_sha256, source_id):
    body = f"{NAMESPACE}|20261105|{plan_sha256}|{source_id}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def route_item(source, plan_sha256, primary):
    reasons, texts, hashes = [], {}, {}
    for model in PRIMARY:
        item = primary[model]
        result = item["result"]
        hashes[model] = item["sha256"]
        if result is None:
            require(item["sha256"] is None, "missing primary cannot have a result hash")
            reasons.append("primary_incomplete")
            continue
        require(is_sha(item["sha256"]) and type(result["text"]) is str
                and type(result["decoder_terminated"]) is bool
                and type(result["quality_flags"]) is list, "invalid primary result projection")
        texts[model] = normalize(result["text"])
        if not result["decoder_terminated"] or not texts[model]:
            reasons.append("primary_incomplete")
        if result["quality_flags"]:
            reasons.append("primary_quality_uncertain")
    if len(texts) == 2 and texts[PRIMARY[0]] != texts[PRIMARY[1]]:
        reasons.append("primary_disagreement")
    if source["quality_uncertain"]:
        reasons.append("source_quality_uncertain")
    sample = agreement_hash(plan_sha256, source["source_id"])
    reasons = [r for r in POLICY["mandatory_reasons"] if r in reasons]
    if not reasons:
        reasons = ["deterministic_20pct" if int(sample, 16) * 5 < 2**256
                   else "two_primary_agree_not_sampled"]
    return {"clip_id": source["clip_id"], "source_id": source["source_id"],
            "primary_result_sha256": hashes, "sample_hash": sample,
            "whisper_state": "not_requested" if reasons == ["two_primary_agree_not_sampled"] else "requested",
            "reasons": reasons}


def read_primary(output, clips, identities, worker_sha):
    found = {}
    for clip in clips:
        found[clip["id"]] = {}
        for model in PRIMARY:
            state, _ = batch.existing_result(output, model, clip, identities[model])
            result = native_audit.validate_cell(output, model, clip, identities[model], worker_sha) if state == "complete" else None
            found[clip["id"]][model] = {"result": result, "sha256":
                sha_file(output / "results" / model / (clip["id"] + ".json")) if result is not None else None}
    return found


def routing_for(request, inputs, mapping, identities):
    found = read_primary(Path(request["output"]), inputs["clips"], identities, request["worker_sha256"])
    return {"schema": "active-asr-selective-routing-v2",
            "input_manifest_sha256": request["inputs"]["sha256"],
            "source_map_sha256": request["source_map"]["sha256"],
            "policy_sha256": request["policy"]["sha256"], "plan_sha256": mapping["plan_sha256"],
            "decoder_source_sha256": request["worker_sha256"],
            "items": [route_item(row, mapping["plan_sha256"], found[row["clip_id"]]) for row in mapping["rows"]]}


def declare(args):
    inputs, _ = validate_map(args.inputs, args.source_map)
    require(batch.read_json(args.policy) == POLICY, "predeclared selective policy differs")
    require(1 <= args.threads <= 4 and 1 <= args.budget_seconds <= 600, "bounded CPU/job budget required")
    runtime, output = args.runtime.resolve(), args.output.absolute()
    require(not output.exists(), "new selective output required")
    identities = {m: batch.model_identity(m, runtime, args.inputs, args.threads) for m in batch.MODELS}
    request = {"schema": "active-asr-selective-request-v2", "created_at": batch.utc(),
        "runtime": str(runtime), "output": str(output),
        "inputs": {"path": str(plain_file(args.inputs)), "sha256": sha_file(args.inputs)},
        "source_map": {"path": str(plain_file(args.source_map)), "sha256": sha_file(args.source_map)},
        "policy": {"path": str(plain_file(args.policy)), "sha256": sha_file(args.policy)},
        "threads": args.threads, "budget_seconds": args.budget_seconds,
        "max_calls": 3 * len(inputs["clips"]), "worker_sha256": sha_file(WORKER),
        "source_files": source_identity(), "model_identities": identities, "boundary": batch.BOUNDARY}
    batch.write_once(args.request, request)
    return request


def verify_request(path):
    request = batch.read_json(plain_file(path))
    require(Path(path).read_bytes() == batch.canonical(request) + b"\n", "canonical request bytes required")
    require(type(request) is dict and set(request) == {
        "schema", "created_at", "runtime", "output", "inputs", "source_map", "policy", "threads",
        "budget_seconds", "max_calls", "worker_sha256", "source_files", "model_identities", "boundary"}
        and request["schema"] == "active-asr-selective-request-v2", "selective request fields")
    for key in ("inputs", "source_map", "policy"):
        row = request[key]
        require(set(row) == {"path", "sha256"} and sha_file(plain_file(row["path"])) == row["sha256"],
                "selective request pinned input changed")
    require(batch.read_json(request["policy"]["path"]) == POLICY, "predeclared selective policy differs")
    require(type(request["threads"]) is int and 1 <= request["threads"] <= 4
            and type(request["budget_seconds"]) is int and 1 <= request["budget_seconds"] <= 600,
            "bounded CPU/job budget required")
    inputs, mapping = validate_map(request["inputs"]["path"], request["source_map"]["path"])
    require(request["max_calls"] == 3 * len(inputs["clips"]) and request["boundary"] == batch.BOUNDARY
            and request["worker_sha256"] == sha_file(WORKER)
            and request["source_files"] == source_identity(), "selective source/boundary changed")
    identities = {m: batch.model_identity(m, Path(request["runtime"]), Path(request["inputs"]["path"]),
                                        request["threads"]) for m in batch.MODELS}
    require(identities == request["model_identities"], "selective runtime/model identity changed")
    return request, inputs, mapping, identities


def expected_matrix(inputs, routing):
    expected = {(m, c["id"]) for m in PRIMARY for c in inputs["clips"]}
    skipped = set()
    for row in routing["items"]:
        (expected if row["whisper_state"] == "requested" else skipped).add(("whisper", row["clip_id"]))
    return expected, skipped


def validate_load(model, loaded, assets):
    """Re-read the native loader's persisted checkpoint/application evidence."""
    require(loaded.get("dtype") == "float32" and loaded.get("device") == "cpu", "native load dtype/device")
    if model == "whisper":
        revision = batch.read_json(WHISPER_SOURCE)["source_revision"]
        require(assets == {"checkpoint_sha256": WHISPER_SHA, "source_revision": revision}
                and loaded.get("checkpoint_sha256") == WHISPER_SHA
                and loaded.get("native_source_revision") == revision
                and loaded.get("dimensions") == {"n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 768,
                    "n_audio_head": 12, "n_audio_layer": 12, "n_vocab": 51865, "n_text_ctx": 448,
                    "n_text_state": 768, "n_text_head": 12, "n_text_layer": 12}, "native Whisper checkpoint/source/geometry")
        return
    spec = batch.read_json(MODEL_LOCK)[model]
    require(assets == {"asset_lock_sha256": spec["asset_lock_sha256"],
                       "source_lock_sha256": spec["source_lock_sha256"]}, "native loaded asset/source locks")
    sys.path.insert(0, str(ROOT / "research/qwen6_asr/core"))
    from asr_stage.architecture import qwen06_schema, sensevoice_schema
    schema = qwen06_schema() if model == "qwen06" else sensevoice_schema()
    verification = loaded["loaded_state_verification"]
    require(verification["complete_checkpoint_verification_passed"] is True, "native full checkpoint verification")
    actual = verification["verified_loaded_state"]
    require(set(actual) == set(schema), "native loaded complete tensor roster")
    for key, row in actual.items():
        require(set(row) == {"shape", "dtype", "sha256"} and
                {k: row[k] for k in ("shape", "dtype")} == schema[key] and is_sha(row["sha256"]),
                "native loaded tensor description")
    if model == "qwen06":
        clean = {k: [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
        checkpoint = next(r["sha256"] for r in spec["asset_lock"]["files"] if r["filename"] == "model.safetensors")
        evidence = verification["checkpoint_evidence"]
        require(verification["schema"] == "qwen-complete-load-receipt-v1" and loaded["loading_info"] == clean
                and verification["loading_info"] == clean and evidence["checkpoint_file_sha256"] == checkpoint
                and evidence["expected_state_schema"] == schema
                and evidence["expected_state_schema_sha256"] == batch.digest(schema)
                and evidence["approved_aliases"] == [] and verification["verified_alias_proofs"] == []
                and set(evidence["tensors"]) == set(schema), "native Qwen checkpoint application")
        for key, row in actual.items():
            require({k: evidence["tensors"][key][k] for k in ("shape", "dtype", "sha256")} == row,
                    "native Qwen loaded tensor differs from checkpoint")
        attention = loaded["effective_attention"]
        names = {f"thinker.audio_tower.layers.{i}.self_attn" for i in range(18)} | {
                 f"thinker.model.layers.{i}.self_attn" for i in range(28)}
        require(len(attention) == len(names) and {r["name"] for r in attention} == names
                and all(r["implementation"] == "eager" for r in attention), "native Qwen eager attention roster")
    else:
        contract = verification["contract"]
        require(verification["schema"] == "sensevoice-exact-apply-receipt-v1"
                and verification["loading_info"] == {"missing_keys": [], "unexpected_keys": []}
                and contract["expected_schema"] == schema and contract["expected_schema_sha256"] == batch.digest(schema)
                and contract["identity_key_mapping_only"] is True and actual == contract["checkpoint_state"]
                and contract["checkpoint_state_sha256"] == batch.digest(actual)
                and verification["verified_loaded_state_sha256"] == batch.digest(actual)
                and loaded["frontend"] == "kaldi-native-fbank" and loaded["dither"] == 0.0
                and loaded["no_vad"] is True and loaded["use_itn"] is False, "native SenseVoice checkpoint/frontend")


def audit_state(request, inputs, mapping, identities):
    output = Path(request["output"])
    routing = batch.read_json(output / "routing.json")
    require(routing == routing_for(request, inputs, mapping, identities), "routing differs from native primaries/source/hash")
    expected, skipped = expected_matrix(inputs, routing)
    clips = {c["id"]: c for c in inputs["clips"]}
    errors, verified, actual = [], set(), {}
    for kind in ("claims", "native", "results"):
        paths = list((output / kind).glob("*/*.json"))
        actual[kind] = {(p.parent.name, p.stem) for p in paths}
        if actual[kind] != expected:
            errors.append({"scope": kind, "missing": sorted(expected - actual[kind]),
                           "extra": sorted(actual[kind] - expected)})
    for model, clip_id in sorted(expected):
        try:
            state, _ = batch.existing_result(output, model, clips[clip_id], identities[model])
            require(state == "complete", "requested result is incomplete/consumed")
            native_audit.validate_cell(output, model, clips[clip_id], identities[model], request["worker_sha256"])
            verified.add((model, clip_id))
        except (ValueError, KeyError, FileNotFoundError) as error:
            errors.append({"scope": "cell", "model": model, "clip_id": clip_id, "error": str(error)})
    job = batch.read_json(output / "job.json")
    require(job["identity"] == {"request_sha256": sha_file(output / "request.json"),
            "max_calls": request["max_calls"], "heavy_model_concurrency": 1,
            "threads": request["threads"]}, "selective job identity changed")
    receipts = [batch.read_json(p) for p in sorted((output / "attempts").glob("*/*.json"))
                if p.name in {m + ".json" for m in batch.MODELS}]
    by_model = {}
    for model in batch.MODELS:
        rows = [r for r in receipts if r["model"] == model]
        selected = {clip_id for m, clip_id in expected if m == model}
        if len(rows) != (1 if selected else 0):
            errors.append({"scope": "worker_receipt_count", "model": model})
        for row in rows:
            if (row.get("schema") != "active-asr-worker-v1" or row.get("identity") != identities[model]
                    or row.get("source_sha256") != request["worker_sha256"]
                    or row.get("attempt") != "selective-001" or set(row.get("selected", [])) != selected
                    or len(row.get("selected", [])) != len(selected) or row.get("reused") != 0
                    or row.get("unresolved_prior_calls") != [] or row.get("boundary") != batch.BOUNDARY
                    or row.get("heavy_lock", {}).get("inherited") is not True
                    or row.get("heavy_lock", {}).get("supervised") is not True
                    or row.get("heavy_lock", {}).get("process_group") != job["supervision"]["process_group"]):
                errors.append({"scope": "worker_identity_or_supervision", "model": model})
            try:
                validate_load(model, batch.read_json(output / "attempts/selective-001" / (model + "-load.json")), row.get("assets"))
            except (ValueError, KeyError, FileNotFoundError, TypeError) as error:
                errors.append({"scope": "native_model_load", "model": model, "error": str(error)})
        if not selected and (output / "attempts/selective-001" / (model + "-load.json")).exists():
            errors.append({"scope": "unrequested_model_load", "model": model})
        counted = sum(r["new_calls"] for r in rows)
        claims = sum(m == model for m, _ in actual["claims"])
        if counted != claims or any(r["status"] != "complete" for r in rows):
            errors.append({"scope": "worker_receipts", "model": model, "claims": claims, "new_calls": counted})
        by_model[model] = {"requested": sum(m == model for m, _ in expected),
            "not_requested": sum(m == model for m, _ in skipped), "verified": sum(m == model for m, _ in verified),
            "claims": claims, "worker_new_calls": counted}
    guard = batch.read_json(output / "network-guard.json")
    require(guard.get("seccomp_installed") is True and guard.get("no_new_privileges") is True
            and guard.get("environment") == research_no_network.ENVIRONMENT, "native network guard missing/different")
    pairs = Counter()
    for path in (output / "claims").glob("*/*.json"):
        claim = batch.read_json(path)
        pairs[(claim.get("model"), claim.get("clip_id"))] += 1
    duplicates = sum(n - 1 for n in pairs.values() if n > 1)
    if duplicates or len(actual["claims"]) > request["max_calls"]:
        errors.append({"scope": "call_budget_or_duplicates"})
    attempt = batch.read_json(output / "attempts/selective-001/summary.json")
    if attempt["status"] != "complete":
        errors.append({"scope": "controller_attempt", "status": attempt["status"]})
    finished_before_deadline = datetime.fromisoformat(attempt["finished_at"]).timestamp() <= job["deadline_unix"]
    if not finished_before_deadline:
        errors.append({"scope": "persisted_deadline_exhausted"})
    return {"status": "complete" if not errors and verified == expected else "incomplete",
        "input_manifest_sha256": request["inputs"]["sha256"], "source_map_sha256": request["source_map"]["sha256"],
        "policy_sha256": request["policy"]["sha256"], "routing_sha256": sha_file(output / "routing.json"),
        "decoder_source_sha256": request["worker_sha256"], "inputs": len(clips),
        "requested_matrix": [list(x) for x in sorted(expected)],
        "not_requested_matrix": [list(x) for x in sorted(skipped)],
        "expected_matrix_cells": len(expected), "verified_matrix_cells": len(verified),
        "original_recognition_calls": len(actual["claims"]), "call_budget": request["max_calls"],
        "duplicate_model_clip_pairs": duplicates, "models": by_model, "errors": errors,
        "finished_before_deadline": finished_before_deadline,
        "boundary": batch.BOUNDARY, "unrequested_is_not_a_negative_or_agreement": True}


def evidence_inventory(request):
    files = native_audit.inventory(Path(request["inputs"]["path"]).parent, "inputs")
    files.update({k: v for k, v in native_audit.inventory(Path(request["output"]), "output").items()
                  if not k.startswith("output/audit/")})
    for name in ("source_map", "policy"):
        path = Path(request[name]["path"])
        files["metadata/" + name + ".json"] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    return files


def seal_or_verify(request_path):
    request, inputs, mapping, identities = verify_request(request_path)
    output = Path(request["output"])
    require((output / "request.json").read_bytes() == Path(request_path).read_bytes(), "persisted request changed")
    state = audit_state(request, inputs, mapping, identities)
    files = evidence_inventory(request)
    audit = output / "audit"
    final = audit / "final-summary.json"
    if final.exists():
        receipt = batch.read_json(final)
        inventory = batch.read_json(audit / "file-hashes.json")
        require(receipt["schema"] == "active-asr-selective-final-audit-v2"
                and receipt["request_sha256"] == sha_file(request_path)
                and receipt["state"] == state and receipt["inventory_sha256"] == sha_file(audit / "file-hashes.json")
                and inventory["files"] == files and inventory["files_sha256"] == batch.digest(files),
                "sealed selective evidence changed")
        return receipt
    audit.mkdir(exist_ok=True)
    hashes = {"schema": "active-asr-selective-inventory-v2",
              "files_sha256": batch.digest(files), "files": files, "file_count": len(files)}
    inventory_path = audit / "file-hashes.json"
    if inventory_path.exists():
        require(inventory_path.read_bytes() == batch.canonical(hashes) + b"\n", "partial audit inventory changed")
    else:
        batch.write_once(inventory_path, hashes)
    receipt = {"schema": "active-asr-selective-final-audit-v2", "created_at": batch.utc(),
        "request_sha256": sha_file(request_path), "inventory_sha256": sha_file(audit / "file-hashes.json"),
        "state": state}
    batch.write_once(final, receipt)
    return receipt


def worker_command(request, model, ids):
    require(model in batch.MODELS and ids and len(ids) == len(set(ids)), "invalid native worker selection")
    return [sys.executable, "-B", str(WORKER), "--model", model, "--runtime", request["runtime"],
            "--inputs", request["inputs"]["path"], "--output", request["output"],
            "--attempt", "selective-001", "--threads", str(request["threads"]), "--ids", ",".join(ids)]


def run(request_path):
    request, inputs, mapping, identities = verify_request(request_path)
    output = Path(request["output"])
    if (output / "audit/final-summary.json").is_file():
        receipt = seal_or_verify(request_path)
        require(receipt["state"]["status"] == "complete", "incomplete attempts are not retried")
        return receipt
    require(not output.exists(), "unsealed/interrupted selective job is consumed; no automatic retry")
    batch.research_runtime.check_interpreter(batch.research_runtime.load_manifest())
    runtime = Path(request["runtime"])
    with heavy_model_slot(runtime / "heavy-model.lock") as lease:
        require(lease.inherited and lease.supervised, "selective ASR requires a bounded supervised queue lease")
        require(os.environ.get("KWS_RESEARCH_RECIPE_SHA256") == sha_file(request_path), "queue recipe binding differs")
        research_no_network.set_offline_environment()
        guard = research_no_network.install()
        output.mkdir(parents=True, exist_ok=False)
        batch.write_once(output / "request.json", request)
        batch.write_once(output / "network-guard.json", guard)
        job = {"schema": "active-asr-selective-job-v2", "created_at": batch.utc(),
               "deadline_unix": time.time() + request["budget_seconds"], "identity": {
                   "request_sha256": sha_file(request_path), "max_calls": request["max_calls"],
                   "heavy_model_concurrency": 1, "threads": request["threads"]},
               "supervision": {"process_id": os.getpid(), "process_group": os.getpgrp(),
                               "inherited_lease": lease.inherited, "supervised": lease.supervised}}
        batch.write_once(output / "job.json", job)
        attempt_dir = output / "attempts/selective-001"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        summary = {"schema": "active-asr-selective-attempt-v2", "started_at": batch.utc(),
                   "status": "running", "models": {}}
        save(attempt_dir / "summary.json", summary)
        env = lease.child_environment()
        env.update(OMP_NUM_THREADS=str(request["threads"]), MKL_NUM_THREADS=str(request["threads"]),
                   OPENBLAS_NUM_THREADS=str(request["threads"]), NUMEXPR_NUM_THREADS=str(request["threads"]),
                   HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
        ids = [c["id"] for c in inputs["clips"]]

        def invoke(model, selected):
            if not selected:
                summary["models"][model] = {"state": "not_requested", "new_calls": 0}
            elif job["deadline_unix"] <= time.time():
                summary["models"][model] = {"state": "budget_exhausted"}
            else:
                with (attempt_dir / (model + ".log")).open("x") as log:
                    started = time.monotonic()
                    code = batch.run_worker(worker_command(request, model, selected), env, log,
                                            runtime, job["deadline_unix"] - time.time())
                summary["models"][model] = {"state": "complete" if code == 0 else "failed",
                    "exit_code": code, "selected_ids": selected, "elapsed_seconds": time.monotonic() - started}
            save(attempt_dir / "summary.json", summary)
            print(json.dumps({"model": model, **summary["models"][model]}, ensure_ascii=False), flush=True)

        for model in PRIMARY:
            invoke(model, ids)
        routing = routing_for(request, inputs, mapping, identities)
        batch.write_once(output / "routing.json", routing)
        invoke("whisper", [r["clip_id"] for r in routing["items"] if r["whisper_state"] == "requested"])
        summary.update(status="complete" if all(r["state"] in ("complete", "not_requested")
            for r in summary["models"].values()) else "incomplete", finished_at=batch.utc())
        save(attempt_dir / "summary.json", summary)
    return seal_or_verify(request_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("declare")
    for flag in ("runtime", "inputs", "source-map", "output", "request"):
        prep.add_argument("--" + flag, type=Path, required=True)
    prep.add_argument("--policy", type=Path, default=POLICY_PATH)
    prep.add_argument("--threads", type=int, default=4)
    prep.add_argument("--budget-seconds", type=int, default=570)
    for command in ("run", "audit"):
        commands.add_parser(command).add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "declare":
        result = declare(args)
        print(json.dumps({"request": str(args.request), "sha256": sha_file(args.request), "max_calls": result["max_calls"]}))
        return 0
    result = run(args.request) if args.command == "run" else seal_or_verify(args.request)
    print(json.dumps({"schema": result["schema"], **result["state"]}, ensure_ascii=False))
    return 0 if result["state"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
