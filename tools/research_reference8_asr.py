#!/usr/bin/env python3
"""One queue-supervised eight-reference ASR batch using the existing smoke/full runner."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_asr_batch as batch
import active_asr_selective as selective
import research_d2_checkpoints as cp
from research_job_lock import heavy_model_slot

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "kws-eight-reference-three-asr-v1"

def require(ok, message):
    if not ok:
        raise ValueError(message)

def phase_contract(clips):
    ids = [x["id"] for x in clips]
    require(len(ids) == len(set(ids)) == 8, "EXACTLY_EIGHT_REFERENCES")
    return [
        {"attempt": "smoke-001", "phase": "smoke", "selected_ids": [ids[0], ids[-1]], "max_new_calls": 6},
        {"attempt": "full-001", "phase": "full", "selected_ids": ids, "max_new_calls": 18},
    ]

def source_records():
    names = set(selective.source_identity())
    names.update(("tools/research_reference8_asr.py", "tools/research_d2_checkpoints.py",
                  "tools/research_queue.py"))
    return {name: cp.pin(ROOT / name) for name in sorted(names)}

def verify_payload(inputs_path, bindings_path, sources_path, exclusions_path):
    inputs = batch.verify_inputs(inputs_path)
    selective.native_audit.input_boundary_audit(inputs, (ROOT / "tools/active_asr_worker.py").read_text())
    bindings, source, exclusions = (batch.read_json(p) for p in
                                    (bindings_path, sources_path, exclusions_path))
    require(bindings["inputs_manifest_sha256"] == batch.sha_file(inputs_path)
            and bindings["source_manifest_sha256"] == batch.sha_file(sources_path)
            and bindings["not_decoder_input"] is True and bindings["target_text_included"] is False,
            "REFERENCE_BINDINGS_CHANGED")
    refs = {r["reference_id"]: r for r in source["references"]}
    require(len(refs) == len(source["references"]) == len(bindings["rows"]) == 8,
            "EIGHT_SOURCE_REFERENCE_JOIN")
    require({x["clip_id"] for x in bindings["rows"]} == {x["id"] for x in inputs["clips"]},
            "REFERENCE_CLIP_ROSTER")
    require(Counter(r["role"] for r in refs.values()) == {"train-candidate": 6, "development-only": 2}
            and len({r["source_family_id"] for r in refs.values()}) == 8, "FROZEN_NEW_SOURCE_SPLIT")
    old = exclusions["rows"]
    require(len(old) == 32, "ORIGINAL32_EXCLUSION_ROSTER")
    for row in bindings["rows"]:
        ref = refs[row["reference_id"]]
        require(all(row[k] == ref[k] for k in ("source_family_id", "speaker_group_id", "role"))
                and row["normalized_wav"]["sha256"] == ref["wav_sha256"]
                and row["normalized_pcm"]["sha256"] == ref["pcm_sha256"]
                and row["original_wav_sha256"] == ref["parent_source_sha256"], "REFERENCE_SOURCE_IDENTITY")
        require(ref["source_family_id"] not in {x["source_family_id"] for x in old}
                and ref["parent_source_sha256"] not in {x["original_sha256"] for x in old}
                and ref["wav_sha256"] not in {x["wav_sha256"] for x in old}
                and ref["pcm_sha256"] not in {x["pcm_sha256"] for x in old}, "OLD32_REFERENCE_OVERLAP")
    return inputs, bindings

def declare(args):
    inputs, _ = verify_payload(args.inputs, args.bindings, args.sources, args.exclusions)
    cp.verify_ack(args.input_checkpoint)
    require(not args.output.exists() and not args.request.exists(), "NEW_ASR_BATCH_REQUIRED")
    runtime = args.runtime.resolve()
    recipe = {
        "schema": SCHEMA, "created_at": batch.utc(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inputs": cp.pin(args.inputs), "bindings": cp.pin(args.bindings), "sources": cp.pin(args.sources),
        "exclusions": cp.pin(args.exclusions), "runtime": str(runtime), "runtime_setup": cp.pin(runtime / "setup.json"),
        "output": str(args.output.absolute()), "models": list(batch.MODELS), "threads": 4,
        "budget_minutes": 9, "absolute_decoder_budget_seconds": 540, "queue_timeout_seconds": 600,
        "max_calls": 24, "automatic_retries": 0, "phases": phase_contract(inputs["clips"]),
        "source_files": source_records(),
        "model_identities": {model: batch.model_identity(model, runtime, args.inputs, 4) for model in batch.MODELS},
        "input_checkpoint": str(args.input_checkpoint.absolute()),
        "input_descriptor": cp.pin(args.input_checkpoint / "capsule.json"),
        "input_ack": cp.pin(args.input_checkpoint / "ack.json"),
        "preparation_checkpoint": str(args.preparation_checkpoint.absolute()),
        "deadline_policy": "One native output/job.json created by smoke; full must reuse its identical SHA and absolute deadline.",
        "human_gold": False, "independent_test": False, "training_admitted": False,
    }
    batch.write_once(args.request, recipe)
    return {"status": "declared_not_run", "request": cp.pin(args.request), "max_calls": 24, "new_model_calls": 0}

def verify_request(path):
    recipe = batch.read_json(path)
    require(recipe["schema"] == SCHEMA and recipe["models"] == list(batch.MODELS)
            and recipe["threads"] == 4 and recipe["budget_minutes"] == 9
            and recipe["absolute_decoder_budget_seconds"] == 540 and recipe["queue_timeout_seconds"] == 600
            and recipe["max_calls"] == 24 and recipe["automatic_retries"] == 0, "FIXED_ASR_CONTRACT")
    for field in ("inputs", "bindings", "sources", "exclusions", "runtime_setup", "input_descriptor", "input_ack"):
        require(cp.pin(recipe[field]["path"]) == recipe[field], "FROZEN_INPUT_CHANGED:" + field)
    require(source_records() == recipe["source_files"], "FROZEN_EXECUTION_SOURCE_CHANGED")
    inputs, _ = verify_payload(*(Path(recipe[k]["path"]) for k in ("inputs", "bindings", "sources", "exclusions")))
    require(recipe["phases"] == phase_contract(inputs["clips"]), "SMOKE_FULL_ROSTER_CHANGED")
    identities = {m: batch.model_identity(m, Path(recipe["runtime"]), Path(recipe["inputs"]["path"]), 4)
                  for m in batch.MODELS}
    require(identities == recipe["model_identities"], "MODEL_OR_RUNTIME_IDENTITY_CHANGED")
    return recipe, inputs

def run(path):
    recipe, inputs = verify_request(path)
    cp.verify_ack(Path(recipe["input_checkpoint"]))
    prep = Path(recipe["preparation_checkpoint"])
    cp.verify_ack(prep)
    require(cp.read(prep / "capsule.json")["context"]["request"] == cp.pin(path),
            "ACTUAL_PREPARATION_ACK_MUST_BIND_THIS_REQUEST")
    require(os.environ.get("KWS_RESEARCH_RECIPE_SHA256") == batch.sha_file(path), "QUEUE_RECIPE_BINDING")
    output = Path(recipe["output"])
    require(not output.exists(), "CONSUMED_OR_EXISTING_NATIVE_OUTPUT_NOT_RETRIED")
    phases, error, first_job = [], None, None
    with heavy_model_slot(Path(recipe["runtime"]) / "heavy-model.lock") as lease:
        require(lease.inherited and lease.supervised, "SUPERVISED_SHARED_QUEUE_SLOT_REQUIRED")
        batch.write_once(output.parent / "sequence-claim.json", {
            "request": cp.pin(path), "started_at": batch.utc(), "max_calls": 24, "automatic_retries": 0,
            "input_ack": recipe["input_ack"], "preparation_ack": cp.pin(prep / "ack.json")})
        try:
            for phase in recipe["phases"]:
                if first_job is not None:
                    require(cp.pin(output / "job.json") == first_job, "NATIVE_DEADLINE_OR_SCOPE_CHANGED")
                    require(batch.read_json(output / "job.json")["deadline_unix"] > time.time(), "ONE_DEADLINE_EXHAUSTED")
                args = argparse.Namespace(runtime=Path(recipe["runtime"]), inputs=Path(recipe["inputs"]["path"]),
                    output=output, attempt=phase["attempt"], phase=phase["phase"], models=recipe["models"],
                    threads=4, budget_minutes=9)
                before_claims = len(list((output / "claims").glob("*/*.json")))
                code = batch.run(args)
                after_claims = len(list((output / "claims").glob("*/*.json")))
                phase_results = len(list((output / "results").glob("*/*.json")))
                require(0 <= after_claims - before_claims <= phase["max_new_calls"], "PHASE_CALL_CAP")
                if code == 0:
                    require(after_claims - before_claims == phase["max_new_calls"]
                            and phase_results == after_claims, "COMPLETE_PHASE_CALL_ACCOUNTING")
                current_job = cp.pin(output / "job.json")
                first_job = current_job if first_job is None else first_job
                require(current_job == first_job, "NATIVE_JOB_REWRITTEN")
                summary = batch.read_json(output / "attempts" / phase["attempt"] / "summary.json")
                phases.append({"phase": phase["phase"], "returncode": code,
                    "new_claims": after_claims - before_claims, "cumulative_claims": after_claims,
                    "summary": cp.pin(output / "attempts" / phase["attempt"] / "summary.json"),
                    "native_job": current_job, "deadline_unix": batch.read_json(output / "job.json")["deadline_unix"]})
                if code != 0:
                    break
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        claims = len(list((output / "claims").glob("*/*.json")))
        results = len(list((output / "results").glob("*/*.json")))
        complete = (len(phases) == 2 and all(x["returncode"] == 0 for x in phases)
                    and claims == results == 24 and error is None)
        require(claims <= 24, "ACTUAL_CALL_CAP_EXCEEDED")
        batch.write_once(output.parent / "sequence-result.json", {
            "schema": SCHEMA, "status": "complete" if complete else "incomplete", "finished_at": batch.utc(),
            "request": cp.pin(path), "phases": phases, "claims": claims, "results": results,
            "error": error, "deadline_was_not_reset": len({x["native_job"]["sha256"] for x in phases}) <= 1,
            "human_gold": False, "automatic_retries": 0})
    return 0 if complete else 1

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("declare")
    for name in ("runtime", "inputs", "bindings", "sources", "exclusions", "input-checkpoint",
                 "preparation-checkpoint", "output", "request"):
        d.add_argument("--" + name, type=Path, required=True)
    for name in ("verify", "run"):
        sub.add_parser(name).add_argument("--request", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == "declare":
        print(json.dumps(declare(args), sort_keys=True))
        return 0
    if args.command == "verify":
        recipe, inputs = verify_request(args.request)
        print(json.dumps({"status": "verified", "references": len(inputs["clips"]),
                          "max_calls": recipe["max_calls"], "new_model_calls": 0}, sort_keys=True))
        return 0
    return run(args.request)

if __name__ == "__main__":
    raise SystemExit(main())

