#!/usr/bin/env python3
"""Read-only export of actual context audio after an immutable queue timeout.

The original generation owner, queue and data parsers remain unchanged. A
consumed unknown and a never-claimed cell have no ASR input. This tool creates
only a new observation package; it cannot generate or retry speech.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import zipfile

SCHEMA = "kws-context-timeout-partial-export-v1"
ML_MODULES = ("numpy", "torch", "onnxruntime", "qwen_tts", "soundfile", "scipy")
NATIVE_FILES = {"call.json", "native.json", "returned.json", "returned-array.npy", "returned-native.wav"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plain(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "PARTIAL_SYMLINK")
    return path


def read(path):
    path = plain(path)
    require(path.is_file() and path.stat().st_size <= 64 * 1024 * 1024, "PARTIAL_FILE_BOUND")
    return path.read_bytes()


def record(path):
    raw = read(path)
    return {"bytes": len(raw), "sha256": sha(raw)}


def load(path):
    return json.loads(read(path))


def leaves(path):
    path = plain(path)
    if not path.exists():
        return set()
    require(path.is_dir(), "PARTIAL_FOLDER_REQUIRED")
    result = set()
    for child in path.iterdir():
        require(plain(child).is_file(), "PARTIAL_CELL_FILE_REQUIRED")
        result.add(child.name)
    return result


def inventory(root):
    root = plain(root)
    names = ("stage.json", "plan.json", "source-manifest.json", "queue/queue.json",
             "queue/request.json", "queue/active-plan.json", "queue/recipes/generate-001.json")
    files = {name: record(root / name) for name in names}
    for folder in (root / "generation", root / "queue/jobs/generate-001"):
        for path in sorted(folder.rglob("*")):
            plain(path)
            if path.is_file():
                files[path.relative_to(root).as_posix()] = record(path)
    return files


def check_timeout(result, claim, *, recipe_sha, queue_sha, claim_sha, definition_sha):
    require(result["schema"] == "kws-local-research-job-result-v1" and
            result["status"] == "timed_out" and result["exit_code"] == -9 and
            result["reason"] == "actual_runner_deadline" and
            result["job_id"] == claim["job_id"] == "generate-001" and
            result["automatic_retry"] is claim["automatic_retry"] is False and
            result["actual_duration_known"] is True and
            result["recipe_sha256"] == claim["recipe_sha256"] == recipe_sha and
            result["queue_sha256"] == claim["queue_sha256"] == queue_sha and
            result["claim_sha256"] == claim_sha and
            result["job_definition_sha256"] == claim["job_definition_sha256"] == definition_sha and
            claim["attempt"] == 1 and
            result["allowed_runner_nanoseconds"] == claim["allowed_runner_nanoseconds"] == 600_000_000_000 and
            result["actual_elapsed_ns"] >= result["allowed_runner_nanoseconds"] and
            result["deadline_overrun_nanoseconds"] ==
                result["actual_elapsed_ns"] - result["allowed_runner_nanoseconds"],
            "PARTIAL_AUTHENTIC_TIMEOUT_REQUIRED")


def check_call(call, cell, ref, generation):
    mode = "ICL" if ref["reference_text_status"] in ("verified", "dataset-provided-content-checked") else "x-vector-only"
    expected = {"schema": "active-tts-generation-call-v1", "cell_id": cell["cell_id"],
                "source_family_id": cell["source_family_id"], "reference_id": ref["reference_id"],
                "reference_wav_sha256": ref["wav_sha256"], "reference_pcm_sha256": ref["pcm_sha256"],
                "reference_text_status": ref["reference_text_status"], "reference_mode": mode,
                "reference_text_passed": mode == "ICL", "seed": cell["seed"], "attempt": 1,
                "generation": generation, "intended_text": cell["prompt"]}
    require(set(call) == set(expected) | {"reference_prompt_seconds"} and
            all(call[k] == v for k, v in expected.items()) and
            type(call["reference_prompt_seconds"]) in (int, float) and
            math.isfinite(call["reference_prompt_seconds"]) and call["reference_prompt_seconds"] >= 0,
            "PARTIAL_NATIVE_CALL_CHANGED")


def classify_cell(generation_root, native_root, cell, ref, plan_sha, generation, weak):
    ident = cell["cell_id"]
    g, n = plain(generation_root) / "cells" / ident, plain(native_root) / ident
    gf, nf = leaves(g), leaves(n)
    if not gf and not nf:
        return {"cell_id": ident, "status": "never-claimed"}, None
    require("started.json" in gf and "call.json" in nf, "PARTIAL_UNOWNED_ARTIFACT")
    weak._d1_started(read(g / "started.json"), cell, ref, plan_sha)
    check_call(load(n / "call.json"), cell, ref, generation)
    if gf == {"started.json"} and nf == {"call.json"}:
        return {"cell_id": ident, "status": "consumed-unknown", "may_retry": False}, None
    success = "completed.json" in gf
    terminal = "completed.json" if success else "failed.json"
    require(nf == NATIVE_FILES and gf == {"started.json", terminal} |
            ({"native-f32.wav", "audio-16k.wav"} if success else set()), "PARTIAL_NATIVE_FILE_ROSTER")
    raw, array, meta = read(n / "returned-native.wav"), read(n / "returned-array.npy"), load(n / "native.json")
    require(meta["schema"] == "active-tts-native-return-v1" and
            meta["array"] == {"bytes": len(array), "sha256": sha(array)} and
            meta["wav"] == {"bytes": len(raw), "sha256": sha(raw)} and
            meta["all_finite"] is True and meta["technical_gate_applied"] is False and
            meta["sample_rate_hz"] == 24000 and meta.get("wav_unavailable_reason") is None,
            "PARTIAL_NATIVE_BYTE_BINDING")
    pcm = weak.float_wav_pcm(raw)
    weak._d1_native_array(array, meta, pcm)
    returned = load(n / "returned.json")
    require(set(returned) == {"generation_seconds", "total_seconds", "native_stop_reason"} and
            returned["native_stop_reason"] == "not_exposed_by_voice_clone_API" and
            all(type(returned[k]) in (int, float) and math.isfinite(returned[k]) and returned[k] >= 0
                for k in ("generation_seconds", "total_seconds")), "PARTIAL_NATIVE_RETURN_RECEIPT")
    if success:
        require(read(g / "native-f32.wav") == raw, "PARTIAL_TECHNICAL_NATIVE_CHANGED")
        weak._d1_completed(read(g / terminal), cell,
            {"derived_pcm": {"frames": (len(pcm) // 4 * 2 + 2) // 3}}, raw, pcm, read(g / "audio-16k.wav"))
    else:
        failure = load(g / terminal)
        require(set(failure) == {"schema", "cell_id", "status", "error_type", "training_admitted"} and
                failure["schema"] == "kws-bulk-clone-failure-v1" and failure["cell_id"] == ident and
                failure["status"] == "FAILED_NO_RETRY" and failure["training_admitted"] is False and
                type(failure["error_type"]) is str and 1 <= len(failure["error_type"]) <= 128,
                "PARTIAL_TECHNICAL_FAILURE_CHANGED")
    return {"cell_id": ident, "status": "completed" if success else "failed-native",
            "original_wav_sha256": sha(raw), "original_pcm_sha256": sha(pcm),
            "frames": len(pcm) // 4, "terminal_sha256": sha(read(g / terminal))}, raw


def collect(request):
    observed_path, owner_root, data_weak = map(plain, (request["observation"], request["owner_root"], request["data_weak"]))
    require(sha(read(observed_path)) == request["observation_sha256"], "PARTIAL_OBSERVATION_PIN_CHANGED")
    require(sha(read(data_weak)) == request["data_weak_sha256"], "PARTIAL_DATA_PARSER_PIN_CHANGED")
    observed = load(observed_path)
    require(observed["schema"] == "kws-context-generation-timeout-observation-v1", "PARTIAL_OBSERVATION_SCHEMA")
    root = plain(observed["source_root"])
    require(inventory(root) == observed["files"], "PARTIAL_ORIGINAL_INVENTORY_CHANGED")
    stage_path = root / "stage.json"
    stage_raw = load(stage_path)
    for name, pin in stage_raw["execution_source"].items():
        require(record(owner_root / name) == pin, "PARTIAL_OWNER_SOURCE_CHANGED")
    sys.path.insert(0, str(owner_root / "tools"))
    owner = importlib.import_module("research_context_generate")
    require(Path(owner.__file__).resolve() == owner_root / "tools/research_context_generate.py", "PARTIAL_OWNER_IMPORT")
    stage, prepared, plan, bulk = owner.load_stage(stage_path)
    q = owner.queue
    qroot, _, definition, queue_sha = q.load_queue(Path(stage["queue_root"]))
    require(qroot == root / "queue" and len(definition["jobs"]) == 1, "PARTIAL_SINGLE_FROZEN_OWNER_REQUIRED")
    job = definition["jobs"][0]
    recipe_path = qroot / "recipes/generate-001.json"
    recipe = load(recipe_path)
    owner.validate_job_recipe(recipe, stage, str(stage_path))
    state = q.result_for(qroot, job, queue_sha)
    result_path, claim_path = qroot / "jobs/generate-001/result.json", qroot / "jobs/generate-001/claim.json"
    result, claim = load(result_path), load(claim_path)
    require(state["status"] == "timed_out" and state["receipt_sha256"] == sha(read(result_path)), "PARTIAL_QUEUE_REPLAY")
    check_timeout(result, claim, recipe_sha=sha(read(recipe_path)), queue_sha=queue_sha,
                  claim_sha=sha(read(claim_path)), definition_sha=q.job_binding(job, queue_sha)["job_definition_sha256"])
    require(observed["queue_result_sha256"] == sha(read(result_path)) and
            observed["queue_claim_sha256"] == sha(read(claim_path)) and
            observed["owner_job_recipe_sha256"] == sha(read(recipe_path)) and
            observed["source_stage_sha256"] == sha(read(stage_path)), "PARTIAL_OWNER_OBSERVATION_JOIN")
    selected = recipe["selected_cells"]
    require(len(selected) == len(set(selected)) == 16 and stage["chunks"] == [selected] and
            selected == [c["cell_id"] for c in plan["cells"]], "PARTIAL_COMPLETE_SIXTEEN_ROSTER")
    native_root = qroot / "jobs/generate-001/outputs/native"
    require({p.name for p in native_root.iterdir()} <= set(selected) and
            {p.name for p in (Path(stage["generation_root"]) / "cells").iterdir()} <= set(selected),
            "PARTIAL_OUTSIDE_OWNER_CELL")
    invocation = load(native_root.parent / "invocation.json")
    require(invocation["selected_cells"] == selected and invocation["max_calls"] == 16 and
            invocation["recipe_sha256"] == sha(read(recipe_path)) and invocation["supervised"] is True and
            invocation["heavy_lock"] == stage["heavy_lock"], "PARTIAL_INVOCATION_OWNER")
    require(data_weak.name == "active_asr_weak_view.py", "PARTIAL_DATA_PARSER_REQUIRED")
    spec = importlib.util.spec_from_file_location("context_partial_weak_parser", data_weak)
    weak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(weak)
    refs = {r["reference_id"]: r for r in plan["references"]}
    plan_sha = sha(bulk.p.canonical(plan))
    weak._d1_lock(read(Path(stage["generation_root"]) / "plan-lock.json"), plan, plan_sha)
    rows, buffer = [], io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for cell in plan["cells"]:
            row, raw = classify_cell(stage["generation_root"], native_root, cell, refs[cell["reference_id"]],
                                     plan_sha, stage["generation"], weak)
            rows.append(row)
            if raw is not None:
                archive.writestr(zipfile.ZipInfo("cells/" + cell["cell_id"] + "/native-f32.wav"), raw)
    counts = dict(Counter(r["status"] for r in rows))
    require(counts == observed["counts"] == {"completed": 12, "failed-native": 1, "consumed-unknown": 1, "never-claimed": 2}
            and [r["status"] for r in rows[-3:]] == ["consumed-unknown", "never-claimed", "never-claimed"]
            and all(r["status"] in ("completed", "failed-native") for r in rows[:13]), "PARTIAL_FROZEN_PREFIX_CLASSIFICATION")
    require(inventory(root) == observed["files"], "PARTIAL_CONCURRENT_MUTATION")
    require(not any(name in sys.modules for name in ML_MODULES), "PARTIAL_MODEL_IMPORT_FORBIDDEN")
    body = buffer.getvalue()
    proof = {"schema": SCHEMA, "request": request, "source_stage": str(stage_path),
        "stage_sha256": sha(read(stage_path)), "job_recipe": str(recipe_path),
        "job_recipe_sha256": sha(read(recipe_path)), "queue_result_sha256": sha(read(result_path)),
        "queue_status": "timed_out", "queue_exit_code": -9, "original_files": observed["files"],
        "owner_execution_source": stage["execution_source"], "producer_source": stage["data_source"],
        "data_parser_sha256": sha(read(data_weak)), "exporter_sha256": sha(read(__file__)),
        "plan": str(prepared / "plan.json"), "plan_sha256": plan_sha,
        "plan_file_sha256": sha(read(prepared / "plan.json")), "generation_root": stage["generation_root"],
        "native_root": str(native_root), "selected_cell_ids": selected, "rows": rows, "counts": counts,
        "observed_cell_ids": [r["cell_id"] for r in rows if r["status"] in ("completed", "failed-native")],
        "consumed_unknown_cell_ids": [r["cell_id"] for r in rows if r["status"] == "consumed-unknown"],
        "never_claimed_cell_ids": [r["cell_id"] for r in rows if r["status"] == "never-claimed"],
        "archive": {"bytes": len(body), "sha256": sha(body)}, "exported_count": 13,
        "new_model_calls": 0, "new_tts_calls": 0, "automatic_retries": 0,
        "recognizer_context_text": None, "human_gold": False, "complete_terminal_coverage": False}
    return proof, body


def export(request, output):
    output = plain(output)
    require(not output.exists(), "PARTIAL_EXPORT_CREATE_ONLY")
    proof, body = collect(request)
    output.mkdir(mode=0o700)
    for name, raw in {"proof.json": canonical(proof), "request.json": canonical(request),
                      "source-original.zip": body, "exporter-source.py": read(__file__)}.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    manifest = {p.name: record(p) for p in sorted(output.iterdir())}
    with (output / "MANIFEST.json").open("xb") as stream:
        stream.write(canonical(manifest))
    return proof


def verify(package, proof_sha):
    package = plain(package)
    require(sha(read(package / "proof.json")) == proof_sha, "PARTIAL_PROOF_PIN_CHANGED")
    manifest = load(package / "MANIFEST.json")
    require(leaves(package) == set(manifest) | {"MANIFEST.json"} and
            all(record(package / k) == v for k, v in manifest.items()), "PARTIAL_EXPORT_MANIFEST_CHANGED")
    proof, body = collect(load(package / "request.json"))
    require(proof == load(package / "proof.json") and body == read(package / "source-original.zip") and
            read(package / "exporter-source.py") == read(__file__), "PARTIAL_REPLAY_CHANGED")
    return {"schema": "kws-context-timeout-partial-readback-v1", "status": "verified",
            "proof_sha256": proof_sha, "observed_rows": 13, "counts": proof["counts"], "new_model_calls": 0}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    for field in ("observation", "observation-sha256", "owner-root", "data-weak", "data-weak-sha256", "output"):
        p.add_argument("--" + field, required=True)
    p = sub.add_parser("verify")
    for field in ("package", "proof-sha256"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--receipt")
    args = cli.parse_args(argv)
    if args.command == "export":
        result = export({"observation": str(plain(args.observation)), "observation_sha256": args.observation_sha256,
                         "owner_root": str(plain(args.owner_root)), "data_weak": str(plain(args.data_weak)),
                         "data_weak_sha256": args.data_weak_sha256}, args.output)
        print(canonical({"schema": result["schema"], "exported_count": result["exported_count"],
                         "counts": result["counts"], "new_model_calls": 0}).decode(), end="")
    else:
        result = verify(args.package, args.proof_sha256)
        if args.receipt:
            with plain(args.receipt).open("xb") as stream:
                stream.write(canonical(result))
        print(canonical(result).decode(), end="")


if __name__ == "__main__":
    main()
