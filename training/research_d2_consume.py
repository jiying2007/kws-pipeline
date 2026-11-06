#!/usr/bin/env python3
"""Verified D2 full-text preparation and one fixed C1 consumption probe.

The probe is 25 steps / 100 train-only draws, not a quality comparison. Existing
D1 source/preparation validators, features, C1, initialization and weighted CTC
loop are reused without modifying their historical contracts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_d1_train as training
import research_d1_fit_diagnostic as diagnostic
import research_d2_inputs as inputs
from research_d1_recipe import row_identity, SEED, BATCH_SIZE
from research_d1_vocab import digest, require

PREP = "kws-d2-consumption-preparation-v1"
JOB = "kws-d2-c1-consumption-job-v1"
RESULT = "kws-d2-c1-consumption-result-v1"
CANDIDATE = "c1-ctc2601"
STEPS = 25
SECONDS = 120
CODE = (*training.CODE_PATHS, "training/research_d1_fit_diagnostic.py",
        "training/research_d1_evaluate.py", "training/research_d2_inputs.py",
        "training/research_d2_consume.py")
QUEUE_OUTPUTS = ("summary.json", "consumption.jsonl", "before.json", "after.json", "final.pt", "MANIFEST.json")


def source_records():
    return {name: training.sha_file(ROOT / name) for name in CODE}


def read_lines(path):
    return [training.io_contract.loads(line) for line in training.io_contract.read_bytes(path, 64 * 1024 * 1024).splitlines()]


def anchor(path, manifest_sha):
    """Execute the unchanged D1 verifier from its exact saved source copy.

    No monkeypatching of ROOT/source_records and no exception for source drift.
    The subprocess is isolated, stdlib-only, and does not construct any model.
    """
    path = training.io_contract.safe_path(path)
    source = path / "execution-source"
    program = '''import json,sys
from pathlib import Path
source,root,expected=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
sys.path[:0]=[str(source),str(source/'training'),str(source/'tools')]
import research_d1_train as t
record,vocabulary,rows=t.verify_prepared(root,expected)
prior=[]
for line in t.io_contract.read_bytes(root/'projected-rows.jsonl',64*1024*1024).splitlines():
 r=t.io_contract.loads(line)
 prior.append({'role':r['source_role'],'source_id':r['source_id'],'source_family_id':r['source_family_id'],
  'speaker_id':r['speaker_id'],'pcm_sha256':r['pcm_sha256'],'wav_sha256':r['file_sha256'],
  'parent_source_sha256':r['parent_source_sha256'],'original_pcm_sha256':r['original_pcm_sha256']})
 prior.append({'role':r['source_role'],'pcm_sha256':r['reference_pcm_sha256'],
  'wav_sha256':r['reference_wav_sha256'],'parent_source_sha256':r['parent_source_sha256']})
print(json.dumps({'manifest_sha256':expected,'source_hashes':record['source_hashes'],
 'vocabulary':vocabulary,'initial':record['initial_checkpoints']['c1-ctc2601'],
 'base_recipe':t.read_json(root/'recipes/0.json'),'prior_sources':prior,
 'original_validator_source':str(source/'training/research_d1_train.py'),
 'original_validator_sha256':t.sha_file(source/'training/research_d1_train.py')}))
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", program, str(source), str(path), manifest_sha],
                            capture_output=True, timeout=60)
    require(result.returncode == 0, "D2_ORIGINAL_D1_VALIDATION_FAILED:" + result.stderr.decode(errors="replace")[-2000:])
    require(len(result.stdout) <= 4 * 1024 * 1024, "D2_ANCHOR_RECEIPT_BOUND")
    return training.io_contract.loads(result.stdout)


def rows_for_probe(rows):
    selected = [row for row in rows if row["split"] == "train" and row["ctc_mask"]]
    require(1 <= len(selected) <= 64, "D2_BOUNDED_KNOWN_TRAIN_COHORT_REQUIRED")
    require(all(row["source_role"] == "train-candidate" and row["ctc_weight"] > 0 and
                row["target_ids"] and not row["quarantined"] for row in selected), "D2_INVALID_TRAIN_COHORT")
    return selected


def prepare(request, output):
    require(set(request) == {"schema", "anchor", "anchor_manifest_sha256", "view_loader", "packages"} and
            request["schema"] == PREP, "D2_PREPARATION_REQUEST")
    output = training.output_directory(output)
    original = anchor(request["anchor"], request["anchor_manifest_sha256"])
    loader, data_source = inputs.load_data_module(request["view_loader"])
    verified, aggregate = inputs.aggregate(request["packages"], loader, data_source["files"],
                                           prior_sources=original["prior_sources"])
    vocabulary = original["vocabulary"]
    projected = [inputs.project_row(row, vocabulary) for row in verified]
    selected = rows_for_probe(projected)
    for row in projected:
        require(training.io_contract.ID.fullmatch(row["source_id"]), "D2_SAFE_SOURCE_ID")
        name = "audio/" + row["source_id"] + ".wav"
        require(training.copy_once(row["audio_path"], output / name) == row["file_sha256"], "D2_AUDIO_COPY_CHANGED")
        row["audio"] = row["audio_path"] = name
    for source, target in ((Path(request["anchor"]) / "initial" / (CANDIDATE + ".pt"), output / "initial.pt"),
                           (Path(request["anchor"]) / "official/tokens.txt", output / "tokens.txt")):
        training.copy_once(source, target)
    require(training.sha_file(output / "initial.pt") == original["initial"]["file_sha256"], "D2_INITIAL_COPY_CHANGED")
    code = source_records()
    for name, expected in code.items():
        require(training.copy_once(ROOT / name, output / "execution-source" / name) == expected, "D2_SOURCE_COPY_CHANGED")
    for name, expected in data_source["files"].items():
        require(training.copy_once(Path(data_source["root"]) / name, output / "data-source" / name) == expected,
                "D2_DATA_SOURCE_COPY_CHANGED")
    for name, value in (("request.json", request), ("anchor.json", original), ("vocabulary.json", vocabulary),
                        ("aggregate.json", aggregate)):
        training.io_contract.write_json(output / name, value)
    training.io_contract.write_new(output / "projected-rows.jsonl", b"".join(training.canonical(row) + b"\n" for row in projected))
    record = {"schema": PREP, "status": "completed", "source_hashes": code, "data_source": data_source,
        "vocabulary_sha256": digest(vocabulary), "initial_checkpoint": original["initial"],
        "anchor_manifest_sha256": original["manifest_sha256"], "aggregate": aggregate,
        "row_bindings": [row_identity(row) for row in projected],
        "train_source_ids": [row["source_id"] for row in selected],
        "projection_reasons": dict(Counter(row["ctc_reason"] for row in projected)),
        "eligible_by_split": dict(Counter(row["split"] for row in projected if row["ctc_mask"])),
        "observed_by_split": dict(Counter(row["split"] for row in projected)),
        "new_model_calls": 0, "human_gold": False, "independent_test": False}
    training.io_contract.write_json(output / "preparation.json", record)
    training.io_contract.write_json(output / "MANIFEST.json", {p.relative_to(output).as_posix(): training.io_contract.file_record(p)
        for p in sorted(output.rglob("*")) if p.is_file()})
    return {"prepared": str(output), "manifest_sha256": training.sha_file(output / "MANIFEST.json"),
            "eligible_by_split": record["eligible_by_split"], "projection_reasons": record["projection_reasons"], "new_model_calls": 0}


def verify_files(root, expected_sha):
    root = training.io_contract.safe_path(root)
    require(training.sha_file(root / "MANIFEST.json") == expected_sha, "D2_MANIFEST_CHANGED")
    manifest = training.read_json(root / "MANIFEST.json")
    require(set(manifest) == {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "MANIFEST.json"},
            "D2_FILE_ROSTER_CHANGED")
    for name, record in manifest.items():
        require(training.io_contract.file_record(root / training.io_contract.relative(name)) == record,
                "D2_FILE_CHANGED:" + name)


def verify_prepared(root, expected_sha):
    root = Path(root)
    verify_files(root, expected_sha)
    record, request = training.read_json(root / "preparation.json"), training.read_json(root / "request.json")
    require(record["schema"] == PREP and record["status"] == "completed" and
            record["source_hashes"] == source_records(), "D2_PREPARED_SOURCE_CHANGED")
    original = anchor(request["anchor"], request["anchor_manifest_sha256"])
    require(original == training.read_json(root / "anchor.json"), "D2_ORIGINAL_ANCHOR_CHANGED")
    vocabulary = training.read_json(root / "vocabulary.json")
    require(vocabulary == original["vocabulary"] and digest(vocabulary) == record["vocabulary_sha256"] and
            training.sha_file(root / "tokens.txt") == vocabulary["base_tokens_sha256"] and
            training.sha_file(root / "initial.pt") == original["initial"]["file_sha256"], "D2_ANCHORED_VOCAB_OR_INIT_CHANGED")
    loader, data_source = inputs.load_data_module(root / "data-source" / inputs.DATA_FILES[0])
    require(data_source["files"] == record["data_source"]["files"], "D2_FROZEN_DATA_SOURCE_CHANGED")
    verified, aggregate = inputs.aggregate(request["packages"], loader, data_source["files"], prior_sources=original["prior_sources"])
    projected = [inputs.project_row(row, vocabulary) for row in verified]
    rows = read_lines(root / "projected-rows.jsonl")
    require(aggregate == record["aggregate"] and [row_identity(row) for row in projected] == record["row_bindings"] ==
            [row_identity(row) for row in rows], "D2_PROJECTED_LABELS_CHANGED")
    for row in rows:
        require(row["audio"] == row["audio_path"] == "audio/" + row["source_id"] + ".wav" and
                training.sha_file(root / row["audio"]) == row["file_sha256"], "D2_RETAINED_AUDIO_CHANGED")
    selected = rows_for_probe(rows)
    require([row["source_id"] for row in selected] == record["train_source_ids"], "D2_TRAIN_ROSTER_CHANGED")
    return record, vocabulary, rows, selected, original["base_recipe"]


def validate_job(job):
    require(set(job) == {"schema", "prepared", "prepared_manifest_sha256", "candidate", "optimizer_steps",
                        "timeout_seconds", "row_binding_sha256", "source_hashes"} and job["schema"] == JOB and
            job["candidate"] == CANDIDATE and job["optimizer_steps"] == STEPS and
            job["timeout_seconds"] == SECONDS and Path(job["prepared"]).is_absolute() and
            training.io_contract.SHA.fullmatch(job["prepared_manifest_sha256"]) and
            training.io_contract.SHA.fullmatch(job["row_binding_sha256"]) and
            job["source_hashes"] == source_records(), "D2_FIXED_PROBE_JOB_CHANGED")
    return job


def declare(prepared, manifest_sha, output):
    _, _, _, rows, _ = verify_prepared(prepared, manifest_sha)
    job = validate_job({"schema": JOB, "prepared": str(Path(prepared).absolute()), "prepared_manifest_sha256": manifest_sha,
        "candidate": CANDIDATE, "optimizer_steps": STEPS, "timeout_seconds": SECONDS,
        "row_binding_sha256": digest([row_identity(row) for row in rows]), "source_hashes": source_records()})
    training.io_contract.write_json(output, job)
    return {"recipe": str(output), "recipe_sha256": training.sha_file(output), "train_rows": len(rows),
            "optimizer_steps": STEPS, "sample_exposures": STEPS * BATCH_SIZE, "new_model_calls": 0}


def probe_recipe(rows, base):
    recipe = diagnostic.tiny_recipe(rows, base, STEPS)
    recipe["schema"] = JOB
    return recipe


def ledger_readback(ledger, recipe, rows):
    gradients = diagnostic.gradient_observation(ledger, recipe, rows)
    for event in ledger:
        for draw in event["draws"]:
            row = next(row for row in rows if row["source_id"] == draw["source_id"])
            require(all(draw[key] == row[key] for key in ("full_text", "target_ids", "pcm_sha256", "global_source_key",
                    "source_family_id", "keyword_targets", "keyword_mask", "keyword_weight", "generation_category")),
                    "D2_CONSUMED_FULL_LABEL_CHANGED")
            expected = draw["raw_length_normalized_ctc"] * draw["ctc_weight"] / len(event["draws"])
            require(math.isclose(expected, draw["weighted_batch_contribution"], rel_tol=1e-6, abs_tol=1e-7),
                    "D2_WEIGHTED_LOSS_CHANGED")
        require(math.isclose(event["loss"], sum(draw["weighted_batch_contribution"] for draw in event["draws"]),
                             rel_tol=1e-6, abs_tol=1e-7), "D2_BATCH_LOSS_CHANGED")
    return gradients


def run(job_path, output, lock):
    job = validate_job(training.read_json(job_path))
    with training.heavy_model_slot(lock) as lease:
        training.queue_binding(job_path, lease)
        output = training.output_directory(output)
        start = time.monotonic()
        deadline = start + SECONDS
        status = {"schema": RESULT, "status": "running", "started_at": training.utc(), "candidate": CANDIDATE,
            "job_recipe_sha256": training.sha_file(job_path), "human_gold": False, "quality_winner": None,
            "development_forward_calls": 0, "asr_or_tts_calls": 0, "automatic_retry": False,
            "scope": "fixed D2 train-only data-consumption probe; no quality optimization claim"}
        training.save_json(output / "status.json", status)
        try:
            runtime = training.research_runtime.load_manifest()
            training.research_runtime.check_interpreter(runtime)
            training.research_runtime.check_packages(runtime, "core")
            prepared = Path(job["prepared"])
            record, vocabulary, all_rows, rows, base = verify_prepared(prepared, job["prepared_manifest_sha256"])
            require(digest([row_identity(row) for row in rows]) == job["row_binding_sha256"], "D2_JOB_ROWS_CHANGED")
            torch, reference = training.torch_runtime()
            import numpy as np
            from frontend import features as frontend_features
            from training_state import atomic_save_checkpoint, load_checkpoint_snapshot, state_identity
            features, observations = [], []
            for row in rows:
                with wave.open(str(prepared / row["audio"]), "rb") as reader:
                    require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes(), reader.getcomptype()) ==
                            (1, 2, 16000, row["samples"], "NONE"), "D2_FEATURE_PCM_FORMAT")
                    pcm = reader.readframes(reader.getnframes())
                require(hashlib.sha256(pcm).hexdigest() == row["pcm_sha256"], "D2_FEATURE_PCM_CHANGED")
                waveform = torch.from_numpy(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768))
                feature = frontend_features(waveform, feature_dim=32, frontend="logmel").contiguous()
                require(tuple(feature.shape) == (row["expected_c1_frames"], 32) and
                        len(feature) >= row["ctc_minimum_frames"] and bool(torch.isfinite(feature).all()), "D2_FEATURE_GEOMETRY")
                features.append(feature)
                observations.append({"source_id": row["source_id"], "pcm_sha256": row["pcm_sha256"],
                    "frames": len(feature), "feature_sha256": training.tensor_sha(feature)})
            torch.manual_seed(SEED)
            model, description = training.build_fresh(CANDIDATE, vocabulary, reference)
            initial, initial_sha = load_checkpoint_snapshot(prepared / "initial.pt")
            anchor_receipt = training.read_json(prepared / "anchor.json")
            require(initial_sha == record["initial_checkpoint"]["file_sha256"] and
                    initial["schema"] == "kws-d1-initial-checkpoint-v2" and initial["candidate"] == CANDIDATE and
                    initial["vocabulary_sha256"] == digest(vocabulary) and initial["optimizer_steps"] == 0 and
                    initial["source_hashes"] == anchor_receipt["source_hashes"] and initial["description"] == description,
                    "D2_EXACT_INITIAL_CHECKPOINT_CHANGED")
            model.load_state_dict(initial["state_dict"], strict=True)
            require(state_identity(model.state_dict()) == initial["float_state_identity"], "D2_INITIAL_LOAD_CHANGED")
            training.copy_once(prepared / "initial.pt", output / "initial.pt")
            training.copy_once(prepared / "projected-rows.jsonl", output / "all-rows.jsonl")
            training.io_contract.write_json(output / "train-rows.json", rows)
            training.io_contract.write_json(output / "feature-observations.json", observations)
            recipe = probe_recipe(rows, base)
            training.io_contract.write_json(output / "exposure-recipe.json", recipe)
            before = diagnostic.observe(model, features, rows, "before", output, deadline)
            spec = recipe["optimizers"]["fresh-rnn64-or-c1"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
            model.train()
            consumed = training.train_batches(model, features, rows, recipe, optimizer, output=output, deadline=deadline)
            gradients = ledger_readback(read_lines(output / "consumption.jsonl"), recipe, rows)
            names = [name for name, p in model.named_parameters() if p.requires_grad]
            changed, delta = diagnostic.parameter_changes(initial["state_dict"], model.state_dict(), names)
            final_identity = state_identity(model.state_dict())
            after = diagnostic.observe(model, features, rows, "after", output, deadline)
            require(state_identity(model.state_dict()) == final_identity, "D2_DIAGNOSTIC_MUTATED_MODEL")
            payload = {"schema": RESULT, "candidate": CANDIDATE, "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "float_state_identity": final_identity,
                "job_recipe_sha256": training.sha_file(job_path), "optimizer_steps": STEPS,
                "initial_checkpoint_sha256": initial_sha, "vocabulary_sha256": digest(vocabulary)}
            atomic_save_checkpoint(payload, output / "final.pt")
            restored, final_sha = load_checkpoint_snapshot(output / "final.pt")
            require(state_identity(restored["state_dict"]) == final_identity, "D2_FINAL_READBACK_CHANGED")
            training.copy_once(job_path, output / "request.json")
            require(consumed["optimizer_steps"] == STEPS and consumed["sample_exposures"] == STEPS * BATCH_SIZE and
                    time.monotonic() < deadline, "D2_PROBE_BUDGET_NOT_CONSUMED")
            status.update(consumed, status="completed", finished_at=training.utc(), wall_seconds=time.monotonic() - start,
                prepared_manifest_sha256=job["prepared_manifest_sha256"], row_binding_sha256=job["row_binding_sha256"],
                vocabulary_sha256=digest(vocabulary), initial_checkpoint_sha256=initial_sha,
                initial_float_state_sha256=initial["float_state_identity"]["sha256"],
                final_float_state_sha256=final_identity["sha256"], final_checkpoint_sha256=final_sha,
                trainable_parameter_names=names, changed_parameter_names=changed, parameter_delta_l2=delta,
                train_rows=len(rows), all_observed_rows=len(all_rows), gradients=gradients,
                before=diagnostic.summarize(before), after=diagnostic.summarize(after),
                training_model_forwards=STEPS * BATCH_SIZE, diagnostic_model_forwards=2 * len(rows),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                consumption_sha256=training.sha_file(output / "consumption.jsonl"),
                consumption_verified=True, projection_reasons=record["projection_reasons"])
            training.save_json(output / "status.json", status)
            training.io_contract.write_json(output / "summary.json", status)
            training.io_contract.write_json(output / "MANIFEST.json", {p.relative_to(output).as_posix(): training.io_contract.file_record(p)
                for p in sorted(output.rglob("*")) if p.is_file()})
            return status
        except BaseException as error:
            status.update(status="failed", error_type=type(error).__name__, error=str(error), wall_seconds=time.monotonic() - start)
            if (output / "consumption.jsonl").exists():
                lines = (output / "consumption.jsonl").read_bytes().splitlines(keepends=True)
                status["completed_ledger_steps"] = sum(line.endswith(b"\n") for line in lines)
            training.save_json(output / "status.json", status)
            raise


def audit(output, expected_sha):
    output = Path(output)
    verify_files(output, expected_sha)
    job, summary = validate_job(training.read_json(output / "request.json")), training.read_json(output / "summary.json")
    record, vocabulary, all_rows, rows, base = verify_prepared(job["prepared"], job["prepared_manifest_sha256"])
    require(summary["schema"] == RESULT and summary["status"] == "completed" and
            summary["job_recipe_sha256"] == training.sha_file(output / "request.json") and
            summary["optimizer_steps"] == STEPS and summary["sample_exposures"] == STEPS * BATCH_SIZE and
            summary["development_forward_calls"] == summary["asr_or_tts_calls"] == 0 and
            training.read_json(output / "train-rows.json") == rows and read_lines(output / "all-rows.jsonl") == all_rows and
            summary["row_binding_sha256"] == job["row_binding_sha256"] == digest([row_identity(row) for row in rows]) and
            summary["vocabulary_sha256"] == digest(vocabulary), "D2_RESULT_SCOPE_CHANGED")
    recipe = probe_recipe(rows, base)
    require(training.read_json(output / "exposure-recipe.json") == recipe and
            ledger_readback(read_lines(output / "consumption.jsonl"), recipe, rows) == summary["gradients"], "D2_LEDGER_CHANGED")
    torch, _ = training.torch_runtime()
    from training_state import load_checkpoint_snapshot, state_identity
    initial, initial_sha = load_checkpoint_snapshot(output / "initial.pt")
    final, final_sha = load_checkpoint_snapshot(output / "final.pt")
    require(initial_sha == summary["initial_checkpoint_sha256"] == record["initial_checkpoint"]["file_sha256"] and
            state_identity(initial["state_dict"]) == initial["float_state_identity"] and
            initial["float_state_identity"]["sha256"] == summary["initial_float_state_sha256"] and
            final_sha == summary["final_checkpoint_sha256"] and final["schema"] == RESULT and
            final["candidate"] == CANDIDATE and final["optimizer_steps"] == STEPS and
            final["vocabulary_sha256"] == digest(vocabulary) and
            state_identity(final["state_dict"]) == final["float_state_identity"] and
            final["float_state_identity"]["sha256"] == summary["final_float_state_sha256"] and
            final["job_recipe_sha256"] == summary["job_recipe_sha256"], "D2_CHECKPOINT_CHANGED")
    changed, delta = diagnostic.parameter_changes(initial["state_dict"], final["state_dict"], summary["trainable_parameter_names"])
    require(changed == summary["changed_parameter_names"] and delta == summary["parameter_delta_l2"], "D2_PARAMETER_DELTA_CHANGED")
    for phase in ("before", "after"):
        observed = training.read_json(output / (phase + ".json"))
        require(len(observed) == len(rows), "D2_NUMERIC_ROSTER_CHANGED")
        for row, item in zip(rows, observed):
            numeric = item["numeric_artifact"]
            path = output / training.io_contract.relative(numeric["path"])
            require(training.io_contract.file_record(path) == {k: numeric[k] for k in ("bytes", "sha256")}, "D2_NUMERIC_FILE_CHANGED")
            saved = torch.load(path, weights_only=True, map_location="cpu")
            require(saved["source_id"] == row["source_id"] and item["label_binding_sha256"] == row_identity(row) and
                    training.tensor_sha(saved["feature"]) == saved["feature_sha256"] and
                    training.tensor_sha(saved["logits"]) == saved["logits_sha256"] and
                    all(item[k] == v for k, v in diagnostic.score(saved["logits"], row).items()), "D2_SAVED_NLL_CHANGED")
        require(diagnostic.summarize(observed) == summary[phase], "D2_NUMERIC_SUMMARY_CHANGED")
    return {"schema": "kws-d2-consumption-readback-v1", "status": "verified", "manifest_sha256": expected_sha,
            "optimizer_steps_replayed": STEPS, "new_optimizer_steps": 0, "new_model_forwards": 0,
            "train_rows": len(rows), "consumption_verified": True, "human_gold": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("anchor", "anchor-manifest-sha256", "view-loader", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--package", action="append", required=True)
    p.add_argument("--package-manifest-sha256", action="append", required=True)
    p = sub.add_parser("declare")
    for name in ("prepared", "prepared-manifest-sha256", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("run")
    for name in ("recipe", "output", "heavy-lock"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("audit")
    for name in ("output", "manifest-sha256", "receipt"):
        p.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        require(len(args.package) == len(args.package_manifest_sha256), "D2_PACKAGE_PIN_COUNT")
        result = prepare({"schema": PREP, "anchor": args.anchor, "anchor_manifest_sha256": args.anchor_manifest_sha256,
            "view_loader": args.view_loader, "packages": [{"path": str(Path(p).absolute()), "manifest_sha256": h}
                for p, h in zip(args.package, args.package_manifest_sha256)]}, Path(args.output))
    elif args.command == "declare":
        result = declare(args.prepared, args.prepared_manifest_sha256, Path(args.output))
    elif args.command == "run":
        result = run(Path(args.recipe), Path(args.output), Path(args.heavy_lock))
    else:
        result = audit(args.output, args.manifest_sha256)
        training.io_contract.write_json(Path(args.receipt), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
