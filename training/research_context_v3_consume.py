#!/usr/bin/env python3
"""One predeclared context-v3 ingestion smoke, with a separate final dev readback.

The original D1 2601-symbol vocabulary, zero-step C1 initialization, optimizer,
features and weighted full-text CTC loop are unchanged. Six train families are
selected before ASR; every dev family stays evaluation-only. This is a data
consumption check, with no fit, quality, generalization or deployment gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_d1_train as training
import research_d1_fit_diagnostic as diagnostic
import research_d2_consume as historical
import research_context_v3_inputs as inputs
import research_context_v3_handoff_inputs as handoff_inputs
import research_d2_checkpoints as checkpoints
from research_d1_recipe import row_identity, SEED, BATCH_SIZE
from research_d1_vocab import digest, require

SELECTION = "kws-context-v3-ingestion-selection-v1"
PREP = "kws-context-v3-ingestion-preparation-v1"
HANDOFF_PREP = "kws-context-v3-handoff-ingestion-preparation-v1"
HANDOFF = "kws-context-v3-metadata-handoff-v1"
HANDOFF_DATA = "kws-context-v3-handoff-ingestion-data-sources-v1"
JOB = "kws-context-v3-ingestion-job-v1"
RESULT = "kws-context-v3-ingestion-result-v1"
STAGE = "kws-context-v3-ingestion-stage-v1"
JOB_ID = "context-v3-ingestion"
OWNER_ENV = "KWS_CONTEXT_V3_INGESTION_OWNER"
INTENT_ENV = "KWS_CONTEXT_V3_INGESTION_INTENT"
CANDIDATE = "c1-ctc2601"
STEPS, SECONDS = 25, 120
CODE = (*historical.CODE, "training/research_context_v3_inputs.py",
        "training/research_context_v3_handoff_inputs.py",
        "training/research_context_v3_consume.py", "tools/research_d2_checkpoints.py")
QUEUE_OUTPUTS = ("summary.json", "consumption.jsonl", "before.json", "after.json",
                 "development.json", "final.pt", "MANIFEST.json")


def source_records():
    return {name: training.sha_file(ROOT / name) for name in CODE}


def pin(path):
    return checkpoints.pin(path)


def checked(value):
    require(set(value) == {"path", "bytes", "sha256"} and pin(value["path"]) == value,
            "V3_INGESTION_PIN_CHANGED")
    return Path(value["path"])


def lines(path):
    return historical.read_lines(path)


def checkpoint_metadata(folder):
    return {name: pin(Path(folder) / name)
            for name in ("capsule.json", "phase-marker.json", "provider-save.json", "ack.json")}


def selection_layout(plan):
    """Only declared source roles and stable family ordering may select rows."""
    require(plan["schema"] == "kws-context-supplement-plan-v3" and len(plan["cells"]) == 72 and
            len({c["cell_id"] for c in plan["cells"]}) == 72 and len(plan["references"]) == 24,
            "V3_INGESTION_FULL_PLAN_REQUIRED")
    families = {}
    for ref in plan["references"]:
        require(ref["role"] in ("train-candidate", "development-only") and
                ref["source_family_id"] not in families, "V3_INGESTION_DISTINCT_FAMILIES")
        families[ref["source_family_id"]] = (ref["role"], ref["speaker_group_id"])
    train = sorted(f for f, (role, _) in families.items() if role == "train-candidate")
    development = sorted(f for f, (role, _) in families.items() if role == "development-only")
    require(len(train) == 18 and len(development) == 6 and
            not ({s for role, s in families.values() if role == "train-candidate"} &
                 {s for role, s in families.values() if role == "development-only"}),
            "V3_INGESTION_FAMILY_SPLIT_CHANGED")
    selected = train[:6]
    for cell in plan["cells"]:
        require((cell["role"], cell["speaker_group_id"]) == families[cell["source_family_id"]],
                "V3_INGESTION_CELL_FAMILY_ROLE_CHANGED")
    counts = Counter(c["source_family_id"] for c in plan["cells"])
    require(counts == Counter({f: 3 for f in families}), "V3_INGESTION_THREE_CELLS_PER_FAMILY")
    return {"train_source_family_ids": selected, "development_source_family_ids": development,
            "train_declared_source_ids": [c["cell_id"] for c in plan["cells"] if c["source_family_id"] in selected],
            "development_declared_source_ids": [c["cell_id"] for c in plan["cells"] if c["source_family_id"] in development]}


def validate_selection(value, plan, original, request):
    fixed = {"schema": SELECTION, "candidate": CANDIDATE, "optimizer_steps": STEPS,
        "sample_exposures": STEPS * BATCH_SIZE, "batch_size": BATCH_SIZE, "seed": SEED,
        "worker_timeout_seconds": SECONDS, "queue_timeout_minutes": 10, "automatic_retries": 0,
        "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .01},
        "family_rule": "first-six-lexicographically-sorted-declared-train-families; all-six-declared-development-families",
        "required_train_coverage": ["observed-K1", "observed-K2", "ordinary-negative-full-text"],
        "human_gold": False, "quality_acceptance_claim": False, "new_model_calls": 0,
        "selection_used_asr_results": False, "selection_used_model_predictions": False,
        **selection_layout(plan)}
    require(all(value.get(k) == v and type(value[k]) is type(v) for k, v in fixed.items()),
            "V3_INGESTION_PREDECLARED_SELECTION_CHANGED")
    require(value["plan_canonical_sha256"] == digest(plan) and
            value["plan_file_sha256"] == request["plan"]["sha256"] and
            value["anchor_manifest_sha256"] == original["manifest_sha256"] == request["anchor_manifest_sha256"] and
            value["full2601_vocabulary_file_sha256"] == plan["vocabulary"]["file_sha256"] ==
                training.sha_file(Path(request["anchor"]) / "vocabulary.json") and
            value["initial_checkpoint"] == {"path": str(Path(request["anchor"]) / "initial" / (CANDIDATE + ".pt")),
                "sha256": original["initial"]["file_sha256"]} and
            original["base_recipe"]["seed"] == SEED and
            original["base_recipe"]["optimizers"]["fresh-rnn64-or-c1"] == value["optimizer"],
            "V3_INGESTION_ORIGINAL_MODEL_RECIPE_CHANGED")
    created = datetime.fromisoformat(value["created_at_utc"])
    require(created.tzinfo is not None, "V3_INGESTION_SELECTION_TIMEZONE_REQUIRED")
    return value


def verify_closed_scope(folder, expected):
    checkpoints.verify_ack(folder)
    actual = training.read_json(Path(folder) / "capsule.json")["context"]
    require(type(actual.get("owner_invocation")) is str and bool(actual["owner_invocation"]) and
            {k: v for k, v in actual.items() if k != "owner_invocation"} ==
            {**expected, "launch_intent": False}, "V3_INGESTION_CLOSED_ACK_SCOPE_CHANGED")
    return checkpoint_metadata(folder)


def handoff_closure(request):
    """Read the exact original-four/new-five closure, without importing its owner."""
    campaign_path = checked(request["campaign"])
    original_validator = checked(request["campaign_validator"])
    handoff_path, validator = checked(request["handoff"]), checked(request["handoff_validator"])
    campaign, value = training.read_json(campaign_path), training.read_json(handoff_path)
    relative = "tools/research_context_v3_metadata_handoff.py"
    require(validator.name == Path(relative).name and validator.parent.name == "tools" and
            original_validator.name == "research_context_v3_campaign.py" and
            value["schema"] == HANDOFF and handoff_path == Path(value["root"]) / "handoff.json" and
            value["parent_campaign"] == request["campaign"] and
            value["original_validator"] == request["campaign_validator"] and
            value["original_stage"] == campaign["stage"] and
            value["pipeline_files"][relative] == {k: request["handoff_validator"][k] for k in ("bytes", "sha256")} and
            value["driver_runtime"] == campaign["driver_runtime"] and
            value["heavy_lock"] == str(Path(campaign["asr_runtime"]) / "heavy-model.lock"),
            "V3_INGESTION_HANDOFF_VALIDATOR_OR_PARENT_CHANGED")
    result = subprocess.run([value["driver_runtime"]["executable"], "-B", str(validator),
                             "status", "--handoff", str(handoff_path)],
                            cwd=validator.parents[1], capture_output=True, timeout=120)
    require(result.returncode == 0 and len(result.stdout) <= 8 * 1024 ** 2,
            "V3_INGESTION_HANDOFF_VALIDATION_FAILED:" + result.stderr.decode(errors="replace")[-1000:])
    status = training.io_contract.loads(result.stdout)
    jobs = [f"generate-{i:03d}" for i in range(1, 10)]
    budget = status["budget"]
    require(status["schema"] == HANDOFF and status["handoff"] == request["handoff"] and
            status["completed_chunks"] == jobs and status["driver_active"] is False and
            status["new_model_calls"] == 0 and budget["unresolved_jobs"] == budget["failed_jobs"] == [] and
            budget["actual_total_is_complete"] is True and
            budget["overrun_nanoseconds"] == budget["new_overrun_nanoseconds"] == 0 and
            budget["tts_claims_consumed"] == 72 and 71 <= budget["asr_claims_consumed"] <= 191 and
            budget["asr_results_saved"] == budget["asr_claims_consumed"] and
            budget["new_tts_claims"] == 32 and 0 <= budget["new_asr_claims"] <= 120 and
            value["original_completed_chunks"] == jobs[:4] and value["continuation_chunks"] == jobs[4:] and
            value["read_only_generation_jobs"] == jobs[4:5] and value["new_generation_jobs"] == jobs[5:] and
            value["total_actual_runner_minutes"] == 180 and value["new_runner_allocation_minutes"] == 90 and
            value["automatic_retries"] == 0 and value["model_concurrency"] == 1,
            "V3_INGESTION_HANDOFF_NOT_TERMINAL_AND_CLOSED")
    root = Path(value["root"])
    final_path, final_folder = root / "complete.json", root / "checkpoints/handoff-complete"
    require(status["completion"] == pin(final_path) and status["completion_checkpoint"] == str(final_folder),
            "V3_INGESTION_HANDOFF_FINAL_LOCATION_CHANGED")
    final = training.read_json(final_path)
    require(final == {"schema": HANDOFF, "status": "completed", "handoff": request["handoff"],
            "completed_chunks": jobs, "budget": budget, "packages": status["packages"], "human_gold": False,
            "automatic_retries": 0, "original_prefix_modified": False},
            "V3_INGESTION_HANDOFF_FINAL_RECEIPT_CHANGED")
    final_checkpoint = verify_closed_scope(final_folder, {"schema": HANDOFF, "phase": "handoff-complete",
        "handoff": request["handoff"], "completion": pin(final_path)})
    readers = {"context-v3-r1": value["original_reader"], "context-v3-metadata-r2": value["metadata_reader"]}
    require(all(reader["revision"] == revision for revision, reader in readers.items()) and
            all(readers["context-v3-r1"]["source_files_sha256"][name] ==
                readers["context-v3-metadata-r2"]["source_files_sha256"][name] for name in inputs.DATA_FILES),
            "V3_INGESTION_HANDOFF_READER_IDENTITY_CHANGED")
    stage = training.read_json(checked(campaign["stage"]))
    closures, packages, starts = [], [], []
    for index, job in enumerate(jobs):
        original = index < 4
        chunk = Path(campaign["root"] if original else value["root"]) / "chunks" / job
        complete = chunk / "complete.json"
        folder = Path(campaign["root"] if original else value["root"]) / "checkpoints" / job / "chunk-complete"
        if original:
            expected = {"schema": "kws-context-v3-campaign-phase-v1", "phase": "chunk-complete",
                "campaign": request["campaign"], "stage": campaign["stage"], "job_id": job,
                "cell_ids": stage["chunks"][index], "max_tts_calls": 8, "max_asr_calls": 24,
                "automatic_retries": 0, "resume_pending_intent_after_restart": False,
                "asr_request": pin(chunk / "asr-request.json"), "asr_queue": pin(chunk / "asr-queue/queue.json"),
                "completion": pin(complete)}
        else:
            expected = {"schema": HANDOFF, "phase": "chunk-complete", "handoff": request["handoff"],
                "parent_campaign": request["campaign"], "job_id": job, "stage": campaign["stage"],
                "reader": value["metadata_reader"], "max_asr_calls": 24, "automatic_retries": 0,
                "pending_intent_restart_allowed": False, "completion": pin(complete)}
            if (chunk / "asr-request.json").exists():
                expected.update(asr_request=pin(chunk / "asr-request.json"), asr_queue=pin(chunk / "asr-queue/queue.json"))
        metadata = verify_closed_scope(folder, expected)
        closures.append({"job_id": job, "completion": pin(complete), "checkpoint": metadata,
                         "reader_revision": "context-v3-r1" if original else "context-v3-metadata-r2"})
        view = chunk / "weak-view"
        if view.exists():
            declaration = {"path": str(view), "manifest_sha256": training.sha_file(view / "MANIFEST.json"),
                "reader": readers["context-v3-r1" if original else "context-v3-metadata-r2"]}
            handoff_inputs.validate_declaration(declaration)
            packages.append(declaration)
        for claim in (chunk / "asr-queue/jobs/asr-selective/outputs/asr/claims").glob("*/*.json"):
            starts.append(datetime.fromisoformat(training.read_json(claim)["started_at"]))
    require(packages == status["packages"] and packages and starts and all(t.tzinfo is not None for t in starts),
            "V3_INGESTION_HANDOFF_PACKAGE_OR_ASR_ROSTER_CHANGED")
    return {"status": status, "campaign": request["campaign"], "handoff": request["handoff"],
        "handoff_validator": request["handoff_validator"], "completion": pin(final_path),
        "completion_checkpoint": final_checkpoint, "closures": closures, "readers": readers,
        "packages": packages, "first_asr_started_at": min(starts).isoformat(), "new_model_calls": 0}


def campaign_closure(request):
    """Authenticate all nine original closures with the original frozen driver.

    Its audio-preparation runtime is deliberately separate from the core train
    runtime. The subprocess is status-only and creates no model or new claim.
    """
    if request["schema"] == HANDOFF_PREP:
        return handoff_closure(request)
    campaign_path, validator = checked(request["campaign"]), checked(request["campaign_validator"])
    campaign = training.read_json(campaign_path)
    require(validator.name == "research_context_v3_campaign.py" and validator.parent.name == "tools" and
            campaign["pipeline_files"]["tools/research_context_v3_campaign.py"] ==
                {k: request["campaign_validator"][k] for k in ("bytes", "sha256")},
            "V3_INGESTION_ORIGINAL_CAMPAIGN_VALIDATOR_CHANGED")
    result = subprocess.run([campaign["driver_runtime"]["executable"], "-B", str(validator),
                             "status", "--campaign", str(campaign_path)],
                            cwd=validator.parents[1], capture_output=True, timeout=90)
    require(result.returncode == 0 and len(result.stdout) <= 4 * 1024 ** 2,
            "V3_INGESTION_CAMPAIGN_VALIDATION_FAILED:" + result.stderr.decode(errors="replace")[-1000:])
    status = training.io_contract.loads(result.stdout)
    jobs = [f"generate-{i:03d}" for i in range(1, 10)]
    budget = status["budget"]
    require(status["schema"] == campaign["schema"] == "kws-context-v3-campaign-v1" and
            campaign["job_ids"] == jobs and status["completed_chunks"] == jobs and
            status["driver_active"] is False and status["max_chunks"] == 9 and
            budget["unresolved_jobs"] == budget["failed_jobs"] == [] and
            budget["actual_total_is_complete"] is True and budget["overrun_nanoseconds"] == 0,
            "V3_INGESTION_CAMPAIGN_NOT_TERMINAL_AND_CLOSED")
    packages, closures, starts = [], [], []
    for job in jobs:
        chunk = Path(campaign["root"]) / "chunks" / job
        complete = chunk / "complete.json"
        folder = Path(campaign["root"]) / "checkpoints" / job / "chunk-complete"
        checkpoints.verify_ack(folder)
        context = training.read_json(folder / "capsule.json")["context"]
        require(context["phase"] == "chunk-complete" and context["job_id"] == job and
                context["launch_intent"] is False and context["campaign"] == request["campaign"] and
                context["completion"] == pin(complete), "V3_INGESTION_CHUNK_ACK_SCOPE_CHANGED")
        closures.append({"job_id": job, "completion": pin(complete), "checkpoint": checkpoint_metadata(folder)})
        view = chunk / "weak-view"
        if view.exists():
            packages.append({"path": str(view), "manifest_sha256": training.sha_file(view / "MANIFEST.json")})
        for claim in (chunk / "asr-queue/jobs/asr-selective/outputs/asr/claims").glob("*/*.json"):
            starts.append(datetime.fromisoformat(training.read_json(claim)["started_at"]))
    require(packages and starts, "V3_INGESTION_NO_OBSERVED_PACKAGES")
    return {"status": status, "campaign": request["campaign"], "closures": closures,
            "packages": packages, "first_asr_started_at": min(starts).isoformat(), "new_model_calls": 0}


def validate_request(request):
    common = {"schema", "anchor", "anchor_manifest_sha256", "plan", "selection",
              "selection_checkpoint", "campaign", "campaign_validator"}
    original = request.get("schema") == PREP and set(request) == common | {"view_loader"}
    mixed = request.get("schema") == HANDOFF_PREP and set(request) == common | {"handoff", "handoff_validator"}
    require((original or mixed) and Path(request["anchor"]).is_absolute() and
        (not original or Path(request["view_loader"]).is_absolute()) and
        Path(request["selection_checkpoint"]).is_absolute(), "V3_INGESTION_PREPARATION_REQUEST")
    for key in ("plan", "selection", "campaign", "campaign_validator", *(("handoff", "handoff_validator") if mixed else ())):
        checked(request[key])
    return request


def verify_original_implementation(original, anchor_root):
    """Keep the sealed D1 implementation, allowing only runtime report metadata."""
    observed = training.source_records()
    runtime_name = "configs/runtime/research-runtime.json"
    require(set(observed) == set(original["source_hashes"]) and
            all(expected == observed[name] for name, expected in original["source_hashes"].items() if name != runtime_name),
            "V3_INGESTION_ORIGINAL_ALGORITHM_SOURCE_CHANGED")
    prior = training.read_json(Path(anchor_root) / "execution-source" / runtime_name)
    current = training.read_json(ROOT / runtime_name)
    require({k: v for k, v in prior.items() if k != "validation"} ==
            {k: v for k, v in current.items() if k != "validation"}, "V3_INGESTION_ORIGINAL_EFFECTIVE_RUNTIME_CHANGED")


def authenticate(request):
    validate_request(request)
    original = historical.anchor(request["anchor"], request["anchor_manifest_sha256"])
    verify_original_implementation(original, request["anchor"])
    plan = training.read_json(request["plan"]["path"])
    selection = validate_selection(training.read_json(request["selection"]["path"]), plan, original, request)
    folder = Path(request["selection_checkpoint"])
    checkpoints.verify_ack(folder)
    scope = training.read_json(folder / "capsule.json")["context"]
    require(scope["schema"] == "kws-context-v3-ingestion-selection-freeze-v1" and
            scope["selection"] == request["selection"] and scope["plan"] == request["plan"] and
            scope["launch_intent"] is False and scope["selection_used_asr_results"] is False and
            scope["selection_used_model_predictions"] is False,
            "V3_INGESTION_SELECTION_ACK_SCOPE_CHANGED")
    closure = campaign_closure(request)
    require(datetime.fromisoformat(selection["created_at_utc"]) < datetime.fromisoformat(closure["first_asr_started_at"]),
            "V3_INGESTION_SELECTION_MUST_PRECEDE_ASR")
    require(training.read_json(request["campaign"]["path"])["plan_sha256"] == digest(plan),
            "V3_INGESTION_CAMPAIGN_PLAN_CHANGED")
    return original, plan, selection, closure


def enrollment(rows, selection, plan, *, mixed_readers=False):
    """Keep every planned cell; no replacement after observed exclusions."""
    cells = {c["cell_id"]: c for c in plan["cells"]}
    observed = {r["source_id"]: r for r in rows}
    require(len(observed) == len(rows) and set(observed) <= set(cells), "V3_INGESTION_OBSERVED_ROSTER")
    train_ids, dev_ids = selection["train_declared_source_ids"], selection["development_declared_source_ids"]
    require(not set(train_ids) & set(dev_ids), "V3_INGESTION_SELECTION_SPLIT_LEAK")
    ledger = []
    for cell in plan["cells"]:
        ident = cell["cell_id"]
        row = observed.get(ident)
        use = "optimizer" if ident in train_ids else "evaluation" if ident in dev_ids else "audit"
        if row is not None:
            allowed = handoff_inputs.ROW_SCHEMAS if mixed_readers else (inputs.ROW_SCHEMA,)
            require(row["consumer"] in allowed and row["source_family_id"] == cell["source_family_id"] and
                    row["speaker_id"] == cell["speaker_group_id"] and row["source_role"] == cell["role"] and
                    row["split"] == ("train" if cell["role"] == "train-candidate" else "development") and
                    row["generation_category"] == cell["generation_category"], "V3_INGESTION_OBSERVED_ROLE_CHANGED")
            if mixed_readers:
                revision = ("context-v3-r1" if row["consumer"] == inputs.ROW_SCHEMA else "context-v3-metadata-r2")
                require(row["source_adapter_schema"] == handoff_inputs.READERS[revision]["adapter"],
                        "V3_INGESTION_ROW_READER_CHANGED")
                if revision == "context-v3-metadata-r2":
                    handoff_inputs.check_relation(row["native_container_relation"], row["generation_outcome"],
                        row["original_pcm_sha256"], row["original_wav_sha256"])
            require(not row["ctc_mask"] or row["target_ids"] and row["full_text"] and row["ctc_weight"] > 0 and
                    not row["quarantined"] and row["generation_outcome"] == "completed",
                    "V3_INGESTION_COMPLETE_TARGET_REQUIRED")
        eligible = row is not None and row["ctc_mask"]
        reason = ("unobserved_no_audio_or_label" if row is None else
                  "outside_predeclared_train_families" if use == "audit" else
                  "eligible_observed_full_text" if eligible else row["ctc_reason"])
        ledger.append({"source_id": ident, "source_family_id": cell["source_family_id"],
            "speaker_id": cell["speaker_group_id"], "source_role": cell["role"], "planned_use": use,
            "observed": row is not None, "eligible": bool(eligible), "reason": reason,
            "optimizer_allowed": use == "optimizer" and bool(eligible),
            "evaluation_allowed": use == "evaluation" and bool(eligible),
            "row_binding_sha256": row_identity(row) if row is not None else None,
            "generation_outcome": row["generation_outcome"] if row else None,
            "label_status": row["label_status"] if row else None,
            "label_reason": row["label_reason"] if row else None,
            "ctc_reason": row["ctc_reason"] if row else None,
            "ctc_weight": row["ctc_weight"] if row else 0.0})
    train = [observed[i] for i in train_ids if i in observed and observed[i]["ctc_mask"]]
    dev = [observed[i] for i in dev_ids if i in observed and observed[i]["ctc_mask"]]
    require(1 <= len(train) <= 18 and len(dev) <= 18 and
            all(r["split"] == "train" for r in train) and all(r["split"] == "development" for r in dev),
            "V3_INGESTION_NO_ELIGIBLE_PREDECLARED_TRAIN")
    coverage = [any(r["keyword_targets"][i] == 1 and r["keyword_mask"][i] for r in train) for i in range(2)]
    coverage.append(any(r["generation_category"] == "ordinary-negative" and r["keyword_targets"] == [0, 0] and
                        r["keyword_mask"] == [True, True] and r["target_ids"] for r in train))
    require(all(coverage), "V3_INGESTION_PREDECLARED_COVERAGE_UNAVAILABLE")
    return train, dev, ledger


def project(request, original, loader_path, closure, plan):
    if request["schema"] == HANDOFF_PREP:
        packages = training.io_contract.loads(training.canonical(closure["packages"]))
        records = {revision: {"source_root": reader["root"],
            "source_files_sha256": reader["source_files_sha256"],
            "retained_relative_root": "data-sources/" + revision}
            for revision, reader in closure["readers"].items()}
        if loader_path is not None:
            # This changes only an explicit loader location. Its complete source
            # bytes remain exactly those that produced the original package.
            for declaration in packages:
                revision = declaration["reader"]["revision"]
                declaration["reader"]["root"] = str(Path(loader_path) / revision)
        rows, aggregate = handoff_inputs.aggregate(packages, expected_plan_sha256=digest(plan),
            vocabulary_path=Path(request["anchor"]) / "vocabulary.json",
            vocabulary_sha256=plan["vocabulary"]["file_sha256"],
            original_tokens_path=Path(request["anchor"]) / "official/tokens.txt",
            prior_sources=original["prior_sources"])
        return rows, aggregate, {"schema": HANDOFF_DATA, "readers": records}
    loader, data_source = inputs.load_data_module(loader_path)
    rows, aggregate = inputs.verified_projection(closure["packages"], loader, data_source["files"],
        expected_plan_sha256=digest(plan), vocabulary_path=Path(request["anchor"]) / "vocabulary.json",
        vocabulary_sha256=plan["vocabulary"]["file_sha256"],
        original_tokens_path=Path(request["anchor"]) / "official/tokens.txt", prior_sources=original["prior_sources"])
    return rows, aggregate, data_source


def copy_data_sources(output, data_source):
    if data_source.get("schema") == HANDOFF_DATA:
        for value in data_source["readers"].values():
            for name, expected in value["source_files_sha256"].items():
                require(training.copy_once(Path(value["source_root"]) / name,
                    output / value["retained_relative_root"] / name) == expected,
                    "V3_INGESTION_DATA_SOURCE_COPY_CHANGED")
    else:
        for name, expected in data_source["files"].items():
            require(training.copy_once(Path(data_source["root"]) / name, output / "data-source" / name) == expected,
                    "V3_INGESTION_DATA_SOURCE_COPY_CHANGED")


def verify_retained_data_sources(root, data_source):
    require(data_source["schema"] == HANDOFF_DATA and set(data_source["readers"]) == set(handoff_inputs.READERS),
            "V3_INGESTION_RETAINED_READER_ROSTER_CHANGED")
    for revision, value in data_source["readers"].items():
        relative = "data-sources/" + revision
        files = value["source_files_sha256"]
        require(value["retained_relative_root"] == relative and
                set(files) == set(handoff_inputs.READERS[revision]["files"]) and
                {name: training.sha_file(Path(root) / relative / name) for name in files} == files,
                "V3_INGESTION_RETAINED_READER_SOURCE_CHANGED")


def verify_replayed_aggregate(original, replayed, root, declarations):
    """Allow a verified byte-copy loader location, preserving every other field."""
    require([{k: parent[k] for k in ("path", "manifest_sha256", "reader")} for parent in original["parents"]] ==
            declarations, "V3_INGESTION_DECLARED_AGGREGATE_PROVENANCE_CHANGED")
    require(original["schema"] == replayed["schema"] == handoff_inputs.SCHEMA and
            {k: v for k, v in original.items() if k != "parents"} ==
            {k: v for k, v in replayed.items() if k != "parents"} and
            len(original["parents"]) == len(replayed["parents"]), "V3_INGESTION_REPLAYED_AGGREGATE_CHANGED")
    for declared, executed in zip(original["parents"], replayed["parents"]):
        require({k: v for k, v in declared.items() if k != "reader"} ==
                {k: v for k, v in executed.items() if k != "reader"} and
                {k: v for k, v in declared["reader"].items() if k != "root"} ==
                {k: v for k, v in executed["reader"].items() if k != "root"} and
                executed["reader"]["root"] == str(Path(root) / "data-sources" / declared["reader"]["revision"]),
                "V3_INGESTION_REPLAYED_READER_IDENTITY_CHANGED")


def prepare(request, output):
    original, plan, selection, closure = authenticate(request)
    mixed = request["schema"] == HANDOFF_PREP
    rows, aggregate, data_source = project(request, original, request.get("view_loader"), closure, plan)
    train, dev, exclusions = enrollment(rows, selection, plan, mixed_readers=mixed)
    output = training.output_directory(output)
    for row in rows:
        require(training.io_contract.ID.fullmatch(row["source_id"]), "V3_INGESTION_SAFE_SOURCE_ID")
        name = "audio/" + row["source_id"] + ".wav"
        require(training.copy_once(row["audio_path"], output / name) == row["file_sha256"], "V3_INGESTION_AUDIO_COPY_CHANGED")
        row["audio"] = row["audio_path"] = name
    for source, target in ((Path(request["anchor"]) / "initial" / (CANDIDATE + ".pt"), "initial.pt"),
                           (Path(request["anchor"]) / "official/tokens.txt", "tokens.txt"),
                           (Path(request["anchor"]) / "vocabulary.json", "vocabulary.json"),
                           (Path(request["selection"]["path"]), "selection.json"),
                           (Path(request["plan"]["path"]), "plan.json")):
        training.copy_once(source, output / target)
    code = source_records()
    for name, expected in code.items():
        require(training.copy_once(ROOT / name, output / "execution-source" / name) == expected, "V3_INGESTION_SOURCE_COPY_CHANGED")
    copy_data_sources(output, data_source)
    for name, value in (("request.json", request), ("anchor.json", original), ("campaign-closure.json", closure),
                        ("aggregate.json", aggregate), ("enrollment.json", exclusions)):
        training.io_contract.write_json(output / name, value)
    training.io_contract.write_new(output / "projected-rows.jsonl", b"".join(training.canonical(row) + b"\n" for row in rows))
    record = {"schema": PREP, "status": "completed", "source_hashes": code, "data_source": data_source,
        "vocabulary_sha256": digest(original["vocabulary"]), "initial_checkpoint": original["initial"],
        "anchor_manifest_sha256": original["manifest_sha256"], "aggregate": aggregate,
        "selection": request["selection"], "selection_checkpoint": checkpoint_metadata(request["selection_checkpoint"]),
        "row_bindings": [row_identity(row) for row in rows],
        "train_source_ids": [r["source_id"] for r in train], "development_source_ids": [r["source_id"] for r in dev],
        "declared_train_rows": 18, "declared_development_rows": 18, "planned_rows": 72,
        "projection_reasons": dict(Counter(row["ctc_reason"] for row in rows)),
        "eligible_by_split": dict(Counter(row["split"] for row in rows if row["ctc_mask"])),
        "observed_by_split": dict(Counter(row["split"] for row in rows)),
        "enrollment_reasons": dict(Counter(row["reason"] for row in exclusions)),
        "development_optimizer_draws": 0, "new_model_calls": 0, "human_gold": False,
        "independent_test": False, "quality_acceptance_claim": False}
    training.io_contract.write_json(output / "preparation.json", record)
    manifest(output)
    return {"prepared": str(output), "manifest_sha256": training.sha_file(output / "MANIFEST.json"),
            "selected_eligible_train": len(train), "selected_eligible_development": len(dev),
            "enrollment_reasons": record["enrollment_reasons"], "new_model_calls": 0}


def manifest(root):
    training.io_contract.write_json(root / "MANIFEST.json", {p.relative_to(root).as_posix(): training.io_contract.file_record(p)
        for p in sorted(root.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"})


def verify_prepared(root, expected_sha):
    root = Path(root)
    historical.verify_files(root, expected_sha)
    record, request = training.read_json(root / "preparation.json"), training.read_json(root / "request.json")
    require(record["schema"] == PREP and record["status"] == "completed" and
            record["source_hashes"] == source_records(), "V3_INGESTION_PREPARED_SOURCE_CHANGED")
    original, plan, selection, closure = authenticate(request)
    require(original == training.read_json(root / "anchor.json") and closure == training.read_json(root / "campaign-closure.json") and
            training.read_json(root / "selection.json") == selection and training.read_json(root / "plan.json") == plan and
            training.sha_file(root / "selection.json") == request["selection"]["sha256"] and
            training.sha_file(root / "plan.json") == request["plan"]["sha256"] and
            record["selection_checkpoint"] == checkpoint_metadata(request["selection_checkpoint"]),
            "V3_INGESTION_ANCHOR_SELECTION_CLOSURE_CHANGED")
    vocabulary = training.read_json(root / "vocabulary.json")
    require(vocabulary == original["vocabulary"] and digest(vocabulary) == record["vocabulary_sha256"] and
            training.sha_file(root / "vocabulary.json") == selection["full2601_vocabulary_file_sha256"] and
            training.sha_file(root / "tokens.txt") == vocabulary["base_tokens_sha256"] and
            training.sha_file(root / "initial.pt") == original["initial"]["file_sha256"], "V3_INGESTION_VOCAB_OR_INIT_CHANGED")
    mixed = request["schema"] == HANDOFF_PREP
    if mixed:
        verify_retained_data_sources(root, record["data_source"])
    loader = root / "data-sources" if mixed else root / "data-source" / inputs.DATA_FILES[0]
    projected, aggregate, data_source = project(request, original, loader, closure, plan)
    rows = lines(root / "projected-rows.jsonl")
    if mixed:
        require(data_source == record["data_source"], "V3_INGESTION_DECLARED_READER_SOURCES_CHANGED")
        verify_replayed_aggregate(record["aggregate"], aggregate, root, closure["packages"])
    else:
        require(data_source["files"] == record["data_source"]["files"] and aggregate == record["aggregate"],
                "V3_INGESTION_PROJECTED_SOURCE_CHANGED")
    require([row_identity(r) for r in projected] == record["row_bindings"] == [row_identity(r) for r in rows],
            "V3_INGESTION_PROJECTED_LABELS_CHANGED")
    for row in rows:
        require(row["audio"] == row["audio_path"] == "audio/" + row["source_id"] + ".wav" and
                training.sha_file(root / row["audio"]) == row["file_sha256"], "V3_INGESTION_RETAINED_AUDIO_CHANGED")
    train, dev, exclusions = enrollment(rows, selection, plan, mixed_readers=mixed)
    require([r["source_id"] for r in train] == record["train_source_ids"] and
            [r["source_id"] for r in dev] == record["development_source_ids"] and
            exclusions == training.read_json(root / "enrollment.json"), "V3_INGESTION_COHORT_OR_EXCLUSIONS_CHANGED")
    return record, vocabulary, rows, train, dev, original["base_recipe"]


def validate_job(job):
    require(set(job) == {"schema", "prepared", "prepared_manifest_sha256", "candidate", "optimizer_steps",
            "timeout_seconds", "train_binding_sha256", "development_binding_sha256", "selection_sha256", "source_hashes"} and
            job["schema"] == JOB and job["candidate"] == CANDIDATE and job["optimizer_steps"] == STEPS and
            job["timeout_seconds"] == SECONDS and Path(job["prepared"]).is_absolute() and
            all(training.io_contract.SHA.fullmatch(job[k]) for k in
                ("prepared_manifest_sha256", "train_binding_sha256", "development_binding_sha256", "selection_sha256")) and
            job["source_hashes"] == source_records(), "V3_INGESTION_FIXED_JOB_CHANGED")
    return job


def declare(prepared, manifest_sha, output):
    record, _, _, train, dev, _ = verify_prepared(prepared, manifest_sha)
    job = validate_job({"schema": JOB, "prepared": str(Path(prepared).absolute()), "prepared_manifest_sha256": manifest_sha,
        "candidate": CANDIDATE, "optimizer_steps": STEPS, "timeout_seconds": SECONDS,
        "train_binding_sha256": digest([row_identity(r) for r in train]),
        "development_binding_sha256": digest([row_identity(r) for r in dev]),
        "selection_sha256": record["selection"]["sha256"], "source_hashes": source_records()})
    training.io_contract.write_json(output, job)
    return {"recipe": str(output), "recipe_sha256": training.sha_file(output), "train_rows": len(train),
            "development_rows": len(dev), "optimizer_steps": STEPS, "sample_exposures": STEPS * BATCH_SIZE,
            "development_optimizer_draws": 0, "new_model_calls": 0}


def probe_recipe(rows, base):
    require(all(r["split"] == "train" and r["source_role"] == "train-candidate" for r in rows),
            "V3_INGESTION_DEVELOPMENT_OPTIMIZER_FORBIDDEN")
    recipe = diagnostic.tiny_recipe(rows, base, STEPS)
    recipe["schema"] = JOB
    return recipe


def observed_summary(rows):
    return {"eligible_rows": len(rows), "status": "observed" if rows else "no_eligible_rows",
            "metrics": diagnostic.summarize(rows) if rows else None}


class WorkerDeadline(BaseException):
    pass


@contextmanager
def worker_deadline(seconds):
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "V3_INGESTION_EXISTING_DEADLINE")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise WorkerDeadline("V3_INGESTION_WORKER_120_SECOND_DEADLINE")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def feature_rows(prepared, rows, torch):
    import numpy as np
    from frontend import features as frontend_features
    features, observations = [], []
    for row in rows:
        with wave.open(str(prepared / row["audio"]), "rb") as reader:
            require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes(), reader.getcomptype()) ==
                    (1, 2, 16000, row["samples"], "NONE"), "V3_INGESTION_FEATURE_PCM_FORMAT")
            pcm = reader.readframes(reader.getnframes())
        require(hashlib.sha256(pcm).hexdigest() == row["pcm_sha256"], "V3_INGESTION_FEATURE_PCM_CHANGED")
        waveform = torch.from_numpy(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768))
        feature = frontend_features(waveform, feature_dim=32, frontend="logmel").contiguous()
        require(tuple(feature.shape) == (row["expected_c1_frames"], 32) and
                len(feature) >= row["ctc_minimum_frames"] and bool(torch.isfinite(feature).all()), "V3_INGESTION_FEATURE_GEOMETRY")
        features.append(feature)
        observations.append({"source_id": row["source_id"], "split": row["split"], "pcm_sha256": row["pcm_sha256"],
            "frames": len(feature), "feature_sha256": training.tensor_sha(feature)})
    return features, observations


def optimizer_readback(saved, parameters, names, spec):
    """Check actual AdamW groups, mapped moments and every persisted step count."""
    import torch
    require(spec == {"name": "AdamW", "lr": .001, "weight_decay": .01} and
            names == list(parameters) and set(saved) == {"state", "param_groups"} and
            len(saved["param_groups"]) == 1, "V3_INGESTION_ADAMW_PARAMETER_MAPPING_CHANGED")
    expected = {"lr": spec["lr"], "betas": (.9, .999), "eps": 1e-8, "weight_decay": spec["weight_decay"],
        "amsgrad": False, "maximize": False, "foreach": None, "capturable": False,
        "differentiable": False, "fused": None, "decoupled_weight_decay": True,
        "params": list(range(len(names)))}
    require(saved["param_groups"][0] == expected and set(saved["state"]) == set(expected["params"]),
            "V3_INGESTION_ADAMW_SETTINGS_CHANGED")
    moments = []
    for index, name in enumerate(names):
        state, parameter = saved["state"][index], parameters[name]
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"} and
                isinstance(state["step"], torch.Tensor) and tuple(state["step"].shape) == () and
                state["step"].dtype == torch.float32 and float(state["step"]) == STEPS,
                "V3_INGESTION_ADAMW_STEP_CHANGED")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            require(isinstance(value, torch.Tensor) and value.shape == parameter.shape and
                    value.dtype == parameter.dtype and bool(torch.isfinite(value).all()) and
                    (key != "exp_avg_sq" or bool((value >= 0).all())), "V3_INGESTION_ADAMW_MOMENTS_CHANGED")
        moments.append({"parameter": name, "step": STEPS, "exp_avg_sha256": training.tensor_sha(state["exp_avg"]),
                        "exp_avg_sq_sha256": training.tensor_sha(state["exp_avg_sq"])})
    return {"name": "AdamW", "settings": {k: list(v) if isinstance(v, tuple) else v for k, v in expected.items() if k != "params"},
            "parameter_names": names, "steps_per_parameter": STEPS, "moments": moments}


def run(job_path, output, lock):
    job = validate_job(training.read_json(job_path))
    with training.heavy_model_slot(lock) as lease:
        training.queue_binding(job_path, lease)
        verify_worker_intent(job_path, lock, output)
        output = training.output_directory(output)
        start, deadline = time.monotonic(), time.monotonic() + SECONDS
        status = {"schema": RESULT, "status": "running", "started_at": training.utc(), "candidate": CANDIDATE,
            "job_recipe_sha256": training.sha_file(job_path), "human_gold": False, "quality_acceptance_claim": False,
            "tiny_fit_passed": None, "generalization_passed": None, "development_optimizer_draws": 0,
            "development_selection_or_tuning": False, "asr_or_tts_calls": 0, "automatic_retry": False,
            "scope": "fixed context-v3 data ingestion smoke with one held-out development readback"}
        training.save_json(output / "status.json", status)
        try:
            with worker_deadline(SECONDS):
                runtime = training.research_runtime.load_manifest()
                training.research_runtime.check_interpreter(runtime)
                training.research_runtime.check_packages(runtime, "core")
                prepared = Path(job["prepared"])
                record, vocabulary, all_rows, rows, dev, base = verify_prepared(prepared, job["prepared_manifest_sha256"])
                require(digest([row_identity(r) for r in rows]) == job["train_binding_sha256"] and
                        digest([row_identity(r) for r in dev]) == job["development_binding_sha256"] and
                        record["selection"]["sha256"] == job["selection_sha256"], "V3_INGESTION_JOB_ROWS_CHANGED")
                torch, reference = training.torch_runtime()
                from training_state import atomic_save_checkpoint, load_checkpoint_snapshot, state_identity
                features, observations = feature_rows(prepared, rows, torch)
                dev_features, dev_observations = feature_rows(prepared, dev, torch)
                torch.manual_seed(SEED)
                model, description = training.build_fresh(CANDIDATE, vocabulary, reference)
                initial, initial_sha = load_checkpoint_snapshot(prepared / "initial.pt")
                original = training.read_json(prepared / "anchor.json")
                require(initial_sha == record["initial_checkpoint"]["file_sha256"] and
                        initial["schema"] == "kws-d1-initial-checkpoint-v2" and initial["candidate"] == CANDIDATE and
                        initial["vocabulary_sha256"] == digest(vocabulary) and initial["optimizer_steps"] == 0 and
                        initial["source_hashes"] == original["source_hashes"] and initial["description"] == description,
                        "V3_INGESTION_EXACT_ZERO_STEP_INITIAL_CHANGED")
                model.load_state_dict(initial["state_dict"], strict=True)
                require(state_identity(model.state_dict()) == initial["float_state_identity"], "V3_INGESTION_INITIAL_LOAD_CHANGED")
                for source, target in ((prepared / "initial.pt", "initial.pt"),
                                       (prepared / "projected-rows.jsonl", "all-rows.jsonl"),
                                       (prepared / "enrollment.json", "enrollment.json"),
                                       (prepared / "selection.json", "selection.json"), (job_path, "request.json")):
                    training.copy_once(source, output / target)
                for name, value in (("train-rows.json", rows), ("development-rows.json", dev),
                                    ("feature-observations.json", observations + dev_observations)):
                    training.io_contract.write_json(output / name, value)
                recipe = probe_recipe(rows, base)
                training.io_contract.write_json(output / "exposure-recipe.json", recipe)
                before = diagnostic.observe(model, features, rows, "before", output, deadline)
                spec = recipe["optimizers"]["fresh-rnn64-or-c1"]
                optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
                model.train()
                consumed = training.train_batches(model, features, rows, recipe, optimizer, output=output, deadline=deadline)
                gradients = historical.ledger_readback(lines(output / "consumption.jsonl"), recipe, rows)
                names = [name for name, p in model.named_parameters() if p.requires_grad]
                changed, delta = diagnostic.parameter_changes(initial["state_dict"], model.state_dict(), names)
                final_identity = state_identity(model.state_dict())
                after = diagnostic.observe(model, features, rows, "after", output, deadline)
                development = diagnostic.observe(model, dev_features, dev, "development", output, deadline)
                require(state_identity(model.state_dict()) == final_identity, "V3_INGESTION_EVALUATION_MUTATED_MODEL")
                payload = {"schema": RESULT, "candidate": CANDIDATE, "state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "float_state_identity": final_identity,
                    "job_recipe_sha256": training.sha_file(job_path), "optimizer_steps": STEPS,
                    "initial_checkpoint_sha256": initial_sha, "vocabulary_sha256": digest(vocabulary)}
                atomic_save_checkpoint(payload, output / "final.pt")
                restored, final_sha = load_checkpoint_snapshot(output / "final.pt")
                require(state_identity(restored["state_dict"]) == final_identity, "V3_INGESTION_FINAL_READBACK_CHANGED")
                optimizer_receipt = optimizer_readback(restored["optimizer_state"], restored["state_dict"], names, spec)
                require(consumed["optimizer_steps"] == STEPS and consumed["sample_exposures"] == STEPS * BATCH_SIZE and
                        time.monotonic() < deadline, "V3_INGESTION_BUDGET_NOT_CONSUMED")
                status.update(consumed, status="completed", finished_at=training.utc(), wall_seconds=time.monotonic() - start,
                    prepared_manifest_sha256=job["prepared_manifest_sha256"], selection_sha256=job["selection_sha256"],
                    train_binding_sha256=job["train_binding_sha256"], development_binding_sha256=job["development_binding_sha256"],
                    vocabulary_sha256=digest(vocabulary), initial_checkpoint_sha256=initial_sha,
                    initial_float_state_sha256=initial["float_state_identity"]["sha256"],
                    final_float_state_sha256=final_identity["sha256"], final_checkpoint_sha256=final_sha,
                    trainable_parameter_names=names, changed_parameter_names=changed, parameter_delta_l2=delta,
                    optimizer_readback=optimizer_receipt,
                    train_rows=len(rows), development_rows=len(dev), all_observed_rows=len(all_rows), gradients=gradients,
                    before=observed_summary(before), after=observed_summary(after), development=observed_summary(development),
                    training_model_forwards=STEPS * BATCH_SIZE, train_diagnostic_model_forwards=2 * len(rows),
                    model_constructions=1,
                    development_forward_calls=len(dev), development_readback_passes=1,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    consumption_sha256=training.sha_file(output / "consumption.jsonl"), consumption_verified=True,
                    projection_reasons=record["projection_reasons"], enrollment_reasons=record["enrollment_reasons"])
                training.save_json(output / "status.json", status)
                training.io_contract.write_json(output / "summary.json", status)
                manifest(output)
                return status
        except BaseException as error:
            status.update(status="failed", error_type=type(error).__name__, error=str(error), wall_seconds=time.monotonic() - start)
            if (output / "consumption.jsonl").exists():
                status["completed_ledger_steps"] = sum(line.endswith(b"\n") for line in
                    (output / "consumption.jsonl").read_bytes().splitlines(keepends=True))
            training.save_json(output / "status.json", status)
            raise


def audit(output, expected_sha):
    output = Path(output)
    historical.verify_files(output, expected_sha)
    job, summary = validate_job(training.read_json(output / "request.json")), training.read_json(output / "summary.json")
    record, vocabulary, all_rows, rows, dev, base = verify_prepared(job["prepared"], job["prepared_manifest_sha256"])
    require(summary["schema"] == RESULT and summary["status"] == "completed" and
            summary["job_recipe_sha256"] == training.sha_file(output / "request.json") and
            summary["optimizer_steps"] == STEPS and summary["sample_exposures"] == STEPS * BATCH_SIZE and
            summary["development_optimizer_draws"] == summary["asr_or_tts_calls"] == 0 and
            summary["development_selection_or_tuning"] is False and summary["quality_acceptance_claim"] is False and
            summary["tiny_fit_passed"] is summary["generalization_passed"] is None and
            summary["development_readback_passes"] == 1 and summary["development_forward_calls"] == len(dev) and
            summary["model_constructions"] == 1 and
            summary["training_model_forwards"] == STEPS * BATCH_SIZE and summary["train_diagnostic_model_forwards"] == 2 * len(rows) and
            training.read_json(output / "train-rows.json") == rows and training.read_json(output / "development-rows.json") == dev and
            lines(output / "all-rows.jsonl") == all_rows and
            training.read_json(output / "enrollment.json") == training.read_json(Path(job["prepared"]) / "enrollment.json") and
            training.sha_file(output / "selection.json") == job["selection_sha256"] == summary["selection_sha256"] and
            summary["train_binding_sha256"] == job["train_binding_sha256"] == digest([row_identity(r) for r in rows]) and
            summary["development_binding_sha256"] == job["development_binding_sha256"] == digest([row_identity(r) for r in dev]) and
            summary["vocabulary_sha256"] == digest(vocabulary), "V3_INGESTION_RESULT_SCOPE_CHANGED")
    recipe = probe_recipe(rows, base)
    require(training.read_json(output / "exposure-recipe.json") == recipe and
            historical.ledger_readback(lines(output / "consumption.jsonl"), recipe, rows) == summary["gradients"],
            "V3_INGESTION_LEDGER_CHANGED")
    torch, _ = training.torch_runtime()
    from training_state import load_checkpoint_snapshot, state_identity
    initial, initial_sha = load_checkpoint_snapshot(output / "initial.pt")
    final, final_sha = load_checkpoint_snapshot(output / "final.pt")
    require(initial_sha == summary["initial_checkpoint_sha256"] == record["initial_checkpoint"]["file_sha256"] and
            state_identity(initial["state_dict"]) == initial["float_state_identity"] and
            initial["float_state_identity"]["sha256"] == summary["initial_float_state_sha256"] and
            final_sha == summary["final_checkpoint_sha256"] and final["schema"] == RESULT and
            final["candidate"] == CANDIDATE and final["optimizer_steps"] == STEPS and
            final["vocabulary_sha256"] == digest(vocabulary) and final["initial_checkpoint_sha256"] == initial_sha and
            state_identity(final["state_dict"]) == final["float_state_identity"] and
            final["float_state_identity"]["sha256"] == summary["final_float_state_sha256"] and
            final["job_recipe_sha256"] == summary["job_recipe_sha256"], "V3_INGESTION_CHECKPOINT_CHANGED")
    changed, delta = diagnostic.parameter_changes(initial["state_dict"], final["state_dict"], summary["trainable_parameter_names"])
    require(changed == summary["changed_parameter_names"] and delta == summary["parameter_delta_l2"], "V3_INGESTION_PARAMETER_DELTA_CHANGED")
    require(optimizer_readback(final["optimizer_state"], final["state_dict"], summary["trainable_parameter_names"],
                              recipe["optimizers"]["fresh-rnn64-or-c1"]) == summary["optimizer_readback"],
            "V3_INGESTION_ADAMW_READBACK_CHANGED")
    features = {r["source_id"]: r for r in training.read_json(output / "feature-observations.json")}
    require(set(features) == {r["source_id"] for r in rows + dev}, "V3_INGESTION_FEATURE_ROSTER_CHANGED")
    for phase, selected in (("before", rows), ("after", rows), ("development", dev)):
        observed = training.read_json(output / (phase + ".json"))
        require(len(observed) == len(selected), "V3_INGESTION_NUMERIC_ROSTER_CHANGED")
        for row, item in zip(selected, observed):
            numeric = item["numeric_artifact"]
            path = output / training.io_contract.relative(numeric["path"])
            require(training.io_contract.file_record(path) == {k: numeric[k] for k in ("bytes", "sha256")},
                    "V3_INGESTION_NUMERIC_FILE_CHANGED")
            saved = torch.load(path, weights_only=True, map_location="cpu")
            feature = features[row["source_id"]]
            require(item["source_id"] == saved["source_id"] == row["source_id"] and
                    item["label_binding_sha256"] == row_identity(row) and item["pcm_sha256"] == row["pcm_sha256"] and
                    training.tensor_sha(saved["feature"]) == saved["feature_sha256"] == feature["feature_sha256"] and
                    feature["split"] == row["split"] and feature["pcm_sha256"] == row["pcm_sha256"] and
                    tuple(saved["feature"].shape) == (feature["frames"], 32) and
                    feature["frames"] == row["expected_c1_frames"] and
                    training.tensor_sha(saved["logits"]) == saved["logits_sha256"] and
                    all(item[k] == v for k, v in diagnostic.score(saved["logits"], row).items()),
                    "V3_INGESTION_SAVED_NUMERIC_CHANGED")
        require(observed_summary(observed) == summary[phase], "V3_INGESTION_NUMERIC_SUMMARY_CHANGED")
    return {"schema": "kws-context-v3-ingestion-readback-v1", "status": "verified", "manifest_sha256": expected_sha,
            "optimizer_steps_replayed": STEPS, "new_optimizer_steps": 0, "new_model_forwards": 0,
            "train_rows": len(rows), "development_rows": len(dev), "development_optimizer_draws": 0,
            "consumption_verified": True, "quality_acceptance_claim": False, "human_gold": False}


def verify_worker_intent(job_path, lock, output):
    owner, folder = os.environ.get(OWNER_ENV), os.environ.get(INTENT_ENV)
    require(owner and folder and Path(folder).is_absolute(), "V3_INGESTION_LIVE_INTENT_OWNER_REQUIRED")
    checkpoints.verify_ack(folder)
    scope = checkpoints.verify_capsule(folder, current_sources=True)["context"]
    stage_path = checked(scope["stage"])
    value = load_stage(stage_path)
    expected = {**stage_scope(stage_path, value), "phase": "before-training",
                "owner_invocation": owner, "launch_intent": True}
    require(scope == expected and Path(folder) == Path(value["root"]) / "checkpoints/before-training" and
            value["recipe"] == pin(job_path) and value["heavy_lock"] == str(Path(lock).absolute()) and
            Path(output).absolute() == Path(value["root"]) / "queue/jobs" / JOB_ID / "outputs" and
            os.environ.get("KWS_RESEARCH_JOB_ID") == JOB_ID and
            os.environ.get("KWS_RESEARCH_RECIPE") == str(Path(job_path).absolute()) and
            os.environ.get("KWS_RESEARCH_OUTPUT_DIR") == str(Path(output).absolute()),
            "V3_INGESTION_WORKER_INTENT_SCOPE_CHANGED")


def runtime_identity():
    runtime = training.research_runtime.load_manifest()
    training.research_runtime.check_interpreter(runtime)
    training.research_runtime.check_packages(runtime, "core")
    return {"python": sys.version.split()[0], "executable": str(Path(sys.executable).absolute()),
            "runtime_manifest_sha256": training.sha_file(ROOT / "configs/runtime/research-runtime.json")}


def queue_definition(recipe, python, heavy_lock):
    q = training.io_contract
    return {"schema": q.SCHEMA, "queue_id": JOB_ID, "global_budget_minutes": 10, "jobs": [{
        "job_id": JOB_ID, "recipe": str(Path(recipe).absolute()), "recipe_sha256": training.sha_file(recipe),
        "argv": [python, "-B", str(ROOT / "training/research_context_v3_consume.py"), "run",
                 "--recipe", "{recipe}", "--output", "{output_dir}", "--heavy-lock", str(heavy_lock)],
        "cwd": str(ROOT), "depends_on": [], "timeout_minutes": 10, "outputs": list(QUEUE_OUTPUTS)}]}


def stage(recipe, output, active_plan, heavy_lock):
    q = training.io_contract
    recipe, output, heavy_lock = q.safe_path(recipe), q.safe_path(output), q.safe_path(heavy_lock)
    job = validate_job(training.read_json(recipe))
    verify_prepared(job["prepared"], job["prepared_manifest_sha256"])
    preparation = training.read_json(Path(job["prepared"]) / "request.json")
    campaign = training.read_json(preparation["campaign"]["path"])
    require(heavy_lock == Path(campaign["asr_runtime"]) / "heavy-model.lock", "V3_INGESTION_CANONICAL_SHARED_LOCK_REQUIRED")
    identity = runtime_identity()
    output = training.output_directory(output)
    request = queue_definition(recipe, identity["executable"], heavy_lock)
    q.write_json(output / "queue-request.json", request)
    q.create(output / "queue", output / "queue-request.json", active_plan, heavy_lock)
    value = {"schema": STAGE, "root": str(output), "queue": pin(output / "queue/queue.json"),
        "request": pin(output / "queue/request.json"), "active_plan": pin(output / "queue/active-plan.json"),
        "recipe": pin(output / "queue/recipes" / (JOB_ID + ".json")), "original_recipe": pin(recipe),
        "heavy_lock": str(heavy_lock), "runtime": identity, "source_hashes": source_records(),
        "automatic_retries": 0, "worker_seconds": SECONDS, "queue_minutes": 10,
        "pending_intent_restart_allowed": False, "slot_wait_seconds": 60, "human_gold": False}
    q.write_json(output / "stage.json", value)
    return {"stage": str(output / "stage.json"), "stage_sha256": training.sha_file(output / "stage.json"),
            "new_model_calls": 0, "queue_minutes": 10, "worker_seconds": SECONDS}


def load_stage(path):
    q = training.io_contract
    value = training.read_json(path)
    require(value["schema"] == STAGE and Path(path).absolute() == Path(value["root"]) / "stage.json" and
            value["automatic_retries"] == 0 and value["worker_seconds"] == SECONDS and value["queue_minutes"] == 10 and
            value["pending_intent_restart_allowed"] is False and value["slot_wait_seconds"] == 60 and value["human_gold"] is False and
            value["source_hashes"] == source_records() and value["runtime"] == runtime_identity(),
            "V3_INGESTION_STAGE_CHANGED")
    for key in ("queue", "request", "active_plan", "recipe", "original_recipe"):
        checked(value[key])
    root, metadata, request, _sha = q.load_queue(Path(value["root"]) / "queue")
    require(request == queue_definition(value["original_recipe"]["path"], value["runtime"]["executable"], value["heavy_lock"]) and
            metadata["heavy_lock"] == value["heavy_lock"] and value["recipe"] == pin(root / "recipes" / (JOB_ID + ".json")),
            "V3_INGESTION_QUEUE_BINDING_CHANGED")
    return value


def stage_sources(value):
    root = Path(value["root"])
    job = validate_job(training.read_json(value["recipe"]["path"]))
    prepared = Path(job["prepared"])
    historical.verify_files(prepared, job["prepared_manifest_sha256"])
    request = training.read_json(prepared / "request.json")
    sources = [root / "stage.json", root / "queue-request.json", Path(value["original_recipe"]["path"]),
               *checkpoints.regular_tree(root / "queue"),
               *checkpoints.regular_tree(prepared), *[ROOT / name for name in CODE]]
    sources += [Path(request[k]["path"]) for k in ("plan", "selection", "campaign", "campaign_validator")]
    sources += [Path(p["path"]) for p in checkpoint_metadata(request["selection_checkpoint"]).values()]
    sources += [Path(request["anchor"]) / "MANIFEST.json", Path(request["anchor"]) / "preparation.json"]
    closure = training.read_json(prepared / "campaign-closure.json")
    for item in closure["closures"]:
        sources += [Path(item["completion"]["path"])]
        sources += [Path(p["path"]) for p in item["checkpoint"].values()]
    if request["schema"] == HANDOFF_PREP:
        handoff = training.read_json(checked(request["handoff"]))
        campaign = training.read_json(checked(request["campaign"]))
        sources += [checked(request[key]) for key in ("handoff", "handoff_validator")]
        sources += [checked(closure["completion"]),
                    *[checked(p) for p in closure["completion_checkpoint"].values()]]
        for revision, reader in closure["readers"].items():
            require(reader["revision"] == revision, "V3_INGESTION_RETAINED_READER_DECLARATION_CHANGED")
            sources += [Path(reader["root"]) / name for name in reader["source_files_sha256"]]
        for owner, manifest in ((request["handoff_validator"], handoff), (request["campaign_validator"], campaign)):
            sources += [Path(owner["path"]).parents[1] / name for name in manifest["pipeline_files"]]
        sources += [checked(handoff[key]) for key in ("original_stage", "initial_status", "failure_receipt")]
        sources += [checked(p) for p in checkpoint_metadata(handoff["failure_checkpoint"]).values()]
        sources += [checked(p) for p in handoff["original_generation_results"].values()]
        sources += [Path(handoff["registration"])]
    for name in ("before-training", "training-terminal"):
        folder = root / "checkpoints" / name
        if (folder / "ack.json").exists():
            sources += [Path(p["path"]) for p in checkpoint_metadata(folder).values()]
    return sources


def stage_scope(stage_path, value):
    return {"schema": STAGE, "stage": pin(stage_path), "recipe": value["recipe"],
        "queue": value["queue"], "heavy_lock": value["heavy_lock"], "automatic_retries": 0,
        "worker_seconds": SECONDS, "queue_minutes": 10, "slot_wait_seconds": 60,
        "pending_intent_restart_allowed": False}


def execute(stage_path, *, checkpoint_gate=None):
    """One exact queue attempt; terminal evidence is saved before readback."""
    q = training.io_contract
    value = load_stage(stage_path)
    root, queue_root = Path(value["root"]), Path(value["root"]) / "queue"
    if checkpoint_gate is None:
        announced = set()
        def on_wait(folder, descriptor):
            if str(folder) not in announced:
                print(json.dumps({"event": "awaiting_durable_save", "checkpoint": str(folder),
                                  "capsule": descriptor["capsule"]}, sort_keys=True), flush=True)
                announced.add(str(folder))
        checkpoint_gate = checkpoints.Gate(on_wait=on_wait)
    scope = stage_scope(stage_path, value)
    with q.exclusive(root / "owner.lock"):
        report = q.inspect(queue_root, reconcile=True)
        state = report["jobs"][JOB_ID]["status"]
        if state == "running":
            return {"status": "running", "new_model_calls": 0}, 75
        if state == "pending":
            folder = root / "checkpoints/before-training"
            checkpoint_gate.checkpoint(folder, {**scope, "phase": "before-training"}, stage_sources(value), intent=True)
            require(OWNER_ENV not in os.environ and INTENT_ENV not in os.environ, "V3_INGESTION_OWNER_ENV_CONFLICT")
            os.environ.update({OWNER_ENV: checkpoint_gate.owner, INTENT_ENV: str(folder)})
            try:
                wait_started = time.monotonic()
                while True:
                    checkpoints.verify_capsule(folder, current_sources=True)
                    report, _code = q.run(queue_root, max_jobs=1)
                    if report["status"] != "waiting_for_heavy_model_slot":
                        break
                    require(report["jobs"][JOB_ID]["status"] == "pending" and
                            not (queue_root / "jobs" / JOB_ID / "claim.json").exists(),
                            "V3_INGESTION_BUSY_QUEUE_ALREADY_CONSUMED")
                    require(time.monotonic() - wait_started < value["slot_wait_seconds"],
                            "V3_INGESTION_SHARED_SLOT_WAIT_EXHAUSTED")
                    time.sleep(.2)
            finally:
                os.environ.pop(OWNER_ENV, None); os.environ.pop(INTENT_ENV, None)
            report = q.inspect(queue_root, reconcile=True)
            state = report["jobs"][JOB_ID]["status"]
        if state in ("pending", "running"):
            return {"status": state, "automatic_retry": False}, 75
        result_path = queue_root / "jobs" / JOB_ID / "result.json"
        require(result_path.is_file(), "V3_INGESTION_AUTHENTIC_TERMINAL_RECEIPT_REQUIRED")
        checkpoint_gate.checkpoint(root / "checkpoints/training-terminal",
            {**scope, "phase": "training-terminal", "terminal_state": state, "queue_result": pin(result_path)},
            stage_sources(value), intent=False)
        if state != "completed":
            return {"status": state, "queue_result": pin(result_path), "automatic_retry": False}, 1
        if report["budget_overrun_nanoseconds"]:
            return {"status": "budget_overrun", "queue_result": pin(result_path),
                    "budget_overrun_nanoseconds": report["budget_overrun_nanoseconds"], "automatic_retry": False}, 1
        output = queue_root / "jobs" / JOB_ID / "outputs"
        receipt = root / "readback.json"
        failure = root / "readback-failure.json"
        if not receipt.exists() and not failure.exists():
            # Fresh interpreter avoids mixing the package's frozen verifier
            # modules with any independently prepared source root in this owner.
            try:
                result = subprocess.run([value["runtime"]["executable"], "-B", str(ROOT / "training/research_context_v3_consume.py"),
                    "audit", "--output", str(output), "--manifest-sha256", training.sha_file(output / "MANIFEST.json"),
                    "--receipt", str(receipt)], capture_output=True, timeout=180)
                stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired as error:
                stdout, stderr, exit_code = error.stdout or b"", error.stderr or b"", "timed_out"
            q.write_new(root / "readback.stdout.log", stdout)
            q.write_new(root / "readback.stderr.log", stderr)
            if exit_code:
                q.write_json(root / "readback-failure.json", {"status": "failed", "exit_code": exit_code,
                    "new_model_forwards": 0, "automatic_retry": False})
        sources = stage_sources(value) + [p for p in (receipt, failure, root / "readback.stdout.log", root / "readback.stderr.log") if p.exists()]
        checkpoint_gate.checkpoint(root / "checkpoints/readback-complete",
            {**scope, "phase": "readback-complete", "queue_result": pin(result_path),
             "readback": pin(receipt) if receipt.exists() else None,
             "failure": pin(failure) if failure.exists() else None}, sources, intent=False)
        return {"status": "completed" if receipt.exists() and not failure.exists() else "readback_failed",
                "readback": str(receipt) if receipt.exists() else None, "quality_acceptance_claim": False}, 1 if failure.exists() else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--request", required=True); p.add_argument("--output", required=True)
    p = commands.add_parser("declare")
    for name in ("prepared", "prepared-manifest-sha256", "output"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("run")
    for name in ("recipe", "output", "heavy-lock"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("audit")
    for name in ("output", "manifest-sha256", "receipt"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("stage")
    for name in ("recipe", "output", "active-plan", "heavy-lock"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("execute")
    p.add_argument("--stage", required=True)
    args = parser.parse_args(argv)
    code = 0
    if args.command == "prepare":
        result = prepare(training.read_json(args.request), Path(args.output))
    elif args.command == "declare":
        result = declare(args.prepared, args.prepared_manifest_sha256, Path(args.output))
    elif args.command == "run":
        result = run(Path(args.recipe), Path(args.output), Path(args.heavy_lock))
    elif args.command == "audit":
        result = audit(args.output, args.manifest_sha256)
        training.io_contract.write_json(Path(args.receipt), result)
    elif args.command == "stage":
        result = stage(args.recipe, args.output, args.active_plan, args.heavy_lock)
    else:
        result, code = execute(Path(args.stage))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
