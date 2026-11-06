#!/usr/bin/env python3
"""Compare two sealed tiny-fit jobs whose only training change is exposure.

This reader constructs no model and performs no inference or optimization. It
authenticates the original independent audit receipts and output files, then
checks the common initialization, rows, optimizer, saved initial logits and
actual consumed-step prefix. A longer job is a separately declared experiment;
neither the earlier job nor its result is resumed, rewritten or reclassified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_d1_fit_diagnostic as diagnostic
import research_d1_train as training
from research_d1_recipe import BATCH_SIZE, row_identity
from research_d1_vocab import digest, require

SCHEMA = "kws-tiny-fit-exposure-comparison-v1"


def load_audited_output(output, audit_path, audit_sha256):
    """Authenticate a retained full audit, without pretending to repeat it."""
    output, audit_path = Path(output), Path(audit_path)
    require(training.io_contract.SHA.fullmatch(audit_sha256) and
            training.sha_file(audit_path) == audit_sha256, "FIT_COMPARE_AUDIT_CHANGED")
    audit = training.read_json(audit_path)
    require(type(audit) is dict and audit.get("status") == "verified" and
            audit.get("new_model_forwards") == audit.get("optimizer_steps") == 0 and
            type(audit.get("tiny_fit_passed")) is bool and
            type(audit.get("manifest_sha256")) is str and
            training.io_contract.SHA.fullmatch(audit["manifest_sha256"]),
            "FIT_COMPARE_INDEPENDENT_AUDIT_REQUIRED")
    manifest_path = output / "MANIFEST.json"
    require(training.sha_file(manifest_path) == audit["manifest_sha256"], "FIT_COMPARE_MANIFEST_CHANGED")
    manifest = training.read_json(manifest_path)
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() and p != manifest_path}
    require(type(manifest) is dict and set(manifest) == actual, "FIT_COMPARE_FILE_ROSTER_CHANGED")
    for name, record in manifest.items():
        require(training.io_contract.file_record(output / training.io_contract.relative(name)) == record,
                "FIT_COMPARE_FILE_CHANGED:" + name)
    job = diagnostic.validate_job(training.read_json(output / "request.json"))
    summary = training.read_json(output / "summary.json")
    require(summary.get("schema") == diagnostic.RESULT_SCHEMA and summary.get("status") == "completed" and
            summary.get("job_recipe_sha256") == training.sha_file(output / "request.json") and
            summary.get("candidate") == job["candidate"] and
            summary.get("prepared_manifest_sha256") == job["prepared_manifest_sha256"] and
            summary.get("diagnostic_source_hashes") == job["diagnostic_source_hashes"] and
            summary.get("row_binding_sha256") == job["row_binding_sha256"], "FIT_COMPARE_JOB_CHANGED")
    for name, sha in job["diagnostic_source_hashes"].items():
        require(training.sha_file(output / "execution-source" / name) == sha,
                "FIT_COMPARE_DIAGNOSTIC_SOURCE_CHANGED")
    rows = training.read_json(output / "selected-rows.json")
    require(type(rows) is list and [row["source_id"] for row in rows] == job["source_ids"] and
            digest([row_identity(row) for row in rows]) == job["row_binding_sha256"] and
            all(row["split"] == "train" and row["source_role"] == "train-candidate" and
                row["label_status"] == "machine_candidate" and row["ctc_mask"] is True and
                0 < row["ctc_weight"] <= 1 and row["human_gold"] is False and
                row["independent_test"] is False for row in rows), "FIT_COMPARE_TRAINING_ROWS_CHANGED")
    recipe = training.read_json(output / "exposure-recipe.json")
    ledger_bytes = training.io_contract.read_bytes(output / "consumption.jsonl", 64 * 1024 * 1024)
    lines = ledger_bytes.splitlines(keepends=True)
    require(lines and all(line.endswith(b"\n") for line in lines), "FIT_COMPARE_PARTIAL_STEP_LEDGER")
    ledger = [training.io_contract.loads(line) for line in lines]
    require(hashlib.sha256(ledger_bytes).hexdigest() == summary["consumption_sha256"] and
            diagnostic.gradient_observation(ledger, recipe, rows) == summary["gradients"] and
            len(ledger) == summary["optimizer_steps"] == job["optimizer_steps"] == recipe["optimizer_steps"] and
            summary["sample_exposures"] == recipe["total_exposures"] == BATCH_SIZE * len(ledger) and
            recipe["exposure_plan_sha256"] == digest(recipe["exposure_plan"]["draws"]),
            "FIT_COMPARE_STEP_ACCOUNTING_CHANGED")
    phases = {}
    for phase in ("before", "after"):
        observations = training.read_json(output / (phase + ".json"))
        require(type(observations) is list and len(observations) == len(rows), "FIT_COMPARE_OBSERVATION_ROSTER")
        for row, observed in zip(rows, observations):
            require(observed["source_id"] == row["source_id"] and
                    observed["label_binding_sha256"] == row_identity(row) and
                    observed["target_ids"] == row["target_ids"] and
                    observed["ctc_weight"] == row["ctc_weight"] and observed["ctc_mask"] is True,
                    "FIT_COMPARE_OBSERVED_LABEL_CHANGED")
            numeric = observed["numeric_artifact"]
            require(training.io_contract.file_record(output / training.io_contract.relative(numeric["path"])) ==
                    {key: numeric[key] for key in ("bytes", "sha256")}, "FIT_COMPARE_NUMERIC_CHANGED")
        require(diagnostic.summarize(observations) == summary[phase], "FIT_COMPARE_SAVED_SUMMARY_CHANGED")
        phases[phase] = observations
    require(summary["development_forward_calls"] == summary["asr_or_tts_calls"] == 0 and
            summary["human_gold"] is False and summary["independent_human_test"] is False and
            summary["automatic_retry"] is False and summary["quality_winner"] is None and
            summary["keyword_loss_used"] is False and
            summary["tiny_fit_passed"] == audit["tiny_fit_passed"] ==
                (summary["after"]["exact_token_matches"] == len(rows) and
                 bool(summary["changed_parameter_names"]) and summary["gradients"]["nonzero_gradient_steps"] > 0),
            "FIT_COMPARE_SCOPE_CHANGED")
    return {"output": str(output.absolute()), "audit": str(audit_path.absolute()),
            "audit_sha256": audit_sha256, "manifest_sha256": audit["manifest_sha256"],
            "job": job, "summary": summary, "rows": rows, "recipe": recipe,
            "ledger_lines": lines, "phases": phases}


def compare(short, long):
    """Require fixed inputs/policy; report any observed common-prefix drift."""
    a, b = short["summary"], long["summary"]
    require(a["optimizer_steps"] < b["optimizer_steps"], "FIT_COMPARE_ORDERED_EXPOSURE_BUDGETS_REQUIRED")
    for key in ("candidate", "prepared_manifest_sha256", "row_binding_sha256", "vocabulary_sha256",
                "initial_checkpoint_sha256", "initial_float_state_sha256", "diagnostic_source_hashes",
                "trainable_parameter_names", "selected_training_rows"):
        require(a[key] == b[key], "FIT_COMPARE_FIXED_CONTROL_CHANGED:" + key)
    require(short["rows"] == long["rows"], "FIT_COMPARE_SELECTED_ROWS_CHANGED")
    omitted = {"exposure_plan", "exposure_plan_sha256", "optimizer_steps", "total_exposures"}
    controls = [{k: v for k, v in item["recipe"].items() if k not in omitted} for item in (short, long)]
    require(controls[0] == controls[1], "FIT_COMPARE_OPTIMIZATION_POLICY_CHANGED")
    short_draws = short["recipe"]["exposure_plan"]["draws"]
    long_draws = long["recipe"]["exposure_plan"]["draws"]
    require(short_draws == long_draws[:len(short_draws)] and
            len(short_draws) == BATCH_SIZE * a["optimizer_steps"] and
            len(long_draws) == BATCH_SIZE * b["optimizer_steps"], "FIT_COMPARE_EXPOSURE_PREFIX_CHANGED")
    for index, draw in enumerate(long_draws):
        row_index = index % len(long["rows"])
        row = long["rows"][row_index]
        expected = {"row_index": row_index, "source_id": row["source_id"], "ctc_weight": row["ctc_weight"]}
        require(draw == expected, "FIT_COMPARE_EXTENDED_SAMPLING_POLICY_CHANGED")
    same_initial_observations = short["phases"]["before"] == long["phases"]["before"]
    prefix = long["ledger_lines"][:len(short["ledger_lines"])]
    first_changed = next((index + 1 for index, (x, y) in
                          enumerate(zip(short["ledger_lines"], prefix)) if x != y), None)
    same_prefix = prefix == short["ledger_lines"]
    controlled = same_initial_observations and same_prefix
    outcomes = {}
    for label, item in (("short", short), ("long", long)):
        summary = item["summary"]
        outcomes[label] = {"manifest_sha256": item["manifest_sha256"], "audit_sha256": item["audit_sha256"],
            "optimizer_steps": summary["optimizer_steps"], "sample_exposures": summary["sample_exposures"],
            "initial_checkpoint_sha256": summary["initial_checkpoint_sha256"],
            "final_checkpoint_sha256": summary["final_checkpoint_sha256"],
            "after": summary["after"], "tiny_fit_passed": summary["tiny_fit_passed"]}
    return {"schema": SCHEMA, "status": "controlled-comparison" if controlled else "numeric-control-mismatch",
        "candidate": a["candidate"], "changed_training_variable": "predeclared optimizer steps and repeated exposures",
        "prepared_manifest_sha256": a["prepared_manifest_sha256"], "row_binding_sha256": a["row_binding_sha256"],
        "vocabulary_sha256": a["vocabulary_sha256"], "selected_training_rows": a["selected_training_rows"],
        "before_observations_and_numeric_files_identical": same_initial_observations,
        "common_optimizer_steps": a["optimizer_steps"], "common_consumption_prefix_byte_identical": same_prefix,
        "first_different_common_step": first_changed, "only_exposure_budget_control_verified": controlled,
        "outcomes": outcomes, "new_model_forwards": 0, "new_optimizer_steps": 0, "asr_or_tts_calls": 0,
        "scope": "known machine-weak training-row memorization only", "human_gold": False,
        "independent_effectiveness_claim": False, "quality_winner": None,
        "prior_result_rewritten": False, "audit_mode": "hash-authenticated independent audit receipts; no new tensor audit",
        "interpretation": ("Compare final memorization under the declared exposure budgets. High blank occupancy alone "
                           "does not imply empty output or training failure; inspect exact complete targets and edit counts."
                           if controlled else "Numeric controls differ. Retain both observations, but do not attribute their "
                           "difference solely to the exposure budget.")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("short", "long"):
        for name in ("output", "audit", "audit-sha256"):
            parser.add_argument("--" + label + "-" + name, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    bundles = [load_audited_output(getattr(args, name + "_output"), getattr(args, name + "_audit"),
                                  getattr(args, name + "_audit_sha256")) for name in ("short", "long")]
    result = compare(*bundles)
    result["comparison_source_sha256"] = training.sha_file(Path(__file__))
    training.io_contract.write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["only_exposure_budget_control_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
