#!/usr/bin/env python3
"""Supplement a completed frozen C1 audit with per-draw loss arithmetic.

Read only the sealed bundle and a hash-bound original audit receipt. Do not
import the training module: its recipe binds the original execution source.
This stdlib check neither loads tensors nor calls a model or optimizer.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_queue as io

SCHEMA = "kws-c1-silence-ledger-supplement-v1"
EXECUTION_SCHEMA = "kws-c1-known-silence-correction-v1"
EXECUTION_SOURCE = "training/research_c1_known_silence.py"
EXECUTION_SHA = "85d7e9a1d1cbc33b103c523455971fd866ede049478f464f44530255ea932d4a"
FROZEN_SOURCE_COMMIT = "d250a12373f68f937fc7c4864b42a682b08240fa"
SOURCE = "tools/research_c1_silence_ledger_audit.py"
STEPS, BATCH = 2000, 4
require = io.require


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "SUPPLEMENT_FINITE_NUMBER_REQUIRED")
    return value


def close(actual, expected, reason, *, absolute=1e-7):
    require(math.isclose(number(actual), number(expected), rel_tol=1e-6, abs_tol=absolute), reason)


def check_ledger(events, recipe):
    """Verify each contribution, including errors that cancel within a batch."""
    require(recipe["schema"] == EXECUTION_SCHEMA and
        type(recipe["optimizer_steps"]) is int and recipe["optimizer_steps"] == STEPS and
        type(recipe["batch_size"]) is int and recipe["batch_size"] == BATCH and
        type(recipe["fixed_ctc_normalizer"]) is float and recipe["fixed_ctc_normalizer"] == 1.0 and
        type(recipe["auxiliary_scale"]) is float and recipe["auxiliary_scale"] == 1.0,
        "SUPPLEMENT_FIXED_LOSS_CONTRACT_CHANGED")
    plan = recipe["exposure_plan"]["draws"]
    require(len(events) == STEPS and len(plan) == STEPS * BATCH, "SUPPLEMENT_LEDGER_COUNT_CHANGED")
    checked = 0
    for step, event in enumerate(events):
        require(type(event["optimizer_step"]) is int and event["optimizer_step"] == step + 1 and
            type(event["draw_offset"]) is int and event["draw_offset"] == step * BATCH and
            len(event["draws"]) == BATCH, "SUPPLEMENT_STEP_ORDER_CHANGED")
        primary, auxiliary, totals = [], [], []
        for item, draw in zip(event["draws"], plan[step * BATCH:(step + 1) * BATCH]):
            require(all(type(item[k]) is type(draw[k]) and item[k] == draw[k] for k in
                ("source_id", "ctc_weight", "offset_ms", "row_visit_index", "known_zero_prefix_frames")),
                "SUPPLEMENT_DRAW_BINDING_CHANGED")
            require(item["ctc_mask"] is True and type(item["auxiliary_scale"]) is float and
                item["auxiliary_scale"] == 1.0, "SUPPLEMENT_LOSS_WEIGHT_CHANGED")
            weight = number(item["ctc_weight"])
            aux_nll = number(item["aux_blank_nll_mean"])
            require(0 < weight <= 1 and aux_nll >= 0 and
                (item["offset_ms"] != 0 or aux_nll == 0), "SUPPLEMENT_LOSS_DOMAIN_CHANGED")
            ctc = number(item["raw_length_normalized_ctc"]) * weight / BATCH
            aux = aux_nll / BATCH
            close(item["weighted_batch_contribution"], ctc, "SUPPLEMENT_PRIMARY_CONTRIBUTION_CHANGED")
            close(item["aux_batch_contribution"], aux, "SUPPLEMENT_AUX_CONTRIBUTION_CHANGED")
            close(item["total_batch_contribution"], ctc + aux, "SUPPLEMENT_DRAW_TOTAL_CHANGED")
            primary.append(ctc)
            auxiliary.append(aux)
            totals.append(number(item["total_batch_contribution"]))
            checked += 1
        close(event["fulltext_ctc_batch_loss"], math.fsum(primary), "SUPPLEMENT_BATCH_PRIMARY_CHANGED", absolute=1e-6)
        close(event["aux_batch_loss"], math.fsum(auxiliary), "SUPPLEMENT_BATCH_AUX_CHANGED", absolute=1e-6)
        close(event["loss"], math.fsum(totals), "SUPPLEMENT_BATCH_TOTAL_CHANGED", absolute=1e-6)
        close(event["loss"], math.fsum(primary) + math.fsum(auxiliary), "SUPPLEMENT_BATCH_COMPONENTS_CHANGED", absolute=1e-6)
    return {"optimizer_steps_checked": len(events), "draw_contributions_checked": checked,
        "per_draw_total_equals_primary_plus_auxiliary": True}


def verify_manifest(output, expected):
    raw = io.read_bytes(output / "MANIFEST.json")
    require(io.sha(raw) == expected, "SUPPLEMENT_MANIFEST_CHANGED")
    manifest = io.loads(raw)
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
        if p.is_file() and p != output / "MANIFEST.json"}
    require(type(manifest) is dict and set(manifest) == actual, "SUPPLEMENT_FILE_ROSTER_CHANGED")
    for name, record in manifest.items():
        require(io.file_record(output / io.relative(name)) == record, "SUPPLEMENT_FILE_CHANGED:" + name)
    return manifest


def read_member(output, manifest, name):
    raw = io.read_bytes(output / name, 64 * 1024 * 1024)
    require({"bytes": len(raw), "sha256": io.sha(raw)} == manifest[name], "SUPPLEMENT_MEMBER_CHANGED:" + name)
    return raw


def audit(output, manifest_sha256, original_audit, original_audit_sha256):
    output, original_audit = io.safe_path(output), io.safe_path(original_audit)
    require(all(type(value) is str and io.SHA.fullmatch(value)
        for value in (manifest_sha256, original_audit_sha256)), "SUPPLEMENT_PIN_REQUIRED")
    audit_raw = io.read_bytes(original_audit)
    require(io.sha(audit_raw) == original_audit_sha256, "SUPPLEMENT_ORIGINAL_AUDIT_CHANGED")
    prior = io.loads(audit_raw)
    expected = {"schema": EXECUTION_SCHEMA, "status": "verified", "manifest_sha256": manifest_sha256,
        "saved_optimizer_steps": STEPS, "saved_training_forwards": STEPS * BATCH, "saved_final_conditions": 24,
        "new_model_constructions": 0, "new_model_forwards": 0, "new_optimizer_steps": 0}
    require(all(type(prior.get(k)) is type(v) and prior[k] == v for k, v in expected.items()),
        "SUPPLEMENT_VERIFIED_ORIGINAL_AUDIT_REQUIRED")
    manifest = verify_manifest(output, manifest_sha256)
    recipe_raw = read_member(output, manifest, "request.json")
    recipe = io.loads(recipe_raw)
    summary = io.loads(read_member(output, manifest, "summary.json"))
    require(summary["schema"] == EXECUTION_SCHEMA and summary["status"] == "completed" and
        summary["recipe_sha256"] == io.sha(recipe_raw), "SUPPLEMENT_COMPLETED_RECIPE_BINDING_REQUIRED")
    require(recipe["source_hashes"][EXECUTION_SOURCE] == EXECUTION_SHA, "SUPPLEMENT_FROZEN_EXECUTION_SOURCE_CHANGED")
    for name, digest in recipe["source_hashes"].items():
        saved = "execution-source/" + io.relative(name)
        require(manifest[saved]["sha256"] == digest, "SUPPLEMENT_EXECUTION_SOURCE_BINDING_CHANGED")
    events = [io.loads(line) for line in read_member(output, manifest, "consumption.jsonl").splitlines()]
    result = check_ledger(events, recipe)
    return {"schema": SCHEMA, "status": "verified", "manifest_sha256": manifest_sha256,
        "original_audit": {"bytes": len(audit_raw), "sha256": original_audit_sha256},
        "frozen_execution_source": {"recorded_commit": FROZEN_SOURCE_COMMIT,
            "path": EXECUTION_SOURCE, "sha256": EXECUTION_SHA},
        "supplement_source_files": {name: io.file_record(ROOT / name) for name in
            (SOURCE, "tools/research_queue.py", "tools/research_job_lock.py")},
        **result, "new_model_constructions": 0, "new_model_forwards": 0, "new_optimizer_steps": 0,
        "scope": "Supplemental sealed-ledger arithmetic; original tensor/optimizer audit remains separate."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "manifest-sha256", "original-audit", "original-audit-sha256", "receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    receipt, output = io.safe_path(args.receipt), io.safe_path(args.output)
    require(receipt != output and output not in receipt.parents, "SUPPLEMENT_RECEIPT_MUST_BE_OUTSIDE_SEALED_OUTPUT")
    result = audit(output, args.manifest_sha256, args.original_audit, args.original_audit_sha256)
    io.write_json(receipt, result)
    print(io.canonical(result).decode().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
