#!/usr/bin/env python3
"""Reconcile authenticated checkpoint copies without reviving a lost queue.

The archived file manifest is an explicit trust anchor supplied by the caller.
This tool verifies bytes and the original receipt chain through a read-only path
map. It never creates model claims, relabels absent work as pending, or changes
an original receipt. The resulting exclusion is deliberately conservative.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_queue as q

SCHEMA = "kws-d2-runtime-loss-v1"
REQUEST = "kws-d2-runtime-loss-request-v1"


def read(path):
    return q.loads(q.read_bytes(path))


def pin(path):
    return {"path": str(q.safe_path(path)), **q.file_record(path)}


def producer_digest(value):
    return hashlib.sha256(q.canonical(value)[:-1]).hexdigest()


def campaign_identity(plan):
    return producer_digest({"parent_plan_sha256": plan["d1_parent"]["plan_sha256"],
                            "stage": plan["stage"], "batch_id": plan["batch_id"]})


def absolute(value):
    q.require(type(value) is str and value.startswith("/") and
              str(PurePosixPath(value)) == value and ".." not in PurePosixPath(value).parts,
              "RECOVERY_ORIGINAL_PATH_REQUIRED")
    return value


class Checkpoints:
    """Resolve original absolute names only from hash-verified manifest rows."""

    def __init__(self, declarations):
        q.require(type(declarations) is list and declarations, "RECOVERY_CHECKPOINTS_REQUIRED")
        self.originals, self.anchors = {}, []
        for item in declarations:
            root = q.safe_path(item["root"])
            expected = item["manifest_sha256"]
            q.require(q.SHA.fullmatch(expected) and q.file_record(root / "MANIFEST.json")["sha256"] == expected,
                      "RECOVERY_MANIFEST_PIN_CHANGED")
            manifest = read(root / "MANIFEST.json")
            q.require(manifest.get("hash_algorithm") == "SHA-256" and
                      manifest.get("self_excluded") == ["MANIFEST.json", "SHA256SUMS"],
                      "RECOVERY_MANIFEST_SCOPE")
            files = manifest["files"]
            q.require(type(files) is dict and 0 < len(files) <= 100000, "RECOVERY_FILE_COUNT")
            actual = set()
            for path in root.rglob("*"):
                q.require(not path.is_symlink(), "RECOVERY_PATH_ALIAS")
                if path.is_file():
                    actual.add(path.relative_to(root).as_posix())
            q.require(actual == set(files) | {"MANIFEST.json", "SHA256SUMS"},
                      "RECOVERY_ARCHIVE_FILE_SET_CHANGED")
            for name, row in files.items():
                path = root / q.relative(name)
                expected_record = {k: row[k] for k in ("bytes", "sha256")}
                q.require(q.file_record(path) == expected_record, "RECOVERY_FILE_CHANGED:" + name)
                original = row.get("source_path")
                if original is not None:
                    original = absolute(original)
                    old = self.originals.get(original)
                    q.require(old is None or old[1] == expected_record,
                              "RECOVERY_CONFLICTING_ORIGINAL_BYTES:" + original)
                    self.originals[original] = (path, expected_record)
            self.anchors.append({"root": str(root), "manifest": pin(root / "MANIFEST.json"),
                                 "verified_files": len(files), "scope": pin(root / "CHECKPOINT_SCOPE.json")})

    def get(self, original, expected=None):
        original = absolute(str(original))
        q.require(original in self.originals, "RECOVERY_REQUIRED_ORIGINAL_ABSENT:" + original)
        path, actual = self.originals[original]
        if expected is not None:
            q.require(actual == {k: expected[k] for k in ("bytes", "sha256")},
                      "RECOVERY_ORIGINAL_PIN_CHANGED:" + original)
        return path

    def value(self, original, expected=None):
        return read(self.get(original, expected))

    def record(self, original):
        return self.originals[absolute(str(original))][1]

    def files(self, original_root):
        prefix = absolute(str(original_root)).rstrip("/") + "/"
        return sorted(name for name in self.originals if name.startswith(prefix))


def verify_job(copies, root, job_id, queue_sha):
    """Authenticate an archived job; never invoke the live queue inspector."""
    root = PurePosixPath(root)
    job = root / "jobs" / job_id
    request = copies.value(root / "request.json")
    metadata = copies.value(root / "queue.json")
    q.require(copies.record(root / "queue.json")["sha256"] == queue_sha and
              copies.record(root / "request.json")["sha256"] == metadata["request_sha256"],
              "RECOVERY_QUEUE_PIN_CHANGED")
    definitions = [item for item in request["jobs"] if item["job_id"] == job_id]
    q.require(len(definitions) == 1, "RECOVERY_JOB_DEFINITION")
    definition = definitions[0]
    result, claim = copies.value(job / "result.json"), copies.value(job / "claim.json")
    definition_sha = q.sha(q.canonical(definition))
    q.require(result.get("schema") == q.RECEIPT_SCHEMA and result.get("status") == "completed" and
              result.get("exit_code") == 0 and result.get("automatic_retry") is False and
              result.get("actual_duration_known") is True and
              type(result.get("actual_elapsed_ns")) is int and result["actual_elapsed_ns"] > 0 and
              claim.get("attempt") == 1 and claim.get("automatic_retry") is False and
              result["claim_sha256"] == copies.record(job / "claim.json")["sha256"],
              "RECOVERY_JOB_TERMINAL_CHANGED")
    for value in (claim, result):
        q.require(value["job_id"] == job_id and value["queue_sha256"] == queue_sha and
                  value["job_definition_sha256"] == definition_sha and
                  value["recipe_sha256"] == definition["recipe_sha256"], "RECOVERY_JOB_OWNER_CHANGED")
    recipe_path = root / "recipes" / (job_id + ".json")
    q.require(copies.record(recipe_path)["sha256"] == definition["recipe_sha256"], "RECOVERY_RECIPE_CHANGED")
    dependencies = copies.record(job / "dependencies.json")
    q.require(dependencies["sha256"] == claim["dependencies_sha256"], "RECOVERY_DEPENDENCIES_CHANGED")
    for name, record in result.get("outputs", {}).items():
        copies.get(job / "outputs" / q.relative(name), record)
    for name, record in result.get("logs", {}).items():
        copies.get(job / q.relative(name), record)
    return result, copies.value(recipe_path)


def reconcile_campaign(copies, original):
    campaign = copies.value(original)
    root = PurePosixPath(campaign["root"])
    q.require(str(root / "campaign.json") == original and campaign["schema"] == "kws-d2-campaign-v1" and
              campaign["automatic_retries"] == 0 and campaign["model_concurrency"] == 1,
              "RECOVERY_CAMPAIGN_SCOPE")
    stage_pin = campaign["stage"]
    stage = copies.value(stage_pin["path"], stage_pin)
    prepared = PurePosixPath(stage["prepared"])
    plan = copies.value(prepared / "plan.json", stage["plan"])
    cells = [row["cell_id"] for row in plan["cells"]]
    chunks = stage["chunks"]
    q.require(len(cells) == len(set(cells)) and [c for chunk in chunks for c in chunk] == cells and
              stage["queue_root"] == campaign["generation_queue"]["path"] and
              stage["automatic_retries"] == 0, "RECOVERY_PLAN_ROSTER_CHANGED")
    job_ids = campaign["job_ids"]
    q.require(job_ids == [f"generate-{i + 1:03d}" for i in range(campaign["max_chunks"])] and
              0 < len(job_ids) <= len(chunks), "RECOVERY_CAMPAIGN_PREFIX")
    sealed, proven, known_ns, stats = [], [], 0, Counter()
    for index, job_id in enumerate(job_ids):
        original_chunk = root / "chunks" / job_id
        completed_path = str(original_chunk / "complete.json")
        if completed_path not in copies.originals:
            continue
        receipt = copies.value(completed_path)
        q.require(receipt["schema"] == "kws-d2-campaign-chunk-complete-v1" and
                  receipt["campaign"] == {"path": original, **copies.record(original)} and
                  receipt["job_id"] == job_id and receipt["status"] in
                  ("completed", "completed_no_asr_eligible_return") and receipt["automatic_retries"] == 0 and
                  receipt["human_gold"] is False, "RECOVERY_CHUNK_RECEIPT_CHANGED")
        for name, record in receipt["files"].items():
            copies.get(original_chunk / q.relative(name), record)
        generation, recipe = verify_job(copies, stage["queue_root"], job_id,
                                       campaign["generation_queue"]["queue_sha256"])
        q.require(recipe["selected_cells"] == chunks[index] and recipe["stage"] == stage_pin,
                  "RECOVERY_SELECTED_CELLS_CHANGED")
        output = PurePosixPath(stage["queue_root"]) / "jobs" / job_id / "outputs"
        summary = copies.value(output / "summary.json")
        q.require(summary["claims"] == len(chunks[index]) and summary["native_generation_calls"] == len(chunks[index]) and
                  [row["cell_id"] for row in summary["rows"]] == chunks[index], "RECOVERY_NATIVE_ROSTER_CHANGED")
        inventory = copies.value(output / "inventory.json")
        for name, record in inventory["files"].items():
            copies.get(output / q.relative(name), record)
        for name, record in inventory["generation_files"].items():
            copies.get(PurePosixPath(stage["generation_root"]) / q.relative(name), record)
        chunk_ns = generation["actual_elapsed_ns"]
        if receipt["statistics"] is not None:
            asr_root = original_chunk / "asr-queue"
            asr, _ = verify_job(copies, str(asr_root), "asr-selective", copies.record(asr_root / "queue.json")["sha256"])
            chunk_ns += asr["actual_elapsed_ns"]
            for name in ("source_items", "machine_candidates", "uncertain", "completed_asr_results"):
                stats[name] += receipt["statistics"][name]
        proven.extend(chunks[index])
        known_ns += chunk_ns
        sealed.append({"job_id": job_id, "cell_ids": chunks[index], "complete": copies.record(completed_path),
                       "native_returns": summary["native_returns"], "outcomes": summary["outcomes"],
                       "actual_runner_nanoseconds": chunk_ns, "statistics": receipt["statistics"]})
    declared = [cell for chunk in chunks[:len(job_ids)] for cell in chunk]
    unknown = [cell for cell in declared if cell not in set(proven)]
    return {"original_campaign": original, "campaign": copies.record(original), "stage": stage_pin,
            "plan_canonical_sha256": producer_digest(plan), "campaign_identity": campaign_identity(plan),
            "declared_cell_ids": declared, "archived_completed_cell_ids": proven,
            "post_checkpoint_outcome_unknown_cell_ids": unknown, "sealed_chunks": sealed,
            "declared_cells": len(declared), "archived_completed_cells": len(proven),
            "post_checkpoint_outcome_unknown_cells": len(unknown), "statistics": dict(stats),
            "actual_known_archived_runner_nanoseconds": known_ns, "actual_post_checkpoint_runner_nanoseconds": None,
            "post_checkpoint_calls_known": False, "post_checkpoint_pending_count": None,
            "resume_allowed": False, "resubmission_allowed": False,
            "status": "retired_after_local_runtime_loss"}


def reconcile(request_path, output):
    request = read(request_path)
    q.require(request["schema"] == REQUEST and request.get("local_runtime_lost") is True and
              request.get("post_checkpoint_execution_was_possible") is True,
              "RECOVERY_EXPLICIT_LOSS_SCOPE_REQUIRED")
    copies = Checkpoints(request["checkpoints"])
    candidates = [name for name in copies.originals if name.endswith("/campaign.json") and
                  copies.value(name).get("schema") == "kws-d2-campaign-v1"]
    q.require(candidates and len(candidates) == len(set(candidates)), "RECOVERY_CAMPAIGNS_REQUIRED")
    campaigns = [reconcile_campaign(copies, name) for name in sorted(candidates)]
    all_ids = [cell for campaign in campaigns for cell in campaign["declared_cell_ids"]]
    q.require(len(all_ids) == len(set(all_ids)), "RECOVERY_CAMPAIGNS_OVERLAP")
    value = {"schema": SCHEMA, "created_at": q.utc(), "request": pin(request_path),
             "checkpoints": copies.anchors, "campaigns": campaigns,
             "declared_cells": sum(v["declared_cells"] for v in campaigns),
             "archived_completed_cells": sum(v["archived_completed_cells"] for v in campaigns),
             "post_checkpoint_outcome_unknown_cells": sum(v["post_checkpoint_outcome_unknown_cells"] for v in campaigns),
             "new_model_calls": 0, "original_files_modified": False,
             "absent_receipt_means_pending": False, "resubmission_allowed": False,
             "unrecorded_calls_or_duration_assumed_zero": False, "human_gold": False,
             "resolution": "Only newly recovered authentic original receipts may refine unknown outcomes; no automatic retry."}
    q.write_json(q.safe_path(output), value)
    return value


def guard_plan(plan, loss_pins):
    """Reject a known retired scope even if its filename or JSON spacing changed."""
    ids = {row["cell_id"] for row in plan["cells"]}
    for loss_pin in loss_pins:
        q.require(pin(loss_pin["path"]) == loss_pin, "D2_HISTORY_LOSS_PIN_CHANGED")
        loss = read(loss_pin["path"])
        q.require(loss["schema"] == SCHEMA and loss["resubmission_allowed"] is False,
                  "D2_HISTORY_LOSS_SCHEMA")
        for campaign in loss["campaigns"]:
            q.require(not ids.intersection(campaign["declared_cell_ids"]) and
                      campaign_identity(plan) != campaign["campaign_identity"],
                      "D2_RETIRED_CAMPAIGN_RESUBMISSION_FORBIDDEN")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recover = commands.add_parser("reconcile")
    recover.add_argument("--request", type=Path, required=True)
    recover.add_argument("--output", type=Path, required=True)
    guard = commands.add_parser("guard-plan")
    guard.add_argument("--plan", type=Path, required=True)
    guard.add_argument("--loss", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "reconcile":
            result = reconcile(args.request, args.output)
            result = {k: result[k] for k in ("schema", "declared_cells", "archived_completed_cells",
                                            "post_checkpoint_outcome_unknown_cells", "new_model_calls")}
        else:
            guard_plan(read(args.plan), [pin(p) for p in args.loss])
            result = {"status": "no_known_retired_scope_overlap", "new_model_calls": 0}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps({"status": "invalid", "reason": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
