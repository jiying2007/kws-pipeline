"""Close the 256 planned cells using immutable views and excluded terminals.

No mutable generation ledger, private operator module, waveform synthesis or
machine-learning import is used. An exclusion is evidence, never a PCM row.
The explicit accounted branch delegates original partial-job authentication to
the retained, source-pinned data loader, then independently closes the roster.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import struct

import research_queue as io

require = io.require
SCHEMA = "kws-d1-native-terminal-coverage-v1"
ACCOUNTED_SCHEMA = "kws-d1-native-accounted-coverage-v2"
ACCOUNTED_PRIOR = "d1-accounted-prior-evidence-v2"
ACCOUNTED_VERIFICATION = "kws-d1-accounted-prior-verification-v2"
SOURCE_REVISION = "43ffea59304121691a84424236543c00a56de588"
TRANSITIONS = ((8, 64), (64, 120), (120, 176), (176, 232), (232, 256))


def plain_digest(value):
    # Bulk plans and cell identities omit LF; queue/terminal files retain LF.
    return io.sha(io.canonical(value)[:-1])


def pin(path):
    path = io.safe_path(path)
    return {"path": str(path), **io.file_record(path)}


def pinned(value):
    require(type(value) is dict and set(value) in ({"path", "sha256"}, {"path", "sha256", "bytes"})
        and type(value["path"]) is str and Path(value["path"]).is_absolute()
        and type(value["sha256"]) is str and io.SHA.fullmatch(value["sha256"]), "D1_COVERAGE_PIN")
    raw = io.read_bytes(value["path"], 64 * 1024 * 1024)
    require(io.sha(raw) == value["sha256"] and ("bytes" not in value or len(raw) == value["bytes"]),
        "D1_COVERAGE_PIN_CHANGED")
    return io.loads(raw)


def excluded_record(row, outcome, stage_sha, recipe_sha):
    """Join the complete original failed terminal and its create-only claim."""
    require(type(row) is dict and row["asr_calls"] == 0 and row["observed_text"] is None
        and row["training_admitted"] is False and type(row["reason"]) is str and row["reason"],
        "D1_COVERAGE_EXCLUDED_IS_UNOBSERVED")
    matches = [item for item in outcome["rows"] if item["cell_id"] == row["cell_id"]]
    require(len(matches) == 1, "D1_COVERAGE_EXCLUDED_OUTCOME_ROSTER")
    item, record = matches[0], row["record"]
    require(item["record"] == record and item["ledger_record_sha256"] == row["ledger_record_sha256"]
        and io.sha(io.canonical(record)) == row["ledger_record_sha256"], "D1_COVERAGE_EXCLUDED_TERMINAL_HASH")
    claim = item["claim"]
    require(record["schema"] == "active-tts-plan-cell-v1" and record["cell_id"] == row["cell_id"]
        and record["status"] == "FAILED_NO_RETRY" and record["origin"] == "extension-call"
        and record["new_generation_calls"] == 1 and record["training_admitted"] is False
        and claim["schema"] == "active-tts-plan-call-claim-v1" and claim["cell_id"] == row["cell_id"]
        and claim["attempt"] == 1 and claim["identity_sha256"] == record["identity_sha256"]
        and io.sha(io.canonical(claim)) == record["claim_sha256"]
        and claim["owner"] == {"stage_sha256": stage_sha, "job_id": row["job_id"],
                                "job_recipe_sha256": recipe_sha}, "D1_COVERAGE_EXCLUDED_CALL_IDENTITY")
    return {"exported_row": row, "original_outcome_row": item}


def native_eligibility(record):
    """Replay the frozen producer's decoder geometry, not its quality gate."""
    name = record["native_wav"]
    if name is None:
        return {"eligible": False, "reason": "NO_DECODABLE_NATIVE_WAV"}
    path = io.safe_path(Path(record["output_root"]) / io.relative(name))
    require(io.file_record(path) == record["files"][name], "D1_COVERAGE_NATIVE_BYTES_CHANGED")
    try:
        require(44 <= path.stat().st_size <= 16_000_000, "ASR_WAV_FILE_BOUND")
        body = io.read_bytes(path, 16_000_000)
        require(body[:4] == b"RIFF" and body[8:12] == b"WAVE" and
            struct.unpack_from("<I", body, 4)[0] + 8 == len(body), "ASR_RIFF_GEOMETRY")
        offset, fmt, data = 12, None, None
        while offset < len(body):
            require(offset + 8 <= len(body), "ASR_RIFF_CHUNK_BOUND")
            name, size = body[offset:offset + 4], struct.unpack_from("<I", body, offset + 4)[0]
            offset += 8
            require(offset + size <= len(body), "ASR_RIFF_CHUNK_BOUND")
            chunk = body[offset:offset + size]
            if name == b"fmt ":
                require(fmt is None and len(chunk) >= 16, "ASR_FORMAT_CHUNK")
                fmt = struct.unpack_from("<HHIIHH", chunk)
            elif name == b"data":
                require(data is None, "ASR_DUPLICATE_DATA_CHUNK")
                data = chunk
            offset += size + size % 2
        require(offset == len(body) and fmt == (3, 1, 24000, 96000, 4, 32) and data is not None,
            "ASR_MONO_FLOAT_24000_REQUIRED")
        require(len(data) % 4 == 0 and 1 <= len(data) // 4 <= 720000, "ASR_FRAME_BOUND")
        maximum = 0.0
        for (sample,) in struct.iter_unpack("<f", data):
            require(math.isfinite(sample), "ASR_NONFINITE_SAMPLE")
            maximum = max(maximum, abs(sample))
        require(maximum <= 1_000_000, "ASR_UNSAFE_NUMERIC_RANGE")
        return {"eligible": True, "reason": None, "frames": len(data) // 4,
                "native_wav_sha256": record["files"][record["native_wav"]]["sha256"]}
    except (ValueError, struct.error) as error:
        return {"eligible": False, "reason": str(error)}


def excluded_native(row):
    eligibility = native_eligibility(row["record"])
    require(eligibility["eligible"] is False and row["reason"] == eligibility["reason"],
        "D1_COVERAGE_EXCLUSION_REQUIRES_UNDECODABLE_NATIVE")
    return eligibility


def excluded_export(entry, plan, plan_sha, plan_file_sha, queue_reader):
    """Replay an export, both completed jobs and original exclusion receipts."""
    index = pinned(entry)
    stage_pin = {"path": index["stage"], "sha256": index["stage_sha256"]}
    stage = pinned(stage_pin)
    roster = [cell["cell_id"] for cell in plan["cells"]]
    prior, target = len(stage["initial_reused_cells"]), stage["target_count"]
    selected = roster[prior:target]
    require((prior, target) in TRANSITIONS and stage["schema"] == "active-tts-extension-stage-v1"
        and stage["source_revision"] == SOURCE_REVISION and stage["target_cells"] == roster[:target]
        and set(stage["initial_reused_cells"]) == set(roster[:prior])
        and stage["chunks"] == [roster[i:min(i + 28, target)] for i in range(prior, target, 28)]
        and stage["maximum_new_calls"] == target - prior and stage["maximum_job_seconds"] == 600
        and stage["stage_budget_minutes"] == 20 and stage["max_retries"] == 0
        and stage["paired_cases_included"] is False and stage["training_admitted"] is False
        and stage["binding"]["plan_canonical_sha256"] == plan_sha
        and stage["binding"]["plan_file_sha256"] == plan_file_sha,
        "D1_COVERAGE_EXCLUDED_STAGE_SCOPE")
    require(index["schema"] == "active-tts-asr-partitions-v1" and index["plan_canonical_sha256"] == plan_sha
        and index["plan_file_sha256"] == plan_file_sha and index["aggregate_cell_count"] == target
        and index["paired_cases_included"] is False and index["training_admitted"] is False
        and index["asr_calls_executed"] == 0 and index["excluded_cells"]
        and set(index["reused_cells_without_new_asr_request"]) == set(roster[:prior]),
        "D1_COVERAGE_EXCLUDED_EXPORT_SCOPE")
    # Generation used the original controller. Its strict source check runs
    # in the independently pinned old-source subprocess, never this new one.
    queue_state = queue_reader(stage["queue_root"])
    queue_root, request = Path(queue_state["root"]), queue_state["request"]
    require(queue_root == io.safe_path(stage["queue_root"]), "D1_COVERAGE_QUEUE_ROOT")
    jobs, new_ids, evidence = {}, set(), [pin(entry["path"]), pin(stage_pin["path"])]
    for binding in index["job_outcomes"]:
        name = binding["job_id"]
        require(name not in jobs, "D1_COVERAGE_DUPLICATE_EXPORT_JOB")
        outcome, job = pinned(binding["outcomes"]), pinned(binding["job_recipe"])
        definitions = [value for value in request["jobs"] if value["job_id"] == name]
        require(len(definitions) == 1, "D1_COVERAGE_TTS_QUEUE_JOB")
        terminal = queue_state["jobs"][name]
        require(terminal["status"] == "completed" and terminal["exit_code"] == 0
            and terminal["actual_duration_known"] is True and 0 <= terminal["actual_elapsed_ns"] <= 600_000_000_000
            and terminal["recipe_sha256"] == binding["job_recipe"]["sha256"]
            and terminal["outputs"]["outcomes.json"] == {key: binding["outcomes"][key] for key in ("bytes", "sha256")},
            "D1_COVERAGE_TTS_TERMINAL_CHANGED")
        require(outcome["schema"] == "active-tts-stage-outcomes-v1" and outcome["stage_sha256"] == index["stage_sha256"]
            and outcome["job_recipe_sha256"] == binding["job_recipe"]["sha256"]
            and outcome["plan_canonical_sha256"] == plan_sha and outcome["binding"] == stage["binding"]
            and outcome["selected_cells"] == job["cells"] == stage["chunks"][job["chunk_index"]]
            and name == job["job_id"] == outcome["job_id"] and outcome["reused_cell_ids"] == []
            and outcome["new_call_ids"] == job["cells"] and not new_ids.intersection(job["cells"]),
            "D1_COVERAGE_EXCLUDED_OUTCOME_BINDING")
        require([row["cell_id"] for row in outcome["rows"]] == job["cells"], "D1_COVERAGE_OUTCOME_ROWS")
        new_ids.update(job["cells"])
        jobs[name] = (binding, outcome)
        evidence.extend([pin(binding["outcomes"]["path"]), pin(binding["job_recipe"]["path"]),
                         pin(queue_root / "jobs" / name / "result.json")])
    require(new_ids == set(selected), "D1_COVERAGE_TTS_NEW_ROSTER")
    eligible, excluded, records = set(), set(), []
    for partition in index["partitions"]:
        name, ids = partition["job_id"], partition["cell_ids"]
        require(name in jobs and 1 <= len(ids) <= 28 and len(set(ids)) == len(ids)
            and set(ids) <= set(jobs[name][1]["new_call_ids"]) and not eligible.intersection(ids)
            and partition["outcomes"] == jobs[name][0]["outcomes"]
            and partition["job_recipe"] == jobs[name][0]["job_recipe"]
            and partition["recognizer_context_text"] is None, "D1_COVERAGE_ELIGIBLE_PARTITION")
        eligible.update(ids)
    for row in index["excluded_cells"]:
        name = row["job_id"]
        require(name in jobs and row["cell_id"] in new_ids and row["cell_id"] not in eligible | excluded
            and row["outcomes"] == jobs[name][0]["outcomes"], "D1_COVERAGE_EXCLUDED_ROSTER")
        retained = excluded_record(row, jobs[name][1], index["stage_sha256"], jobs[name][0]["job_recipe"]["sha256"])
        record = row["record"]
        cell = next(value for value in plan["cells"] if value["cell_id"] == row["cell_id"])
        refs = [value for value in plan["references"] if value["reference_id"] == cell["reference_id"]]
        require(len(refs) == 1 and record["identity_sha256"] == plain_digest(
            {"binding": stage["binding"], "cell": cell, "reference": refs[0]}), "D1_COVERAGE_EXCLUDED_CELL_IDENTITY")
        require(record["output_root"] == str(queue_root / "jobs" / name / "outputs"), "D1_COVERAGE_OUTPUT_ROOT")
        for relative, expected in record["files"].items():
            require(io.file_record(Path(record["output_root"]) / io.relative(relative)) == expected,
                "D1_COVERAGE_RETAINED_FAILED_ARTIFACT_CHANGED")
        retained["asr_eligibility_replayed"] = excluded_native(row)
        records.append(retained)
        excluded.add(row["cell_id"])
    require(eligible | excluded == new_ids and index["new_generation_cell_count"] == len(new_ids)
        and index["asr_eligible_new_cell_count"] == len(eligible)
        and index["asr_excluded_new_cell_count"] == len(excluded), "D1_COVERAGE_EXPORT_PARTITION_INCOMPLETE")
    return records, evidence


def close_roster(plan, rows, excluded, previous_ids):
    """256 planned terminals = retained observable rows + explicit exclusions."""
    cells = {cell["cell_id"]: cell for cell in plan["cells"]}
    observed = {row["source_id"] for row in rows}
    omitted = {item["exported_row"]["cell_id"] for item in excluded}
    require(len(cells) == len(plan["cells"]) == 256 and len(observed) == len(rows)
        and len(omitted) == len(excluded) and not observed & omitted and observed | omitted == set(cells),
        "D1_COVERAGE_COMPLETE_TERMINAL_PARTITION")
    require(len(previous_ids) == len(set(previous_ids)) == 64 and set(previous_ids) <= observed
        and set(previous_ids) == {cell["cell_id"] for cell in plan["cells"][:64]},
        "D1_COVERAGE_PREVIOUS64_CANNOT_BE_EXCLUDED")
    for row in rows:
        cell = cells[row["source_id"]]
        require(row["source_family_id"] == cell["source_family_id"] and row["source_role"] == cell["role"]
            and row["split"] == ("train" if cell["role"] == "train-candidate" else "development"),
            "D1_COVERAGE_OBSERVED_SOURCE_ROLE")
    unobservable = [{"source_id": item["exported_row"]["cell_id"],
        "source_family_id": cells[item["exported_row"]["cell_id"]]["source_family_id"],
        "source_role": cells[item["exported_row"]["cell_id"]]["role"],
        "split": "train" if cells[item["exported_row"]["cell_id"]]["role"] == "train-candidate" else "development",
        "new_native_call": False, "observed_text": None, "training_admitted": False,
        "keyword_known": [False, False], "keyword_weight": [0, 0], **item} for item in excluded]
    return {"planned_cells": 256, "observable_cells": len(rows), "excluded_cells": len(excluded),
        "reused_native_cells": 64, "new_native_cells": len(rows) - 64,
        "observed_source_ids": sorted(observed), "excluded_source_ids": sorted(omitted),
        "unobservable_by_split": dict(Counter(row["split"] for row in unobservable)),
        "unobservable": unobservable, "complete_terminal_coverage": True}


def close_accounted_roster(plan, rows, excluded, unknown, previous_ids):
    """Close accounted cells without turning a consumed unknown into a return."""
    cells = {cell["cell_id"]: cell for cell in plan["cells"]}
    observed = {row["source_id"] for row in rows}
    omitted = {item["source_id"] for item in excluded}
    consumed = {item["source_id"] for item in unknown}
    require(len(cells) == len(plan["cells"]) == 256 and len(observed) == len(rows)
        and len(omitted) == len(excluded) and len(consumed) == len(unknown)
        and not observed & omitted and not observed & consumed and not omitted & consumed
        and observed | omitted | consumed == set(cells), "D1_ACCOUNTED_EXACT_DISJOINT_PARTITION")
    require(len(previous_ids) == len(set(previous_ids)) == 64 and set(previous_ids) <= observed
        and set(previous_ids) == {cell["cell_id"] for cell in plan["cells"][:64]},
        "D1_ACCOUNTED_PREVIOUS64_MUST_REMAIN_OBSERVABLE")
    for row in rows:
        cell = cells[row["source_id"]]
        require(row["source_family_id"] == cell["source_family_id"] and row["source_role"] == cell["role"]
            and row["split"] == ("train" if cell["role"] == "train-candidate" else "development"),
            "D1_ACCOUNTED_OBSERVED_SOURCE_ROLE")
    unobservable = []
    for values, status in ((excluded, "returned-unrecognizable"), (unknown, "unobservable-consumed")):
        for item in values:
            cell = cells[item["source_id"]]
            require(item["status"] == status and item["source_family_id"] == cell["source_family_id"]
                and item["source_role"] == cell["role"] and item["asr_calls"] == 0
                and item["observed_text"] is None and item["training_admitted"] is False
                and type(item["reason"]) is str and bool(item["reason"])
                and type(item["original_proof"]) is dict and bool(item["original_proof"]),
                "D1_ACCOUNTED_UNOBSERVABLE_RECORD")
            # The data verifier authenticates the complete original proof. These
            # are coverage entries only: no PCM, target or observation is made.
            unobservable.append({"source_id": item["source_id"], "source_family_id": cell["source_family_id"],
                "source_role": cell["role"],
                "split": "train" if cell["role"] == "train-candidate" else "development",
                "status": status, "reason": item["reason"], "new_native_call": False,
                "asr_calls": 0, "observed_text": None, "training_admitted": False,
                "keyword_known": [False, False], "keyword_weight": [0, 0],
                "original_proof": item["original_proof"]})
    return {"planned_cells": 256, "accounted_cells": len(observed | omitted | consumed),
        "observable_cells": len(rows), "returned_generation_cells": len(rows) + len(excluded),
        "excluded_returned_cells": len(excluded), "consumed_unknown_cells": len(unknown),
        "unobservable_cells": len(excluded) + len(unknown), "reused_native_cells": 64,
        "new_native_cells": len(rows) - 64, "observed_source_ids": sorted(observed),
        "excluded_source_ids": sorted(omitted), "consumed_unknown_source_ids": sorted(consumed),
        "unobservable_by_split": dict(Counter(row["split"] for row in unobservable)),
        "unobservable": unobservable, "accounted_complete": True,
        "complete_terminal_coverage": not unknown}


def bind_accounted(job, record, rows, previous_ids, loader):
    require(callable(getattr(loader, "verify_accounted_prior", None)), "D1_ACCOUNTED_RETAINED_VERIFIER_REQUIRED")
    binding = job["final_prior_evidence"]
    verified = loader.verify_accounted_prior(binding["path"], binding["sha256"], expected_count=256)
    require(type(verified) is dict and verified["schema"] == ACCOUNTED_VERIFICATION
        and verified["plan_sha256"] == job["plan_sha256"] and verified["accounted_prefix_count"] == 256
        and verified["pending_ids"] == [] and verified["accounted_complete"] is True
        and type(verified["ml_imports"]) is int and verified["ml_imports"] == 0
        and type(verified["new_model_calls"]) is int and verified["new_model_calls"] == 0,
        "D1_ACCOUNTED_REAL_CLOSED_PRIOR_REQUIRED")
    plan = verified["plan"]
    require(plain_digest(plan) == job["plan_sha256"] and sorted(cell["cell_id"] for cell in plan["cells"]) ==
        job["source_ids"], "D1_ACCOUNTED_ORIGINAL_PLAN_CHANGED")
    parents = [{key: value[key] for key in ("path", "manifest_sha256")} for value in record["aggregate"]["parents"]]
    require(verified["views"] == parents, "D1_ACCOUNTED_PREPARED_PARENTS_CHANGED")
    ids = verified["observed_ids"]
    require(type(ids) is list and len(ids) == len(set(ids)) == len(rows)
        and set(ids) == {row["source_id"] for row in rows}
        and ids == [cell["cell_id"] for cell in plan["cells"] if cell["cell_id"] in set(ids)],
        "D1_ACCOUNTED_OBSERVATION_ROSTER_CHANGED")
    excluded, unknown = verified["excluded_returned"], verified["unobservable_consumed"]
    require(type(excluded) is list and type(unknown) is list, "D1_ACCOUNTED_UNOBSERVABLE_LISTS")
    closed = close_accounted_roster(plan, rows, excluded, unknown, previous_ids)
    require(type(verified["complete_terminal_coverage"]) is bool
        and verified["complete_terminal_coverage"] == closed["complete_terminal_coverage"],
        "D1_ACCOUNTED_UNKNOWN_IS_NOT_GENERATION_TERMINAL")
    evidence, by_path = [pin(binding["path"])], {}
    require(type(verified["evidence_files"]) is list and verified["evidence_files"],
        "D1_ACCOUNTED_ORIGINAL_EVIDENCE_REQUIRED")
    for entry in evidence + verified["evidence_files"]:
        require(type(entry) is dict and set(entry) == {"path", "bytes", "sha256"}
            and type(entry["path"]) is str and Path(entry["path"]).is_absolute(), "D1_ACCOUNTED_EVIDENCE_PIN")
        actual = pin(entry["path"])
        require(actual == entry and (entry["path"] not in by_path or by_path[entry["path"]] == entry),
            "D1_ACCOUNTED_ORIGINAL_EVIDENCE_CHANGED")
        by_path[entry["path"]] = entry
    return {"schema": ACCOUNTED_SCHEMA, "final_prior_evidence": binding,
        "plan_sha256": job["plan_sha256"], "views": parents, "evidence_files": list(by_path.values()),
        "source_verifier_schema": verified["schema"], "ml_imports": 0, "new_model_calls": 0, **closed}


def bind(job, record, rows, previous_ids, queue_reader, *, loader=None):
    spec = pinned(job["final_prior_evidence"])
    if type(spec) is dict and spec.get("schema") == ACCOUNTED_PRIOR:
        return bind_accounted(job, record, rows, previous_ids, loader)
    require(type(spec) is dict and set(spec) == {"schema", "plan_sha256", "prefix_count", "views", "excluded_exports"}
        and spec["schema"] == "d1-sealed-prior-evidence-v1" and spec["prefix_count"] == 256
        and spec["plan_sha256"] == job["plan_sha256"], "D1_COVERAGE_FINAL_CLOSED_PRIOR")
    require(spec["views"] == [{key: value[key] for key in ("path", "manifest_sha256")}
        for value in record["aggregate"]["parents"]], "D1_COVERAGE_PREPARED_PARENTS_CHANGED")
    first = spec["views"][0]
    manifest = pinned({"path": str(Path(first["path"]) / "MANIFEST.json"), "sha256": first["manifest_sha256"]})
    plan_path = Path(first["path"]) / "evidence/source/d1/plan.json"
    plan_sha = manifest["evidence/source/d1/plan.json"]
    plan = pinned({"path": str(plan_path), "sha256": plan_sha})
    require(plain_digest(plan) == job["plan_sha256"] and sorted(cell["cell_id"] for cell in plan["cells"]) ==
        job["source_ids"], "D1_COVERAGE_ORIGINAL_PLAN_CHANGED")
    excluded, evidence = [], [pin(job["final_prior_evidence"]["path"]), pin(plan_path)]
    seen_exports = set()
    for entry in spec["excluded_exports"]:
        require(entry["sha256"] not in seen_exports, "D1_COVERAGE_DUPLICATE_EXCLUDED_EXPORT")
        seen_exports.add(entry["sha256"])
        values, pins = excluded_export(entry, plan, job["plan_sha256"], plan_sha, queue_reader)
        excluded.extend(values)
        evidence.extend(pins)
    return {"schema": SCHEMA, "final_prior_evidence": job["final_prior_evidence"],
        "plan_sha256": job["plan_sha256"], "views": spec["views"], "evidence_files": evidence,
        **close_roster(plan, rows, excluded, previous_ids)}
