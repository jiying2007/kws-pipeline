#!/usr/bin/env python3
"""Prepare exact controlled Chinese input cohorts without model/audio execution.

The existing owner-held preparation and independent R1 audit are inputs, not
new labels or a new permission grant.  This module never chooses a training
budget, samples rows, loads weights, decodes audio, or contacts an executor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

from research_d1_recipe import row_identity
from research_d1_vocab import digest, fulltext_targets, require

SCHEMA = "kws-controlled-chinese-input-plan-v1"
DIMENSIONS = (
    "source_id", "source_family_id", "speaker_id", "reference_id",
    "reference_wav_sha256", "reference_pcm_sha256", "parent_source_sha256",
    "original_pcm_sha256", "pcm_sha256",
)
KEYWORDS = ("你好小窝", "小窝小窝")
TABLES = {
    "increment_train_candidate": ("train_non_near_candidates_140.jsonl",
        "weak_research_candidate_pending_execution_route", "train"),
    "increment_development": ("development_57.jsonl", "development_no_training", "development"),
    "increment_nearphone_quarantine": ("nearphone_quarantine_14.jsonl", "nearphone_quarantine", "train"),
    "increment_unknown_zero_loss": ("unknown_zero_loss_44.jsonl", "unknown_zero_loss", None),
}
AUDITED_ROLES = {
    "increment_train_candidate": "train_non_near_140",
    "increment_development": "development_57",
    "increment_nearphone_quarantine": "nearphone_quarantine_14",
}
SOURCE_FILES = (
    "training/research_controlled_chinese_plan.py",
    "training/research_d1_recipe.py", "training/research_d1_vocab.py",
)
LABEL_FIELDS = (
    "full_text", "target_ids", "ctc_mask", "ctc_weight",
    "keyword_targets", "keyword_mask", "keyword_weight",
)
MAX_METADATA_BYTES = 128 * 1024 * 1024


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hash_string(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def safe_member(value):
    require(type(value) is str and bool(value) and "\\" not in value and "\x00" not in value,
            "invalid relative member")
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and
            all(part not in ("", ".", "..") for part in path.parts), "unsafe relative member")
    return value


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)

    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)


def file_bytes(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and
            not any(p.is_symlink() for p in path.parents), "regular nonsymlink input required")
    require(path.stat().st_size <= MAX_METADATA_BYTES, "metadata input too large")
    return path.read_bytes()


def identity(raw):
    return {"bytes": len(raw), "sha256": sha(raw)}


def matches(raw, record):
    require(type(record) is dict and type(record.get("bytes")) is int and
            record["bytes"] >= 0 and hash_string(record.get("sha256")), "invalid file identity")
    require(identity(raw) == {k: record[k] for k in ("bytes", "sha256")}, "file identity mismatch")


def indexed_archive(path, expected_sha256):
    raw = file_bytes(path)
    require(hash_string(expected_sha256) and sha(raw) == expected_sha256, "audit archive hash changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        require(len(names) == len(set(names)) and "FILE_MANIFEST.json" in names, "audit ZIP member set")
        require(sum(item.file_size for item in infos) <= MAX_METADATA_BYTES, "expanded metadata too large")
        for item in infos:
            safe_member(item.filename)
            require(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16), "nonregular ZIP member")
        bodies = {name: archive.read(name) for name in names}
    manifest = strict_json(bodies["FILE_MANIFEST.json"])
    require(manifest.get("schema") == "kws-v4-chinese-increment-metadata-manifest-v1" and
            type(manifest.get("files")) is dict, "audit manifest schema")
    require(set(bodies) == set(manifest["files"]) | {"FILE_MANIFEST.json"}, "audit ZIP is not closed")
    for name, record in manifest["files"].items():
        matches(bodies[name], record)
    return bodies, identity(raw)


def saved_isolation(directory, expected_manifest_sha256):
    root = Path(directory)
    raw = file_bytes(root / "FILE_MANIFEST.json")
    require(hash_string(expected_manifest_sha256) and sha(raw) == expected_manifest_sha256,
            "saved isolation manifest changed")
    manifest = strict_json(raw)
    require(manifest.get("schema") == "private-independent-audit-evidence-v1" and
            type(manifest.get("files")) is list, "saved isolation manifest schema")
    names = [safe_member(record["name"]) for record in manifest["files"]]
    require(len(names) == len(set(names)), "duplicate isolation member")
    bodies = {}
    for record in manifest["files"]:
        body = file_bytes(root / record["name"])
        matches(body, record)
        bodies[record["name"]] = body
    required = {"independent_overlap_audit.json", "current_identity_rows.json",
                "r1_complete_identity_rows.json", "r1_actual_binding_checks.json", "verified_input_members.json"}
    require(required <= set(bodies), "missing saved isolation evidence")
    values = {name: strict_json(bodies[name]) for name in required}
    return values, {"manifest": identity(raw), "members": {name: identity(bodies[name]) for name in sorted(required)}}


def rows(raw):
    result = [strict_json(line) for line in raw.splitlines() if line.strip()]
    require(all(type(row) is dict for row in result), "row object required")
    return result


def index_by(values, key):
    require(type(values) is list and all(type(row) is dict and type(row.get(key)) is str and row[key]
                                      for row in values), "invalid indexed rows")
    result = {row[key]: row for row in values}
    require(len(result) == len(values), "duplicate indexed identity")
    return result


def validate_labels(row, vocabulary):
    require(type(row.get("samples")) is int and row["samples"] > 0, "invalid sample count")
    require(type(row.get("ctc_mask")) is bool and type(row.get("ctc_weight")) in (int, float) and
            math.isfinite(row["ctc_weight"]) and 0 <= row["ctc_weight"] <= 1, "invalid CTC mask or weight")
    require(row.get("vocabulary_sha256") == digest(vocabulary), "row vocabulary changed")
    for field in ("keyword_targets", "keyword_mask", "keyword_weight"):
        require(type(row.get(field)) is list and len(row[field]) == len(KEYWORDS), "invalid keyword shape")
    active = row["ctc_mask"]
    minimum = None
    frames = max(0, 1 + (row["samples"] - 400) // 160) // 3
    if active:
        projection = fulltext_targets(row["full_text"], vocabulary)
        require(projection["supported"] and row["target_ids"] == projection["target_ids"] and
                type(row["target_ids"]) is list and all(type(item) is int for item in row["target_ids"]) and
                row["ctc_weight"] > 0, "confirmed full-text CTC changed")
        minimum = len(row["target_ids"]) + sum(a == b for a, b in zip(row["target_ids"], row["target_ids"][1:]))
        require(minimum <= frames, "R1 native CTC geometry infeasible")
    else:
        require(row["ctc_weight"] == 0 and row["target_ids"] is None, "unknown CTC must have zero loss")
    for word, target, mask, weight in zip(KEYWORDS, row["keyword_targets"], row["keyword_mask"], row["keyword_weight"]):
        require(type(mask) is bool and type(weight) in (int, float) and math.isfinite(weight) and 0 <= weight <= 1,
                "invalid keyword mask or weight")
        if mask:
            require(active and type(target) is int and target == int(word in row["full_text"]) and weight > 0,
                    "keyword target differs from observed text")
        else:
            require(target is None and weight == 0, "unknown is not an absent keyword")
    require((row["keyword_mask"] == [True, True]) if active else (row["keyword_mask"] == [False, False]),
            "full-text label mask changed")
    for key, value in (("minimum_ctc_frames", minimum), ("r1_output_frames", frames)):
        if key in row:
            require(row[key] == value, "saved CTC geometry changed")
    return {key: row[key] for key in LABEL_FIELDS}, {"minimum_ctc_frames": minimum, "r1_output_frames": frames}


def validate_identity(row):
    require(all(type(row.get(key)) is str and bool(row[key]) for key in DIMENSIONS), "missing source identity")
    require(all(hash_string(row[key]) for key in DIMENSIONS if key.endswith("_sha256")), "invalid source hash")
    require(row.get("split") in ("train", "development") and
            row.get("source_role") == ("train-candidate" if row["split"] == "train" else "development-only"),
            "original split or source role changed")


def role_scope(group, row):
    candidate = group in ("increment_train_candidate", "r1_train_reference")
    development = group in ("increment_development", "r1_development_reference")
    require(not candidate or row["split"] == "train" and row["ctc_mask"],
            "unknown or development row entered train candidate pool")
    require(group != "increment_train_candidate" or row["generation_category"] != "near-negative",
            "new quarantined row entered train candidate pool")
    return {
        "group": group, "original_split": row["split"], "original_source_role": row["source_role"],
        "train_pool_candidate": candidate, "development_only": development,
        "nearphone_quarantined": group == "increment_nearphone_quarantine",
        "unknown_zero_loss": group in ("increment_unknown_zero_loss", "r1_unknown_zero_loss"),
        "candidate_ctc_mask": candidate and row["ctc_mask"],
        "candidate_ctc_weight": row["ctc_weight"] if candidate else 0.0,
        "optimizer_authorized": False, "effective_optimizer_mask": False, "effective_optimizer_weight": 0.0,
        "keyword_loss_authorized": False,
    }


def enforce_separation(entries):
    for field in ("source_id", "pcm_sha256", "original_pcm_sha256"):
        values = [entry["source_identity"][field] for entry in entries]
        require(len(values) == len(set(values)), "duplicate source or PCM in combined input cohorts")
    for field in DIMENSIONS:
        roles = {}
        for entry in entries:
            value = entry["source_identity"][field]
            role = entry["scope"]["original_split"]
            require(value not in roles or roles[value] == role, "source alias crosses train/development")
            roles[value] = role


def source_inventory():
    root = Path(__file__).resolve().parents[1]
    return {name: identity(file_bytes(root / name)) for name in SOURCE_FILES}


def original_locator(locator, package, archives, manifest):
    require(type(locator) is dict and set(locator) == {"archive", "member", "sha256"}, "audio locator schema")
    require(locator["archive"] in package["archives"] and locator["archive"] in archives,
            "audio archive outside original package")
    safe_member(locator["member"])
    prefix = package["source_archive_member_prefix"]
    require(type(prefix) is str and prefix.endswith("/") and locator["member"].startswith(prefix),
            "audio member outside original package")
    relative = locator["member"][len(prefix):]
    safe_member(relative)
    require(manifest.get(relative) == locator["sha256"] and hash_string(locator["sha256"]),
            "audio locator differs from original manifest")
    record = archives[locator["archive"]]
    return {"archive": record["name"], "archive_bytes": record["bytes"], "archive_sha256": record["sha256"],
            "member": locator["member"], "member_sha256": locator["sha256"],
            "bytes_verified_during_this_preparation": False}


def prepare_increment(bodies, vocabulary, summary, isolation):
    archives = index_by(strict_json(bodies["source_archives.json"]), "name")
    packages = index_by(strict_json(bodies["source_packages.json"]), "id")
    for archive in archives.values():
        safe_member(archive["name"])
        require(type(archive["bytes"]) is int and archive["bytes"] > 0 and hash_string(archive["sha256"]),
                "source archive identity invalid")
    manifests = {}
    for pid, package in packages.items():
        name = safe_member(package["original_manifest_saved_as"])
        raw = bodies[name]
        require(sha(raw) == package["manifest_sha256"], "original package manifest changed")
        manifest = strict_json(raw)
        require(type(manifest) is dict and manifest.get("report.json") == package["report_sha256"],
                "package report binding changed")
        require(type(package["archives"]) is list and set(package["archives"]) <= set(archives), "package archive set changed")
        manifests[pid] = manifest
    audited = index_by(isolation["current_identity_rows.json"], "source_id")
    entries, counts, audited_seen = [], {}, set()
    for group, (member, usage, split) in TABLES.items():
        group_rows = rows(bodies[member])
        counts[group] = len(group_rows)
        require(summary["prepared_usage_counts"].get(usage) == len(group_rows), "prepared group count changed")
        for ordinal, row in enumerate(group_rows):
            validate_identity(row)
            require(row.get("schema") == "kws-v4-restored-existing-input-row-v1" and row.get("prepared_usage") == usage and
                    (split is None or row["split"] == split), "prepared row scope changed")
            require(row.get("generation_category") in ("full-positive", "ordinary-negative", "partial-negative", "near-negative") and
                    row.get("label_status") == ("machine_candidate" if row["ctc_mask"] else "uncertain"),
                    "prepared label status or category changed")
            for field in ("formal_admission", "human_gold", "independent_test", "new_admission_granted",
                          "optimizer_loss_authorized_by_this_preparation", "public_executor_ready"):
                require(row.get(field) is False, "prepared metadata cannot grant new permission")
            require(row.get("globally_unused") is False and type(row.get("previously_used_in_historical_head_experiments")) is bool,
                    "source reuse history missing")
            require(row.get("wav_pcm_source_labels_verified") is True and
                    row.get("original_projection_sha256") == summary["source_projection_sha256"] and
                    hash_string(row.get("original_projection_row_canonical_sha256")), "original projection binding changed")
            require((group == "increment_unknown_zero_loss") == (not row["ctc_mask"]), "unknown label role changed")
            require(row.get("included_in_non_near_candidate_set") is (group == "increment_train_candidate"),
                    "non-near candidate membership changed")
            if group == "increment_nearphone_quarantine":
                require(row["generation_category"] == "near-negative", "nearphone quarantine changed")
            labels, geometry = validate_labels(row, vocabulary)
            if group in AUDITED_ROLES:
                expected = {"current_role": AUDITED_ROLES[group], **{k: row[k] for k in DIMENSIONS}}
                require(audited.get(row["source_id"]) == expected, "row differs from saved independent isolation population")
                audited_seen.add(row["source_id"])
            package = packages[row["source_package_id"]]
            require(row["parent_manifest_sha256"] == package["manifest_sha256"] and
                    row["parent_report_sha256"] == package["report_sha256"], "row parent package differs")
            locator = {name: original_locator(row[name], package, archives, manifests[package["id"]])
                       for name in ("normalized_audio_locator", "original_audio_locator")}
            entries.append({
                "key": "increment/" + row["source_id"], "source_identity": {k: row[k] for k in DIMENSIONS},
                "scope": role_scope(group, row), "labels": labels, "geometry": geometry,
                "audio": locator, "source_observation": row,
                "binding": {"audit_member": member, "row_ordinal": ordinal,
                    "audit_member_sha256": sha(bodies[member]), "prepared_row_canonical_sha256": digest(row),
                    "original_projection_row_canonical_sha256": row["original_projection_row_canonical_sha256"]},
            })
    require(audited_seen == set(audited), "saved isolation cohort is not fully consumed")
    require(sum(counts.values()) == summary["actual_observed_16k_wav_pcm_verified"] ==
            summary["actual_observed_original_wav_pcm_verified"], "observed audio row count changed")
    require(len(archives) == summary["source_archives"] and len(packages) == summary["restored_native_packages"],
            "source container count changed")
    return entries, counts, {name: {k: row[k] for k in ("name", "bytes", "sha256")} for name, row in archives.items()}


def prepare_r1(recipe, capsule_manifest, references, isolation, vocabulary):
    require(recipe.get("human_gold") is False and recipe.get("independent_test") is False and
            recipe.get("generation_intent_is_supervision") is False and recipe.get("development_optimizer_draws") == 0,
            "retained R1 source recipe scope changed")
    require(recipe.get("vocabulary_sha256") == digest(vocabulary), "R1 and increment vocabularies differ")
    require(capsule_manifest.get("schema") == "kws-d2-durable-checkpoint-v1", "R1 capsule manifest schema")
    files_by_path = {}
    for name, record in capsule_manifest["files"].items():
        safe_member(name)
        require(type(record.get("path")) is str and record["path"].startswith("/") and
                record["path"] not in files_by_path, "capsule original path is ambiguous")
        require(type(record.get("bytes")) is int and record["bytes"] >= 0 and hash_string(record.get("sha256")),
                "capsule file identity invalid")
        files_by_path[record["path"]] = (name, record)
    train = index_by(recipe["train"], "source_id")
    development = index_by(recipe["development"], "source_id")
    evaluation = index_by(recipe["evaluation_rows"], "source_id")
    unknown = recipe["new_zero_loss_rows"]
    require(type(unknown) is list and len(unknown) == len(set(unknown)) and
            all(type(value) is str and value for value in unknown), "R1 zero-loss roster invalid")
    require(not set(train) & set(development) and not set(unknown) & (set(train) | set(development)) and
            set(evaluation) == set(train) | set(development) | set(unknown), "R1 original roles changed")
    for key, cohort in (("train_binding_sha256", recipe["train"]),
                        ("development_binding_sha256", recipe["development"]),
                        ("evaluation_binding_sha256", recipe["evaluation_rows"])):
        require(recipe.get(key) == digest([row_identity(row) for row in cohort]), "R1 original cohort binding changed")
    require([row["source_id"] for row in recipe["evaluation_rows"][:len(train)]] == list(train),
            "R1 original evaluation order changed")
    observed = isolation["r1_complete_identity_rows.json"]
    checks = isolation["r1_actual_binding_checks.json"]
    require([row["source_id"] for row in recipe["evaluation_rows"]] == [row["source_id"] for row in observed] ==
            [row["source_id"] for row in checks], "R1 actual binding order changed")
    report = isolation["independent_overlap_audit.json"]
    capsule = report["inputs"]["r1_recipe_capsule"]
    owner = report["inputs"]["r1_model_owner_archive"]
    capref = references["first64_original_evidence_and_r1_source_recipe_archive"]
    modelref = references["r1_model_owner_archive"]
    require(capsule_manifest["files"][capref["r1_recipe_member"]]["path"] == capref["r1_recipe_original_path"] and
            report["recipe_member_verified"]["member"] == capref["r1_recipe_member"] and
            isolation["verified_input_members.json"]["recipe"] == report["recipe_member_verified"],
            "R1 recipe member/path provenance changed")
    require(capsule["sha256"] == capref["sha256"] and owner["sha256"] == modelref["sha256"] and
            owner["bytes"] == modelref["bytes"], "R1 archive provenance changed")
    require(report["r1_actual_feature_bindings_sha256"] == modelref["actual_feature_bindings_sha256"],
            "R1 actual feature binding file changed")
    members = index_by(report["r1_selected_members_verified"], "member")
    model_members = [item for name, item in members.items() if name.endswith("/model.pt")]
    plan_members = [item for name, item in members.items() if name.endswith("/training/plan.json")]
    require(len(model_members) == len(plan_members) == 1 and
            model_members[0]["sha256"] == report["r1_checkpoint_sha256"] and
            plan_members[0]["sha256"] == report["r1_plan_sha256"], "actual R1 checkpoint/plan binding changed")
    entries = []
    for ordinal, (row, old_identity, check) in enumerate(zip(recipe["evaluation_rows"], observed, checks)):
        validate_identity(row)
        sid = row["source_id"]
        role = "train" if sid in train else "development" if sid in development else "unknown_zero_loss"
        group = {"train": "r1_train_reference", "development": "r1_development_reference",
                 "unknown_zero_loss": "r1_unknown_zero_loss"}[role]
        require(old_identity == {"r1_effective_role": role, **{key: row[key] for key in DIMENSIONS}},
                "R1 row differs from independently saved identity")
        label_sha = row_identity(row)
        require(check["r1_effective_role"] == role and check["split"] == row["split"] and
                check["pcm_sha256"] == row["pcm_sha256"] and check["match"] is True and
                check["computed_label_binding_sha256"] == check["saved_label_binding_sha256"] == label_sha,
                "R1 actual label/PCM binding changed")
        if sid in train or sid in development:
            original = train[sid] if sid in train else development[sid]
            require(row_identity(original) == label_sha and row["ctc_mask"] is True and
                    row["split"] == role, "R1 retained eligible role changed")
        else:
            require(row["ctc_mask"] is False and row["ctc_weight"] == 0, "R1 unknown row acquired loss")
        require(row.get("human_gold") is False and row.get("independent_test") is False,
                "R1 research labels promoted")
        labels, geometry = validate_labels(row, vocabulary)
        audio = recipe["audio_files"][sid]
        member, record = files_by_path[audio["path"]]
        require({k: audio[k] for k in ("bytes", "sha256")} == {k: record[k] for k in ("bytes", "sha256")} and
                audio["sha256"] == row["file_sha256"], "R1 original audio path/hash changed")
        entries.append({
            "key": "r1/" + sid, "source_identity": {key: row[key] for key in DIMENSIONS},
            "scope": role_scope(group, row), "labels": labels, "geometry": geometry,
            "audio": {"normalized_audio_locator": {"archive": capref["name"], "archive_bytes": capsule["bytes"],
                "archive_sha256": capsule["sha256"], "member": member, "member_bytes": record["bytes"],
                "member_sha256": record["sha256"], "original_recipe_audio_path": audio["path"],
                "bytes_verified_during_this_preparation": False}},
            "source_observation": row,
            "binding": {"recipe_evaluation_ordinal": ordinal, "original_label_binding_sha256": label_sha,
                "actual_binding": check},
        })
    counts = {"train": len(train), "development": len(development), "unknown_zero_loss": len(unknown)}
    require(report["r1_binding_checks"]["rows"] == len(entries) and
            report["r1_binding_checks"]["effective_role_counts"] == counts, "saved R1 cohort counts changed")
    excluded = recipe.get("excluded_original_rows", [])
    require(type(excluded) is list and not {row["source_id"] for row in excluded} & set(evaluation),
            "historically excluded source row entered R1 cohort")
    return entries, {
        "recipe_archive": {"name": capref["name"], "bytes": capsule["bytes"], "sha256": capsule["sha256"]},
        "recipe_member": capref["r1_recipe_member"], "recipe_sha256": capref["r1_recipe_sha256"],
        "historical_source_recipe_schema": recipe["schema"],
        "historical_recipe_execution_parameters_inherited": False,
        "retained_checkpoint_archive": {"name": modelref["name"], "bytes": owner["bytes"], "sha256": owner["sha256"]},
        "retained_checkpoint_member": model_members[0], "retained_plan_member": plan_members[0],
        "actual_feature_bindings_sha256": report["r1_actual_feature_bindings_sha256"],
        "original_role_counts": counts,
        "historically_excluded_rows": [{"source_id": row["source_id"], "label_binding_sha256": row_identity(row)}
                                       for row in excluded],
        "new_replay_or_checkpoint_load_authorized": False,
    }


def build_plan(audit, audit_sha256, isolation_dir, isolation_manifest_sha256, r1_recipe, capsule_manifest):
    bodies, audit_identity = indexed_archive(audit, audit_sha256)
    isolation, isolation_identity = saved_isolation(isolation_dir, isolation_manifest_sha256)
    summary = strict_json(bodies["summary.json"])
    require(summary.get("schema") == "kws-v4-chinese-existing-input-preparation-v1" and summary.get("completed") is True,
            "prepared input summary incomplete")
    for field in ("formal_admission", "human_gold", "independent_test", "new_admission_granted",
                  "public_audio_upload_allowed", "public_executor_ready"):
        require(summary.get(field) is False, "input summary scope changed")
    report = isolation["independent_overlap_audit.json"]
    require(report.get("result") == "PASS_NO_IDENTITY_OVERLAP" and
            report.get("inherited_overlap_flags_used_as_evidence") is False and
            report.get("archive_identity_and_bytes_verified") is True and
            report["inputs"]["current_audit"]["sha256"] == audit_sha256 and
            report["inputs"]["current_audit"]["bytes"] == audit_identity["bytes"], "independent isolation evidence differs")
    require(report.get("identity_dimensions") == list(DIMENSIONS) and report.get("nonzero_field_intersections") == 0,
            "independent source isolation did not pass")
    require(report.get("comparison_groups") == len(AUDITED_ROLES) * 2 and
            report.get("field_comparisons") == len(AUDITED_ROLES) * 2 * len(DIMENSIONS), "saved isolation scope changed")
    vocabulary_raw = bodies["original_frozen_vocabulary.json"]
    vocabulary = strict_json(vocabulary_raw)
    require(sha(vocabulary_raw) == summary["frozen_vocabulary_file_sha256"] and
            vocabulary.get("output_count") == 2601 and vocabulary.get("blank_id") == 0 and
            vocabulary.get("observed_development_text_used") is False, "frozen vocabulary changed")
    references = strict_json(bodies["r1_and_first64_source_references.json"])
    recipe_raw, manifest_raw = file_bytes(r1_recipe), file_bytes(capsule_manifest)
    ref = references["first64_original_evidence_and_r1_source_recipe_archive"]
    require(sha(recipe_raw) == ref["r1_recipe_sha256"] == summary["r1_recipe_sha256"] and
            sha(manifest_raw) == ref["manifest_sha256"] == report["capsule_manifest_sha256"], "R1 recipe source changed")
    matches(recipe_raw, report["recipe_member_verified"])
    capsule = strict_json(manifest_raw)
    matches(recipe_raw, capsule["files"][ref["r1_recipe_member"]])
    increment, counts, archives = prepare_increment(bodies, vocabulary, summary, isolation)
    old, r1_binding = prepare_r1(strict_json(recipe_raw), capsule, references, isolation, vocabulary)
    all_rows = increment + old
    enforce_separation(all_rows)
    no_audio = strict_json(bodies["consumed_unknown_no_retry.json"])
    require(type(no_audio) is list and len(no_audio) == summary["consumed_unknown_no_audio_no_retry"], "no-audio denominator changed")
    index_by(no_audio, "source_id")
    for row in no_audio:
        require(row.get("has_audio") is False and row.get("may_retry") is False and row.get("ctc_mask") is False and
                row.get("ctc_weight") == 0 and row.get("training_admitted") is False and
                row.get("observed_text") is None, "consumed no-audio terminal changed")
        require(row["source_id"] not in {entry["source_identity"]["source_id"] for entry in all_rows}, "no-audio row entered cohort")
    cohorts = {group: [entry["key"] for entry in all_rows if entry["scope"]["group"] == group]
               for group in (*TABLES, "r1_train_reference", "r1_development_reference", "r1_unknown_zero_loss")}
    return {
        "schema": SCHEMA, "status": "input_projection_complete_execution_not_authorized",
        "source_code": source_inventory(),
        "inputs": {"increment_audit_zip": audit_identity, "saved_isolation": isolation_identity,
            "r1_recipe": identity(recipe_raw), "r1_capsule_manifest": identity(manifest_raw)},
        "vocabulary": {"file": identity(vocabulary_raw), "canonical_sha256": digest(vocabulary),
            "output_count": 2601, "blank_id": 0, "extended_during_this_preparation": False},
        "r1_binding": r1_binding, "source_audio_archives": archives,
        "saved_isolation_scope": {"current_known_rows": len(isolation["current_identity_rows.json"]),
            "current_unknown_rows_outside_saved_audit": counts["increment_unknown_zero_loss"],
            "field_comparisons": report["field_comparisons"], "nonzero_field_intersections": 0,
            "auditor_reexecuted": False},
        "combined_role_validation": {"rows": len(all_rows), "identity_dimensions": list(DIMENSIONS),
            "cross_train_development_aliases": 0, "source_ids_and_original_normalized_pcm_unique": True,
            "acoustic_independence_claimed": False},
        "cohorts": cohorts, "counts": {name: len(keys) for name, keys in cohorts.items()},
        "rows": all_rows, "consumed_no_audio_no_retry": no_audio,
        "execution": {"optimizer_authorized": False, "optimizer_steps": None, "batch_size": None,
            "seed": None, "sampling_schedule": None, "replay_selection": None,
            "public_audio_upload_allowed": False, "public_executor_ready": False,
            "requires_separate_finite_recipe": True, "requires_permitted_controlled_executor": True,
            "requires_original_audio_member_and_pcm_verification": True},
        "scope": {"formal_admission": False, "human_gold": False, "independent_test": False,
            "new_labels": False, "new_vocabulary": False, "audio_decodes": 0, "model_calls": 0,
            "source_audio_bytes_reverified_this_preparation": False,
            "isolation_auditor_reexecuted": False, "identity_nonoverlap_implies_acoustic_independence": False,
            "historical_243_row_plan_reused": False, "historical_128_step_budget_reused": False},
    }


def summarize(plan):
    """A text-free aggregate suitable for review; the plan remains private."""
    require(plan.get("schema") == SCHEMA, "input plan schema")
    groups = {}
    for name in plan["cohorts"]:
        selected = [row for row in plan["rows"] if row["scope"]["group"] == name]
        groups[name] = {
            "rows": len(selected), "samples": sum(row["source_observation"]["samples"] for row in selected),
            "observed_ctc_weight_counts": dict(sorted(Counter(str(row["labels"]["ctc_weight"]) for row in selected).items())),
            "generation_category_counts": dict(sorted(Counter(row["source_observation"]["generation_category"] for row in selected).items())),
            "candidate_ctc_rows": sum(row["scope"]["candidate_ctc_mask"] for row in selected),
            "effective_optimizer_rows": sum(row["scope"]["effective_optimizer_mask"] for row in selected),
        }
    return {"schema": "kws-controlled-chinese-input-plan-summary-v1", "plan_canonical_sha256": digest(plan),
            "status": plan["status"], "groups": groups, "saved_isolation_scope": plan["saved_isolation_scope"],
            "combined_role_validation": plan["combined_role_validation"],
            "consumed_no_audio_not_in_row_total": len(plan["consumed_no_audio_no_retry"]),
            "execution": plan["execution"], "scope": plan["scope"]}


def write_new_json(path, value, *, protected_roots):
    path = Path(path).absolute()
    require(".." not in path.parts and path.parent.is_dir() and
            not any(parent.is_symlink() for parent in (path, *path.parents)), "unsafe private output path")
    roots = [Path(__file__).resolve().parents[1], *(Path(root).resolve() for root in protected_roots)]
    require(not any(path.is_relative_to(root) for root in roots),
            "private output must be outside repository and input evidence directories")
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
    return identity(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--isolation-dir", type=Path, required=True)
    parser.add_argument("--isolation-manifest-sha256", required=True)
    parser.add_argument("--r1-recipe", type=Path, required=True)
    parser.add_argument("--capsule-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = build_plan(args.audit, args.audit_sha256, args.isolation_dir, args.isolation_manifest_sha256,
                      args.r1_recipe, args.capsule_manifest)
    if args.mode == "prepare":
        result = write_new_json(args.plan, plan, protected_roots=(args.audit.parent, args.isolation_dir,
            args.r1_recipe.parent, args.capsule_manifest.parent))
    else:
        raw = file_bytes(args.plan)
        require(digest(strict_json(raw)) == digest(plan), "saved projection differs from exact original inputs")
        result = identity(raw)
    print(json.dumps({"status": "prepared" if args.mode == "prepare" else "verified", "plan": result,
        "counts": plan["counts"], "optimizer_authorized": False, "audio_decodes": 0, "model_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
