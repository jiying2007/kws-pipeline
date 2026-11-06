#!/usr/bin/env python3
"""Replay explicitly pinned r1/r2 context packages into the fixed full2601 map.

Each package is replayed in a fresh process. The parent imports no data reader,
so a caller's frozen generation modules cannot be replaced or reused implicitly.
Original r1 projection rows keep their original schema and values. The additive
metadata projection has its own schema and retains both native container hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import research_context_v3_inputs as original
import research_d1_vocab as vocabulary_contract
from research_d1_vocab import canonical, digest, fulltext_targets, require

SCHEMA = "kws-context-v3-handoff-verified-aggregate-v1"
CONSUMER = "kws-context-v3-handoff-fulltext-projection-v1"
METADATA_AGGREGATE_SCHEMA = "kws-context-v3-metadata-verified-aggregate-v1"
METADATA_ROW_SCHEMA = "kws-context-v3-metadata-fulltext-projection-v1"
ROW_SCHEMAS = (original.ROW_SCHEMA, METADATA_ROW_SCHEMA)
CONTAINER_POLICY = "kws-native-wav-peak-timestamp-plus-one-second-v1"
KEYWORDS = original.KEYWORDS
SHA = re.compile(r"[0-9a-f]{64}\Z")
PIPELINE_FILES = ("training/research_context_v3_handoff_inputs.py",
                  "training/research_context_v3_inputs.py", "training/research_d1_vocab.py")
READERS = {
    "context-v3-r1": {
        "files": original.DATA_FILES,
        "schema": "kws-active-context-v3-research-weak-view-v1",
        "adapter": "kws-active-context-v3-native-source-v1"},
    "context-v3-metadata-r2": {
        "files": ("automation/active_context_v3_metadata_weak_view.py", *original.DATA_FILES),
        "schema": "kws-active-context-v3-metadata-research-weak-view-v1",
        "adapter": "kws-active-context-v3-metadata-native-source-v1"},
}
ROLE_FIELDS = ("source_family_id", "speaker_id", "pcm_sha256", "wav_sha256", "parent_source_sha256")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def absolute(value):
    path = Path(value).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            "HANDOFF_PATH_ALIAS")
    return path


def read_bytes(path, maximum=64 * 1024 * 1024):
    path = absolute(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and 0 < info.st_size <= maximum,
                "HANDOFF_REGULAR_BOUNDED_FILE")
        raw = stream.read(maximum + 1)
    require(len(raw) == info.st_size, "HANDOFF_FILE_CHANGED_DURING_READ")
    return raw


def loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "HANDOFF_DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("HANDOFF_NONFINITE_JSON:" + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def write_new(path, value):
    path = absolute(path)
    require(path.parent.is_dir(), "HANDOFF_OUTPUT_PARENT_REQUIRED")
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_identity():
    result = {name: sha(read_bytes(ROOT / name)) for name in PIPELINE_FILES}
    require(sha(read_bytes(original.__file__)) == result["training/research_context_v3_inputs.py"] and
            sha(read_bytes(vocabulary_contract.__file__)) == result["training/research_d1_vocab.py"],
            "HANDOFF_MIXED_PIPELINE_MODULES")
    return result


def validate_declaration(value):
    require(type(value) is dict and set(value) == {"path", "manifest_sha256", "reader"} and
            type(value["path"]) is str and Path(value["path"]).is_absolute() and
            type(value["manifest_sha256"]) is str and SHA.fullmatch(value["manifest_sha256"]),
            "HANDOFF_PACKAGE_DECLARATION")
    package = absolute(value["path"])
    require(str(package) == value["path"] and package.is_dir(), "HANDOFF_PACKAGE_PATH")
    reader = value["reader"]
    require(type(reader) is dict and set(reader) == {"revision", "root", "source_files_sha256"} and
            type(reader["revision"]) is str and reader["revision"] in READERS and
            type(reader["root"]) is str and Path(reader["root"]).is_absolute(), "HANDOFF_READER_DECLARATION")
    root, spec = absolute(reader["root"]), READERS[reader["revision"]]
    files = reader["source_files_sha256"]
    require(str(root) == reader["root"] and root.is_dir() and type(files) is dict and
            set(files) == set(spec["files"]) and all(type(v) is str and SHA.fullmatch(v) for v in files.values()),
            "HANDOFF_COMPLETE_READER_ROSTER")
    require({name: sha(read_bytes(root / name)) for name in spec["files"]} == files,
            "HANDOFF_READER_SOURCE_CHANGED")
    require(sha(read_bytes(package / "MANIFEST.json")) == value["manifest_sha256"], "HANDOFF_PACKAGE_PIN_CHANGED")
    return spec


def _load_reader(value):
    """Worker-only data imports, after authenticating every declared source."""
    spec = validate_declaration(value)
    root = absolute(value["reader"]["root"])
    path = root / spec["files"][0]
    name = path.stem
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    require(tuple(module.PRODUCER_FILES) == spec["files"] and module.SCHEMA == spec["schema"] and
            module.SOURCE_ADAPTER == spec["adapter"], "HANDOFF_EXECUTED_READER_REVISION")
    validate_declaration(value)
    return module


def check_relation(value, outcome, pcm_sha256, wav_sha256):
    if outcome == "failed-native":
        require(value is None, "HANDOFF_FAILED_NATIVE_HAS_NO_RECONCILIATION")
        return
    require(outcome == "completed" and type(value) is dict and value.get("schema") == CONTAINER_POLICY,
            "HANDOFF_CONTAINER_RELATION_SCHEMA")
    flags = {"all_other_container_bytes_equal": True, "original_native_array_to_pcm_join_verified": True,
             "signal_samples_changed": False, "original_files_rewritten": False,
             "technical_gate_changed": False, "generation_outcome_changed": False}
    require(all(value.get(key) is expected for key, expected in flags.items()), "HANDOFF_CONTAINER_RELATION_BOUNDARY")
    for key in ("returned_wav", "bulk_generation_wav", "native_array"):
        record = value.get(key)
        require(type(record) is dict and set(record) == {"bytes", "sha256"} and
                type(record["bytes"]) is int and record["bytes"] > 0 and type(record["sha256"]) is str and
                SHA.fullmatch(record["sha256"]), "HANDOFF_CONTAINER_ORIGINAL_HASH")
    require(value["native_float32_pcm_sha256"] == pcm_sha256 and value["returned_wav"]["sha256"] == wav_sha256 and
            type(value["native_frames"]) is int and value["native_frames"] > 0 and value["native_rate"] == 24000,
            "HANDOFF_CONTAINER_NATIVE_IDENTITY")
    bulk, returned = value["bulk_peak_timestamp"], value["returned_peak_timestamp"]
    differences = value["differing_byte_offsets"]
    if value["relation"] == "byte-identical":
        require(value["returned_wav"] == value["bulk_generation_wav"] and differences == [] and bulk == returned and
                (bulk is None or type(bulk) is int and 0 <= bulk <= 0xffffffff), "HANDOFF_IDENTICAL_CONTAINER_CHANGED")
    else:
        require(value["relation"] == "peak-timestamp-plus-one-second" and
                type(bulk) is type(returned) is int and 0 <= returned < bulk <= 0xffffffff and bulk == returned + 1 and
                value["returned_wav"]["bytes"] == value["bulk_generation_wav"]["bytes"] and
                value["returned_wav"]["sha256"] != value["bulk_generation_wav"]["sha256"] and
                type(differences) is list and differences and differences == sorted(set(differences)) and
                all(type(i) is int and 60 <= i < 64 for i in differences), "HANDOFF_ONE_SECOND_CONTAINER_RELATION")


def project_metadata_row(row, vocabulary):
    """Keep observed text and label evidence; never shorten it to four tokens."""
    require(row["consumer"] == "context-v3-metadata-verified-full-transcript-v1" and
            row["split"] in ("train", "development") and
            row["source_role"] == ("train-candidate" if row["split"] == "train" else "development-only") and
            row["purpose"] == "technical-research-only" and row["human_gold"] is False and
            row["independent_test"] is False and row["context_stage"] == "context-supplement-v3" and
            row["source_adapter_schema"] == "kws-active-context-v3-metadata-native-source-v1" and
            row["global_source_key"] == [row["source_plan_sha256"], row["source_id"]], "CONTEXT_V3_ROW_SCOPE")
    require(row["generation_outcome"] in ("completed", "failed-native") and
            row["label_status"] in ("machine_candidate", "uncertain"), "CONTEXT_V3_OBSERVED_STATUS")
    require(row["requested_vocabulary_sha256"] == digest(vocabulary), "CONTEXT_V3_REQUESTED_AND_CONSUMER_VOCABULARY_DIFFER")
    require("native_container_relation" in row, "HANDOFF_CONTAINER_RELATION_REQUIRED")
    check_relation(row["native_container_relation"], row["generation_outcome"],
                   row["original_pcm_sha256"], row["original_wav_sha256"])
    confirmed = row["label_status"] == "machine_candidate"
    intended = row["intended_keyword_id"]
    positive = (row["generation_category"] == "full-positive" and row["intended_kind"] == "positive-intent" and
                type(intended) is int and intended in (1, 2))
    negative = (row["generation_category"] == "ordinary-negative" and row["intended_kind"] == "speech-negative-intent" and
                intended is None)
    require(positive or negative, "CONTEXT_V3_INTENDED_TARGET_ID")
    require(not confirmed or type(row["full_text"]) is str and bool(row["full_text"]), "CONTEXT_V3_CONFIRMED_FULL_TEXT_REQUIRED")
    require(type(row["ctc_mask"]) is bool and row["ctc_mask"] == confirmed and
            type(row["ctc_weight"]) in (int, float) and
            (0 < row["ctc_weight"] <= 1 if confirmed else row["ctc_weight"] == 0), "CONTEXT_V3_LABEL_WEIGHT_MASK")
    require(len(row["keyword_targets"]) == len(row["keyword_mask"]) == len(row["keyword_weight"]) == 2,
            "CONTEXT_V3_KEYWORD_SHAPE")
    for word, target, mask, weight in zip(KEYWORDS, row["keyword_targets"], row["keyword_mask"], row["keyword_weight"]):
        require(type(mask) is bool and type(weight) in (int, float), "CONTEXT_V3_KEYWORD_MASK_TYPE")
        require((confirmed and type(target) is int and target == int(word in row["full_text"]) and 0 < weight <= 1)
                if mask else target is None and weight == 0, "CONTEXT_V3_UNKNOWN_IS_NOT_NEGATIVE")
    require(not confirmed or row["keyword_mask"] == [True, True], "CONTEXT_V3_CONFIRMED_COMPLETE_KEYWORD_LABELS")
    require(not (confirmed and positive) or row["keyword_targets"][intended - 1] == 1,
            "CONTEXT_V3_INTENDED_WAKE_NOT_OBSERVED")
    require(not (confirmed and negative) or row["keyword_targets"] == [0, 0] and
            all(word not in row["full_text"] for word in KEYWORDS), "CONTEXT_V3_NEGATIVE_WAKE_NOT_ABSENT")
    if row["generation_outcome"] == "failed-native":
        require(not confirmed and row["full_text"] is None, "CONTEXT_V3_FAILED_NATIVE_ZERO_LOSS")
    projection = fulltext_targets(row["full_text"] if confirmed else None, vocabulary)
    require(type(row["samples"]) is int and row["samples"] > 0, "CONTEXT_V3_PCM_FRAME_COUNT")
    frames = max(0, 1 + (row["samples"] - 400) // 320)
    minimum = None
    if projection["supported"]:
        ids = projection["target_ids"]
        minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if frames < minimum:
            projection.update(supported=False, reason="unpadded_c1_ctc_geometry_infeasible", target_ids=None)
    eligible = confirmed and projection["supported"]
    return {**row, "consumer": METADATA_ROW_SCHEMA, "vocabulary_sha256": digest(vocabulary),
            "observed_label": {key: row[key] for key in
                ("full_text", "ctc_mask", "ctc_weight", "keyword_targets", "keyword_mask", "keyword_weight")},
            "ctc_mask": eligible, "ctc_weight": row["ctc_weight"] if eligible else 0.0,
            "target_ids": projection["target_ids"] if eligible else None,
            "keyword_targets": list(row["keyword_targets"]) if eligible else [None, None],
            "keyword_mask": list(row["keyword_mask"]) if eligible else [False, False],
            "keyword_weight": list(row["keyword_weight"]) if eligible else [0.0, 0.0],
            "ctc_reason": projection["reason"], "oov_characters": projection["oov_characters"],
            "expected_c1_frames": frames, "ctc_minimum_frames": minimum,
            "keyword_loss_used": False, "quarantined": not eligible}


def _aggregate_metadata(packages, loader, source_files, *, expected_plan_sha256, prior_sources=()):
    """All packages are replayed; global cell, PCM and split aliases are checked."""
    require(type(packages) is list and 1 <= len(packages) <= 72 and
            loader.SCHEMA == "kws-active-context-v3-metadata-research-weak-view-v1" and
            loader.SOURCE_ADAPTER == "kws-active-context-v3-metadata-native-source-v1" and
            type(expected_plan_sha256) is str and loader.weak.SHA.fullmatch(expected_plan_sha256),
            "CONTEXT_V3_PACKAGE_ROSTER")
    weak = loader.weak
    roles = weak.SourceRoles()
    identities, original_pcm, package_ids, rows, parents = set(), set(), set(), [], []
    previous, plan_cells = [], None
    for row in prior_sources:
        roles.add(row["role"], family=row.get("source_family_id"), speaker=row.get("speaker_id"),
                  pcm_sha256=row.get("pcm_sha256"), wav_sha256=row.get("wav_sha256"),
                  parent_source_sha256=row.get("parent_source_sha256"))
        if row.get("source_id"):
            identities.add(row["source_id"])
        if row.get("original_pcm_sha256"):
            original_pcm.add(row["original_pcm_sha256"])
        previous.append(row)
    for declaration in packages:
        require(set(declaration) == {"path", "manifest_sha256"} and Path(declaration["path"]).is_absolute(),
                "CONTEXT_V3_PACKAGE_DECLARATION")
        package = Path(declaration["path"])
        manifest_raw = loader.read_bytes(package / "MANIFEST.json")
        identity = sha(manifest_raw)
        require(identity == declaration["manifest_sha256"] and identity not in package_ids,
                "CONTEXT_V3_PACKAGE_IDENTITY_OR_DUPLICATE")
        package_ids.add(identity)
        # This must happen before accessing any projected or filtered export.
        report = loader.verify_view(package, expected_manifest_sha256=identity, other_split_sources=previous)
        require(report["producer"]["source_files_sha256"] == source_files,
                "CONTEXT_V3_EXECUTED_VERIFIER_DIFFERS_FROM_PRODUCER")
        inputs_raw = loader.read_bytes(package / "evidence/inputs.json")
        inputs = loader.loads(inputs_raw)
        report_raw = loader.read_bytes(package / "report.json", 32 * 1024 * 1024)
        require(loader.loads(report_raw) == report and loader.read_bytes(package / "MANIFEST.json") == manifest_raw,
                "CONTEXT_V3_PACKAGE_CHANGED_DURING_REPLAY")
        require([row["item_id"] for row in report["items"]] == [clip["id"] for clip in inputs["clips"]],
                "CONTEXT_V3_AUDIO_ROSTER")
        adapter = report["source_adapter"]
        require(report["schema"] == loader.SCHEMA and adapter["schema"] == loader.SOURCE_ADAPTER and
                adapter["stage"] == "context-supplement-v3" and adapter["plan_sha256"] == expected_plan_sha256 and
                adapter["inherited_d1_cells_reissued"] == adapter["inherited_d2_cells_reissued"] ==
                adapter["inherited_context_v1_cells_reissued"] == adapter["inherited_context_v2_cells_reissued"] == 0 and
                adapter["vocabulary_preflight_scope"] == "requested_text_only",
                "CONTEXT_V3_PLAN_AND_SOURCE_SCOPE")
        plan = loader.loads(loader.read_bytes(package / "evidence/source/context-v3/plan.json", 8 * 1024 * 1024))
        current_cells = {cell["cell_id"] for cell in plan["cells"]}
        require(digest(plan) == expected_plan_sha256 and plan["schema"] == "kws-context-supplement-plan-v3" and
                len(current_cells) == len(plan["cells"]) == 72 and
                (plan_cells is None or plan_cells == current_cells), "CONTEXT_V3_FROZEN_CELL_SCOPE")
        require(adapter["context_parent_plan_sha256"] == plan["context_parent"]["plan_sha256"] and
                digest(plan["context_parent"]["plan"]) == plan["context_parent"]["plan_sha256"] and
                plan["context_parent"]["plan"]["schema"] == "kws-context-supplement-plan-v2" and
                adapter["vocabulary_sha256"] == plan["vocabulary"]["canonical_sha256"] == digest(plan["vocabulary"]["value"]) and
                adapter["vocabulary_file_sha256"] == plan["vocabulary"]["file_sha256"],
                "CONTEXT_V3_PARENT_OR_VOCABULARY_CHANGED")
        plan_cells = current_cells
        require(adapter["native_container_policy"] == CONTAINER_POLICY and
                adapter["reconciled_peak_timestamp_cell_ids"] == [item["source"]["source_id"]
                    for item in report["items"] if item["source"]["native_container_relation"] is not None and
                    item["source"]["native_container_relation"]["relation"] == "peak-timestamp-plus-one-second"],
                "HANDOFF_METADATA_POLICY_OR_RECONCILED_ROSTER")
        for raw, clip in zip(report["items"], inputs["clips"]):
            source = raw["source"]
            require(source["source_id"] in plan_cells and source["source_plan_sha256"] == expected_plan_sha256,
                    "CONTEXT_V3_OBSERVED_CELL_OUTSIDE_PLAN")
            require(source["source_id"] not in identities, "CONTEXT_V3_DUPLICATE_GLOBAL_SOURCE_ID")
            require(clip["original_pcm"]["sha256"] not in original_pcm, "CONTEXT_V3_DUPLICATE_NATIVE_PCM")
            identities.add(source["source_id"])
            original_pcm.add(clip["original_pcm"]["sha256"])
            weak.add_source_role(roles, source, clip, raw["audio_derivation"])
            previous.append({"role": source["role"], "source_family_id": source["source_family_id"],
                "speaker_id": source["speaker_id"], "pcm_sha256": raw["audio_derivation"]["pcm_sha256"],
                "wav_sha256": raw["audio_derivation"]["file_sha256"]})
            row = weak.training_row(raw, "fsmn-fulltext")
            row.update(audio=str(package / weak.relative(raw["audio"])),
                audio_path=str(package / weak.relative(raw["audio"])), consumer="context-v3-metadata-verified-full-transcript-v1",
                label_status=raw["decision"]["status"], label_reason=raw["decision"]["reason"],
                global_source_key=[source["source_plan_sha256"], source["source_id"]],
                source_role=source["role"], parent_source_sha256=source["parent_source_sha256"],
                reference_wav_sha256=source["reference_wav_sha256"], reference_pcm_sha256=source["reference_pcm_sha256"],
                retained_row_sha256=digest(raw), parent_manifest_sha256=identity,
                parent_report_sha256=sha(report_raw), parent_inputs_sha256=sha(inputs_raw),
                source_adapter_schema=adapter["schema"], source_adapter_sha256=digest(adapter),
                native_container_relation=source["native_container_relation"],
                context_stage=adapter["stage"], asr_transcripts=raw["asr"],
                context_observation=raw["context_observation"],
                intended_keyword_id=source["intended_keyword_id"],
                requested_context_class=source["requested_context_class"],
                requested_keyword_occurrences=source["requested_keyword_occurrences"],
                context_parent_plan_sha256=adapter["context_parent_plan_sha256"],
                requested_vocabulary_sha256=adapter["vocabulary_sha256"],
                requested_vocabulary_file_sha256=adapter["vocabulary_file_sha256"],
                requested_text_preflight=source["requested_text_preflight"],
                explicit_additive_revision=source["explicit_additive_revision"],
                same_requested_text_as_parent=source["same_requested_text_as_parent"],
                normalized_prompt_sha256=source["normalized_prompt_sha256"],
                request_payload_sha256=source["request_payload_sha256"],
                historical_prompt_reuse=source["historical_prompt_reuse"])
            rows.append(row)
        parents.append({**declaration, "observed_rows": len(report["items"]),
                        "plan_sha256": adapter["plan_sha256"], "stage": adapter["stage"]})
    require(len(rows) <= 72, "CONTEXT_V3_BOUNDED_NEW_OBSERVATIONS")
    rows.sort(key=lambda row: (row["source_family_id"], row["source_id"]))
    return rows, {"schema": METADATA_AGGREGATE_SCHEMA, "parents": parents, "plan_sha256": expected_plan_sha256, "observed_rows": len(rows),
        "planned_rows": len(plan_cells), "complete_observed_coverage": {row["source_id"] for row in rows} == plan_cells,
        "unobserved_cell_ids": sorted(plan_cells - {row["source_id"] for row in rows}),
        "by_split": dict(Counter(row["split"] for row in rows)), "source_id_global_deduplication": True,
        "all_rows_replayed_before_projection": True, "human_gold": False, "new_model_calls": 0}


def _projection_summary(rows, report, identity, consumer):
    return {**report, "consumer": consumer, "vocabulary": identity,
        "eligible_rows": sum(row["ctc_mask"] for row in rows),
        "zero_weight_rows": sum(not row["ctc_mask"] for row in rows),
        "projection_reasons": dict(sorted(Counter(row["ctc_reason"] for row in rows).items())),
        "full_observed_transcripts_retained": True,
        "eligible_rows_by_category": dict(sorted(Counter(row["generation_category"] for row in rows if row["ctc_mask"]).items())),
        "ordinary_speech_uses_complete_nonblank_targets": True}


def role_records(report, inputs):
    """Carry every alias checked by the frozen reader into the next process."""
    records = []
    for item, clip in zip(report["items"], inputs["clips"]):
        source, measured = item["source"], item["audio_derivation"]
        role = source["role"]
        records.append({"role": role, "source_id": source["source_id"],
            "original_pcm_sha256": clip["original_pcm"]["sha256"],
            "source_family_id": source["source_family_id"], "speaker_id": source["speaker_id"],
            "pcm_sha256": clip["original_pcm"]["sha256"], "wav_sha256": clip["original_wav"]["sha256"]})
        if "reference_id" in source:
            records.append({"role": role, "pcm_sha256": source["reference_pcm_sha256"],
                "wav_sha256": source["reference_wav_sha256"], "parent_source_sha256": source["parent_source_sha256"]})
            if source.get("generation_outcome") != "failed-native":
                audio = source["original_generation_row"]["audio"]
                records.append({"role": role, "pcm_sha256": audio["pcm_sha256"], "wav_sha256": audio["wav_sha256"]})
        records.append({"role": role, "pcm_sha256": measured["pcm_sha256"], "wav_sha256": measured["file_sha256"]})
        relation = source.get("native_container_relation")
        if relation is not None:
            records.append({"role": role, "wav_sha256": relation["bulk_generation_wav"]["sha256"]})
    return records


class GlobalSources:
    """Check aliases and native reuse even across different reader processes."""
    def __init__(self):
        self.roles, self.cells, self.native_pcm = {}, set(), set()

    def add(self, source):
        require(type(source) is dict and source.get("role") in ("train-candidate", "development-only"),
                "HANDOFF_SOURCE_SPLIT_ROLE")
        role = source["role"]
        for field, kind in (("source_family_id", "family"), ("speaker_id", "speaker"),
                            ("pcm_sha256", "pcm"), ("wav_sha256", "file"), ("parent_source_sha256", "file"),
                            ("reference_pcm_sha256", "pcm"), ("reference_wav_sha256", "file")):
            value = source.get(field)
            if value is None:
                continue
            require(type(value) is str and value, "HANDOFF_SOURCE_ALIAS_VALUE")
            require(self.roles.get((kind, value), role) == role, "HANDOFF_SOURCE_SPLIT_LEAK:" + kind)
            self.roles[(kind, value)] = role
        for field, seen, error in (("source_id", self.cells, "HANDOFF_DUPLICATE_GLOBAL_SOURCE_ID"),
                                    ("original_pcm_sha256", self.native_pcm, "HANDOFF_DUPLICATE_NATIVE_PCM")):
            value = source.get(field)
            if value is not None:
                require(type(value) is str and value and value not in seen, error)
                seen.add(value)


def _worker(request):
    require(type(request) is dict and set(request) == {"package", "options", "prior_sources", "pipeline_source_sha256"} and
            request["pipeline_source_sha256"] == source_identity(), "HANDOFF_WORKER_SOURCE_BINDING")
    value, options = request["package"], request["options"]
    require(type(options) is dict and set(options) == {"expected_plan_sha256", "vocabulary_path",
            "vocabulary_sha256", "original_tokens_path"} and type(request["prior_sources"]) is list,
            "HANDOFF_WORKER_OPTIONS")
    loader = _load_reader(value)
    package = absolute(value["path"])
    manifest_raw = read_bytes(package / "MANIFEST.json")
    manifest = loads(manifest_raw)
    report_raw = read_bytes(package / "report.json")
    inputs_raw = read_bytes(package / "evidence/inputs.json")
    require(sha(report_raw) == manifest["report.json"] and sha(inputs_raw) == manifest["evidence/inputs.json"],
            "HANDOFF_REPLAY_INPUT_PINS")
    declaration = {key: value[key] for key in ("path", "manifest_sha256")}
    files = value["reader"]["source_files_sha256"]
    if value["reader"]["revision"] == "context-v3-r1":
        rows, report = original.verified_projection([declaration], loader, files, **options,
                                                   prior_sources=request["prior_sources"])
    else:
        vocabulary, identity = original.load_vocabulary(options["vocabulary_path"], options["vocabulary_sha256"],
                                                        options["original_tokens_path"], loader)
        raw_rows, report = _aggregate_metadata([declaration], loader, files,
            expected_plan_sha256=options["expected_plan_sha256"], prior_sources=request["prior_sources"])
        require(all(row["requested_vocabulary_sha256"] == identity["canonical_sha256"] and
                    row["requested_vocabulary_file_sha256"] == identity["file_sha256"] for row in raw_rows),
                "HANDOFF_CONSUMER_VOCABULARY_DIFFERS_FROM_PLAN")
        rows = [project_metadata_row(row, vocabulary) for row in raw_rows]
        report = _projection_summary(rows, report, identity, METADATA_ROW_SCHEMA)
    require(read_bytes(package / "MANIFEST.json") == manifest_raw and read_bytes(package / "report.json") == report_raw and
            read_bytes(package / "evidence/inputs.json") == inputs_raw, "HANDOFF_PACKAGE_CHANGED_DURING_REPLAY")
    retained, inputs = loads(report_raw), loads(inputs_raw)
    require(retained["producer"]["source_files_sha256"] == files, "HANDOFF_EXECUTED_READER_DIFFERS_FROM_PRODUCER")
    spec = validate_declaration(value)
    require(retained["schema"] == spec["schema"] and retained["source_adapter"]["schema"] == spec["adapter"],
            "HANDOFF_RETAINED_READER_REVISION")
    plan = loads(read_bytes(package / "evidence/source/context-v3/plan.json"))
    require(digest(plan) == options["expected_plan_sha256"], "HANDOFF_PLAN_CHANGED")
    require(source_identity() == request["pipeline_source_sha256"], "HANDOFF_WORKER_SOURCE_CHANGED")
    return {"declaration": value, "rows": rows, "report": report,
            "plan_cell_ids": [cell["cell_id"] for cell in plan["cells"]],
            "prior_records": role_records(retained, inputs),
            "pipeline_source_sha256": request["pipeline_source_sha256"], "worker_pid": os.getpid()}


def _run_worker(value, options, previous, sources):
    with tempfile.TemporaryDirectory(prefix="context-v3-handoff-replay-") as temporary:
        root = Path(temporary)
        request, output, log = root / "request.json", root / "output.json", root / "worker.log"
        write_new(request, {"package": value, "options": options, "prior_sources": previous,
                            "pipeline_source_sha256": sources})
        optimization = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
        with log.open("xb") as stream:
            result = subprocess.run([sys.executable, "-I", "-S", "-B", *optimization, str(Path(__file__).absolute()),
                                     "_replay-package", "--request", str(request), "--output", str(output)],
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=120, check=False)
        detail = log.read_bytes()[-4096:].decode("utf-8", errors="replace")
        require(result.returncode == 0 and output.is_file(), "HANDOFF_PACKAGE_REPLAY_FAILED:" + detail)
        response = loads(read_bytes(output))
    require(response["declaration"] == value and response["pipeline_source_sha256"] == sources and
            type(response["worker_pid"]) is int and response["worker_pid"] != os.getpid(),
            "HANDOFF_WORKER_RESPONSE_BINDING")
    return response


def aggregate(packages, *, expected_plan_sha256, vocabulary_path, vocabulary_sha256, original_tokens_path,
              prior_sources=()):
    """Return all replayed rows and a deterministic report, with no model calls."""
    require(type(packages) is list and 1 <= len(packages) <= 72 and type(expected_plan_sha256) is str and
            SHA.fullmatch(expected_plan_sha256) and type(vocabulary_sha256) is str and SHA.fullmatch(vocabulary_sha256),
            "HANDOFF_BOUNDED_DECLARATION")
    packages = loads(canonical(packages))
    options = {"expected_plan_sha256": expected_plan_sha256, "vocabulary_path": str(absolute(vocabulary_path)),
               "vocabulary_sha256": vocabulary_sha256, "original_tokens_path": str(absolute(original_tokens_path))}
    vocabulary_raw = read_bytes(options["vocabulary_path"])
    token_raw = read_bytes(options["original_tokens_path"])
    require(sha(vocabulary_raw) == vocabulary_sha256, "HANDOFF_VOCABULARY_PIN_CHANGED")
    package_ids = set()
    for value in packages:
        validate_declaration(value)
        require(value["manifest_sha256"] not in package_ids, "HANDOFF_DUPLICATE_PACKAGE")
        package_ids.add(value["manifest_sha256"])
    require(type(prior_sources) in (list, tuple), "HANDOFF_PRIOR_SOURCE_ROSTER")
    previous = loads(canonical(list(prior_sources)))
    global_sources = GlobalSources()
    for source in previous:
        global_sources.add(source)
    rows, parents, expected_cells, vocabulary_identity = [], [], None, None
    sources = source_identity()
    for value in packages:
        response = _run_worker(value, options, previous, sources)
        validate_declaration(value)
        report, observed = response["report"], response["rows"]
        require(type(observed) is list and observed and report["observed_rows"] == len(observed) and
                report["plan_sha256"] == expected_plan_sha256 and report["new_model_calls"] == 0 and
                report["human_gold"] is False, "HANDOFF_VERIFIED_WORKER_SCOPE")
        cells = response["plan_cell_ids"]
        require(type(cells) is list and len(cells) == len(set(cells)) == 72 and
                (expected_cells is None or cells == expected_cells), "HANDOFF_MIXED_FROZEN_PLANS")
        expected_cells = cells
        require(vocabulary_identity is None or vocabulary_identity == report["vocabulary"],
                "HANDOFF_MIXED_VOCABULARIES")
        vocabulary_identity = report["vocabulary"]
        spec = READERS[value["reader"]["revision"]]
        consumer = original.ROW_SCHEMA if value["reader"]["revision"] == "context-v3-r1" else METADATA_ROW_SCHEMA
        require(all(row["source_id"] in cells and row["source_plan_sha256"] == expected_plan_sha256 and
                    row["source_adapter_schema"] == spec["adapter"] and row["consumer"] == consumer and
                    row["parent_manifest_sha256"] == value["manifest_sha256"] for row in observed),
                "HANDOFF_ROW_READER_BINDING")
        source_records = response["prior_records"]
        require([item["source_id"] for item in source_records if "source_id" in item] ==
                [item["source"]["source_id"] for item in loads(read_bytes(Path(value["path"]) / "report.json"))["items"]],
                "HANDOFF_ALIAS_ROSTER_CHANGED")
        for source in source_records:
            global_sources.add(source)
        previous.extend(source_records)
        rows.extend(observed)
        parents.append({**value, "observed_rows": len(observed), "source_adapter_schema": spec["adapter"],
                        "row_consumer": consumer, "source_ids": sorted(row["source_id"] for row in observed)})
        require(source_identity() == sources, "HANDOFF_PROJECTION_SOURCE_CHANGED")
    require(len(rows) <= 72 and len({row["source_id"] for row in rows}) == len(rows), "HANDOFF_BOUNDED_GLOBAL_CELLS")
    require(read_bytes(options["vocabulary_path"]) == vocabulary_raw and
            read_bytes(options["original_tokens_path"]) == token_raw, "HANDOFF_VOCABULARY_CHANGED_DURING_REPLAY")
    for value in packages:
        validate_declaration(value)
    rows.sort(key=lambda row: (row["source_family_id"], row["source_id"]))
    observed_ids = {row["source_id"] for row in rows}
    report = {"schema": SCHEMA, "parents": parents, "plan_sha256": expected_plan_sha256,
        "observed_rows": len(rows), "planned_rows": len(expected_cells),
        "complete_observed_coverage": observed_ids == set(expected_cells),
        "unobserved_cell_ids": sorted(set(expected_cells) - observed_ids),
        "by_split": dict(Counter(row["split"] for row in rows)), "source_id_global_deduplication": True,
        "native_pcm_global_deduplication": True, "cross_reader_reference_aliases_checked": True,
        "all_rows_replayed_before_projection": True, "fresh_process_per_package": True,
        "row_consumer_schemas": dict(sorted(Counter(row["consumer"] for row in rows).items())),
        "reader_revision_rows": dict(sorted(Counter({revision: sum(p["observed_rows"] for p in parents
            if p["reader"]["revision"] == revision) for revision in READERS}).items())),
        "metadata_reconciled_cell_ids": sorted(row["source_id"] for row in rows if
            row.get("native_container_relation") is not None and
            row["native_container_relation"]["relation"] == "peak-timestamp-plus-one-second"),
        "projection_source_files_sha256": sources, "human_gold": False, "new_model_calls": 0,
        "original_packages_rewritten": False, "reader_source_substitution": False}
    return rows, _projection_summary(rows, report, vocabulary_identity, CONSUMER)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    project = commands.add_parser("project")
    project.add_argument("--declaration", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("_replay-package")
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not absolute(args.output).exists(), "HANDOFF_CREATE_ONLY_OUTPUT")
    if args.command == "_replay-package":
        result = _worker(loads(read_bytes(args.request)))
    else:
        declaration = loads(read_bytes(args.declaration))
        require(type(declaration) is dict and set(declaration) in (
            {"packages", "expected_plan_sha256", "vocabulary_path", "vocabulary_sha256", "original_tokens_path"},
            {"packages", "expected_plan_sha256", "vocabulary_path", "vocabulary_sha256", "original_tokens_path", "prior_sources"}),
            "HANDOFF_PROJECTION_DECLARATION")
        rows, report = aggregate(**declaration)
        result = {"rows": rows, "report": report}
    write_new(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
