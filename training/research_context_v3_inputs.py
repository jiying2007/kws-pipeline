"""Context-v3 verified full-transcript aggregation; no model imports or calls.

Replay every native/ASR row with the v3 producer before projecting it into the
unchanged D1 2601-symbol vocabulary. Uncertain and unsupported rows stay in the
audit population, with no objective weight. Batch-local clip IDs are not IDs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import sys

from research_d1_vocab import digest, fulltext_targets, require, validate_vocabulary

SCHEMA = "kws-context-v3-verified-aggregate-v1"
ROW_SCHEMA = "kws-context-v3-fulltext-projection-v1"
KEYWORDS = ("你好小窝", "小窝小窝")
DATA_FILES = ("automation/active_context_v3_weak_view.py", "automation/active_asr_weak_view.py",
              "research/auto-tts-voice-clone-20261102/context_supplement_v3.py",
              "research/auto-tts-voice-clone-20261102/context_supplement_v2.py",
              "research/auto-tts-voice-clone-20261102/context_supplement.py",
              "research/auto-tts-voice-clone-20261102/d2_pool.py",
              "research/auto-tts-voice-clone-20261102/bulk_pool.py",
              "research/auto-tts-voice-clone-20261102/bulk_runner.py",
              "research/auto-tts-voice-clone-20261102/runner.py",
              "research/auto-tts-voice-clone-20261102/campaign.py",
              "kws_data_contract/__init__.py", "kws_data_contract/identity.py",
              "kws_data_contract/reference_text.py", "kws_data_contract/source_registry.py")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_data_module(path):
    path = Path(path).absolute()
    root = path.parents[1]
    require(path == root / DATA_FILES[0] and not any(p.is_symlink() for p in (path, *path.parents)),
            "CONTEXT_V3_LOADER_PATH")
    for name, module in list(sys.modules.items()):
        if name in ("active_context_v3_weak_view", "active_asr_weak_view", "context_supplement_v3", "context_supplement_v2", "context_supplement", "d2_pool",
                    "bulk_pool", "bulk_runner", "runner", "campaign") or \
                name == "kws_data_contract" or name.startswith("kws_data_contract."):
            origin = getattr(module, "__file__", None)
            require(origin and Path(origin).resolve().is_relative_to(root), "CONTEXT_V3_MIXED_DATA_MODULES")
    before = {name: sha((root / name).read_bytes()) for name in DATA_FILES}
    spec = importlib.util.spec_from_file_location("active_context_v3_weak_view", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(tuple(module.PRODUCER_FILES) == DATA_FILES and
            before == {name: sha((root / name).read_bytes()) for name in DATA_FILES},
            "CONTEXT_V3_LOADER_SOURCE_CHANGED")
    return module, {"root": str(root), "files": before}


def project_row(row, vocabulary):
    """Keep observed text and label evidence; never shorten it to four tokens."""
    require(row["consumer"] == "context-v3-verified-full-transcript-v1" and
            row["split"] in ("train", "development") and
            row["source_role"] == ("train-candidate" if row["split"] == "train" else "development-only") and
            row["purpose"] == "technical-research-only" and row["human_gold"] is False and
            row["independent_test"] is False and row["context_stage"] == "context-supplement-v3" and
            row["source_adapter_schema"] == "kws-active-context-v3-native-source-v1" and
            row["global_source_key"] == [row["source_plan_sha256"], row["source_id"]], "CONTEXT_V3_ROW_SCOPE")
    require(row["generation_outcome"] in ("completed", "failed-native") and
            row["label_status"] in ("machine_candidate", "uncertain"), "CONTEXT_V3_OBSERVED_STATUS")
    require(row["requested_vocabulary_sha256"] == digest(vocabulary), "CONTEXT_V3_REQUESTED_AND_CONSUMER_VOCABULARY_DIFFER")
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
    return {**row, "consumer": ROW_SCHEMA, "vocabulary_sha256": digest(vocabulary),
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


def aggregate(packages, loader, source_files, *, expected_plan_sha256, prior_sources=()):
    """All packages are replayed; global cell, PCM and split aliases are checked."""
    require(type(packages) is list and 1 <= len(packages) <= 72 and
            loader.SCHEMA == "kws-active-context-v3-research-weak-view-v1" and
            loader.SOURCE_ADAPTER == "kws-active-context-v3-native-source-v1" and
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
                audio_path=str(package / weak.relative(raw["audio"])), consumer="context-v3-verified-full-transcript-v1",
                label_status=raw["decision"]["status"], label_reason=raw["decision"]["reason"],
                global_source_key=[source["source_plan_sha256"], source["source_id"]],
                source_role=source["role"], parent_source_sha256=source["parent_source_sha256"],
                reference_wav_sha256=source["reference_wav_sha256"], reference_pcm_sha256=source["reference_pcm_sha256"],
                retained_row_sha256=digest(raw), parent_manifest_sha256=identity,
                parent_report_sha256=sha(report_raw), parent_inputs_sha256=sha(inputs_raw),
                source_adapter_schema=adapter["schema"], source_adapter_sha256=digest(adapter),
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
    return rows, {"schema": SCHEMA, "parents": parents, "plan_sha256": expected_plan_sha256, "observed_rows": len(rows),
        "planned_rows": len(plan_cells), "complete_observed_coverage": {row["source_id"] for row in rows} == plan_cells,
        "unobserved_cell_ids": sorted(plan_cells - {row["source_id"] for row in rows}),
        "by_split": dict(Counter(row["split"] for row in rows)), "source_id_global_deduplication": True,
        "all_rows_replayed_before_projection": True, "human_gold": False, "new_model_calls": 0}


def load_vocabulary(path, expected_file_sha256, original_tokens_path, reader):
    """Authenticate the existing full2601 vocabulary; never extend it from data."""
    path, original_tokens_path = Path(path).absolute(), Path(original_tokens_path).absolute()
    raw = reader.read_bytes(path)
    require(sha(raw) == expected_file_sha256, "CONTEXT_V3_VOCABULARY_FILE_PIN_CHANGED")
    value = reader.loads(raw)
    original = reader.read_bytes(original_tokens_path, 65536)
    validate_vocabulary(value, original)
    return value, {"path": str(path), "file_sha256": sha(raw), "canonical_sha256": digest(value),
                   "original_tokens_sha256": sha(original), "output_count": value["output_count"],
                   "blank_id": value["blank_id"], "observed_text_used_for_construction": False}


def verified_projection(packages, loader, source_files, *, expected_plan_sha256, vocabulary_path,
                        vocabulary_sha256, original_tokens_path, prior_sources=()):
    """Complete native/ASR replay followed by the separate pinned consumer map."""
    vocabulary, identity = load_vocabulary(vocabulary_path, vocabulary_sha256, original_tokens_path, loader)
    rows, report = aggregate(packages, loader, source_files, expected_plan_sha256=expected_plan_sha256,
                             prior_sources=prior_sources)
    require(all(row["requested_vocabulary_sha256"] == identity["canonical_sha256"] and
                row["requested_vocabulary_file_sha256"] == identity["file_sha256"] for row in rows),
            "CONTEXT_V3_CONSUMER_VOCABULARY_DIFFERS_FROM_PLAN")
    projected = [project_row(row, vocabulary) for row in rows]
    return projected, {**report, "consumer": ROW_SCHEMA, "vocabulary": identity,
        "eligible_rows": sum(row["ctc_mask"] for row in projected),
        "zero_weight_rows": sum(not row["ctc_mask"] for row in projected),
        "projection_reasons": dict(sorted(Counter(row["ctc_reason"] for row in projected).items())),
        "full_observed_transcripts_retained": True,
        "eligible_rows_by_category": dict(sorted(Counter(row["generation_category"] for row in projected
                                                          if row["ctc_mask"]).items())),
        "ordinary_speech_uses_complete_nonblank_targets": True}
