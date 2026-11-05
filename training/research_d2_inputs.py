"""D2-only verified full-transcript aggregation; no model imports or calls.

Replay every native/ASR row with the D2 producer before projecting it into the
unchanged D1 2601-symbol vocabulary. Uncertain and unsupported rows stay in the
audit population, with no objective weight. Batch-local clip IDs are not IDs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import sys

from research_d1_vocab import digest, fulltext_targets, require

SCHEMA = "kws-d2-verified-aggregate-v1"
ROW_SCHEMA = "kws-d2-fulltext-projection-v1"
KEYWORDS = ("你好小窝", "小窝小窝")
DATA_FILES = ("automation/active_d2_weak_view.py", "automation/active_asr_weak_view.py",
              "research/auto-tts-voice-clone-20261102/d2_pool.py",
              "research/auto-tts-voice-clone-20261102/bulk_pool.py",
              "kws_data_contract/__init__.py", "kws_data_contract/identity.py",
              "kws_data_contract/reference_text.py", "kws_data_contract/source_registry.py")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_data_module(path):
    path = Path(path).absolute()
    root = path.parents[1]
    require(path == root / DATA_FILES[0] and not any(p.is_symlink() for p in (path, *path.parents)),
            "D2_LOADER_PATH")
    for name, module in list(sys.modules.items()):
        if name in ("active_d2_weak_view", "active_asr_weak_view", "d2_pool", "bulk_pool") or \
                name == "kws_data_contract" or name.startswith("kws_data_contract."):
            origin = getattr(module, "__file__", None)
            require(origin and Path(origin).resolve().is_relative_to(root), "D2_MIXED_DATA_MODULES")
    before = {name: sha((root / name).read_bytes()) for name in DATA_FILES}
    spec = importlib.util.spec_from_file_location("active_d2_weak_view", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(tuple(module.PRODUCER_FILES) == DATA_FILES and
            before == {name: sha((root / name).read_bytes()) for name in DATA_FILES},
            "D2_LOADER_SOURCE_CHANGED")
    return module, {"root": str(root), "files": before}


def project_row(row, vocabulary):
    """Keep observed text and label evidence; never shorten it to four tokens."""
    require(row["consumer"] == "d2-verified-full-transcript-v1" and
            row["split"] in ("train", "development") and
            row["source_role"] == ("train-candidate" if row["split"] == "train" else "development-only") and
            row["purpose"] == "technical-research-only" and row["human_gold"] is False and
            row["independent_test"] is False, "D2_ROW_SCOPE")
    require(row["generation_outcome"] in ("completed", "failed-native") and
            row["label_status"] in ("machine_candidate", "uncertain"), "D2_OBSERVED_STATUS")
    confirmed = row["label_status"] == "machine_candidate"
    require(not confirmed or type(row["full_text"]) is str and bool(row["full_text"]), "D2_CONFIRMED_FULL_TEXT_REQUIRED")
    require(type(row["ctc_mask"]) is bool and row["ctc_mask"] == confirmed and
            type(row["ctc_weight"]) in (int, float) and
            (0 < row["ctc_weight"] <= 1 if confirmed else row["ctc_weight"] == 0), "D2_LABEL_WEIGHT_MASK")
    require(len(row["keyword_targets"]) == len(row["keyword_mask"]) == len(row["keyword_weight"]) == 2,
            "D2_KEYWORD_SHAPE")
    for word, target, mask, weight in zip(KEYWORDS, row["keyword_targets"], row["keyword_mask"], row["keyword_weight"]):
        require(type(mask) is bool and type(weight) in (int, float), "D2_KEYWORD_MASK_TYPE")
        require((confirmed and type(target) is int and target == int(word in row["full_text"]) and 0 < weight <= 1)
                if mask else target is None and weight == 0, "D2_UNKNOWN_IS_NOT_NEGATIVE")
    if row["generation_outcome"] == "failed-native":
        require(not confirmed and row["full_text"] is None, "D2_FAILED_NATIVE_ZERO_LOSS")
    projection = fulltext_targets(row["full_text"] if confirmed else None, vocabulary)
    require(type(row["samples"]) is int and row["samples"] > 0, "D2_PCM_FRAME_COUNT")
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


def aggregate(packages, loader, source_files, *, prior_sources=()):
    """All packages are replayed; global cell, PCM and split aliases are checked."""
    require(type(packages) is list and 1 <= len(packages) <= 256, "D2_PACKAGE_ROSTER")
    weak = loader.weak
    roles = weak.SourceRoles()
    identities, original_pcm, package_ids, rows, parents = set(), set(), set(), [], []
    previous = []
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
                "D2_PACKAGE_DECLARATION")
        package = Path(declaration["path"])
        manifest_raw = loader.read_bytes(package / "MANIFEST.json")
        identity = sha(manifest_raw)
        require(identity == declaration["manifest_sha256"] and identity not in package_ids,
                "D2_PACKAGE_IDENTITY_OR_DUPLICATE")
        package_ids.add(identity)
        # This must happen before accessing any projected or filtered export.
        report = loader.verify_view(package, expected_manifest_sha256=identity, other_split_sources=previous)
        require(report["producer"]["source_files_sha256"] == source_files,
                "D2_EXECUTED_VERIFIER_DIFFERS_FROM_PRODUCER")
        inputs_raw = loader.read_bytes(package / "evidence/inputs.json")
        inputs = loader.loads(inputs_raw)
        report_raw = loader.read_bytes(package / "report.json", 32 * 1024 * 1024)
        require(loader.loads(report_raw) == report and loader.read_bytes(package / "MANIFEST.json") == manifest_raw,
                "D2_PACKAGE_CHANGED_DURING_REPLAY")
        require([row["item_id"] for row in report["items"]] == [clip["id"] for clip in inputs["clips"]],
                "D2_AUDIO_ROSTER")
        adapter = report["source_adapter"]
        for raw, clip in zip(report["items"], inputs["clips"]):
            source = raw["source"]
            require(source["source_id"] not in identities, "D2_DUPLICATE_GLOBAL_SOURCE_ID")
            require(clip["original_pcm"]["sha256"] not in original_pcm, "D2_DUPLICATE_NATIVE_PCM")
            identities.add(source["source_id"])
            original_pcm.add(clip["original_pcm"]["sha256"])
            weak.add_source_role(roles, source, clip, raw["audio_derivation"])
            previous.append({"role": source["role"], "source_family_id": source["source_family_id"],
                "speaker_id": source["speaker_id"], "pcm_sha256": raw["audio_derivation"]["pcm_sha256"],
                "wav_sha256": raw["audio_derivation"]["file_sha256"]})
            row = weak.training_row(raw, "fsmn-fulltext")
            row.update(audio=str(package / weak.relative(raw["audio"])),
                audio_path=str(package / weak.relative(raw["audio"])), consumer="d2-verified-full-transcript-v1",
                label_status=raw["decision"]["status"], label_reason=raw["decision"]["reason"],
                global_source_key=[source["source_plan_sha256"], source["source_id"]],
                source_role=source["role"], parent_source_sha256=source["parent_source_sha256"],
                reference_wav_sha256=source["reference_wav_sha256"], reference_pcm_sha256=source["reference_pcm_sha256"],
                retained_row_sha256=digest(raw), parent_manifest_sha256=identity,
                parent_report_sha256=sha(report_raw), parent_inputs_sha256=sha(inputs_raw),
                source_adapter_sha256=digest(adapter), d2_stage=adapter["stage"], asr_transcripts=raw["asr"])
            rows.append(row)
        parents.append({**declaration, "observed_rows": len(report["items"]),
                        "plan_sha256": adapter["plan_sha256"], "stage": adapter["stage"]})
    require(len(rows) <= 1792, "D2_BOUNDED_NEW_OBSERVATIONS")
    rows.sort(key=lambda row: (row["source_family_id"], row["source_id"]))
    return rows, {"schema": SCHEMA, "parents": parents, "observed_rows": len(rows),
        "by_split": dict(Counter(row["split"] for row in rows)), "source_id_global_deduplication": True,
        "all_rows_replayed_before_projection": True, "human_gold": False, "new_model_calls": 0}
