#!/usr/bin/env python3
"""Project frozen public64 ASR bytes into bounded weak keyword research inputs.

This reuses the completed native audit. It never imports a recognizer, opens a
WAV decoder, executes saved code, invents word times, or grants CTC admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from speech_asr_consensus import NORMALIZATION, normalized_transcript
from speech_label_admission import PUNCTUATION

SCHEMA = "kws-v4-d01-public-zh64-keyword-snapshot-v1"
MANIFEST = "kws-v4-d01-public-zh64-keyword-package-v1"
CONSUMER = "kws-v4-d01-weak-presence-inputs-v1"
POLICY = "three-complete-native-literal-positive-only-v1"
KEYWORDS = ("你好小窝", "小窝小窝")
PROVIDERS = ("qwen06", "sensevoice", "whisper")
HARD_BOUNDARIES = frozenset("。！？.!?\r\n\v\f\u0085\u2028\u2029")
WEAK_WEIGHT = 0.25
SPLITS = {"Vivian": "train", "Dylan": "train", "Serena": "dev", "Uncle_Fu": "dev"}
FILES = ("snapshot.json", "keyword-evidence.jsonl", "present-evidence.jsonl")
SOURCE_FILES = (
    "research/v4-d01-public-zh64-keywords-v1/project.py",
    "tools/speech_asr_consensus.py", "tools/speech_label_admission.py",
)
INPUTS = {
    "union": {"path": "audits/public64_v2_cross_run.json", "bytes": 174367,
              "sha256": "539425066b70d812690bf435f6199293789b4daeeadbe6ccba1852214502b010"},
    "first": {"path": "artifacts/public-zh64-first-asr-43608230908.zip", "bytes": 42715116,
              "sha256": "e731141401bea5220c5fdb9dd066c044ece7bb860fb615a6cc50768acffe40e6",
              "run_id": "43608230908", "members": 3746},
    "tail": {"path": "artifacts/public-zh64-whisper16-v2-43619056902.zip", "bytes": 18783858,
             "sha256": "a397ac2ea006e098ebd19deb7e5855dc908665393fe0047aa832919ac35af08e",
             "run_id": "43619056902", "members": 1514},
}
SOURCE_ZIP_SHA = "0704e28608f636fa14e8bfa30ebc5ed5b870b9576ec8c80fbe15da9e05033ac5"
SOURCE_POLICY_SHA = "e5bace7a7297e925ccb6e9c6ab90871fe62b145472bb3c625c3b9852cdd4323a"
AUDITS = {
    "first": "db8de962c1ed04ba03e07ede4629009f9a552d76d0691391a773bde208cec9ef",
    "stopped": "f7da1b3e3b55d951beb85fed15e0d90935ca8822f21c382653f2ced329a87978",
    "tail": "e82841ad1cb90b5d0c80fcec6ff14d841a8105f5460a596cb3cee5867bb25fd6",
}
LEGACY_FALSE = (
    "training_admitted", "formal_admission", "optimizer_loss_authorized",
    "human_gold", "natural_negative_qualified", "evaluation_or_production_use",
)
PROTOCOL = {
    "policy": POLICY,
    "normalization": "fixed-punctuation-removal-only-v1",
    "keyword_scope": "clip-level machine weak presence; no acoustic ground truth",
    "positive_rule": "all three complete nonempty native texts contain the full literal keyword",
    "absence_rule": "ASR non-detection never establishes absence",
    "quarantine_rule": "inherited nearphone quarantine overrides every positive vote",
    "extra_text_rule": "retain all text; literal prefix/suffix evidence is not a full transcript target",
    "hard_boundary_characters": sorted(HARD_BOUNDARIES),
    "hard_boundary_rule": "a positive text span may not cross a sentence, line or paragraph boundary",
    "normalization_aliases": [],
    "requested_text_used_as_supervision": False,
    "protocol_declared_after_all_192_saved_outputs": True,
    "allowed_use": "controlled-noncommercial-research-clip-keyword-presence",
    "weak_loss_weight": WEAK_WEIGHT,
    "weight_is_estimated_label_accuracy": False,
    "train_role": "masked positive-only keyword loss within a separately declared research recipe",
    "dev_role": "masked weak-presence development checks; never optimizer or independent evaluation",
    "positive_supplement_only": True,
    "standalone_training_corpus_qualified": False,
    "ctc_allowed": False,
    "event_timing_supervision_allowed": False,
    "formal_silver_admission_completed": False,
    "independent_test_allowed": False,
    "human_gold": False,
    "new_model_calls": 0,
    "new_training_runs": 0,
    "new_audio_transformations": 0,
    "native_audit_repeated": False,
}


def require(condition, message):
    if not condition:
        raise ValueError("D01_" + message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record(raw):
    return {"bytes": len(raw), "sha256": sha(raw)}


def loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError("D01_NONFINITE_JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def read(path, limit=64 * 1024 * 1024):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "SYMLINK_INPUT")
    require(path.is_file() and 0 < path.stat().st_size <= limit, "REGULAR_BOUNDED_INPUT")
    raw = path.read_bytes()
    require(0 < len(raw) <= limit, "INPUT_SIZE")
    return raw


def source_identity():
    return {name: record(read(ROOT / name)) for name in SOURCE_FILES}


class SavedArchive:
    """Only JSON/index metadata are read; original audio stays in its saved ZIP."""
    def __init__(self, raw, expected):
        require(record(raw) == {k: expected[k] for k in ("bytes", "sha256")}, "ARCHIVE_PIN")
        self.expected = expected
        self.zip = zipfile.ZipFile(io.BytesIO(raw))
        names = self.zip.namelist()
        self.index = loads(self.zip.read("artifact-index.json"))["files"]
        require(len(names) == len(set(names)) == expected["members"] and
                set(names) == set(self.index) | {"artifact-index.json"}, "SAVED_MEMBER_ROSTER")

    def get(self, name):
        require(name in self.index, "SAVED_MEMBER_MISSING")
        raw = self.zip.read(name)
        require(record(raw) == self.index[name], "SAVED_MEMBER_BINDING")
        return raw

    def json(self, name):
        return loads(self.get(name))

    def ref(self, name, expected_sha=None):
        require(name in self.index, "EVIDENCE_MEMBER_MISSING")
        pin = self.index[name]
        require(expected_sha is None or pin["sha256"] == expected_sha, "EVIDENCE_REFERENCE_SHA")
        return {"artifact_sha256": self.expected["sha256"], "member": name, **pin}


def literal_spans(text, keyword):
    """Offsets refer only to saved Unicode text, never samples or word times."""
    require(type(text) is str and len(text) <= 65536, "TEXT_TYPE_SIZE")
    require(keyword in KEYWORDS, "KEYWORD_IDENTITY")
    kept = [i for i, char in enumerate(text) if char not in PUNCTUATION]
    normalized = normalized_transcript(text)
    require(normalized == "".join(text[i] for i in kept), "NORMALIZER_MAPPING")
    spans, offset = [], 0
    while True:
        start = normalized.find(keyword, offset)
        if start < 0:
            break
        end = start + len(keyword)
        boundary_indices = [i for i in range(kept[start], kept[end - 1] + 1)
                            if text[i] in HARD_BOUNDARIES]
        spans.append({"normalized_start_character": start, "normalized_end_character": end,
                      "raw_character_indices": kept[start:end],
                      "raw_start_character": kept[start], "raw_end_character": kept[end - 1] + 1,
                      "raw_hard_boundary_character_indices": boundary_indices,
                      "eligible_literal_span": not boundary_indices,
                      "retained_normalized_prefix": normalized[:start],
                      "retained_normalized_suffix": normalized[end:]})
        offset = start + 1
    return normalized, spans


def project_keyword(texts, complete, *, keyword_id, quarantine, split):
    """Pure projection for verified saved inputs; fixtures do not gain authority."""
    require(type(texts) is dict and set(texts) == set(PROVIDERS), "EXACT_THREE_TEXTS")
    require(type(complete) is dict and set(complete) == set(PROVIDERS) and
            all(type(v) is bool for v in complete.values()), "EXACT_THREE_COMPLETION_FLAGS")
    require(type(keyword_id) is int and keyword_id in (1, 2), "KEYWORD_ID")
    require(type(quarantine) is bool and split in ("train", "dev"), "QUARANTINE_ROLE")
    keyword = KEYWORDS[keyword_id - 1]
    matches, normalized = {}, {}
    for family in PROVIDERS:
        normalized[family], matches[family] = literal_spans(texts[family], keyword)
    votes = {family: any(span["eligible_literal_span"] for span in matches[family]) for family in PROVIDERS}
    clean = all(complete.values()) and all(normalized.values())
    if quarantine:
        state, reason = "unknown", "inherited_nearphone_quarantine"
    elif not clean:
        state, reason = "unknown", "three_complete_nonempty_native_outputs_required"
    elif all(votes.values()):
        state, reason = "present", "three_complete_literal_full_keyword_votes"
    elif any(votes.values()):
        state, reason = "unknown", "keyword_presence_disagreement"
    else:
        state, reason = "unknown", "non_detection_does_not_establish_absence"
    present = state == "present"
    return {"keyword_id": keyword_id, "keyword": keyword, "state": state, "reason": reason,
            "evidence_grade": "machine_weak_present" if present else "uncertain",
            "literal_votes": votes, "text_matches": matches,
            "weak_target": 1 if present else None, "weak_mask": present,
            "weak_weight": WEAK_WEIGHT if present else 0.0,
            "research_keyword_presence_eligible": present,
            "research_optimizer_eligible": present and split == "train",
            "research_development_eligible": present and split == "dev",
            "allowed_use": ("research-clip-keyword-presence-loss" if split == "train" else
                            "research-clip-keyword-presence-development") if present else "saved-evidence-analysis",
            "acoustic_ground_truth_state": "unknown", "event_times": None,
            "event_timing_supervision_allowed": False, "ctc_allowed": False}


def _map(rows, key):
    require(type(rows) is list and all(type(r) is dict for r in rows), "ROW_OBJECTS")
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), "DUPLICATE_ROW_ID")
    return result


def load_saved(delivery):
    delivery = Path(delivery)
    raw = {name: read(delivery / pin["path"]) for name, pin in INPUTS.items()}
    for name, pin in INPUTS.items():
        require(record(raw[name]) == {k: pin[k] for k in ("bytes", "sha256")}, "PINNED_INPUT_" + name)
    union = loads(raw["union"])
    require(union["audit_passed"] is True and union["all_192_native_observations_complete"] is True and
            union["actual_complete_native_observations"] == 192 and union["unresolved_slots"] == 0,
            "UPSTREAM_COMPLETE_AUDIT")
    require(all(union.get(k) is False for k in LEGACY_FALSE), "HISTORICAL_BOUNDARY")
    require([union[k] for k in ("original_native_audit_sha256", "second_native_audit_sha256",
            "third_native_audit_sha256")] == [AUDITS[k] for k in ("first", "stopped", "tail")],
            "UPSTREAM_NATIVE_AUDITS")
    first, tail = (SavedArchive(raw[k], INPUTS[k]) for k in ("first", "tail"))
    original = first.get("source/original-public64.zip")
    require(sha(original) == SOURCE_ZIP_SHA, "ORIGINAL_SOURCE_ARCHIVE")
    with zipfile.ZipFile(io.BytesIO(original)) as source_zip:
        policy_raw = source_zip.read("policy.json")
    require(sha(policy_raw) == SOURCE_POLICY_SHA, "ORIGINAL_SOURCE_POLICY")
    source_policy = loads(policy_raw)
    require(source_policy["use"] == "controlled-noncommercial-research" and
            source_policy["requested_text_is_observed_transcript"] is False and
            source_policy["training_admitted"] is False, "SOURCE_RESEARCH_PURPOSE")
    require(NORMALIZATION == PROTOCOL["normalization"], "NATIVE_NORMALIZER_VERSION")
    for archive in (first, tail):
        policy = archive.json("policy/recognizer-policy.json")
        require(policy["normalization"] == NORMALIZATION and policy["target_conditioning_allowed"] is False,
                "SAVED_NATIVE_POLICY")
    source = first.json("derived/source-map.json")
    require(source["source_artifact_sha256"] == SOURCE_ZIP_SHA and source["source_policy_preserved"] is True,
            "ORIGINAL_TO_DERIVED_SOURCE")
    sources = _map(source["rows"], "audio_id")
    locks = _map(first.json("derived/input-lock.json")["clips"], "audio_id")
    compared = _map(union["comparison"]["rows"], "audio_id")
    expected_ids = {f"clip-{i:06d}" for i in range(1, 65)}
    require(set(sources) == set(locks) == set(compared) == expected_ids, "EXACT_64_SOURCE_ROWS")
    combined = tail.json("combined-observations.json")
    require(combined["lineage_run_ids"] == ["43608230908", "43614492600", "43619056902"] and
            combined["middle_run_audio_calls"] == 0 and combined["parent_observations_policy_preserved"] is True,
            "THREE_RUN_LINEAGE")
    require(set(combined["providers"]) == set(PROVIDERS), "PROVIDER_ROSTER")
    observations = {p: _map(combined["providers"][p], "audio_id") for p in PROVIDERS}
    require(all(set(v) == expected_ids for v in observations.values()), "EXACT_192_OBSERVATIONS")
    return union, first, tail, source_policy, sources, locks, compared, observations


def build_snapshot(delivery):
    union, first, tail, source_policy, sources, locks, compared, observations = load_saved(delivery)
    rows, identities = [], set()
    for audio_id in sorted(sources):
        source, lock, old = sources[audio_id], locks[audio_id], compared[audio_id]
        split, quarantine = source["research_split"], source["nearphone_quarantined"]
        require(SPLITS[source["preset"]] == split and type(quarantine) is bool, "FROZEN_PRESET_ROLE")
        require(source["original_intended_split"] == "train_only" and
                source["training_admitted"] is False and source["optimizer_loss_authorized"] is False,
                "ORIGINAL_ROLE_AND_AUTHORITY")
        for key in ("producer_ordinal", "preset", "research_split", "source_cell_id", "native_wav_sha256",
                    "native_pcm_sha256", "derived_wav_sha256", "derived_pcm_sha256", "nearphone_quarantined"):
            require(source[key] == old[key] and type(source[key]) is type(old[key]), "SOURCE_REPORT_JOIN_" + key)
        require(lock["wav_sha256"] == source["derived_wav_sha256"] and
                lock["pcm_sha256"] == source["derived_pcm_sha256"] and
                lock["frames"] == source["derived_frames"] and
                lock["source_id"] == "public64-" + source["native_pcm_sha256"] and
                lock["request_id"] == "zh64-" + lock["wav_sha256"], "EXACT_AUDIO_REQUEST_JOIN")
        require(source["native_pcm_sha256"] not in identities, "DUPLICATE_SOURCE_PCM")
        identities.add(source["native_pcm_sha256"])
        evidence = {}
        for family in PROVIDERS:
            observed = observations[family][audio_id]
            require(observed["status"] == old["provider_states"][family] == "complete_native_observation" and
                    observed["attempt_consumed"] is True and observed["actual_text"] == old["native_text"][family],
                    "SAVED_NATIVE_TEXT_JOIN")
            require(all(observed[k] == lock[k] for k in ("wav_sha256", "pcm_sha256", "frames", "source_id", "request_id")),
                    "SAVED_NATIVE_INPUT_JOIN")
            lineage = old["provider_lineage"][family]
            archive = tail if family == "whisper" and int(audio_id[-6:]) > 48 else first
            require(lineage["run_id"] == archive.expected["run_id"] and
                    lineage["policy_sha256"] == archive.index["policy/recognizer-policy.json"]["sha256"],
                    "SAVED_NATIVE_RUN_POLICY")
            batch = (int(audio_id[-6:]) - 1) // 16
            base = f"inference/batch-{batch}/{family}/output/{observed['native_receipt']}"
            receipts = {key: archive.ref(base + "/" + filename, observed[key + "_sha256"])
                        for key, filename in (("started", "started.json"), ("completed", "completed.json"),
                                              ("raw_output", "raw-output.json"))}
            evidence[family] = {**observed, "origin": dict(lineage), "receipt_members": receipts,
                                "audit_result_reused": True}
        texts = {p: evidence[p]["actual_text"] for p in PROVIDERS}
        keywords = [project_keyword(texts, dict.fromkeys(PROVIDERS, True), keyword_id=k,
                                   quarantine=quarantine, split=split) for k in (1, 2)]
        rows.append({"audio_id": audio_id, "source_id": lock["source_id"],
                     "global_source_key": [SOURCE_ZIP_SHA, source["source_cell_id"]],
                     "source_identity": dict(source), "research_split": split,
                     "split_group": "frozen-public64-preset:" + source["preset"],
                     "real_speaker_independence": "not_established",
                     "qwen_tts_and_qwen_asr_are_independent_ground_truth": False,
                     "audio": {**first.ref("derived/audio/" + lock["wav_sha256"] + ".wav", lock["wav_sha256"]),
                               "pcm_sha256": lock["pcm_sha256"], "sample_rate_hz": 16000,
                               "channels": 1, "frames": lock["frames"]},
                     "provider_evidence": evidence, "keywords": keywords,
                     "historical_fulltext_comparison": {
                         "three_way_agreement": old["three_of_three_complete_nonempty_fulltext_agreement"],
                         "state": old["comparison_state"],
                         "normalization": union["comparison"]["normalization"]},
                     "historical_authority": {k: old[k] for k in LEGACY_FALSE},
                     "label_provenance": {"status": "PENDING_EXTERNAL_VERIFICATION",
                                          "scope": "formal admission remains pending; new research protocol is separate"},
                     "training_admitted": False, "formal_admission": False, "human_gold": False,
                     "independent_test": False, "ctc_mask": False, "target_ids": None,
                     "event_times": None})
    flat = flatten(rows)
    states = Counter(r["state"] for r in flat)
    by_keyword = {str(k): {state: sum(r["keyword_id"] == k and r["state"] == state for r in flat)
                          for state in ("present", "absent", "unknown")} for k in (1, 2)}
    summary = {"source_rows": len(rows), "keyword_slots": len(flat),
               "states": {state: states[state] for state in ("present", "absent", "unknown")},
               "by_keyword": by_keyword,
               "research_train_present_slots": sum(r["research_optimizer_eligible"] for r in flat),
               "research_dev_present_slots": sum(r["research_development_eligible"] for r in flat),
               "nearphone_quarantined_rows": sum(r["source_identity"]["nearphone_quarantined"] for r in rows),
               "nearphone_quarantined_keyword_slots": sum(r["nearphone_quarantined"] for r in flat),
               "new_present_from_fulltext_disagreement": [r["audio_id"] for r in rows
                    if not r["historical_fulltext_comparison"]["three_way_agreement"] and
                    any(k["state"] == "present" for k in r["keywords"])],
               "research_split_rows": dict(Counter(r["research_split"] for r in rows)),
               "event_times_known": 0, "formal_admitted_rows": 0}
    return {"schema": SCHEMA, "protocol": dict(PROTOCOL), "inputs": INPUTS,
            "producer_source_files": source_identity(), "reused_native_audit_sha256": AUDITS,
            "original_source_policy": source_policy, "original_source_policy_sha256": SOURCE_POLICY_SHA,
            "historical_authority": {k: union[k] for k in LEGACY_FALSE},
            "training_admitted": False, "formal_admission": False, "human_gold": False,
            "independent_test": False, "summary": summary, "rows": rows}


def flatten(rows):
    return [{"audio_id": row["audio_id"], "source_id": row["source_id"],
             "global_source_key": row["global_source_key"], "research_split": row["research_split"],
             "preset": row["source_identity"]["preset"],
             "nearphone_quarantined": row["source_identity"]["nearphone_quarantined"],
             "audio": row["audio"], "native_evidence_row_sha256": sha(canonical(row["provider_evidence"])),
             "historical_authority": row["historical_authority"], **keyword}
            for row in rows for keyword in row["keywords"]]


def payloads(snapshot):
    flat = flatten(snapshot["rows"])
    return {"snapshot.json": canonical(snapshot) + b"\n",
            "keyword-evidence.jsonl": b"".join(canonical(r) + b"\n" for r in flat),
            "present-evidence.jsonl": b"".join(canonical(r) + b"\n" for r in flat if r["state"] == "present")}


def create(delivery, output):
    snapshot = build_snapshot(delivery)
    output = Path(output).absolute()
    require(not output.exists() and output.parent.is_dir() and
            not output.is_relative_to(Path(delivery).absolute()) and not output.is_relative_to(ROOT),
            "NEW_OUTPUT_OUTSIDE_SOURCES")
    require(not any(p.is_symlink() for p in output.parents), "OUTPUT_SYMLINK_PARENT")
    contents = payloads(snapshot)
    manifest = {"schema": MANIFEST, "policy": POLICY, "inputs": INPUTS,
                "producer_source_files": source_identity(),
                "files": {name: record(raw) for name, raw in contents.items()}}
    output.mkdir(mode=0o700)
    for name, raw in {**contents, "MANIFEST.json": canonical(manifest) + b"\n"}.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    return {"status": "created", "manifest_sha256": sha(read(output / "MANIFEST.json")),
            "summary": snapshot["summary"]}


def verify(delivery, package, expected_manifest_sha256):
    package = Path(package)
    raw = read(package / "MANIFEST.json")
    require(sha(raw) == expected_manifest_sha256, "PACKAGE_MANIFEST_PIN")
    manifest = loads(raw)
    require(set(manifest) == {"schema", "policy", "inputs", "producer_source_files", "files"} and
            manifest["schema"] == MANIFEST and manifest["policy"] == POLICY and
            manifest["inputs"] == INPUTS and manifest["producer_source_files"] == source_identity(),
            "PACKAGE_PROTOCOL_SOURCE_IDENTITY")
    require(set(manifest["files"]) == set(FILES) and
            {p.name for p in package.iterdir()} == set(FILES) | {"MANIFEST.json"}, "PACKAGE_FILE_ROSTER")
    expected = build_snapshot(delivery)
    for name, derived in payloads(expected).items():
        actual = read(package / name)
        require(record(actual) == manifest["files"][name] and actual == derived, "PACKAGE_NOT_SAVED_PROJECTION")
    return expected


def research_inputs(snapshot, phase):
    """Consume a verified snapshot as weak clip-presence data, never CTC/event times.

    Call through consume() for an external package. This pure function is also
    used by fixtures; its inputs alone cannot grant data or model authority.
    """
    require(phase in ("train", "development"), "RESEARCH_PHASE_ONLY")
    require(snapshot.get("schema") == SCHEMA and canonical(snapshot.get("protocol")) == canonical(PROTOCOL),
            "CONSUMER_PROTOCOL")
    selected, seen = [], set()
    split = "train" if phase == "train" else "dev"
    for row in snapshot["rows"]:
        require(row["audio_id"] not in seen, "CONSUMER_DUPLICATE_SOURCE")
        seen.add(row["audio_id"])
        require(row["research_split"] == SPLITS[row["source_identity"]["preset"]], "CONSUMER_ROLE")
        require(all(row["historical_authority"].get(k) is False for k in LEGACY_FALSE) and
                row["training_admitted"] is False and row["formal_admission"] is False and
                row["human_gold"] is False and row["independent_test"] is False and
                row["event_times"] is None and row["ctc_mask"] is False and row["target_ids"] is None,
                "CONSUMER_NO_FORMAL_CTC_TIMING_PROMOTION")
        texts = {p: row["provider_evidence"][p]["actual_text"] for p in PROVIDERS}
        complete = {p: row["provider_evidence"][p]["status"] == "complete_native_observation" for p in PROVIDERS}
        projected = [project_keyword(texts, complete, keyword_id=k,
                     quarantine=row["source_identity"]["nearphone_quarantined"],
                     split=row["research_split"]) for k in (1, 2)]
        require(canonical(row["keywords"]) == canonical(projected), "CONSUMER_KEYWORD_LABEL_CHANGED")
        if row["research_split"] != split or not any(k["weak_mask"] for k in projected):
            continue
        selected.append({"consumer": CONSUMER, "source_id": row["source_id"], "audio_id": row["audio_id"],
                         "global_source_key": row["global_source_key"], "audio": row["audio"],
                         "research_split": split, "phase": phase,
                         "allowed_use": "research-clip-keyword-presence-" + ("loss" if phase == "train" else "development"),
                         "keyword_targets": [k["weak_target"] for k in projected],
                         "keyword_mask": [k["weak_mask"] for k in projected],
                         "keyword_weight": [k["weak_weight"] for k in projected],
                         "research_optimizer_allowed": phase == "train",
                         "human_gold": False, "independent_test": False,
                         "training_admitted": False, "formal_admission": False,
                         "ctc_mask": False, "ctc_weight": 0.0, "target_ids": None,
                         "full_text": None, "event_times": None, "event_timing_supervision_allowed": False,
                         "label_provenance": row["label_provenance"],
                         "historical_authority": row["historical_authority"],
                         "evidence_row_sha256": sha(canonical(row))})
    return {"schema": CONSUMER, "phase": phase, "policy": POLICY,
            "positive_supplement_only": True, "standalone_training_corpus_qualified": False,
            "new_training_runs": 0, "rows": selected,
            "selected_rows": len(selected), "known_positive_keyword_slots": sum(sum(r["keyword_mask"]) for r in selected)}


def consume(delivery, package, expected_manifest_sha256, phase):
    result = research_inputs(verify(delivery, package, expected_manifest_sha256), phase)
    result["snapshot_manifest_sha256"] = expected_manifest_sha256
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify", "consume"))
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--phase", choices=("train", "development"))
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            require(args.output is not None and args.package is None and args.manifest_sha256 is None and
                    args.phase is None, "CREATE_ARGUMENTS")
            result = create(args.delivery, args.output)
        else:
            require(args.output is None and args.package is not None and args.manifest_sha256 is not None,
                    "VERIFY_ARGUMENTS")
            if args.command == "verify":
                require(args.phase is None, "VERIFY_PHASE")
                result = {"status": "verified", "summary": verify(args.delivery, args.package, args.manifest_sha256)["summary"]}
            else:
                require(args.phase is not None, "CONSUME_PHASE")
                result = consume(args.delivery, args.package, args.manifest_sha256, args.phase)
        print(canonical(result).decode("utf-8"))
        return 0
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        print("D01_REJECTED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
