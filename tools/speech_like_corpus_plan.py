#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from attach_speech_like_labels import normalize_label  # noqa: E402
from corpus_identity import inspect_pcm16_wav, rebind_audio_path  # noqa: E402
from source_identity import source_identity_namespace  # noqa: E402
from speech_label_admission import (REAL_MODE, FIXTURE_MODE, admission_from_reviews,
                                    fixture_admission, latest_review)  # noqa: E402
from source_lineage_admission import LINEAGE_FIELDS, validate_review_record, validate_admission  # noqa: E402

from speech_asr_consensus import (admission_from_consensus, validate_consensus_admission,
                                  validate_consensus_record, require_training_label_authority,
                                  has_machine_consensus, LABEL_SOURCE)
from asr_source_resolver import source_verification_session, resolve_source_context

POLICY = "speech-like-corpus-plan-v1"
VOICE_CLASS = "speech-like-provider-voice-slot-v1"
INTENT_CLASS = "speech-like-request-label-intent-v1"
MANIFEST_CLASS = "speech-like-synthetic-recording-v1"
LABEL_CLASS = "speech-like-training-label-v1"
SPLITS = ("train", "calibration", "test", "qualification")
ALLOWED_KINDS = {"positive", "confusable", "negative"}
ALLOWED_PROVIDER_GROUPS = {"train", "generalization-search", "generalization-freeze"}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _jsonl_text(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_jsonl_text(rows), encoding="utf-8")


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def normalize_plan(path: pathlib.Path) -> dict:
    value = load_object(path)
    if int(value.get("schema_version", 0)) != 1 or value.get("policy") != POLICY:
        raise ValueError("speech-like corpus plan identity mismatch")
    if str(value.get("locale", "")).lower() not in {"zh-cn", "zh_cn", "cmn-hans-cn"}:
        raise ValueError("speech-like corpus plan must be Mandarin zh-CN")
    roles = value.get("split_roles")
    if not isinstance(roles, dict) or set(roles) != set(SPLITS):
        raise ValueError("split_roles must define train/calibration/test/qualification")
    seen_slots: set[str] = set()
    normalized_roles: dict[str, dict] = {}
    for split in SPLITS:
        item = roles[split]
        if not isinstance(item, dict):
            raise ValueError(f"split role {split} must be an object")
        group = require_text(item.get("provider_group"), f"split role {split}.provider_group")
        if group not in ALLOWED_PROVIDER_GROUPS:
            raise ValueError(f"split role {split}: unsupported provider group {group}")
        slots = item.get("voice_slots")
        if not isinstance(slots, list) or not slots:
            raise ValueError(f"split role {split}.voice_slots must be non-empty")
        normalized_slots: list[str] = []
        for raw in slots:
            slot = require_text(raw, f"split role {split}.voice slot")
            if slot in seen_slots:
                raise ValueError(f"voice slot appears in multiple splits: {slot}")
            seen_slots.add(slot)
            normalized_slots.append(slot)
        normalized_roles[split] = {"provider_group": group, "voice_slots": normalized_slots}

    utterances = value.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        raise ValueError("utterances must be non-empty")
    ids: set[str] = set()
    normalized_utterances: list[dict] = []
    positive_keywords: set[int] = set()
    for index, row in enumerate(utterances):
        if not isinstance(row, dict):
            raise ValueError(f"utterance {index} must be an object")
        utterance_id = require_text(row.get("id"), f"utterance {index}.id")
        if utterance_id in ids:
            raise ValueError(f"duplicate utterance id: {utterance_id}")
        ids.add(utterance_id)
        kind = require_text(row.get("kind"), f"utterance {utterance_id}.kind")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"utterance {utterance_id}: unsupported kind {kind}")
        text = require_text(row.get("text"), f"utterance {utterance_id}.text")
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(not isinstance(token, str) or not token for token in tokens):
            raise ValueError(f"utterance {utterance_id}.tokens must be non-empty strings")
        keyword_id = row.get("keyword_id")
        if kind == "positive":
            if isinstance(keyword_id, bool) or not isinstance(keyword_id, int) or keyword_id <= 0:
                raise ValueError(f"positive utterance {utterance_id} requires positive integer keyword_id")
            positive_keywords.add(keyword_id)
        elif keyword_id is not None:
            raise ValueError(f"non-positive utterance {utterance_id} must use keyword_id=null")
        normalized_utterances.append(
            {
                "id": utterance_id,
                "kind": kind,
                "keyword_id": keyword_id,
                "text": text,
                "tokens": list(tokens),
            }
        )
    if len(positive_keywords) < 2:
        raise ValueError("corpus plan must cover at least two positive keyword ids")

    constraints = value.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError("speech-like corpus plan constraints must be an object")
    pause_separator = require_text(
        constraints.get("deterministic_pause_separator"),
        "constraints.deterministic_pause_separator",
    )
    if pause_separator != "。":
        raise ValueError(
            "deterministic pause separator must be the pinned sherpa-safe full stop '。'"
        )
    by_id = {row["id"]: row for row in normalized_utterances}
    pause_ids = sorted(row_id for row_id in by_id if row_id.endswith("-pause"))
    if not pause_ids:
        raise ValueError("corpus plan must define positive deterministic pause variants")
    for pause_id in pause_ids:
        exact_id = pause_id.removesuffix("-pause") + "-exact"
        if exact_id not in by_id:
            raise ValueError(f"{pause_id}: matching exact utterance is missing")
        pause = by_id[pause_id]
        exact = by_id[exact_id]
        if pause["kind"] != "positive" or exact["kind"] != "positive":
            raise ValueError(f"{pause_id}: exact/pause pair must both be positive")
        if pause["keyword_id"] != exact["keyword_id"] or pause["tokens"] != exact["tokens"]:
            raise ValueError(f"{pause_id}: exact/pause pair keyword/tokens drifted")
        if pause["text"].count(pause_separator) != 1:
            raise ValueError(
                f"{pause_id}: pause text must contain exactly one deterministic separator"
            )
        if pause["text"].replace(pause_separator, "") != exact["text"]:
            raise ValueError(
                f"{pause_id}: pause text must derive from exact text only by the separator"
            )

    return {
        "roles": normalized_roles,
        "utterances": normalized_utterances,
        "deterministic_pause_separator": pause_separator,
        "plan_sha256": sha256_file(path),
    }


def normalize_inventory(path: pathlib.Path, expected_slots: set[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    voice_owner: dict[str, str] = {}
    for index, row in enumerate(load_jsonl(path), 1):
        if int(row.get("schema_version", 0)) != 1 or row.get("evidence_class") != VOICE_CLASS:
            raise ValueError(f"{path}:{index}: voice inventory identity mismatch")
        slot = require_text(row.get("slot"), f"{path}:{index}.slot")
        voice_id = require_text(row.get("voice_id"), f"{path}:{index}.voice_id")
        if slot in result:
            raise ValueError(f"duplicate voice slot in inventory: {slot}")
        previous = voice_owner.get(voice_id)
        if previous is not None:
            raise ValueError(f"voice_id {voice_id} is reused by slots {previous}/{slot}")
        parameters = row.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"{path}:{index}.parameters must be an object")
        normalized_parameters: dict[str, str | int | float] = {}
        for key, value in parameters.items():
            name = require_text(key, f"{path}:{index}.parameter key")
            if name in {"executable", "text", "output"} or name.startswith("asset:"):
                raise ValueError(f"voice inventory uses reserved provider parameter: {name}")
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError(f"voice inventory parameter {name} must be string/number")
            normalized_parameters[name] = value
        result[slot] = {"voice_id": voice_id, "parameters": normalized_parameters}
        voice_owner[voice_id] = slot
    if set(result) != expected_slots:
        missing = sorted(expected_slots - set(result))
        extra = sorted(set(result) - expected_slots)
        raise ValueError(f"voice inventory must map exactly the plan slots; missing={missing} extra={extra}")
    return result


def build_requests(plan_path: pathlib.Path, inventory_path: pathlib.Path) -> tuple[list[dict], list[dict], dict]:
    plan = normalize_plan(plan_path)
    expected_slots = {
        slot
        for role in plan["roles"].values()
        for slot in role["voice_slots"]
    }
    inventory = normalize_inventory(inventory_path, expected_slots)
    requests: list[dict] = []
    intents: list[dict] = []
    split_counts: dict[str, int] = {split: 0 for split in SPLITS}
    group_counts: dict[str, int] = {group: 0 for group in ALLOWED_PROVIDER_GROUPS}
    seen_sources: set[str] = set()
    for split in SPLITS:
        role = plan["roles"][split]
        group = role["provider_group"]
        for slot in role["voice_slots"]:
            voice = inventory[slot]
            for utterance in plan["utterances"]:
                request_id = f"{split}:{slot}:{utterance['id']}"
                source_id = f"speech-like:{request_id}"
                if source_id in seen_sources:
                    raise ValueError(f"duplicate derived source_id: {source_id}")
                seen_sources.add(source_id)
                requests.append(
                    {
                        "schema_version": 1,
                        "group": group,
                        "text": utterance["text"],
                        "tokens": utterance["tokens"],
                        "voice_id": voice["voice_id"],
                        "source_id": source_id,
                        "parameters": voice["parameters"],
                    }
                )
                intents.append(
                    {
                        "schema_version": 1,
                        "evidence_class": INTENT_CLASS,
                        "request_id": request_id,
                        "provider_group": group,
                        "split": split,
                        "kind": utterance["kind"],
                        "keyword_id": utterance["keyword_id"],
                        "text": utterance["text"],
                        "tokens": utterance["tokens"],
                        "voice_id": voice["voice_id"],
                        "source_id": source_id,
                    }
                )
                split_counts[split] += 1
                group_counts[group] += 1
    summary = {
        "schema_version": 1,
        "evidence_class": "speech-like-corpus-request-plan-v1",
        "policy": POLICY,
        "plan_sha256": plan["plan_sha256"],
        "voice_inventory_sha256": sha256_file(inventory_path),
        "requests": len(requests),
        "split_counts": split_counts,
        "provider_group_counts": group_counts,
        "voice_slots": len(expected_slots),
        "utterances": len(plan["utterances"]),
        "deterministic_pause_separator": plan["deterministic_pause_separator"],
        "protected_evidence_used": False,
    }
    summary["request_set_sha256"] = canonical_sha256(requests)
    summary["intent_set_sha256"] = canonical_sha256(intents)
    return requests, intents, summary


def intent_key(group: str, row: dict) -> tuple[str, str, str, str, tuple[str, ...]]:
    tokens = row.get("tokens")
    if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
        raise ValueError("generated/intended tokens must be a string list")
    return (
        group,
        require_text(row.get("voice_id"), "voice_id"),
        require_text(row.get("source_id"), "source_id"),
        require_text(row.get("text"), "text"),
        tuple(tokens),
    )


def _review_admissions(path: pathlib.Path, intent_map: dict, generated: dict, *, partial: bool = False) -> tuple[dict, dict]:
    """Require complete supplied review histories and bind their latest actual labels."""
    histories: dict[tuple[str, str], list[dict]] = {}
    for row in load_jsonl(path):
        validate_review_record(row)
        histories.setdefault((row["source_id"], row["file_sha256"]), []).append(row)
    expected = {(row["source_id"], row["file_sha256"]) for row in generated.values()}
    if set(histories) - expected or (not partial and set(histories) != expected):
        raise ValueError("audio review contains missing, unplanned or stale recordings")
    admissions = {}
    reviewers = set()
    review_sha = sha256_file(path)
    for key, intent in intent_map.items():
        source = generated[key]
        identity = (source["source_id"], source["file_sha256"])
        if identity not in histories:
            continue
        require_training_label_authority(source)
        require_training_label_authority(intent)
        latest = latest_review(histories[identity])
        if latest.get("intended_text") != intent["text"]:
            raise ValueError(f"audio review intended_text differs from plan: source_id={identity[0]}")
        if latest["pcm_sha256"] != source.get("pcm_sha256"):
            raise ValueError("audio review pcm_sha256 mismatch")
        for field in LINEAGE_FIELDS:
            if field in source and source[field] != latest.get(field):
                raise ValueError(f"audio review {field} differs from source metadata")
        source_admission = source.get("admission")
        if source_admission is not None:
            if not isinstance(source_admission, dict) or source_admission.get("mode") != REAL_MODE:
                raise ValueError("synthetic fixture recordings cannot promote to real training")
        admissions[identity] = admission_from_reviews(histories[identity], review_sha)
        reviewers.add(latest["reviewer_id"])
    return admissions, {"review_sha256": review_sha, "recordings": len(histories),
                        "reviewer_count": len(reviewers), "revision_policy": "latest-complete-supplied-history-v1"}


def validate_audio_review(path: pathlib.Path, intent_map: dict, generated: dict) -> dict:
    return _review_admissions(path, intent_map, generated)[1]


def load_planned_recordings(intents_path: pathlib.Path, generated_root: pathlib.Path, *, planned_groups_only: bool = False) -> tuple[dict, dict, dict]:
    intents = load_jsonl(intents_path)
    intent_map: dict[tuple[str, str, str, str, tuple[str, ...]], dict] = {}
    for index, row in enumerate(intents, 1):
        if int(row.get("schema_version", 0)) != 1 or row.get("evidence_class") != INTENT_CLASS:
            raise ValueError(f"{intents_path}:{index}: label intent identity mismatch")
        split = str(row.get("split", ""))
        group = str(row.get("provider_group", ""))
        kind = str(row.get("kind", ""))
        if split not in SPLITS or group not in ALLOWED_PROVIDER_GROUPS or kind not in ALLOWED_KINDS:
            raise ValueError(f"{intents_path}:{index}: invalid split/group/kind")
        key = intent_key(group, row)
        if key in intent_map:
            raise ValueError(f"duplicate label intent identity: {key}")
        intent_map[key] = row

    generated: dict[tuple[str, str, str, str, tuple[str, ...]], dict] = {}
    source_manifest_sha: dict[str, str] = {}
    groups = {key[0] for key in intent_map} if planned_groups_only else ALLOWED_PROVIDER_GROUPS
    for group in sorted(groups):
        manifest = generated_root / group / "manifest.jsonl"
        if not manifest.is_file():
            raise ValueError(f"generated provider manifest is missing: {manifest}")
        source_manifest_sha[group] = sha256_file(manifest)
        for index, row in enumerate(load_jsonl(manifest), 1):
            if int(row.get("schema_version", 0)) != 1 or row.get("evidence_class") != MANIFEST_CLASS:
                raise ValueError(f"{manifest}:{index}: generated manifest identity mismatch")
            if row.get("provider_kind") == "tone":
                raise ValueError(f"{manifest}:{index}: tone provider is forbidden")
            key = intent_key(group, row)
            if key in generated:
                raise ValueError(f"duplicate generated recording identity: {key}")
            audio_raw = require_text(row.get("audio"), f"{manifest}:{index}.audio")
            audio = pathlib.Path(audio_raw)
            audio = audio.resolve() if audio.is_absolute() else (manifest.parent / audio).resolve()
            normalized = rebind_audio_path(row, str(audio), field="audio", label=f"{manifest}:{index}")
            generated[key] = normalized

    if set(generated) != set(intent_map):
        missing = sorted(set(intent_map) - set(generated))
        extra = sorted(set(generated) - set(intent_map))
        raise ValueError(f"generated corpus does not match planned requests; missing={len(missing)} extra={len(extra)}")

    return intent_map, generated, source_manifest_sha


def inspect_consensus_recordings(intents_path: pathlib.Path, generated_root: pathlib.Path,
                                 receipts_path: pathlib.Path, *, expected_count: int | None = None) -> dict:
    """Read-only eligibility check, never an authenticated training admission."""
    intent_map, generated, manifests = load_planned_recordings(
        intents_path, generated_root, planned_groups_only=True)
    if expected_count is not None and (expected_count < 1 or len(generated) != expected_count):
        raise ValueError("recording count differs from expected count")
    receipts = {}
    for record in load_jsonl(receipts_path):
        validate_consensus_record(record)
        identity = (record["source_id"], record["file_sha256"])
        if identity in receipts:
            raise ValueError("duplicate machine-consensus recording")
        receipts[identity] = record
    if set(receipts) != {(row["source_id"], row["file_sha256"]) for row in generated.values()}:
        raise ValueError("machine consensus contains missing, unplanned or stale recordings")
    seen_pcm, seen_sources = set(), set()
    receipt_sha = sha256_file(receipts_path)
    for key, intent in intent_map.items():
        source = generated[key]
        record = receipts[(source["source_id"], source["file_sha256"])]
        if intent["split"] != "train" or intent["provider_group"] != "train":
            raise ValueError("machine consensus only labels train-role source audio")
        for metadata in (source, intent):
            for field in ("role", "allowed_role", "split_role", "source_role", "split"):
                if field in metadata and metadata[field] != "train":
                    raise ValueError(f"machine consensus cannot reclassify source/intent {field}")
            if (metadata.get("source_screen_only") is True or metadata.get("human_gold") is True
                    or metadata.get("review_origin") == "human"):
                raise ValueError("machine consensus cannot reclassify protected roles or human gold")
        if source.get("synthetic") is not True:
            raise ValueError("this synthetic-source inspector requires explicit synthetic=true")
        if source.get("admission") is not None:
            raise ValueError("machine consensus cannot relabel an already admitted/fixture/historical source")
        inspected = inspect_pcm16_wav(pathlib.Path(source["audio"]))
        for field in ("file_sha256", "pcm_sha256"):
            if source.get(field) != inspected[field] or record[field] != inspected[field]:
                raise ValueError(f"machine consensus {field} differs from measured WAV")
        if source["source_id"] in seen_sources or inspected["pcm_sha256"] in seen_pcm:
            raise ValueError("duplicate machine-consensus source/decoded PCM")
        seen_sources.add(source["source_id"]); seen_pcm.add(inspected["pcm_sha256"])
        if record["intended_text"] != intent["text"]:
            raise ValueError("machine consensus intended text differs from original plan")
        for field in LINEAGE_FIELDS:
            if source.get(field) != record.get(field):
                raise ValueError(f"machine consensus {field} differs from source provenance")
        for field, value in record["source_provenance"].items():
            if source.get(field) != value or type(source.get(field)) is not type(value):
                raise ValueError(f"machine consensus {field} differs from source origin/rights/role")
        admission = admission_from_consensus(record, receipt_sha)
        validate_consensus_admission(admission, split=intent["split"], frames=inspected["frames"])
    return {"schema_version": 1, "evidence_class": "speech-three-asr-eligibility-inspection-v1",
            "status": "PENDING_EXTERNAL_VERIFICATION", "label_eligible_recordings": len(receipts),
            "training_admitted_recordings": 0, "ctc_training_allowed": False,
            "qualification_allowed": False, "evaluation_gold_allowed": False,
            "authentication_independently_verified": False,
            "consensus_receipt_sha256": receipt_sha, "intents_sha256": sha256_file(intents_path),
            "source_manifests": manifests,
            "limits": ["Self-authored verified=true and hashes do not authenticate recognizers or outputs.",
                       "Rights references and source roles are checked for consistency, not independently certified.",
                       "Frame coverage does not prove human acoustic completeness or absence of shared ASR error.",
                       "Existing split, rights, coverage, ancestry, resource and independent qualification gates remain required."]}


def _consensus_admissions(path: pathlib.Path, intent_map: dict, generated: dict, *,
                          evidence_dir=None, verification_session=None) -> tuple[dict, dict]:
    """Create machine admissions only after resolving immutable retained bytes."""
    records = {}
    for record in load_jsonl(path):
        validate_consensus_record(record)
        identity = (record["source_id"], record["file_sha256"])
        if identity in records:
            raise ValueError("duplicate machine-consensus recording")
        records[identity] = record
    expected = {(source["source_id"], source["file_sha256"]) for source in generated.values()}
    if set(records) - expected:
        raise ValueError("machine consensus contains unplanned or stale recordings")
    admissions = {}
    receipt_sha = sha256_file(path)
    for key, intent in intent_map.items():
        source = generated[key]
        identity = (source["source_id"], source["file_sha256"])
        if identity not in records:
            continue
        record = records[identity]
        if intent["split"] != "train" or intent["provider_group"] != "train":
            raise ValueError("machine consensus only labels train-role source audio")
        for metadata in (source, intent):
            if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
                raise ValueError("machine source/intent schema_version must be canonical integer 1")
            for field in ("role", "allowed_role", "split_role", "source_role", "split"):
                if field in metadata and metadata[field] != "train":
                    raise ValueError(f"machine consensus cannot reclassify source/intent {field}")
            if (metadata.get("source_screen_only") is True or metadata.get("human_gold") is True
                    or metadata.get("review_origin") == "human"):
                raise ValueError("machine consensus cannot reclassify protected roles or human gold")
            for field, expected_value in (("label_origin", "machine-consensus"), ("label_source", LABEL_SOURCE)):
                if field in metadata and metadata[field] != expected_value:
                    raise ValueError("machine consensus cannot rewrite an existing label origin")
            if metadata.get("admission") is not None:
                raise ValueError("machine consensus cannot relabel an already admitted/fixture/historical source")
        if source.get("synthetic") is not True:
            raise ValueError("speech-like admission requires explicit synthetic=true")
        measured = inspect_pcm16_wav(pathlib.Path(source["audio"]))
        for field in ("file_sha256", "pcm_sha256"):
            if source.get(field) != measured[field] or record[field] != measured[field]:
                raise ValueError(f"machine consensus {field} differs from measured WAV")
        if "frames" in source and (type(source["frames"]) is not int or source["frames"] != measured["frames"]):
            raise ValueError("machine source frames differ from measured WAV")
        if record["intended_text"] != intent["text"]:
            raise ValueError("machine consensus intended text differs from original plan")
        if "actual_text" in source:
            raise ValueError("already labeled source requires explicit source-bound supersession; no in-place relabeling")
        for field in LINEAGE_FIELDS:
            if source.get(field) != record.get(field):
                raise ValueError(f"machine consensus {field} differs from source provenance")
        for field, value in record["source_provenance"].items():
            if source.get(field) != value or type(source.get(field)) is not type(value):
                raise ValueError(f"machine consensus {field} differs from source origin/rights/role")
            if field in intent and (intent[field] != value or type(intent[field]) is not type(value)):
                raise ValueError(f"machine consensus {field} differs from intent origin/rights/role")
        context = resolve_source_context(record, evidence_dir=evidence_dir, session=verification_session)
        admission = admission_from_consensus(record, receipt_sha, verification_context=context)
        candidate = {**source, "admission": admission, "split": intent["split"]}
        require_training_label_authority(candidate, allow_machine=True, split=intent["split"],
                                         frames=measured["frames"], evidence_dir=evidence_dir,
                                         verification_session=verification_session)
        validate_admission(admission, mode=REAL_MODE, source_id=source["source_id"],
                           file_sha256=measured["file_sha256"], pcm_sha256=measured["pcm_sha256"],
                           text=record["actual_text"], tokens=record["actual_tokens"],
                           kind=record["kind"], keyword_id=record["keyword_id"], provenance=source,
                           split=intent["split"], frames=measured["frames"], evidence_dir=evidence_dir,
                           verification_session=verification_session)
        admissions[identity] = admission
    return admissions, {"consensus_receipt_sha256": receipt_sha, "recordings": len(admissions),
                        "label_origin": "machine-consensus", "qualification_allowed": False,
                        "evaluation_gold_allowed": False}


def _admitted_recordings(intents_path, intent_map, generated, manifests, *, audio_review=None,
                         asr_consensus=None, evidence_dir=None, verification_session=None):
    if audio_review is None and asr_consensus is None:
        raise ValueError("mandatory human audio-review v2 or trusted three-ASR consensus is required for future training admission")
    human, review = (_review_admissions(audio_review, intent_map, generated, partial=asr_consensus is not None)
                    if audio_review is not None else ({}, None))
    machine, consensus = (_consensus_admissions(asr_consensus, intent_map, generated, evidence_dir=evidence_dir,
                                                verification_session=verification_session)
                          if asr_consensus is not None else ({}, None))
    if set(human) & set(machine):
        raise ValueError("human and machine admission coverage must be disjoint")
    admissions = {**human, **machine}
    if set(admissions) != {(row["source_id"], row["file_sha256"]) for row in generated.values()}:
        raise ValueError("human/machine admissions contain missing, unplanned or stale recordings")
    seen_sources, seen_pcm, voice_splits, lineage_owners, split_counts = set(), set(), {}, {}, {}
    for key, intent in intent_map.items():
        row = generated[key]
        if row.get("synthetic") is not True:
            raise ValueError("reviewed admission requires synthetic=true")
        normalize_label({**intent, "evidence_class": LABEL_CLASS}, "planned label")
        source_id = require_text(row.get("source_id"), "source_id")
        voice_id = require_text(row.get("voice_id"), "voice_id")
        split = intent["split"]
        if source_id in seen_sources:
            raise ValueError("duplicate source_id in admitted batch")
        seen_sources.add(source_id)
        if voice_id in voice_splits and voice_splits[voice_id] != split:
            raise ValueError("voice_id crosses splits")
        voice_splits[voice_id] = split
        measured = inspect_pcm16_wav(pathlib.Path(row["audio"]))
        for field in ("file_sha256", "pcm_sha256"):
            if row.get(field) != measured[field]:
                raise ValueError(f"admitted WAV {field} mismatch")
        if measured["pcm_sha256"] in seen_pcm:
            raise ValueError("duplicate decoded PCM in admitted batch")
        seen_pcm.add(measured["pcm_sha256"])
        admission = admissions[(source_id, row["file_sha256"])]
        receipt = (admission["consensus_record"] if has_machine_consensus(admission)
                   else latest_review(admission["review_history"]))
        for field in ("source_id", *LINEAGE_FIELDS):
            if field in receipt:
                identity = (source_identity_namespace(field), receipt[field])
                previous = lineage_owners.get(identity)
                if previous is not None and previous != split:
                    raise ValueError(f"{field} crosses splits")
                lineage_owners[identity] = split
        split_counts[split] = split_counts.get(split, 0) + 1
    summary = {"schema_version": 2, "evidence_class": "speech-like-reviewed-admission-v2",
               "admission_mode": REAL_MODE, "ctc_training_allowed": True, "allowed_purpose": "ctc-training-only",
               "scope": "internal-development-only", "qualification_allowed": False,
               "intents_sha256": sha256_file(intents_path), "source_manifests": manifests,
               "recordings": len(generated), "split_counts": split_counts,
               "human_review_recordings": len(human), "machine_consensus_recordings": len(machine),
               "generator_family_independence_verified": False,
               "review_history_globally_current_verified": False,
               "limits": ["Receipt binding does not independently prove listening or reviewer identity.",
                          "Supplied history does not prove that no later revision exists in an external review catalog.",
                          "Provider IDs do not prove distinct model families or rights to source audio.",
                          "Machine labels are training-only and never human/evaluation/qualification gold.",
                          "No product qualification or new holdout claim, including legacy qualification split names."]}
    if review is not None:
        summary["audio_review"] = review
    if consensus is not None:
        summary["asr_consensus"] = consensus
    return admissions, summary


def admit_reviewed_recordings(intents_path: pathlib.Path, generated_root: pathlib.Path,
                              audio_review: pathlib.Path | None = None, *, expected_count: int | None = None,
                              asr_consensus: pathlib.Path | None = None, evidence_dir=None) -> dict:
    """Read-only integrity gate for supplied human or trusted-source machine labels."""
    with source_verification_session(evidence_dir) as session:
        intent_map, generated, manifests = load_planned_recordings(
            intents_path, generated_root, planned_groups_only=True)
        if expected_count is not None and (type(expected_count) is not int or expected_count < 1 or len(generated) != expected_count):
            raise ValueError("recording count differs from expected count")
        return _admitted_recordings(intents_path, intent_map, generated, manifests, audio_review=audio_review,
                                    asr_consensus=asr_consensus, evidence_dir=evidence_dir,
                                    verification_session=session)[1]


def materialize_labels(intents_path: pathlib.Path, generated_root: pathlib.Path,
                       output_root: pathlib.Path, *, audio_review: pathlib.Path | None = None,
                       admission_mode: str = REAL_MODE, asr_consensus: pathlib.Path | None = None,
                       evidence_dir=None) -> dict:
    with source_verification_session(evidence_dir) as session:
        summary, payloads, targets = _materialize_labels(
            intents_path, generated_root, output_root, audio_review=audio_review,
            admission_mode=admission_mode, asr_consensus=asr_consensus,
            evidence_dir=evidence_dir, verification_session=session)
    # The context's final strong recheck must succeed before formal admissions
    # are published. No further evidence-dependent validation follows writes.
    created = []
    try:
        _check_materialized_audio(targets)
        for path, payload in payloads:
            with path.open("x", encoding="utf-8") as output:
                created.append(path)
                output.write(payload)
        _check_materialized_audio(targets)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return summary


def _copy_audio_exclusive(source: pathlib.Path, target: pathlib.Path) -> None:
    """Never overwrite a target created after output-path preflight."""
    try:
        with source.open("rb") as original, target.open("xb") as output:
            shutil.copyfileobj(original, output)
    except FileExistsError as exc:
        raise ValueError(f"labeled audio target already exists: {target}") from exc


def _check_materialized_audio(targets: list[tuple[pathlib.Path, str]]) -> None:
    for target, expected in targets:
        if target.is_symlink() or not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"materialized audio identity changed before publication: {target}")


def _materialize_labels(intents_path, generated_root, output_root, *, audio_review,
                        admission_mode, asr_consensus, evidence_dir, verification_session):
    if admission_mode not in {REAL_MODE, FIXTURE_MODE}:
        raise ValueError("new materialization must use reviewed-real-v1 or synthetic-fixture-v1")
    if admission_mode == REAL_MODE and audio_review is None and asr_consensus is None:
        raise ValueError("mandatory human audio-review v2 or trusted three-ASR consensus is required for future training admission")
    if admission_mode == FIXTURE_MODE and (audio_review is not None or asr_consensus is not None):
        raise ValueError("fixture lane cannot claim human review or machine consensus")
    intent_map, generated, source_manifest_sha = load_planned_recordings(
        intents_path, generated_root, planned_groups_only=asr_consensus is not None)
    admissions, admission = ({}, None)
    if admission_mode == REAL_MODE:
        admissions, admission = _admitted_recordings(
            intents_path, intent_map, generated, source_manifest_sha, audio_review=audio_review,
            asr_consensus=asr_consensus, evidence_dir=evidence_dir, verification_session=verification_session)
    split_rows: dict[str, list[dict]] = {split: [] for split in SPLITS}
    split_labels: dict[str, list[dict]] = {split: [] for split in SPLITS}
    split_labeled: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for key, intent in intent_map.items():
        row = dict(generated[key])
        if row.get("synthetic") is not True:
            raise ValueError("speech-like materialization requires explicit synthetic=true")
        measured = inspect_pcm16_wav(pathlib.Path(row["audio"]))
        for field in ("file_sha256", "pcm_sha256"):
            if row.get(field) != measured[field]:
                raise ValueError(f"materialization source {field} mismatch")
        actual_label, label_source = intent, "synthetic-fixture-plan-v1"
        split = str(intent["split"])
        if admission_mode == REAL_MODE:
            row["admission"] = admissions[(row["source_id"], row["file_sha256"])]
            machine = has_machine_consensus(row["admission"])
            receipt = row["admission"]["consensus_record"] if machine else latest_review(row["admission"]["review_history"])
            actual_label = receipt
            row["intended_text"] = row["text"]
            row["intended_tokens"] = row["tokens"]
            row["text"] = row["actual_text"] = receipt["actual_text"]
            row["tokens"] = receipt["actual_tokens"]
            for field in LINEAGE_FIELDS:
                if field in receipt:
                    row[field] = receipt[field]
            if machine:
                row["label_origin"] = "machine-consensus"
                row["split"] = split
                label_source = LABEL_SOURCE
            else:
                label_source = "human-actual-transcript-v2"
        else:
            # Preserve machine tags until this guard, even when the caller
            # explicitly asks to materialize a synthetic diagnostic fixture.
            require_training_label_authority(row)
            require_training_label_authority(intent)
            row["admission"] = fixture_admission()
        label = {"schema_version": 1, "evidence_class": LABEL_CLASS,
                 "voice_id": row["voice_id"], "source_id": row["source_id"], "file_sha256": row["file_sha256"],
                 "kind": actual_label["kind"], "keyword_id": actual_label["keyword_id"], "label_source": label_source}
        label_value = normalize_label(label, "materialized label")
        label_identity = {field: label[field] for field in ("voice_id", "source_id", "file_sha256")}
        label_identity.update(label_value)
        # Validate and prepare attached labels before creating any output.
        labeled = {**row, **label_value, "label_provenance_sha256": canonical_sha256(label_identity)}
        if admission_mode == REAL_MODE:
            require_training_label_authority(labeled, allow_machine=True, split=split, frames=measured["frames"],
                                             evidence_dir=evidence_dir, verification_session=verification_session)
            validate_admission(labeled["admission"], mode=REAL_MODE, source_id=labeled["source_id"],
                               file_sha256=measured["file_sha256"], pcm_sha256=measured["pcm_sha256"],
                               text=labeled["actual_text"], tokens=labeled["tokens"], kind=labeled["kind"],
                               keyword_id=labeled["keyword_id"], provenance=labeled, split=split, frames=measured["frames"],
                               evidence_dir=evidence_dir, verification_session=verification_session)
        split_rows[split].append(row)
        split_labels[split].append(label)
        split_labeled[split].append(labeled)

    # Detect every collision and stale evidence snapshot before the first mkdir,
    # audio copy or manifest write. The outer session rechecks again on exit.
    for split, rows in split_rows.items():
        for name in ("manifest.jsonl", "labels.jsonl", "labeled-manifest.jsonl"):
            target = output_root / split / name
            if target.exists() or target.is_symlink():
                raise ValueError(f"{split}: labeled metadata target already exists: {target}")
        for ordinal, row in enumerate(rows):
            target = output_root / split / "audio" / f"recording-{ordinal:06d}-{row['file_sha256'][:12]}.wav"
            if target.exists():
                raise ValueError(f"{split}: labeled audio target already exists: {target}")
    verification_session.checkpoint()
    split_summary: dict[str, dict] = {}
    payloads: list[tuple[pathlib.Path, str]] = []
    portable_splits: dict[str, list[dict]] = {}
    targets: list[tuple[pathlib.Path, str]] = []
    for split in SPLITS:
        split_root = output_root / split
        audio_dir = split_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        portable_rows: list[dict] = []
        for ordinal, row in enumerate(split_rows[split]):
            source = pathlib.Path(str(row["audio"])).resolve()
            file_sha = require_text(row.get("file_sha256"), f"{split} row {ordinal}.file_sha256")
            target = audio_dir / f"recording-{ordinal:06d}-{file_sha[:12]}.wav"
            if target.exists():
                raise ValueError(f"{split}: labeled audio target already exists: {target}")
            _copy_audio_exclusive(source, target)
            if sha256_file(target) != file_sha:
                raise ValueError(f"{split}: labeled audio sha256 mismatch after copy")
            targets.append((target, file_sha))
            portable = rebind_audio_path(row, target.relative_to(split_root).as_posix(), field="audio")
            portable_rows.append(portable)
        portable_splits[split] = portable_rows
    # A later copy must not invalidate an earlier target while leaving a
    # successful admission already published. Verify the entire copied set.
    _check_materialized_audio(targets)
    verification_session.checkpoint()
    for split in SPLITS:
        split_root = output_root / split
        raw_manifest = split_root / "manifest.jsonl"
        labels_path = split_root / "labels.jsonl"
        labeled_manifest = split_root / "labeled-manifest.jsonl"
        portable_rows = portable_splits[split]
        enriched = [rebind_audio_path(labeled, portable["audio"], field="audio")
                    for labeled, portable in zip(split_labeled[split], portable_rows)]
        labeled_payload = _jsonl_text(enriched)
        payloads.extend(((raw_manifest, _jsonl_text(portable_rows)),
                         (labels_path, _jsonl_text(split_labels[split])),
                         (labeled_manifest, labeled_payload)))
        split_summary[split] = {
            "recordings": len(enriched),
            "positive": sum(1 for row in enriched if row["kind"] == "positive"),
            "confusable": sum(1 for row in enriched if row["kind"] == "confusable"),
            "negative": sum(1 for row in enriched if row["kind"] == "negative"),
            "labeled_manifest": str(labeled_manifest),
            "labeled_manifest_sha256": hashlib.sha256(labeled_payload.encode("utf-8")).hexdigest(),
        }
    _check_materialized_audio(targets)
    summary = {
        "schema_version": 1,
        "evidence_class": "speech-like-corpus-materialization-v1",
        "admission_mode": admission_mode,
        "ctc_training_allowed": admission_mode == REAL_MODE,
        "policy": POLICY,
        "intents_sha256": sha256_file(intents_path),
        "source_manifests": source_manifest_sha,
        "splits": split_summary,
        "recordings": sum(item["recordings"] for item in split_summary.values()),
        "protected_evidence_used": False,
        **({"reviewed_admission": admission} if admission is not None else {}),
    }
    return summary, payloads, targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan and materialize immutable speech-like TTS corpora.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--plan", required=True, type=pathlib.Path)
    build.add_argument("--voice-inventory", required=True, type=pathlib.Path)
    build.add_argument("--requests", required=True, type=pathlib.Path)
    build.add_argument("--intents", required=True, type=pathlib.Path)
    build.add_argument("--summary", required=True, type=pathlib.Path)

    materialize = sub.add_parser("materialize")
    materialize.add_argument("--intents", required=True, type=pathlib.Path)
    materialize.add_argument("--generated-root", required=True, type=pathlib.Path)
    materialize.add_argument("--output-root", required=True, type=pathlib.Path)
    materialize.add_argument("--summary", required=True, type=pathlib.Path)

    materialize.add_argument("--audio-review", type=pathlib.Path)
    materialize.add_argument("--asr-consensus", type=pathlib.Path, help="Supplied three-ASR consensus records for train rows")
    materialize.add_argument("--asr-evidence-dir", type=pathlib.Path, help="Local evidence storage only; never a trust-root override")
    materialize.add_argument("--admission-mode", choices=(REAL_MODE, FIXTURE_MODE), default=REAL_MODE)

    admission = sub.add_parser("admit-reviewed", help="Read-only hash-bound admission of already-reviewed WAVs")
    admission.add_argument("--intents", required=True, type=pathlib.Path)
    admission.add_argument("--generated-root", required=True, type=pathlib.Path)
    admission.add_argument("--audio-review", type=pathlib.Path)
    admission.add_argument("--asr-consensus", type=pathlib.Path, help="Supplied three-ASR consensus records for train rows")
    admission.add_argument("--asr-evidence-dir", type=pathlib.Path, help="Local evidence storage only; never a trust-root override")
    admission.add_argument("--expected-count", required=True, type=int)

    consensus = sub.add_parser("inspect-three-asr", help="Read-only prospective machine-label check; never training admission")
    consensus.add_argument("--intents", required=True, type=pathlib.Path)
    consensus.add_argument("--generated-root", required=True, type=pathlib.Path)
    consensus.add_argument("--consensus-receipts", required=True, type=pathlib.Path)
    consensus.add_argument("--expected-count", required=True, type=int)

    args = parser.parse_args()
    if args.command == "inspect-three-asr":
        summary = inspect_consensus_recordings(args.intents.resolve(), args.generated_root.resolve(),
                                               args.consensus_receipts.resolve(), expected_count=args.expected_count)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.command == "admit-reviewed":
        summary = admit_reviewed_recordings(args.intents.resolve(), args.generated_root.resolve(),
                                            args.audio_review.resolve() if args.audio_review else None,
                                            expected_count=args.expected_count,
                                            asr_consensus=args.asr_consensus.resolve() if args.asr_consensus else None,
                                            evidence_dir=args.asr_evidence_dir)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.command == "build":
        requests, intents, summary = build_requests(args.plan.resolve(), args.voice_inventory.resolve())
        write_jsonl(args.requests.resolve(), requests)
        write_jsonl(args.intents.resolve(), intents)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"speech-like corpus plan: requests={summary['requests']} voices={summary['voice_slots']} "
            f"utterances={summary['utterances']}"
        )
        return 0

    summary = materialize_labels(args.intents.resolve(), args.generated_root.resolve(), args.output_root.resolve(),
                                 audio_review=args.audio_review.resolve() if args.audio_review else None,
                                 admission_mode=args.admission_mode,
                                 asr_consensus=args.asr_consensus.resolve() if args.asr_consensus else None,
                                 evidence_dir=args.asr_evidence_dir)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"speech-like corpus materialized: recordings={summary['recordings']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, EOFError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
