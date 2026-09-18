#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from attach_speech_like_labels import attach as attach_labels  # noqa: E402

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


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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


def materialize_labels(intents_path: pathlib.Path, generated_root: pathlib.Path, output_root: pathlib.Path) -> dict:
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
    for group in sorted(ALLOWED_PROVIDER_GROUPS):
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
            normalized = dict(row)
            normalized["audio"] = str(audio)
            generated[key] = normalized

    if set(generated) != set(intent_map):
        missing = sorted(set(intent_map) - set(generated))
        extra = sorted(set(generated) - set(intent_map))
        raise ValueError(f"generated corpus does not match planned requests; missing={len(missing)} extra={len(extra)}")

    split_rows: dict[str, list[dict]] = {split: [] for split in SPLITS}
    split_labels: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for key, intent in intent_map.items():
        row = generated[key]
        split = str(intent["split"])
        split_rows[split].append(row)
        split_labels[split].append(
            {
                "schema_version": 1,
                "evidence_class": LABEL_CLASS,
                "voice_id": row["voice_id"],
                "source_id": row["source_id"],
                "file_sha256": row["file_sha256"],
                "kind": intent["kind"],
                "keyword_id": intent["keyword_id"],
                "label_source": POLICY,
            }
        )

    split_summary: dict[str, dict] = {}
    for split in SPLITS:
        split_root = output_root / split
        raw_manifest = split_root / "manifest.jsonl"
        labels_path = split_root / "labels.jsonl"
        labeled_manifest = split_root / "labeled-manifest.jsonl"
        audio_dir = split_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        portable_rows: list[dict] = []
        for ordinal, row in enumerate(split_rows[split]):
            source = pathlib.Path(str(row["audio"])).resolve()
            file_sha = require_text(row.get("file_sha256"), f"{split} row {ordinal}.file_sha256")
            target = audio_dir / f"recording-{ordinal:06d}-{file_sha[:12]}.wav"
            if target.exists():
                raise ValueError(f"{split}: labeled audio target already exists: {target}")
            shutil.copyfile(source, target)
            if sha256_file(target) != file_sha:
                raise ValueError(f"{split}: labeled audio sha256 mismatch after copy")
            portable = dict(row)
            portable["audio"] = target.relative_to(split_root).as_posix()
            portable_rows.append(portable)
        write_jsonl(raw_manifest, portable_rows)
        write_jsonl(labels_path, split_labels[split])
        enriched = attach_labels(raw_manifest, labels_path)
        write_jsonl(labeled_manifest, enriched)
        split_summary[split] = {
            "recordings": len(enriched),
            "positive": sum(1 for row in enriched if row["kind"] == "positive"),
            "confusable": sum(1 for row in enriched if row["kind"] == "confusable"),
            "negative": sum(1 for row in enriched if row["kind"] == "negative"),
            "labeled_manifest": str(labeled_manifest),
            "labeled_manifest_sha256": sha256_file(labeled_manifest),
        }
    return {
        "schema_version": 1,
        "evidence_class": "speech-like-corpus-materialization-v1",
        "policy": POLICY,
        "intents_sha256": sha256_file(intents_path),
        "source_manifests": source_manifest_sha,
        "splits": split_summary,
        "recordings": sum(item["recordings"] for item in split_summary.values()),
        "protected_evidence_used": False,
    }


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

    args = parser.parse_args()
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

    summary = materialize_labels(args.intents.resolve(), args.generated_root.resolve(), args.output_root.resolve())
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"speech-like corpus materialized: recordings={summary['recordings']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
