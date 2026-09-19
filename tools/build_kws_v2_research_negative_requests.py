#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

EVIDENCE_CLASS = "kws-v2-research-negative-utterance-plan-v1"
VOICE_CLASS = "speech-like-provider-voice-slot-v1"
SPLITS = ("train", "calibration", "test")


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


def validate_plan(path: pathlib.Path) -> dict:
    value = load_object(path)
    if int(value.get("schema_version", 0)) != 1 or value.get("evidence_class") != EVIDENCE_CLASS:
        raise ValueError("research negative plan identity mismatch")
    if value.get("evidence_scope") != "research-only":
        raise ValueError("research negative plan must remain research-only")
    if value.get("target_policy") != "empty-target-nonwake":
        raise ValueError("research negative plan target policy drifted")
    if value.get("protected_evidence_used") is not False:
        raise ValueError("research negative plan cannot use protected evidence")
    if value.get("qualification_split_allowed") is not False:
        raise ValueError("research negative plan cannot request qualification voices")
    roles = value.get("split_voice_slots")
    if not isinstance(roles, dict) or set(roles) != set(SPLITS):
        raise ValueError("research negative split roles must be exactly train/calibration/test")
    seen_slots: set[str] = set()
    for split in SPLITS:
        slots = roles[split]
        if not isinstance(slots, list) or not slots:
            raise ValueError(f"{split} voice slots must be non-empty")
        if any(not isinstance(slot, str) or not slot for slot in slots):
            raise ValueError(f"{split} contains invalid voice slot")
        overlap = seen_slots.intersection(slots)
        if overlap:
            raise ValueError(f"voice slots overlap across research splits: {sorted(overlap)}")
        seen_slots.update(slots)
        if any(slot.startswith("freeze-") for slot in slots):
            raise ValueError("qualification/freeze voice slot is forbidden in research negatives")
    utterances = value.get("utterances")
    if not isinstance(utterances, list) or len(utterances) < 24:
        raise ValueError("research negative plan needs at least 24 utterances")
    ids: set[str] = set()
    categories: set[str] = set()
    for row in utterances:
        if not isinstance(row, dict):
            raise ValueError("research negative utterances must be objects")
        uid = str(row.get("id", "")).strip()
        text = str(row.get("text", "")).strip()
        category = str(row.get("category", "")).strip()
        tokens = row.get("provider_tokens")
        if not uid or uid in ids or not text or not category:
            raise ValueError(f"invalid research negative utterance: {row}")
        if not isinstance(tokens, list) or not tokens or any(not isinstance(v, str) or not v for v in tokens):
            raise ValueError(f"{uid}: provider_tokens must be non-empty strings")
        ids.add(uid)
        categories.add(category)
    if len(categories) < 5:
        raise ValueError("research negative plan must cover at least five linguistic categories")
    return value


def load_inventory(path: pathlib.Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for index, row in enumerate(load_jsonl(path), 1):
        if int(row.get("schema_version", 0)) != 1 or row.get("evidence_class") != VOICE_CLASS:
            raise ValueError(f"{path}:{index}: voice inventory identity mismatch")
        slot = str(row.get("slot", "")).strip()
        voice_id = str(row.get("voice_id", "")).strip()
        parameters = row.get("parameters", {})
        if not slot or not voice_id or not isinstance(parameters, dict):
            raise ValueError(f"{path}:{index}: invalid voice inventory row")
        if slot in result:
            raise ValueError(f"duplicate voice slot: {slot}")
        result[slot] = {"voice_id": voice_id, "parameters": parameters}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build research-only ordinary-speech negative TTS requests from a verified Stage-A voice inventory."
    )
    parser.add_argument(
        "--plan",
        type=pathlib.Path,
        default=pathlib.Path("configs/training/kws-v2-research-negative-utterances-v1.json"),
    )
    parser.add_argument("--voice-inventory", required=True, type=pathlib.Path)
    parser.add_argument("--requests", required=True, type=pathlib.Path)
    parser.add_argument("--index", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    inventory_path = args.voice_inventory.resolve()
    plan = validate_plan(plan_path)
    inventory = load_inventory(inventory_path)

    requests: list[dict] = []
    index_rows: list[dict] = []
    seen_sources: set[str] = set()
    for split in SPLITS:
        group = "train" if split == "train" else "generalization-search"
        for slot in plan["split_voice_slots"][split]:
            if slot not in inventory:
                raise ValueError(f"research negative voice slot absent from inventory: {slot}")
            voice = inventory[slot]
            for utterance in plan["utterances"]:
                source_id = f"kws-v2-research-negative:{split}:{slot}:{utterance['id']}"
                if source_id in seen_sources:
                    raise ValueError(f"duplicate research negative source: {source_id}")
                seen_sources.add(source_id)
                request = {
                    "schema_version": 1,
                    "group": group,
                    "text": str(utterance["text"]),
                    "tokens": list(utterance["provider_tokens"]),
                    "voice_id": str(voice["voice_id"]),
                    "source_id": source_id,
                    "parameters": dict(voice["parameters"]),
                }
                requests.append(request)
                index_rows.append(
                    {
                        "schema_version": 1,
                        "evidence_class": "kws-v2-research-negative-request-v1",
                        "evidence_scope": "research-only",
                        "split": split,
                        "category": str(utterance["category"]),
                        "utterance_id": str(utterance["id"]),
                        "text": str(utterance["text"]),
                        "voice_slot": slot,
                        "voice_id": str(voice["voice_id"]),
                        "source_id": source_id,
                        "target_policy": "empty-target-nonwake",
                        "protected_evidence_used": False,
                    }
                )

    write_jsonl(args.requests.resolve(), requests)
    write_jsonl(args.index.resolve(), index_rows)
    split_counts = {
        split: sum(1 for row in index_rows if row["split"] == split) for split in SPLITS
    }
    value = {
        "schema_version": 1,
        "evidence_class": "kws-v2-research-negative-request-set-v1",
        "evidence_scope": "research-only",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "voice_inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "requests": len(requests),
        "split_counts": split_counts,
        "utterances": len(plan["utterances"]),
        "voice_slots": sum(len(plan["split_voice_slots"][split]) for split in SPLITS),
        "request_set_sha256": canonical_sha256(requests),
        "index_set_sha256": canonical_sha256(index_rows),
        "target_policy": "empty-target-nonwake",
        "qualification_split_consumed": False,
        "protected_evidence_used": False,
    }
    target = args.summary.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"research-negative-requests: requests={len(requests)} "
        f"train={split_counts['train']} calibration={split_counts['calibration']} test={split_counts['test']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
