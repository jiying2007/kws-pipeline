#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import unicodedata
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from attach_speech_like_labels import attach as attach_labels  # noqa: E402

POLICY = "speech-like-corpus-plan-v1"
VOICE_CLASS = "speech-like-provider-voice-slot-v1"
INTENT_CLASS = "speech-like-request-label-intent-v1"
MANIFEST_CLASS = "speech-like-synthetic-recording-v1"
LABEL_CLASS = "speech-like-training-label-v1"
ASR_REVIEW_CLASS = "speech-like-asr-review-v1"
ASR_STANDARD = "exact-normalized-text-v1"
CONTIGUOUS_MAX_SILENCE_MS = 120
SPLITS = ("train", "calibration", "test", "qualification")
ALLOWED_KINDS = {"positive", "confusable", "negative"}
ALLOWED_PROVIDER_GROUPS = {"train", "generalization-search", "generalization-freeze"}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


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


def validate_cross_provider_holdout(rows_by_split: dict[str, list[dict]]) -> dict[str, dict]:
    """Require independent TTS identities for train, search, and freeze cohorts."""
    identities: dict[str, dict] = {}
    for split in SPLITS:
        rows = rows_by_split.get(split, [])
        if not rows:
            raise ValueError(f"cross-provider holdout requires recordings in {split}")
        providers: dict[str, str] = {}
        for index, row in enumerate(rows, 1):
            name = require_text(row.get("provider_name"), f"{split} row {index}.provider_name")
            digest = row.get("provider_identity_sha256")
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{split} row {index}: provider_identity_sha256 is required")
            previous = providers.get(name)
            if previous is not None and previous != digest:
                raise ValueError(f"{split}: provider_name has multiple asset identities: {name}")
            providers[name] = digest
        if len(set(providers.values())) != len(providers):
            raise ValueError(f"{split}: multiple provider names reuse one asset identity")
        if split != "train" and len(providers) != 1:
            raise ValueError(f"{split}: one pinned TTS provider is required")
        identities[split] = {"providers": [
            {"provider_name": name, "provider_identity_sha256": providers[name]}
            for name in sorted(providers)
        ]}

    for left, right in (("train", "calibration"), ("train", "test"),
                        ("train", "qualification"), ("calibration", "qualification"),
                        ("test", "qualification")):
        left_names = {item["provider_name"] for item in identities[left]["providers"]}
        right_names = {item["provider_name"] for item in identities[right]["providers"]}
        left_digests = {item["provider_identity_sha256"] for item in identities[left]["providers"]}
        right_digests = {item["provider_identity_sha256"] for item in identities[right]["providers"]}
        if left_names & right_names or left_digests & right_digests:
            raise ValueError(f"cross-provider holdout reuses a TTS provider in {left}/{right}")
    if identities["calibration"] != identities["test"]:
        raise ValueError("calibration/test must share the pinned search provider")
    return identities


def validate_audio_review(path: pathlib.Path, intent_map: dict, generated: dict) -> dict:
    """Bind a human audio verdict to every planned recording and its WAV hash."""
    reviews: dict[tuple[str, str], dict] = {}
    reviewers: set[str] = set()
    for index, row in enumerate(load_jsonl(path), 1):
        if row.get("schema_version") != 1 or row.get("evidence_class") != "speech-like-audio-review-v1":
            raise ValueError(f"{path}:{index}: audio review identity mismatch")
        source_id = require_text(row.get("source_id"), f"{path}:{index}.source_id")
        file_sha = row.get("file_sha256")
        if not isinstance(file_sha, str) or SHA256_RE.fullmatch(file_sha) is None:
            raise ValueError(f"{path}:{index}: file_sha256 must be lowercase SHA-256")
        key = (source_id, file_sha)
        if key in reviews:
            raise ValueError(f"{path}:{index}: duplicate audio review identity")
        reviewers.add(require_text(row.get("reviewer_id"), f"{path}:{index}.reviewer_id"))
        reviews[key] = row

    expected: set[tuple[str, str]] = set()
    for key, intent in intent_map.items():
        source = generated[key]
        identity = (str(source["source_id"]), str(source["file_sha256"]))
        expected.add(identity)
        review = reviews.get(identity)
        if review is None:
            raise ValueError(f"audio review is missing for source_id={identity[0]}")
        if (review.get("intended_text") != intent["text"]
                or review.get("kind") != intent["kind"]
                or review.get("keyword_id") != intent["keyword_id"]):
            raise ValueError(f"audio review label differs from plan: source_id={identity[0]}")
        if review.get("verdict") != "accepted":
            raise ValueError(f"audio review is not accepted: source_id={identity[0]}")
    if set(reviews) != expected:
        raise ValueError("audio review contains unplanned or stale recordings")
    return {"review_sha256": sha256_file(path), "recordings": len(reviews),
            "reviewer_count": len(reviewers)}


def normalize_asr_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(char for char in normalized
                   if not char.isspace() and not unicodedata.category(char).startswith("P"))


def internal_silence_ms(audio: pathlib.Path) -> int:
    """Measure the longest low-energy run between the first/last active 10 ms frames."""
    with wave.open(str(audio), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (
                16000, 1, 2, "NONE"):
            raise ValueError(f"{audio}: continuity input must be 16 kHz mono PCM16")
        samples = array.array("h")
        samples.frombytes(stream.readframes(stream.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    frame_samples = 160
    levels = [math.sqrt(sum(float(sample * sample) for sample in samples[i:i + frame_samples])
                        / frame_samples)
              for i in range(0, len(samples) - frame_samples + 1, frame_samples)]
    if not levels or max(levels) < 100:
        return len(levels) * 10
    threshold = max(levels) * 0.01
    active = [i for i, level in enumerate(levels) if level >= threshold]
    longest = current = 0
    for index in range(active[0], active[-1] + 1):
        if levels[index] < threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest * 10


def generate_asr_review(manifests: list[pathlib.Path], audio_root: pathlib.Path | None,
                        asr_binary: pathlib.Path, asr_model: pathlib.Path,
                        asr_tokens: pathlib.Path, output: pathlib.Path,
                        intents: pathlib.Path | None = None) -> dict:
    """Use a fixed ASR binary/model to conservatively screen synthetic WAVs."""
    if output.exists():
        raise ValueError(f"ASR review output already exists: {output}")
    binary_sha = sha256_file(asr_binary)
    model_sha = sha256_file(asr_model)
    tokens_sha = sha256_file(asr_tokens)
    intent_by_source: dict[str, dict] = {}
    if intents is not None:
        for index, intent in enumerate(load_jsonl(intents), 1):
            if intent.get("schema_version") != 1 or intent.get("evidence_class") != INTENT_CLASS:
                raise ValueError(f"{intents}:{index}: label intent identity mismatch")
            source_id = require_text(intent.get("source_id"), f"{intents}:{index}.source_id")
            if source_id in intent_by_source:
                raise ValueError(f"{intents}:{index}: duplicate source_id")
            intent_by_source[source_id] = intent
    reviewed: list[dict] = []
    seen: set[tuple[str, str]] = set()
    seen_sources: set[str] = set()
    for manifest, index, row in (
        (manifest, index, row)
        for manifest in manifests
        for index, row in enumerate(load_jsonl(manifest), 1)
    ):
        source_id = require_text(row.get("source_id"), f"{manifest}:{index}.source_id")
        if source_id in seen_sources:
            raise ValueError(f"{manifest}:{index}: duplicate source_id")
        seen_sources.add(source_id)
        file_sha = require_text(row.get("file_sha256"), f"{manifest}:{index}.file_sha256")
        if SHA256_RE.fullmatch(file_sha) is None:
            raise ValueError(f"{manifest}:{index}: invalid WAV SHA-256")
        identity = (source_id, file_sha)
        if identity in seen:
            raise ValueError(f"{manifest}:{index}: duplicate WAV identity")
        seen.add(identity)
        intended = require_text(row.get("text"), f"{manifest}:{index}.text")
        intent = intent_by_source.get(source_id) if intents is not None else None
        if intents is not None and intent is None:
            raise ValueError(f"{manifest}:{index}: no matching label intent")
        if intent is not None and intent.get("text") != intended:
            raise ValueError(f"{manifest}:{index}: generated text differs from intent")
        kind = require_text(row.get("kind") if row.get("kind") is not None else
                            intent.get("kind") if intent is not None else None,
                            f"{manifest}:{index}.kind")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"{manifest}:{index}: invalid kind")
        keyword_id = row.get("keyword_id") if "keyword_id" in row else (
            intent.get("keyword_id") if intent is not None else None)
        if intent is not None and (kind != intent.get("kind") or keyword_id != intent.get("keyword_id")):
            raise ValueError(f"{manifest}:{index}: generated label differs from intent")
        if (kind == "positive" and keyword_id not in (1, 2)) or (kind != "positive" and keyword_id is not None):
            raise ValueError(f"{manifest}:{index}: invalid keyword_id for kind")
        audio_name = require_text(row.get("audio_path") or row.get("audio"),
                                  f"{manifest}:{index}.audio")
        audio = pathlib.Path(audio_name)
        if not audio.is_absolute():
            audio = (audio_root if audio_root is not None else manifest.parent) / audio
        audio = audio.resolve(strict=True)
        if sha256_file(audio) != file_sha:
            raise ValueError(f"{manifest}:{index}: WAV SHA-256 mismatch")
        with wave.open(str(audio), "rb") as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (
                    16000, 1, 2, "NONE"):
                raise ValueError(f"{manifest}:{index}: ASR input must be 16 kHz mono PCM16")
        result = subprocess.run(
            [str(asr_binary), f"--zipformer-ctc-model={asr_model}",
             f"--tokens={asr_tokens}", "--num-threads=2", str(audio)],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=60,
        )
        transcripts = [json.loads(line)["text"] for line in result.stdout.splitlines()
                       if line.startswith("{") and '"text"' in line]
        if len(transcripts) != 1 or not isinstance(transcripts[0], str):
            raise ValueError(f"{manifest}:{index}: ASR output must contain one transcript")
        asr_text = transcripts[0]
        accepted = bool(normalize_asr_text(intended)) and normalize_asr_text(asr_text) == normalize_asr_text(intended)
        reviewed.append({
            "schema_version": 1, "evidence_class": ASR_REVIEW_CLASS,
            "asr_standard": ASR_STANDARD, "source_id": source_id,
            "file_sha256": file_sha, "intended_text": intended,
            "kind": kind, "keyword_id": keyword_id, "asr_text": asr_text,
            "verdict": "accepted" if accepted else "rejected",
            "asr_binary_sha256": binary_sha, "asr_model_sha256": model_sha,
            "asr_tokens_sha256": tokens_sha,
            "usage_class": row.get("usage_class"),
        })
    if intents is not None and {row["source_id"] for row in reviewed} != set(intent_by_source):
        raise ValueError("ASR review manifests do not cover all planned source IDs")
    write_jsonl(output, reviewed)
    return {"recordings": len(reviewed),
            "accepted": sum(row["verdict"] == "accepted" for row in reviewed),
            "rejected": sum(row["verdict"] == "rejected" for row in reviewed),
            "review_sha256": sha256_file(output),
            "asr_binary_sha256": binary_sha, "asr_model_sha256": model_sha,
            "asr_tokens_sha256": tokens_sha}


def validate_asr_review(path: pathlib.Path, intent_map: dict, generated: dict) -> dict:
    """Keep ASR evidence distinct from human listening receipts."""
    reviews: dict[tuple[str, str], dict] = {}
    engines: set[tuple[str, str, str]] = set()
    for index, row in enumerate(load_jsonl(path), 1):
        if row.get("schema_version") != 1 or row.get("evidence_class") != ASR_REVIEW_CLASS or row.get("asr_standard") != ASR_STANDARD:
            raise ValueError(f"{path}:{index}: ASR review identity mismatch")
        identity = (require_text(row.get("source_id"), f"{path}:{index}.source_id"),
                    require_text(row.get("file_sha256"), f"{path}:{index}.file_sha256"))
        if SHA256_RE.fullmatch(identity[1]) is None or identity in reviews:
            raise ValueError(f"{path}:{index}: invalid or duplicate WAV identity")
        engine = tuple(require_text(row.get(field), f"{path}:{index}.{field}")
                       for field in ("asr_binary_sha256", "asr_model_sha256", "asr_tokens_sha256"))
        if any(SHA256_RE.fullmatch(value) is None for value in engine):
            raise ValueError(f"{path}:{index}: invalid ASR engine SHA-256")
        engines.add(engine)
        intended = require_text(row.get("intended_text"), f"{path}:{index}.intended_text")
        asr_text = row.get("asr_text")
        if not isinstance(asr_text, str):
            raise ValueError(f"{path}:{index}: invalid ASR transcript")
        expected_verdict = ("accepted" if normalize_asr_text(asr_text) == normalize_asr_text(intended)
                            and bool(normalize_asr_text(intended)) else "rejected")
        if row.get("verdict") != expected_verdict:
            raise ValueError(f"{path}:{index}: ASR verdict differs from transcript")
        reviews[identity] = row
    if len(engines) != 1:
        raise ValueError("ASR review contains multiple engine identities")
    expected: set[tuple[str, str]] = set()
    for key, intent in intent_map.items():
        source = generated[key]
        identity = (str(source["source_id"]), str(source["file_sha256"]))
        expected.add(identity)
        review = reviews.get(identity)
        if review is None:
            raise ValueError(f"ASR review is missing for source_id={identity[0]}")
        if (review.get("intended_text") != intent["text"] or review.get("kind") != intent["kind"]
                or review.get("keyword_id") != intent["keyword_id"]):
            raise ValueError(f"ASR review label differs from plan: source_id={identity[0]}")
        if review.get("verdict") != "accepted":
            raise ValueError(f"ASR review is not accepted: source_id={identity[0]}")
        # ASR verifies lexical content; a continuous target also needs a bounded
        # internal gap. Explicitly punctuated plan variants keep their own label.
        if (intent["kind"] == "positive" and not any(
                unicodedata.category(char).startswith("P") for char in intent["text"])):
            audio = source.get("audio")
            if audio is None:
                raise ValueError(f"ASR continuity check is missing audio: source_id={identity[0]}")
            if internal_silence_ms(pathlib.Path(str(audio))) >= CONTIGUOUS_MAX_SILENCE_MS:
                raise ValueError(f"ASR accepted text but positive has long internal silence: source_id={identity[0]}")
    if set(reviews) != expected:
        raise ValueError("ASR review contains unplanned or stale recordings")
    engine = next(iter(engines))
    return {"review_sha256": sha256_file(path), "recordings": len(reviews),
            "asr_standard": ASR_STANDARD, "asr_binary_sha256": engine[0],
            "asr_model_sha256": engine[1], "asr_tokens_sha256": engine[2],
            "contiguous_max_internal_silence_ms": CONTIGUOUS_MAX_SILENCE_MS}


def validate_mixed_reviews(audio_review: pathlib.Path, asr_review: pathlib.Path,
                           intent_map: dict, generated: dict) -> tuple[dict, dict]:
    """Require disjoint, complete human/ASR coverage for a mixed research corpus."""
    expected = {(str(generated[key]["source_id"]), str(generated[key]["file_sha256"])): key
                for key in intent_map}
    if len(expected) != len(intent_map):
        raise ValueError("planned review WAV identities are not unique")

    def identities(path: pathlib.Path) -> set[tuple[str, str]]:
        values = [(require_text(row.get("source_id"), f"{path}.source_id"),
                   require_text(row.get("file_sha256"), f"{path}.file_sha256"))
                  for row in load_jsonl(path)]
        if len(set(values)) != len(values):
            raise ValueError(f"{path}: duplicate review WAV identity")
        return set(values)

    human = identities(audio_review)
    machine = identities(asr_review)
    if human & machine:
        raise ValueError("human and ASR reviews overlap on a WAV identity")
    if human | machine != set(expected):
        raise ValueError("mixed reviews do not cover exactly the planned WAV identities")
    human_intents = {expected[identity]: intent_map[expected[identity]] for identity in human}
    machine_intents = {expected[identity]: intent_map[expected[identity]] for identity in machine}
    return (validate_audio_review(audio_review, human_intents, generated),
            validate_asr_review(asr_review, machine_intents, generated))


def merge_generated_manifests(inputs: list[pathlib.Path], output: pathlib.Path) -> dict:
    """Combine independently generated providers without losing their audio roots."""
    if len(inputs) < 2:
        raise ValueError("provider merge requires at least two manifests")
    if output.exists():
        raise ValueError(f"provider merge output already exists: {output}")
    merged: list[dict] = []
    sources: set[str] = set()
    file_hashes: set[str] = set()
    providers: set[tuple[str, str]] = set()
    for manifest in inputs:
        for index, row in enumerate(load_jsonl(manifest), 1):
            if row.get("schema_version") != 1 or row.get("evidence_class") != MANIFEST_CLASS:
                raise ValueError(f"{manifest}:{index}: generated manifest identity mismatch")
            source_id = require_text(row.get("source_id"), f"{manifest}:{index}.source_id")
            provider_name = require_text(row.get("provider_name"), f"{manifest}:{index}.provider_name")
            provider_sha = row.get("provider_identity_sha256")
            file_sha = row.get("file_sha256")
            if (not isinstance(provider_sha, str) or SHA256_RE.fullmatch(provider_sha) is None
                    or not isinstance(file_sha, str) or SHA256_RE.fullmatch(file_sha) is None):
                raise ValueError(f"{manifest}:{index}: provider/audio SHA-256 is required")
            if source_id in sources or file_sha in file_hashes:
                raise ValueError(f"{manifest}:{index}: duplicate source or audio identity")
            raw_audio = require_text(row.get("audio"), f"{manifest}:{index}.audio")
            audio = pathlib.Path(raw_audio)
            audio = audio.resolve() if audio.is_absolute() else (manifest.parent / audio).resolve()
            if not audio.is_file() or sha256_file(audio) != file_sha:
                raise ValueError(f"{manifest}:{index}: WAV missing or SHA-256 mismatch")
            normalized = dict(row)
            normalized["audio"] = str(audio)
            merged.append(normalized)
            sources.add(source_id)
            file_hashes.add(file_sha)
            providers.add((provider_name, provider_sha))
    if len(providers) < 2:
        raise ValueError("provider merge did not contain two distinct TTS providers")
    merged.sort(key=lambda row: (str(row["provider_name"]), str(row["source_id"])))
    write_jsonl(output, merged)
    return {"recordings": len(merged), "providers": len(providers),
            "manifest_sha256": sha256_file(output)}


def materialize_labels(intents_path: pathlib.Path, generated_root: pathlib.Path,
                       output_root: pathlib.Path, *, require_cross_provider_holdout: bool = False,
                       audio_review: pathlib.Path | None = None,
                       asr_review: pathlib.Path | None = None) -> dict:
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

    if audio_review is not None and asr_review is not None:
        review_summary, asr_summary = validate_mixed_reviews(
            audio_review, asr_review, intent_map, generated)
    else:
        review_summary = (
            validate_audio_review(audio_review, intent_map, generated)
            if audio_review is not None else None
        )
        asr_summary = (
            validate_asr_review(asr_review, intent_map, generated)
            if asr_review is not None else None
        )

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

    provider_holdout = (
        validate_cross_provider_holdout(split_rows)
        if require_cross_provider_holdout else None
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
    summary = {
        "schema_version": 1,
        "evidence_class": "speech-like-corpus-materialization-v1",
        "policy": POLICY,
        "intents_sha256": sha256_file(intents_path),
        "source_manifests": source_manifest_sha,
        "splits": split_summary,
        "recordings": sum(item["recordings"] for item in split_summary.values()),
        "protected_evidence_used": False,
    }
    if provider_holdout is not None:
        summary["cross_provider_holdout"] = provider_holdout
    if review_summary is not None:
        summary["audio_review"] = review_summary
    if asr_summary is not None:
        summary["asr_review"] = asr_summary
    return summary


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
    materialize.add_argument("--require-cross-provider-holdout", action="store_true")
    materialize.add_argument("--audio-review", type=pathlib.Path)
    materialize.add_argument("--asr-review", type=pathlib.Path)

    asr = sub.add_parser("asr-review", help="Hash-bound exact-ASR screening of synthetic WAVs")
    asr.add_argument("--manifest", required=True, action="append", type=pathlib.Path)
    asr.add_argument("--audio-root", type=pathlib.Path)
    asr.add_argument("--intents", type=pathlib.Path)
    asr.add_argument("--asr-binary", required=True, type=pathlib.Path)
    asr.add_argument("--asr-model", required=True, type=pathlib.Path)
    asr.add_argument("--asr-tokens", required=True, type=pathlib.Path)
    asr.add_argument("--output", required=True, type=pathlib.Path)

    merge = sub.add_parser("merge", help="Merge disjoint provider manifests into one group")
    merge.add_argument("--input-manifest", required=True, action="append", type=pathlib.Path)
    merge.add_argument("--output-manifest", required=True, type=pathlib.Path)

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

    if args.command == "merge":
        summary = merge_generated_manifests(
            [path.resolve() for path in args.input_manifest], args.output_manifest.resolve()
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    if args.command == "asr-review":
        summary = generate_asr_review(
            [path.resolve() for path in args.manifest],
            args.audio_root.resolve() if args.audio_root is not None else None,
            args.asr_binary.resolve(strict=True), args.asr_model.resolve(strict=True),
            args.asr_tokens.resolve(strict=True), args.output.resolve(),
            intents=args.intents.resolve(strict=True) if args.intents is not None else None,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    summary = materialize_labels(
        args.intents.resolve(), args.generated_root.resolve(), args.output_root.resolve(),
        require_cross_provider_holdout=args.require_cross_provider_holdout,
        audio_review=args.audio_review.resolve() if args.audio_review is not None else None,
        asr_review=args.asr_review.resolve() if args.asr_review is not None else None,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"speech-like corpus materialized: recordings={summary['recordings']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
