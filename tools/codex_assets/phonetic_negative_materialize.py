#!/usr/bin/env python3
"""Materialize a bounded, ASR-observed phonetic negative research cohort."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import wave

from tools.speech_like_corpus_plan import normalize_asr_text

WAKE_TEXTS = frozenset(("你好小窝", "小窝小窝"))
POLICY = "phonetic-negative-equivalence-v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_rows(path: pathlib.Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return rows


def read_tokens(path: pathlib.Path) -> dict[str, int]:
    tokens: dict[str, int] = {}
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid token row: {line}")
        name, raw_id = parts
        token_id = int(raw_id)
        if name in tokens or token_id in ids or token_id < 0:
            raise ValueError("duplicate or invalid token ID")
        tokens[name] = token_id
        ids.add(token_id)
    if tokens.get("<blk>") != 0 or sorted(ids) != list(range(len(ids))):
        raise ValueError("vocabulary must be contiguous and blank-first")
    return tokens


def materialize(
    *, policy_path: pathlib.Path, tokens_path: pathlib.Path,
    candidates_path: pathlib.Path, asr_path: pathlib.Path,
    speaker_split_path: pathlib.Path, existing_manifest_path: pathlib.Path,
) -> tuple[list[dict], dict]:
    policy = read_object(policy_path)
    if (policy.get("schema_version") != 1 or policy.get("policy") != POLICY
            or policy.get("scope") != "development-only-negative-competitors"
            or policy.get("positive_acceptance") != "exact-normalized-text-v1-only"):
        raise ValueError("phonetic-negative policy identity mismatch")
    cases: dict[str, dict] = {}
    vocabulary = read_tokens(tokens_path)
    for case in policy.get("cases", []):
        if not isinstance(case, dict):
            raise ValueError("phonetic case must be an object")
        official = str(case.get("official_text", ""))
        aliases = case.get("accepted_asr_text")
        names = case.get("tokens")
        minimum = case.get("minimum_rows")
        if (not official or official in cases or normalize_asr_text(official) in WAKE_TEXTS
                or not isinstance(aliases, list) or not aliases
                or not isinstance(names, list) or not names
                or not isinstance(minimum, int) or minimum < 1):
            raise ValueError("invalid negative-only phonetic case")
        if (any(not isinstance(value, str) or not normalize_asr_text(value)
                or normalize_asr_text(value) in WAKE_TEXTS for value in aliases)
                or any(not isinstance(name, str) or name not in vocabulary
                       or vocabulary[name] == 0 for name in names)):
            raise ValueError("phonetic case aliases or tokens are invalid")
        for wake in (("ni3", "hao3", "xiao3", "wo1"),
                     ("xiao3", "wo1", "xiao3", "wo1")):
            if any(tuple(names[index:index + 4]) == wake
                   for index in range(max(0, len(names) - 3))):
                raise ValueError("negative phonetic case contains a full wake target")
        cases[official] = case
    if not cases:
        raise ValueError("phonetic policy has no negative cases")
    split = read_object(speaker_split_path)
    train_speakers = set(split.get("train_speakers", []))
    holdout_speakers = set(split.get("holdout_speakers", []))
    if not train_speakers or train_speakers & holdout_speakers:
        raise ValueError("speaker split is missing or overlapping")
    existing_wavs = {row["file_sha256"] for row in read_rows(existing_manifest_path)}
    reviews = read_rows(asr_path)
    by_source = {row.get("source_id"): row for row in reviews}
    if len(reviews) != len(by_source):
        raise ValueError("duplicate ASR source ID")
    candidates = read_rows(candidates_path)
    if len(candidates) != len(reviews):
        raise ValueError("ASR review does not cover all candidates")
    accepted: list[dict] = []
    counts: collections.Counter[str] = collections.Counter()
    used_wavs: set[str] = set()
    for source in candidates:
        source_id = source.get("source_id")
        receipt = by_source.pop(source_id, None)
        audio = pathlib.Path(str(source.get("audio_path", "")))
        digest = source.get("file_sha256")
        if (receipt is None or receipt.get("schema_version") != 1
                or receipt.get("evidence_class") != "qwen3-asr-himia-competitor-screen-v1"
                or receipt.get("source_id") != source_id
                or receipt.get("recording") != source.get("recording")
                or receipt.get("kind") != "confusable"
                or receipt.get("keyword_id") is not None
                or source.get("kind") != "confusable"
                or source.get("keyword_id") is not None
                or source.get("speaker_id") not in train_speakers
                or source.get("speaker_id") in holdout_speakers
                or not audio.is_file() or not isinstance(digest, str)
                or sha256_file(audio) != digest or receipt.get("file_sha256") != digest
                or receipt.get("intended_text") != source.get("text")
                or receipt.get("asr_model_revision") != policy.get("asr_model_revision")
                or receipt.get("asr_model_sha256") != policy.get("asr_model_sha256")):
            raise ValueError("candidate, WAV, ASR or speaker identity mismatch")
        with wave.open(str(audio), "rb") as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (
                    16000, 1, 2, "NONE") or stream.getnframes() <= 0:
                raise ValueError("candidate audio is not mono 16-kHz PCM16 WAV")
        case = cases.get(str(source.get("text")))
        if case is None:
            continue
        observed = normalize_asr_text(str(receipt.get("asr_text", "")))
        exact = observed == normalize_asr_text(str(source.get("text", "")))
        if receipt.get("verdict") != ("accepted" if exact else "rejected"):
            raise ValueError("ASR exact verdict differs from transcript")
        if observed not in {normalize_asr_text(value) for value in case["accepted_asr_text"]}:
            continue
        if digest in existing_wavs or digest in used_wavs:
            raise ValueError("training WAV overlap or duplicate")
        used_wavs.add(digest)
        accepted.append({
            "recording": source["recording"], "audio": str(audio),
            "tokens": [vocabulary[name] for name in case["tokens"]],
            "keyword_id": None, "kind": "confusable",
            "speaker_id": "himia-" + source["speaker_id"],
            "session_id": source.get("session_id"), "source_id": source_id,
            "file_sha256": digest, "official_text": source["text"],
            "observed_asr_text": receipt["asr_text"],
            "asr_exact_verdict": receipt.get("verdict"),
            "phonetic_negative_policy": POLICY,
            "phonetic_negative_policy_sha256": sha256_file(policy_path),
            "asr_review_sha256": sha256_file(asr_path),
            "usage_class": source.get("usage_class"),
        })
        counts[source["text"]] += 1
    if by_source:
        raise ValueError("ASR review has unmatched recordings")
    if any(counts[text] < case["minimum_rows"] for text, case in cases.items()):
        raise ValueError(f"phonetic case below minimum: {dict(counts)}")
    summary = {
        "accepted": len(accepted), "by_official_text": dict(counts),
        "policy_sha256": sha256_file(policy_path),
        "tokens_sha256": sha256_file(tokens_path),
        "candidates_sha256": sha256_file(candidates_path),
        "asr_review_sha256": sha256_file(asr_path),
        "speaker_split_sha256": sha256_file(speaker_split_path),
        "existing_manifest_sha256": sha256_file(existing_manifest_path),
    }
    return accepted, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--candidates", required=True, type=pathlib.Path)
    parser.add_argument("--asr-review", required=True, type=pathlib.Path)
    parser.add_argument("--speaker-split", required=True, type=pathlib.Path)
    parser.add_argument("--existing-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows, summary = materialize(
        policy_path=args.policy, tokens_path=args.tokens,
        candidates_path=args.candidates, asr_path=args.asr_review,
        speaker_split_path=args.speaker_split,
        existing_manifest_path=args.existing_manifest,
    )
    if not args.dry_run:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                       for row in rows), encoding="utf-8")
        summary["manifest_sha256"] = sha256_file(args.output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
