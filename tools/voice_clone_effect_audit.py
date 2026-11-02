#!/usr/bin/env python3
"""Offline effectiveness audit for authorized voice-clone output.

This reads the real generated WAV bytes and optional *untrusted* ASR-screen
observations. Matching recognizer text is diagnostic only: it is NOT a
speech-label-admission receipt, a human-gold annotation, or release evidence.
No TTS, ASR, network request, training, or model promotion is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from authorized_voice_clone_batch import CHALLENGE_FAMILIES, KINDS, POLICY, WAKE_TEXTS  # noqa: E402
from corpus_identity import inspect_pcm16_wav, sha256_file  # noqa: E402
from speech_asr_consensus import normalized_transcript  # noqa: E402

AUDIT_POLICY = "voice-clone-effect-screen-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
SCREEN_MATCH = "REQUESTED_TEXT_MATCH_3X_UNVERIFIED"
SCREEN_NONE = "NOT_SCREENED"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(obj, dict), f"{path.name}: JSON object required")
    return obj


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require(bool(line.strip()), f"{path.name}:{index}: blank line")
        row = json.loads(line)
        require(isinstance(row, dict), f"{path.name}:{index}: JSON object required")
        rows.append(row)
        require(len(rows) <= 32768, "manifest/screen exceeds clone attempt ceiling")
    return rows


def private_audio_path(root: Path, relative: object) -> Path:
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "WAV path must be a relative POSIX path")
    name = PurePosixPath(relative)
    require(not name.is_absolute() and len(name.parts) >= 3 and name.parts[0] == "audio"
            and all(p not in ("", ".", "..") for p in name.parts),
            "WAV path escapes the campaign audio subtree")
    path = root
    for part in name.parts:
        path = path / part
        require(not path.is_symlink(), "WAV path cannot contain symlinks")
    resolved = path.resolve(strict=True)
    require(resolved.is_file() and root in resolved.parents,
            "WAV must be a regular file inside the campaign")
    return resolved


def valid_sha(value: object) -> bool:
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def clean_text(value: object) -> str:
    require(isinstance(value, str) and bool(value.strip()), "utterance text must be nonempty")
    return normalized_transcript(value)


def load_screen(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    indexed = {}
    for row in read_jsonl(path):
        item = row.get("item_id")
        require(isinstance(item, str) and item and item not in indexed, "duplicate/invalid screen item_id")
        require(valid_sha(row.get("file_sha256")), "ASR screen needs bound WAV SHA-256")
        observations = row.get("observations")
        require(isinstance(observations, list) and len(observations) == 3,
                "ASR screen must have three observations (not three authenticated votes)")
        families, texts = set(), []
        for observation in observations:
            require(isinstance(observation, dict), "invalid ASR observation")
            family = observation.get("model_family")
            require(isinstance(family, str) and family.strip() and family not in families,
                    "ASR screen families must be named and distinct")
            families.add(family)
            raw_text = observation.get("raw_text")
            require(isinstance(raw_text, str), "ASR screen raw_text must be a string (empty permitted)")
            texts.append(normalized_transcript(raw_text))
        indexed[item] = {"file_sha256": row["file_sha256"], "texts": texts,
                         "model_families": sorted(families)}
    return indexed


def classify(row: dict, screen: dict | None) -> str:
    if screen is None:
        return SCREEN_NONE
    requested = clean_text(row["text_requested"])
    texts = screen["texts"]
    # A substring in an ASR transcript cannot prove what the microphone
    # actually captured. Never automatically label an uncertain negative.
    if row["requested_intent"] != "wake" and any(
            any(keyword in text for keyword in WAKE_TEXTS) for text in texts):
        return "NEGATIVE_WAKE_CONFUSION_REVIEW"
    if not any(texts):
        return "ASR_ALL_EMPTY_UNVERIFIED"
    if all(text == requested for text in texts):
        return SCREEN_MATCH
    if row["requested_intent"] == "wake":
        return "WAKE_TEXT_MISS_OR_DISAGREEMENT"
    if any(not text for text in texts):
        return "ASR_PARTIAL_EMPTY_OR_DISAGREEMENT"
    return "NEGATIVE_TEXT_MISS_OR_DISAGREEMENT"


def audit(campaign_root: Path, screen_path: Path | None = None) -> dict:
    root = campaign_root.resolve(strict=True)
    manifest = root / "manifest.jsonl"
    summary_path = root / "summary.json"
    require(manifest.is_file() and summary_path.is_file(), "completed clone manifest/summary missing")
    summary = read_json(summary_path)
    require(summary.get("policy") == POLICY and summary.get("training_admitted") is False
            and summary.get("qualification_allowed") is False,
            "clone summary is not the expected unadmitted generation receipt")
    require(valid_sha(summary.get("campaign_sha256")) and
            summary.get("manifest_sha256") == sha256_file(manifest),
            "clone summary/manifest identity mismatch")
    rows = read_jsonl(manifest)
    require(len(rows) == summary.get("recordings") and type(summary.get("recordings")) is int
            and 0 < len(rows) <= 32768, "clone manifest count mismatch")
    screens = load_screen(screen_path)
    ids, paths, reference_owner, speaker_family, source_groups = set(), set(), {}, {}, defaultdict(set)
    pcm_owner, all_counts, per_intent = {}, Counter(), defaultdict(Counter)
    per_family, per_text, per_challenge = (defaultdict(Counter), defaultdict(Counter),
                                           defaultdict(Counter))
    gaps = []
    for row in rows:
        require(row.get("schema_version") == 1 and
                row.get("evidence_class") == "unreviewed-synthetic-voice-clone-v1" and
                row.get("synthetic") is True and
                row.get("training_admitted") is False and
                row.get("qualification_allowed") is False and
                row.get("campaign_sha256") == summary["campaign_sha256"] and
                row.get("human_gold") is not True and
                row.get("evaluation_gold_allowed") is not True,
                "row has invalid synthetic identity or unauthorized authority claim")
        item, speaker, family = (row.get(key) for key in
                                 ("item_id", "speaker_id", "source_family_id"))
        require(all(isinstance(v, str) and v.strip() for v in (item, speaker, family))
                and row.get("source_id") == item and item not in ids,
                "invalid/duplicate source and speaker identities")
        ids.add(item)
        require(row.get("requested_intent") in KINDS, "unknown requested intent")
        challenge = row.get("challenge_family", "unclassified_legacy")
        if challenge != "unclassified_legacy":
            require(isinstance(challenge, str) and challenge in CHALLENGE_FAMILIES[row["requested_intent"]],
                    "challenge family conflicts with declared intent")
        require("challenge_family" not in row or challenge != "unclassified_legacy",
                "reserved legacy challenge family cannot be supplied as provenance")
        challenge_key = row["requested_intent"] + ":" + challenge
        requested = clean_text(row.get("text_requested"))
        if row["requested_intent"] == "wake":
            require(requested in WAKE_TEXTS, "wake text does not match either product keyword")
        else:
            require(all(wake not in requested for wake in WAKE_TEXTS),
                    "negative request contains a full wake phrase")
        reference = row.get("reference_audio_sha256")
        require(valid_sha(reference) and valid_sha(row.get("rights_evidence_sha256"))
                and valid_sha(row.get("provider_identity_sha256"))
                and row.get("derivation_family_id") == "clone-" + reference,
                "reference/rights/provider lineage is missing or inconsistent")
        if speaker in speaker_family:
            require(speaker_family[speaker] == family, "speaker reassigned between source families")
        speaker_family[speaker] = family
        if reference in reference_owner:
            require(reference_owner[reference] == speaker,
                    "one reference cannot represent two original speakers")
        reference_owner[reference] = speaker
        source_groups[family].add(speaker)
        wav = private_audio_path(root, row.get("audio"))
        require(str(wav) not in paths, "duplicate generated WAV path")
        paths.add(str(wav))
        actual = inspect_pcm16_wav(wav)
        require(all(actual[key] == row.get(key) for key in
                    ("file_sha256", "pcm_sha256", "frames")) and
                type(row.get("duration_s")) in (float, int) and
                math.isclose(actual["duration_s"], row["duration_s"], abs_tol=1e-9),
                "generated PCM/file identity mismatch")
        pcm = actual["pcm_sha256"]
        if pcm in pcm_owner:
            require(pcm_owner[pcm] == (family, speaker),
                    "identical decoded PCM duplicated across original speakers/families")
            all_counts["duplicate_pcm_within_speaker"] += 1
        else:
            pcm_owner[pcm] = (family, speaker)
        screen = screens.get(item)
        if screen is not None:
            require(screen["file_sha256"] == actual["file_sha256"],
                    "ASR screen is bound to a different WAV")
        state = classify(row, screen)
        all_counts[state] += 1
        per_intent[row["requested_intent"]][state] += 1
        per_family[family][state] += 1
        per_text[requested][state] += 1
        per_challenge[challenge_key][state] += 1
    require(set(screens) <= ids, "ASR screen contains non-campaign item")
    wake_coverage = {wake: sum(1 for row in rows if row["requested_intent"] == "wake"
                              and clean_text(row["text_requested"]) == wake)
                     for wake in sorted(WAKE_TEXTS)}
    if any(not count for count in wake_coverage.values()):
        gaps.append("MISSING_PRODUCT_KEYWORD_GENERATION")
    if not screens or len(screens) < len(rows):
        gaps.append("THREE_ASR_SCREENING_INCOMPLETE")
    if all_counts["WAKE_TEXT_MISS_OR_DISAGREEMENT"]:
        gaps.append("POSITIVE_PRONUNCIATION_YIELD_NEEDS_IMPROVEMENT")
    if all_counts["NEGATIVE_WAKE_CONFUSION_REVIEW"]:
        gaps.append("NEGATIVE_WAKE_CONFUSION_NEEDS_REVIEW")
    if all_counts["ASR_ALL_EMPTY_UNVERIFIED"] or all_counts["ASR_PARTIAL_EMPTY_OR_DISAGREEMENT"]:
        gaps.append("EMPTY_ASR_SCREEN_IS_NOT_SILENCE_EVIDENCE")
    if len(source_groups) == 1:
        gaps.append("SINGLE_SOURCE_FAMILY_NO_CROSS_FAMILY_GENERALIZATION")
    # This tool has no real-human/AFE/continuous-stream/board results. Their
    # missingness must never be converted into a fabricated PASS.
    gaps.extend(["INDEPENDENT_REAL_HUMAN_FINAL_AFE_NOT_EVALUATED",
                 "CONTINUOUS_FAR_FRR_NOT_EVALUATED",
                 "SSC305_HARDWARE_NOT_EVALUATED"])
    return {
        "schema_version": 1, "policy": AUDIT_POLICY,
        "campaign_sha256": summary["campaign_sha256"], "manifest_sha256": summary["manifest_sha256"],
        "asr_screen_sha256": sha256_file(screen_path) if screen_path else None,
        "total_generated": len(rows), "original_speaker_roster": len(speaker_family),
        "source_family_roster": len(source_groups), "unique_output_pcm": len(pcm_owner),
        "wake_keyword_requested_counts": wake_coverage,
        "screened": len(screens), "screen_counts": dict(sorted(all_counts.items())),
        "screen_by_intent": {key: dict(sorted(per_intent[key].items())) for key in KINDS},
        "screen_by_source_family": {key: dict(sorted(value.items()))
                                    for key, value in sorted(per_family.items())},
        "screen_by_requested_text": {key: dict(sorted(value.items()))
                                     for key, value in sorted(per_text.items())},
        "screen_by_challenge_family": {key: dict(sorted(value.items()))
                                       for key, value in sorted(per_challenge.items())},
        "gaps": sorted(set(gaps)),
        "label_authentication_verified": False,
        "training_admitted": False, "independent_real_speakers_verified": False,
        "qualification_allowed": False, "shipping_approved": False,
        "conclusion": "SCREENING_ONLY_NOT_TRAINING_OR_QUALIFICATION",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--asr-screen", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="new private JSON summary; existing files are never overwritten")
    args = parser.parse_args(argv)
    result = audit(args.campaign_root, args.asr_screen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"policy": AUDIT_POLICY, "conclusion": result["conclusion"],
                      "screened": result["screened"], "gaps": result["gaps"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
