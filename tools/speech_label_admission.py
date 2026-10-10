"""Fail-closed labels for future speech bases; receipts are supplied, never generated.

The bounded mapper deliberately rejects unsupported speech rather than silently
turning OOV text into an empty CTC target. This validates receipt consistency, not
whether a human really listened or whether an asserted speaker is authentic.
"""
from __future__ import annotations

import hashlib
import json
import re

REAL_MODE = "reviewed-real-v1"
FIXTURE_MODE = "synthetic-fixture-v1"
HISTORICAL_MODE = "historical-frozen-diagnostic-v1"
MODES = (REAL_MODE, FIXTURE_MODE, HISTORICAL_MODE)
REVIEW_CLASS = "speech-like-audio-review-v2"
TRANSCRIPT_POLICY = "xiaowo-four-syllable-transcript-v1"
CHAR_TOKENS = {"你": "ni3", "好": "hao3", "小": "xiao3", "窝": "wo1"}
PUNCTUATION = frozenset("。，！？、；：…,.!?;: \t\n\r")
LINEAGE_FIELDS = ("source_family_id", "speaker_id", "reference_audio_sha256", "session_id", "derivation_family_id")
WAKE_TOKENS = {1: ["ni3", "hao3", "xiao3", "wo1"], 2: ["xiao3", "wo1", "xiao3", "wo1"]}
SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def actual_tokens(text: object) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("actual_text must be text")
    result = []
    for char in text:
        if char in PUNCTUATION:
            continue
        if char not in CHAR_TOKENS:
            raise ValueError(f"actual_text has unsupported/OOV transcript character: {char!r}; exclude or extend the reviewed transcript policy")
        result.append(CHAR_TOKENS[char])
    return result


def validate_review_record(row: dict) -> None:
    if type(row.get("schema_version")) is not int or row["schema_version"] != 2 or row.get("evidence_class") != REVIEW_CLASS:
        raise ValueError("audio review identity mismatch; future training requires review v2")
    for field in ("source_id", "reviewer_id", "source_family_id", "intended_text"):
        require_text(row.get(field), f"review.{field}")
    for field in ("file_sha256", "pcm_sha256"):
        require_sha(row.get(field), f"review.{field}")
    revision = row.get("review_revision")
    if type(revision) is not int or revision < 1:
        raise ValueError("review_revision must be a positive integer")
    if revision == 1:
        if "supersedes_review_sha256" not in row or row["supersedes_review_sha256"] is not None:
            raise ValueError("first review must explicitly supersede null")
    else:
        require_sha(row.get("supersedes_review_sha256"), "review.supersedes_review_sha256")
    if row.get("review_origin") != "human" or row.get("allowed_purpose") != "ctc-training-only":
        raise ValueError("audio review must be human-origin and allowed for CTC training; fixtures cannot promote")
    if row.get("transcript_policy") != TRANSCRIPT_POLICY:
        raise ValueError("audio review transcript policy mismatch")
    if row.get("verdict") not in {"accepted", "rejected", "uncertain"}:
        raise ValueError("audio review verdict is invalid")
    if type(row.get("acoustic_complete")) is not bool or type(row.get("speech_present")) is not bool:
        raise ValueError("audio review requires explicit acoustic_complete and speech_present booleans")
    if "keyword_id" not in row:
        raise ValueError("audio review keyword_id is missing")
    kind = row.get("kind")
    keyword = row["keyword_id"]
    if kind not in {"positive", "confusable", "negative", "background"}:
        raise ValueError("audio review kind is invalid")
    if kind == "positive":
        if type(keyword) is not int or keyword < 1:
            raise ValueError("audio review positive keyword_id must be a positive integer")
    elif keyword is not None:
        raise ValueError("audio review non-positive keyword_id must be null")
    names = row.get("actual_tokens")
    if not isinstance(names, list) or any(not isinstance(item, str) for item in names):
        raise ValueError("actual_tokens must be a string list")
    if not isinstance(row.get("actual_text"), str):
        raise ValueError("actual_text must be text, including for rejected observations")
    # Rejected/uncertain revisions preserve what was heard (including OOV) so a
    # correction can supersede them. Only accepted claims can become CTC labels.
    if row["verdict"] == "accepted":
        if names != actual_tokens(row["actual_text"]):
            raise ValueError("actual_text and actual_tokens disagree")
        if kind == "background":
            if names or row["actual_text"] != "" or row["speech_present"] is not False:
                raise ValueError("background requires reviewed absence of speech and empty actual text/targets")
        elif not names or row["speech_present"] is not True:
            raise ValueError("speech requires complete non-empty actual transcription; unknown is not blank")
        if kind == "positive" and WAKE_TOKENS.get(keyword) != names:
            raise ValueError("positive actual target differs from canonical keyword_id")
        if kind in {"negative", "confusable"}:
            for wake in WAKE_TOKENS.values():
                position = 0
                for token in names:
                    if token == wake[position]:
                        position += 1
                        if position == len(wake):
                            raise ValueError("negative actual target contains a canonical wake path")
    for field in LINEAGE_FIELDS:
        if field in row:
            require_sha(row[field], f"review.{field}") if field.endswith("sha256") else require_text(row[field], f"review.{field}")


def latest_review(chain: list[dict], *, require_accepted: bool = True) -> dict:
    if not isinstance(chain, list) or not chain or any(not isinstance(row, dict) for row in chain):
        raise ValueError("audio review revision history must be a non-empty object list")
    ordered = sorted(chain, key=lambda row: row.get("review_revision", 0) if type(row.get("review_revision")) is int else 0)
    identity = None
    previous = None
    for revision, row in enumerate(ordered, 1):
        validate_review_record(row)
        current = (row["source_id"], row["file_sha256"], row["pcm_sha256"])
        if identity is not None and current != identity:
            raise ValueError("review revision changes recording identity")
        identity = current
        if row["review_revision"] != revision or row["supersedes_review_sha256"] != previous:
            raise ValueError("audio review revision chain is incomplete, duplicate, stale, or ambiguous")
        previous = canonical_sha256(row)
    latest = ordered[-1]
    if require_accepted and (latest["verdict"] != "accepted" or latest["acoustic_complete"] is not True):
        raise ValueError("latest audio review is not accepted/acoustically complete")
    return latest


def admission_from_reviews(chain: list[dict], review_sha256: str) -> dict:
    latest = latest_review(chain)
    return {"mode": REAL_MODE, "allowed_purpose": "ctc-training-only", "ctc_training_allowed": True,
            "review_sha256": require_sha(review_sha256, "review_sha256"),
            "review_record_sha256": canonical_sha256(latest),
            "review_history": sorted(chain, key=lambda row: row["review_revision"])}


def fixture_admission() -> dict:
    return {"mode": FIXTURE_MODE, "allowed_purpose": "synthetic-contract-test-only", "ctc_training_allowed": False}


def validate_admission(admission: object, *, mode: str, source_id: str,
                       file_sha256: str, pcm_sha256: str, text: object, tokens: object,
                       kind: str, keyword_id: object, provenance: dict) -> None:
    if not isinstance(admission, dict) or admission.get("mode") != mode:
        raise ValueError("mandatory audio admission mode is missing or mismatched")
    if mode == FIXTURE_MODE:
        if admission != fixture_admission():
            raise ValueError("synthetic fixture admission cannot promote to training")
        return
    if mode != REAL_MODE:
        raise ValueError("historical frozen data cannot be newly materialized")
    if admission.get("ctc_training_allowed") is not True or admission.get("allowed_purpose") != "ctc-training-only":
        raise ValueError("audio admission is not allowed for CTC training")
    require_sha(admission.get("review_sha256"), "admission.review_sha256")
    latest = latest_review(admission.get("review_history"))
    if canonical_sha256(latest) != admission.get("review_record_sha256"):
        raise ValueError("audio admission review record hash mismatch")
    for field, value in (("source_id", source_id), ("file_sha256", file_sha256), ("pcm_sha256", pcm_sha256),
                         ("actual_text", text), ("actual_tokens", tokens), ("kind", kind), ("keyword_id", keyword_id)):
        if latest.get(field) != value:
            raise ValueError(f"audio admission {field} mismatch")
    for field in LINEAGE_FIELDS:
        if latest.get(field) != provenance.get(field):
            raise ValueError(f"audio admission {field} mismatch")
