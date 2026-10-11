"""Training-only three-ASR receipts. This module never runs a recognizer.

A future trusted external verifier must authenticate the pinned recognizer
identities and captured outputs. Here we only validate a supplied report; hashes
and a typed ``verified`` flag are not themselves an authentication mechanism.
No machine receipt is a human observation or qualification reference.
"""
from __future__ import annotations

import copy
import datetime
import math

from speech_label_admission import (LINEAGE_FIELDS, PUNCTUATION, REAL_MODE,
                                    TRANSCRIPT_POLICY, WAKE_TOKENS, actual_tokens,
                                    canonical_sha256, latest_review, require_sha,
                                    require_text)

CONSENSUS_CLASS = "speech-three-asr-consensus-v1"
ADMISSION_CLASS = "speech-three-asr-training-admission-v1"
POLICY_CLASS = "speech-three-asr-recognizer-policy-v1"
RESULT_CLASS = "speech-asr-complete-result-v1"
VERIFICATION_CLASS = "speech-asr-authentication-verification-v1"
NORMALIZATION = "fixed-punctuation-removal-only-v1"
LABEL_SOURCE = "machine-three-asr-consensus-v1"


def normalized_transcript(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("ASR transcript must be text")
    # No synonym, homophone, simplified/traditional, case, or Unicode folding.
    return "".join(char for char in value if char not in PUNCTUATION)


def _identity(value: dict, version: int, evidence: str, label: str) -> None:
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != version or value.get("evidence_class") != evidence):
        raise ValueError(f"{label} schema/evidence identity mismatch")


def validate_recognizer_policy(policy: dict) -> dict[str, dict]:
    _identity(policy, 1, POLICY_CLASS, "ASR policy")
    if (policy.get("normalization") != NORMALIZATION
            or policy.get("frozen_before_recognition") is not True
            or policy.get("target_conditioning_allowed") is not False):
        raise ValueError("ASR policy must be frozen with conservative normalization and no target conditioning")
    recognizers = policy.get("recognizers")
    if not isinstance(recognizers, list) or len(recognizers) != 3:
        raise ValueError("ASR policy requires exactly three recognizers")
    identities, models, artifacts, families = {}, set(), set(), set()
    for row in recognizers:
        if not isinstance(row, dict):
            raise ValueError("ASR recognizer identity must be an object")
        for field in ("recognizer_id", "provider_id", "model_id", "model_revision", "model_family"):
            require_text(row.get(field), f"recognizer.{field}")
        artifact = require_sha(row.get("model_sha256"), "recognizer.model_sha256")
        require_sha(row.get("native_decode_config_sha256"), "recognizer.native_decode_config_sha256")
        config = row.get("decode_config")
        if not isinstance(config, dict) or set(config) != {
            "language", "task", "beam_size", "temperature", "prompt", "hotwords",
        }:
            raise ValueError("ASR decode config must use the complete frozen v1 fields")
        if (config["language"] != "zh" or config["task"] != "transcribe"
                or config["prompt"] != "" or config["hotwords"] != []):
            raise ValueError("ASR decoding cannot use prompts, target hotwords, or translation")
        if type(config["beam_size"]) is not int or config["beam_size"] < 1:
            raise ValueError("ASR beam_size must be a positive integer")
        temperature = config["temperature"]
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("ASR temperature must be finite and nonnegative")
        if row.get("decode_config_sha256") != canonical_sha256(config):
            raise ValueError("ASR decode config hash mismatch")
        # Repeated passes, aliases, versions, or prompts of one model do not vote.
        model = (row["provider_id"], row["model_id"])
        if (row["recognizer_id"] in identities or model in models or artifact in artifacts
                or row["model_family"] in families):
            raise ValueError("ASR consensus requires three distinct recognizers, models, and model families")
        identities[row["recognizer_id"]] = row
        models.add(model)
        artifacts.add(artifact)
        families.add(row["model_family"])
    return identities


def human_conflicts(history: list[dict], record: dict) -> tuple[bool, bool]:
    if not isinstance(history, list):
        raise ValueError("ASR consensus must explicitly retain supplied human review history")
    if not history:
        return False, False
    latest = latest_review(history, require_accepted=False)
    conflicts = []
    for review in history:
        for field in ("source_id", "file_sha256", "pcm_sha256", "intended_text", *LINEAGE_FIELDS):
            if review.get(field) != record.get(field):
                raise ValueError(f"ASR/human review {field} mismatch")
        conflicts.append(review["verdict"] != "accepted" or not review["acoustic_complete"]
                         or normalized_transcript(review["actual_text"]) != record.get("actual_text")
                         or review["actual_tokens"] != record.get("actual_tokens")
                         or review["kind"] != record.get("kind")
                         or review["keyword_id"] != record.get("keyword_id"))
    return conflicts[history.index(latest)], any(conflicts)


def validate_consensus_record(record: dict) -> None:
    _identity(record, 1, CONSENSUS_CLASS, "ASR consensus")
    if (record.get("label_origin") != "machine-consensus"
            or record.get("allowed_purpose") != "ctc-training-only"
            or record.get("qualification_allowed") is not False
            or record.get("evaluation_gold_allowed") is not False
            or record.get("split_role") != "train"):
        raise ValueError("machine consensus is training-only, never human/evaluation/qualification gold")
    for field in ("source_id", "source_family_id", "intended_text"):
        require_text(record.get(field), f"consensus.{field}")
    for field in ("file_sha256", "pcm_sha256"):
        require_sha(record.get(field), f"consensus.{field}")
    for field in LINEAGE_FIELDS:
        if field in record:
            (require_sha if field.endswith("sha256") else require_text)(record[field], f"consensus.{field}")
    provenance = record.get("source_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "synthetic", "source_type", "source_role", "license_id", "rights_evidence_sha256",
    }:
        raise ValueError("ASR consensus must bind source origin, role, license, and rights evidence")
    if (type(provenance["synthetic"]) is not bool or provenance["source_role"] != "train"
            or provenance["source_type"] != ("synthetic-speech" if provenance["synthetic"] else "recorded-speech")):
        raise ValueError("ASR labels cannot promote fixture/screen/heldout sources or relabel their origin")
    require_text(provenance["license_id"], "source license_id")
    require_sha(provenance["rights_evidence_sha256"], "source rights_evidence_sha256")
    forbidden = {"human_gold", "gold", "review_origin", "reviewer_id", "acoustic_complete", "source_screen_only"}
    if forbidden.intersection(record):
        raise ValueError("machine consensus cannot carry human-gold, human-completeness, or source-screen claims")
    frames = record.get("audio_frames")
    if type(frames) is not int or frames <= 0:
        raise ValueError("ASR source audio_frames must be a positive integer")
    if record.get("transcript_policy") != TRANSCRIPT_POLICY:
        raise ValueError("ASR consensus transcript policy mismatch")
    policy = record.get("recognizer_policy")
    identities = validate_recognizer_policy(policy)
    policy_sha = canonical_sha256(policy)
    if record.get("recognizer_policy_sha256") != policy_sha:
        raise ValueError("ASR recognizer policy hash mismatch")
    observations = record.get("observations")
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("ASR consensus requires exactly three actual outputs; missing output is not a vote")
    seen, transcripts = set(), []
    for observation in observations:
        _identity(observation, 1, RESULT_CLASS, "ASR output")
        recognizer_id = observation.get("recognizer_id")
        if not isinstance(recognizer_id, str) or recognizer_id not in identities or recognizer_id in seen:
            raise ValueError("ASR outputs must cover each distinct frozen recognizer exactly once")
        seen.add(recognizer_id)
        identity = identities[recognizer_id]
        if (observation.get("recognizer_identity_sha256") != canonical_sha256(identity)
                or observation.get("recognizer_policy_sha256") != policy_sha):
            raise ValueError("ASR output recognizer/model/decode identity differs from frozen policy")
        for field in ("source_id", "file_sha256", "pcm_sha256", "audio_frames"):
            if observation.get(field) != record[field] or type(observation.get(field)) is not type(record[field]):
                raise ValueError(f"ASR output {field} differs from exact source recording")
        if (type(observation.get("start_frame")) is not int or observation["start_frame"] != 0
                or type(observation.get("end_frame")) is not int or observation["end_frame"] != frames
                or observation.get("status") != "success" or observation.get("origin") != "asr"
                or observation.get("transcript_complete") is not True
                or observation.get("coverage_complete") is not True
                or observation.get("speech_present") is not True):
            raise ValueError("ASR vote must be successful complete-source speech transcription, not partial/human/silence")
        require_text(observation.get("request_id"), "ASR request_id")
        require_sha(observation.get("raw_output_sha256"), "ASR raw_output_sha256")
        transcripts.append(normalized_transcript(observation.get("actual_text")))
    if len(set(transcripts)) != 1 or not transcripts[0]:
        raise ValueError("ASR transcripts are not unanimous complete speech; quarantine")
    if record.get("actual_text") != transcripts[0]:
        raise ValueError("ASR consensus text differs from conservative unanimous transcript")
    names = actual_tokens(transcripts[0])
    if record.get("actual_tokens") != names or not names:
        raise ValueError("ASR consensus tokens disagree with actual transcript; voiced OOV is not blank")
    kind, keyword = record.get("kind"), record.get("keyword_id")
    if kind == "positive":
        if type(keyword) is not int or WAKE_TOKENS.get(keyword) != names:
            raise ValueError("ASR positive target differs from canonical keyword")
    elif kind in {"negative", "confusable"} and keyword is None:
        for wake in WAKE_TOKENS.values():
            position = 0
            for token in names:
                if token == wake[position]:
                    position += 1
                    if position == len(wake):
                        raise ValueError("ASR negative target contains a canonical wake path")
    else:
        raise ValueError("ASR consensus supports speech labels only; silence/background needs independent review")
    if "authentication_verification" not in record:
        raise ValueError("ASR external authentication report must be explicitly supplied or null")
    verification = record["authentication_verification"]
    if verification is not None:
        _identity(verification, 1, VERIFICATION_CLASS, "ASR external authentication")
        if (verification.get("verified") is not True or verification.get("subject_kind") != CONSENSUS_CLASS
                or verification.get("recognizer_policy_sha256") != policy_sha
                or verification.get("observations_sha256") != canonical_sha256(observations)):
            raise ValueError("ASR external authentication report must bind actual outputs and frozen identities")
        for field in ("issuer", "trust_policy", "verified_at_utc"):
            require_text(verification.get(field), f"authentication.{field}")
        require_sha(verification.get("verification_evidence_sha256"), "authentication.verification_evidence_sha256")
        stamp = verification["verified_at_utc"]
        try:
            parsed = datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("ASR authentication timestamp must be valid UTC") from exc
        if not stamp.endswith("Z") or parsed.utcoffset() != datetime.timedelta(0):
            raise ValueError("ASR authentication timestamp must be UTC")
    current, historical = human_conflicts(record.get("human_review_history"), record)
    if record.get("human_conflict") is not current or record.get("historical_human_conflict") is not historical:
        raise ValueError("ASR/human conflict flags differ from preserved observations")
    if current:
        raise ValueError("ASR consensus conflicts with latest human observation; quarantine without rewriting history")


def admission_from_consensus(record: dict, receipt_sha256: str) -> dict:
    validate_consensus_record(record)
    return {"schema_version": 1, "evidence_class": ADMISSION_CLASS, "mode": REAL_MODE,
            "allowed_purpose": "ctc-training-only", "ctc_training_allowed": False,
            "label_eligible": True, "status": "PENDING_EXTERNAL_VERIFICATION",
            "qualification_allowed": False, "evaluation_gold_allowed": False,
            "label_origin": "machine-consensus", "label_source": LABEL_SOURCE,
            "authentication_independently_verified": False,
            "consensus_receipt_sha256": require_sha(receipt_sha256, "consensus_receipt_sha256"),
            "consensus_record_sha256": canonical_sha256(record),
            "consensus_record": copy.deepcopy(record)}


def validate_consensus_admission(admission: dict, *, split: object, frames: object = None,
                                 for_training: bool = False) -> dict:
    _identity(admission, 1, ADMISSION_CLASS, "ASR admission")
    if split != "train":
        raise ValueError("machine-consensus labels require train role; evaluation/qualification gold is forbidden")
    record = admission.get("consensus_record")
    validate_consensus_record(record)
    expected = admission_from_consensus(record, admission.get("consensus_receipt_sha256"))
    if admission != expected:
        raise ValueError("ASR admission fields/hash/purpose differ from validated machine receipt")
    if frames is not None and (type(frames) is not int or record["audio_frames"] != frames):
        raise ValueError("ASR complete-transcript frame coverage differs from measured source")
    if for_training:
        # No trusted out-of-band verifier is connected here. Never turn a
        # caller-authored JSON assertion or matching checksum into authority.
        raise ValueError("PENDING_EXTERNAL_VERIFICATION: machine consensus is label-eligible, not training-admitted")
    return record


def has_machine_consensus(value: object) -> bool:
    """Recognize nested provenance even when a caller changes one outer tag."""
    if isinstance(value, list):
        return any(has_machine_consensus(item) for item in value)
    if not isinstance(value, dict):
        return False
    if (value.get("evidence_class") in (CONSENSUS_CLASS, ADMISSION_CLASS)
            or value.get("label_origin") == "machine-consensus"
            or value.get("label_source") == LABEL_SOURCE or "consensus_record" in value
            or value.get("status") == "PENDING_EXTERNAL_VERIFICATION"):
        return True
    return any(has_machine_consensus(value.get(key)) for key in
               ("admission", "speech_like_provenance", "source_provenance", "label_provenance",
                "consensus_record", "review_history", "human_review_history"))


def require_independent_reference_labels(rows: list[dict]) -> None:
    """Machine-only labels may be inspected diagnostically, never qualification gold."""
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("qualification references must be an object list")
    for row in rows:
        if has_machine_consensus(row):
            raise ValueError("qualification requires independent reference labels; machine consensus is training-label evidence only")


def require_training_label_authority(value: object) -> None:
    """Reject pending machine lineage before any projection can erase its tags."""
    if has_machine_consensus(value):
        raise ValueError("PENDING_EXTERNAL_VERIFICATION: machine label lineage cannot be admitted as human training evidence")
