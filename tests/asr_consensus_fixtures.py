"""Invented metadata only. No real ASR, human observations, or audio evidence."""
from speech_label_admission import TRANSCRIPT_POLICY, canonical_sha256
from speech_asr_consensus import (CONSENSUS_CLASS, POLICY_CLASS, RESULT_CLASS,
                                  VERIFICATION_CLASS, NORMALIZATION)


def consensus_fixture(source=None):
    source = source or {}
    policy = {"schema_version": 1, "evidence_class": POLICY_CLASS,
              "normalization": NORMALIZATION, "frozen_before_recognition": True,
              "target_conditioning_allowed": False, "recognizers": []}
    for index in range(3):
        config = {"language": "zh", "task": "transcribe", "beam_size": 1,
                  "temperature": 0, "prompt": "", "hotwords": []}
        policy["recognizers"].append({"recognizer_id": f"invented-recognizer-{index}",
            "provider_id": f"invented-provider-{index}", "model_id": f"invented-model-{index}",
            "model_revision": "fixture-revision", "model_family": f"invented-family-{index}",
            "model_sha256": str(index + 1) * 64, "native_decode_config_sha256": str(index + 6) * 64,
            "decode_config": config,
            "decode_config_sha256": canonical_sha256(config)})
    record = {"schema_version": 1, "evidence_class": CONSENSUS_CLASS,
              "label_origin": "machine-consensus", "allowed_purpose": "ctc-training-only",
              "qualification_allowed": False, "evaluation_gold_allowed": False, "split_role": "train",
              "source_provenance": {"synthetic": True, "source_type": "synthetic-speech",
                  "source_role": "train", "license_id": source.get("license_id", "fixture-only"),
                  "rights_evidence_sha256": "9" * 64},
              "source_id": source.get("source_id", "invented-source"),
              "source_family_id": source.get("source_family_id", "invented-source-family"),
              "file_sha256": source.get("file_sha256", "a" * 64),
              "pcm_sha256": source.get("pcm_sha256", "b" * 64),
              "audio_frames": source.get("frames", 160), "intended_text": "你好小窝",
              "actual_text": "你好小窝", "actual_tokens": ["ni3", "hao3", "xiao3", "wo1"],
              "transcript_policy": TRANSCRIPT_POLICY, "kind": "positive", "keyword_id": 1,
              "recognizer_policy": policy, "recognizer_policy_sha256": canonical_sha256(policy),
              "human_review_history": [], "human_conflict": False,
              "historical_human_conflict": False, "observations": []}
    for index, identity in enumerate(policy["recognizers"]):
        record["observations"].append({"schema_version": 1, "evidence_class": RESULT_CLASS,
            "recognizer_id": identity["recognizer_id"], "recognizer_identity_sha256": canonical_sha256(identity),
            "recognizer_policy_sha256": record["recognizer_policy_sha256"],
            **{key: record[key] for key in ("source_id", "file_sha256", "pcm_sha256", "audio_frames")},
            "start_frame": 0, "end_frame": record["audio_frames"], "status": "success", "origin": "asr",
            "transcript_complete": True, "coverage_complete": True, "speech_present": True,
            "request_id": f"invented-request-{index}", "raw_output_sha256": str(index + 4) * 64,
            "actual_text": "你好，小窝。" if index == 0 else "你好小窝"})
    record["authentication_verification"] = {"schema_version": 1, "evidence_class": VERIFICATION_CLASS,
        "verified": True, "subject_kind": CONSENSUS_CLASS, "issuer": "invented-test-verifier",
        "trust_policy": "invented-test-policy-not-authentication", "verified_at_utc": "2026-10-11T00:00:00Z",
        "verification_evidence_sha256": "e" * 64,
        "recognizer_policy_sha256": record["recognizer_policy_sha256"],
        "observations_sha256": canonical_sha256(record["observations"])}
    return record


def rebind_fixture(record):
    """Rebind hashes after invented test mutations, never authenticate evidence."""
    policy = record["recognizer_policy"]
    record["recognizer_policy_sha256"] = canonical_sha256(policy)
    for observation, identity in zip(record["observations"], policy["recognizers"]):
        observation["recognizer_id"] = identity["recognizer_id"]
        observation["recognizer_identity_sha256"] = canonical_sha256(identity)
        observation["recognizer_policy_sha256"] = record["recognizer_policy_sha256"]
    record["authentication_verification"]["recognizer_policy_sha256"] = record["recognizer_policy_sha256"]
    record["authentication_verification"]["observations_sha256"] = canonical_sha256(record["observations"])
    return record
