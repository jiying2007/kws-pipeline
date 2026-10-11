"""Training-only three-ASR receipts. This module never runs a recognizer.

Structural JSON and caller-authored verification reports never grant authority.
The separate code-rooted resolver may establish a fresh per-process source proof
from retained source/rights/model/config/capture bytes. Production train roots
are currently empty; adding one requires a reviewed control-plane code update.
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
CONSENSUS_CLASS_V2 = "speech-three-asr-consensus-v2"
POLICY_CLASS_V2 = "speech-three-asr-recognizer-policy-v2"
RESULT_CLASS_V2 = "speech-asr-complete-result-v2"
ADMITTED_STATUS = "VERIFIED_SOURCE_TRAIN_LABEL"


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
    native_v2 = isinstance(policy, dict) and policy.get("evidence_class") == POLICY_CLASS_V2
    _identity(policy, 2 if native_v2 else 1, POLICY_CLASS_V2 if native_v2 else POLICY_CLASS, "ASR policy")
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
        if native_v2:
            from asr_native_contracts import validate_native_config
            config = row.get("native_decode_config")
            validate_native_config(config)
            if config["model_family"] != row["model_family"]:
                raise ValueError("native adapter family cannot be relabeled to manufacture independent votes")
            if row["native_decode_config_sha256"] != canonical_sha256(config):
                raise ValueError("ASR native configuration canonical hash mismatch")
            if "decode_config" in row or "decode_config_sha256" in row:
                raise ValueError("native-v2 cannot manufacture generic decoding defaults")
        else:
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
        if review.get("training_use_prohibited") is True or review.get("training_use_allowed") is False:
            raise ValueError("explicit human sample-use restriction forbids training")
        for field in ("source_id", "file_sha256", "pcm_sha256", "intended_text", *LINEAGE_FIELDS):
            if review.get(field) != record.get(field):
                raise ValueError(f"ASR/human review {field} mismatch")
        conflicts.append(review["verdict"] != "accepted" or not review["acoustic_complete"]
                         or normalized_transcript(review["actual_text"]) != record.get("actual_text")
                         or review["actual_tokens"] != record.get("actual_tokens")
                         or review["kind"] != record.get("kind")
                         or review["keyword_id"] != record.get("keyword_id"))
    return conflicts[history.index(latest)], any(conflicts)


def validate_raw_human_references(history: object) -> bool:
    """Only hashes/relative artifact references belong in persisted v2 labels."""
    if not isinstance(history, list):
        raise ValueError("native-v2 must explicitly retain raw human observation references")
    previous = None
    for revision, reference in enumerate(history, 1):
        _identity(reference, 1, "speech-raw-human-observation-reference-v1", "raw human reference")
        if set(reference) != {"schema_version", "evidence_class", "observation_revision",
                              "supersedes_observation_sha256", "observation_sha256", "artifact"}:
            raise ValueError("raw human reference cannot embed private note bodies or extra claims")
        if (type(reference["observation_revision"]) is not int or reference["observation_revision"] != revision
                or reference["supersedes_observation_sha256"] != previous):
            raise ValueError("raw human references must retain complete ordered revisions")
        previous = require_sha(reference["observation_sha256"], "raw human observation hash")
        artifact = reference["artifact"]
        if not isinstance(artifact, dict) or set(artifact) != {"archive", "member"}:
            raise ValueError("raw human observation must reference an exact retained artifact")
        for field in artifact:
            require_text(artifact[field], "raw human artifact reference")
    return bool(history)


def raw_human_conflicts(history: object, record: dict, *, typed_conflicts=None) -> tuple[bool, bool]:
    """Preserve raw listening observations without inventing training-review v2."""
    if not isinstance(history, list):
        raise ValueError("native-v2 must explicitly retain its supplied raw human observation history")
    if not history:
        return False, False
    previous = None
    conflicts = []
    ordered = sorted(history, key=lambda row: row.get("observation_revision", 0)
                     if isinstance(row, dict) and type(row.get("observation_revision")) is int else 0)
    for revision, observation in enumerate(ordered, 1):
        _identity(observation, 1, "speech-raw-human-observation-v1", "raw human observation")
        if (type(observation.get("observation_revision")) is not int or observation.get("observation_revision") != revision
                or observation.get("supersedes_observation_sha256") != previous):
            raise ValueError("raw human observation history is incomplete or forked")
        if {"verdict", "allowed_purpose", "actual_tokens", "acoustic_complete"}.intersection(observation):
            raise ValueError("raw human observations cannot fabricate training-review fields")
        for field in ("source_id", "file_sha256", "pcm_sha256"):
            if type(observation.get(field)) is not type(record[field]) or observation.get(field) != record[field]:
                raise ValueError("raw human observation belongs to another source")
        # A wrapper can bind preexisting unstructured bytes without rewriting the
        # note or manufacturing a human training-review receipt.
        if "original_observation_artifact" in observation:
            ref = observation["original_observation_artifact"]
            if not isinstance(ref, dict) or set(ref) != {"archive", "member"}:
                raise ValueError("original human note must reference retained bytes")
            require_sha(observation.get("original_observation_sha256"), "original human note hash")
            if "original_observation" in observation:
                raise ValueError("referenced original note must not be duplicated inline")
        else:
            require_text(observation.get("original_observation"), "original human observation")
        text = observation.get("observed_text")
        if text is not None and not isinstance(text, str):
            raise ValueError("raw heard words must remain text or unknown, including OOV")
        certainty = observation.get("certainty")
        if certainty not in {"clear", "uncertain", "unintelligible", "unknown"}:
            raise ValueError("raw human certainty is missing")
        if observation.get("training_use_prohibited") is True or observation.get("training_use_allowed") is False:
            raise ValueError("explicit raw human sample-use restriction forbids training")
        original_format = observation.get("original_format", "raw-note")
        if original_format == "human-review-v2":
            if not isinstance(typed_conflicts, dict) or revision not in typed_conflicts:
                raise ValueError("typed original human review requires its complete source-bound revision chain")
            conflicts.append(typed_conflicts[revision])
        elif original_format == "raw-note":
            conflicts.append(certainty != "clear" or text is None or normalized_transcript(text) != record["actual_text"])
        else:
            raise ValueError("unsupported original human observation format")
        previous = canonical_sha256(observation)
    return conflicts[-1], any(conflicts)


def validate_consensus_record(record: dict) -> None:
    native_v2 = isinstance(record, dict) and record.get("evidence_class") == CONSENSUS_CLASS_V2
    _identity(record, 2 if native_v2 else 1, CONSENSUS_CLASS_V2 if native_v2 else CONSENSUS_CLASS, "ASR consensus")
    if native_v2:
        if record.get("human_review_history") != []:
            raise ValueError("native-v2 human histories must use private source-bound references, not inline bodies")
        if record.get("acoustic_verbatim_completeness") != "unknown":
            raise ValueError("machine consensus cannot establish acoustic verbatim completeness")
        if record.get("label_selection_policy") != "trusted-three-asr-preferred-training-v2":
            raise ValueError("native-v2 requires explicit trusted-three-ASR training-label priority")
        if record.get("training_use_prohibited") is not False:
            raise ValueError("explicit training-use prohibition or unknown purpose blocks consensus")
    if record.get("training_use_prohibited") is True or record.get("training_use_allowed") is False:
        raise ValueError("explicit sample-use restriction forbids machine training")
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
    if (policy["evidence_class"] == POLICY_CLASS_V2) is not native_v2:
        raise ValueError("consensus and recognizer policy versions differ")
    policy_sha = canonical_sha256(policy)
    if record.get("recognizer_policy_sha256") != policy_sha:
        raise ValueError("ASR recognizer policy hash mismatch")
    observations = record.get("observations")
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("ASR consensus requires exactly three actual outputs; missing output is not a vote")
    seen, transcripts = set(), []
    for observation in observations:
        _identity(observation, 2 if native_v2 else 1, RESULT_CLASS_V2 if native_v2 else RESULT_CLASS, "ASR output")
        if native_v2 and {"acoustic_complete", "speech_present", "coverage_complete", "transcript_complete",
                          "human_gold", "review_origin"}.intersection(observation):
            raise ValueError("native-v2 output cannot manufacture human/speech/acoustic-completeness assertions")
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
                or (not native_v2 and (observation.get("transcript_complete") is not True
                    or observation.get("coverage_complete") is not True or observation.get("speech_present") is not True))
                or (native_v2 and (observation.get("input_coverage_complete") is not True
                    or observation.get("decoder_output_complete") is not True))):
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
        if (verification.get("verified") is not True or verification.get("subject_kind") != record["evidence_class"]
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
    if native_v2 and validate_raw_human_references(record.get("human_observation_history")):
        # Private original observations are resolved only inside the trusted
        # evidence reader. Structural JSON can retain flags but cannot verify
        # their content or act as source authority.
        for flag in ("human_conflict", "historical_human_conflict"):
            if type(record.get(flag)) is not bool:
                raise ValueError("native-v2 human conflict flags must be explicit booleans")
        if (current and not record["human_conflict"] or historical and not record["historical_human_conflict"]
                or record["human_conflict"] and not record["historical_human_conflict"]):
            raise ValueError("human conflict flags erase known review history")
        current, historical = record["human_conflict"], record["historical_human_conflict"]
    if record.get("human_conflict") is not current or record.get("historical_human_conflict") is not historical:
        raise ValueError("ASR/human conflict flags differ from preserved observations")
    if current and not native_v2:
        raise ValueError("ASR consensus conflicts with latest human observation; quarantine without rewriting history")


def admission_from_consensus(record: dict, receipt_sha256: str, *, verification_context=None) -> dict:
    validate_consensus_record(record)
    result = {"schema_version": 1, "evidence_class": ADMISSION_CLASS, "mode": REAL_MODE,
              "allowed_purpose": "ctc-training-only", "ctc_training_allowed": False,
              "label_eligible": True, "status": "PENDING_EXTERNAL_VERIFICATION",
              "qualification_allowed": False, "evaluation_gold_allowed": False,
              "label_origin": "machine-consensus", "label_source": LABEL_SOURCE,
              "authentication_independently_verified": False,
              "consensus_receipt_sha256": require_sha(receipt_sha256, "consensus_receipt_sha256"),
              "consensus_record_sha256": canonical_sha256(record), "consensus_record": copy.deepcopy(record)}
    if verification_context is not None:
        from asr_source_resolver import verify_source_context
        proof = verify_source_context(verification_context, record)
        result.update(ctc_training_allowed=True, status=ADMITTED_STATUS,
                      authentication_independently_verified=True, source_verification=proof)
    return result


def validate_consensus_admission(admission: dict, *, split: object, frames: object = None,
                                 for_training: bool = False, verification_context=None,
                                 evidence_dir=None, verification_session=None) -> dict:
    _identity(admission, 1, ADMISSION_CLASS, "ASR admission")
    if split != "train":
        raise ValueError("machine-consensus labels require train role; evaluation/qualification gold is forbidden")
    record = admission.get("consensus_record")
    validate_consensus_record(record)
    if admission.get("status") == ADMITTED_STATUS:
        from asr_source_resolver import resolve_source_context
        if verification_context is None:
            verification_context = resolve_source_context(record, evidence_dir=evidence_dir, session=verification_session)
    elif verification_context is not None:
        raise ValueError("pending receipt cannot silently become admitted during consumption")
    expected = admission_from_consensus(record, admission.get("consensus_receipt_sha256"),
                                       verification_context=verification_context)
    if canonical_sha256(admission) != canonical_sha256(expected):
        raise ValueError("ASR admission fields/hash/purpose differ from validated machine receipt")
    if frames is not None and (type(frames) is not int or record["audio_frames"] != frames):
        raise ValueError("ASR complete-transcript frame coverage differs from measured source")
    if for_training and expected["ctc_training_allowed"] is not True:
        raise ValueError("PENDING_EXTERNAL_VERIFICATION: machine consensus is label-eligible, not training-admitted")
    return record


def has_machine_consensus(value: object) -> bool:
    """Recognize nested provenance even when a caller changes one outer tag."""
    if isinstance(value, list):
        return any(has_machine_consensus(item) for item in value)
    if not isinstance(value, dict):
        return False
    if (value.get("evidence_class") in (CONSENSUS_CLASS, CONSENSUS_CLASS_V2, ADMISSION_CLASS)
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


def require_training_label_authority(value: object, *, allow_machine: bool = False,
                                     split=None, frames=None, evidence_dir=None, verification_session=None) -> None:
    """Never replace machine provenance with a human-looking outer receipt."""
    if not has_machine_consensus(value):
        return
    if not allow_machine or not isinstance(value, dict):
        raise ValueError("PENDING_EXTERNAL_VERIFICATION: machine label lineage cannot be admitted as human training evidence")
    admission = value.get("admission")
    if not isinstance(admission, dict) or admission.get("evidence_class") != ADMISSION_CLASS:
        raise ValueError("PENDING_EXTERNAL_VERIFICATION: machine source requires its original typed consensus admission")
    record = validate_consensus_admission(admission, split=split, frames=frames, for_training=True,
                                         evidence_dir=evidence_dir, verification_session=verification_session)
    provenance = value.get("speech_like_provenance", value)
    metadata_values = [value, provenance]
    for item in (value, provenance):
        metadata_values.extend(item[key] for key in ("source_provenance", "label_provenance") if key in item)
    for metadata in metadata_values:
        if not isinstance(metadata, dict):
            raise ValueError("machine source provenance must be an object")
        for field in ("role", "allowed_role", "split_role", "source_role", "split"):
            if field in metadata and metadata[field] != "train":
                raise ValueError("machine source/row role cannot be changed")
        if (metadata.get("source_screen_only") is True or metadata.get("human_gold") is True
                or metadata.get("review_origin") == "human"
                or metadata.get("training_use_prohibited") is True or metadata.get("training_use_allowed") is False):
            raise ValueError("machine source cannot become human gold or override restricted source use")
        for field, expected in (("label_origin", "machine-consensus"), ("label_source", LABEL_SOURCE)):
            if field in metadata and metadata[field] != expected:
                raise ValueError("machine label provenance cannot be rewritten")
        for field, expected in record["source_provenance"].items():
            if field in metadata and (metadata[field] != expected or type(metadata[field]) is not type(expected)):
                raise ValueError(f"machine source {field} differs from trusted origin/rights/role")
