"""Strict candidate-relative V7 synthetic machine admission.

No model inference, TTS, optimizer or human-gold claims occur here. The module
verifies the original PCM/WAV and four blind observer captures, then creates two
independent candidate manifests. A successful data snapshot is only eligible
for a separately budgeted G3 training decision; it never authorizes an optimizer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import unicodedata
import wave

SCHEMA = "english-kws-dual-track-v7-machine-admission-v1"
CORPUS_SCHEMA = "english-kws-dual-track-v7-kokoro-corpus-v1"
EVIDENCE_SCHEMA = "english-kws-dual-track-v7-kokoro-machine-evidence-v1"
CAPTURE_SCHEMA = "english-kws-dual-track-v7-kokoro-observer-capture-v1"
ROLES = ("train", "development")
CANDIDATES = ("VOSEN", "MERO")
IDENTITY_KEYS = ("recognizer_id", "model_family", "model_sha256", "decode_config_sha256")
SOURCE_KEYS = ("source_group_id", "source_speaker_id", "source_speaker_index", "voice_family", "role")
CELL_KEYS = ("cell_id", "requested_text", "requested_class", "pronunciation_hypothesis")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_REL = re.compile(r"[A-Za-z0-9_./-]+\Z")

MODEL_PINS = {
    "qwen": ("qwen3-asr", "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea",
             "6ef4ab0c3a329ac618980f42a00ebff601c08a768b8f341256b85ffaffb3bca1"),
    "whisper": ("whisper", "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
                "439b51d2e7e5f7ce4ade748a87c3d3e6d5b6b66d2acaec893bfc3bf673af04a7"),
    "parakeet": ("parakeet-tdt", "e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e",
                 "c83bcf97f4c3f718264df1ecac8f013a276bdf7024f3ed39b255f85ad4bb9b6d"),
    "phoneme": ("wav2vec2-xlsr", "04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551",
                "2b37c89e02afc6b44b8109a9ae38303eca75411564f53b3d61d86985de5af441"),
}

EXPECTED_CELLS = {
    "vosen-slow": ("Hey Vosen.", 1, "target", "heɪ vˈoʊsən", 1.08, {"VOSEN": 1, "MERO": 0}),
    "vosen-normal": ("Hey Vosen.", 1, "target", "heɪ vˈoʊsən", 1.0, {"VOSEN": 1, "MERO": 0}),
    "vosen-fast": ("Hey Vosen.", 1, "target", "heɪ vˈoʊsən", 0.92, {"VOSEN": 1, "MERO": 0}),
    "mero-slow": ("Hey Mero.", 2, "target", "heɪ mˈɛɹoʊ", 1.08, {"VOSEN": 0, "MERO": 1}),
    "mero-normal": ("Hey Mero.", 2, "target", "heɪ mˈɛɹoʊ", 1.0, {"VOSEN": 0, "MERO": 1}),
    "mero-fast": ("Hey Mero.", 2, "target", "heɪ mˈɛɹoʊ", 0.92, {"VOSEN": 0, "MERO": 1}),
    "bosen-s": ("Hey Bosen.", 0, "contrast-negative", "heɪ bˈoʊsən", 1.0, {"VOSEN": 0, "MERO": 0}),
    "bosen-z": ("Hey Bosen.", 0, "contrast-negative", "heɪ bˈoʊzən", 1.0, {"VOSEN": 0, "MERO": 0}),
    "memo": ("Hey Memo.", 0, "contrast-negative", "heɪ mˈɛmoʊ", 1.0, {"VOSEN": 0, "MERO": 0}),
    "nero": ("Hey Nero.", 0, "contrast-negative", "heɪ nˈɛɹoʊ", 1.0, {"VOSEN": 0, "MERO": 0}),
    "hello": ("Hello there.", 0, "ordinary-negative", "həlˈoʊ ðɛɹ", 1.0, {"VOSEN": 0, "MERO": 0}),
    "morning": ("Good morning.", 0, "ordinary-negative", "ɡʊd ˈmɔɹnɪŋ", 1.0, {"VOSEN": 0, "MERO": 0}),
}

ALIASES = {
    "VOSEN": ("hey vosen", "hey vohsen"),
    "MERO": ("hey mero", "hey merrow"),
}
EXCLUDED_TRANSCRIPTS = {
    "hey meero", "hay meero", "hey miro", "hay miro",
    "hey vozen", "hay vozen", "hey vohzen", "hay vohzen",
    "hey vozan", "hay vozan", "hey vozin", "hay vozin",
}
PHONE_PATTERNS = {
    "VOSEN": re.compile(r"h?eɪv(?:oʊ|ou|o)s[əɐʌ]n\Z"),
    "MERO": re.compile(r"h?eɪmɛr(?:oʊ|ou|o)\Z"),
}
ORDINARY_TEXTS = {"hello there", "good morning"}
RULES = {
    "rule_id": "english-kws-dual-track-v7-candidate-relative-v1",
    "normalization": "unicode-nfkc-casefold-punctuation-to-space-v1",
    "aliases": ALIASES,
    "excluded_transcript_spellings": sorted(EXCLUDED_TRANSCRIPTS),
    "phone_patterns": {key: value.pattern for key, value in PHONE_PATTERNS.items()},
    "phone_normalization": "join-tokens-strip-stress-length-and-ties-map-rhotics-v1",
    "positive_min_votes": 2,
    "silver_votes": 3,
    "weak_votes": 2,
    "cross_trigger_requires_other_phone_match": True,
    "negative_requires_explicit_expected_transcript_and_no_wake_phone": True,
    "unknown_is_background": False,
    "positive_min_samples": 6400,
    "positive_max_samples": 28672,
    "minimum_positive_rows_per_group": 2,
    "minimum_negative_rows_per_group": 4,
    "minimum_balanced_groups": {"train": 6, "development": 2},
    "tiers": {"silver": 1.0, "weak": 0.5},
    "locked_test_never_optimizer_or_mining": True,
}
RULES_SHA256 = hashlib.sha256(json.dumps(
    RULES, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode("utf-8")).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def normalized_text(text):
    require(isinstance(text, str), "TRANSCRIPT_MUST_BE_TEXT")
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(char).startswith("P") else char
                            for char in text).split())


def normalized_phones(tokens):
    require(isinstance(tokens, list) and all(isinstance(token, str) and token for token in tokens),
            "PHONE_TOKENS_REQUIRED")
    value = unicodedata.normalize("NFKC", "".join(tokens))
    value = re.sub(r"[\sˈˌːˑ͜͡]", "", value)
    return value.replace("ɹ", "r").replace("ɾ", "r")


def phone_candidate(tokens):
    value = normalized_phones(tokens)
    matches = [name for name, pattern in PHONE_PATTERNS.items() if pattern.fullmatch(value)]
    return matches[0] if len(matches) == 1 else ("AMBIGUOUS" if matches else None)


def safe_relative(name):
    require(type(name) is str and name and SAFE_REL.fullmatch(name) is not None,
            "SAFE_RELATIVE_PATH")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name
            and not any(part in ("", ".", "..", ".git") for part in path.parts),
            "SAFE_RELATIVE_PATH")
    return name


def read_member(root, relative, expected_sha, *, limit=4 * 1024 * 1024):
    name = safe_relative(relative)
    require(type(expected_sha) is str and HEX64.fullmatch(expected_sha) is not None,
            "MEMBER_SHA256")
    path = Path(root)
    for part in PurePosixPath(name).parts:
        path = path / part
        require(not path.is_symlink(), "SYMLINK_MEMBER")
    require(path.is_file() and path.stat().st_size <= limit, "REGULAR_MEMBER_AND_SIZE")
    raw = path.read_bytes()
    require(sha(raw) == expected_sha, "MEMBER_SHA256_MISMATCH")
    return raw


def verify_pcm(row, root):
    pcm = read_member(root, row["pcm_path"], row["pcm_sha256"], limit=1024 * 1024)
    wav = read_member(root, row["wav_path"], row["wav_sha256"], limit=4 * 1024 * 1024)
    with wave.open(io.BytesIO(wav), "rb") as stream:
        samples = stream.getnframes()
        require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype())
                == (1, 2, 16000, "NONE"), "ORIGINAL_MONO_PCM16_16K")
        require(samples == row["pcm_samples"] and len(pcm) == samples * 2, "ORIGINAL_SAMPLE_COUNT")
        decoded = stream.readframes(samples)
    require(decoded == pcm and sha(decoded) == row["pcm_sha256"], "WAV_PCM_BYTE_IDENTITY")
    require(row["sample_rate"] == 16000 and row["channels"] == 1, "PCM_DECLARED_GEOMETRY")
    if row["kind"] == "target":
        require(6400 <= samples <= 28672, "MODEL_INPUT_GEOMETRY")
    return pcm


def observer_identity(provider):
    family, model_sha, decode_sha = MODEL_PINS[provider]
    return {
        "recognizer_id": provider,
        "model_family": family,
        "model_sha256": model_sha,
        "decode_config_sha256": decode_sha,
    }


def verify_observation(observation, provider, row, root, *, phones=False):
    if not isinstance(observation, dict):
        return None
    identity = observer_identity(provider)
    require(all(observation.get(key) == value for key, value in identity.items()),
            "OBSERVER_IDENTITY:" + provider)
    require(observation.get("id") == row["id"]
            and observation.get("schema") == CAPTURE_SCHEMA
            and observation.get("status") == "success"
            and observation.get("source_pcm_sha256") == row["pcm_sha256"]
            and observation.get("source_wav_sha256") == row["wav_sha256"]
            and observation.get("source_samples") == row["pcm_samples"]
            and observation.get("target_conditioned") is False
            and observation.get("input_coverage_complete") is True
            and observation.get("decoder_output_complete") is True,
            "OBSERVER_SOURCE_AND_BLIND_BINDING:" + provider)
    raw = read_member(root, observation["capture_path"], observation["capture_sha256"], limit=1024 * 1024)
    capture = json.loads(raw, object_pairs_hook=unique_pairs)
    require(capture == {key: value for key, value in observation.items()
                        if key not in ("capture_path", "capture_sha256")},
            "CAPTURE_BYTES_MATCH_INDEX:" + provider)
    if phones:
        tokens = capture.get("phone_tokens")
        require(isinstance(tokens, list) and 1 <= len(tokens) <= 300
                and all(isinstance(token, str) and 0 < len(token) <= 40 for token in tokens),
                "PHONE_CAPTURE_CONTENT")
        return tokens
    text = capture.get("raw_text")
    require(isinstance(text, str) and 0 < len(text.strip()) <= 512
            and not any(ord(char) < 32 for char in text.strip()), "ASR_CAPTURE_CONTENT:" + provider)
    return normalized_text(text)


def _votes(texts):
    result = {"VOSEN": 0, "MERO": 0}
    normalized = [normalized_text(text) for text in texts]
    for text in normalized:
        if text in EXCLUDED_TRANSCRIPTS:
            return None, None, "SUPERSEDED_PRONUNCIATION_TRANSCRIPT"
        for candidate, aliases in ALIASES.items():
            if text in aliases:
                result[candidate] += 1
    return result, normalized, None


def decide_candidate(row, candidate, texts, phones):
    """Return (tier, label, reason, observed_phrase) for one candidate."""
    require(candidate in CANDIDATES, "KNOWN_CANDIDATE")
    if texts is None or len(texts) != 3 or any(not isinstance(text, str) or not text for text in texts):
        return "unknown", None, "MISSING_OR_INCOMPLETE_ASR", None
    if phones is None or not phones:
        return "unknown", None, "MISSING_OR_INCOMPLETE_PHONE", None
    votes, normalized, error = _votes(texts)
    if error:
        return "unknown", None, error, None
    if votes is None:
        return "unknown", None, "CONTRADICTORY_TRANSCRIPT", None
    other = "MERO" if candidate == "VOSEN" else "VOSEN"
    phone = phone_candidate(phones)
    if phone == "AMBIGUOUS":
        return "unknown", None, "AMBIGUOUS_PHONE_CLASS", None
    target_votes = votes[candidate]
    other_votes = votes[other]
    expected_class = row["candidate_labels"][candidate]
    if expected_class == 1:
        if target_votes < 2 or other_votes != 0 or phone != candidate:
            return "unknown", None, "NO_JOINT_CANONICAL_TARGET_SUPPORT", phone
        return ("silver" if target_votes == 3 else "weak"), 1, "JOINT_ASR_AND_PHONE_TARGET_SUPPORT", candidate
    if row["kind"] == "target":
        if other_votes < 2 or target_votes != 0 or phone != other:
            return "unknown", None, "NO_JOINT_CROSS_TRIGGER_NEGATIVE_SUPPORT", phone
        return ("silver" if other_votes == 3 else "weak"), 0, "OTHER_BRAND_CROSS_TRIGGER_NEGATIVE", other
    if target_votes or other_votes or phone is not None:
        return "unknown", None, "WAKE_PHRASE_OR_PHONE_PRESENT_IN_NEGATIVE", phone
    expected = normalized_text(row["requested_text"])
    exact_votes = sum(text == expected for text in normalized)
    if exact_votes < 2:
        return "unknown", None, "NEGATIVE_NOT_EXPLICITLY_RECOGNIZED", None
    if row["kind"] not in ("contrast-negative", "ordinary-negative"):
        return "unknown", None, "UNREGISTERED_NEGATIVE_KIND", None
    return ("silver" if exact_votes == 3 else "weak"), 0, "EXPLICIT_NEGATIVE_TRANSCRIPT_AND_NO_WAKE_PHONE", None


def validate_plan(plan, profile):
    require(type(plan) is dict and plan.get("execution_stage") == "G2_SOURCE_GENERATION_AND_BLIND_OBSERVATION_ONLY",
            "V7_G2_EXECUTION_PLAN")
    require(plan.get("source_count") == 96 and plan.get("candidate_ids") == ["hey-vosen-v7", "hey-mero-v7"]
            and plan.get("observer_providers") == ["qwen", "whisper", "parakeet", "phoneme"],
            "V7_EXACT_SOURCE_AND_CANDIDATE_ROSTER")
    require(plan.get("source_role_counts") == {"train": 6, "development": 2},
            "V7_SOURCE_ROLE_COUNTS")
    require(plan.get("positive_min_samples") == 6400 and plan.get("positive_max_samples") == 28672,
            "V7_POSITIVE_GEOMETRY")
    require(plan.get("source_plan_sha256") and HEX64.fullmatch(plan["source_plan_sha256"]),
            "V7_SOURCE_PLAN_HASH")
    require(plan.get("training_allowed") is False and plan.get("promotion_allowed") is False
            and plan.get("new_kws_calls") == 0 and plan.get("optimizer_updates") == 0,
            "G2_CANNOT_AUTHORIZE_TRAINING_OR_PROMOTION")
    require(profile.get("schema") == "english-kws-pronunciation-profile-v4"
            and {item.get("id"): item.get("ipa") for item in profile.get("phrases", [])}
            == {"hey_mero": "/heɪ ˈmɛroʊ/", "hey_vosen": "/heɪ ˈvoʊsən/"}
            and all(item.get("final_pronunciation_frozen") is True for item in profile["phrases"]),
            "EXACT_CANONICAL_BRAND_PRONUNCIATIONS")
    return plan


def _validate_corpus(corpus, evidence, plan):
    require(corpus.get("schema") == CORPUS_SCHEMA and evidence.get("schema") == EVIDENCE_SCHEMA,
            "V7_CORPUS_AND_EVIDENCE_SCHEMA")
    require(corpus.get("recipe_sha256") == plan.get("recipe_sha256")
            and corpus.get("preflight_sha256"), "V7_RECIPE_AND_PREFLIGHT_BINDING")
    rows = corpus.get("rows", [])
    observations = evidence.get("rows", [])
    require(len(rows) == 96 and len(observations) == 96, "COMPLETE_144_ROW_DENOMINATOR")
    statuses = [row.get("generation_status") for row in rows]
    allowed_statuses = {"success", "FAILED_NO_RETRY", "UNKNOWN_AFTER_START_NO_RETRY",
                        "POSITIVE_GEOMETRY_QUARANTINED", "MODEL_INPUT_GEOMETRY_QUARANTINED", "DUPLICATE_PCM_QUARANTINED",
                        "NOT_STARTED"}
    require(corpus.get("declared_count") == 96 and all(status in allowed_statuses for status in statuses)
            and corpus.get("valid_count") == statuses.count("success"),
            "144_ROW_TERMINAL_STATUS_AND_VALID_COUNT")
    require(len({row.get("id") for row in rows}) == 96
            and {row.get("id") for row in rows} == {row.get("id") for row in observations},
            "SOURCE_OBSERVER_ROSTER_IDENTITY")
    groups = corpus.get("source_groups", [])
    require(len(groups) == 8 and len({group.get("source_group_id") for group in groups}) == 8
            and len({group.get("source_speaker_id") for group in groups}) == 8,
            "EIGHT_UNIQUE_SOURCE_GROUPS_AND_SPEAKERS")
    require(Counter(group.get("role") for group in groups)
            == Counter({"train": 6, "development": 2}),
            "SOURCE_GROUP_6_2_2_2")
    group_by_id = {group["source_group_id"]: group for group in groups}
    cells_seen = set()
    rows_by_id = {}
    for row in rows:
        group = group_by_id.get(row.get("source_group_id"))
        cell = EXPECTED_CELLS.get(row.get("cell_id"))
        require(group is not None and cell is not None
                and all(row.get(key) == group[key] for key in SOURCE_KEYS),
                "ROW_SOURCE_GROUP_AND_ROLE_BINDING")
        expected_text, expected_class, expected_kind, expected_ipa, expected_scale, labels = cell
        require(row.get("requested_text") == expected_text
                and row.get("requested_class") == expected_class
                and row.get("kind") == expected_kind
                and row.get("raw_phonemes") == expected_ipa
                and row.get("length_scale") == expected_scale
                and row.get("candidate_labels") == labels,
                "ROW_CELL_IPA_SCALE_AND_CANDIDATE_RELATIVE_LABEL_BINDING")
        pair = (row["source_group_id"], row["cell_id"])
        require(pair not in cells_seen, "DUPLICATE_SOURCE_CELL")
        cells_seen.add(pair)
        rows_by_id[row["id"]] = row
    require(len(cells_seen) == 96, "EXACT_8_BY_12_SOURCE_CELL_ROSTER")
    evidence_by_id = {item["id"]: item for item in observations}
    return rows, evidence_by_id, group_by_id


def build_admission(corpus, evidence, root, *, profile, plan, open_roles=ROLES):
    """Build candidate-relative manifests while opening only explicitly authorized roles.

    G3 passes ("train", "development"). Calibration and locked-test rows remain
    sealed placeholders: their audio, observer captures and labels are not read.
    """
    validate_plan(plan, profile)
    open_roles = tuple(open_roles)
    require(open_roles and len(set(open_roles)) == len(open_roles)
            and set(open_roles) <= set(ROLES)
            and {"train", "development"} <= set(open_roles),
            "AUTHORIZED_ADMISSION_ROLES")
    rows, evidence_by_id, groups = _validate_corpus(corpus, evidence, plan)
    per_candidate = {}
    for candidate in CANDIDATES:
        decisions = []
        for row in rows:
            packet = evidence_by_id[row["id"]]
            role = row["role"]
            common = {
                "id": row["id"],
                **{key: row[key] for key in SOURCE_KEYS},
                **{key: row[key] for key in ("cell_id", "requested_text", "requested_class", "kind",
                                              "length_scale", "raw_phonemes", "pronunciation_hypothesis")},
                "request_sha256": row["request_sha256"],
                "candidate_id": "hey-vosen-v7" if candidate == "VOSEN" else "hey-mero-v7",
                "candidate": candidate,
                "human_gold": False,
                "final_native_pronunciation_verified": False,
                "independent_real_evaluation_allowed": False,
            }
            if role not in open_roles:
                # Do not expose target text, requested labels, PCM hashes or observer evidence
                # for calibration/locked-test rows before their dedicated evaluation stage.
                sealed_common = {
                    "id": row["id"],
                    **{key: row[key] for key in SOURCE_KEYS},
                    "candidate_id": "hey-vosen-v7" if candidate == "VOSEN" else "hey-mero-v7",
                    "candidate": candidate,
                    "human_gold": False,
                    "final_native_pronunciation_verified": False,
                    "independent_real_evaluation_allowed": False,
                }
                decisions.append({
                    **sealed_common,
                    "source_receipt_sha256": None,
                    "actual_phonemes": None,
                    "actual_phoneme_ids": None,
                    "generation_status": "NOT_OPENED",
                    "source_generation_status": row["generation_status"],
                    "pcm_sha256": None,
                    "wav_sha256": None,
                    "pcm_samples": None,
                    "research_class": None,
                    "tier": "unknown",
                    "disposition": "SEALED",
                    "reason": "SEALED_ROLE_NOT_OPENED",
                    "observed_phrase": None,
                    "label_kind": "none",
                    "evidence_refs": [],
                    "balanced_source_group": False,
                    "training_eligible": False,
                    "optimizer_allowed": False,
                    "training_class": None,
                    "training_weight": 0.0,
                    "calibration_sealed": role == "calibration",
                    "locked_test_sealed": role == "locked-test",
                })
                continue
            by_id = {}
            if row["generation_status"] == "success":
                verify_pcm(row, root)
                observations = packet.get("observations")
                require(isinstance(observations, list) and len(observations) <= 3
                        and all(isinstance(item, dict) for item in observations),
                        "BLIND_ASR_OBSERVATION_LIST")
                provider_ids = [item.get("recognizer_id") for item in observations]
                require(len(set(provider_ids)) == len(provider_ids)
                        and set(provider_ids) <= {"qwen", "whisper", "parakeet"},
                        "NO_DUPLICATE_OR_UNKNOWN_ASR_PROVIDER")
                by_id = {item["recognizer_id"]: item for item in observations}
                texts = ([verify_observation(by_id[provider], provider, row, root)
                          for provider in ("qwen", "whisper", "parakeet")]
                         if set(by_id) == {"qwen", "whisper", "parakeet"} else None)
                phone_observation = packet.get("phone_observation")
                phone = (verify_observation(phone_observation, "phoneme", row, root, phones=True)
                         if isinstance(phone_observation, dict) else None)
            else:
                require(not packet.get("observations") and packet.get("phone_observation") is None,
                        "FAILED_OR_QUARANTINED_SOURCE_HAS_NO_OBSERVER_EVIDENCE")
                texts = None
                phone = None
            tier, label, reason, observed = decide_candidate(row, candidate, texts, phone)
            if row["generation_status"] != "success":
                tier, label, reason, observed = "unknown", None, "SOURCE_NOT_VALID_NO_RETRY", None
            refs = []
            for provider in ("qwen", "whisper", "parakeet"):
                item = by_id.get(provider)
                if isinstance(item, dict) and item.get("capture_path"):
                    refs.append({"recognizer_id": provider, "capture_path": item["capture_path"],
                                 "capture_sha256": item["capture_sha256"]})
            phone_item = packet.get("phone_observation")
            if isinstance(phone_item, dict) and phone_item.get("capture_path"):
                refs.append({"recognizer_id": "phoneme", "capture_path": phone_item["capture_path"],
                             "capture_sha256": phone_item["capture_sha256"]})
            decisions.append({
                **common,
                "source_receipt_sha256": row.get("source_receipt_sha256"),
                "actual_phonemes": row.get("actual_phonemes"),
                "actual_phoneme_ids": row.get("actual_phoneme_ids"),
                "generation_status": row["generation_status"],
                "source_generation_status": row["generation_status"],
                "pcm_sha256": row["pcm_sha256"],
                "wav_sha256": row["wav_sha256"],
                "pcm_samples": row["pcm_samples"],
                "research_class": label,
                "tier": tier,
                "disposition": "UNKNOWN" if tier == "unknown" else tier.upper() + "_RESEARCH",
                "reason": reason,
                "observed_phrase": observed,
                "label_kind": "synthetic_" + tier + "_machine_research",
                "evidence_refs": refs,
                "calibration_sealed": False,
                "locked_test_sealed": False,
            })
        supported = defaultdict(lambda: Counter())
        for item in decisions:
            if item["tier"] in ("silver", "weak"):
                supported[item["source_group_id"]][str(item["research_class"])] += 1
        balanced = set()
        for group_id, counts in supported.items():
            if counts["1"] >= 2 and counts["0"] >= 4:
                balanced.add(group_id)
        balanced_counts = Counter(groups[group_id]["role"] for group_id in balanced)
        snapshot = all(balanced_counts[role] >= plan["source_role_counts"][role]
                       for role in open_roles)
        for item in decisions:
            eligible = item["source_group_id"] in balanced and item["tier"] in ("silver", "weak")
            item.update(
                balanced_source_group=item["source_group_id"] in balanced,
                training_eligible=eligible and item["role"] == "train",
                optimizer_allowed=False,
                training_class=item["research_class"] if eligible and item["role"] == "train" else None,
                training_weight={"silver": 1.0, "weak": 0.5}.get(item["tier"], 0.0)
                    if eligible and item["role"] == "train" else 0.0,
                calibration_sealed=item["role"] == "calibration",
                locked_test_sealed=item["role"] == "locked-test",
            )
        phrase = "Hey Vosen" if candidate == "VOSEN" else "Hey Mero"
        ipa = "/heɪ ˈvoʊ.sən/" if candidate == "VOSEN" else "/heɪ ˈmɛroʊ/"
        tokens = ["HEY", "VOH", "SEN"] if candidate == "VOSEN" else ["HEY", "MEH", "ROH"]
        manifest = {
            "schema": "english-kws-dual-track-v7-candidate-manifest-v1",
            "candidate_id": "hey-vosen-v7" if candidate == "VOSEN" else "hey-mero-v7",
            "candidate": candidate,
            "phrase": phrase,
            "locale": "en-US",
            "ipa": ipa,
            "ctc_tokens": tokens,
            "ctc_token_ids": [1, 2, 3],
            "source_plan_sha256": plan["source_plan_sha256"],
            "source_group_roles": sorted(groups.values(), key=lambda group: group["source_group_id"]),
            "split_counts": dict(plan["source_role_counts"]),
            "minimum_balanced_groups_per_role": {role: plan["source_role_counts"][role]
                                                   for role in open_roles},
            "admission_open_roles": list(open_roles),
            "balanced_groups_by_role": {role: balanced_counts[role] for role in ROLES},
            "minimum_positive_rows_per_group": 2,
            "minimum_negative_rows_per_group": 4,
            "training_snapshot_allowed": snapshot,
            "optimizer_allowed": False,
            "calibration_sealed": "calibration" not in open_roles,
            "locked_test_sealed_from_training_and_mining": "locked-test" not in open_roles,
            "rows": decisions,
            "tier_counts": dict(sorted(Counter(item["tier"] for item in decisions).items())),
            "opened_role_tier_counts": {
                role: dict(sorted(Counter(item["tier"] for item in decisions
                                           if item["role"] == role).items()))
                for role in open_roles
            },
            "human_gold": False,
            "independent_real_quality_claims_allowed": False,
            "physical_ssc305_claims_allowed": False,
            "unknown_is_background": False,
        }
        per_candidate[candidate] = {
            "manifest": manifest,
            "manifest_sha256": sha(canonical(manifest)),
            "training_snapshot_allowed": snapshot,
            "balanced_groups_by_role": {role: balanced_counts[role] for role in ROLES},
            "tier_counts": dict(sorted(Counter(item["tier"] for item in decisions).items())),
            "opened_role_tier_counts": manifest["opened_role_tier_counts"],
            "eligible_train_rows": sum(item["training_eligible"] for item in decisions),
        }
    summary = {
        "schema": SCHEMA,
        "rule_id": RULES["rule_id"],
        "rules_sha256": RULES_SHA256,
        "source_plan_sha256": plan["source_plan_sha256"],
        "source_run": corpus.get("run"),
        "corpus_sha256": sha(canonical(corpus)),
        "evidence_sha256": sha(canonical(evidence)),
        "opened_roles": list(open_roles),
        "calibration_pcm_opened": "calibration" in open_roles,
        "locked_test_pcm_opened": "locked-test" in open_roles,
        "candidate_results": {
            candidate: {key: value for key, value in result.items() if key != "manifest"}
            for candidate, result in per_candidate.items()
        },
        "data_admitted": all(result["training_snapshot_allowed"] for result in per_candidate.values()),
        "training_authorized": False,
        "optimizer_updates": 0,
        "kws_model_forwards": 0,
        "human_gold": False,
        "independent_real_quality_claims_allowed": False,
        "physical_ssc305_claims_allowed": False,
        "locked_test_opened_for_training_or_mining": False,
        "unknown_is_background": False,
    }
    return {"summary": summary, "candidate_manifests": {
        candidate: value["manifest"] for candidate, value in per_candidate.items()
    }}


__all__ = [
    "SCHEMA", "RULES", "RULES_SHA256", "MODEL_PINS", "CANDIDATES", "ROLES",
    "normalized_text", "normalized_phones", "phone_candidate", "decide_candidate",
    "validate_plan", "build_admission", "canonical", "sha",
]
