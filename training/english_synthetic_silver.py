"""English synthetic research admission from retained PCM and machine captures.

This module runs no recognizer/model and grants no human-gold or device claims.
The reviewed experiment pins the plan, this module, profile and source job; its
artifact hash authenticates the bundle. JSON alone is not source authentication.
Three compatible ASR observations plus independent phones form silver; two form
a separately named weak tier. Unknown is never converted to background.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import unicodedata
import wave

RULE_ID = "english-synthetic-machine-admission-v1"
ROLES = ("train", "development", "calibration", "locked-test")
IDENTITY = ("recognizer_id", "model_family", "model_sha256", "decode_config_sha256")
SOURCE = ("source_group_id", "source_speaker_id", "source_speaker_index", "voice_family", "role")
CELL = ("cell_id", "requested_text", "requested_class", "pronunciation_hypothesis")
SEMANTICS = IDENTITY + ("source_pcm_sha256", "source_wav_sha256", "source_samples", "status",
                        "target_conditioned", "input_coverage_complete", "decoder_output_complete")
ORDINARY = ("What time is it?", "Please turn on the kitchen light.",
            "The package is by the front door.", "Could you open the window?")
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
RULES = {
    "rule_id": RULE_ID,
    "normalization": "unicode-nfkc-casefold-punctuation-to-space-v1",
    "aliases": {
        "1": ["hey vosen", "hey vozen", "hey vozan", "hey vohsen", "hey vohzen"],
        "2": ["hey mero", "hey meero", "hey miro", "hey merrow", "hey marrow"],
    },
    "phone_patterns": {
        "mero-us-eh-v1": [2, r"h?eɪmɛr(?:oʊ|ou|o)"],
        "mero-us-i-v1": [2, r"h?eɪmir(?:oʊ|ou|o)"],
        "vosen-us-s-v1": [1, r"h?eɪv(?:oʊ|ou|o)s[əɐʌ]n"],
        "vosen-us-z-v1": [1, r"h?eɪv(?:oʊ|ou|o)z[əɐʌ]n"],
    },
    "phone_normalization": "join-tokens-strip-stress-length-and-ties-map-rhotics-v1",
    "phone_scope": "whole-phrase-family-not-stress-vowel-length-or-final-native-truth",
    "model_pins": MODEL_PINS,
    "silver_votes": 3,
    "weak_votes": 2,
    "third_recognizer_other_wake_class_veto": True,
    "contradictory_core_phrases": {
        "1": ["hey frozen", "hey chosen", "hey rosen", "hey wosen"],
        "2": ["hey hero", "hey nero", "hey zero", "hey memo"],
    },
    "missing_hey_h_allowed_only_when_all_three_texts_support_target": True,
    "ordinary_sentences": list(ORDINARY),
    "pcm": {"sample_rate": 16000, "channels": 1, "sample_width": 2,
            "min_samples": 6400, "max_samples": 160000,
            "min_rms": 0.0001, "max_rail_fraction": 0.01},
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


RULES_SHA256 = sha(canonical(RULES))


def normalized_text(text):
    require(isinstance(text, str), "TRANSCRIPT_MUST_BE_TEXT")
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c
                             for c in text).split())


def normalized_phones(tokens):
    require(isinstance(tokens, list) and all(isinstance(t, str) and t for t in tokens),
            "PHONE_TOKENS_REQUIRED")
    value = unicodedata.normalize("NFKC", "".join(tokens))
    value = re.sub(r"[\sˈˌːˑ͜͡]", "", value)
    return value.replace("ɹ", "r").replace("ɾ", "r")


def read_member(root, relative, expected_sha, *, limit=4 * 1024 * 1024):
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "INVALID_MEMBER_PATH")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and all(p not in ("", ".", "..") for p in path.parts)
            and path.as_posix() == relative, "INVALID_MEMBER_PATH")
    require(isinstance(expected_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha),
            "INVALID_MEMBER_SHA256")
    current = Path(root)
    require(current.is_dir() and not current.is_symlink(), "INVALID_EVIDENCE_ROOT")
    for part in path.parts:
        current = current / part
        require(not current.is_symlink(), "SYMLINK_MEMBER_FORBIDDEN")
    require(current.is_file() and 0 < current.stat().st_size <= limit, "MEMBER_MISSING_OR_SIZE")
    raw = current.read_bytes()
    require(sha(raw) == expected_sha, "MEMBER_HASH_MISMATCH")
    return raw


def verify_pcm(row, root):
    require(row.get("sample_rate") == 16000 and row.get("channels") == 1
            and type(row.get("pcm_samples")) is int, "PCM_METADATA")
    raw = read_member(root, row.get("wav_path"), row.get("wav_sha256"))
    try:
        with wave.open(io.BytesIO(raw)) as stream:
            require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                     stream.getcomptype()) == (1, 2, 16000, "NONE"), "PCM_FORMAT")
            frames = stream.getnframes()
            pcm = stream.readframes(frames)
            require(len(pcm) == 2 * frames and frames == row["pcm_samples"], "PCM_FRAME_COUNT")
    except (wave.Error, EOFError) as exc:
        raise ValueError("INVALID_WAV") from exc
    require(sha(pcm) == row.get("pcm_sha256"), "PCM_HASH_MISMATCH")
    # pcm_path is a raw little-endian PCM companion when supplied. WAV is the
    # authoritative trainer input; do not interpret a pathname as decoded PCM.
    if row.get("pcm_path") is not None:
        require(read_member(root, row["pcm_path"], row["pcm_sha256"]) == pcm, "PCM_COMPANION")
    limits = RULES["pcm"]
    if not limits["min_samples"] <= frames <= limits["max_samples"]:
        return False, "DURATION_OUTSIDE_RESEARCH_RULE"
    samples = struct.unpack("<" + "h" * frames, pcm)
    mean_square = sum(x * x for x in samples) / frames / (32768 * 32768)
    rail = sum(abs(x) >= 32767 for x in samples) / frames
    return (mean_square >= limits["min_rms"] ** 2 and rail <= limits["max_rail_fraction"],
            "PCM_ENERGY_OR_RAIL_RULE")


def verify_observation(observation, identity, row, root, *, phones=False):
    if not isinstance(observation, dict) or observation.get("status") in (None, "missing"):
        return None
    for key in IDENTITY:
        require(observation.get(key) == identity.get(key), "RECOGNIZER_IDENTITY_DRIFT")
    for field, source in (("source_pcm_sha256", "pcm_sha256"),
                          ("source_wav_sha256", "wav_sha256"),
                          ("source_samples", "pcm_samples")):
        require(type(observation.get(field)) is type(row.get(source)) and
                observation.get(field) == row.get(source), "OBSERVATION_SOURCE_DRIFT")
    raw = read_member(root, observation.get("capture_path"), observation.get("capture_sha256"))
    capture = json.loads(raw)
    value_field = "phone_tokens" if phones else "raw_text"
    require(isinstance(capture, dict) and all(key in capture and key in observation
            and capture[key] == observation[key] for key in SEMANTICS + (value_field,)),
            "CAPTURE_SEMANTIC_MISMATCH")
    if (observation.get("status") != "success" or observation.get("target_conditioned") is not False
            or observation.get("input_coverage_complete") is not True
            or observation.get("decoder_output_complete") is not True):
        return None
    return normalized_phones(observation[value_field]) if phones else normalized_text(observation[value_field])


def decide_label(row, texts, phones):
    """Return evidence tier, class and observed hypothesis without target KWS use."""
    if any(t is None or not t for t in texts) or phones is None or not phones:
        return "unknown", None, None, "MISSING_OR_INCOMPLETE_MACHINE_OBSERVATION"
    expected = row["requested_class"]
    if expected in (1, 2):
        votes = sum(text in RULES["aliases"][str(expected)] for text in texts)
        other = 3 - expected
        conflicts = RULES["aliases"][str(other)] + RULES["contradictory_core_phrases"][str(expected)]
        if any(any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text)
                   for alias in conflicts) for text in texts):
            return "unknown", None, None, "CONTRADICTORY_WAKE_CLASS"
        matches = [name for name, (kind, pattern) in RULES["phone_patterns"].items()
                   if kind == expected and re.fullmatch(pattern, phones)]
        if not matches or votes < 2 or (not phones.startswith("h") and votes != 3):
            return "unknown", None, None, "NO_JOINT_WHOLE_PHRASE_SUPPORT"
        return ("silver" if votes == 3 else "weak"), expected, matches[0], "JOINT_MACHINE_RESEARCH_SUPPORT"
    require(row["requested_text"] in ORDINARY and expected == 0, "ORDINARY_REQUEST_NOT_PREDECLARED")
    target_seen = any(any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text)
                         for aliases in RULES["aliases"].values() for alias in aliases) for text in texts)
    target_seen |= any(re.search(pattern, phones) is not None
                       for _, pattern in RULES["phone_patterns"].values())
    target_seen |= any(re.search(r"\b(?:hey|hay)\b", text) is not None for text in texts)
    votes = sum(text == normalized_text(row["requested_text"]) for text in texts)
    if target_seen or votes < 2:
        return "unknown", None, None, "ORDINARY_SENTENCE_NOT_SUPPORTED_OR_WAKE_OBSERVED"
    return ("silver" if votes == 3 else "weak"), 0, None, "ORDINARY_SENTENCE_MACHINE_SUPPORT"


def validate_plan(plan, profile):
    require(profile.get("schema") == "english-kws-pronunciation-profile-v3"
            and profile.get("research_and_software_delivery", {}).get("allowed") is True
            and profile["research_and_software_delivery"].get("human_recording_precondition") is False
            and profile["research_and_software_delivery"].get("physical_ssc305_precondition") is False,
            "SYNTHETIC_RESEARCH_PROFILE_REQUIRED")
    policy = plan.get("silver_admission", {})
    require(policy.get("rule_id") == RULE_ID and policy.get("rules_sha256") == RULES_SHA256,
            "MACHINE_RULE_NOT_PINNED")
    recognizers = policy.get("recognizers", [])
    phone = policy.get("phone_recognizer", {})
    require(len(recognizers) == 3 and {r.get("recognizer_id") for r in recognizers}
            == {"qwen", "whisper", "parakeet"}, "THREE_DISTINCT_RECOGNIZERS_REQUIRED")
    all_models = recognizers + [phone]
    for identity in all_models:
        require(all(isinstance(identity.get(k), str) and identity[k] for k in IDENTITY), "MODEL_IDENTITY")
        require(all(re.fullmatch(r"[0-9a-f]{64}", identity[k]) for k in
                    ("model_sha256", "decode_config_sha256")), "MODEL_IDENTITY_DIGEST")
        require(tuple(identity[k] for k in IDENTITY[1:]) == MODEL_PINS.get(identity["recognizer_id"]),
                "FROZEN_NATIVE_OBSERVER_PIN")
    require(phone["recognizer_id"] == "phoneme", "INDEPENDENT_PHONE_OBSERVER_ID")
    require(len({r["model_family"] for r in all_models}) == 4
            and len({r["model_sha256"] for r in all_models}) == 4,
            "INDEPENDENT_MODEL_FAMILIES_REQUIRED")
    groups, cells = policy.get("source_groups", []), policy.get("cells", [])
    require(len(groups) == 24 and len(cells) == 8, "FROZEN_24_BY_8_DESIGN")
    require(Counter(g.get("role") for g in groups) == Counter({"train": 12, "development": 4,
            "calibration": 4, "locked-test": 4}), "FROZEN_SOURCE_ROLE_COUNTS")
    require({g.get("source_speaker_index") for g in groups} == set(range(10, 34)), "NEW_SOURCE_INDICES")
    for field in SOURCE[:-1]:
        require(len({g.get(field) for g in groups}) == len(groups)
                and all(g.get(field) is not None for g in groups), "SOURCE_ANCESTRY_ALIAS_OR_MISSING")
    excluded = policy.get("excluded_source_speaker_ids")
    require(isinstance(excluded, list) and excluded and len(excluded) == len(set(excluded)), "EXPOSURE_EXCLUSIONS_REQUIRED")
    require(not set(excluded).intersection(g["source_speaker_id"] for g in groups), "EXPOSED_SPEAKER_REUSED")
    require(len({c.get("cell_id") for c in cells}) == 8 and
            Counter(c.get("requested_class") for c in cells) == Counter({0: 4, 1: 2, 2: 2}), "BALANCED_EIGHT_CELLS")
    require({c.get("requested_text") for c in cells if c["requested_class"] == 0} == set(ORDINARY), "ORDINARY_CELLS")
    for cell in cells:
        require(type(cell.get("requested_class")) is int, "CLASS_MUST_BE_INTEGER")
        hypothesis = cell.get("pronunciation_hypothesis")
        require(hypothesis == "ordinary-negative-v1" if cell["requested_class"] == 0 else
                hypothesis in RULES["phone_patterns"] and
                RULES["phone_patterns"][hypothesis][0] == cell["requested_class"], "HYPOTHESIS_CLASS")
    tiers = policy.get("supervised_tiers")
    weights = policy.get("training_tier_weights")
    require(isinstance(tiers, list) and "silver" in tiers and set(tiers) <= {"silver", "weak"}
            and len(tiers) == len(set(tiers)) and isinstance(weights, dict) and set(weights) == set(tiers), "EXPLICIT_TRAINING_TIERS")
    require(all(type(w) in (int, float) and 0 < w <= 1 for w in weights.values())
            and weights["silver"] == 1 and ("weak" not in weights or weights["weak"] < 1), "EXPLICIT_TIER_WEIGHTS")
    minimum = policy.get("minimum_balanced_groups", {})
    require(set(minimum) == set(ROLES) and all(type(minimum[r]) is int and
            1 <= minimum[r] <= (12 if r == "train" else 4) for r in ROLES), "PREDECLARED_SOURCE_MINIMUM")
    return policy


def build_admission(corpus, evidence, root, *, profile, plan):
    policy = validate_plan(plan, profile)
    require(corpus.get("schema") == "english-synthetic192-corpus-v1"
            and evidence.get("schema") == "english-synthetic-machine-evidence-v1", "CORPUS_OR_EVIDENCE_SCHEMA")
    rows, observations = corpus.get("rows", []), evidence.get("rows", [])
    require(len(rows) == 192 and len(observations) == 192, "COMPLETE_INTENDED_ROSTER_REQUIRED")
    require(len({r.get("id") for r in rows}) == 192 and {r["id"] for r in rows}
            == {r.get("id") for r in observations}, "SOURCE_OBSERVATION_ROSTER")
    observed = {r["id"]: r for r in observations}
    groups = {g["source_group_id"]: g for g in policy["source_groups"]}
    cells = {c["cell_id"]: c for c in policy["cells"]}
    seen, pcm_roles, result = set(), {}, []
    for row in rows:
        group, cell = groups.get(row.get("source_group_id")), cells.get(row.get("cell_id"))
        require(group is not None and cell is not None and all(row.get(k) == group[k] for k in SOURCE)
                and all(row.get(k) == cell[k] for k in CELL), "ROW_PLAN_OR_SPLIT_DRIFT")
        pair = (row["source_group_id"], row["cell_id"])
        require(pair not in seen, "DUPLICATE_SOURCE_CELL")
        seen.add(pair)
        tier, label, matched, reason = "unknown", None, None, "GENERATION_UNAVAILABLE"
        refs = []
        if row.get("generation_status") == "success":
            quality, quality_reason = verify_pcm(row, root)
            require(row["pcm_sha256"] not in pcm_roles, "DUPLICATE_PCM_IDENTITY")
            pcm_roles[row["pcm_sha256"]] = row["role"]
            packet = observed[row["id"]]
            obs = packet.get("observations", [])
            require(isinstance(obs, list) and len({o.get("recognizer_id") for o in obs}) == len(obs)
                    and {o.get("recognizer_id") for o in obs} <= {"qwen", "whisper", "parakeet"}, "OBSERVATION_ROSTER")
            by_id = {o["recognizer_id"]: o for o in obs}
            texts = [verify_observation(by_id.get(identity["recognizer_id"]), identity, row, root)
                     for identity in policy["recognizers"]]
            phone_obs = packet.get("phone_observation")
            phones = verify_observation(phone_obs, policy["phone_recognizer"], row, root, phones=True)
            refs = [{k: o[k] for k in ("recognizer_id", "capture_path", "capture_sha256")}
                    for o in obs + ([phone_obs] if isinstance(phone_obs, dict) else []) if o.get("capture_path")]
            if quality:
                tier, label, matched, reason = decide_label(row, texts, phones)
            else:
                reason = quality_reason
        result.append({"id": row["id"], **{k: row[k] for k in SOURCE + CELL},
                       "pcm_sha256": row.get("pcm_sha256"), "wav_sha256": row.get("wav_sha256"),
                       "disposition": "UNKNOWN" if tier == "unknown" else tier.upper() + "_RESEARCH",
                       "tier": tier, "research_class": label, "matched_pronunciation_hypothesis": matched,
                       "label_kind": "synthetic_" + tier + "_machine_research", "reason": reason,
                       "evidence_refs": refs, "human_gold": False, "final_native_pronunciation_verified": False,
                       "independent_real_evaluation_allowed": False})
    supported = defaultdict(set)
    for row in result:
        if row["tier"] in policy["supervised_tiers"]:
            supported[row["source_group_id"]].add(row["research_class"])
    balanced = {g for g, labels in supported.items() if labels == {0, 1, 2}}
    counts = Counter(groups[g]["role"] for g in balanced)
    for row in result:
        allowed = row["source_group_id"] in balanced and row["tier"] in policy["supervised_tiers"]
        row.update(supervision_allowed=allowed, training_allowed=allowed and row["role"] == "train",
                   training_class=row["research_class"] if allowed and row["role"] == "train" else None,
                   training_weight=policy["training_tier_weights"].get(row["tier"], 0)
                   if allowed and row["role"] == "train" else 0)
    enough = all(counts[r] >= policy["minimum_balanced_groups"][r] for r in ROLES)
    return {"schema": "english-synthetic-research-admission-v1", "rule_id": RULE_ID,
            "rules_sha256": RULES_SHA256, "plan_sha256": sha(canonical(plan)),
            "profile_sha256": sha(canonical(profile)), "corpus_sha256": sha(canonical(corpus)),
            "evidence_sha256": sha(canonical(evidence)), "rows": result,
            "tier_counts": dict(sorted(Counter(r["tier"] for r in result).items())),
            "balanced_groups_by_role": {r: counts[r] for r in ROLES},
            "minimum_balanced_groups": policy["minimum_balanced_groups"],
            "training_snapshot_allowed": enough,
            "human_gold": False, "independent_real_quality_claims_allowed": False,
            "physical_ssc305_claims_allowed": False,
            "source_authentication": "requires_reviewed_worker_plan_and_exact_artifact_pin",
            "unknown_is_background": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "evidence", "root", "profile", "plan", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    result = build_admission(read(args.corpus), read(args.evidence), args.root,
                             profile=read(args.profile), plan=read(args.plan))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["training_snapshot_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
