"""English wake-word research-only whole-clip label bridge.

No CTC/Chinese-phone remapping, ASR voting, IPA auto-truth, audio generation,
model loading, or training permission. It reads one exact three-source review
receipt from an existing *exposed synthetic* English study. Every mapped
intent is a weak diagnostic target, never an authenticated acoustic label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

POLICY = "english-wholeclip-weak-intent-bridge-v1"
SOURCE_SCHEMA = "english63-independent-evidence-joined-review-priority-v1"
SOURCE_BYTES_SHA256 = "89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12"
SOURCE_ZIP_SHA256 = {
    "asr": "3786baa9fd3575437a05e97956fafdf3298eac23de2538b485b8342ef4d25133",
    "dtw": "64efe60c4b8c45cf130b2b55b4be6674f665dbcf7d667548c463dda15d951e18",
    "ipa": "a70ee89c76bf25f8a0a6b82a5a72c46128782e7f2020fb7655c06845de131be0",
}
SOURCE_PAYLOAD_SHA256 = {
    "asr": "60054a3532491631a47b094ea7ff2bc86ef16530ff510f968bc9edcdd253a9ad",
    "dtw": "6dfe169bb37a8264e281270c3eb6aaa3d49135a79a2eb6761cffb69b1b5681f4",
    "ipa": "093e18093a52bffa68aa60082cc490fdc3f9129eb1d60a209c2fef1d29e9e151",
}
# Source order is frozen. The English whole-recording R0 head used three
# classes, not Mandarin's four-phone CTC representation.
CLASSES = {"background": 0, "Hey Vosen": 1, "Hey Mero": 2}
REQUESTS = {
    "Hey Vosen": ("wake", "Hey Vosen"),
    "Hey, Vosen!": ("wake", "Hey Vosen"),
    "Hey Mero": ("wake", "Hey Mero"),
    "Hey, Mero!": ("wake", "Hey Mero"),
    "Hey Wosen": ("near_miss", None),
    "Hey Miro": ("near_miss", None),
    "Hey Rosen": ("near_miss", None),
    "What time is it?": ("ordinary_negative", "background"),
    "Hey, can you hear me?": ("ordinary_negative", "background"),
}
TRAIN_VOICES = ("dylan", "eric", "ono_anna", "sohee", "vivian")
DEVELOPMENT_VOICES = ("serena", "uncle_fu")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CELL = re.compile(r"en-[0-9a-f]{24}\Z")


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def role(voice: str) -> str:
    if voice in TRAIN_VOICES:
        return "train-candidate"
    if voice in DEVELOPMENT_VOICES:
        return "development-only"
    raise ValueError("ENGLISH_UNKNOWN_VOICE_FAMILY")


def require_machine_only(record: dict, tag: str) -> None:
    need(type(record) is dict and
         all(record.get(field) is False for field in
             ("human_gold", "training_admitted", "trusted_negative",
              "phonetic_label_authenticated")),
         "ENGLISH_UNAUTHORIZED_MACHINE_LABEL_PROMOTION_" + tag)


def read_verified_report(path: Path) -> dict:
    need(path.is_absolute() and path.is_file() and not path.is_symlink() and
         1000 <= path.stat().st_size <= 2 * 1024 * 1024,
         "ENGLISH_REVIEW_SOURCE_FILE_INVALID")
    raw = path.read_bytes()
    need(sha(raw) == SOURCE_BYTES_SHA256, "ENGLISH_REVIEW_SOURCE_HASH_CHANGED")
    return json.loads(raw)


def verify_three_source_report(report: dict) -> list[dict]:
    need(type(report) is dict and report.get("schema") == SOURCE_SCHEMA and
         report.get("status") == "MACHINE_EVIDENCE_JOINED_REVIEW_NOT_GOLD" and
         report.get("source_archive_sha256") == SOURCE_ZIP_SHA256 and
         report.get("source_payload_sha256") == SOURCE_PAYLOAD_SHA256 and
         report.get("source_tts_run_id") == 43242278962 and
         report.get("original_three_asr_run_id") == 43246104418 and
         report.get("source_ipa_run_id") == 43340235981 and
         report.get("audio_rows_reused") == 63 and
         report.get("asr_observations_reused") == 189 and
         report.get("new_model_calls") == 0 and
         report.get("voice_count_synthetic") == 7 and
         report.get("review_wake_pairs") == 14 and
         report.get("nearphone_challenges") == 21 and
         report.get("training_admitted") is False and
         report.get("human_gold") is False and
         report.get("independent_human_validation") == "NOT_RUN",
         "ENGLISH_REVIEW_SOURCE_POLICY_CHANGED")
    rows = report.get("rows")
    need(type(rows) is list and len(rows) == 63,
         "ENGLISH_SOURCE_ROW_COUNT")
    observed = set()
    wavs, pcms, ids = set(), set(), set()
    for row in rows:
        require_machine_only(row, "SOURCE_ROW")
        voice = row.get("voice_id")
        requested = row.get("requested_text")
        need(type(voice) is str and (voice in TRAIN_VOICES or
             voice in DEVELOPMENT_VOICES) and
             requested in REQUESTS and
             row.get("source_family_id") == "qwen3-preset-" + voice and
             row.get("role") == role(voice) and
             row.get("requested_intent") == REQUESTS[requested][0] and
             row.get("independent_evaluation") is False,
             "ENGLISH_SOURCE_ROLE_OR_INTENT_DRIFT")
        ident = row.get("cell_id")
        wav = row.get("wav_sha256")
        pcm = row.get("pcm_sha256")
        need(type(ident) is str and CELL.fullmatch(ident) is not None and
             type(wav) is str and HEX64.fullmatch(wav) is not None and
             type(pcm) is str and HEX64.fullmatch(pcm) is not None and
             ident not in ids and wav not in wavs and pcm not in pcms and
             (voice, requested) not in observed,
             "ENGLISH_DUPLICATE_OR_UNBOUND_AUDIO")
        ids.add(ident)
        wavs.add(wav)
        pcms.add(pcm)
        observed.add((voice, requested))
        asr = row.get("asr_observations")
        need(type(asr) is dict and set(asr) == {"qwen", "whisper", "parakeet"} and
             all(type(x) is dict and type(x.get("text")) is str and
                 type(x.get("normalized")) is str and
                 type(x.get("native_sha256")) is str and
                 HEX64.fullmatch(x["native_sha256"]) is not None
                 for x in asr.values()),
             "ENGLISH_ASR_NOT_INDEPENDENT_OR_INCOMPLETE")
        ipa = row.get("ipa_tokens")
        need(type(ipa) is list and 2 <= len(ipa) <= 300 and
             all(type(x) is str and 0 < len(x) <= 40 for x in ipa),
             "ENGLISH_NO_NATIVE_PHONES")
    need(observed == {(v, p) for v in TRAIN_VOICES + DEVELOPMENT_VOICES
                      for p in REQUESTS}, "ENGLISH_SOURCE_ROSTER_INCOMPLETE")
    pairs = report.get("wake_pairs")
    need(type(pairs) is list and len(pairs) == 14,
         "ENGLISH_WAKE_REVIEW_PAIR_COUNT")
    seen_pairs = set()
    for pair in pairs:
        require_machine_only(pair, "REVIEW_PAIR")
        voice = pair.get("voice_id")
        name = pair.get("wake_candidate")
        need(voice in TRAIN_VOICES + DEVELOPMENT_VOICES and
             name in ("vosen", "mero") and
             pair.get("role") == role(voice) and
             pair.get("source_family_id") == "qwen3-preset-" + voice and
             pair.get("machine_only_priority") is True and
             (voice, name) not in seen_pairs,
             "ENGLISH_PAIR_ROLE_OR_LABEL_DRIFT")
        seen_pairs.add((voice, name))
        expected = ["Hey Vosen", "Hey, Vosen!"] if name == "vosen" else [
            "Hey Mero", "Hey, Mero!"]
        lookup = {r["requested_text"]:r["cell_id"] for r in rows
                  if r["voice_id"] == voice}
        need(pair.get("requested_texts") == expected and
             pair.get("cell_ids") == [lookup[p] for p in expected],
             "ENGLISH_PAIR_CELL_ID_DRIFT")
    need(seen_pairs == {(v, n) for v in TRAIN_VOICES + DEVELOPMENT_VOICES
                        for n in ("vosen", "mero")},
         "ENGLISH_PAIR_ROSTER_INCOMPLETE")
    return rows


def diagnostic_plan(report: dict) -> dict:
    """Build a *non-executable* weak-intent plan, not training admission."""
    rows = verify_three_source_report(report)
    output = []
    for row in sorted(rows, key=lambda r:r["cell_id"]):
        phrase = row["requested_text"]
        intent, label = REQUESTS[phrase]
        development = role(row["voice_id"]) == "development-only"
        if intent == "near_miss":
            action = "QUARANTINE_AMBIGUOUS_NEARPHONE"
            weak_class = None
        elif development:
            action = "EXPOSED_DEVELOPMENT_NO_TUNE"
            weak_class = CLASSES[label]
        else:
            action = "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY"
            weak_class = CLASSES[label]
        output.append({
            "cell_id": row["cell_id"],
            "wav_sha256": row["wav_sha256"],
            "pcm_sha256": row["pcm_sha256"],
            "source_family_id": row["source_family_id"],
            "role": row["role"], "intent_requested": intent,
            "weak_wholeclip_class": weak_class,
            "action": action,
            "human_gold": False,
            "training_admitted": False,
            "trusted_negative": False,
            "ctc_target_ids": None,
        })
    tally = {name:sum(r["action"] == name for r in output) for name in (
        "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY",
        "EXPOSED_DEVELOPMENT_NO_TUNE",
        "QUARANTINE_AMBIGUOUS_NEARPHONE")}
    need(tally == {
        "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY":30,
        "EXPOSED_DEVELOPMENT_NO_TUNE":12,
        "QUARANTINE_AMBIGUOUS_NEARPHONE":21},
        "ENGLISH_EXPOSURE_COUNT_DRIFT")
    return {
        "schema":POLICY,
        "status":"WEAK_INTENT_DIAGNOSTIC_NOT_TRAINING_ADMISSION",
        "source_report_sha256":SOURCE_BYTES_SHA256,
        "locale":"en-US",
        "head_kind":"whole-recording-3-class-not-ctc",
        "class_roster":CLASSES,
        "counts":tally, "rows":output,
        "new_audio_calls":0, "new_asr_calls":0, "new_ipa_calls":0,
        "new_kws_model_calls":0, "optimizer_steps":0,
        "machine_observation_is_truth":False,
        "human_gold":False, "training_admitted":False,
        "admitted_training_rows":0,
        "independent_real_human_test":"NOT_RUN",
        "physical_ssc305":"NOT_RUN",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-review",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args(argv)
    try:
        if args.source_review is None:
            need(args.output is None,"ENGLISH_EXPLICIT_SOURCE_REQUIRED")
            print(json.dumps({"schema":POLICY,"source_report_sha256":
                SOURCE_BYTES_SHA256,"admitted_training_rows":0,"model_calls":0},
                sort_keys=True))
            return 0
        need(args.output is not None and args.output.is_absolute() and
             args.output.parent.is_dir() and not args.output.exists() and
             not args.output.parent.is_symlink(),
             "ENGLISH_OUTPUT_CREATE_ONLY")
        result = diagnostic_plan(read_verified_report(args.source_review))
        with args.output.open("x",encoding="utf-8") as f:
            json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2)
            f.write("\n")
        args.output.chmod(0o600)
        print(json.dumps({"status":result["status"],"counts":result["counts"],
                          "admitted_training_rows":0},sort_keys=True))
        return 0
    except (ValueError,OSError,json.JSONDecodeError) as exc:
        print("ENGLISH_LABEL_BRIDGE_STOP: "+type(exc).__name__+": "+str(exc),
              file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
