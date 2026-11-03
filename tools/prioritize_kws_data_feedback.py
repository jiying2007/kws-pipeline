#!/usr/bin/env python3
"""Prioritize bounded next KWS data work using independently read-back evidence.

Never generate speech, call ASR, infer true labels, train, access held-out
qualification, or authorize scaling. A machine-screen report is a *development
diagnostic*, so every task is a source/evidence inspection request only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_data_bulk_effect_boundary import (POLICY as INPUT_POLICY, SCREEN_STATES,
                                           canonical_sha, read_json)

POLICY = "kws-effect-feedback-priorities-v1"


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def derive(report: dict) -> dict:
    require(report.get("schema_version") == 1 and report.get("policy") == INPUT_POLICY
            and report.get("conclusion") == "INDEPENDENT_DEVELOPMENT_DIAGNOSTIC_ONLY",
            "UNSUPPORTED_EFFECT_REPORT")
    for field in ("source_rights_authenticated", "asr_source_authenticated",
                  "human_gold", "training_admitted", "qualification_allowed"):
        require(report.get(field) is False, "UNEXPECTED_EFFECT_AUTHORITY")
    require("shipping_approved" not in report, "UNEXPECTED_EFFECT_AUTHORITY")
    for field in ("kws_data_plan_sha256", "kws_data_screen_sha256", "adapter_lock_sha256"):
        value = report.get(field)
        require(isinstance(value, str) and len(value) == 64 and
                all(c in "0123456789abcdef" for c in value),
                "MISSING_SOURCE_IDENTITY")
    rows = report.get("rows")
    total = report.get("total_planned")
    require(type(total) is int and 1 <= total <= 16384 and
            isinstance(rows, list) and len(rows) == total,
            "EFFECT_ROW_COUNT")
    ids, current, challenges, keyword_counts, source_counts, original_roles = (
        set(), Counter(), defaultdict(Counter), defaultdict(Counter),
        defaultdict(Counter), {})
    generated = 0
    unique_pcm = set()
    tasks = []

    def task(priority: str, code: str, scope: dict, observed: int, action: str) -> None:
        require(priority in ("P0", "P1", "P2") and type(observed) is int and observed >= 0,
                "INVALID_TASK")
        tasks.append({"priority": priority, "code": code, "scope": scope,
                      "observed": observed, "action": action,
                      "new_model_or_generator_calls_authorized": 0,
                      "training_admitted": False, "qualification_allowed": False})

    for row in rows:
        require(isinstance(row, dict) and
                isinstance(row.get("cell_id"), str) and row["cell_id"] not in ids and
                isinstance(row.get("source_family_id"), str) and row["source_family_id"] and
                isinstance(row.get("speaker_group_id"), str) and row["speaker_group_id"] and
                row.get("role") in ("train-candidate", "development-only") and
                isinstance(row.get("requested_challenge"), str) and
                row["requested_challenge"] and
                row.get("screen_status") in SCREEN_STATES and
                row.get("synthetic") is True and
                row.get("training_admitted") is False and
                row.get("qualification_allowed") is False,
                "INVALID_EFFECT_ROW")
        ids.add(row["cell_id"])
        for key in ("source:" + row["source_family_id"],
                    "speaker:" + row["speaker_group_id"]):
            require(key not in original_roles or original_roles[key] == row["role"],
                    "EFFECT_GROUP_ROLE_LEAKAGE")
            original_roles[key] = row["role"]
        kw = row.get("requested_keyword_id")
        require(type(kw) is int and kw in (1, 2) or kw is None, "INVALID_KEYWORD_ID")
        status = row["screen_status"]
        current[status] += 1
        challenges[row["requested_challenge"]][status] += 1
        source_counts[row["source_family_id"]][status] += 1
        if kw is not None:
            keyword_counts[str(kw)][status] += 1
        wav = row.get("generated_audio")
        if wav is None:
            require(status == "NO_GENERATION", "MISSING_AUDIO_WITH_NONEMPTY_SCREEN")
        else:
            require(isinstance(wav, dict) and
                    all(isinstance(wav.get(name), str) and
                        len(wav[name]) == 64 and
                        all(c in "0123456789abcdef" for c in wav[name])
                        for name in ("file_sha256", "pcm_sha256")) and
                    type(wav.get("frames")) is int and wav["frames"] > 0,
                    "BAD_VERIFIED_AUDIO_IDENTITY")
            generated += 1
            unique_pcm.add(wav["pcm_sha256"])
            require(status != "NO_GENERATION", "UNREPORTED_GENERATED_AUDIO")
    expected_counts = dict(current)
    observed_counts = report.get("screen_counts")
    require(isinstance(observed_counts, dict) and
            {k: v for k, v in observed_counts.items() if k in SCREEN_STATES} == expected_counts,
            "EFFECT_COUNTS_DRIFT")
    duplicate_count = observed_counts.get("duplicate_pcm_within_original_group", 0)
    require(type(duplicate_count) is int and
            duplicate_count == generated - len(unique_pcm) and
            type(report.get("unique_pcm_verified")) is int and
            report["unique_pcm_verified"] == len(unique_pcm) and
            type(report.get("total_generated_audio_verified")) is int and
            report["total_generated_audio_verified"] == generated,
            "EFFECT_DUPLICATE_OR_AUDIO_COUNT_DRIFT")
    require(report.get("screen_by_challenge") == {
                k: dict(v) for k, v in challenges.items()} and
            report.get("screen_by_source_family") == {
                k: dict(v) for k, v in source_counts.items()} and
            report.get("screen_by_keyword") == {
                # The source producer emits both known keyword buckets, including
                # a valid empty bucket when a keyword had no scheduled examples.
                k: dict(keyword_counts[k]) for k in ("1", "2")} and
            report.get("unique_source_families_declared") == len(source_counts),
            "EFFECT_STRATUM_COUNTS_DRIFT")

    # The highest priority is to prevent a false or incomplete negative from
    # being recycled into training. No inference from requested intent alone.
    for challenge, group in sorted(challenges.items()):
        wrong_wake = group.get("NEGATIVE_SPOKE_WAKE_OR_ASR_CONFLICT", 0)
        if wrong_wake:
            task("P0", "QUARANTINE_POSSIBLE_WAKE_IN_NEGATIVE",
                 {"requested_challenge": challenge}, wrong_wake,
                 "Inspect original waveform/native ASR disagreement and source identity; do not admit as a negative")
        incomplete = group.get("ASR_INCOMPLETE_HOLD", 0) + group.get("NO_GENERATION", 0)
        if incomplete:
            task("P1", "RESOLVE_UNFINISHED_SOURCE_OR_ASR",
                 {"requested_challenge": challenge}, incomplete,
                 "Trace immutable attempted and completed receipts; do not silently retry consumed calls")
        mispronounced = group.get("ASR_DISAGREEMENT_OR_TTS_MISPRONUNCIATION", 0)
        if mispronounced:
            task("P1", "DIAGNOSE_PRONUNCIATION_OR_ASR",
                 {"requested_challenge": challenge}, mispronounced,
                 "Inspect source-bound three-ASR native outputs and acoustic completeness before a new generation campaign")
    for keyword in ("1", "2"):
        group = keyword_counts[keyword]
        matched = group.get("MACHINE_POSITIVE_CANDIDATE", 0)
        observed = sum(group.values())
        if not matched:
            task("P0", "NO_MACHINE_SCREENED_KEYWORD_POSITIVE",
                 {"keyword_id": int(keyword)}, observed,
                 "Repair pronunciation/voice/prompt coverage using already authorised sources; freeze a new bounded campaign")
        else:
            task("P1", "VERIFY_POSITIVE_SOURCE_LABELS",
                 {"keyword_id": int(keyword)}, matched,
                 "Authenticate actual spoken labels and original source rights before CTC training; machine screen is not gold")
    if len(source_counts) < 2:
        task("P1", "SOURCE_FAMILY_GENERALIZATION_UNPROVEN",
             {"source_families_declared": len(source_counts)}, len(rows),
             "Use independently approved distinct source families for development comparison; never count clones as new real people")
    if duplicate_count:
        task("P1", "REPEATED_PCM_DOES_NOT_ADD_DIVERSITY",
             {}, duplicate_count, "Investigate repeated generated PCM; do not inflate speaker or acoustic coverage")
    task("P0", "INDEPENDENT_RESEARCH_EVIDENCE_MISSING", {}, 0,
         "Obtain genuinely reviewed source permissions, authentic label receipts, independent final-AFE human longstreams and SSC305 target-board evidence")
    tasks.sort(key=lambda r: ({"P0": 0, "P1": 1, "P2": 2}[r["priority"]],
                              r["code"], json.dumps(r["scope"], sort_keys=True)))
    return {
        "schema_version": 1, "policy": POLICY,
        "input_effect_sha256": canonical_sha(report),
        "source_plan_sha256": report["kws_data_plan_sha256"],
        "source_screen_sha256": report["kws_data_screen_sha256"],
        "total_rows": total, "generated_verified": generated,
        "challenge_count": len(challenges), "source_families_declared": len(source_counts),
        "ranked_actions": tasks, "next_campaign_authorized": False,
        "new_model_or_generator_calls_authorized": 0,
        "source_rights_authenticated": False, "asr_source_authenticated": False,
        "human_gold": False, "training_admitted": False,
        "qualification_allowed": False,
        "conclusion": "DEVELOPMENT_DATA_GAPS_ONLY_NOT_AN_EXECUTION_PERMISSION",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = read_json(args.effect)
    result = derive(report)
    dest = args.output.resolve()
    require(dest != args.effect.resolve() and
            dest != ROOT and ROOT not in dest.parents, "PRIVATE_OUTPUT_ONLY")
    require(dest.parent.exists() and dest.parent.is_dir(), "PRIVATE_OUTPUT_DIRECTORY_REQUIRED")
    data = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(dest, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({"policy": POLICY, "actions": len(result["ranked_actions"]),
                      "conclusion": result["conclusion"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
