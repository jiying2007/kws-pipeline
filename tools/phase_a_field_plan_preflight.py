#!/usr/bin/env python3
"""Preflight the *planned* kws-data field pilot against Phase-A capture quotas.

Planning is not recording. No human identity, consent, true transcript, AFE,
qualification audio, independent duration, FAR/FRR or device evidence is
authenticated here. Output remains explicitly NON-QUALIFIED in every case.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
POLICY = "phase-a-field-pilot-planning-coverage-v1"
FIELD_SCHEMA = "xiaowo-field-collection-plan-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
IDENT = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
WORDS = {"你好小窝": 1, "小窝小窝": 2}
CONFUSABLES = frozenset(("你好小屋", "你好小吴", "你好小王",
                         "小屋小屋", "小沃小沃", "你好你好"))
BACKGROUND_PROMPT = "录制真实家庭自然交谈或环境声，保留所有意外唤醒词"
ROLES = frozenset(("train-candidate", "development", "sealed-evaluation"))
SLOT_FIELDS = {
    "slot_id", "speaker_id", "session_id", "source_family_id", "role",
    "room_id", "device_id", "afe_profile_sha256", "distance_m",
    "azimuth_deg", "repetition", "prompt", "intent_kind",
    "intended_keyword_id", "audio_recorded", "human_verbatim_reviewed",
    "rights_approved", "training_admitted",
}


def need(ok: bool, label: str) -> None:
    if not ok:
        raise ValueError(label)


def source_json(path: Path) -> dict:
    need(not path.is_symlink() and path.is_file() and
         stat.S_ISREG(path.stat().st_mode) and path.stat().st_nlink == 1 and
         0 < path.stat().st_size <= 8 * 1024 * 1024,
         "INPUT_NOT_BOUNDED_REGULAR_FILE")
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, "DUPLICATE_INPUT_JSON_KEY")
            result[k] = v
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))
    need(type(value) is dict, "INPUT_MUST_BE_OBJECT")
    return value


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def identifier(value: object, label: str) -> str:
    need(type(value) is str and IDENT.fullmatch(value) is not None, label)
    return value


def expected_tasks() -> set[tuple]:
    tasks = set()
    for distance in (1, 3, 5):
        for azimuth in (0, 180):
            for word in WORDS:
                for repetition in (1, 2):
                    tasks.add(("wake-intent", word, distance, azimuth, repetition))
    for distance in (3, 5):
        for word in CONFUSABLES:
            tasks.add(("confusable-intent", word, distance, 0, 1))
    tasks.add(("continuous-negative-candidate", BACKGROUND_PROMPT, 3, 0, 1))
    need(len(tasks) == 37, "INTERNAL_FIELD_TASK_COUNT")
    return tasks


def slot_identity(row: dict) -> str:
    labels = ("speaker_id", "session_id", "distance_m",
              "azimuth_deg", "repetition", "prompt")
    key = {name: row[name] for name in labels}
    return "s-" + canonical_sha(key)[:24]


def assess(plan: dict, policy: dict, expected_plan_sha256: str) -> dict:
    need(type(expected_plan_sha256) is str and
         SHA.fullmatch(expected_plan_sha256) is not None and
         canonical_sha(plan) == expected_plan_sha256,
         "INDEPENDENT_FROZEN_FIELD_PLAN_DIGEST_MISMATCH")
    need(type(plan) is dict and plan.get("schema") == FIELD_SCHEMA and
         plan.get("source_type") == "human-field-recording-candidate" and
         plan.get("scope") == "private-local-pilot-only" and
         plan.get("roles_are_declared_not_authenticated") is True and
         all(plan.get(k) is False for k in
             ("human_rights_authenticated", "audio_labels_verified",
              "training_admitted", "qualification_allowed", "shipping_approved")) and
         type(plan.get("model_calls")) is int and plan["model_calls"] == 0,
         "FIELD_PLAN_CANNOT_CLAIM_CAPTURE_OR_AUTHORITY")
    need(type(policy) is dict and policy.get("schema_version") == 1 and
         policy.get("policy_id") == "xiaowo-real-human-phase-a-v1" and
         policy.get("shipping_approved") is False,
         "UNEXPECTED_PHASE_A_POLICY")
    minimums = policy.get("minimums")
    slices = policy.get("critical_positive_slices")
    negative_bands = policy.get("critical_negative_exposure_hours")
    need(type(minimums) is dict and type(slices) is list and
         type(negative_bands) is dict and slices and negative_bands,
         "PHASE_A_COVERAGE_POLICY_MISSING")
    for field in ("speakers", "sessions_per_speaker", "expected_wakes_total",
                  "expected_wakes_per_keyword", "negative_audio_hours"):
        need(type(minimums.get(field)) is int and minimums[field] > 0,
             "PHASE_A_QUOTA_NOT_POSITIVE_INTEGER")
    for name, hours in negative_bands.items():
        need(type(name) is str and type(hours) in (int, float) and hours > 0,
             "INVALID_NEGATIVE_EXPOSURE_POLICY")
    slots = plan.get("slots")
    need(type(slots) is list and 1 <= len(slots) <= 3700 and
         type(plan.get("participants")) is int and 1 <= plan["participants"] <= 100 and
         len(slots) == 37 * plan["participants"],
         "FIELD_PLAN_NOT_COMPLETE_37_SLOT_SESSIONS")
    expected = expected_tasks()
    session_tasks: dict[tuple[str, str], set] = defaultdict(set)
    session_details, family_role, speaker_role, session_owner = {}, {}, {}, {}
    ids = set()
    counts = {role: Counter() for role in ROLES}
    planned_kw = Counter()
    planned_distance = defaultdict(Counter)
    rear_orientation_candidates = Counter()
    sealed_sessions = defaultdict(set)
    for row in slots:
        need(type(row) is dict and set(row) == SLOT_FIELDS,
             "FIELD_SLOT_SCHEMA_UNEXPECTED")
        speaker = identifier(row.get("speaker_id"), "INVALID_SPEAKER_ID")
        session = identifier(row.get("session_id"), "INVALID_SESSION_ID")
        family = identifier(row.get("source_family_id"), "INVALID_SOURCE_FAMILY_ID")
        identifier(row.get("room_id"), "INVALID_ROOM_ID")
        identifier(row.get("device_id"), "INVALID_DEVICE_ID")
        role = row.get("role")
        need(type(role) is str and role in ROLES,
             "UNKNOWN_FIELD_ROLE")
        need(row.get("afe_profile_sha256") is None or
             (type(row["afe_profile_sha256"]) is str and
              SHA.fullmatch(row["afe_profile_sha256"]) is not None),
             "INVALID_AFE_IDENTITY")
        need(all(row.get(key) is False for key in
                 ("audio_recorded", "human_verbatim_reviewed",
                  "rights_approved", "training_admitted")),
             "FIELD_SLOT_CANNOT_CLAIM_OBSERVED_AUDIO_OR_AUTHORITY")
        for value in (row.get("distance_m"), row.get("azimuth_deg"), row.get("repetition")):
            need(type(value) is int, "FIELD_GEOMETRY_TYPE")
        task = (row.get("intent_kind"), row.get("prompt"), row["distance_m"],
                row["azimuth_deg"], row["repetition"])
        need(task in expected and
             row.get("intended_keyword_id") ==
             (WORDS[row["prompt"]] if row["prompt"] in WORDS else None) and
             type(row.get("slot_id")) is str and row["slot_id"] == slot_identity(row)
             and row["slot_id"] not in ids,
             "FIELD_INTENT_OR_SLOT_ID_CHANGED")
        ids.add(row["slot_id"])
        group = (speaker, session)
        details = (family, role, row["room_id"], row["device_id"], row["afe_profile_sha256"])
        need(group not in session_details or session_details[group] == details,
             "SESSION_METADATA_DRIFT")
        session_details[group] = details
        need(task not in session_tasks[group],
             "DUPLICATE_FIELD_SESSION_TASK")
        session_tasks[group].add(task)
        need(session not in session_owner or session_owner[session] == speaker,
             "SESSION_SHARED_BY_DIFFERENT_SPEAKERS")
        session_owner[session] = speaker
        for index, value, owners in (("source", family, family_role),
                                     ("speaker", speaker, speaker_role)):
            need(value not in owners or owners[value] == role,
                 "FIELD_SPEAKER_OR_FAMILY_CROSSES_SPLITS")
            owners[value] = role
        counts[role][row["intent_kind"]] += 1
        if role == "sealed-evaluation":
            sealed_sessions[speaker].add(session)
            if row["intent_kind"] == "wake-intent":
                kw = WORDS[row["prompt"]]
                planned_kw[kw] += 1
                planned_distance[kw][row["distance_m"]] += 1
                if row["azimuth_deg"] == 180:
                    rear_orientation_candidates[kw] += 1
    need(len(session_tasks) == plan["participants"] and
         all(tasks == expected for tasks in session_tasks.values()),
         "FIELD_SESSION_TASK_MATRIX_DRIFT")
    # "Sealed" is a plan role, not proof of a fresh independent sealed corpus.
    eligible = len(sealed_sessions)
    missing_second_sessions = sum(len(session_ids) < minimums["sessions_per_speaker"]
                                  for session_ids in sealed_sessions.values())
    min_kw = minimums["expected_wakes_per_keyword"]
    total_wakes = sum(planned_kw.values())
    planned_gaps = {
        "declared_sealed_speakers_missing": max(0, minimums["speakers"] - eligible),
        "declared_sealed_speakers_short_of_session_min": missing_second_sessions,
        "planned_wakes_total_missing": max(0, minimums["expected_wakes_total"] - total_wakes),
        "planned_wakes_per_keyword_missing": {
            str(k): max(0, min_kw - planned_kw[k]) for k in (1, 2)
        },
    }
    slice_rows = []
    for rule in slices:
        need(type(rule) is dict and type(rule.get("name")) is str and
             type(rule.get("min_expected_per_keyword")) is int and
             rule["min_expected_per_keyword"] > 0,
             "INVALID_PHASE_A_POSITIVE_SLICE")
        by_word = {}
        for k in (1, 2):
            if rule.get("field") == "distance_m":
                lower = rule.get("min_exclusive", rule.get("min", 0))
                upper = rule.get("max", 5)
                need(type(lower) in (int, float) and type(upper) in (int, float),
                     "INVALID_PHASE_A_DISTANCE_BOUNDS")
                possible = sum(count for d, count in planned_distance[k].items()
                               if (d > lower if "min_exclusive" in rule else d >= lower)
                               and d <= upper)
            elif type(rule.get("tag")) is str:
                # The field plan has NO verified event tags. In particular,
                # azimuth=180 is not a witnessed "rear" annotation.
                possible = 0
            else:
                raise ValueError("UNSUPPORTED_PHASE_A_SLICE")
            by_word[str(k)] = {
                "planned_candidate_slots": possible,
                "planned_deficit": max(0, rule["min_expected_per_keyword"] - possible),
                "verified_labelled_events": 0,
            }
        slice_rows.append({"name": rule["name"], "requested_support": by_word,
                           "actual_qualification_support_verified": False})
    return {
        "schema_version": 1, "policy": POLICY,
        "source_plan_sha256": expected_plan_sha256,
        "phase_a_policy_sha256": canonical_sha(policy),
        "deployment_tag": policy.get("deployment_tag"),
        "declared_session_participants": plan["participants"],
        "declared_role_slot_counts": {k: dict(v) for k, v in sorted(counts.items())},
        "declared_sealed_speakers": eligible,
        "declared_sealed_sessions": sum(len(x) for x in sealed_sessions.values()),
        "sealed_requested_wakes_by_keyword": {
            str(k): planned_kw[k] for k in (1, 2)},
        "sealed_rear_azimuth_candidate_wakes": {
            str(k): rear_orientation_candidates[k] for k in (1, 2)},
        "sealed_continuous_negative_prompt_slots":
            counts["sealed-evaluation"]["continuous-negative-candidate"],
        "planned_quota_gaps": planned_gaps,
        "critical_positive_slice_plans": slice_rows,
        "critical_negative_exposure_hours_missing": dict(sorted(negative_bands.items())),
        "verified_negative_audio_hours": 0,
        "verified_real_speakers": 0, "verified_actual_keyword_events": 0,
        "actual_spoken_labels_verified": False,
        "final_afe_identity_verified": False,
        "speaker_independence_verified": False,
        "rights_authenticated": False, "training_admitted": False,
        "qualification_allowed": False, "shipping_approved": False,
        "conclusion": "FIELD_PLAN_COVERAGE_ONLY_REAL_RECORDINGS_REQUIRED",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field-plan", type=Path, required=True)
    p.add_argument("--phase-a-policy", type=Path, required=True)
    p.add_argument("--expected-plan-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    result = assess(source_json(args.field_plan), source_json(args.phase_a_policy),
                    args.expected_plan_sha256)
    output = args.output.resolve()
    need(output != args.field_plan.resolve() and
         output != args.phase_a_policy.resolve() and
         output != ROOT and ROOT not in output.parents and
         output.parent.is_dir(), "PRIVATE_CREATE_ONLY_OUTPUT_REQUIRED")
    raw = (json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    handle = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(handle, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"policy": POLICY, "conclusion": result["conclusion"],
                      "declared_sealed_speakers": result["declared_sealed_speakers"],
                      "planned_quota_gaps": result["planned_quota_gaps"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
