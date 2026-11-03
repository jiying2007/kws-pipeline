#!/usr/bin/env python3
"""Build private, executable *capture requests* for missing Phase-A scenes.

This only schedules supplementary field recording tasks derived from complete,
independently frozen kws-data v1 field-pilot plans. It does not record a
microphone, classify speech, create training labels, grant consent, run an AFE
or test KWS/SSC305. Planned duration/event counts are NEVER observed evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path

from phase_a_field_plan_preflight import (
    ROOT, assess_many, canonical_sha, need, source_json,
)

POLICY = "phase-a-supplemental-capture-request-v1"
SCENES = {
    "rear": {"distance_m": 5, "azimuth_deg": 180},
    "playback": {"distance_m": 5, "azimuth_deg": 0},
    "double_talk": {"distance_m": 3, "azimuth_deg": 0},
    "robot_motion": {"distance_m": 3, "azimuth_deg": 0},
}
WORDS = {"你好小窝": 1, "小窝小窝": 2}
NEGATIVE_SCENES = ("household", "speech_confusion", "playback", "robot_motion")
BUFFER_FACTOR = 1.10
MAX_SEGMENT_SECONDS = 600
MAX_TASKS = 4096


def _task_id(record: dict) -> str:
    # Original speaker/session identities are already pseudonyms from the
    # separately frozen private field plan. No external name or raw audio.
    keys = ("speaker_id", "session_id", "source_family_id", "kind",
            "scene", "prompt_requested", "keyword_id_requested",
            "distance_m_requested", "azimuth_deg_requested", "repetition")
    return "supp-" + canonical_sha({k: record[k] for k in keys})[:24]


def _sessions(plans: list[dict]) -> dict[tuple[str, str], dict]:
    data = {}
    for plan in plans:
        for slot in plan["slots"]:
            if slot["role"] != "sealed-evaluation":
                continue
            key = (slot["speaker_id"], slot["session_id"])
            value = {k: slot[k] for k in (
                "speaker_id", "session_id", "source_family_id",
                "role", "room_id", "device_id", "afe_profile_sha256")}
            if key in data:
                need(data[key] == value, "CAPTURE_SESSION_METADATA_DRIFT")
            else:
                data[key] = value
    return dict(sorted(data.items()))


def build(plans: list[dict], policy: dict, frozen_hashes: list[str]) -> dict:
    source = assess_many(plans, policy, frozen_hashes)
    minimums = policy["minimums"]
    sessions = _sessions(plans)
    sealed_people = defaultdict(set)
    families = {}
    for speaker, session_id in sessions:
        sealed_people[speaker].add(session_id)
        families[speaker] = sessions[(speaker, session_id)]["source_family_id"]
    need(len(sealed_people) >= minimums["speakers"] and
         len(set(families.values())) >= minimums["speakers"] and
         all(len(v) >= minimums["sessions_per_speaker"]
             for v in sealed_people.values()),
         "FRESH_SEALED_COHORT_MINIMUM_NOT_DECLARED")
    number_of_sessions = len(sessions)

    # Work from the immutable product policy, not a hardcoded/looser target.
    positive = {rule["name"]: rule for rule in policy["critical_positive_slices"]
                if isinstance(rule, dict) and "tag" in rule}
    need(set(SCENES) <= set(positive) and all(
        positive[k]["tag"] == k and
        type(positive[k].get("min_expected_per_keyword")) is int and
        positive[k]["min_expected_per_keyword"] > 0 for k in SCENES),
        "MISSING_REQUIRED_CRITICAL_POSITIVE_TAGS")

    exposure = policy["critical_negative_exposure_hours"]
    need(set(exposure) == set(NEGATIVE_SCENES) and
         all(type(v) in (int, float) and 0 < v < float("inf")
             for v in exposure.values()),
         "INVALID_PHASE_A_NEGATIVE_SCENES")
    hours = {name: float(exposure[name]) for name in NEGATIVE_SCENES}
    # The mandatory total may exceed the sum of named critical strata.
    # Assign the uncovered portion to ordinary household capture; do not
    # double count the same planned seconds in multiple strata.
    remaining = max(0.0, float(minimums["negative_audio_hours"]) - sum(hours.values()))
    hours["household"] += remaining
    seconds = {name: math.ceil(value * BUFFER_FACTOR * 3600 / number_of_sessions)
               for name, value in hours.items()}
    need(all(1 <= n <= MAX_SEGMENT_SECONDS for n in seconds.values()),
         "NEGATIVE_SEGMENTS_OVER_10_MINUTES_EXPAND_SEALED_COHORT")

    per_scene_repeats = {
        name: math.ceil(int(positive[name]["min_expected_per_keyword"])
                        * BUFFER_FACTOR / number_of_sessions)
        for name in SCENES
    }
    need(all(1 <= reps <= 4 for reps in per_scene_repeats.values()),
         "POSITIVE_SCENE_REPETITIONS_OVER_BOUNDED_PILOT")

    tasks, ids = [], set()
    positive_counts = defaultdict(Counter)
    negative_seconds = Counter()
    for key, session in sessions.items():
        for scene in sorted(SCENES):
            geometry = SCENES[scene]
            for text, keyword_id in sorted(WORDS.items(), key=lambda x: x[1]):
                for rep in range(1, per_scene_repeats[scene] + 1):
                    item = {
                        **session,
                        "kind": "wake-intent",
                        "scene": scene,
                        "prompt_requested": text,
                        "keyword_id_requested": keyword_id,
                        "distance_m_requested": geometry["distance_m"],
                        "azimuth_deg_requested": geometry["azimuth_deg"],
                        "repetition": rep,
                        "tags_requested": [scene],
                        "duration_s_requested": None,
                        "actual_spoken_keyword_verified": False,
                        "actual_acoustic_tags_verified": False,
                        "rights_authenticated": False,
                        "training_admitted": False,
                        "qualification_allowed": False,
                    }
                    item["request_id"] = _task_id(item)
                    need(item["request_id"] not in ids, "DUPLICATE_CAPTURE_REQUEST_ID")
                    ids.add(item["request_id"])
                    tasks.append(item)
                    positive_counts[scene][str(keyword_id)] += 1
        for scene in NEGATIVE_SCENES:
            item = {
                **session,
                "kind": "continuous-negative-candidate",
                "scene": scene,
                "prompt_requested": None,
                "keyword_id_requested": None,
                "distance_m_requested": 3,
                "azimuth_deg_requested": 0,
                "repetition": 1,
                "tags_requested": [scene],
                "duration_s_requested": seconds[scene],
                "actual_spoken_keyword_verified": False,
                "actual_acoustic_tags_verified": False,
                "rights_authenticated": False,
                "training_admitted": False,
                "qualification_allowed": False,
            }
            item["request_id"] = _task_id(item)
            need(item["request_id"] not in ids, "DUPLICATE_CAPTURE_REQUEST_ID")
            ids.add(item["request_id"])
            tasks.append(item)
            negative_seconds[scene] += seconds[scene]
    tasks.sort(key=lambda x: x["request_id"])
    need(len(tasks) <= MAX_TASKS, "SUPPLEMENTAL_TASK_BUDGET_EXCEEDED")
    for name, rule in positive.items():
        need(all(positive_counts[name][str(k)] >=
                 rule["min_expected_per_keyword"] for k in (1, 2)),
             "PLANNED_CRITICAL_POSITIVE_QUOTA_MISSING")
    need(all(negative_seconds[name] / 3600 >= exposure[name]
             for name in NEGATIVE_SCENES) and
         sum(negative_seconds.values()) / 3600 >= minimums["negative_audio_hours"],
         "PLANNED_NEGATIVE_DURATION_QUOTA_MISSING")

    return {
        "schema_version": 1, "policy": POLICY,
        "frozen_source_plan_sha256s": sorted(frozen_hashes),
        "derived_field_plan_sha256": source["source_plan_sha256"],
        "phase_a_policy_sha256": canonical_sha(policy),
        "deployment_tag": policy["deployment_tag"],
        "declared_sealed_speakers": len(sealed_people),
        "declared_sealed_source_families": len(set(families.values())),
        "declared_sealed_sessions": number_of_sessions,
        "new_capture_requests": len(tasks),
        "capture_requests_sha256": canonical_sha(tasks),
        "requested_positive_by_scene_keyword": {
            s: dict(sorted(c.items())) for s, c in sorted(positive_counts.items())},
        "requested_negative_duration_s_per_session": dict(sorted(seconds.items())),
        "requested_negative_hours_by_scene": {
            s: round(n / 3600, 6) for s, n in sorted(negative_seconds.items())},
        "requested_negative_hours_total": round(sum(negative_seconds.values()) / 3600, 6),
        "requests": tasks,
        "observed_human_speakers": 0,
        "observed_wake_events": 0,
        "observed_post_afe_negative_hours": 0,
        "actual_acoustic_tags_verified": False,
        "source_rights_authenticated": False,
        "independent_human_labels_verified": False,
        "final_afe_verified": False,
        "physical_ssc305_verified": False,
        "training_admitted": False,
        "qualification_allowed": False,
        "shipping_approved": False,
        "conclusion": "SUPPLEMENTARY_CAPTURE_INTENT_ONLY_NOT_ACOUSTIC_EVIDENCE",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-plan", type=Path, action="append", required=True)
    parser.add_argument("--expected-plan-sha256", action="append", required=True)
    parser.add_argument("--phase-a-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args(argv)
    need(1 <= len(a.field_plan) == len(a.expected_plan_sha256) <= 8,
         "FIELD_PLAN_HASH_COUNT")
    plans = [source_json(path, max_bytes=1024 * 1024) for path in a.field_plan]
    policy = source_json(a.phase_a_policy)
    result = build(plans, policy, a.expected_plan_sha256)
    output = a.output.resolve()
    need(output != ROOT and ROOT not in output.parents and
         all(output != path.resolve() for path in
             (*a.field_plan, a.phase_a_policy)) and output.parent.is_dir(),
         "PRIVATE_CREATE_ONLY_OUTPUT_REQUIRED")
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "policy": POLICY, "capture_requests": len(result["requests"]),
        "requested_negative_hours_total": result["requested_negative_hours_total"],
        "conclusion": result["conclusion"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
