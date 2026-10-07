"""Saved-only exact98 comparison; never executes a frontend, model, or decoder."""
from collections import Counter
import hashlib
import pathlib

from contracts import require
from compare_saved_scores import compare
from score_endpoint import (SCORE_SCHEMA, FROZEN_MANIFEST_SHA, FROZEN_GEOMETRY_SHA,
                            frozen_document)
import candidate_control

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE_SHA256 = "33ac29d8f668fbede4409eb5bdb0990f8c5ed3714664e38aff426b9ec6606336"
HISTORICAL_SHA256 = "59702fd6dc6d2746a3c9798eecf0a23c56bc57b2059306715ae55f712336c4c3"
GROUP_COUNTS = {"Qwen20": 20, "D20": 20, "Cosy12": 12, "N0": 3,
                "DEMAND": 1, "FLEURS20": 20, "Cosy22": 22}


def signature(row):
    return [(e["keyword"], e["start_frame"], e["end_frame"], e["available_samples"])
            for e in row["events"]]


def event_signature(rows):
    return [(r["recording"], signature(r)) for r in rows]


def counters(row):
    require(row.get("complete") is True and row.get("feed_finish_records_validated") is True,
            "incomplete trace is never a negative")
    require(type(row.get("events")) is list, "complete event list required")
    require(all(type(e.get("keyword")) is int and e["keyword"] in (1, 2)
                for e in row["events"]), "keyword IDs required")
    return Counter(e["keyword"] for e in row["events"])


def qwen_preservation(original, candidate):
    # Keep the reviewed old guards, but do not reuse its gain requirement.
    result = compare(original, candidate)
    return {"all20": result["all20"], "regressions": result["regressions"],
            "no_regressions": not result["regressions"],
            "improvement_required": False, "qualification": False}


def cosy_rows(rows):
    result = {}
    for row in rows:
        if row["group"] == "Cosy22":
            alias = row["alias"]
            require(alias not in result, "duplicate Cosy22 alias")
            counters(row)
            result[alias] = {"complete": row["complete"], "error": row.get("error"),
                             "events": [event["keyword"] for event in row["events"]]}
    return result


def historical_drift(rows, historical):
    """Discrete native signatures only; no reexecution or new event adjudication."""
    by_name = {r["recording"]: r for r in rows}
    result = {}
    for group in ("Qwen20", "Cosy12", "N0", "DEMAND", "FLEURS20"):
        changes = []
        for row in historical["groups"][group]["rows"]:
            name = row["recording"]
            if group == "Cosy12":
                before = [( {"你好小窝": 1, "小窝小窝": 2}[e["keyword"]],
                            round(e["start"] * 100), round(e["end"] * 100),
                            e["available_samples"]) for e in row["native_events"]]
            else:
                before = signature(row)
            after = signature(by_name[name])
            if before != after:
                changes.append({"recording": name, "historical": before, "original_arm": after})
        result[group] = {"discrete_drift": bool(changes), "changes": changes}
    result["D20"] = {"discrete_drift": None,
                     "scope": "No historical native D20 baseline asserted; paired original is the baseline"}
    return result


def compare_pair(original, candidate):
    require(original["schema"] == candidate["schema"] == SCORE_SCHEMA,
            "endpoint98 validated score schema required")
    require(original["status"] == candidate["status"] ==
            "COMPLETE_TRACE_ONLY_NOT_QUALITY_OR_RESOURCE_PASS", "complete validated trace reports required")
    old, new = original["streams"], candidate["streams"]
    require(len(old) == len(new) == 98, "exact98")
    manifest = frozen_document("manifest.json", FROZEN_MANIFEST_SHA)
    expected = manifest["rows"]
    require([r["recording"] for r in old] == [r["recording"] for r in new] ==
            [r["recording"] for r in expected], "exact98 source order")
    require(Counter(r["group"] for r in old) == Counter(GROUP_COUNTS), "exact98 groups")
    for report in (original, candidate):
        for key, value in (("manifest_sha256", FROZEN_MANIFEST_SHA),
                           ("geometry_sha256", FROZEN_GEOMETRY_SHA)):
            require(report["source_hashes"][key] == value, "saved score metadata binding " + key)
    for a, b, frozen in zip(old, new, expected):
        for key, value in frozen.items():
            require(a.get(key) == b.get(key) == value, "source/label drift " + key)
        counters(a)
        counters(b)

    hist = frozen_document("FROZEN-HISTORICAL-EVIDENCE.json", HISTORICAL_SHA256)
    oq = [r for r in old if r["group"] == "Qwen20"]
    nq = [r for r in new if r["group"] == "Qwen20"]
    qwen_historical = qwen_preservation(hist["qwen_frozen_scoring_baseline"]["clips"], nq)
    qwen_paired = qwen_preservation(oq, nq)
    drift = historical_drift(old, hist)
    require(hashlib.sha256((ROOT / "src/candidate_control.py").read_bytes()).hexdigest() == GATE_SHA256,
            "approved Cosy endpoint gate source drift")
    cosy = candidate_control.cosy_endpoint_gate(cosy_rows(old), cosy_rows(new))

    rows, regressions, unadjudicated = [], [], []
    for a, b in zip(old, new):
        ca, cb = counters(a), counters(b)
        group, name, flags = a["group"], a["recording"], []
        if group == "D20":
            target = a["target_ids"]
            presence = [int(any(target[i:i+4] == kw for i in range(len(target)-3)))
                        for kw in ([1, 2, 3, 4], [3, 4, 3, 4])]
            for k in (1, 2):
                if presence[k-1] and ca[k] > 0 and cb[k] == 0:
                    flags.append("lost_weak_positive_K" + str(k))
                if not presence[k-1] and cb[k] > ca[k]:
                    flags.append("new_weak_negative_event_K" + str(k))
                if presence[k-1] and max(cb[k]-1, 0) > max(ca[k]-1, 0):
                    flags.append("new_weak_repeat_K" + str(k))
        if group == "N0" and any(cb[k] > ca[k] for k in (1, 2)):
            flags.append("new_constructed_control_event")
        if flags:
            regressions.append({"recording": name, "flags": flags})
        changed = signature(a) != signature(b)
        changed_fields = (["events"] if changed else []) + (["event_scores"] if
            [e.get("score") for e in a["events"]] != [e.get("score") for e in b["events"]] else []) + [
            key for key in ("greedy_ids", "token_edits", "greedy_exact") if a.get(key) != b.get(key)]
        descriptive = group in ("Cosy12", "DEMAND", "FLEURS20")
        if descriptive and changed_fields:
            unadjudicated.append({"recording": name, "group": group,
                                 "changed_fields": changed_fields, "adjudication": "UNADJUDICATED"})
        rows.append({"recording": name, "group": group, "alias": a.get("alias"),
                     "old_events": a["events"], "candidate_events": b["events"],
                     "old_counts": dict(ca), "candidate_counts": dict(cb),
                     "event_behavior_changed": changed, "changed_fields": changed_fields,
                     "old_greedy_ids": a["greedy_ids"], "candidate_greedy_ids": b["greedy_ids"],
                     "old_token_edits": a["token_edits"], "candidate_token_edits": b["token_edits"],
                     "weak_or_control_regression_flags": flags,
                     "truth_status": "UNADJUDICATED" if descriptive else
                         a.get("label_strength", a.get("label", a.get("kind", "UNKNOWN")))})

    baseline_drift = any(r["discrete_drift"] for r in drift.values()) or cosy["status"] == "STOP_BASELINE_DRIFT"
    old76_ok = qwen_historical["no_regressions"] and qwen_paired["no_regressions"] and not regressions
    status = ("BASELINE_DRIFT" if baseline_drift else
              "OLD76_REGRESSION" if not old76_ok else
              cosy["status"] if cosy["status"] != "FIT_ENDPOINT_GAIN" else
              "REVIEW_REQUIRED_UNADJUDICATED_CHANGES" if unadjudicated else
              "EXPOSED_DEVELOPMENT_SIGNAL_ONLY")
    resources = {}
    for key in ("wall_ns", "process_cpu_ns", "maxrss_kib", "minor_faults", "major_faults",
                "block_input_ops", "block_output_ops", "decoder_rows_decoded"):
        x, y = original["native_run_end"].get(key), candidate["native_run_end"].get(key)
        resources[key] = {"original": x, "candidate": y,
                          "delta": y-x if x is not None and y is not None else None,
                          "ratio": y/x if x and y is not None else None}
    return {"schema": "a20-cosy49-step300-paired-endpoint98-comparison-v1", "status": status,
            "cosy22_endpoint_gate": cosy, "cosy22_gate_source_sha256": GATE_SHA256,
            "qwen_original_discrete_drift": drift["Qwen20"]["discrete_drift"],
            "original_historical_drift": drift,
            "qwen_preservation_against_historical": qwen_historical,
            "qwen_preservation_against_paired_original": qwen_paired,
            "old76_no_regressions": old76_ok, "old76_improvement_required": False,
            "all98": rows, "secondary_regressions": regressions,
            "unadjudicated_behavior_changes": unadjudicated,
            "native_resource_differences": resources,
            "native_wall_and_cpu_timing_scope": original["native_wall_and_cpu_timing_scope"],
            "whole_process_timing_scope": original["whole_process_timing_scope"],
            "resource_scope": "DESCRIPTIVE_SINGLE_PAIR; process/supervisor overhead and optional counters remain in resource evidence",
            "false_activation_rate_per_hour": None,
            "qualification": False, "quality_pass": False, "heldout_or_generalization_pass": False,
            "greedy_exact_used_for_wake_gate": False,
            "retry_or_training_extension_authorized": False, "next_stage_authorized": False}
