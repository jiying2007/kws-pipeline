#!/usr/bin/env python3
"""Select development KWS candidate models by comparable error/resource evidence.

No model inference/training/download is performed; no synthetic result becomes
real-human/SSC305 or shipping qualification. The input scorecards must be
produced under the *same* frozen development corpus and v4 protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compare_dataset_iterations import (  # noqa: E402
    DATASET_IDENTITY_POLICY, FAR_POLICY, compare, far_policy,
    framing, metric, dataset_identity_policy,
)

POLICY = "source-bound-kws-multilingual-development-tournament-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
IDENT = re.compile(r"[a-z][a-z0-9_-]{1,63}\Z")
ONE_SIDED_Z95 = 1.6448536269514722
METRICS = ("frr_upper_95", "far_upper_95_per_hour", "p95_post_end_latency_ms",
           "cpu_ms_per_s", "rss_kib", "p99_block_ms")
RESOURCE_FIELDS = {"measurement_class", "measurement_receipt_sha256",
                   "cpu_ms_per_s", "rss_kib", "p99_block_ms"}
BUDGET_FIELDS = {"min_expected_wakes", "min_negative_hours", "max_frr_upper_95",
                 "max_far_upper_95_per_hour", "max_latency_ms",
                 "max_cpu_ms_per_s", "max_rss_kib", "max_p99_block_ms",
                 "frr_point_tolerance"}
MAX_CANDIDATES = 8


def need(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def json_pairs(pairs):
    result = {}
    for k, v in pairs:
        need(k not in result, "DUPLICATE_JSON_FIELD")
        result[k] = v
    return result


def read_json(path: Path, *, max_bytes: int = 12 * 1024 * 1024) -> dict:
    need(not path.is_symlink() and path.is_file() and
         stat.S_ISREG(path.stat().st_mode) and path.stat().st_nlink == 1 and
         1 <= path.stat().st_size <= max_bytes, "INPUT_NOT_BOUNDED_REGULAR")
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=json_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ValueError("NONFINITE_JSON")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("INVALID_JSON") from exc
    need(type(value) is dict, "INPUT_NOT_OBJECT")
    return value


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def number(value: object, label: str, *, strictly_positive: bool = False) -> float:
    need(type(value) in (int, float), "INVALID_NUMERIC_TYPE:" + label)
    result = float(value)
    need(math.isfinite(result) and (result > 0 if strictly_positive else result >= 0),
         "INVALID_NUMERIC_RANGE:" + label)
    return result


def digest(value: object, n: int, code: str) -> str:
    need(type(value) is str and (HEX64 if n == 64 else HEX40).fullmatch(value) is not None,
         code)
    return value


def resource(row: object) -> dict:
    need(type(row) is dict and set(row) == RESOURCE_FIELDS and
         row["measurement_class"] == "host-development-only", "RESOURCE_EVIDENCE_UNQUALIFIED")
    receipt_sha = digest(row["measurement_receipt_sha256"], 64,
                         "RESOURCE_RECEIPT_SHA_MISSING")
    return {"measurement_class": "host-development-only",
            "measurement_receipt_sha256": receipt_sha,
            **{key: number(row[key], key, strictly_positive=True)
               for key in ("cpu_ms_per_s", "rss_kib", "p99_block_ms")}}


def load_entry(entry: dict, *, expected_role: str) -> tuple[dict, dict, dict]:
    need(type(entry) is dict and set(entry) == {
        "id", "scorecard_path", "scorecard_sha256", "resource"},
         "TOURNAMENT_ENTRY_FIELDS")
    ident = entry["id"]
    need(type(ident) is str and IDENT.fullmatch(ident) is not None,
         "TOURNAMENT_ENTRY_ID")
    name = Path(entry["scorecard_path"])
    need(name.is_absolute() and name.resolve() != ROOT and
         ROOT not in name.resolve().parents, "PRIVATE_SCORECARD_PATH_REQUIRED")
    card = read_json(name)
    expected_sha = digest(entry["scorecard_sha256"], 64, "SCORECARD_SHA_NOT_PINNED")
    need(sha(name.read_bytes()) == expected_sha, "SCORECARD_BYTES_CHANGED")
    need(type(card.get("schema_version")) is int and card["schema_version"] == 4,
         "SCORECARD_V4_REQUIRED")
    need(type(card.get("identity")) is dict and
         card["identity"].get("variable") == "model",
         "MODEL_VARIABLE_ONLY")
    need(dataset_identity_policy(card, ident) == DATASET_IDENTITY_POLICY and
         far_policy(card, ident) == FAR_POLICY,
         "LEGACY_METRIC_POLICY_REJECTED")
    ident_sha = card["identity"]
    need(type(ident_sha.get("code_sha")) is str and
         HEX40.fullmatch(ident_sha["code_sha"]) is not None and
         type(ident_sha.get("model_id")) is str and
         HEX64.fullmatch(ident_sha["model_id"]) is not None and
         type(ident_sha.get("dataset_id")) is str and
         HEX64.fullmatch(ident_sha["dataset_id"]) is not None,
         "UNPINNED_MODEL_OR_DATASET_IDENTITY")
    artifact = card.get("artifacts")
    need(type(artifact) is dict and
         type(artifact.get("references_sha256")) is str and
         HEX64.fullmatch(artifact["references_sha256"]) is not None,
         "SCORECARD_REFERENCE_BINDING_MISSING")
    counts = card["metrics"]
    expected = counts.get("expected")
    fails = counts.get("false_rejects")
    matched = counts.get("matched")
    need(type(expected) is int and expected > 0 and
         type(fails) is int and 0 <= fails <= expected and
         type(matched) is int and matched == expected - fails,
         "INVALID_POSITIVE_EVENT_COUNTS")
    frr = metric(card, "frr")
    need(0 <= frr <= 1 and abs(frr - fails / expected) < 1e-12,
         "FRR_DISAGREES_WITH_EVENT_COUNTS")
    latency = metric(card, "p95_post_end_latency_ms")
    need(latency >= 0, "NEGATIVE_LATENCY")
    exposure = card["exposure"]
    need(type(exposure) is dict, "MISSING_NEGATIVE_EXPOSURE")
    negative_hours = number(exposure.get("exposure_hours"), "exposure_hours",
                            strictly_positive=True)
    # One-sided Wilson 95% upper bound uses the entire number of real
    # expected events declared by the scorecard; it is development-only,
    # NOT a product qualification reference authority.
    z = ONE_SIDED_Z95
    p = fails / expected
    wilson = ((p + z * z / (2 * expected) +
               z * math.sqrt(p * (1-p) / expected + z*z / (4*expected*expected))) /
              (1 + z*z / expected))
    point = {"expected_wakes": expected, "false_rejects": fails,
             "frr": frr, "frr_upper_95": wilson,
             "far_upper_95_per_hour": number(
                 card["far_rate"]["upper_bound_95_per_hour"], "far_bound"),
             "p95_post_end_latency_ms": latency,
             "negative_exposure_hours": negative_hours}
    return card, point, resource(entry["resource"])


def dominates(a: dict, b: dict) -> bool:
    return all(a[k] <= b[k] + 1e-12 for k in METRICS) and \
        any(a[k] < b[k] - 1e-12 for k in METRICS)


def select(plan: dict) -> dict:
    need(type(plan) is dict and set(plan) == {
        "schema_version", "policy", "locale", "role",
        "sealed_or_formal_data_used", "model_calls_authorized",
        "baseline", "candidates", "budgets"} and
         plan["schema_version"] == 1 and plan["policy"] == POLICY and
         plan["locale"] in ("zh-CN", "en-US") and
         plan["role"] == "development-only" and
         plan["sealed_or_formal_data_used"] is False and
         plan["model_calls_authorized"] is False,
         "DEVELOPMENT_ONLY_PRE_REGISTRATION_REQUIRED")
    raw_budget = plan["budgets"]
    need(type(raw_budget) is dict and set(raw_budget) == BUDGET_FIELDS,
         "TOURNAMENT_BUDGET_SCHEMA")
    b = {k: number(value, k, strictly_positive=k not in
                   ("frr_point_tolerance",))
         for k, value in raw_budget.items()}
    need(type(raw_budget["min_expected_wakes"]) is int and
         1 <= raw_budget["min_expected_wakes"] <= 1000000 and
         b["max_frr_upper_95"] <= 1 and b["frr_point_tolerance"] <= 0.05,
         "INVALID_DEVELOPMENT_BUDGET")
    raw_candidates = plan["candidates"]
    need(type(raw_candidates) is list and
         1 <= len(raw_candidates) <= MAX_CANDIDATES, "TOURNAMENT_CANDIDATE_BUDGET")
    original, base, base_resources = load_entry(plan["baseline"], expected_role="development-only")
    ids = {plan["baseline"]["id"]}
    model_ids = {original["identity"]["model_id"]}
    required_identity = original["identity"]
    references = original["artifacts"]["references_sha256"]
    candidates = []
    for entry in raw_candidates:
        need(type(entry) is dict and entry.get("id") not in ids,
             "DUPLICATE_CANDIDATE")
        ids.add(entry["id"])
        card, scores, resources = load_entry(entry, expected_role="development-only")
        # These are immutable existing development result files, never a
        # synthetic task matrix, formal/sealed holdout or new inference call.
        framing(original, card)
        ident = card["identity"]
        need(card["artifacts"]["references_sha256"] == references and
             ident["dataset_id"] == required_identity["dataset_id"] and
             ident["code_sha"] == required_identity["code_sha"] and
             ident["seed"] == required_identity["seed"] and
             ident["protocol"] == required_identity["protocol"] and
             scores["expected_wakes"] == base["expected_wakes"] and
             scores["negative_exposure_hours"] == base["negative_exposure_hours"],
             "NONCOMPARABLE_DEVELOPMENT_COHORT")
        need(ident["model_id"] not in model_ids, "REPEATED_CANDIDATE_MODEL")
        model_ids.add(ident["model_id"])
        # Reuse existing producer's FAR/FRR/latency comparison mathematics.
        comparison = compare(original, card, b["max_far_upper_95_per_hour"],
                             b["frr_point_tolerance"], b["max_latency_ms"])
        values = {**scores, **resources}
        reasons = []
        for field, observed, ceiling in [
            ("positive-support", scores["expected_wakes"], b["min_expected_wakes"]),
            ("negative-exposure", scores["negative_exposure_hours"], b["min_negative_hours"])]:
            if observed < ceiling:
                reasons.append("too-small-" + field)
        for label, observed, ceiling in [
            ("frr-confidence", scores["frr_upper_95"], b["max_frr_upper_95"]),
            ("far-confidence", scores["far_upper_95_per_hour"], b["max_far_upper_95_per_hour"]),
            ("latency", scores["p95_post_end_latency_ms"], b["max_latency_ms"]),
            ("cpu", resources["cpu_ms_per_s"], b["max_cpu_ms_per_s"]),
            ("rss", resources["rss_kib"], b["max_rss_kib"]),
            ("block", resources["p99_block_ms"], b["max_p99_block_ms"])]:
            if observed > ceiling:
                reasons.append(label + "-over-development-budget")
        if scores["frr"] > base["frr"] + b["frr_point_tolerance"]:
            reasons.append("positive-recall-regressed")
        if scores["far_upper_95_per_hour"] > base["far_upper_95_per_hour"] + 1e-12:
            reasons.append("false-accept-confidence-regressed")
        # Do not declare a winner simply because a scorecard exists.
        if comparison["verdict"] == "regressed":
            reasons.append("registered-comparator-regression")
        candidates.append({
            "id": entry["id"],
            "scorecard_sha256": entry["scorecard_sha256"],
            "model_id": ident["model_id"],
            "development_scores": values,
            "paired_verdict": comparison["verdict"],
            "eligible_for_development_rank": not reasons,
            "blockers": sorted(set(reasons)),
        })
    valid = [x for x in candidates if x["eligible_for_development_rank"]]
    frontier = [x for x in valid if not any(
        dominates(y["development_scores"], x["development_scores"])
        for y in valid if x["id"] != y["id"])]
    frontier.sort(key=lambda x: (
        x["development_scores"]["frr_upper_95"],
        x["development_scores"]["far_upper_95_per_hour"],
        x["development_scores"]["p95_post_end_latency_ms"],
        x["development_scores"]["cpu_ms_per_s"], x["id"]))
    chosen = frontier[0]["id"] if frontier else None
    return {
        "schema_version": 1, "policy": POLICY, "locale": plan["locale"],
        "source_role_declared": "development-only",
        "source_role_authentication_verified": False,
        "fixed_development_dataset_id": required_identity["dataset_id"],
        "fixed_code_sha": required_identity["code_sha"],
        "fixed_protocol": required_identity["protocol"],
        "fixed_references_sha256": references,
        "control_model_id": required_identity["model_id"],
        "control_development_scores": {**base, **base_resources},
        "budgets": b, "candidate_count": len(candidates),
        "candidate_results": sorted(candidates, key=lambda x: x["id"]),
        "pareto_frontier_ids": [x["id"] for x in frontier],
        "research_leader_id": chosen,
        "research_leader_only": True,
        "production_model_selected": False,
        "production_promotion_allowed": False,
        "training_admitted": False,
        "independent_real_human_verified": False,
        "physical_ssc305_verified": False,
        "shipping_approved": False,
        "new_model_or_training_calls_authorized": 0,
        "conclusion": "DEVELOPMENT_MODEL_COMPARISON_ONLY_NOT_PRODUCTION_QUALIFICATION",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = select(read_json(args.plan, max_bytes=1024 * 1024))
    dest = args.output.resolve()
    need(dest != ROOT and ROOT not in dest.parents and
         dest != args.plan.resolve() and dest.parent.is_dir(),
         "PRIVATE_CREATE_ONLY_OUTPUT_REQUIRED")
    raw = (json.dumps(result, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(dest, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"policy": POLICY, "locale": result["locale"],
                      "research_leader_id": result["research_leader_id"],
                      "production_model_selected": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
