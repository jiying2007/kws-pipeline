#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re

SHA256_RE = re.compile(r"[0-9a-f]{64}")


def load_object(path: pathlib.Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def normalize_request(path: pathlib.Path) -> dict:
    value = load_object(path, "request")
    required = {
        "schema_version",
        "evidence_class",
        "development_only",
        "selection_feedback_allowed",
        "source_run_id",
        "source_artifact_id",
        "source_head_sha",
        "case_id",
        "expected_keyword_id",
        "expected_classification",
        "expected_model_sha256",
        "expected_keyword_pack_sha256",
    }
    if set(value) != required:
        raise ValueError(
            "request fields mismatch: "
            f"missing={sorted(required-set(value))} "
            f"extra={sorted(set(value)-required)}"
        )
    if value["schema_version"] != 1:
        raise ValueError("request schema_version must be 1")
    if value["evidence_class"] != "retained-frozen-far-context-path-summary-request-v1":
        raise ValueError("request evidence_class mismatch")
    if value["development_only"] is not True:
        raise ValueError("request must be development_only")
    if value["selection_feedback_allowed"] is not False:
        raise ValueError("request must not feed selection")
    for key in ("source_run_id", "source_artifact_id", "expected_keyword_id"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(value["source_head_sha"], str) or re.fullmatch(
        r"[0-9a-f]{40}", value["source_head_sha"]
    ) is None:
        raise ValueError("source_head_sha must be lowercase git SHA")
    if not isinstance(value["case_id"], str) or not value["case_id"]:
        raise ValueError("case_id must be non-empty")
    if value["expected_classification"] not in {
        "mixed-stream-context-sufficient",
        "longer-stream-state-or-history-required",
    }:
        raise ValueError("expected_classification is unsupported")
    for key in ("expected_model_sha256", "expected_keyword_pack_sha256"):
        if not isinstance(value[key], str) or SHA256_RE.fullmatch(value[key]) is None:
            raise ValueError(f"{key} must be lowercase SHA256")
    return dict(value)


def normalize_snapshot(value: object, label: str) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be object or null")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label}.provenance is missing")
    token_advances = int(provenance.get("token_advances", -1))
    exact = int(provenance.get("exact_top_advances", -1))
    fuzzy = int(provenance.get("fuzzy_advances", -1))
    root_exact = int(provenance.get("root_exact_starts", -1))
    root_ambiguous = int(provenance.get("root_ambiguous_starts", -1))
    if min(token_advances, exact, fuzzy, root_exact, root_ambiguous) < 0:
        raise ValueError(f"{label} token provenance contains negative counters")
    if root_exact + root_ambiguous + exact + fuzzy != token_advances:
        raise ValueError(f"{label} token provenance accounting is inconsistent")
    events = provenance.get("fuzzy_events")
    if not isinstance(events, list) or len(events) != fuzzy:
        raise ValueError(f"{label} fuzzy-event count mismatch")
    normalized_events = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"{label}.fuzzy_events[{index}] must be object")
        normalized_events.append(
            {
                "frame_index": int(event["frame_index"]),
                "depth": int(event["depth"]),
                "target_token": int(event["target_token"]),
                "top_token": int(event["top_token"]),
                "logit_gap": finite(event["logit_gap"], f"{label}.fuzzy_events[{index}].logit_gap"),
                "target_rank": int(event["target_rank"]),
            }
        )
    if fuzzy > 0 and root_ambiguous > 0:
        mode = "contains-ambiguous-root-and-fuzzy-advance"
    elif fuzzy > 0:
        mode = "contains-fuzzy-advance"
    elif root_ambiguous > 0:
        mode = "contains-ambiguous-root-start"
    else:
        mode = "exact-top-only"
    return {
        "retention_log": finite(value["retention_log"], f"{label}.retention_log"),
        "confidence": finite(value["confidence"], f"{label}.confidence"),
        "path_mode": mode,
        "provenance": {
            "token_advances": token_advances,
            "exact_top_advances": exact,
            "fuzzy_advances": fuzzy,
            "fuzzy_logit_gap_sum": finite(
                provenance.get("fuzzy_logit_gap_sum", 0.0),
                f"{label}.fuzzy_logit_gap_sum",
            ),
            "fuzzy_logit_gap_max": finite(
                provenance.get("fuzzy_logit_gap_max", 0.0),
                f"{label}.fuzzy_logit_gap_max",
            ),
            "fuzzy_target_rank_sum": int(provenance.get("fuzzy_target_rank_sum", 0)),
            "fuzzy_target_rank_max": int(provenance.get("fuzzy_target_rank_max", 0)),
            "fuzzy_events": normalized_events,
            "root_exact_starts": root_exact,
            "root_ambiguous_starts": root_ambiguous,
            "same_token_retentions": int(provenance.get("same_token_retentions", 0)),
            "blank_retentions": int(provenance.get("blank_retentions", 0)),
        },
    }


def summarize(*, request: dict, retained_root: pathlib.Path) -> dict:
    analysis_root = retained_root / "build/frozen-far-mixed-context-analysis"
    summary = load_object(
        analysis_root / "stream-context-summary.json",
        "stream-context summary",
    )
    if summary.get("evidence_class") != "frozen-far-stream-context-replay-evidence-v1":
        raise ValueError("stream-context summary evidence_class mismatch")
    if summary.get("development_only") is not True:
        raise ValueError("stream-context summary must be development-only")
    if summary.get("selection_feedback_allowed") is not False:
        raise ValueError("stream-context summary must not feed selection")
    if summary.get("model_sha256") != request["expected_model_sha256"]:
        raise ValueError("retained model SHA mismatch")
    if summary.get("keyword_pack_sha256") != request["expected_keyword_pack_sha256"]:
        raise ValueError("retained keyword-pack SHA mismatch")
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise ValueError("stream-context summary cases are missing")
    matches = [
        row for row in cases
        if isinstance(row, dict)
        and isinstance(row.get("case"), dict)
        and row["case"].get("case_id") == request["case_id"]
    ]
    if len(matches) != 1:
        raise ValueError("requested retained case is missing or duplicated")
    row = matches[0]
    if row.get("classification") != request["expected_classification"]:
        raise ValueError("retained case classification drifted")
    case = row["case"]
    if int(case.get("detected_keyword_id", -1)) != request["expected_keyword_id"]:
        raise ValueError("retained keyword id drifted")
    result_path = (
        analysis_root
        / request["case_id"]
        / "stream-context-result.json"
    )
    result = load_object(result_path, "stream-context case result")
    if result.get("classification") != request["expected_classification"]:
        raise ValueError("case result classification drifted")
    analysis = result.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("case analysis is missing")
    if analysis.get("direct_replay_exact_parity") is not True:
        raise ValueError("direct/replay parity is not exact")
    target_path = analysis.get("target_keyword_path")
    if not isinstance(target_path, dict):
        raise ValueError("target keyword path is missing")
    if int(target_path.get("keyword_id", -1)) != request["expected_keyword_id"]:
        raise ValueError("target path keyword id drifted")
    selected = target_path.get("selected_path")
    if not isinstance(selected, dict):
        raise ValueError("selected_path summary is missing")
    snapshots = {
        key: normalize_snapshot(selected.get(key), f"selected_path.{key}")
        for key in (
            "best_retention",
            "best_confidence_retention_pass",
            "best_eligible",
        )
    }
    eligible = snapshots["best_eligible"]
    if int(target_path.get("detections", 0)) > 0 and eligible is None:
        raise ValueError("detected keyword lacks best_eligible path evidence")
    interpretation = (
        "best-eligible-path-contains-fuzzy-advance"
        if eligible is not None and eligible["path_mode"] == "contains-fuzzy-advance"
        else "best-eligible-path-exact-top-only"
        if eligible is not None and eligible["path_mode"] == "exact-top-only"
        else "best-eligible-path-unresolved"
    )
    return {
        "schema_version": 1,
        "evidence_class": "retained-frozen-far-context-path-summary-v1",
        "development_only": True,
        "selection_feedback_allowed": False,
        "source": {
            "run_id": request["source_run_id"],
            "artifact_id": request["source_artifact_id"],
            "head_sha": request["source_head_sha"],
            "case_id": request["case_id"],
        },
        "model_sha256": summary["model_sha256"],
        "keyword_pack_sha256": summary["keyword_pack_sha256"],
        "classification": row["classification"],
        "direct_replay_exact_parity": True,
        "target_keyword_path": {
            key: target_path.get(key)
            for key in (
                "keyword_id",
                "target_depth",
                "max_depth_reached",
                "root_alive_frames",
                "terminal_alive_frames",
                "terminal_speech_alive_frames",
                "terminal_retention_pass_frames",
                "terminal_threshold_pass_frames",
                "pending_frames",
                "detections",
                "threshold",
                "max_terminal_confidence",
                "max_terminal_retention_log",
            )
        },
        "selected_path": snapshots,
        "interpretation": interpretation,
        "scope_note": (
            "selected_path snapshots summarize the strongest retained eligible paths; "
            "they are not, by themselves, proof of the exact emitted detection backpointer"
        ),
    }


def self_test() -> None:
    exact = normalize_snapshot(
        {
            "retention_log": -0.1,
            "confidence": 0.9,
            "provenance": {
                "token_advances": 4,
                "exact_top_advances": 3,
                "fuzzy_advances": 0,
                "fuzzy_logit_gap_sum": 0.0,
                "fuzzy_logit_gap_max": 0.0,
                "fuzzy_target_rank_sum": 0,
                "fuzzy_target_rank_max": 0,
                "fuzzy_events": [],
                "root_exact_starts": 1,
                "root_ambiguous_starts": 0,
                "same_token_retentions": 0,
                "blank_retentions": 2,
            },
        },
        "exact",
    )
    assert exact is not None and exact["path_mode"] == "exact-top-only"
    fuzzy = normalize_snapshot(
        {
            "retention_log": -0.2,
            "confidence": 0.8,
            "provenance": {
                "token_advances": 4,
                "exact_top_advances": 2,
                "fuzzy_advances": 1,
                "fuzzy_logit_gap_sum": 0.2,
                "fuzzy_logit_gap_max": 0.2,
                "fuzzy_target_rank_sum": 2,
                "fuzzy_target_rank_max": 2,
                "fuzzy_events": [
                    {
                        "frame_index": 12,
                        "depth": 3,
                        "target_token": 2,
                        "top_token": 4,
                        "logit_gap": 0.2,
                        "target_rank": 2,
                    }
                ],
                "root_exact_starts": 1,
                "root_ambiguous_starts": 0,
                "same_token_retentions": 1,
                "blank_retentions": 1,
            },
        },
        "fuzzy",
    )
    assert fuzzy is not None and fuzzy["path_mode"] == "contains-fuzzy-advance"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-request", action="store_true")
    parser.add_argument("--request", type=pathlib.Path)
    parser.add_argument("--retained-root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("retained frozen FAR context path summary self-test: PASS")
        return 0
    if args.verify_request:
        if args.request is None:
            parser.error("--verify-request requires --request")
        value = normalize_request(args.request.resolve())
        print(json.dumps(value, sort_keys=True, allow_nan=False))
        return 0
    if args.request is None or args.retained_root is None or args.output is None:
        parser.error("--request, --retained-root and --output are required")
    request = normalize_request(args.request.resolve())
    result = summarize(
        request=request,
        retained_root=args.retained_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_id": result["source"]["case_id"],
                "classification": result["classification"],
                "interpretation": result["interpretation"],
                "target_keyword_path": result["target_keyword_path"],
                "best_eligible": result["selected_path"]["best_eligible"],
                "scope_note": result["scope_note"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
