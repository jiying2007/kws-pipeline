#!/usr/bin/env python3
"""Compare existing paired replay receipts; never execute or synthesize traces."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

from contracts import (ARMS, ContractError, MODEL_SHA, PACK_SHA, canonical_sha,
                       input_binding, is_sha, load_json, number, require,
                       score_events, sha256_file, validate_arm_receipt,
                       validate_manifest, validate_plan, validate_saved_inputs, validate_detection_values)


def compare_saved(plan, manifest, receipts, results_root, *, reference_receipt=None):
    validate_plan(plan)
    validate_manifest(manifest)
    require(manifest["purpose"] == plan["purpose"], "current comparator is fixed mechanism-only")
    require(isinstance(receipts, dict), "paired receipts must be an object")
    require(set(receipts) == set(ARMS), "exactly both paired receipts required")
    events = {arm: validate_arm_receipt(receipts[arm], arm, plan, manifest, results_root)
              for arm in ARMS}
    require(len({r["decoder_executable_sha256"] for r in receipts.values()}) == 1,
            "both arms must use the same research executable with only child-admission mode changed")
    if reference_receipt is not None:
        events["production_reference"] = validate_arm_receipt(reference_receipt, "production_reference",
            plan, manifest, results_root, reference=True)
    per_trace = []
    for row in manifest["recordings"]:
        counts = {}
        event_rows = {}
        for arm in ARMS:
            found = [event for event in events[arm] if event["recording"] == row["recording"]]
            event_rows[arm] = found
            counts[arm] = {"total": len(found), "by_keyword": dict(sorted(Counter(
                str(event["keyword_id"]) for event in found).items()))}
        item = {"recording": row["recording"], "trace_sha256": row["trace"]["sha256"],
                          "overlap_group": row["overlap_group"], "initial_state": row["initial_state"],
                          "counts": counts, "events": event_rows,
                          "single_fuzzy_wo_minus_strict_events": counts[ARMS[1]]["total"] - counts[ARMS[0]]["total"],
                          "expected_events": None if row["expected"] is None else len(row["expected"]),
                          "annotation_role": row["annotation_role"]}
        if reference_receipt is not None:
            reference_events = [event for event in events["production_reference"] if event["recording"] == row["recording"]]
            item["production_reference"] = {"reference_only": True, "total": len(reference_events),
                "by_keyword": dict(sorted(Counter(str(event["keyword_id"]) for event in reference_events).items())),
                "events": reference_events}
        per_trace.append(item)
    return {
        "schema": "kws-saved-paired-comparison-v1", "status": "SAVED_MECHANISM_COMPARISON_COMPLETE",
        "purpose": "saved_synthetic_mechanism_regression", "threshold": 0.55,
        "production_comparison_limit": "Both research arms narrow the production default fuzzy-child policy. B versus A improvement does not establish improvement over production.",
        "production_reference": {"status": "SAVED_CONTROLLED_REFERENCE" if reference_receipt is not None else "NOT_RUN",
            "reference_only": True, "selector_candidate": False,
            "decoder_executable_sha256": reference_receipt["decoder_executable_sha256"] if reference_receipt else None},
        "input_binding": input_binding(plan, manifest),
        "decoder_executable_sha256": receipts[ARMS[0]]["decoder_executable_sha256"],
        "recordings": per_trace,
        "aggregate_event_count": None, "aggregate_exposure_hours": None,
        "aggregation_limit": "Traces overlap. Per-trace counts are mechanism observations; do not sum into FAR.",
        "qualification": {"status": "INSUFFICIENT_EVIDENCE", "matched_far_status": "NOT_RUN",
                          "ordinary_real_negative_hours": 0.0, "independent_accent_positives": 0,
                          "independent_farfield_positives": 0, "frr": None, "far_per_hour": None,
                          "negative_far_upper95_per_hour": None},
        "historical_initial_state_equivalence_established": False,
        "literal_wu_recognition_claimed": False, "training_admission": False,
        "acoustic_forward_calls_by_comparator": 0, "decoder_replay_calls_by_comparator": 0,
        "receipt_limit": "Receipts bind saved outputs and declared successful full-trace invocations; they are not independent execution attestations.",
    }


def score_qualified(manifest, detections, *, split, arm, threshold, replay):
    """Future event scoring only, given saved full-replay provenance.

    Not invoked by the current mechanism CLI. No threshold is derived by filtering
    detections. The fixture tests do not establish a real measurement.
    """
    validate_manifest(manifest)
    require(split in ("dev", "test") and arm in ARMS, "qualified split/arm required")
    number(threshold, "threshold")
    require(0 < threshold <= 1, "threshold must be in (0,1]")
    rows = [row for row in manifest["recordings"] if row["split"] == split]
    require(rows, "qualified split empty")
    require(isinstance(replay, dict) and isinstance(replay.get("contract"), dict), "full replay must be an object")
    decoder_sha = replay["contract"].get("decoder_executable_sha256")
    require(is_sha(decoder_sha), "future research decoder executable identity required")
    controls = deepcopy(load_json(Path(__file__).resolve().parent / "plan.json")["shared_controls"])
    del controls["threshold"]
    expected_replay = {"kind": "full_decoder_replay", "split": split, "arm": arm,
                       "threshold": threshold, "model_sha256": MODEL_SHA, "keyword_pack_sha256": PACK_SHA,
                       "manifest_canonical_sha256": canonical_sha(manifest),
                       "decoder_executable_sha256": decoder_sha, "shared_controls_sha256": canonical_sha(controls),
                       "trace_sha256_by_recording": {row["recording"]: row["trace"]["sha256"] for row in rows}}
    require(isinstance(replay, dict), "full replay must be an object")
    require(replay.get("contract") == expected_replay and is_sha(replay.get("receipt_sha256")),
            "each threshold needs its own complete saved decoder replay receipt; filtering is forbidden")
    validate_detection_values(detections)
    refs = score_events.validate_recordings(rows)
    found = score_events.validate_detections(detections, refs)
    summary, _, _ = score_events.score(refs, found,
        score_events.DEFAULT_PRE_TOLERANCE_MS / 1000,
        score_events.DEFAULT_POST_TOLERANCE_MS / 1000)
    # The reused scorer reports 0 FRR with no positives; preserve non-estimability here.
    if summary["expected"] == 0:
        summary["frr"] = None
    enough = summary["expected"] > 0 and summary["negative_recording_audio_hours"] > 0
    return {"status": "QUALIFIED_RESEARCH_MEASUREMENT" if enough else "INSUFFICIENT_EVIDENCE",
            "split": split, "arm": arm, "threshold": threshold,
            "manifest_canonical_sha256": canonical_sha(manifest),
            "denominator_sha256": canonical_sha(rows), "replay_receipt_sha256": replay["receipt_sha256"],
            "full_decoder_replay": True, "replay_contract": expected_replay, "summary": summary,
            "qualification_claimed": False,
            "scoring_inputs": {"manifest": deepcopy(manifest), "detections": deepcopy(detections)}}


def validate_measurement_report(report):
    """Recompute metrics and contract identities before trusting a saved summary."""
    require(isinstance(report, dict), "measurement report must be an object")
    inputs = report.get("scoring_inputs")
    require(isinstance(inputs, dict), "retained scoring inputs required")
    rebuilt = score_qualified(inputs.get("manifest"), inputs.get("detections"),
        split=report.get("split"), arm=report.get("arm"), threshold=report.get("threshold"),
        replay={"contract": report.get("replay_contract"), "receipt_sha256": report.get("replay_receipt_sha256")})
    require(report == rebuilt, "saved measurement report/summary differs from recomputed scoring inputs")
    return report


def select_dev_thresholds(reports, *, max_negative_far_upper95_per_hour):
    """Future dev-only selection under the same caller-supplied upper95 FAR budget."""
    budget = number(max_negative_far_upper95_per_hour, "explicit caller FAR budget")
    require(isinstance(reports, list) and reports, "development reports required")
    groups = {arm: [] for arm in ARMS}
    for report in reports:
        require(isinstance(report, dict), "development report must be an object")
        require(report.get("split") == "dev", "test-set threshold selection forbidden")
        require(report.get("arm") in ARMS and report.get("full_decoder_replay") is True and
                is_sha(report.get("replay_receipt_sha256")), "full decoder replay report required")
        require(report.get("status") == "QUALIFIED_RESEARCH_MEASUREMENT", "INSUFFICIENT_EVIDENCE")
        contract = report.get("replay_contract", {})
        require(contract.get("threshold") == report["threshold"] and contract.get("arm") == report["arm"]
                and contract.get("split") == "dev" and
                contract.get("manifest_canonical_sha256") == report["manifest_canonical_sha256"],
                "development report differs from its full replay contract")
        validate_measurement_report(report)
        groups[report["arm"]].append(report)
    require(len({r["replay_receipt_sha256"] for r in reports}) == len(reports),
            "distinct arm/threshold contracts cannot reuse one replay receipt")
    require(len({r["manifest_canonical_sha256"] for r in reports}) == 1 and
            len({r["denominator_sha256"] for r in reports}) == 1, "paired development inputs/denominators differ")
    require(len({r["replay_contract"]["decoder_executable_sha256"] for r in reports}) == 1 and
            len({r["replay_contract"]["shared_controls_sha256"] for r in reports}) == 1,
            "paired decoder executable/policy inputs differ")
    grids = [sorted(r["threshold"] for r in groups[arm]) for arm in ARMS]
    require(grids[0] and grids[0] == grids[1] and len(grids[0]) == len(set(grids[0])),
            "both arms need the same unique fully replayed threshold grid")
    selected = {}
    for arm, candidates in groups.items():
        eligible = []
        for report in candidates:
            summary = report["summary"]
            require(summary["expected"] > 0 and summary["negative_recording_audio_hours"] > 0,
                    "INSUFFICIENT_EVIDENCE: positive truth and nonzero ordinary negative hours required")
            bound = summary["negative_recording_far_upper_95_per_hour"]
            require(bound is not None, "INSUFFICIENT_EVIDENCE: no FAR bound")
            number(bound, "negative upper95 FAR")
            number(summary["frr"], "FRR")
            require(summary["negative_recording_far_confidence"] == .95 and
                    summary["negative_recording_far_policy"] == "negative-only-recordings-poisson-upper-v1",
                    "negative-only Poisson95 FAR semantics required")
            if bound <= budget:
                eligible.append(report)
        if not eligible:
            return {"status": "NO_DEVELOPMENT_THRESHOLD_MEETS_BUDGET", "selected": None,
                    "max_negative_far_upper95_per_hour": budget}
        winner = min(eligible, key=lambda r: (r["summary"]["frr"],
                       r["summary"]["negative_recording_far_upper_95_per_hour"], r["threshold"]))
        selected[arm] = winner["threshold"]
    return {"status": "DEVELOPMENT_THRESHOLDS_FROZEN", "selected": selected,
            "max_negative_far_upper95_per_hour": budget,
            "manifest_canonical_sha256": reports[0]["manifest_canonical_sha256"],
            "development_reports_sha256": canonical_sha(reports),
            "development_reports": deepcopy(reports),
            "decoder_executable_sha256": reports[0]["replay_contract"]["decoder_executable_sha256"],
            "shared_controls_sha256": reports[0]["replay_contract"]["shared_controls_sha256"],
            "tie_break": "lowest FRR, then lowest negative-only upper95 FAR, then lowest threshold",
            "test_selection_allowed": False}


def evaluate_frozen_test(selection, reports):
    """Accept exactly one held-out replay at each already frozen arm threshold."""
    require(isinstance(selection, dict), "selection must be an object")
    require(selection.get("status") == "DEVELOPMENT_THRESHOLDS_FROZEN", "development selection not frozen")
    recomputed = select_dev_thresholds(selection.get("development_reports"),
        max_negative_far_upper95_per_hour=selection.get("max_negative_far_upper95_per_hour"))
    require(selection == recomputed, "frozen development selection differs from retained development evidence")
    require(isinstance(reports, list) and len(reports) == 2 and all(isinstance(r, dict) for r in reports)
            and {r.get("arm") for r in reports} == set(ARMS),
            "exactly one test report per arm required; no test selection")
    all_receipts = [r["replay_receipt_sha256"] for r in selection["development_reports"]]
    all_receipts += [r.get("replay_receipt_sha256") for r in reports]
    require(all(is_sha(value) for value in all_receipts) and len(set(all_receipts)) == len(all_receipts),
            "distinct dev/test/arm/threshold contracts cannot reuse one replay receipt")
    for report in reports:
        require(report.get("status") == "QUALIFIED_RESEARCH_MEASUREMENT", "INSUFFICIENT_EVIDENCE: frozen test measurements incomplete")
        require(report.get("split") == "test" and report.get("threshold") == selection["selected"][report["arm"]],
                "test threshold must equal development-frozen threshold")
        require(report.get("manifest_canonical_sha256") == selection["manifest_canonical_sha256"] and
                report.get("full_decoder_replay") is True and is_sha(report.get("replay_receipt_sha256")),
                "frozen test input/replay binding mismatch")
        contract = report.get("replay_contract", {})
        require(contract.get("threshold") == report["threshold"] and contract.get("arm") == report["arm"]
                and contract.get("split") == "test", "test report differs from its full replay contract")
        require(contract.get("decoder_executable_sha256") == selection.get("decoder_executable_sha256") and
                contract.get("shared_controls_sha256") == selection.get("shared_controls_sha256"),
                "test decoder executable/policy differs from development freeze")
        validate_measurement_report(report)
    require(reports[0]["denominator_sha256"] == reports[1]["denominator_sha256"], "test arm denominators differ")
    return {"status": "FROZEN_TEST_REPORTED", "selection": selection, "arms": reports,
            "qualification_claimed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--plan", type=Path, default=here / "plan.json")
    parser.add_argument("--manifest", type=Path, default=here / "data_manifest.json")
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--strict-receipt", type=Path)
    parser.add_argument("--fuzzy-receipt", type=Path)
    parser.add_argument("--reference-receipt", type=Path, help="optional fresh unmodified-production saved reference; never a selector candidate")
    parser.add_argument("--verify-inputs-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        plan, manifest = load_json(args.plan), load_json(args.manifest)
        verification = validate_saved_inputs(plan, manifest, args.trace_root)
        if args.verify_inputs_only:
            require(not any((args.results_root, args.strict_receipt, args.fuzzy_receipt, args.reference_receipt)),
                    "input-only validation must not also claim saved results")
            result = {**verification, "result_status": "NOT_RUN", "qualification_status": "INSUFFICIENT_EVIDENCE"}
        else:
            require(all((args.results_root, args.strict_receipt, args.fuzzy_receipt)),
                    "both saved arm receipts and results root required; missing results never trigger replay")
            result = compare_saved(plan, manifest, {
                ARMS[0]: load_json(args.strict_receipt), ARMS[1]: load_json(args.fuzzy_receipt)}, args.results_root,
                reference_receipt=load_json(args.reference_receipt) if args.reference_receipt else None)
            result["receipt_sha256"] = {ARMS[0]: sha256_file(args.strict_receipt),
                                        ARMS[1]: sha256_file(args.fuzzy_receipt)}
            if args.reference_receipt:
                result["receipt_sha256"]["production_reference"] = sha256_file(args.reference_receipt)
        result["plan_file_sha256"] = sha256_file(args.plan)
        result["manifest_file_sha256"] = sha256_file(args.manifest)
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            require(not args.output.exists(), "output already exists; preserve saved evidence")
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
        return 0
    except (ContractError, ValueError, KeyError, TypeError, OSError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
