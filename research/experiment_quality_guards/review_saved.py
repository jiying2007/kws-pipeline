"""Read public JSON fixtures only; no model, ASR, audio or network operation."""
import argparse
import json
from pathlib import Path
import quality_gates as gates


def report(fixture):
    coverage = gates.coverage_admission(fixture["coverage_rows"], fixture["declarations"], fixture["coverage_policy"], fixture["coverage_policy_sha256"])
    identities = gates.identity_audit(fixture["old6_rows"], fixture["identity_history"])
    identities.pop("ledger")
    events = gates.event_comparison(fixture["old6_rows"], fixture["old6_original"], fixture["old6_candidate"],
                                   {"historical_training_greedy_exact": "49/49", "scope": "training fit only"})
    timing = gates.availability_report(fixture["timing_illustration_original"], fixture["timing_illustration_candidate"], fixture["timing_illustration_sample_rates"])
    aggregate = fixture["historical_old98_aggregate"]
    rss = aggregate["lifetime_rss_kib"]
    sampled = aggregate["sampled_rss_bytes"]
    positives = [row for row in events["rows"] if row["expected"]]
    negatives = [row for row in events["rows"] if not row["expected"]]
    counts = {arm: {"wake_recordings_detected": sum(not row[arm]["misses"] for row in positives),
                    "wake_recordings": len(positives),
                    "nonwake_recordings_with_false_events": sum(bool(row[arm]["false_keywords"]) for row in negatives),
                    "nonwake_recordings": len(negatives)} for arm in ("original", "candidate")}
    return {"status": "PURE_REGRESSION_REPORT", "new_model_asr_training_calls": 0,
            "evidence_basis": fixture["evidence_basis"],
            "coverage": coverage, "coverage_scope": "invented missing-positive examples",
            "historical_coverage_summary": fixture["historical_coverage_summary"],
            "old6_lineage": identities, "old6_event_comparison": events,
            "old6_descriptive_counts": counts,
            "historical_old98_aggregate": aggregate,
            "timing_illustration": timing,
            "supervision_illustration": {arm: gates.supervision_interpretation(r) for arm, r in fixture["supervision_illustration"].items()},
            "old98_rss": gates.rss_comparison(rss["original"], rss["candidate"], sampled["original"], sampled["candidate"]),
            "implemented_controls": ["frozen actual-label coverage", "lineage/exposure audit", "event delta accounting", "unique retained-event availability", "supervision/recovery interpretation"],
            "hypotheses": ["More complete positive coverage or a different training objective may help; untested"],
            "remaining_unknowns": ["acoustic quality improvement", "real ASR accuracy", "unseen-voice generalization", "word-end/service latency", "board resource behavior"],
            "candidate_adopted": False, "scientific_qualification": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent / "fixtures/public_regressions.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(json.loads(args.fixture.read_text()))
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
