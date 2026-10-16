"""Shared saved-event scoring; never rank exposed-data arms as a winner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_events(references, arm_events, *, mode, pre_tolerance_s=0.0, post_tolerance_s=0.0):
    from eval.score_events import (validate_clip_references, validate_recordings,
        validate_detections, score_clip_presence, score, finite_float)
    if set(arm_events) != {"ctc", "direct"}:
        raise ValueError("exactly ctc and direct event streams are required")
    for value in (pre_tolerance_s, post_tolerance_s):
        if finite_float(value, "event tolerance") < 0:
            raise ValueError("event tolerances must be nonnegative")
    if mode == "clip-presence":
        recordings = validate_clip_references(references)
        if pre_tolerance_s or post_tolerance_s:
            raise ValueError("clip presence has no acoustic timing tolerances")
    elif mode == "event-windows":
        if any(row.get("annotation_status") != "complete" for row in references):
            raise ValueError("event scoring requires complete acoustic event annotations")
        recordings = validate_recordings(references)
    else:
        raise ValueError("choose clip-presence or event-windows explicitly")
    if not recordings:
        raise ValueError("scoring requires at least one recording")
    summaries = {}
    for arm, events in arm_events.items():
        detections = validate_detections(events, recordings)
        if mode == "clip-presence":
            summaries[arm] = score_clip_presence(recordings, detections)
        else:
            summary, false_accepts, false_rejects = score(
                recordings, detections, pre_tolerance_s, post_tolerance_s)
            summaries[arm] = dict(summary=summary, false_accepts=false_accepts,
                                  false_rejects=false_rejects)
    return {"schema": "same-reference-event-diagnostic-v1", "mode": mode,
        "evidence_scope": "exposed-development-diagnostic-only", "winner": None,
        "qualification_allowed": False, "independent_evaluation": False,
        "pre_tolerance_s": pre_tolerance_s, "post_tolerance_s": post_tolerance_s,
        "arms": summaries}


def main():
    from eval.score_events import load_jsonl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ctc-events", type=Path, required=True)
    parser.add_argument("--direct-events", type=Path, required=True)
    parser.add_argument("--mode", choices=("clip-presence", "event-windows"), required=True)
    parser.add_argument("--pre-tolerance-s", type=float, default=0.0)
    parser.add_argument("--post-tolerance-s", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_events(load_jsonl(args.references),
        {"ctc": load_jsonl(args.ctc_events), "direct": load_jsonl(args.direct_events)},
        mode=args.mode, pre_tolerance_s=args.pre_tolerance_s,
        post_tolerance_s=args.post_tolerance_s)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
