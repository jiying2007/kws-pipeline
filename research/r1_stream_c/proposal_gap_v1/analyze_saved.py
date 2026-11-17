#!/usr/bin/env python3
"""Private read-only R1 argmax analysis: fixed baseline versus predeclared gap20.

Input identities below were already published with R1. Output contains private
source/label bindings: never commit the output directory to a public repository.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import zipfile

from proposal import WORDS, event_blank_runs, replay
from saved_numeric import read_argmax, require

HERE = Path(__file__).resolve().parent
PINS = {
    "KWS_V3_Research_20261107.zip": "c1876f4046c18045ff01f53bd27e88ce4195b2eb82e6f4977f9b2e9db3754454",
    "numeric/RECOVERY_INDEX.json": "ec88e02ccb80c1092b41a3a2a9a7153f97f2a3348ac21e3a7690ed5dcbe4efc7",
}
FROZEN_SHA = "493625580bc5bcb4f3c320aff9fe95898d8f8a23422e84f135df182d55b72504"
RUN_PREFIX = "KWS_V3_Research_20261107/experiments/fsmn-selected-dev/run/"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_file(path, maximum=128 * 1024 * 1024):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file(), "regular nonsymlink input")
    require(path.stat().st_size <= maximum, "bounded input")
    return path.read_bytes()


def collapsed(tokens):
    result, last = [], None
    for token in tokens:
        if token and token != last:
            result.append(token)
        last = token
    return result


def analyze(root, output):
    import io
    root, output = Path(root).absolute(), Path(output).absolute()
    require(not output.exists() and output.parent.is_dir() and
            not output.is_relative_to(root) and not output.is_relative_to(HERE.parents[2]) and
            not any(p.is_symlink() for p in output.parents), "new private output outside source/input tree required")
    frozen_raw = safe_file(HERE / "frozen_candidate.json")
    require(digest(frozen_raw) == FROZEN_SHA, "frozen gap20 declaration changed")
    raw = {path: safe_file(root / path) for path in PINS}
    for path, expected in PINS.items():
        require(digest(raw[path]) == expected, "pinned input changed")
    index = json.loads(raw["numeric/RECOVERY_INDEX.json"])
    locations = {k: v for k, v in index["numeric_locations"].items() if k.startswith("fsmn-tail/")}
    require(len(locations) == 134, "fixed 134 numeric observations required")
    parts = {p["name"]: p for p in index["parts"] if p["name"].startswith("fsmn-tail-")}
    require(len(parts) == 2, "two fixed numeric archives required")
    archives = {}
    for name, pin in parts.items():
        require(Path(name).name == name, "archive basename required")
        body = safe_file(root / "numeric" / name)
        require(len(body) == pin["bytes"] and digest(body) == pin["sha256"], "numeric archive pin")
        archive = zipfile.ZipFile(io.BytesIO(body))
        selected = {v["member"] for v in locations.values() if v["archive"] == name}
        require(len(selected) == pin["numeric_conditions"] and len(archive.namelist()) == len(selected) + 1 and
                set(archive.namelist()) == selected | {"MANIFEST.json"}, "closed numeric archive")
        archives[name] = archive
    with zipfile.ZipFile(io.BytesIO(raw["KWS_V3_Research_20261107.zip"])) as package:
        observation_raw = package.read(RUN_PREFIX + "observations.jsonl")
        manifest = json.loads(package.read(RUN_PREFIX + "MANIFEST.json"))
        require(digest(observation_raw) == manifest["observations.jsonl"]["sha256"], "observations manifest")
        observations = [json.loads(line) for line in observation_raw.splitlines()]
    require(len(observations) == 134 and len({r["source_id"] for r in observations}) == 134, "fixed distinct roster")
    output.mkdir(mode=0o700)
    rows, counts, event_gaps = [], Counter(), []
    roles = Counter()
    for row in observations:
        numeric = row["numeric_artifact"]
        loc = locations["fsmn-tail/" + numeric["path"]]
        require(loc["member"] == numeric["path"] and loc["sha256"] == numeric["sha256"] and
                loc["bytes"] == numeric["bytes"], "observation/numeric location binding")
        body = archives[loc["archive"]].read(loc["member"])
        require(len(body) == loc["bytes"] and digest(body) == loc["sha256"], "numeric member pin")
        saved = read_argmax(body)
        require(all(saved[k] == row[k] for k in ("source_id", "feature_sha256", "logits_sha256")), "numeric observation identity")
        tokens = saved["tokens"]
        require(collapsed(tokens) == row["greedy_token_ids"], "saved greedy sequence binding")
        baseline, candidate = replay(tokens, 0), replay(tokens, 20)
        known = all(k["known"] for k in row["keywords"])
        role = row["split"] if known else "unknown"
        roles[role] += 1
        counts["model_frames"] += len(tokens)
        counts["identical_event_lists"] += baseline["events"] == candidate["events"]
        counts["baseline_events"] += len(baseline["events"])
        counts["candidate_events"] += len(candidate["events"])
        counts["prefix_expirations"] += candidate["prefix_expirations"]
        stats = []
        for keyword in row["keywords"]:
            k = keyword["keyword_id"]
            require(keyword["token_ids"] == list(WORDS[k - 1]), "fixed keyword IDs")
            old = any(e["keyword_id"] == k for e in baseline["events"])
            new = any(e["keyword_id"] == k for e in candidate["events"])
            stats.append({"keyword_id": k, "known": keyword["known"], "reference": keyword["reference"],
                          "baseline_hit": old, "candidate_hit": new})
            if keyword["known"]:
                label = "positive" if keyword["reference"] == 1 else "weak_negative"
                prefix = f"{role}/k{k}/{label}"
                counts[prefix + "/rows"] += 1
                counts[prefix + "/baseline_hit_rows"] += old
                counts[prefix + "/candidate_hit_rows"] += new
        gaps = [{"keyword_id": e["keyword_id"], "max_internal_blank_frames": event_blank_runs(tokens, e),
                 "span_frames": e["token_frames"][-1] - e["token_frames"][0]} for e in baseline["events"]]
        event_gaps.extend(gaps)
        rows.append({"source_id": row["source_id"], "role": role, "pcm_sha256": row["pcm_sha256"],
                     "label_binding_sha256": row["label_binding_sha256"], "numeric": loc,
                     "tokens": tokens, "baseline": baseline, "gap20": candidate,
                     "keywords": stats, "baseline_event_gaps": gaps})
    require(roles == Counter({"train": 96, "development": 34, "unknown": 4}), "fixed original source roles")
    require(counts["model_frames"] == 12534, "fixed unique saved output frames")
    for archive in archives.values():
        archive.close()
    summary = {"schema": "r1-gap20-saved-output-comparison-v1", "status": "COMPLETED_SAVED_ARGMAX_POSTPROCESSING",
               "frozen_candidate_sha256": FROZEN_SHA, "input_pins": PINS, "numeric_archive_pins": list(parts.values()),
               "roles": dict(roles), "counts": dict(sorted(counts.items())),
               "baseline_event_gap_max_frames": max((e["max_internal_blank_frames"] for e in event_gaps), default=None),
               "baseline_event_gap_histogram": dict(sorted(Counter(str(e["max_internal_blank_frames"]) for e in event_gaps).items())),
               "new_acoustic_model_calls": 0, "training_runs": 0, "new_audio_calls": 0, "new_asr_calls": 0,
               "source_record_resets_preserved": True, "constructed_continuous_audio": False,
               "natural_far_frr_or_word_end_latency": "NOT_ESTABLISHED", "production_promoted": False,
               "private_per_record_output": True,
               "analysis_source_sha256": {p.name: digest(p.read_bytes()) for p in
                   (HERE / "proposal.py", HERE / "saved_numeric.py", HERE / "analyze_saved.py")}}
    for name, value in (("private-records.json", rows), ("aggregate.json", summary)):
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.output), sort_keys=True))
