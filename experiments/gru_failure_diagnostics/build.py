#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

from iterate_domain import evaluate  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import load_config  # noqa: E402

POLICY = "gru-open-shadow-failure-lexical-diagnostics-v1"
EXPECTED_COUNTS = {
    941102: {"false_rejects": 1, "false_accepts": 0},
    941105: {"false_rejects": 1, "false_accepts": 0},
    941107: {"false_rejects": 0, "false_accepts": 2},
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def effective_config(cfg: dict, seed: int) -> dict:
    rendered = copy.deepcopy(cfg)
    rendered["seed"] = seed
    rendered.pop("qualification_holdout_seed", None)
    rendered.pop("retired_qualification_holdout_seeds", None)
    scenes = rendered["domains"]["scenes_per_example"]
    for split in ("train", "calibration", "test"):
        scenes[split] = 1
    scenes["qualification"] = 8
    return rendered


def recording_index(name: str) -> int:
    prefix = "domain-qualification-"
    if not name.startswith(prefix):
        raise ValueError(f"unexpected qualification recording name: {name}")
    value = name[len(prefix):]
    if not value.isdigit():
        raise ValueError(f"invalid qualification recording ordinal: {name}")
    return int(value)


def compact_source(row: dict) -> dict:
    scene = row.get("scene")
    if not isinstance(scene, dict):
        raise ValueError("domain row is missing scene metadata")
    return {
        "kind": str(row.get("kind")),
        "family_id": str(row.get("family_id")),
        "variant": int(row.get("variant", 0)),
        "source_keyword_id": row.get("keyword_id"),
        "tokens": [str(v) for v in row.get("tokens", [])],
        "target_ids": [int(v) for v in row.get("target_ids", [])],
        "domain_id": str(row.get("domain_id")),
        "scene_seed": int(row.get("scene_seed")),
        "scene": {
            key: scene.get(key)
            for key in (
                "distance_m",
                "distance_band",
                "azimuth_deg",
                "rt60_s",
                "snr_db",
                "noise_profile",
                "playback_sir_db",
                "room_id",
                "rir_id",
                "afe_latency_samples",
            )
            if key in scene
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--seed", action="append", type=int, required=True)
    args = parser.parse_args()

    seeds = [int(v) for v in args.seed]
    if seeds != [941102, 941105, 941107]:
        raise ValueError("diagnostic seeds must be exactly 941102,941105,941107")

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    formal = {
        int(cfg["qualification_holdout_seed"]),
        *[int(v) for v in cfg.get("retired_qualification_holdout_seeds", [])],
    }
    old_shadow = {int(v) for v in cfg["shadow_qualification"]["seeds"]}
    if set(seeds) & formal or set(seeds) & old_shadow:
        raise ValueError("diagnostic seed overlaps formal or original shadow namespace")

    work = args.work_dir.resolve()
    manifest_path = work / "domain-loop-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["records"][0]
    model = pathlib.Path(str(record["model"])).resolve()
    pack = pathlib.Path(str(record["pack"])).resolve()
    if not model.is_file() or not pack.is_file():
        raise ValueError("reproduced GRU model/keyword pack is missing")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict] = []
    all_events: list[dict] = []

    for seed in seeds:
        seed_root = output / f"seed-{seed}"
        effective_path = seed_root / "effective-config.json"
        effective_path.parent.mkdir(parents=True, exist_ok=True)
        effective_path.write_text(
            json.dumps(effective_config(cfg, seed), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dataset = seed_root / "dataset"
        render_domain_dataset(effective_path, dataset, curriculum_weights=None)
        base, _ = evaluate(
            runner=args.runner.resolve(),
            model=model,
            pack=pack,
            references=dataset / "qualification.references.jsonl",
            output=seed_root / "eval",
        )
        expected = EXPECTED_COUNTS[seed]
        actual = {
            "false_rejects": int(base["false_rejects"]),
            "false_accepts": int(base["false_accepts"]),
        }
        if actual != expected:
            raise ValueError(
                f"seed {seed} failure reproduction drifted: actual={actual} expected={expected}"
            )

        qualification_rows = [
            row for row in load_jsonl(dataset / "domain-index.jsonl")
            if str(row.get("split")) == "qualification"
        ]
        refs = load_jsonl(dataset / "qualification.references.jsonl")
        if len(qualification_rows) != len(refs):
            raise ValueError(f"seed {seed}: qualification index/reference cardinality drifted")

        events: list[dict] = []
        for event_type, path in (
            ("false_reject", seed_root / "eval" / "false-rejects.jsonl"),
            ("false_accept", seed_root / "eval" / "false-positives.jsonl"),
        ):
            for event in load_jsonl(path):
                name = str(event["recording"])
                ordinal = recording_index(name)
                if ordinal >= len(qualification_rows):
                    raise ValueError(f"seed {seed}: recording ordinal out of range: {name}")
                source = qualification_rows[ordinal]
                reference = refs[ordinal]
                if str(reference["recording"]) != name:
                    raise ValueError(f"seed {seed}: reference/index recording order drifted")
                row = {
                    "seed": seed,
                    "event_type": event_type,
                    "recording": name,
                    "detected_keyword_id": (
                        int(event["keyword_id"]) if event_type == "false_accept" else None
                    ),
                    "expected_keyword_id": (
                        int(event["keyword_id"]) if event_type == "false_reject" else None
                    ),
                    "confidence": (
                        float(event["confidence"]) if event_type == "false_accept" else None
                    ),
                    "source": compact_source(source),
                }
                events.append(row)
                all_events.append(row)

        seed_results.append(
            {
                "seed": seed,
                "false_rejects": actual["false_rejects"],
                "false_accepts": actual["false_accepts"],
                "domain_index_sha256": sha256_file(dataset / "domain-index.jsonl"),
                "references_sha256": sha256_file(dataset / "qualification.references.jsonl"),
                "events": events,
            }
        )

    result = {
        "schema_version": 1,
        "evidence_class": "development-only-open-shadow-failure-diagnostics",
        "policy": POLICY,
        "formal_qualification_used": False,
        "formal_seed_consumed": False,
        "open_development_seeds": seeds,
        "model_sha256": sha256_file(model),
        "keyword_pack_sha256": sha256_file(pack),
        "development_manifest_sha256": sha256_file(manifest_path),
        "failure_event_count": len(all_events),
        "seeds": seed_results,
    }
    if len(all_events) != 4:
        raise ValueError(f"expected exactly 4 reproduced failure events, got {len(all_events)}")
    path = output / "gru-failure-diagnostics.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
