#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

from iterate_domain import evaluate  # noqa: E402

POLICY = "gru-v2-calibration-failure-diagnostics-v1"
EXPECTED_MODEL_SHA256 = "8dc7d85505147fb6c9b402b2b07f6f0b047335207ca954800ac5e311c46a3c7d"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def recording_index(name: str) -> int:
    prefix = "domain-calibration-"
    if not name.startswith(prefix):
        raise ValueError(f"unexpected calibration recording name: {name}")
    value = name[len(prefix):]
    if not value.isdigit():
        raise ValueError(f"invalid calibration recording ordinal: {name}")
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
    parser.add_argument("--shared-data", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-root", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-run-id", required=True, type=int)
    parser.add_argument("--candidate-artifact-digest", required=True)
    args = parser.parse_args()

    shared_path = args.shared_data.resolve()
    shared = load_json(shared_path)
    if shared.get("policy") != "model-family-shared-development-data-v1":
        raise ValueError("shared data policy mismatch")
    if shared.get("formal_qualification_used") is not False:
        raise ValueError("shared data touched formal qualification")
    if int(shared.get("formal_seed", -1)) != 271843:
        raise ValueError("formal seed identity drifted")

    candidate_root = args.candidate_root.resolve()
    summary_path = candidate_root / "candidate-summary.json"
    manifest_path = candidate_root / "domain-loop-manifest.json"
    model = candidate_root / "best" / "model.kwm"
    pack = candidate_root / "best" / "keywords.kwk"
    for path in (summary_path, manifest_path, model, pack):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"candidate artifact member missing: {path}")

    summary = load_json(summary_path)
    manifest = load_json(manifest_path)
    if summary.get("policy") != "shadow-blind-gru-generalization-v2":
        raise ValueError("candidate summary policy mismatch")
    if summary.get("formal_qualification_used") is not False:
        raise ValueError("candidate touched formal qualification")
    if summary.get("shadow_used") is not False or summary.get("shadow_used_for_selection") is not False:
        raise ValueError("candidate used shadow evidence")
    if [int(v) for v in summary.get("reserved_untouched_shadow_seeds", [])] != list(range(951101, 951109)):
        raise ValueError("reserved GRU shadow arena drifted")
    if manifest.get("formal_qualification_used") is not False:
        raise ValueError("candidate manifest touched formal qualification")
    if manifest.get("development_qualified") is not False:
        raise ValueError("diagnostic expects failed development candidate")

    if sha256_file(model) != EXPECTED_MODEL_SHA256 or str(summary.get("model_sha256")) != EXPECTED_MODEL_SHA256:
        raise ValueError("candidate model identity drifted")
    if bool(summary.get("calibration_gate")) is not False:
        raise ValueError("candidate calibration unexpectedly passed")
    if bool(summary.get("test_gate")) is not True:
        raise ValueError("candidate test unexpectedly failed")
    if bool(summary.get("development_qualification_gate")) is not True:
        raise ValueError("candidate development qualification unexpectedly failed")
    calibration = summary["calibration"]
    if int(calibration.get("false_rejects", -1)) != 1 or int(calibration.get("false_accepts", -1)) != 0:
        raise ValueError("candidate calibration failure count drifted")
    per_keyword = calibration.get("per_keyword", {})
    if int(per_keyword.get("1", {}).get("false_rejects", -1)) != 0:
        raise ValueError("keyword 1 calibration count drifted")
    if int(per_keyword.get("2", {}).get("false_rejects", -1)) != 1:
        raise ValueError("keyword 2 calibration count drifted")

    references = pathlib.Path(str(shared["canonical_calibration_references"]))
    domain_index = references.parent / "domain-index.jsonl"
    if not references.is_file() or not domain_index.is_file():
        raise ValueError("canonical calibration corpus is missing")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base, _ = evaluate(
        runner=args.runner.resolve(),
        model=model,
        pack=pack,
        references=references,
        output=output / "eval",
    )
    reproduced = {
        "false_rejects": int(base["false_rejects"]),
        "false_accepts": int(base["false_accepts"]),
    }
    if reproduced != {"false_rejects": 1, "false_accepts": 0}:
        raise ValueError(f"calibration reproduction drifted: {reproduced}")

    failures = load_jsonl(output / "eval" / "false-rejects.jsonl")
    false_positives = load_jsonl(output / "eval" / "false-positives.jsonl")
    if len(failures) != 1 or false_positives:
        raise ValueError("expected exactly one reproduced calibration false reject")
    event = failures[0]
    name = str(event["recording"])
    ordinal = recording_index(name)

    calibration_rows = [
        row for row in load_jsonl(domain_index)
        if str(row.get("split")) == "calibration"
    ]
    refs = load_jsonl(references)
    if len(calibration_rows) != len(refs):
        raise ValueError("calibration index/reference cardinality drifted")
    if ordinal >= len(calibration_rows):
        raise ValueError(f"calibration recording ordinal out of range: {name}")
    source = calibration_rows[ordinal]
    reference = refs[ordinal]
    if str(reference["recording"]) != name:
        raise ValueError("calibration reference/index recording order drifted")
    if int(event["keyword_id"]) != 2 or int(source.get("keyword_id", -1)) != 2:
        raise ValueError("reproduced failure is not keyword 2")

    result = {
        "schema_version": 1,
        "evidence_class": "development-only-gru-v2-calibration-failure-diagnostics",
        "policy": POLICY,
        "formal_qualification_used": False,
        "formal_seed_consumed": False,
        "shadow_used": False,
        "shadow_seed_consumed": False,
        "candidate_run_id": args.candidate_run_id,
        "candidate_artifact_digest": str(args.candidate_artifact_digest),
        "candidate_model_sha256": sha256_file(model),
        "candidate_pack_sha256": sha256_file(pack),
        "candidate_summary_sha256": sha256_file(summary_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "shared_data_sha256": sha256_file(shared_path),
        "calibration_references_sha256": sha256_file(references),
        "domain_index_sha256": sha256_file(domain_index),
        "reproduced": reproduced,
        "event": {
            "event_type": "false_reject",
            "recording": name,
            "expected_keyword_id": int(event["keyword_id"]),
            "source": compact_source(source),
        },
    }
    path = output / "gru-v2-calibration-failure-diagnostics.json"
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
