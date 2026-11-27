"""Build the two independent V7 synthetic admission manifests from one immutable G2 packet.

This stage performs byte/label verification only. It does not run TTS, ASR, KWS,
model inference, calibration, optimization or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import english_kws_dual_track_v7_silver as admission


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "REGULAR_INPUT_JSON")
    return json.loads(path.read_bytes(), object_pairs_hook=admission.unique_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def write_once(path, value):
    path = Path(path)
    raw = admission.canonical(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    return {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def run(packet, output):
    packet = Path(packet).resolve()
    output = Path(output)
    require(packet.is_dir() and not packet.is_symlink(), "IMMUTABLE_G2_PACKET_DIRECTORY")
    require(not output.exists() and output.parent.is_dir(), "NEW_ADMISSION_OUTPUT_DIRECTORY")
    corpus = read_json(packet / "CORPUS_MANIFEST.json")
    evidence = read_json(packet / "MACHINE_EVIDENCE.json")
    plan = read_json(packet / "EXECUTION_PLAN.json")
    profile = read_json(ROOT / "configs/research/english-pronunciation-profile-20261115.json")
    joined = admission.build_admission(corpus, evidence, packet, profile=profile, plan=plan)
    output.mkdir(mode=0o700)
    files = []
    summary = joined["summary"]
    files.append(write_once(output / "SUMMARY.json", summary))
    for candidate in admission.CANDIDATES:
        manifest = joined["candidate_manifests"][candidate]
        name = "HEY_" + candidate + "_V7_MANIFEST.json"
        files.append(write_once(output / name, manifest))
    gate = {
        "schema": "english-kws-dual-track-v7-g3-training-readiness-v1",
        "status": "READY_FOR_G3_BUDGET_DECLARATION" if summary["data_admitted"] else "STOP_INCOMPLETE_SOURCE_ADMISSION",
        "data_admitted": summary["data_admitted"],
        "training_authorized": False,
        "optimizer_updates": 0,
        "kws_model_forwards": 0,
        "requires_independent_candidate_budgets": True,
        "candidate_manifests": {
            candidate: {
                "sha256": joined["candidate_results"][candidate]["manifest_sha256"],
                "training_snapshot_allowed": joined["candidate_results"][candidate]["training_snapshot_allowed"],
                "eligible_train_rows": joined["candidate_results"][candidate]["eligible_train_rows"],
                "balanced_groups_by_role": joined["candidate_results"][candidate]["balanced_groups_by_role"],
            }
            for candidate in admission.CANDIDATES
        },
        "development_used_for_training": False,
        "calibration_used_for_training": False,
        "locked_test_used_for_training_or_mining": False,
        "locked_kws_audio_opened": False,
        "human_gold": False,
        "independent_real_quality_claims_allowed": False,
        "physical_ssc305_claims_allowed": False,
        "product_promotion_allowed": False,
    }
    files.append(write_once(output / "G3_READINESS.json", gate))
    inventory = {
        "schema": "english-kws-dual-track-v7-admission-files-v1",
        "packet_corpus_sha256": summary["corpus_sha256"],
        "packet_evidence_sha256": summary["evidence_sha256"],
        "files": files,
    }
    write_once(output / "FILES.json", inventory)
    print(json.dumps({
        "status": gate["status"],
        "data_admitted": gate["data_admitted"],
        "training_authorized": False,
        "candidate_manifests": gate["candidate_manifests"],
        "new_kws_model_calls": 0,
        "optimizer_updates": 0,
    }, sort_keys=True))
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.packet, args.output)


if __name__ == "__main__":
    main()
