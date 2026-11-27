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


def verify_packet_inventory(packet):
    inventory_path = packet / "FILES.json"
    inventory = read_json(inventory_path)
    require(isinstance(inventory.get("schema"), str)
            and inventory["schema"].startswith("english-kws-dual-track-v7")
            and isinstance(inventory.get("files"), list), "G2_PACKET_FILE_INVENTORY")
    seen = set()
    total_bytes = 0
    for item in inventory["files"]:
        require(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"},
                "G2_PACKET_FILE_ENTRY")
        name = admission.safe_relative(item["path"])
        require(name not in seen and type(item["bytes"]) is int and item["bytes"] >= 0,
                "G2_PACKET_FILE_UNIQUE_AND_SIZED")
        raw = admission.read_member(packet, name, item["sha256"], limit=4 * 1024 * 1024)
        require(len(raw) == item["bytes"], "G2_PACKET_FILE_BYTE_COUNT")
        seen.add(name)
        total_bytes += len(raw)
    required = {"CORPUS_MANIFEST.json", "MACHINE_EVIDENCE.json", "EXECUTION_PLAN.json"}
    require(required <= seen and "FILES.json" not in seen, "G2_PACKET_REQUIRED_FILES")
    raw = inventory_path.read_bytes()
    return {"files_manifest_sha256": admission.sha(raw), "file_count": len(seen),
            "verified_bytes": total_bytes}


def run(packet, output):
    packet_input = Path(packet)
    require(packet_input.is_dir() and not packet_input.is_symlink(), "IMMUTABLE_G2_PACKET_DIRECTORY")
    packet = packet_input.resolve()
    output = Path(output)
    require(not output.exists() and output.parent.is_dir(), "NEW_ADMISSION_OUTPUT_DIRECTORY")
    packet_identity = verify_packet_inventory(packet)
    corpus = read_json(packet / "CORPUS_MANIFEST.json")
    evidence = read_json(packet / "MACHINE_EVIDENCE.json")
    plan = read_json(packet / "EXECUTION_PLAN.json")
    profile = read_json(ROOT / "configs/research/english-pronunciation-profile-20261115.json")
    joined = admission.build_admission(corpus, evidence, packet, profile=profile, plan=plan)
    output.mkdir(mode=0o700)
    files = []
    summary = joined["summary"]
    candidate_results = summary["candidate_results"]
    files.append(write_once(output / "SUMMARY.json", summary))
    for candidate in admission.CANDIDATES:
        manifest = joined["candidate_manifests"][candidate]
        name = "HEY_" + candidate + "_V7_MANIFEST.json"
        files.append(write_once(output / name, manifest))
    gate = {
        "schema": "english-kws-dual-track-v7-g3-training-readiness-v1",
        "source_run": summary.get("source_run"),
        "packet_files_manifest_sha256": packet_identity["files_manifest_sha256"],
        "packet_file_count": packet_identity["file_count"],
        "packet_verified_bytes": packet_identity["verified_bytes"],
        "status": "READY_FOR_G3_BUDGET_DECLARATION" if summary["data_admitted"] else "STOP_INCOMPLETE_SOURCE_ADMISSION",
        "data_admitted": summary["data_admitted"],
        "training_authorized": False,
        "optimizer_updates": 0,
        "kws_model_forwards": 0,
        "requires_independent_candidate_budgets": True,
        "candidate_manifests": {
            candidate: {
                "sha256": candidate_results[candidate]["manifest_sha256"],
                "training_snapshot_allowed": candidate_results[candidate]["training_snapshot_allowed"],
                "eligible_train_rows": candidate_results[candidate]["eligible_train_rows"],
                "balanced_groups_by_role": candidate_results[candidate]["balanced_groups_by_role"],
                "tier_counts": candidate_results[candidate]["tier_counts"],
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
