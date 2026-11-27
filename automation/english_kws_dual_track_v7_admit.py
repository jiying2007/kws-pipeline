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


def build_readiness(joined, packet_identity):
    summary = joined["summary"]
    candidate_results = summary["candidate_results"]
    return {
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



def write_bytes_once(path, raw):
    path = Path(path)
    require(isinstance(raw, bytes), "SNAPSHOT_BYTES_REQUIRED")
    require(not path.exists() and not path.is_symlink(), "NEW_SNAPSHOT_FILE")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    return {"path": path.name, "bytes": len(raw), "sha256": admission.sha(raw)}


def build_training_snapshot(packet, output, candidate, manifest, source_run, packet_identity):
    """Create a candidate-only TRAIN+DEV artifact; calibration/locked bytes never enter it."""
    require(candidate in admission.CANDIDATES, "KNOWN_SNAPSHOT_CANDIDATE")
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "NEW_CANDIDATE_SNAPSHOT_DIRECTORY")
    rows = [row for row in manifest.get("rows", [])
            if row.get("role") in ("train", "development")]
    groups = [group for group in manifest.get("source_group_roles", [])
              if group.get("role") in ("train", "development")]
    role_counts = Counter(group.get("role") for group in groups)
    require(role_counts == Counter({"train": 6, "development": 2}),
            "TRAIN_DEV_ONLY_SOURCE_GROUP_COUNTS")
    require(len({group.get("source_group_id") for group in groups}) == 8
            and len({group.get("source_speaker_id") for group in groups}) == 8,
            "TRAIN_DEV_SOURCE_GROUP_IDENTITY")
    require(rows and all(row.get("role") in ("train", "development") for row in rows),
            "TRAIN_DEV_ONLY_ROWS")
    require(not any(row.get("role") in ("calibration", "locked-test") for row in rows),
            "NO_CALIBRATION_OR_LOCKED_ROWS")
    output.mkdir(mode=0o700)
    snapshot_rows = []
    file_entries = []
    total_bytes = 0
    for row in rows:
        row_id = admission.safe_relative(row.get("id"))
        require("/" not in row_id and "\\" not in row_id, "SAFE_SNAPSHOT_ROW_ID")
        pcm = admission.read_member(packet, row["pcm_path"], row["pcm_sha256"], limit=1024 * 1024)
        wav = admission.read_member(packet, row["wav_path"], row["wav_sha256"], limit=4 * 1024 * 1024)
        require(len(pcm) == row["pcm_samples"] * 2, "SNAPSHOT_PCM_GEOMETRY")
        require(admission.sha(pcm) == row["pcm_sha256"]
                and admission.sha(wav) == row["wav_sha256"], "SNAPSHOT_SOURCE_BYTE_IDENTITY")
        pcm_name = "audio/" + row_id + ".pcm"
        wav_name = "audio/" + row_id + ".wav"
        file_entries.append(write_bytes_once(output / pcm_name, pcm) | {"path": pcm_name})
        file_entries.append(write_bytes_once(output / wav_name, wav) | {"path": wav_name})
        total_bytes += len(pcm) + len(wav)
        kept = {
            key: value for key, value in row.items()
            if key not in {
                "evidence_refs", "observed_phrase", "human_gold",
                "final_native_pronunciation_verified",
                "independent_real_quality_claims_allowed",
            }
        }
        kept["pcm_path"] = pcm_name
        kept["wav_path"] = wav_name
        snapshot_rows.append(kept)
    snapshot_rows.sort(key=lambda item: item["id"])
    snapshot_manifest = {
        "schema": "english-kws-dual-track-v7-training-snapshot-manifest-v1",
        "candidate": candidate,
        "candidate_id": manifest["candidate_id"],
        "phrase": manifest["phrase"],
        "locale": manifest["locale"],
        "ipa": manifest["ipa"],
        "ctc_tokens": manifest["ctc_tokens"],
        "ctc_token_ids": manifest["ctc_token_ids"],
        "source_plan_sha256": manifest["source_plan_sha256"],
        "source_run": source_run,
        "source_groups": groups,
        "role_counts": {"train": 6, "development": 2},
        "rows": snapshot_rows,
        "training_rows": sum(row["role"] == "train" and row["training_eligible"] for row in snapshot_rows),
        "development_rows": sum(row["role"] == "development" and row["tier"] in ("silver", "weak")
                                and row["research_class"] in (0, 1) for row in snapshot_rows),
        "training_snapshot_allowed": True,
        "calibration_rows_included": False,
        "locked_test_rows_included": False,
        "calibration_labels_included": False,
        "locked_test_labels_included": False,
        "calibration_audio_included": False,
        "locked_test_audio_included": False,
        "optimizer_updates": 0,
        "kws_model_forwards": 0,
        "product_promotion_allowed": False,
    }
    manifest_raw = admission.canonical(snapshot_manifest) + b"\\n"
    write_bytes_once(output / "MANIFEST.json", manifest_raw)
    file_entries.append({
        "path": "MANIFEST.json", "bytes": len(manifest_raw),
        "sha256": admission.sha(manifest_raw),
    })
    file_entries.sort(key=lambda item: item["path"])
    inventory = {
        "schema": "english-kws-dual-track-v7-training-snapshot-files-v1",
        "candidate": candidate,
        "files": file_entries,
    }
    inventory_raw = admission.canonical(inventory) + b"\\n"
    write_bytes_once(output / "FILES.json", inventory_raw)
    snapshot = {
        "schema": "english-kws-dual-track-v7-training-snapshot-v1",
        "candidate": candidate,
        "candidate_id": manifest["candidate_id"],
        "source_run": source_run,
        "source_plan_sha256": manifest["source_plan_sha256"],
        "source_candidate_manifest_sha256": admission.sha(admission.canonical(manifest)),
        "source_packet_files_manifest_sha256": packet_identity["files_manifest_sha256"],
        "manifest_sha256": admission.sha(manifest_raw),
        "files_manifest_sha256": admission.sha(inventory_raw),
        "file_count": len(file_entries),
        "verified_bytes": total_bytes + len(manifest_raw),
        "training_rows": snapshot_manifest["training_rows"],
        "development_rows": snapshot_manifest["development_rows"],
        "role_counts": {"train": 6, "development": 2},
        "calibration_rows_included": False,
        "locked_test_rows_included": False,
        "calibration_labels_included": False,
        "locked_test_labels_included": False,
        "calibration_audio_included": False,
        "locked_test_audio_included": False,
        "locked_test_access_in_training_job": False,
        "optimizer_updates": 0,
        "kws_model_forwards": 0,
        "product_promotion_allowed": False,
    }
    write_once(output / "SNAPSHOT.json", snapshot)
    return snapshot


def build_training_snapshots(packet, output, joined, packet_identity):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "NEW_TRAINING_SNAPSHOT_ROOT")
    output.mkdir(mode=0o700, parents=True)
    result = {}
    for candidate in admission.CANDIDATES:
        manifest = joined["candidate_manifests"][candidate]
        require(manifest["training_snapshot_allowed"] is True,
                "CANDIDATE_NOT_READY_FOR_TRAINING_SNAPSHOT")
        result[candidate] = build_training_snapshot(
            packet, output / candidate, candidate, manifest,
            joined["summary"].get("source_run"), packet_identity)
    return result


def run(packet, output, snapshots_output=None):
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
    gate = build_readiness(joined, packet_identity)
    files.append(write_once(output / "G3_READINESS.json", gate))
    snapshots = None
    if snapshots_output is not None and gate["data_admitted"]:
        snapshots = build_training_snapshots(packet, snapshots_output, joined, packet_identity)
        snapshot_index = {
            "schema": "english-kws-dual-track-v7-training-snapshots-index-v1",
            "candidates": snapshots,
            "calibration_or_locked_content_included": False,
        }
        files.append(write_once(output / "TRAINING_SNAPSHOTS.json", snapshot_index))
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
        "training_snapshots_created": snapshots is not None,
        "training_snapshot_candidates": sorted(snapshots) if snapshots is not None else [],
        "new_kws_model_calls": 0,
        "optimizer_updates": 0,
    }, sort_keys=True))
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots-output", type=Path)
    args = parser.parse_args()
    run(args.packet, args.output, args.snapshots_output)


if __name__ == "__main__":
    main()
