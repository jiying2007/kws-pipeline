#!/usr/bin/env python3
"""Saved-only analysis after acquisition; never launches a child/model or writes remotely."""
import argparse
import json
import pathlib

from paired_supervisor import load, sha, require, file_ref, regular, ROOT, write_new
from score_endpoint import score
from compare_endpoint import compare_pair

SAVED_FILE_CAP = 32 * 1024**2
ARMS = (("original_A20", "original_native_bundle"),
        ("candidate_cosy49_step300", "candidate_native_bundle"))


def bounded_load(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= SAVED_FILE_CAP,
            "saved analysis input file cap/shape")
    return load(path)


def analyze(directory, readback_receipt, output):
    # This command never repairs a failed attempt or invokes the collector.
    summary = bounded_load(directory / "pair-summary.json")
    require(summary.get("schema") == "a20-endpoint98-pair-acquisition-v1" and
            summary["status"] == "PAIR_ACQUIRED_PENDING_PRIVATE_READBACK_AND_INDEPENDENT_SAVED_AUDIT",
            "pair failed or incomplete")
    require(summary.get("results") == [
        {"arm": arm, "status": "COMPLETE_ACQUISITION_PENDING_TRACE_AUDIT"} for arm, _ in ARMS],
        "exact original then Cosy49 arm order required")
    receipt = bounded_load(readback_receipt)
    require(receipt.get("schema") == "a20-endpoint98-raw-private-readback-v1" and
            receipt.get("status") == "PASS", "exact persistence/readback receipt required")
    require(receipt.get("pair_summary_sha256") == sha(directory / "pair-summary.json"),
            "readback summary binding")
    seen = set()
    for ref in summary["files"]:
        require(ref["path"] not in seen, "duplicate saved evidence path")
        seen.add(ref["path"])
        path = regular(directory, ref["path"])
        require(type(ref["bytes"]) is int and 0 <= ref["bytes"] <= SAVED_FILE_CAP and
                path.stat().st_size == ref["bytes"] and sha(path) == ref["sha256"],
                "saved evidence drift or size cap")
    require({"release-used.json", *(arm + ".raw.jsonl" for arm, _ in ARMS)} <= seen,
            "required saved evidence missing from bound summary")
    release = bounded_load(directory / "release-used.json")
    require(release.get("schema") == "a20-endpoint98-paired-release-v1", "endpoint98 release required")
    reports = []
    for arm, key in ARMS:
        require(type(release[key]["bytes"]) is int and 0 <= release[key]["bytes"] <= SAVED_FILE_CAP,
                "native bundle metadata size cap")
        bundle = bounded_load(file_ref(release[key]))
        bindings = {"protocol_sha256": release["protocol_sha256"],
                    "model_sha256": bundle["payload"]["sha256"],
                    "library_sha256": bundle["library"]["sha256"],
                    "decoder_config_sha256": release["decoder_config_sha256"]}
        reports.append(score(directory / (arm + ".raw.jsonl"), ROOT / "metadata/manifest.json",
                             ROOT / "metadata/geometry.json", bindings))
    comparison = compare_pair(*reports)
    output.mkdir(exist_ok=False)
    # Separate saved-analysis outputs, outside the acquisition 32 MiB budget.
    for name, report in zip(("original-score.json", "candidate-score.json", "comparison.json"),
                            reports + [comparison]):
        write_new(output / name, report, SAVED_FILE_CAP)
    write_new(output / "identity.json", {
        "pair_summary_sha256": sha(directory / "pair-summary.json"),
        "readback_receipt_sha256": sha(readback_receipt),
        "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
                  for p in sorted(output.iterdir())],
        "independent_saved_audit_status": "PENDING", "qualification": False}, SAVED_FILE_CAP)
    return {"status": comparison["status"], "independent_saved_audit_status": "PENDING",
            "new_model_or_decoder_calls": 0, "next_stage_authorized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--private-readback-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.private_readback_receipt, args.output)))
