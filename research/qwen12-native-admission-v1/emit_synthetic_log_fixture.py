"""Public deterministic legacy log transport fixture, with no ASR dependencies.

This test does not open the real campaign's claim or read real audio/models.
Every payload byte is generated from a fixed public seed or this test's source.
Root must read the actual CI job log back through the connector and validate it
before releasing the separate real-recognition campaign.
"""
from pathlib import Path
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from log_transport import MAX_COMPRESSED, MAX_MEMBER, build_package, canonical, parse_job_log, sha

SEED = 20261011

def emit(run_identity, stream):
    # Near-incompressible deterministic synthetic bytes exercise >98% of the
    # original 8MiB ZIP cap without changing the historical log fixture route.
    cap = min(MAX_COMPRESSED, MAX_MEMBER)
    size = cap - 65536
    raw = random.Random(SEED).randbytes(size)
    with tempfile.TemporaryDirectory(prefix="kws-synthetic-log-") as directory:
        root = Path(directory)
        (root / "synthetic-payload.fixture").write_bytes(raw)
        (root / "synthetic-description.json").write_bytes(canonical({
            "schema": "synthetic-native-log-transport-fixture-v1",
            "synthetic_fixture": True, "public_seed": SEED,
            "generated_bytes": size, "generated_sha256": sha(raw),
            "model_downloads": 0, "model_loads": 0, "recognition_attempts": 0,
            "training_runs": 0, "real_batch_claim_touched": False}))
        payload, manifest, lines = build_package(root,
            ["synthetic-description.json", "synthetic-payload.fixture"], run_identity)
    if not cap * 98 // 100 <= len(payload) <= cap:
        raise ValueError("synthetic package did not exercise the admitted maximum-size range")
    parsed, parsed_manifest, receipt = parse_job_log("\n".join(lines), run_identity)
    if parsed != payload or parsed_manifest != manifest:
        raise ValueError("local fixture readback mismatch")
    print("SYNTHETIC_TRANSPORT_ONLY " + json.dumps({"archive_bytes": len(payload),
        "archive_sha256": sha(payload), "decoded_transport_bytes": sum(len(line) + 1 for line in lines),
        "zero_recognition_attempts": True, "live_connector_readback_still_required": True}, sort_keys=True), file=stream, flush=True)
    for line in lines:
        print(line, file=stream, flush=True)
    return receipt

def main():
    if len(sys.argv) != 1:
        raise ValueError("no arguments; this fixture accepts only current CI identity")
    run_identity = {"repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "job_key": os.environ["GITHUB_JOB"], "executed_head_sha": os.environ["GITHUB_SHA"]}
    emit(run_identity, sys.stdout)

if __name__ == "__main__":
    main()
