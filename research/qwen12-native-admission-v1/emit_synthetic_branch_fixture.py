"""Fixed public-seed near-cap ZIP fixture. Zero model/audio/ASR or publication."""
from pathlib import Path
import os
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from log_transport import MAX_COMPRESSED, build_package, canonical, require, sha
from branch_transport import bound_identity, write_package

SEED = 20261011


def build_synthetic_package(run_identity):
    bound_identity(run_identity, "synthetic")
    size = MAX_COMPRESSED - 65536
    raw = random.Random(SEED).randbytes(size)
    with tempfile.TemporaryDirectory(prefix="kws-synthetic-branch-") as directory:
        root = Path(directory)
        (root / "synthetic-payload.fixture").write_bytes(raw)
        (root / "synthetic-description.json").write_bytes(canonical({
            "schema": "synthetic-native-log-transport-fixture-v1",
            "synthetic_fixture": True, "public_seed": SEED,
            "generated_bytes": size, "generated_sha256": sha(raw),
            "model_downloads": 0, "model_loads": 0, "recognition_attempts": 0,
            "training_runs": 0, "real_batch_claim_touched": False}))
        payload, manifest, _ = build_package(root,
            ["synthetic-description.json", "synthetic-payload.fixture"], run_identity)
    require(MAX_COMPRESSED * 98 // 100 <= len(payload) <= MAX_COMPRESSED,
            "synthetic package outside admitted near-cap range")
    return payload, manifest


def verify_synthetic(payload, run_identity):
    expected, _ = build_synthetic_package(run_identity)
    require(payload == expected, "synthetic package must be exact public-seed fixture")


def main():
    require(len(sys.argv) == 1, "no arguments")
    run_identity = {"repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "job_key": os.environ["GITHUB_JOB"], "executed_head_sha": os.environ["GITHUB_SHA"]}
    payload, _ = build_synthetic_package(run_identity)
    write_package(os.environ["RUNNER_TEMP"], payload, run_identity, "synthetic")
    print("SYNTHETIC_TRANSPORT_ONLY " + canonical({"archive_bytes": len(payload),
        "archive_sha256": sha(payload), "zero_recognition_attempts": True,
        "public_seed": SEED, "live_connector_readback_still_required": True}).decode("ascii"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical({"status": "synthetic_package_failed", "error_type": type(error).__name__}).decode("ascii"))
        raise SystemExit(2)
