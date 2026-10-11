"""Fixed public-seed near-cap ZIP fixture. Zero model/audio/ASR or publication."""
from pathlib import Path
import os
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from log_transport import MAX_COMPRESSED, MAX_MEMBER, build_package, canonical, require, sha
from branch_transport import bound_identity, write_package

SEED = 20261011


def build_synthetic_package(run_identity, purpose="synthetic"):
    require(purpose in ("synthetic", "synthetic12m"), "fixed synthetic fixture purpose")
    bound_identity(run_identity, purpose)
    cap = MAX_COMPRESSED if purpose == "synthetic12m" else 8 * 1024 * 1024
    size = cap - 65536
    raw = random.Random(SEED).randbytes(size)
    with tempfile.TemporaryDirectory(prefix="kws-synthetic-branch-") as directory:
        root = Path(directory)
        members = (["synthetic-payload.fixture"] if purpose == "synthetic" else
            [f"synthetic-payload-{i:04d}.fixture" for i in range((size + MAX_MEMBER - 1) // MAX_MEMBER)])
        for i, name in enumerate(members):
            (root / name).write_bytes(raw[i * MAX_MEMBER:(i + 1) * MAX_MEMBER])
        (root / "synthetic-description.json").write_bytes(canonical({
            "schema": ("synthetic-native-log-transport-fixture-v1" if purpose == "synthetic" else
                       "synthetic-native-branch-transport-12m-fixture-v1"),
            "synthetic_fixture": True, "public_seed": SEED,
            "generated_bytes": size, "generated_sha256": sha(raw),
            "model_downloads": 0, "model_loads": 0, "recognition_attempts": 0,
            "training_runs": 0, "real_batch_claim_touched": False}))
        payload, manifest, _ = build_package(root,
            ["synthetic-description.json", *members], run_identity)
    require(cap * 98 // 100 <= len(payload) <= cap,
            "synthetic package outside admitted near-cap range")
    return payload, manifest


def verify_synthetic(payload, run_identity, purpose="synthetic"):
    expected, _ = build_synthetic_package(run_identity, purpose)
    require(payload == expected, "synthetic package must be exact public-seed fixture")


def main():
    require(len(sys.argv) == 1, "no arguments")
    run_identity = {"repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "job_key": os.environ["GITHUB_JOB"], "executed_head_sha": os.environ["GITHUB_SHA"]}
    payload, _ = build_synthetic_package(run_identity, "synthetic12m")
    write_package(os.environ["RUNNER_TEMP"], payload, run_identity, "synthetic12m")
    print("SYNTHETIC_TRANSPORT_ONLY " + canonical({"archive_bytes": len(payload),
        "archive_sha256": sha(payload), "zero_recognition_attempts": True,
        "public_seed": SEED, "live_connector_readback_still_required": True}).decode("ascii"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical({"status": "synthetic_package_failed", "error_type": type(error).__name__}).decode("ascii"))
        raise SystemExit(2)
