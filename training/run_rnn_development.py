#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

import iterate_rnn_development as loop  # noqa: E402
import run_gru_development as shared  # noqa: E402

ROTATION_POLICY = "rnn-development-train-acoustic-round-rotation-v1"
NEGATIVE_STRESS_POLICY = "rnn-development-negative-stress-support-v1"


def _retain_rnn_identity(work: pathlib.Path) -> None:
    manifest_path = work / "development-loop-manifest.json"
    freeze_path = work / "frozen-candidate" / "freeze-manifest.json"
    manifest = shared.load_object(manifest_path)
    manifest["model_family"] = loop.MODEL_FAMILY
    manifest["architecture"] = loop.ARCHITECTURE
    shared.write_object(manifest_path, manifest)

    freeze = shared.load_object(freeze_path)
    freeze["model_family"] = loop.MODEL_FAMILY
    freeze["architecture"] = loop.ARCHITECTURE
    code = freeze.setdefault("training_code_sha256", {})
    if not isinstance(code, dict):
        raise ValueError("freeze training_code_sha256 must be an object")
    wrapper = pathlib.Path(__file__).resolve()
    code[wrapper.relative_to(ROOT).as_posix()] = shared.sha256_file(wrapper)
    shared.write_object(freeze_path, freeze)


def main() -> int:
    policy_path = shared._argument_path("--policy")
    work = shared._argument_path("--work-dir")
    # Keep run_gru_development.loop bound to the shared curriculum module so its
    # renderer hook patches render_domain_dataset there. iterate_rnn_development
    # delegates into that same module after swapping only trainer/export/freeze identity.
    shared.POLICY = ROTATION_POLICY
    shared.NEGATIVE_STRESS_POLICY = NEGATIVE_STRESS_POLICY
    rotations, negative_stress_rounds = shared.install_rotation(policy_path)
    code = int(loop.main())
    if code == 0:
        shared.retain_rotation_evidence(work, rotations, negative_stress_rounds)
        _retain_rnn_identity(work)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
