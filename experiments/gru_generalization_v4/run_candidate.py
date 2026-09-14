#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

from iterate_domain import sha256_file  # noqa: E402

POLICY = "shadow-blind-gru-generalization-v4-targeted-confusable-replay"
EVIDENCE_CLASS = "development-only-shadow-blind-gru-generalization-v4"
FAILURE_REPLAY_POLICY = "gru-v4-development-confusable-resynthesis-v1"
FAILURE_REPLAY_CLASS = "training-only-gru-v4-development-confusable-resynthesis"
EXPECTED_REPLAY_EXAMPLES = 16
EXPECTED_REPLAY_REPEAT = 3


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_failure_replay(path: pathlib.Path) -> dict:
    evidence = load_json(path)
    if evidence.get("evidence_class") != FAILURE_REPLAY_CLASS:
        raise ValueError("unexpected GRU V4 failure replay evidence class")
    if evidence.get("policy") != FAILURE_REPLAY_POLICY:
        raise ValueError("unexpected GRU V4 failure replay policy")
    if evidence.get("formal_qualification_used") is not False:
        raise ValueError("GRU V4 replay touched formal qualification")
    if evidence.get("shadow_used") is not False:
        raise ValueError("GRU V4 replay touched reserved shadow")
    if evidence.get("source_wav_bytes_copied") is not False:
        raise ValueError("GRU V4 replay copied development evaluation WAV bytes")
    if evidence.get("development_qualification_feedback_used") is not True:
        raise ValueError("GRU V4 replay must explicitly declare development qualification feedback")
    if int(evidence.get("examples", -1)) != EXPECTED_REPLAY_EXAMPLES:
        raise ValueError("GRU V4 targeted replay example count drifted")
    if int(evidence.get("effective_exposure_with_repeat_3", -1)) != EXPECTED_REPLAY_EXAMPLES * EXPECTED_REPLAY_REPEAT:
        raise ValueError("GRU V4 targeted replay exposure drifted")
    manifest = pathlib.Path(str(evidence["manifest"]))
    if not manifest.is_file() or sha256_file(manifest) != str(evidence["manifest_sha256"]):
        raise ValueError("GRU V4 targeted replay manifest binding failed")
    return evidence


def nonempty_lines(path: pathlib.Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--shared-data", required=True, type=pathlib.Path)
    parser.add_argument("--replay-evidence", required=True, type=pathlib.Path)
    parser.add_argument("--v3-training-evidence", required=True, type=pathlib.Path)
    parser.add_argument("--v3-failure-replay-evidence", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    shared_path = args.shared_data.resolve()
    shared = load_json(shared_path)
    if shared.get("policy") != "model-family-shared-development-data-v1":
        raise ValueError("shared model-family data policy mismatch")
    if shared.get("formal_qualification_used") is not False:
        raise ValueError("shared data touched formal qualification")
    failure_replay_path = args.v3_failure_replay_evidence.resolve()
    failure_replay = validate_failure_replay(failure_replay_path)

    output = args.output.resolve()
    overlay_root = output / "v4-input"
    overlay_root.mkdir(parents=True, exist_ok=True)
    original_failure = pathlib.Path(str(shared["failure_manifest"]))
    targeted = pathlib.Path(str(failure_replay["manifest"]))
    original_lines = nonempty_lines(original_failure) if int(shared.get("failure_replay_examples", 0)) > 0 else []
    targeted_lines = nonempty_lines(targeted)
    if len(targeted_lines) != EXPECTED_REPLAY_EXAMPLES:
        raise ValueError("GRU V4 targeted replay manifest cardinality drifted")
    combined_lines = [*original_lines, *targeted_lines]
    if len(set(combined_lines)) != len(combined_lines):
        raise ValueError("GRU V4 combined replay contains duplicate manifest rows")
    combined = overlay_root / "combined-development-failure-replay.tsv"
    combined.write_text("\n".join(combined_lines) + "\n", encoding="utf-8")

    overlay = copy.deepcopy(shared)
    overlay["failure_manifest"] = str(combined.resolve())
    overlay["failure_manifest_sha256"] = sha256_file(combined)
    overlay["failure_replay_examples"] = len(combined_lines)
    overlay["v4_overlay"] = {
        "policy": POLICY,
        "development_qualification_feedback_used_for_training": True,
        "source_failure_replay_sha256": sha256_file(failure_replay_path),
        "targeted_examples": EXPECTED_REPLAY_EXAMPLES,
        "targeted_effective_exposure": EXPECTED_REPLAY_EXAMPLES * EXPECTED_REPLAY_REPEAT,
        "source_wav_bytes_copied": False,
        "formal_qualification_used": False,
        "shadow_used": False,
    }
    overlay_path = overlay_root / "shared-data-v4-overlay.json"
    overlay_path.write_text(
        json.dumps(overlay, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(ROOT / "experiments" / "gru_generalization_v3" / "run_candidate.py"),
        "--config", str(args.config.resolve()),
        "--shared-data", str(overlay_path),
        "--replay-evidence", str(args.replay_evidence.resolve()),
        "--v3-training-evidence", str(args.v3_training_evidence.resolve()),
        "--runner", str(args.runner.resolve()),
        "--output", str(output),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in (0, 1):
        return completed.returncode

    summary_path = output / "candidate-summary.json"
    manifest_path = output / "domain-loop-manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ValueError("GRU V3 base runner did not retain candidate evidence")
    summary = load_json(summary_path)
    manifest = load_json(manifest_path)
    if int(summary.get("parameters", {}).get("replay_exposure_repeat", -1)) != EXPECTED_REPLAY_REPEAT:
        raise ValueError("base runner replay exposure contract drifted")
    if summary.get("formal_qualification_used") is not False or summary.get("shadow_used") is not False:
        raise ValueError("base runner consumed forbidden evaluation evidence")

    base_policy = str(summary.get("policy"))
    summary["base_runner_policy"] = base_policy
    summary["policy"] = POLICY
    summary["development_qualification_feedback_used_for_training"] = True
    summary["development_qualification_independent"] = False
    summary["development_qualification_gate_independent"] = False
    summary["v3_failure_diagnostics_used_for_training"] = True
    summary["v3_failure_source_wav_bytes_copied"] = False
    summary["v3_failure_replay_evidence_sha256"] = sha256_file(failure_replay_path)
    summary["v3_failure_replay_manifest_sha256"] = str(failure_replay["manifest_sha256"])
    summary["parameters"]["v3_failure_replay_examples"] = EXPECTED_REPLAY_EXAMPLES
    summary["parameters"]["v3_failure_effective_exposure"] = EXPECTED_REPLAY_EXAMPLES * EXPECTED_REPLAY_REPEAT
    summary["parameters"]["v3_failure_focus_keyword_id"] = int(failure_replay["focus_keyword_id"])
    summary["independent_shadow_required_for_acceptance"] = True
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    manifest["base_runner_policy"] = str(manifest.get("policy"))
    manifest["policy"] = POLICY
    manifest["evidence_class"] = EVIDENCE_CLASS
    manifest["development_qualification_feedback_used_for_training"] = True
    manifest["development_qualification_independent"] = False
    manifest["qualification_replay_passed"] = bool(manifest.get("qualification_qualified"))
    manifest["independent_shadow_required_for_acceptance"] = True
    manifest["qualified"] = False
    manifest["formal_qualification_used"] = False
    manifest["v3_failure_replay_evidence_sha256"] = sha256_file(failure_replay_path)
    manifest["v3_failure_replay_manifest_sha256"] = str(failure_replay["manifest_sha256"])
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "policy": POLICY,
        "base_runner_exit_code": completed.returncode,
        "calibration_gate": bool(summary.get("calibration_gate")),
        "test_gate": bool(summary.get("test_gate")),
        "development_qualification_replay_gate": bool(summary.get("development_qualification_gate")),
        "independent_shadow_required_for_acceptance": True,
        "v3_failure_replay_examples": EXPECTED_REPLAY_EXAMPLES,
        "v3_failure_effective_exposure": EXPECTED_REPLAY_EXAMPLES * EXPECTED_REPLAY_REPEAT,
    }, ensure_ascii=False, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
