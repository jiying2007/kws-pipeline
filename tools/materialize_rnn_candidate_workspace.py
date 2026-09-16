#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
from verify_rnn_frozen_candidate import verify  # noqa: E402


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def materialize(candidate: pathlib.Path, output: pathlib.Path) -> dict:
    candidate = candidate.resolve(); output = output.resolve()
    freeze = verify(candidate); evidence = load_object(candidate / "selection-evidence.json")
    if output.exists(): shutil.rmtree(output)
    best = output / "best"; best.mkdir(parents=True, exist_ok=True)
    for name in ("model.kwm", "model.pt", "model-provenance.json", "keywords.kwk", "keywords.tsv"):
        shutil.copy2(candidate / name, best / name)
    shutil.copy2(candidate / "source-config.json", output / "frozen-source-config.json")
    shutil.copy2(candidate / "development-wav-sha256.json", output / "frozen-development-wav-sha256.json")
    shutil.copy2(candidate / "freeze-manifest.json", output / "frozen-candidate-manifest.json")
    shutil.copy2(candidate / "selection-evidence.json", output / "frozen-selection-evidence.json")
    selected_round = int(evidence["selected_round"]); selected_frontend = str(evidence["selected_frontend"])
    record = {
        "round": selected_round, "frontend": selected_frontend, "candidate": 0,
        "score": float(evidence["selected_score"]),
        "model": str((best / "model.kwm").resolve()), "model_sha256": str(freeze["model_sha256"]),
        "checkpoint": str((best / "model.pt").resolve()),
        "provenance": str((best / "model-provenance.json").resolve()),
        "keywords": str((best / "keywords.tsv").resolve()), "pack": str((best / "keywords.kwk").resolve()),
        "calibration": evidence["calibration"], "calibration_domains": evidence["calibration_domains"],
        "test": evidence["test"], "test_domains": evidence["test_domains"],
        "calibration_gate": True, "test_gate": True, "formal_qualification_used": False,
        "source_freeze_policy": str(freeze["policy"]),
    }
    manifest = {
        "schema_version": 1, "evidence_class": "verified-frozen-rnn-candidate-workspace",
        "policy": "frozen-rnn-candidate-workspace-v1", "model_family": "rnn",
        "development_qualified": True, "records": [record],
        "candidate_selection": {
            "policy": "latest-strict-gate-passing-round", "eligible_rounds": [selected_round],
            "selected_round": selected_round, "selected_frontend": selected_frontend,
            "selected_score": float(evidence["selected_score"]),
            "qualification_used_for_selection": False, "shadow_used_for_selection": False,
            "formal_qualification_used_for_selection": False,
        },
        "frozen_candidate": {
            "policy": str(freeze["policy"]), "model_family": "rnn", "model_sha256": str(freeze["model_sha256"]),
            "selection_evidence_sha256": str(freeze["selection_evidence_sha256"]),
            "development_wav_identities_sha256": str(freeze["development_wav_identities_sha256"]),
            "candidate_stage_feedback_allowed": False,
        },
    }
    (output / "domain-loop-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidate", required=True, type=pathlib.Path); parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(); value = materialize(args.candidate, args.output)
    print(json.dumps({"materialized": True, "model_family": "rnn", "selected_round": value["candidate_selection"]["selected_round"], "model_sha256": value["frozen_candidate"]["model_sha256"]}, sort_keys=True)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr); raise SystemExit(2)
