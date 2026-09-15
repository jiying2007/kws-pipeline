#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "tools" / "finalize_gru_frozen_candidate.py"
VERIFIER = ROOT / "tools" / "verify_gru_stability_evidence.py"


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GruStabilityEvidenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.finalizer = load_module(FINALIZER, "gru_stability_evidence_finalizer_contract")
        cls.verifier = load_module(VERIFIER, "gru_stability_evidence_contract")

    def _candidate(self, root: pathlib.Path, gates: list[tuple[bool, bool]]) -> pathlib.Path:
        candidate = root / "candidate"
        candidate.mkdir()
        required = 2
        round_gates = [
            {"round": index, "calibration_gate": cal, "test_gate": test}
            for index, (cal, test) in enumerate(gates)
        ]
        observed = self.verifier.terminal_strict_streak(round_gates)
        policy = {
            "policy": "gru-development-curriculum-loop-v1",
            "stable_strict_pass_rounds": required,
        }
        selection = {
            "stable_strict_pass_rounds_required": required,
            "stable_strict_pass_rounds_observed": observed,
        }
        stability = {
            "schema_version": 1,
            "policy": "gru-development-stability-evidence-v1",
            "source_policy": "gru-development-curriculum-loop-v1",
            "development_only": True,
            "stable_strict_pass_rounds_required": required,
            "stable_strict_pass_rounds_observed": observed,
            "round_gates": round_gates,
            "qualification_used": False,
            "shadow_used": False,
            "formal_qualification_used": False,
        }
        (candidate / "source-development-policy.json").write_text(
            json.dumps(policy, sort_keys=True) + "\n", encoding="utf-8"
        )
        (candidate / "selection-evidence.json").write_text(
            json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8"
        )
        stability_path = candidate / "stability-evidence.json"
        stability_path.write_text(json.dumps(stability, sort_keys=True) + "\n", encoding="utf-8")
        freeze = {
            "policy": "gru-frozen-candidate-v1",
            "stability_evidence_sha256": self.verifier.sha256_file(stability_path),
            "stable_strict_pass_rounds_required": required,
            "stable_strict_pass_rounds_observed": observed,
        }
        (candidate / "freeze-manifest.json").write_text(
            json.dumps(freeze, sort_keys=True) + "\n", encoding="utf-8"
        )
        return candidate

    def test_finalizer_and_verifier_share_stability_policy(self) -> None:
        self.assertEqual(
            self.finalizer.STABILITY_EVIDENCE_POLICY,
            self.verifier.STABILITY_POLICY,
        )

    def test_finalizer_round_gate_projection_is_recomputable(self) -> None:
        records = [
            {"round": 0, "calibration_gate": False, "test_gate": False, "score": 3.0},
            {"round": 1, "calibration_gate": True, "test_gate": True, "score": 2.0},
            {"round": 2, "calibration_gate": True, "test_gate": True, "score": 1.0},
        ]
        projected = self.finalizer.round_gate_evidence(records)
        self.assertEqual(self.verifier.terminal_strict_streak(projected), 2)
        self.assertEqual(
            projected,
            [
                {"round": 0, "calibration_gate": False, "test_gate": False},
                {"round": 1, "calibration_gate": True, "test_gate": True},
                {"round": 2, "calibration_gate": True, "test_gate": True},
            ],
        )

    def test_valid_terminal_streak_is_independently_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate = self._candidate(
                pathlib.Path(temp), [(False, False), (True, True), (True, True)]
            )
            result = self.verifier.verify(candidate)
            self.assertEqual(result["stable_strict_pass_rounds_required"], 2)
            self.assertEqual(result["stable_strict_pass_rounds_observed"], 2)

    def test_historical_non_terminal_streak_is_rejected_even_with_matching_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate = self._candidate(
                pathlib.Path(temp), [(True, True), (True, True), (False, False)]
            )
            with self.assertRaisesRegex(ValueError, "does not meet required"):
                self.verifier.verify(candidate)

    def test_non_contiguous_round_evidence_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.verifier.terminal_strict_streak(
                [
                    {"round": 0, "calibration_gate": True, "test_gate": True},
                    {"round": 2, "calibration_gate": True, "test_gate": True},
                ]
            )


if __name__ == "__main__":
    unittest.main()
