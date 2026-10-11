#!/usr/bin/env python3
"""Stdlib-only checks for bounded numeric CI and preserved old job scopes."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import candidate_numeric_ci_changes as changes
from check_workflow_path_filters import exact_path_filters


class CandidateNumericCIContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (ROOT / changes.WORKFLOW).read_text()
        cls.numeric = (ROOT / "tests/test_candidate_encoder_numeric.py").read_text()

    def test_candidate_only_does_not_admit_training_or_data_jobs(self):
        paths = ["research/candidate_encoders/torch_models.py",
                 "research/candidate_encoders/README.md",
                 "tests/test_candidate_encoder_numeric.py",
                 "tests/test_candidate_encoder_numeric_ci.py"]
        self.assertEqual(changes.scopes(paths), (False, True))
        self.assertEqual(changes.scopes(["README.md"]), (False, False))
        self.assertEqual(changes.scopes(["research/candidate_encoders-lookalike/x.py"]), (False, False))

    def test_every_existing_filter_preserves_original_jobs_in_mixed_changes(self):
        filters = {path for _, path in exact_path_filters(ROOT / changes.WORKFLOW)}
        original = filters - changes.CANDIDATE_EXACT - {changes.CANDIDATE_GLOB}
        self.assertGreater(len(original), 40)
        for path in original:
            with self.subTest(path=path):
                self.assertTrue(changes.scopes([path])[0])
                self.assertEqual(changes.scopes([path, "research/candidate_encoders/torch_models.py"]), (True, True))
        for path in ("training/reproducibility_smoke.py", "training/reproducibility_trace.py",
                     "training/compare_reproducibility_trace.py", "training/model.py",
                     "configs/training/product-speech-like-base-v1.json"):
            self.assertIn(path, original)
        self.assertEqual(changes.scopes(["tools/candidate_numeric_ci_changes.py"]), (True, True))

    def test_scopes_fail_closed_on_unsupported_original_filters(self):
        with patch.object(changes, "exact_path_filters", return_value=[(1, "training/**")]):
            with self.assertRaises(ValueError):
                changes.scopes(["README.md"])

    def test_complete_delta_helper_and_original_push_branches(self):
        with patch.object(changes, "changed_paths", return_value=["training/model.py"]) as changed:
            payload = {"before": "a" * 40}
            for ref in changes.SUPPORTED_PUSH_REFS:
                self.assertEqual(changes.event_paths("push", payload, ref), ["training/model.py"])
                changed.assert_called_with("push", payload, "refs/heads/main")
            changes.event_paths("pull_request", payload, "refs/pull/1/merge")
            changed.assert_called_with("pull_request", payload, "refs/pull/1/merge")
        for event, ref in (("push", "refs/heads/other"), ("workflow_dispatch", "refs/heads/main")):
            with self.assertRaises(ValueError):
                changes.event_paths(event, {}, ref)

    def test_existing_job_conditions_and_fail_closed_compare(self):
        for name in ("contract", "live-base-binding", "training-reproducibility-replica"):
            self.assertIn("  " + name + ":\n    needs: select-scope\n    if: ${{ needs.select-scope.outputs.existing_required == 'true' }}\n", self.workflow)
        self.assertIn("    needs: [select-scope, training-reproducibility-replica]\n    if: ${{ always() && needs.select-scope.outputs.existing_required == 'true' }}\n", self.workflow)
        self.assertIn('replica: [a, b]', self.workflow)
        for command in ("Train unobserved parity reference", "Train deterministic fixture",
                        "Verify observation leaves actual training unchanged",
                        "Enforce numerical-class reproducibility policy"):
            self.assertIn(command, self.workflow)

    def test_numeric_step_is_bounded_and_reuses_existing_runtime(self):
        job = self.workflow.split("  objective-math:\n", 1)[1].split("  live-base-binding:\n", 1)[0]
        step = job.split("      - name: Candidate encoder random-feature numeric contracts\n", 1)[1].split("      - name:", 1)[0]
        self.assertIn("timeout-minutes: 5", step)
        self.assertIn(".venv-objective/bin/python -B tests/test_candidate_encoder_numeric.py", step)
        self.assertIn("--report build/candidate-encoder-numeric.json", step)
        self.assertIn("needs.select-scope.outputs.candidate_required == 'true'", step)
        self.assertIn("torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu", job)
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertIn(variable + ': "1"', job)
        self.assertEqual(self.workflow.count("tests/test_candidate_encoder_numeric.py \\"), 1)
        self.assertNotIn("continue-on-error", step)
        self.assertLess(job.index("      - name: Decoder-aligned objective math contract"),
                        job.index("      - name: Candidate encoder random-feature numeric contracts"))

    def test_source_has_no_optimizer_checkpoint_audio_or_download_calls(self):
        tree = ast.parse(self.numeric)
        compile(tree, "candidate numeric contract", "exec")
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = {node.module if isinstance(node, ast.ImportFrom) else alias.name
                 for node in imports for alias in (node.names if isinstance(node, ast.Import) else [None])}
        for name in names:
            self.assertFalse(name.startswith(("requests", "urllib", "torchaudio", "soundfile", "wave")), name)
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        for forbidden in ("torch.load", "torch.save", "torch.hub.load", "torch.optim.SGD", "torch.optim.Adam",
                          "optimizer.step", "subprocess.run", "subprocess.Popen"):
            self.assertNotIn(forbidden, calls)
        self.assertIn("SEED = 20261011", self.numeric)
        self.assertIn("torch.use_deterministic_algorithms(True)", self.numeric)
        self.assertNotIn("skipTest", self.numeric)
        self.assertNotIn("@unittest.skip", self.numeric)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
