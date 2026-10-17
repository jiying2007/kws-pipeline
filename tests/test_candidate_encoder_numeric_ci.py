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

    def test_direct_only_and_shared_scorer_scopes(self):
        for path in ("research/direct_kws/model.py", "research/direct_kws/events.py",
                     "research/direct_kws/README.md", "tests/test_direct_kws_numeric.py"):
            self.assertEqual(changes.scopes([path]), (False, True))
            self.assertEqual(changes.scopes([path, "training/train_ctc.py"]), (True, True))
        self.assertEqual(changes.scopes(["research/direct_kws-lookalike/x.py"]), (False, False))
        for path in ("eval/score_events.py", "eval/context_policy.py"):
            self.assertEqual(changes.scopes([path]), (True, True))
        filters = [path for _, path in exact_path_filters(ROOT / changes.WORKFLOW)]
        for path in (*changes.CANDIDATE_GLOBS, "tests/test_direct_kws_numeric.py"):
            self.assertEqual(filters.count(path), 2)

    def test_direct_fit_scopes_and_bounded_fixture(self):
        for path in ("research/direct_fit/runner.py", "research/direct_fit/registry_data.py",
                     "tests/test_direct_fit_numeric.py", "research/kws-candidate-registry-v1/candidates.json",
                     "research/kws-candidate-registry-v1/build_registry.py", "research/kws-candidate-registry-v1/input-pins.json"):
            self.assertEqual(changes.scopes([path]), (False, True))
        for path in ("training/export_model.py", "training/training_state.py", "training/training_admission.py",
                     "training/frontend.py", "tools/asr_source_resolver.py"):
            self.assertEqual(changes.scopes([path]), (True, True))
        self.assertEqual(changes.scopes(["research/direct_fit-lookalike/x.py"]), (False, False))
        filters = [path for _, path in exact_path_filters(ROOT / changes.WORKFLOW)]
        for path in ("research/direct_fit/**", "tests/test_direct_fit_numeric.py",
                     "research/kws-candidate-registry-v1/candidates.json",
                     "research/kws-candidate-registry-v1/build_registry.py", "research/kws-candidate-registry-v1/input-pins.json"):
            self.assertEqual(filters.count(path), 2)
        job = self.workflow.split("  objective-math:\n", 1)[1].split("  live-base-binding:\n", 1)[0]
        step = job.split("      - name: Direct fit checkpoint and registry software fixtures\n", 1)[1].split("      - name:", 1)[0]
        self.assertIn("timeout-minutes: 5", step)
        self.assertIn("needs.select-scope.outputs.candidate_required == 'true'", step)
        self.assertIn(".venv-objective/bin/python -B tests/test_direct_fit_numeric.py", step)
        self.assertIn("--report build/direct-fit-fixture.json", step)
        self.assertNotIn("continue-on-error", step)
        artifact = job.split("      - name: Retain direct fit software fixture receipt\n", 1)[1]
        self.assertIn("path: build/direct-fit-fixture.json", artifact)
        self.assertIn("if-no-files-found: error", artifact)
        self.assertNotIn("path: build/\n", artifact)

    def test_direct_fit_delivery_scopes_and_fixture(self):
        for path in ("research/direct_fit/result_delivery.py", "research/direct_fit/README-delivery.md",
                     "tests/test_direct_fit_delivery.py"):
            self.assertEqual(changes.scopes([path]), (False, True))
            self.assertEqual(changes.scopes([path, "training/train_ctc.py"]), (True, True))
        filters = [path for _, path in exact_path_filters(ROOT / changes.WORKFLOW)]
        self.assertEqual(filters.count("tests/test_direct_fit_delivery.py"), 2)
        job = self.workflow.split("  objective-math:\n", 1)[1].split("  live-base-binding:\n", 1)[0]
        step = job.split("      - name: Direct fit bounded delivery software fixtures\n", 1)[1].split("      - name:", 1)[0]
        self.assertIn("timeout-minutes: 2", step)
        self.assertIn("needs.select-scope.outputs.candidate_required == 'true'", step)
        self.assertIn("python3 -B tests/test_direct_fit_delivery.py", step)
        self.assertIn("--report build/direct-fit-delivery-fixture.json", step)
        self.assertNotIn("continue-on-error", step)
        artifact = job.split("      - name: Retain direct fit delivery software fixture receipt\n", 1)[1]
        self.assertIn("path: build/direct-fit-delivery-fixture.json", artifact)
        self.assertIn("if-no-files-found: error", artifact)
        self.assertNotIn("path: build/\n", artifact)

    def test_direct_fixture_step_and_receipt_are_bounded(self):
        job = self.workflow.split("  objective-math:\n", 1)[1].split("  live-base-binding:\n", 1)[0]
        step = job.split("      - name: Direct KWS three-class software fixture contracts\n", 1)[1].split("      - name:", 1)[0]
        self.assertIn("timeout-minutes: 5", step)
        self.assertIn("needs.select-scope.outputs.candidate_required == 'true'", step)
        self.assertIn(".venv-objective/bin/python -B tests/test_direct_kws_numeric.py", step)
        self.assertIn("--report build/direct-kws-fixture.json", step)
        self.assertNotIn("continue-on-error", step)
        artifact = job.split("      - name: Retain direct KWS software fixture receipt\n", 1)[1]
        self.assertIn("always() && needs.select-scope.outputs.candidate_required == 'true'", artifact)
        self.assertIn("path: build/direct-kws-fixture.json", artifact)
        self.assertIn("if-no-files-found: error", artifact)
        self.assertNotIn("path: build/\n", artifact)

    def test_every_existing_filter_preserves_original_jobs_in_mixed_changes(self):
        filters = {path for _, path in exact_path_filters(ROOT / changes.WORKFLOW)}
        original = filters - changes.CANDIDATE_EXACT - changes.CANDIDATE_GLOBS
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

    def test_continuous_intake_complete_exact_dependency_closure(self):
        # The actual CI suite reads every package member through its manifest,
        # plus the separately archived ASR results/independent audit.
        prefixes = ("research/alimeeting-continuous-intake-v1",
                    "research/tts-pilot-screening-v1/evidence/asr-run-38573117922")
        dependencies = {path.relative_to(ROOT).as_posix()
                        for prefix in prefixes for path in (ROOT / prefix).rglob("*")
                        if path.is_file() and "__pycache__" not in path.parts
                        and path.suffix != ".pyc"}
        self.assertGreaterEqual(len(dependencies), 204)
        filters = [path for _, path in exact_path_filters(ROOT / changes.WORKFLOW)]
        for dependency in sorted(dependencies):
            with self.subTest(dependency=dependency):
                self.assertEqual(filters.count(dependency), 2,
                                 "complete exact coverage in both push and PR filters")
                self.assertEqual(changes.scopes([dependency]), (True, False))
                self.assertEqual(changes.scopes([dependency,
                                 "research/candidate_encoders/torch_models.py"]), (True, True))

    def test_actual_product_scopes_entrypoint_accepts_current_workflow(self):
        self.assertEqual(changes.scopes([]), (False, False))
        self.assertEqual(changes.scopes([
            "research/alimeeting-continuous-intake-v1/test_suite.py",
            "research/tts-pilot-screening-v1/evidence/asr-run-38573117922/report.json"
        ]), (True, False))

    def test_non_candidate_glob_still_rejected_with_complete_candidate_filters(self):
        filters = exact_path_filters(ROOT / changes.WORKFLOW)
        with patch.object(changes, "exact_path_filters",
                          return_value=filters + [(99999, "research/alimeeting-continuous-intake-v1/**")]):
            with self.assertRaisesRegex(ValueError, "existing product filters must be nonempty exact paths"):
                changes.scopes(["research/alimeeting-continuous-intake-v1/test_suite.py"])

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

    def test_research_smoke_dependency_paths_remain_triggered(self):
        # Read code, never import Torch or execute the research runner. Follow
        # top-level local imports from every source bound into its checkpoints.
        import fnmatch
        from types import SimpleNamespace
        runner = ROOT / "research/model_comparison/runner.py"
        tree = ast.parse(runner.read_text())
        bound = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "SOURCE_PATHS"
                             for t in node.targets))
        required = set(ast.literal_eval(bound)) | {
            "research/model_comparison/runner.py", "research/model_comparison/compare.py",
            "research/model_comparison/pilot.example.json",
            "research/model_comparison/keywords.example.json",
            "tests/test_model_comparison_numeric.py", "tools/kws_vocab.py",
            "keywords/tokens.example.txt", "eval/score_events.py", "eval/context_policy.py",
        }
        pending, visited = list(required), set()
        def imports(nodes):
            for node in nodes:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    yield node
                elif isinstance(node, (ast.If, ast.Try, ast.With)):
                    for field in ("body", "orelse", "finalbody"):
                        yield from imports(getattr(node, field, []))
                    for handler in getattr(node, "handlers", []):
                        yield from imports(handler.body)
        while pending:
            relative = pending.pop()
            if relative in visited or not relative.endswith(".py"):
                continue
            visited.add(relative)
            source = ROOT / relative
            self.assertTrue(source.is_file(), relative)
            for node in imports(ast.parse(source.read_text()).body):
                modules = ([node.module] if isinstance(node, ast.ImportFrom)
                           else [alias.name for alias in node.names])
                for module in modules:
                    if not module:
                        continue
                    suffix = Path(*module.split(".")).with_suffix(".py")
                    if isinstance(node, ast.ImportFrom) and node.level:
                        bases = [source.parents[node.level - 1]]
                    else:
                        bases = [source.parent, ROOT / "training", ROOT / "tools",
                                 ROOT / "eval", ROOT]
                    resolved = next((base / suffix for base in bases
                                     if (base / suffix).is_file()), None)
                    if resolved is not None:
                        dependency = resolved.relative_to(ROOT).as_posix()
                        required.add(dependency)
                        pending.append(dependency)
        # Inspect each trigger separately: a union can hide a PR-only/push-only omission.
        for event in ("push", "pull_request"):
            lines, active = [], False
            for line in self.workflow.splitlines():
                if line == "  " + event + ":":
                    active = True
                elif active and line.strip() and len(line) - len(line.lstrip()) <= 2:
                    break
                if active:
                    lines.append(line)
            section = "\n".join(lines)
            filters = {value for _, value in exact_path_filters(
                SimpleNamespace(read_text=lambda **kwargs: section))}
            self.assertTrue(filters, event)
            for path in sorted(required):
                with self.subTest(event=event, dependency=path):
                    self.assertTrue(any(fnmatch.fnmatchcase(path, pattern) for pattern in filters), path)
                    self.assertTrue(any(changes.scopes([path])), path)
        job = self.workflow.split("  objective-math:\n", 1)[1].split("  live-base-binding:\n", 1)[0]
        step = job.split("      - name: Research model train and streaming numeric smoke\n", 1)[1].split("      - name:", 1)[0]
        self.assertNotIn("if:", step)
        self.assertNotIn("continue-on-error", step)
        self.assertIn("timeout 120s .venv-objective/bin/python -B tests/test_model_comparison_numeric.py", step)
        self.assertIn("timeout-minutes: 3", step)
        artifact = job.split("      - name: Retain research model numeric smoke receipt\n", 1)[1].split("      - name:", 1)[0]
        self.assertIn("path: build/model-comparison-numeric-receipt.json", artifact)
        self.assertNotIn("path: build/\n", artifact)

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
