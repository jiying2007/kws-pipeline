#!/usr/bin/env python3
"""Network-free contracts for required review coverage and source-retention CI."""
from pathlib import Path
import re
import subprocess
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[3]


def jobs(text):
    body = text.split('\njobs:\n', 1)[1]
    entries = re.split(r'(?m)^  (?=[a-z][a-z0-9-]*:\s*$)', body)
    return {part.split(':', 1)[0]: part for part in entries if part.strip()}


class CIContracts(unittest.TestCase):
    def setUp(self):
        self.ci = (ROOT / '.github/workflows/ci.yml').read_text()
        self.source = (ROOT / '.github/workflows/research-source-consolidation.yml').read_text()
        self.quality = (ROOT / '.github/workflows/experiment-quality-guards.yml').read_text()

    def test_required_hosted_checks_fail_when_shared_checks_fail(self):
        job = jobs(self.ci)['hosted']
        self.assertIn('    needs: [python-contracts, research-sources]\n', job)
        self.assertIn('    if: ${{ always() }}\n', job)
        self.assertIn('CONTRACT_RESULT: ${{ needs.python-contracts.result }}', job)
        self.assertIn('test "$CONTRACT_RESULT" = success', job)
        self.assertIn('cc: [gcc, clang]', job)

    def test_independent_checks_run_once_and_runner_checks_stay_matrixed(self):
        shared = jobs(self.ci)['python-contracts']
        hosted = jobs(self.ci)['hosted']
        corpus_commands = [
            'python3 tests/test_corpus_identity.py',
            'python3 -S -B -O tests/test_corpus_identity.py',
            'python3 -S -B -OO tests/test_corpus_identity.py',
        ]
        self.assertEqual([line.strip() for line in shared.splitlines()
                          if 'tests/test_corpus_identity.py' in line], corpus_commands)
        self.assertEqual(self.ci.count('test_corpus_identity.py'), 3)
        self.assertNotIn('test_corpus_identity.py', hosted)
        for name in ('test_domain_scene.py', 'test_statistical_bounds.py',
                     'test_domain_curriculum.py', 'test_false_reject_mining.py',
                     'test_domain_metrics.py', 'test_training_diagnostics.py',
                     'test_acoustic_alignment_diagnostic.py',
                     'test_decoder_boundary_references.py',
                     'test_decoder_boundary_product_references.py',
                     'test_decoder_boundary_requalification.py',
                     'test_decoder_boundary_separability.py',
                     'test_acoustic_boundary_segmentation.py',
                     'test_recorded_verdict_booleans.py'):
            self.assertEqual(self.ci.count(name), 1)
            self.assertIn(name, shared)
            self.assertNotIn(name, hosted)
        for command in ('test_frontend_parity.py', 'test_domain_loop.py --runner',
                        'test_long_far.py --runner', 'test_board_bench.py --runner',
                        'test_parameter_contract.py',
                        'test_decoder_policy_replay.py --path-runner ./build/kws_decoder_path_replay',
                        'check_runtime_purity.py',
                        'check_reproducible_sdk.py', 'ctest --test-dir build'):
            self.assertIn(command, hosted)
        self.assertEqual(self.ci.count('test_decoder_policy_replay.py'), 1)
        self.assertNotIn('test_decoder_policy_replay.py', shared)
        benchmark = hosted.split('      - name: Real-artifact board benchmark contract\n', 1)[1].split('      - name:', 1)[0]
        self.assertIn('CC: ${{ matrix.cc }}', benchmark)
        self.assertIn('python3 -B tests/test_board_bench.py --runner ./build/kws_board_bench', benchmark)
        for mode in ('-O', '-OO'):
            self.assertIn('python3 -B ' + mode + ' tests/test_board_bench.py --statistics-only', benchmark)

    def test_only_ordinary_same_pr_ci_is_cancelled(self):
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", self.ci)
        self.assertIn("format('pr-{0}', github.event.pull_request.number)", self.ci)
        self.assertIn("format('run-{0}', github.run_id)", self.ci)
        self.assertNotIn('cancel-in-progress:', self.source)
        for name, job in jobs(self.ci).items():
            if name != 'research-sources':
                self.assertRegex(job, r'    timeout-minutes: (15|30)\n')

    def test_retention_is_local_and_required_for_every_pr(self):
        # ci must run even for README-only/unrelated PRs. A path-filtered
        # research job is not itself a safe required check.
        triggers = self.ci.split('\njobs:\n', 1)[0]
        self.assertIn('  pull_request:\n', triggers)
        for forbidden in ('paths:', 'paths-ignore:', 'branches-ignore:'):
            self.assertNotIn(forbidden, triggers)
        shared = jobs(self.ci)['python-contracts']
        self.assertNotIn('    if:', shared)
        for command in ('tools/verify_research_sources.py',
                        'source_tests/test_source_retention.py',
                        'source_tests/test_publication_index.py',
                        'source_tests/test_archive_delivery.py',
                        'source_tests/test_ci_structure.py'):
            self.assertIn(command, shared)
            expected = 3 if command.endswith(('test_source_retention.py', 'test_ci_structure.py')) else 1
            self.assertEqual(self.ci.count(command), expected)
            self.assertNotIn(command, self.source.split('\njobs:\n', 1)[1])
        for network in ('--fetch', 'pip install', 'setup-python', 'needs:',
                        'continue-on-error:'):
            self.assertNotIn(network, shared)
        self.assertNotIn('local-source-inventory:', self.source)
        expensive = jobs(self.source)['offline-source-checks']
        self.assertIn('needs: decoder-scratch', expensive)
        self.assertIn('if: ${{ always() }}', expensive)
        self.assertIn('test "$SCRATCH_RESULT" = success', expensive)
        self.assertIn('cc: [gcc, clang]', jobs(self.source)['decoder-scratch'])

    def test_required_gate_executes_and_fails_closed_for_all_other_results(self):
        hosted = jobs(self.ci)['hosted']
        first_step = hosted.split('    steps:\n', 1)[1].split('      - uses:', 1)[0]
        command = textwrap.dedent(first_step.split('        run: |\n', 1)[1])
        self.assertNotIn('continue-on-error:', hosted)
        self.assertNotIn('        if:', first_step)
        for contract in ('success', 'failure', 'cancelled', 'skipped', ''):
            for required in ('true', 'false', ''):
                for result in ('success', 'failure', 'cancelled', 'skipped', ''):
                    completed = subprocess.run(['bash', '-e', '-c', command], env={
                        'CONTRACT_RESULT': contract, 'RESEARCH_REQUIRED': required,
                        'RESEARCH_RESULT': result})
                    expected = contract == 'success' and (
                        (required == 'true' and result == 'success') or
                        (required == 'false' and result == 'skipped'))
                    self.assertEqual(completed.returncode == 0, expected,
                                     (contract, required, result))

    def test_expensive_checks_called_once_and_inherit_read_only_permissions(self):
        caller = jobs(self.ci)['research-sources']
        self.assertIn('needs: python-contracts', caller)
        self.assertIn("if: ${{ needs.python-contracts.outputs.research_required == 'true' }}", caller)
        self.assertIn('uses: ./.github/workflows/research-source-consolidation.yml', caller)
        self.assertIn('research_required: ${{ steps.research_changes.outputs.required }}', self.ci)
        self.assertIn('fetch-depth: 2', jobs(self.ci)['python-contracts'])
        self.assertIn('run: python3 -B tools/research_ci_changes.py', self.ci)
        self.assertIn('  workflow_call:', self.source)
        trigger = self.source.split('permissions:', 1)[0]
        self.assertNotIn('pull_request:', trigger)
        self.assertNotIn('main', trigger)
        self.assertIn("branches: ['consolidate/**']", trigger)
        self.assertNotIn('concurrency:', self.source)
        for workflow in (self.ci, self.source):
            self.assertIn('permissions:\n  contents: read\n', workflow)
        self.assertNotIn('continue-on-error:', self.source)

    def test_legacy_consolidation_checks_are_required_without_duplicate_pr_runs(self):
        shared = jobs(self.ci)['python-contracts']
        for command in ('tools/verify_research_consolidation.py',
                        'research/consolidation/tests/test_research_consolidation.py',
                        'research/consolidation/tests/test_cleanup_retention.py',
                        'tools/test_inventory.py --root research/consolidation/tests --workflow .github/workflows/ci.yml'):
            self.assertIn(command, shared)
            self.assertEqual(self.ci.count(command), 1)
        legacy = (ROOT / '.github/workflows/research-consolidation.yml').read_text()
        trigger = legacy.split('permissions:', 1)[0]
        self.assertNotIn('  pull_request:', trigger)
        self.assertIn('    branches: ["consolidate/**"]', trigger)
        self.assertNotRegex(trigger, r'branches:.*\bmain\b')
        self.assertNotIn('"consolidate/**"', self.ci.split('permissions:', 1)[0])
        self.assertNotIn('    if:', shared)
        self.assertNotIn('continue-on-error:', shared)

    def test_all_quality_review_files_run_in_required_chain_in_all_python_modes(self):
        shared = jobs(self.ci)['python-contracts']
        hosted = jobs(self.ci)['hosted']
        files = {path.name for path in (ROOT / 'research/experiment_quality_guards/tests').glob('test_*.py')}
        self.assertEqual(files, {'test_quality_gates.py', 'test_review_counterexamples.py',
                                'test_admission_cli.py', 'test_public_report.py',
                                'test_screen32_human_review.py'})
        for name in sorted(files):
            for mode in ('', ' -O', ' -OO'):
                command = ('python3 -S -B' + mode + ' -m unittest discover -s '
                           'research/experiment_quality_guards/tests -p ' + name + ' -v')
                self.assertEqual(shared.count(command + '\n'), 1, command)
                self.assertNotIn(command, hosted)
                self.assertEqual(self.quality.count(command + '\n'), 1, command)
            self.assertEqual(self.ci.count(name), 3)
        self.assertIn('tools/test_inventory.py --root research/experiment_quality_guards/tests '
                      '--workflow .github/workflows/ci.yml', shared)
        self.assertNotIn('    if:', shared)
        self.assertNotIn('continue-on-error:', shared)
        self.assertIn('"$RUNNER_TEMP/check-tracked-worktree.sh" "Saved review and dataset quality guards"', shared)

    def test_standalone_quality_workflow_is_explicit_manual_only(self):
        trigger = self.quality.split('permissions:', 1)[0]
        self.assertIn('on:\n  workflow_dispatch:\n', trigger)
        for event in ('push:', 'pull_request:', 'pull_request_target:', 'schedule:',
                      'workflow_run:', 'workflow_call:', 'paths:'):
            self.assertNotIn(event, trigger)
        self.assertIn('permissions:\n  contents: read\n', self.quality)
        self.assertNotIn('continue-on-error:', self.quality)
        self.assertNotIn('uses: ./.github/workflows/experiment-quality-guards.yml', self.ci)

    def test_new_identity_guards_keep_optimized_coverage(self):
        shared = jobs(self.ci)['python-contracts']
        for mode in ('', ' -O', ' -OO'):
            self.assertEqual(shared.count('python3 -S -B' + mode + ' tests/test_dataset_audit.py\n'), 1)
            for name in ('test_source_retention.py', 'test_ci_structure.py'):
                self.assertEqual(shared.count('python3 -B' + mode +
                    ' research/consolidation/source_tests/' + name + '\n'), 1)

    def test_full_screen32_freeze_suite_remains_on_canonical_path(self):
        shared = jobs(self.ci)['python-contracts']
        for mode in ('', ' -O', ' -OO'):
            self.assertIn('python3 -B' + mode + ' research/source-screen32-v1/test_contract.py\n', shared)
        # test_contract's load_tests also imports the hosted-run contract suite;
        # its saved source review must not be mistaken for a two-file freeze.
        self.assertNotIn('python3 -B research/source-screen32-v1/contract.py\n', shared)
        self.assertIn('screen32_human_review.py packet --output "$RUNNER_TEMP/screen32-historical-source-packet.json"', shared)
        contract = (ROOT / 'research/source-screen32-v1/test_contract.py').read_text()
        self.assertIn('def load_tests(', contract)
        self.assertIn('test_hosted_run', contract)

    def test_cache_key_is_pinned_and_harness_verifies_hits(self):
        self.assertIn("hashFiles('research/offline-evidence-safety-v1/PUBLIC-DEPENDENCY.json')", self.source)
        self.assertNotIn('restore-keys:', self.source)
        self.assertIn('--fetch-helper --helper-cache-dir', self.source)


if __name__ == '__main__':
    unittest.main()
