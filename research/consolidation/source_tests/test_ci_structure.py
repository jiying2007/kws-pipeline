#!/usr/bin/env python3
"""Network-free regression checks for the intentionally small CI refactor."""
from pathlib import Path
import re
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

    def test_required_hosted_checks_fail_when_shared_checks_fail(self):
        job = jobs(self.ci)['hosted']
        self.assertIn('    needs: python-contracts\n', job)
        self.assertIn('    if: ${{ always() }}\n', job)
        self.assertIn('CONTRACT_RESULT: ${{ needs.python-contracts.result }}', job)
        self.assertIn('run: test "$CONTRACT_RESULT" = success', job)
        self.assertIn('cc: [gcc, clang]', job)

    def test_independent_checks_run_once_and_runner_checks_stay_matrixed(self):
        shared = jobs(self.ci)['python-contracts']
        hosted = jobs(self.ci)['hosted']
        for name in ('test_dataset_audit.py', 'test_corpus_identity.py',
                     'test_domain_scene.py', 'test_statistical_bounds.py',
                     'test_domain_curriculum.py', 'test_false_reject_mining.py'):
            self.assertEqual(self.ci.count(name), 1)
            self.assertIn(name, shared)
            self.assertNotIn(name, hosted)
        for command in ('test_frontend_parity.py', 'test_domain_loop.py --runner',
                        'test_long_far.py --runner', 'test_board_bench.py --runner',
                        'test_parameter_contract.py', 'check_runtime_purity.py',
                        'check_reproducible_sdk.py', 'ctest --test-dir build'):
            self.assertIn(command, hosted)

    def test_only_ordinary_same_pr_ci_is_cancelled(self):
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", self.ci)
        self.assertIn("format('pr-{0}', github.event.pull_request.number)", self.ci)
        self.assertIn("format('run-{0}', github.run_id)", self.ci)
        self.assertNotIn('cancel-in-progress:', self.source)
        for job in jobs(self.ci).values():
            self.assertRegex(job, r'    timeout-minutes: (15|30)\n')

    def test_retention_is_local_and_all_workflows_trigger_it(self):
        self.assertEqual(self.source.count("      - '.github/workflows/**'"), 2)
        local = jobs(self.source)['local-source-inventory']
        self.assertIn('tools/verify_research_sources.py', local)
        for network in ('--fetch', 'pip install', 'setup-python', 'needs:'):
            self.assertNotIn(network, local)
        required = jobs(self.source)['offline-source-checks']
        self.assertIn('needs: [local-source-inventory, decoder-scratch]', required)
        self.assertIn('if: ${{ always() }}', required)
        self.assertIn('test "$INVENTORY_RESULT" = success && test "$SCRATCH_RESULT" = success', required)
        self.assertIn('cc: [gcc, clang]', jobs(self.source)['decoder-scratch'])

    def test_cache_key_is_pinned_and_harness_verifies_hits(self):
        self.assertIn("hashFiles('research/offline-evidence-safety-v1/PUBLIC-DEPENDENCY.json')", self.source)
        self.assertNotIn('restore-keys:', self.source)
        self.assertIn('--fetch-helper --helper-cache-dir', self.source)


if __name__ == '__main__':
    unittest.main()
