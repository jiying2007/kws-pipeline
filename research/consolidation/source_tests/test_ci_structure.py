#!/usr/bin/env python3
"""Network-free contracts for required review coverage and source-retention CI."""
from pathlib import Path
import ast
import os
import tempfile
import re
import subprocess
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[3]


def jobs(text):
    body = text.split('\njobs:\n', 1)[1]
    entries = re.split(r'(?m)^  (?=[a-z][a-z0-9_-]*:\s*$)', body)
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



ASR18_MODULES = (
    'test_asr18_binding', 'test_asr18_contract', 'test_asr18_entrypoints',
    'test_asr18_freeze', 'test_asr18_failure_observer', 'test_publisher_observation', 'test_asr18_host', 'test_asr18_metadata',
    'test_asr18_raw_data_reader', 'test_asr18_receipts', 'test_asr18_resources',
    'test_asr18_transport', 'test_asr18_worker',
)
ASR18_JOB_IF = "${{ github.repository == 'jiying2007/kws-pipeline' && github.event_name == 'push' && github.ref == 'refs/heads/research/k1-punctuation6-asr18-r1' && github.event.created == true && github.event.deleted == false && github.event.repository.private == false && github.run_attempt == 1 }}"
ASR18_HOST_IF = "${{ success() && steps.asr18_contracts.outcome == 'success' }}"
ASR18_PUBLISH_IF = "${{ always() && steps.host.outputs.started == 'true' }}"



def asr18_job_route(condition, context):
    """Evaluate only the fixed equality grammar in the actual job guard."""
    if not condition.startswith('${{ ') or not condition.endswith(' }}'):
        raise ValueError('job condition delimiters')
    results = []
    for clause in condition[4:-3].split(' && '):
        match = re.fullmatch(r"(github\.[a-z_.]+) == ('[^']*'|true|false|[1-9][0-9]*)", clause)
        if not match or match[1] not in context:
            raise ValueError('unsupported or missing job context')
        literal = match[2]
        expected = (literal == 'true' if literal in ('true', 'false') else
                    ast.literal_eval(literal))
        actual = context[match[1]]
        results.append(type(actual) is type(expected) and actual == expected)
    return all(results)


def named_step(text, name):
    matches = [part for part in text.split('      - ') if part.startswith('name: ' + name + '\n')]
    if len(matches) != 1:
        raise ValueError('missing or duplicate exact named step: ' + name)
    return matches[0]


def step_if(step):
    return re.search(r'^        if: (.+)$', step, re.M).group(1)


def contract_command(step):
    return textwrap.dedent(step.split('        run: |\n', 1)[1])


def outcome_route(condition, *, success=False, host='', contracts='', started=''):
    """Only the fixed comparison grammar used by the actual two step guards."""
    if not condition.startswith('${{ ') or not condition.endswith(' }}'):
        raise ValueError('condition delimiters')
    values = {'host': host, 'asr18_contracts': contracts}
    results = []
    for clause in condition[4:-3].split(' && '):
        if clause == 'always()':
            results.append(True)
        elif clause == 'success()':
            results.append(success)
        elif clause == "steps.host.outputs.started == 'true'":
            results.append(started == 'true')
        else:
            match = re.fullmatch(r"steps\.(host|asr18_contracts)\.outcome (==|!=) ('[^']*')", clause)
            if not match:
                raise ValueError('unsupported outcome clause')
            value = ast.literal_eval(match[3])
            results.append(values[match[1]] == value if match[2] == '==' else values[match[1]] != value)
    return all(results)


class ASR18CIContracts(unittest.TestCase):
    def setUp(self):
        self.ci = (ROOT / '.github/workflows/ci.yml').read_text()
        self.workflow = (ROOT / '.github/workflows/k1-punctuation6-asr18-v1.yml').read_text()
        self.template = (ROOT / 'research/k1-punctuation6-asr18-v1/workflow.yml').read_text()

    def expected_command(self):
        program = ('import sys,unittest; from pathlib import Path; '
                   'sys.path.insert(0,str(Path("research/k1-punctuation6-asr18-v1").resolve())); '
                   'suite=unittest.defaultTestLoader.loadTestsFromNames('
                   + repr(ASR18_MODULES).replace("'", '"') + '); '
                   'result=unittest.TextTestRunner(verbosity=2).run(suite); '
                   'raise SystemExit(not result.wasSuccessful())')
        return ("set -euo pipefail\nasr18_tests='" + program + "'\n" + ''.join(
            'python3 -I -S -B' + mode + ' -c "$asr18_tests"\n' for mode in ('', ' -O', ' -OO')))

    def check_wiring(self, ci, workflow, template):
        self.assertEqual(workflow, template)
        self.assertNotIn('INERT RESEARCH TEMPLATE', workflow)
        self.assertEqual(workflow.split('permissions:', 1)[0],
            'name: K1 punctuation6 ASR18 r1 once\n'
            'on:\n  push:\n    branches:\n      - research/k1-punctuation6-asr18-r1\n')
        self.assertNotIn("refs/heads/research/k1-punctuation6-asr18-v1", workflow)
        shared = jobs(ci)['python-contracts']
        self.assertNotIn('    if:', shared)
        self.assertEqual(set(jobs(workflow)), {'asr18'})
        execution = jobs(workflow)['asr18']
        self.assertIn('    if: ' + ASR18_JOB_IF + '\n', execution)
        self.assertIn('concurrency:\n  group: k1-punctuation6-asr18-r1\n  cancel-in-progress: false\n', workflow)
        for name, body in (('ci', shared), ('asr18', execution)):
            step = named_step(body, 'K1 punctuation6 ASR18 offline source contracts')
            self.assertIn('        id: asr18_contracts\n', step)
            self.assertNotIn('        if:', step)
            self.assertNotIn('continue-on-error:', step)
            self.assertNotIn('        env:', step)
            self.assertEqual(contract_command(step), self.expected_command(), name)
        host = named_step(execution, 'Admit complete new input and run fixed ASR18 once')
        publish = named_step(execution, 'Publish complete audio-free bounded evidence')
        self.assertIn('        id: host\n', host)
        self.assertEqual(step_if(host), ASR18_HOST_IF)
        self.assertEqual(step_if(publish), ASR18_PUBLISH_IF)
        self.assertNotIn('continue-on-error:', execution)
        self.assertLess(execution.index('id: asr18_contracts'), execution.index('id: host'))
        self.assertLess(execution.index('id: host'), execution.index('name: Publish complete'))
        self.assertEqual(contract_command(host), "printf 'started=true\\n' >> \"$GITHUB_OUTPUT\"\n"
                         'python3 -I -B research/k1-punctuation6-asr18-v1/hosted_run.py\n')
        self.assertIn('python3 -I -B research/k1-punctuation6-asr18-v1/evidence_publisher.py\n', publish)
        self.assertIn('"$RUNNER_TEMP/check-tracked-worktree.sh" "K1 punctuation6 ASR18 contracts"', shared)

    def test_fixed_all_modules_run_in_required_ci_and_before_execution(self):
        self.check_wiring(self.ci, self.workflow, self.template)
        actual = {p.stem for p in (ROOT / 'research/k1-punctuation6-asr18-v1').iterdir()
                  if p.is_file() and (p.name.startswith('test_asr18_') and p.suffix == '.py'
                                     or p.name == 'test_publisher_observation.py')}
        self.assertEqual(actual, set(ASR18_MODULES))
        self.assertEqual(len(ASR18_MODULES), 13)
        for name in ASR18_MODULES:
            self.assertTrue((ROOT / 'research/k1-punctuation6-asr18-v1' / (name + '.py')).is_file())

    def test_rejects_missing_modes_discovery_skips_and_bypassed_routes(self):
        mutations = (
            ('ci', 'python3 -I -S -B -OO -c "$asr18_tests"', 'true'),
            ('ci', 'loadTestsFromNames', 'discover'),
            ('ci', 'test_asr18_worker', 'test_asr18_not_reviewed'),
            ('ci', '        id: asr18_contracts\n', '        id: asr18_contracts\n        if: false\n'),
            ('workflow', ASR18_HOST_IF, '${{ always() }}'),
            ('workflow', ASR18_PUBLISH_IF, "${{ always() && steps.host.outcome != 'skipped' }}"),
            ('workflow', "github.run_attempt == 1", 'github.run_attempt >= 1'),
            ('workflow', '      - research/k1-punctuation6-asr18-r1\n',
             '      - research/k1-punctuation6-asr18-v1\n'),
            ('workflow', '      - research/k1-punctuation6-asr18-r1\n',
             '      - research/k1-punctuation6-asr18-r1\n      - research/k1-punctuation6-asr18-v1\n'),
            ('workflow', "github.ref == 'refs/heads/research/k1-punctuation6-asr18-r1'",
             "github.ref == 'refs/heads/research/k1-punctuation6-asr18-v1'"),
            ('workflow', 'github.event.created == true', 'github.event.created == false'),
            ('workflow', 'group: k1-punctuation6-asr18-r1', 'group: k1-punctuation6-asr18-v1'),
            ('workflow', 'id: host\n', 'id: host\n        continue-on-error: true\n'),
        )
        for target, before, after in mutations:
            with self.subTest(target=target, before=before):
                ci = self.ci.replace(before, after) if target == 'ci' else self.ci
                workflow = self.workflow.replace(before, after) if target == 'workflow' else self.workflow
                with self.assertRaises(AssertionError):
                    self.check_wiring(ci, workflow, workflow)

    def test_actual_job_guard_allows_only_fresh_independent_r1(self):
        job = jobs(self.workflow)['asr18']
        condition = re.search(r'^    if: (.+)$', job, re.M).group(1)
        context = {
            'github.repository': 'jiying2007/kws-pipeline',
            'github.event_name': 'push',
            'github.ref': 'refs/heads/research/k1-punctuation6-asr18-r1',
            'github.event.created': True,
            'github.event.deleted': False,
            'github.event.repository.private': False,
            'github.run_attempt': 1,
        }
        self.assertTrue(asr18_job_route(condition, context))
        for key, alternatives in (
            ('github.repository', ('other/kws-pipeline', 'jiying2007/other')),
            ('github.event_name', ('workflow_dispatch', 'workflow_call', 'pull_request', 'schedule')),
            ('github.ref', ('refs/heads/research/k1-punctuation6-asr18-v1',
                            'refs/heads/research/k1-punctuation6-asr18-v2',
                            'refs/heads/research/k1-punctuation6-asr18-r10',
                            'refs/heads/research/k1-punctuation6-asr18-r1-extra',
                            'refs/heads/research/k1-punctuation6-asr18-r2',
                            'refs/heads/main', 'refs/tags/research/k1-punctuation6-asr18-r1')),
            ('github.event.created', (False, None)),
            ('github.event.deleted', (True, None)),
            ('github.event.repository.private', (True, None)),
            ('github.run_attempt', (0, 2, 3, None)),
        ):
            for value in alternatives:
                with self.subTest(key=key, value=value):
                    self.assertFalse(asr18_job_route(condition, dict(context, **{key: value})))

    def test_rejects_removing_each_fixed_module(self):
        for name in ASR18_MODULES:
            with self.subTest(name=name):
                altered = self.ci.replace('"' + name + '"', '"test_not_approved"')
                self.assertNotEqual(altered, self.ci)
                with self.assertRaises(AssertionError):
                    self.check_wiring(altered, self.workflow, self.template)

    def test_contract_shell_failure_stops_later_modes_and_host_marker(self):
        command = contract_command(named_step(self.workflow, 'K1 punctuation6 ASR18 offline source contracts'))
        host = contract_command(named_step(self.workflow, 'Admit complete new input and run fixed ASR18 once'))
        with tempfile.TemporaryDirectory(prefix='asr18-ci-shell-') as name:
            root = Path(name)
            fake = root / 'python3'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
                            'count=$(wc -l < "$CALL_LOG")\n'
                            'if [ "$count" = "$FAIL_AT" ]; then exit 7; fi\n')
            fake.chmod(0o755)
            for fail_at in (0, 1, 2, 3, 4):
                with self.subTest(fail_at=fail_at):
                    log = root / ('calls-' + str(fail_at))
                    output = root / ('output-' + str(fail_at))
                    run = subprocess.run(['bash', '-c', command + host],
                        env={'PATH': str(root) + os.pathsep + '/usr/bin:/bin',
                             'CALL_LOG': str(log), 'FAIL_AT': str(fail_at), 'GITHUB_OUTPUT': str(output)},
                        text=True, capture_output=True, check=False)
                    self.assertEqual(len(log.read_text().splitlines()), fail_at or 4)
                    self.assertEqual(run.returncode == 0, fail_at == 0)
                    self.assertEqual(output.exists(), fail_at in (0, 4))
                    if output.exists():
                        self.assertEqual(output.read_text(), 'started=true\n')

    def test_actual_step_guard_routing_requires_recorded_host_start(self):
        host_if = step_if(named_step(self.workflow, 'Admit complete new input and run fixed ASR18 once'))
        publish_if = step_if(named_step(self.workflow, 'Publish complete audio-free bounded evidence'))
        for contracts in ('', 'success', 'failure', 'cancelled', 'skipped'):
            for previous_success in (False, True):
                with self.subTest(contracts=contracts, previous_success=previous_success):
                    self.assertEqual(outcome_route(host_if, success=previous_success, contracts=contracts),
                                     previous_success and contracts == 'success')
        for host in ('', 'skipped', 'success', 'failure', 'cancelled'):
            for started in ('', 'false', 'true'):
                with self.subTest(host=host, started=started):
                    # An outcome label alone never proves that the step began.
                    self.assertEqual(outcome_route(publish_if, host=host, started=started), started == 'true')



PREPARE_MODULES = ('test_prepare_only', 'test_prepare_transport', 'test_prepare_hosted', 'test_prepare_launch_ci')
PREPARE_PATH = 'research/k1-punctuation6-prepare-diagnostic-v1'
PREPARE_JOB_IF = "${{ github.repository == 'jiying2007/kws-pipeline' && github.event_name == 'push' && github.ref == 'refs/heads/research/k1-punctuation6-prepare-diagnostic-v1' && github.event.created == true && github.event.deleted == false && github.event.repository.private == false && github.run_attempt == 1 }}"
PREPARE_HOST_IF = "${{ success() && steps.prepare_contracts.outcome == 'success' }}"


class PrepareDiagnosticCIContracts(unittest.TestCase):
    def setUp(self):
        self.ci = (ROOT / '.github/workflows/ci.yml').read_text()
        self.workflow = (ROOT / '.github/workflows/k1-punctuation6-prepare-diagnostic-v1.yml').read_text()
        self.template = (ROOT / PREPARE_PATH / 'workflow.yml').read_text()

    def expected_command(self):
        program = ('import sys,unittest; from pathlib import Path; '
                   'sys.path.insert(0,str(Path("' + PREPARE_PATH + '").resolve())); '
                   'suite=unittest.defaultTestLoader.loadTestsFromNames('
                   + repr(PREPARE_MODULES).replace("'", '"') + '); '
                   'result=unittest.TextTestRunner(verbosity=2).run(suite); '
                   'raise SystemExit(not result.wasSuccessful())')
        return ("set -euo pipefail\nprepare_tests='" + program + "'\n" + ''.join(
            'python3 -I -S -B' + mode + ' -c "$prepare_tests"\n' for mode in ('', ' -O', ' -OO')))

    def check_wiring(self, ci, workflow, template):
        self.assertEqual(workflow, template)
        self.assertEqual(workflow.split('permissions:', 1)[0],
            'name: K1 prepare-only diagnostic once\n'
            'on:\n  push:\n    branches:\n      - research/k1-punctuation6-prepare-diagnostic-v1\n')
        self.assertEqual(set(jobs(workflow)), {'prepare_diagnostic'})
        execution = jobs(workflow)['prepare_diagnostic']
        self.assertIn('    if: ' + PREPARE_JOB_IF + '\n', execution)
        self.assertIn('concurrency:\n  group: k1-punctuation6-prepare-diagnostic-v1\n  cancel-in-progress: false\n', workflow)
        shared = jobs(ci)['python-contracts']
        self.assertNotIn('    if:', shared)
        self.assertIn('    runs-on: ubuntu-24.04\n', shared)
        launch = named_step(shared, 'K1 prepare Docker create inspect remove regression')
        self.assertEqual(launch,
            'name: K1 prepare Docker create inspect remove regression\n'
            '        id: prepare_launch_ci\n'
            '        timeout-minutes: 8\n'
            '        run: python3 -I -S -B ' + PREPARE_PATH + '/prepare_launch_ci.py\n')
        self.assertEqual(ci.count('/prepare_launch_ci.py'), 1)
        self.assertNotIn('/prepare_launch_ci.py', workflow)
        self.assertLess(shared.index('id: prepare_contracts'), shared.index('id: prepare_launch_ci'))
        self.assertIn('"$RUNNER_TEMP/check-tracked-worktree.sh" "K1 prepare launch regression"', shared)
        for name, body in (('ci', shared), ('prepare', execution)):
            step = named_step(body, 'K1 punctuation6 prepare diagnostic offline source contracts')
            self.assertIn('        id: prepare_contracts\n', step)
            self.assertIn('        timeout-minutes: 5\n', step)
            self.assertNotIn('        if:', step)
            self.assertNotIn('continue-on-error:', step)
            self.assertNotIn('        env:', step)
            self.assertEqual(contract_command(step), self.expected_command(), name)
        host = named_step(execution, 'Admit fixed six inputs and prepare three models without recognition')
        publish = named_step(execution, 'Publish complete audio-free prepare diagnostic evidence')
        self.assertIn('        id: host\n', host)
        self.assertEqual(step_if(host), PREPARE_HOST_IF)
        self.assertEqual(step_if(publish), ASR18_PUBLISH_IF)
        self.assertNotIn('continue-on-error:', execution)
        self.assertLess(execution.index('id: prepare_contracts'), execution.index('id: host'))
        self.assertLess(execution.index('id: host'), execution.index('name: Publish complete'))
        self.assertEqual(contract_command(host), "printf 'started=true\\n' >> \"$GITHUB_OUTPUT\"\n"
                         'python3 -I -B ' + PREPARE_PATH + '/prepare_run.py\n')
        self.assertIn('python3 -I -B ' + PREPARE_PATH + '/diagnostic_publisher.py\n', publish)
        self.assertIn('export KWS_PREPARE_FREEZE_SHA256=', publish)
        self.assertNotIn('KWS_SOURCE_FREEZE_SHA256', publish)
        self.assertIn('"$RUNNER_TEMP/check-tracked-worktree.sh" "K1 prepare diagnostic contracts"', shared)
        for forbidden in ('prepare_run.py', 'diagnostic_publisher.py', 'setup-python',
                          'pip install', 'load_model(', '.transcribe(', '.generate('):
            self.assertNotIn(forbidden, shared)
        for forbidden in ('upload-artifact', 'actions/cache', 'workflow_dispatch:',
                          'workflow_call:', 'pull_request:', 'schedule:', 'workflow_run:'):
            self.assertNotIn(forbidden, workflow)

    def test_fixed_four_modules_run_in_required_ci_and_before_prepare(self):
        self.check_wiring(self.ci, self.workflow, self.template)
        actual = {p.stem for p in (ROOT / PREPARE_PATH).iterdir()
                  if p.is_file() and p.name.startswith('test_') and p.suffix == '.py'}
        self.assertEqual(actual, set(PREPARE_MODULES))
        self.assertEqual(len(PREPARE_MODULES), 4)

    def test_rejects_missing_modes_discovery_skips_and_bypassed_routes(self):
        mutations = (
            ('ci', 'python3 -I -S -B -OO -c "$prepare_tests"', 'true'),
            ('ci', 'loadTestsFromNames', 'discover'),
            ('ci', 'test_prepare_hosted', 'test_prepare_unreviewed'),
            ('ci', '    runs-on: ubuntu-24.04', '    runs-on: ubuntu-latest'),
            ('ci', '        id: prepare_launch_ci\n', '        id: prepare_launch_ci\n        if: false\n'),
            ('ci', '/prepare_launch_ci.py', '/prepare_run.py'),
            ('ci', '        id: prepare_launch_ci\n', '        id: prepare_launch_ci\n        env:\n          GITHUB_TOKEN: ${{ github.token }}\n'),
            ('ci', '        id: prepare_contracts\n', '        id: prepare_contracts\n        if: false\n'),
            ('workflow', PREPARE_HOST_IF, '${{ always() }}'),
            ('workflow', ASR18_PUBLISH_IF, "${{ always() && steps.host.outcome != 'skipped' }}"),
            ('workflow', 'github.run_attempt == 1', 'github.run_attempt >= 1'),
            ('workflow', 'github.event.created == true', 'github.event.created == false'),
            ('workflow', '      - research/k1-punctuation6-prepare-diagnostic-v1\n',
             '      - research/k1-punctuation6-prepare-diagnostic-v1\n      - main\n'),
            ('workflow', "github.ref == 'refs/heads/research/k1-punctuation6-prepare-diagnostic-v1'",
             "github.ref == 'refs/heads/research/k1-punctuation6-asr18-r1'"),
            ('workflow', 'group: k1-punctuation6-prepare-diagnostic-v1', 'group: k1-punctuation6-prepare-diagnostic-v2'),
            ('workflow', 'id: host\n', 'id: host\n        continue-on-error: true\n'),
        )
        for target, before, after in mutations:
            with self.subTest(target=target, before=before):
                ci = self.ci.replace(before, after) if target == 'ci' else self.ci
                workflow = self.workflow.replace(before, after) if target == 'workflow' else self.workflow
                with self.assertRaises(AssertionError):
                    self.check_wiring(ci, workflow, workflow)

    def test_actual_job_guard_allows_only_fresh_diagnostic_attempt_one(self):
        condition = re.search(r'^    if: (.+)$', jobs(self.workflow)['prepare_diagnostic'], re.M).group(1)
        context = {
            'github.repository': 'jiying2007/kws-pipeline', 'github.event_name': 'push',
            'github.ref': 'refs/heads/research/k1-punctuation6-prepare-diagnostic-v1',
            'github.event.created': True, 'github.event.deleted': False,
            'github.event.repository.private': False, 'github.run_attempt': 1,
        }
        self.assertTrue(asr18_job_route(condition, context))
        for key, alternatives in (
            ('github.repository', ('other/kws-pipeline', 'jiying2007/other')),
            ('github.event_name', ('workflow_dispatch', 'workflow_call', 'pull_request', 'schedule')),
            ('github.ref', ('refs/heads/research/k1-punctuation6-asr18-v1',
                            'refs/heads/research/k1-punctuation6-asr18-r1',
                            'refs/heads/research/k1-punctuation6-prepare-diagnostic-v10',
                            'refs/heads/research/k1-punctuation6-prepare-diagnostic-v1-extra',
                            'refs/heads/research/k1-punctuation6-prepare-diagnostic-v2',
                            'refs/heads/main', 'refs/tags/research/k1-punctuation6-prepare-diagnostic-v1')),
            ('github.event.created', (False, None)), ('github.event.deleted', (True, None)),
            ('github.event.repository.private', (True, None)),
            ('github.run_attempt', (0, 2, 3, None, '1', True)),
        ):
            for value in alternatives:
                with self.subTest(key=key, value=value):
                    self.assertFalse(asr18_job_route(condition, dict(context, **{key: value})))

    def test_rejects_removing_each_fixed_module(self):
        for name in PREPARE_MODULES:
            with self.subTest(name=name):
                altered = self.ci.replace('"' + name + '"', '"test_not_approved"')
                self.assertNotEqual(altered, self.ci)
                with self.assertRaises(AssertionError):
                    self.check_wiring(altered, self.workflow, self.template)

    def test_contract_shell_failure_stops_later_modes_and_host_marker(self):
        command = contract_command(named_step(self.workflow, 'K1 punctuation6 prepare diagnostic offline source contracts'))
        host = contract_command(named_step(self.workflow, 'Admit fixed six inputs and prepare three models without recognition'))
        with tempfile.TemporaryDirectory(prefix='prepare-ci-shell-') as name:
            root = Path(name)
            fake = root / 'python3'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
                            'count=$(wc -l < "$CALL_LOG")\n'
                            'if [ "$count" = "$FAIL_AT" ]; then exit 7; fi\n')
            fake.chmod(0o755)
            for fail_at in (0, 1, 2, 3, 4):
                with self.subTest(fail_at=fail_at):
                    log = root / ('calls-' + str(fail_at))
                    output = root / ('output-' + str(fail_at))
                    run = subprocess.run(['bash', '-c', command + host],
                        env={'PATH': str(root) + os.pathsep + '/usr/bin:/bin',
                             'CALL_LOG': str(log), 'FAIL_AT': str(fail_at), 'GITHUB_OUTPUT': str(output)},
                        text=True, capture_output=True, check=False)
                    self.assertEqual(len(log.read_text().splitlines()), fail_at or 4)
                    self.assertEqual(run.returncode == 0, fail_at == 0)
                    self.assertEqual(output.exists(), fail_at in (0, 4))
                    if output.exists():
                        self.assertEqual(output.read_text(), 'started=true\n')

if __name__ == '__main__':
    unittest.main()
