"""Invented source/receipt/trace tests; no compiler, model, network or execute call."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mel_protocol_under_test', HERE / 'mel_specialization_once.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PIPELINE = HERE.parents[1] if HERE.parent.name == 'research' else HERE.parent / 'kws-pipeline'
helper = m.load_trace_helpers(PIPELINE)


def invented_trace(value=0.0):
    raw = bytearray(helper.TRACE_MAGIC)
    for sample in helper.SAMPLES:
        for stage, count in helper.STAGE_COUNTS.items():
            raw.extend(struct.pack('<IIQ', stage, count, sample))
            raw.extend(struct.pack('<f', value) * count)
    return bytes(raw)


TRACE = invented_trace()


def enabled_pins():
    pins = m.pins_template()
    pins.update(enabled=True, worker_code_sha256='1' * 64, candidate_source_sha256='2' * 64)
    return pins


def guard_values():
    env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': m.REPOSITORY, 'GITHUB_EVENT_NAME': 'push',
        'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_ID': '999999', 'GITHUB_SHA': 'a' * 40}
    event = {'before': m.BASELINE_HEAD, 'after': env['GITHUB_SHA'], 'ref': 'refs/heads/main',
        'repository': {'full_name': m.REPOSITORY, 'private': False}}
    return env, event


def api_values():
    run = {'id': int(m.BASELINE['run_id']), 'head_sha': m.BASELINE_HEAD, 'run_attempt': 1,
        'event': 'push', 'head_branch': 'main', 'status': 'completed', 'conclusion': 'success',
        'repository': {'full_name': m.REPOSITORY, 'private': False}}
    artifact = {'id': m.BASELINE['artifact_id'], 'name': m.BASELINE['artifact_name'],
        'size_in_bytes': m.BASELINE['zip_bytes'], 'digest': 'sha256:' + m.BASELINE['zip_sha256'], 'expired': False,
        'workflow_run': {'id': int(m.BASELINE['run_id']), 'head_sha': m.BASELINE_HEAD, 'head_branch': 'main'}}
    return run, artifact


def timing(backend):
    return {'returncode': 0, 'wall_seconds': 1.0, 'host_child_cpu_seconds': 0.9,
        'stdout': json.dumps(m.EXPECTED_SUMMARIES[backend]), 'stderr': '', 'scope': 'invented fixture only'}


def saved_case():
    return {'pcm': bytes(96000), 'reference': TRACE,
        'old': {b: {'trace': TRACE, 'timing': timing(b), 'process_summary': m.EXPECTED_SUMMARIES[b]}
                for b in m.BACKENDS}}


def zip_file(path, files):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    raw = path.read_bytes()
    return {'zip_bytes': len(raw), 'zip_sha256': m.sha(raw), 'zip_members': len(files)}


def fixture_archive(root, mutate=None, after_manifest=None):
    """Entirely invented bytes, with fixture-only hashes replacing artifact pins."""
    prefix = 'backbone-parity-result/'
    files = {}
    pcm = {case: struct.pack('<h', index + 1) * 48000 for index, case in enumerate(m.CASES)}
    pcm_sha = {case: m.sha(raw) for case, raw in pcm.items()}
    generated = {'fixed_models.c': b'/* invented; never compiled */\n', 'fixed_models.h': b'/* invented header */\n'}
    generated['export_manifest.json'] = m.canonical({'schema': 'fixed-feature-pure-export-v1',
        'generated_c_sha256': m.sha(generated['fixed_models.c']), 'real_inference_performed': False,
        'numerical_parity_established': False, 'ssc305_qualification': False})
    generated_pins = {name: m.identity(raw) for name, raw in generated.items()}
    binding = {'verification_pipeline_commit': m.BASELINE_HEAD, 'consumed_original_static_generated_bytes': True,
        'head_used': False, 'head_weights_used': False,
        'frontend_sources_sha256': {name: m.FROZEN_SOURCES[m.FRONTEND_DIR + '/' + name]
            for name in ('fixed_frontend.c', 'fixed_frontend.h', 'frontend_replay.c')},
        'static_export': {'generated': generated_pins}}
    result = {'schema': 'english-fixed-shared-backbone-parity-v1', 'profile': 'FIXED_SHARED_BACKBONE_ONLY',
        'status': 'EXECUTED_BACKBONE_WITHIN_PREDECLARED_LIMITS', 'run_id': m.BASELINE['run_id'],
        'github_sha': m.BASELINE_HEAD, 'run_attempt': '1', 'passed': True, 'test_backend': False,
        'no_retry': True, 'plan': helper.PLAN, 'plan_sha256': m.BASELINE_PLAN_SHA,
        'source_binding': binding, **dict.fromkeys(m.FALSE_FLAGS, False),
        'accounting': {'onnx_session_attempts': 156, 'onnx_session_completions': 156,
            'reference_frontend_windows_completed': 78,
            'c_process_attempts': dict.fromkeys(m.BACKENDS, 3), 'c_process_completions': dict.fromkeys(m.BACKENDS, 3),
            'c_frontend_windows_completed': dict.fromkeys(m.BACKENDS, 78), 'head_evaluations': 0,
            'optimizer_updates': 0, 'new_tts_asr_phone_calls': 0, 'real_audio_cases': 0}, 'comparisons': []}
    for case in m.CASES:
        files[prefix + case + '/input.pcm'] = pcm[case]
        files[prefix + case + '/reference.bin'] = TRACE
        for backend in m.BACKENDS:
            receipt = timing(backend)
            files[prefix + case + '/' + backend + '.bin'] = TRACE
            files[prefix + case + '/' + backend + '-timing.json'] = m.canonical(receipt)
            result['comparisons'].append({'case': case, 'backend': backend, 'passed': True,
                'head_checked': False, 'embeddings': 26, 'pcm_sha256': pcm_sha[case],
                'backbone': {'passed': True}, 'process_summary': m.EXPECTED_SUMMARIES[backend], 'timing': receipt})
    for name, raw in generated.items():
        files[prefix + 'generated/' + name] = raw
    files[prefix + 'SOURCE_BINDING.json'] = m.canonical(binding)
    files[prefix + 'RESULT.json'] = m.canonical(result)
    if mutate:
        mutate(files, result)
        files[prefix + 'RESULT.json'] = m.canonical(result)
    while len(files) < 49:
        files[prefix + 'invented-' + str(len(files)) + '.txt'] = b'invented, no executable data\n'
    manifest = {'schema': result['schema'], 'profile': result['profile'], 'plan': helper.PLAN,
        'plan_sha256': m.BASELINE_PLAN_SHA, **dict.fromkeys(m.FALSE_FLAGS, False),
        'files': {name[len(prefix):]: m.identity(raw) for name, raw in files.items()}}
    files[prefix + 'MANIFEST.json'] = m.canonical(manifest)
    for i in range(11):
        files['invented-evidence/' + str(i) + '.txt'] = b'invented metadata\n'
    if after_manifest:
        after_manifest(files)
    path = root / 'invented.zip'
    pin = {**m.BASELINE, **zip_file(path, files), 'result_sha256': m.sha(files[prefix + 'RESULT.json'])}
    return path, pin, generated_pins, pcm_sha


class ProtocolTests(unittest.TestCase):
    def test_default_is_dormant_and_imports_no_ml(self):
        with mock.patch.object(m, 'execute', side_effect=AssertionError('execute forbidden')), \
             mock.patch.object(m.subprocess, 'Popen', side_effect=AssertionError('process forbidden')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main([]), 0)
        self.assertEqual(json.loads(output.getvalue())['c_processes'], 0)
        self.assertFalse({'numpy', 'onnx', 'onnxruntime', 'torch'} & set(sys.modules))

    def test_separate_plan_keeps_zero_new_reference_and_old_tolerances(self):
        self.assertNotEqual(m.FIXED_PLAN_SHA, m.BASELINE_PLAN_SHA)
        self.assertEqual(m.sha(m.canonical(m.PLAN)), m.FIXED_PLAN_SHA)
        self.assertEqual(m.LIMITS, helper.LIMITS)
        self.assertEqual(m.PLAN['maximum_c_processes_per_backend'], 3)
        self.assertEqual(m.PLAN['new_reference_onnx_session_runs'], 0)
        self.assertEqual(m.PLAN['old_implementation_processes'], 0)
        self.assertEqual(m.PLAN['total_timeout_seconds'], 1800)

    def test_template_stays_disabled_without_concrete_source_identities(self):
        pins = m.pins_template()
        self.assertIs(pins['enabled'], False)
        self.assertIsNone(pins['candidate_source_sha256'])
        with self.assertRaisesRegex(ValueError, 'DISABLED'):
            m.validate_pins(pins)

    def test_guard_accepts_only_the_pure_first_successor_fixture(self):
        env, event = guard_values()
        m.execution_guard(enabled_pins(), env, event, '1' * 64)
        for key, value in (('GITHUB_ACTIONS', 'false'), ('GITHUB_EVENT_NAME', 'workflow_dispatch'),
                           ('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_RUN_ID', m.BASELINE['run_id']),
                           ('GITHUB_REF', 'refs/heads/other'), ('GITHUB_SHA', 'b' * 40)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.execution_guard(enabled_pins(), {**env, key: value}, event, '1' * 64)
        for change in ({'before': 'c' * 40}, {'repository': {'full_name': m.REPOSITORY, 'private': True}}):
            with self.assertRaises(ValueError):
                m.execution_guard(enabled_pins(), env, {**event, **change}, '1' * 64)

    def test_pin_types_limits_and_old_candidate_cannot_be_relaxed(self):
        for change in ({'no_retry': 1}, {'head_parity_established': 0}, {'extra': 'x'},
                       {'candidate_source_sha256': m.FROZEN_SOURCES[m.FRONTEND_DIR + '/fixed_frontend.c']},
                       {'worker_code_sha256': None}, {'plan_sha256': m.BASELINE_PLAN_SHA}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_pins({**enabled_pins(), **change})

    def test_api_metadata_does_not_mix_old_source_with_new_execution(self):
        run, artifact = api_values()
        m.baseline_api_evidence(run, artifact)
        for change in ({'head_sha': 'a' * 40}, {'run_attempt': True}, {'conclusion': 'failure'}):
            with self.assertRaises(ValueError):
                m.baseline_api_evidence({**run, **change}, artifact)
        for change in ({'expired': True}, {'digest': 'sha256:' + '0' * 64}, {'size_in_bytes': True},
                       {'workflow_run': {'id': 999999, 'head_sha': m.BASELINE_HEAD, 'head_branch': 'main'}}):
            with self.assertRaises(ValueError):
                m.baseline_api_evidence(run, {**artifact, **change})

    def test_archive_rejects_escape_symlink_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for index, names in enumerate((['../escape'], ['same', 'same'], ['link'])):
                path = root / (str(index) + '.zip')
                with zipfile.ZipFile(path, 'w') as archive:
                    for name in names:
                        info = zipfile.ZipInfo(name)
                        if name == 'link':
                            info.external_attr = (stat.S_IFLNK | 0o777) << 16
                        archive.writestr(info, b'invented')
                pin = {'zip_bytes': path.stat().st_size, 'zip_sha256': m.sha(path.read_bytes()), 'zip_members': len(names)}
                with self.assertRaises(ValueError):
                    m.checked_archive(path, pin)

    def test_invented_closed_archive_is_read_without_processes_or_models(self):
        with tempfile.TemporaryDirectory() as root:
            path, pin, generated, pcm = fixture_archive(Path(root))
            with mock.patch.multiple(m, BASELINE=pin, GENERATED=generated, PCM_SHA=pcm), \
                 mock.patch.object(m.subprocess, 'Popen', side_effect=AssertionError('no process')):
                result = m.read_baseline(path, helper)
            self.assertEqual(result['closed_manifest_files'], 49)
            self.assertEqual(set(result['cases']), set(m.CASES))
            self.assertEqual(result['cases']['zero']['reference'], TRACE)

    def test_manifest_binds_every_member_instead_of_trusting_result_summary(self):
        for change in (lambda files: files.__setitem__('backbone-parity-result/zero/native.bin', b'changed'),
                       lambda files: files.__setitem__('backbone-parity-result/unregistered.txt', b'extra')):
            with tempfile.TemporaryDirectory() as root:
                path, pin, generated, pcm = fixture_archive(Path(root), after_manifest=change)
                with mock.patch.multiple(m, BASELINE=pin, GENERATED=generated, PCM_SHA=pcm), self.assertRaises(ValueError):
                    m.read_baseline(path, helper)

    def test_raw_trace_clock_is_checked_even_with_consistent_fixture_hashes(self):
        def mutate(files, _result):
            raw = bytearray(TRACE)
            struct.pack_into('<Q', raw, 16, 16001)
            files['backbone-parity-result/zero/native.bin'] = bytes(raw)
        with tempfile.TemporaryDirectory() as root:
            path, pin, generated, pcm = fixture_archive(Path(root), mutate=mutate)
            with mock.patch.multiple(m, BASELINE=pin, GENERATED=generated, PCM_SHA=pcm), \
                 self.assertRaisesRegex(ValueError, 'TRACE_STAGE_COUNT_CLOCK'):
                m.read_baseline(path, helper)

    def test_source_run_and_full_six_roster_are_required(self):
        def mutate(_files, result):
            result['comparisons'][-1] = result['comparisons'][0]
        with tempfile.TemporaryDirectory() as root:
            path, pin, generated, pcm = fixture_archive(Path(root), mutate=mutate)
            with mock.patch.multiple(m, BASELINE=pin, GENERATED=generated, PCM_SHA=pcm), self.assertRaises(ValueError):
                m.read_baseline(path, helper)

    def test_candidate_requires_all_three_gates(self):
        report = m.compare_candidate(helper, 'zero', 'native', TRACE, timing('native'), saved_case())
        self.assertTrue(report['passed'])
        self.assertTrue(report['bit_identity']['passed'])
        self.assertTrue(report['unchanged_workspace_and_state'])

    def test_numeric_tolerance_does_not_waive_a_single_trace_byte(self):
        raw = bytearray(TRACE)
        struct.pack_into('<f', raw, 24, 1e-8)
        report = m.compare_candidate(helper, 'zero', 'native', bytes(raw), timing('native'), saved_case())
        self.assertTrue(report['saved_ort_comparison']['passed'])
        self.assertFalse(report['bit_identity']['passed'])
        self.assertFalse(report['passed'])

    def test_signed_zero_must_remain_bit_identical(self):
        raw = bytearray(TRACE)
        struct.pack_into('<f', raw, 24, -0.0)
        report = m.compare_candidate(helper, 'zero', 'arm', bytes(raw), timing('arm'), saved_case())
        self.assertTrue(report['saved_ort_comparison']['passed'])
        self.assertFalse(report['passed'])

    def test_workspace_or_state_change_is_a_failure_with_identical_trace(self):
        for field in ('workspace_bytes', 'frontend_state_bytes'):
            receipt = timing('native')
            summary = json.loads(receipt['stdout'])
            summary[field] += 4
            receipt['stdout'] = json.dumps(summary)
            report = m.compare_candidate(helper, 'zero', 'native', TRACE, receipt, saved_case())
            self.assertTrue(report['bit_identity']['passed'])
            self.assertFalse(report['passed'])

    def test_nonfinite_or_head_trace_is_not_a_comparison_pass(self):
        for field, value in (('float', float('nan')), ('stage', 4)):
            raw = bytearray(TRACE)
            struct.pack_into('<f' if field == 'float' else '<I', raw, 24 if field == 'float' else 8, value)
            with self.assertRaises(ValueError):
                m.compare_candidate(helper, 'zero', 'native', bytes(raw), timing('native'), saved_case())

    def test_process_reservation_rejects_fourth_replay_and_second_build(self):
        account = m.fresh_accounting()
        for backend in m.BACKENDS:
            m.reserve_attempt(account, 'build', backend)
            with self.assertRaises(ValueError):
                m.reserve_attempt(account, 'build', backend)
            for _ in range(3):
                m.reserve_attempt(account, 'c_process', backend)
            with self.assertRaises(ValueError):
                m.reserve_attempt(account, 'c_process', backend)
        self.assertEqual(account['new_onnx_session_attempts'], 0)
        self.assertEqual(account['c_process_attempts'], {'native': 3, 'arm': 3})

    def test_compile_command_selects_new_candidate_and_no_head_or_old_c(self):
        for backend in m.BACKENDS:
            command = [str(x) for x in m.build_command('cc', backend, Path('/invented'), Path('/generated'), '/out')]
            self.assertIn('/invented/' + m.CANDIDATE, command)
            self.assertNotIn('/invented/' + m.FRONTEND_DIR + '/fixed_frontend.c', command)
            self.assertFalse(any('HEAD' in part or 'head.c' in part or 'onnx' in part for part in command))
            self.assertIn('-fno-fast-math', command)
            self.assertIn('-ffp-contract=off', command)
            self.assertEqual('-static' in command, backend == 'arm')

    def test_source_preflight_requires_exact_clean_head_and_tracked_fixed_paths(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            worker = root / m.PROFILE_DIR / 'mel_specialization_once.py'
            pins_path = root / m.PROFILE_DIR / 'mel_specialization_pins.json'
            contents = {**{name: b'invented original source' for name in m.FROZEN_SOURCES},
                m.CANDIDATE: b'invented candidate source',
                m.PROFILE_DIR + '/mel_specialization_once.py': b'invented worker source',
                m.PROFILE_DIR + '/mel_specialization_pins.json': b'{}'}
            for name, raw in contents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            hashes = {name: m.sha(contents[name]) for name in m.FROZEN_SOURCES}
            pins = {'worker_code_sha256': m.sha(worker.read_bytes()),
                    'candidate_source_sha256': m.sha(contents[m.CANDIDATE])}
            with mock.patch.object(m, '__file__', str(worker)), mock.patch.object(m, 'FROZEN_SOURCES', hashes), \
                 mock.patch.object(m.subprocess, 'check_output', side_effect=['a' * 40 + '\n', '']), \
                 mock.patch.object(m.subprocess, 'run') as tracked:
                record = m.source_preflight(root, pins, 'a' * 40, pins_path)
                self.assertEqual(record['execution_pipeline_commit'], 'a' * 40)
                self.assertIn('--error-unmatch', tracked.call_args.args[0])
                self.assertIn(m.CANDIDATE, tracked.call_args.args[0])
            with mock.patch.object(m, '__file__', str(worker)), \
                 mock.patch.object(m.subprocess, 'check_output', side_effect=['a' * 40 + '\n', '?? untracked']), \
                 self.assertRaisesRegex(ValueError, 'EXACT_CLEAN_EXECUTION_HEAD'):
                m.source_preflight(root, pins, 'a' * 40, pins_path)

    def test_interrupted_child_is_killed_once_and_has_failure_receipt(self):
        child = mock.Mock(pid=999999, returncode=-9)
        child.communicate.side_effect = [subprocess.TimeoutExpired('invented-command', 1), ('partial', 'interrupted')]
        with tempfile.TemporaryDirectory() as root, \
             mock.patch.object(m.subprocess, 'Popen', return_value=child) as spawn, \
             mock.patch.object(m.os, 'killpg') as kill:
            path = Path(root) / 'timing.json'
            with self.assertRaises(subprocess.TimeoutExpired):
                m.timed_process(['invented-command'], 1, path)
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt['status'], 'FAILED_OR_INTERRUPTED_NO_RETRY')
            self.assertEqual(receipt['retry_count'], 0)
            self.assertEqual(spawn.call_count, 1)
            kill.assert_called_once_with(child.pid, m.signal.SIGKILL)

    def test_duplicate_json_nonfinite_and_overwrite_fail_closed(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                m.json_bytes(raw)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'receipt.json'
            m.once(path, {'status': 'STARTED'})
            with self.assertRaises(FileExistsError):
                m.once(path, {'status': 'replacement'})


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    unittest.main()
