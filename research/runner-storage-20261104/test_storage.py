"""Invented fixtures only: no actual disk-capacity, Docker, model or remote calls."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('collect_storage', HERE / 'collect_storage.py')
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)


def fixtures():
    env = dict(GITHUB_REPOSITORY=storage.REPOSITORY, GITHUB_REF=storage.REF,
               GITHUB_EVENT_NAME='push', GITHUB_RUN_NUMBER='1', GITHUB_RUN_ATTEMPT='1',
               GITHUB_ACTIONS='true', RUNNER_ENVIRONMENT='github-hosted', RUNNER_OS='Linux',
               RUNNER_ARCH='X64', GITHUB_SHA='a' * 40, GITHUB_WORKFLOW_SHA='a' * 40,
               GITHUB_RUN_ID='41822449999', GITHUB_WORKSPACE=str(storage.ROOT),
               ImageOS='ubuntu24', ImageVersion='20261103.1.0',
               GITHUB_WORKFLOW_REF=storage.REPOSITORY + '/' + storage.WORKFLOW + '@' + storage.REF)
    event = dict(created=True, deleted=False, before='0' * 40, after='a' * 40,
                 ref=storage.REF, repository=dict(private=False, full_name=storage.REPOSITORY))
    return env, event


def counters(device=2049, free=None):
    if free is None:
        free = storage.RESERVE_BYTES
    stat = NS(st_dev=device)
    vfs = NS(f_frsize=4096, f_bsize=4096, f_blocks=20000000, f_bfree=10000000,
             f_bavail=9000000, f_files=1000000, f_ffree=900000, f_favail=800000)
    usage = NS(total=100 * 1024 ** 3, used=40 * 1024 ** 3, free=free)
    return stat, vfs, usage


def measured(free=None):
    stat, vfs, usage = counters(free=free)
    return storage.measure_one('workspace', '/PRIVATE-TEST-PATH',
                               Mock(return_value=stat), Mock(return_value=vfs),
                               Mock(return_value=usage))


class AdmissionTests(unittest.TestCase):
    def test_exact_identity(self):
        env, event = fixtures()
        identity = storage.admission(env, event, storage.ROOT)
        self.assertEqual(identity['head_sha'], 'a' * 40)
        self.assertEqual(identity['run_id'], 41822449999)
        self.assertEqual(identity['run_number'], 1)
        self.assertEqual(identity['run_attempt'], 1)
        self.assertEqual(identity['workflow_sha'], 'a' * 40)
        self.assertEqual(identity['runner_image'],
                         dict(os='ubuntu24', version='20261103.1.0', metadata_complete=True))
        self.assertEqual(identity['run_url'],
                         'https://github.com/jiying2007/kws-pipeline/actions/runs/41822449999')

    def test_every_environment_gate(self):
        env, event = fixtures()
        mutations = dict(GITHUB_REPOSITORY='other/repo', GITHUB_REF='refs/heads/main',
                         GITHUB_EVENT_NAME='workflow_dispatch', GITHUB_RUN_NUMBER='2',
                         GITHUB_RUN_ATTEMPT='2', GITHUB_ACTIONS='false',
                         RUNNER_ENVIRONMENT='self-hosted', RUNNER_OS='Windows', RUNNER_ARCH='ARM64',
                         GITHUB_SHA='not-a-sha', GITHUB_WORKFLOW_SHA='b' * 40,
                         GITHUB_RUN_ID='0', GITHUB_WORKFLOW_REF='unreviewed',
                         GITHUB_WORKSPACE='/invented-wrong-workspace')
        for key, value in mutations.items():
            with self.subTest(key=key):
                altered = dict(env, **{key: value})
                with self.assertRaises(ValueError):
                    storage.admission(altered, event, storage.ROOT)

    def test_every_event_gate(self):
        env, event = fixtures()
        mutations = dict(created=False, deleted=True, before='b' * 40,
                         after='b' * 40, ref='refs/heads/main',
                         repository=dict(private=True, full_name=storage.REPOSITORY))
        for key, value in mutations.items():
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    storage.admission(dict(env), dict(event, **{key: value}), storage.ROOT)
        for repository in (None, {}, dict(private=False, full_name='other/repo')):
            with self.assertRaises(ValueError):
                storage.admission(env, dict(event, repository=repository), storage.ROOT)

    def test_true_boolean_not_integer_or_string(self):
        env, event = fixtures()
        for created in (1, 'true', None):
            with self.assertRaises(ValueError):
                storage.admission(env, dict(event, created=created), storage.ROOT)
        for private in (0, 'false', None):
            altered = copy.deepcopy(event)
            altered['repository']['private'] = private
            with self.assertRaises(ValueError):
                storage.admission(env, altered, storage.ROOT)

    def test_no_extra_event_or_environment_content_in_identity(self):
        env, event = fixtures()
        env['UNRELATED_SECRET'] = 'NEVER-EXPORT'
        event['arbitrary'] = {'private_path': '/NEVER-EXPORT'}
        identity = json.dumps(storage.admission(env, event, storage.ROOT))
        self.assertNotIn('NEVER-EXPORT', identity)
        self.assertNotIn('GITHUB_WORKSPACE', identity)

    def test_image_metadata_is_optional_and_fixed_format_only(self):
        env, event = fixtures()
        for value in ('PRIVATE-SECRET', '20261103.1.0\nPRIVATE-SECRET', '1.2.3', None, '1' * 200):
            altered = dict(env, ImageOS='PRIVATE-SECRET', ImageVersion=value)
            identity = storage.admission(altered, event, storage.ROOT)
            self.assertEqual(identity['runner_image'],
                             dict(os=None, version=None, metadata_complete=False))
            self.assertNotIn('PRIVATE-SECRET', json.dumps(identity))
        env.pop('ImageOS')
        env.pop('ImageVersion')
        self.assertFalse(storage.admission(env, event, storage.ROOT)['runner_image']['metadata_complete'])


class MeasurementTests(unittest.TestCase):
    def test_exact_three_targets_and_one_call_per_api(self):
        stat, vfs, usage = counters()
        stat_fn = Mock(return_value=stat)
        vfs_fn = Mock(return_value=vfs)
        usage_fn = Mock(return_value=usage)
        report = storage.collect('/invented-workspace', {}, 'b' * 64,
                                 stat_fn, vfs_fn, usage_fn, '2026-11-04T00:00:00Z')
        expected = ['/', '/invented-workspace', '/var/lib/docker']
        for fn in (stat_fn, vfs_fn, usage_fn):
            self.assertEqual([call.args[0] for call in fn.call_args_list], expected)
        self.assertEqual(report['status'], 'measured_all')
        self.assertEqual([row['label'] for row in report['targets']],
                         ['root', 'workspace', 'docker_root'])
        self.assertEqual(len(report['shared_devices']), 3)
        self.assertTrue(all(row['same_device'] for row in report['shared_devices']))
        self.assertNotIn('/invented-workspace', json.dumps(report))

    def test_inode_capacity_and_device_aggregates(self):
        row = measured()
        self.assertEqual(row['disk_usage']['available_bytes'], storage.RESERVE_BYTES)
        self.assertEqual(row['statvfs']['available_bytes'], 9000000 * 4096)
        self.assertEqual(row['statvfs']['free_bytes'], 10000000 * 4096)
        self.assertEqual(row['statvfs']['available_inodes'], 800000)
        self.assertEqual(row['device_major'], os.major(2049))
        self.assertEqual(row['device_minor'], os.minor(2049))
        self.assertNotIn('PRIVATE', json.dumps(row))

    def test_original_reserve_boundary_is_informational(self):
        self.assertEqual(storage.RESERVE_BYTES, 26843545600)
        self.assertFalse(measured(storage.RESERVE_BYTES - 1)['meets_unchanged_prepare_reserve'])
        self.assertTrue(measured(storage.RESERVE_BYTES)['meets_unchanged_prepare_reserve'])
        self.assertEqual(measured(0)['status'], 'measured')

    def test_distinct_snapshots_need_not_match_and_minimum_is_conservative(self):
        stat, vfs, usage = counters(free=storage.RESERVE_BYTES + 4096)
        vfs.f_bavail = (storage.RESERVE_BYTES // 4096) - 1
        row = storage.measure_one('workspace', '/fixture', Mock(return_value=stat),
                                  Mock(return_value=vfs), Mock(return_value=usage))
        self.assertEqual(row['status'], 'measured')
        self.assertNotEqual(row['disk_usage']['available_bytes'], row['statvfs']['available_bytes'])
        self.assertEqual(row['conservative_available_bytes'], storage.RESERVE_BYTES - 4096)
        self.assertFalse(row['meets_unchanged_prepare_reserve'])

    def test_shared_device_false_is_not_separate_mount_claim(self):
        _, vfs, usage = counters()
        report = storage.collect('/fixture', {}, 'b' * 64,
                                 Mock(side_effect=[NS(st_dev=1), NS(st_dev=2), NS(st_dev=2)]),
                                 Mock(return_value=vfs), Mock(return_value=usage), 'fixture-time')
        self.assertEqual([row['same_device'] for row in report['shared_devices']],
                         [False, False, True])
        self.assertTrue(report['device_equality_is_not_mount_identity'])

    def test_missing_target_sanitizes_exception_and_preserves_other_targets(self):
        stat, vfs, usage = counters()
        stat_fn = Mock(side_effect=[stat, stat, FileNotFoundError(2, 'SECRET-BODY', '/private/file')])
        report = storage.collect('/fixture', {}, 'b' * 64, stat_fn,
                                 Mock(return_value=vfs), Mock(return_value=usage), 'fixture-time')
        self.assertEqual(report['status'], 'measured_partial')
        self.assertEqual(report['targets'][2], dict(label='docker_root', status='unavailable',
                                                   error_kind='os_error', errno=2))
        self.assertEqual([row['same_device'] for row in report['shared_devices']], [True, None, None])
        self.assertNotIn('SECRET', json.dumps(report))
        self.assertNotIn('/private/file', json.dumps(report))

    def test_failure_does_not_retry_or_call_later_api(self):
        stat_fn = Mock(side_effect=PermissionError(13, 'SECRET'))
        vfs_fn, usage_fn = Mock(), Mock()
        row = storage.measure_one('docker_root', '/fixed', stat_fn, vfs_fn, usage_fn)
        self.assertEqual(row['errno'], 13)
        self.assertEqual(stat_fn.call_count, 1)
        vfs_fn.assert_not_called()
        usage_fn.assert_not_called()

    def test_each_capacity_api_error_has_only_numeric_errno(self):
        stat, vfs, usage = counters()
        for failing in ('vfs', 'usage'):
            a, b, c = Mock(return_value=stat), Mock(return_value=vfs), Mock(return_value=usage)
            (b if failing == 'vfs' else c).side_effect = OSError(5, '/PRIVATE-SECRET')
            row = storage.measure_one('root', '/', a, b, c)
            self.assertEqual(row, dict(label='root', status='unavailable', error_kind='os_error', errno=5))

    def test_invalid_counters_fail_closed(self):
        for field, value in (('f_frsize', 0), ('f_blocks', -1), ('f_bfree', 20000001),
                             ('f_bavail', True), ('f_favail', 900001), ('f_files', 'SECRET')):
            stat, vfs, usage = counters()
            setattr(vfs, field, value)
            row = storage.measure_one('root', '/', Mock(return_value=stat),
                                      Mock(return_value=vfs), Mock(return_value=usage))
            self.assertEqual(row['error_kind'], 'invalid_aggregate')
            self.assertNotIn('SECRET', json.dumps(row))

    def test_invalid_disk_usage_fails_closed(self):
        stat, vfs, usage = counters()
        usage.free = usage.total + 1
        row = storage.measure_one('root', '/', Mock(return_value=stat),
                                  Mock(return_value=vfs), Mock(return_value=usage))
        self.assertEqual(row['error_kind'], 'invalid_aggregate')

    def test_fixed_label(self):
        fn = Mock()
        with self.assertRaises(ValueError):
            storage.measure_one('/PRIVATE-LABEL', '/', fn, fn, fn)
        fn.assert_not_called()

    def test_no_model_cleanup_or_previous_run_capacity_claims(self):
        stat, vfs, usage = counters()
        report = storage.collect('/fixture', {}, 'b' * 64, Mock(return_value=stat),
                                 Mock(return_value=vfs), Mock(return_value=usage), 'fixture-time')
        for key in ('recursive_scans', 'docker_commands', 'model_calls', 'model_downloads',
                    'package_installs', 'cleanup_actions', 'experiment_retries'):
            self.assertEqual(report[key], 0)
        self.assertFalse(report['prepare_reserve_changed'])
        self.assertFalse(report['previous_prepare_disk_bytes_observed'])
        self.assertFalse(report['measurement_is_previous_prepare_runner'])
        self.assertFalse(report['authorizes_model_execution'])
        self.assertEqual(report['previous_prepare_run_id'], 41822443746)


class SourceAndWorkflowTests(unittest.TestCase):
    def test_checked_in_source_freeze(self):
        digest = storage.verify_sources(storage.ROOT)
        expected = hashlib.sha256((storage.ROOT / storage.LOCK_PATH).read_bytes()).hexdigest()
        self.assertEqual(digest, expected)

    def fixture_root(self, root):
        for relative in storage.SOURCE_FILES + (storage.LOCK_PATH,):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((storage.ROOT / relative).read_bytes())

    def test_modified_source_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture_root(root)
            (root / storage.WORKFLOW).write_text('unreviewed')
            with self.assertRaisesRegex(ValueError, 'source freeze'):
                storage.verify_sources(root)

    def test_extra_lock_path_fails_before_reading_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture_root(root)
            path = root / storage.LOCK_PATH
            lock = json.loads(path.read_text())
            lock['files']['/PRIVATE-NOT-READ'] = 'a' * 64
            path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, 'exact source set'):
                storage.verify_sources(root)

    def test_source_symlink_fails_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture_root(root)
            target = root / storage.WORKFLOW
            target.rename(target.with_suffix('.fixture'))
            target.symlink_to('/PRIVATE-NOT-READ')
            with self.assertRaisesRegex(ValueError, 'symlink forbidden'):
                storage.verify_sources(root)

    def test_bounded_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            path.write_bytes(b'12345')
            self.assertEqual(storage.bounded_read(path, 5), b'12345')
            with self.assertRaisesRegex(ValueError, 'bounded input'):
                storage.bounded_read(path, 4)

    def test_python_38_syntax_and_closed_standard_library_imports(self):
        for path in (HERE / 'collect_storage.py', HERE / 'test_storage.py'):
            ast.parse(path.read_text(), feature_version=(3, 8))
        tree = ast.parse((HERE / 'collect_storage.py').read_text())
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                modules.add(node.module)
        self.assertEqual(modules, {'pathlib', 'datetime', 'hashlib', 'json', 'os', 're', 'shutil', 'sys'})
        forbidden = {'walk', 'rglob', 'glob', 'scandir', 'listdir', 'system', 'popen',
                     'unlink', 'rmdir', 'remove', 'rmtree', 'exec', 'eval', 'spawn', 'fork'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else '')
                self.assertNotIn(name, forbidden)

    def test_exact_one_job_pins_guards_and_single_aggregate_artifact(self):
        workflow = (storage.ROOT / storage.WORKFLOW).read_text()
        self.assertIn('timeout-minutes: 5\n', workflow)
        self.assertIn('runs-on: ubuntu-24.04\n', workflow)
        self.assertEqual(workflow.count('    runs-on:'), 1)
        self.assertEqual(workflow.count('      - uses:'), 2)
        self.assertIn('actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1', workflow)
        self.assertIn('actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02', workflow)
        self.assertIn('persist-credentials: false', workflow)
        self.assertIn('ref: ${{ github.sha }}', workflow)
        for condition in ("github.repository == 'jiying2007/kws-pipeline'",
                          "github.event_name == 'push'", "github.ref == '" + storage.REF + "'",
                          'github.event.created == true', 'github.event.deleted == false',
                          'github.event.repository.private == false',
                          'github.run_number == 1', 'github.run_attempt == 1'):
            self.assertIn(condition, workflow)
        self.assertIn('path: ' + storage.OUTPUT + '/status.json\n', workflow)
        self.assertEqual(workflow.count('          path:'), 1)
        self.assertIn('contents: read', workflow)
        self.assertNotIn('workflow_dispatch', workflow)
        self.assertNotIn('schedule:', workflow)
        self.assertNotIn('sudo', workflow)
        self.assertNotIn('docker ', workflow)
        self.assertNotIn('pip ', workflow)
        self.assertNotIn('apt ', workflow)
        self.assertIn("for opt in '' -O -OO; do", workflow)


class RunTests(unittest.TestCase):
    def test_actual_main_rejects_unadmitted_environment_without_capacity_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event_path = root / 'synthetic-event.json'
            event_path.write_text('{}')
            output = io.StringIO()
            with patch.object(storage, 'ROOT', root), \
                    patch.dict(storage.os.environ, {'GITHUB_EVENT_PATH': str(event_path)}, clear=True), \
                    patch.object(storage.os, 'stat', side_effect=AssertionError('no stat')) as stat, \
                    patch.object(storage.os, 'statvfs', side_effect=AssertionError('no vfs')) as vfs, \
                    patch.object(storage.shutil, 'disk_usage', side_effect=AssertionError('no usage')) as usage, \
                    contextlib.redirect_stderr(output):
                self.assertEqual(storage.main(), 2)
                stat.assert_not_called()
                vfs.assert_not_called()
                usage.assert_not_called()
            self.assertFalse((root / storage.OUTPUT).exists())
            self.assertNotIn(str(root), output.getvalue())

    def test_run_writes_only_report_once_and_refuses_second_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event_path = root / 'synthetic-event.json'
            event_path.write_text('{}')
            report = {'status': 'measured_all', 'fixture_only': True}
            with patch.object(storage, 'ROOT', root), \
                    patch.object(storage, 'admission', return_value={'fixture': True}), \
                    patch.object(storage, 'verify_sources', return_value='b' * 64), \
                    patch.object(storage, 'collect', return_value=report) as collect:
                self.assertEqual(storage.run({'GITHUB_EVENT_PATH': str(event_path)}), 0)
                self.assertEqual(collect.call_count, 1)
                output = root / storage.OUTPUT
                self.assertEqual([p.name for p in output.iterdir()], ['status.json'])
                self.assertEqual(json.loads((output / 'status.json').read_text()), report)
                with self.assertRaises(FileExistsError):
                    storage.run({'GITHUB_EVENT_PATH': str(event_path)})
                self.assertEqual(collect.call_count, 1)

    def test_partial_report_is_written_but_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event_path = root / 'synthetic-event.json'
            event_path.write_text('{}')
            with patch.object(storage, 'ROOT', root), \
                    patch.object(storage, 'admission', return_value={}), \
                    patch.object(storage, 'verify_sources', return_value='b' * 64), \
                    patch.object(storage, 'collect', return_value={'status': 'measured_partial'}):
                self.assertEqual(storage.run({'GITHUB_EVENT_PATH': str(event_path)}), 2)
            self.assertEqual(json.loads((root / storage.OUTPUT / 'status.json').read_text()),
                             {'status': 'measured_partial'})

    def test_main_does_not_print_exception_body_or_traceback(self):
        output = io.StringIO()
        with patch.object(storage, 'run', side_effect=RuntimeError('/PRIVATE-SECRET')), \
                contextlib.redirect_stderr(output):
            self.assertEqual(storage.main(), 2)
        self.assertNotIn('PRIVATE', output.getvalue())
        self.assertNotIn('Traceback', output.getvalue())
        self.assertEqual(len(output.getvalue().splitlines()), 1)

    def test_rejected_admission_never_collects_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event_path = root / 'synthetic-event.json'
            event_path.write_text('{}')
            with patch.object(storage, 'ROOT', root), \
                    patch.object(storage, 'collect') as collect:
                with self.assertRaises(ValueError):
                    storage.run({'GITHUB_EVENT_PATH': str(event_path)})
                collect.assert_not_called()
            self.assertFalse((root / storage.OUTPUT).exists())


if __name__ == '__main__':
    unittest.main()
