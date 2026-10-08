"""Independent offline regression matrix for the reviewed atomic Git transport.

Every Git integration transaction targets a disposable local bare repository.
No credentials, hosted Git service, remote mutation, or network access is used.
"""
from __future__ import annotations
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import unittest
from unittest.mock import patch

from atomic_git_test_fixtures import GitFixture, run_git, ZERO, ANCHOR, ORIGINAL, SOURCE

ROOT = Path(__file__).resolve().parent


def module(name):
    if name in sys.modules:
        return sys.modules[name]
    location = ROOT / (name + '.py')
    if not location.exists():
        location = ROOT.parent / 'tools' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, location)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


G = module('git_atomic_guard_20261008')
T = module('git_atomic_prune_20261008')
GUARD_PATH = Path(G.__file__)


def binding(count=57, remote='https://github.com/jiying2007/kws-pipeline.git'):
    return {'remote': remote, 'anchor': ANCHOR, 'archive_oid': 'a' * 40,
            'delete_refs': [{'name': f'refs/heads/frozen/topic-{n:03}', 'before_oid': 'b' * 40}
                            for n in range(count)], 'local_ref': SOURCE}


def fixture_binding(f):
    b = binding(f.count, str(f.remote))
    b['archive_oid'] = f.archive
    b['delete_refs'] = [{'name': ref, 'before_oid': oid} for ref, oid in f.candidates.items()]
    return b


def rows(b):
    return ([f"{b['local_ref']} {b['archive_oid']} {b['anchor']} {ZERO}"] +
            [f"(delete) {ZERO} {r['name']} {r['before_oid']}" for r in b['delete_refs']])


def encoded(lines):
    return ('\n'.join(lines) + '\n').encode('ascii')


def successful_porcelain(b):
    return ('To ' + b['remote'] + '\n' + ''.join(
        f"-\t:{r['name']}\t[deleted]\n" for r in b['delete_refs']) +
        f"*\t{b['local_ref']}:{b['anchor']}\t[new tag]\nDone\n").encode('ascii')


class GuardRowsTests(unittest.TestCase):
    def setUp(self):
        self.b = binding()
        self.good = rows(self.b)

    def validate(self, value, b=None, remote_name=None, remote_url=None):
        b = self.b if b is None else b
        return G.validate_rows(value, b, b['remote'] if remote_name is None else remote_name,
                               b['remote'] if remote_url is None else remote_url)

    def reject(self, lines):
        with self.assertRaises(ValueError):
            self.validate(encoded(lines) if isinstance(lines, list) else lines)

    def test_exact_58_and_23_rows_accept(self):
        for count in (57, 22):
            with self.subTest(count=count):
                b = binding(count)
                self.validate(encoded(rows(b)), b)

    def test_row_order_is_not_significant(self):
        self.validate(encoded(list(reversed(self.good))))

    def test_empty_or_missing_anchor_or_missing_candidate_rejects(self):
        for value in ([], self.good[1:], self.good[:-1]):
            with self.subTest(rows=len(value)):
                self.reject(value)

    def test_duplicate_anchor_candidate_or_extra_ref_rejects(self):
        for extra in (self.good[0], self.good[1], f'(delete) {ZERO} refs/heads/extra ' + 'b' * 40):
            self.reject(self.good + [extra])

    def test_protected_main_original_tag_held_and_extra_target_rejects(self):
        for ref in ('refs/heads/main', ORIGINAL, 'refs/heads/feature/held-one', 'refs/tags/extra'):
            value = self.good.copy()
            value[-1] = f'(delete) {ZERO} {ref} ' + 'b' * 40
            self.reject(value)

    def test_anchor_already_exists_same_or_other_target_rejects(self):
        for oid in (self.b['archive_oid'], 'c' * 40):
            value = self.good.copy()
            value[0] = f'{SOURCE} {self.b["archive_oid"]} {ANCHOR} {oid}'
            self.reject(value)

    def test_wrong_archive_candidate_oid_and_missing_candidate_zero_rejects(self):
        for index, field, replacement in ((0, 1, 'c' * 40), (1, 3, 'c' * 40), (1, 3, ZERO),
                                           (1, 1, 'b' * 40), (0, 1, ZERO)):
            value = self.good.copy()
            parts = value[index].split(' ')
            parts[field] = replacement
            value[index] = ' '.join(parts)
            self.reject(value)

    def test_unexpected_source_ref_or_delete_marker_rejects(self):
        for index, replacement in ((0, 'refs/heads/main'), (1, 'delete'), (1, '(null)'),
                                    (1, 'refs/heads/frozen/topic-000')):
            value = self.good.copy()
            parts = value[index].split(' ')
            parts[0] = replacement
            value[index] = ' '.join(parts)
            self.reject(value)

    def test_remote_name_or_url_mismatch_rejects(self):
        for kwargs in ({'remote_name': 'origin'}, {'remote_url': 'https://example.invalid/repo.git'},
                       {'remote_name': 'https://github.com/other/repo.git'}):
            with self.assertRaises(ValueError):
                self.validate(encoded(self.good), **kwargs)

    def test_malformed_rows_strict_ascii_control_and_whitespace_rejects(self):
        raw = encoded(self.good)
        variants = [raw.replace(b' ', b'\t', 1), raw.replace(b' ', b'  ', 1),
                    b' ' + raw, raw + b'\n', raw.replace(b'\n', b'\r\n'),
                    raw.replace(b'refs/', b'refs/\x00', 1), raw.replace(b'refs/', b'refs/\x7f', 1),
                    raw.replace(b'refs/', b'refs/\xff', 1), raw.replace(b'refs/', b'refs/\xc2\xa0', 1),
                    raw.replace(b'\n', b'\v', 1), raw.replace(b'\n', b'\f', 1),
                    raw.replace(b'a' * 40, b'A' * 40, 1),
                    raw.replace(b'a' * 40, b'a' * 39, 1),
                    raw.replace(b'a' * 40, b'a' * 41, 1),
                    encoded([' '.join(reversed(self.good[0].split(' '))), *self.good[1:]]),
                    encoded([self.good[0] + ' extra', *self.good[1:]])]
        for number, variant in enumerate(variants):
            with self.subTest(case=number):
                self.reject(variant)

    def test_oversized_input_rejects(self):
        self.reject(encoded(self.good) + b'X' * (1024 * 1024))

    def test_malformed_or_duplicate_binding_rejects(self):
        changes = [('archive_oid', 'not-an-oid'), ('anchor', 'refs/heads/main'),
                   ('local_ref', 'refs/heads/main'), ('delete_refs', [])]
        for key, value in changes:
            b = copy.deepcopy(self.b)
            b[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(encoded(self.good), b)
        b = copy.deepcopy(self.b)
        b['delete_refs'].append(copy.deepcopy(b['delete_refs'][0]))
        with self.assertRaises(ValueError):
            self.validate(encoded(self.good), b)


class HookAndCommandTests(unittest.TestCase):
    def setUp(self):
        self.fixture = GitFixture(22)
        self.addCleanup(self.fixture.close)
        self.b = fixture_binding(self.fixture)
        self.guard = self.fixture.hooks / 'pre-push'
        self.guard.write_bytes(GUARD_PATH.read_bytes())
        self.guard.chmod(0o700)
        self.digest = hashlib.sha256(self.guard.read_bytes()).hexdigest()

    def test_reviewed_hook_accepts(self):
        T.validate_hook(self.fixture.hooks, self.digest)

    def test_missing_nonexecutable_modified_symlink_directory_hook_reject(self):
        initial = self.guard.read_bytes()
        for case in ('missing', 'nonexecutable', 'modified', 'symlink', 'directory'):
            if self.guard.is_symlink() or self.guard.is_file():
                self.guard.unlink()
            elif self.guard.exists():
                self.guard.rmdir()
            self.guard.write_bytes(initial)
            self.guard.chmod(0o700)
            if case == 'missing':
                self.guard.unlink()
            elif case == 'nonexecutable':
                self.guard.chmod(0o600)
            elif case == 'modified':
                self.guard.write_bytes(initial + b'\n# changed\n')
            elif case == 'symlink':
                self.guard.unlink()
                self.guard.symlink_to(GUARD_PATH)
            else:
                self.guard.unlink()
                self.guard.mkdir()
            with self.subTest(case=case), self.assertRaises(Exception):
                T.validate_hook(self.fixture.hooks, self.digest)

    def test_wrong_interpreter_or_additional_hook_rejects(self):
        self.guard.write_bytes(b'#!/bin/sh\nexit 0\n')
        self.guard.chmod(0o700)
        with self.assertRaises(T.Stop):
            T.validate_hook(self.fixture.hooks, hashlib.sha256(self.guard.read_bytes()).hexdigest())
        self.guard.write_bytes(GUARD_PATH.read_bytes())
        self.guard.chmod(0o700)
        (self.fixture.hooks / 'post-push').write_text('#!/bin/sh\nexit 0\n')
        with self.assertRaises(T.Stop):
            T.validate_hook(self.fixture.hooks, self.digest)

    def test_exact_command_targets_and_explicit_leases(self):
        b = binding()
        command = T.build_command(self.fixture.local, self.fixture.hooks, b)
        self.assertIn('--atomic', command)
        self.assertIn('--porcelain', command)
        self.assertIn(f"--force-with-lease={ANCHOR}:", command)
        for r in b['delete_refs']:
            self.assertIn(f"--force-with-lease={r['name']}:{r['before_oid']}", command)
            self.assertIn(':' + r['name'], command)
        self.assertIn(f'{SOURCE}:{ANCHOR}', command)
        self.assertEqual(sum(arg == f'{SOURCE}:{ANCHOR}' for arg in command), 1)
        forbidden = ('--force', '--mirror', '--all', '--prune', '--no-verify', '--follow-tags', '--recurse-submodules')
        for option in forbidden:
            self.assertNotIn(option, command)
        self.assertFalse(any(arg.startswith('+') for arg in command))
        self.assertFalse(any('*' in arg for arg in command))
        self.assertIn('protocol.allow=never', command)
        self.assertIn('protocol.https.allow=always', command)
        self.assertNotIn('protocol.file.allow=always', command)
        hooks_settings = [arg for arg in command if arg.startswith('core.hooksPath=')]
        self.assertEqual(hooks_settings, [f'core.hooksPath={self.fixture.hooks}'])
        targets = {arg[1:] for arg in command if arg.startswith(':refs/')}
        self.assertEqual(targets, {r['name'] for r in b['delete_refs']})


class PorcelainTests(unittest.TestCase):
    def test_exact_success_accepted_for_both_counts(self):
        for count in (57, 22):
            b = binding(count)
            T.parse_porcelain(successful_porcelain(b), b)

    def test_missing_duplicate_unknown_rejected_or_uptodate_rows(self):
        b = binding(22)
        raw = successful_porcelain(b)
        first = f"-\t:{b['delete_refs'][0]['name']}\t[deleted]\n".encode()
        anchor = f'*\t{SOURCE}:{ANCHOR}\t[new tag]\n'.encode()
        for value in (raw.replace(first, b'', 1), raw.replace(anchor, b'', 1),
                      raw.replace(first, first + first, 1),
                      raw.replace(first, first.replace(b'-\t', b'!\t'), 1),
                      raw.replace(anchor, anchor.replace(b'*\t', b'=\t'), 1),
                      raw.replace(first, first.replace(b'[deleted]', b'[rejected]'), 1),
                      raw.replace(b'Done\n', b'?\tmalformed\tunknown\nDone\n'),
                      raw + b'\xff'):
            with self.subTest(output=value[-100:]), self.assertRaises(Exception):
                T.parse_porcelain(value, b)




class NativeGitIntegrationTests(unittest.TestCase):
    def prepare(self, count=57):
        f = GitFixture(count)
        self.addCleanup(f.close)
        b = fixture_binding(f)
        hook = f.hooks / 'pre-push'
        hook.write_bytes(GUARD_PATH.read_bytes())
        hook.chmod(0o700)
        manifest = f.root / 'binding.json'
        payload = json.dumps(b, sort_keys=True).encode('ascii')
        manifest.write_bytes(payload)
        f.receipt = f.root / 'hook-receipt.json'
        f.binding = b
        f.env = {'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent',
                 'LANG': 'C', 'LC_ALL': 'C', 'GIT_CONFIG_NOSYSTEM': '1',
                 'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_TERMINAL_PROMPT': '0',
                 'KWS_GIT_BINDING': str(manifest),
                 'KWS_GIT_BINDING_SHA256': hashlib.sha256(payload).hexdigest(),
                 'KWS_GIT_HOOK_RECEIPT': str(f.receipt)}
        f.pushes = 0
        return f

    def push(self, f, extra=()):
        command = T.build_command(f.local, f.hooks, f.binding)
        # Only the test harness enables file transport to its disposable fixture.
        # The production builder is asserted to remain HTTPS-only above.
        command[1:1] = ['-c', 'protocol.file.allow=always']
        if extra:
            position = command.index('push') + 1
            command[position:position] = list(extra)
        f.pushes += 1
        return subprocess.run(command, env=f.env, capture_output=True, check=False, timeout=20)

    def assert_blocked(self, f, before, result):
        self.assertNotEqual(result.returncode, 0, result.stdout.decode(errors='replace'))
        self.assertEqual(f.refs(), before)
        self.assertNotIn(ANCHOR, f.refs()) if ANCHOR not in before else None
        self.assertEqual(f.pushes, 1)

    def test_exact_57_plus_1_and_22_plus_1_transactions(self):
        for count in (57, 22):
            with self.subTest(count=count):
                f = self.prepare(count)
                result = self.push(f)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                T.parse_porcelain(result.stdout, f.binding)
                self.assertEqual(f.refs(), {**f.preserved, ANCHOR: f.archive})
                self.assertEqual(json.loads(f.receipt.read_text()),
                                 {'anchor_old_oid': ZERO, 'outgoing_rows': count + 1})
                self.assertEqual(f.pushes, 1)
                for label, tag in (('original', ORIGINAL), ('new-anchor', ANCHOR)):
                    restored = f.root / (label + '-restore.git')
                    run_git('init', '--bare', restored)
                    run_git('-C', restored, '-c', 'protocol.file.allow=always', 'fetch', '--no-tags',
                            f.remote, tag + ':refs/recovery/archive')
                    for oid in set(f.candidates.values()):
                        run_git('-C', restored, 'cat-file', '-e', oid + '^{commit}')
                    fetched = run_git('-C', restored, 'rev-parse', 'refs/recovery/archive').stdout.strip().decode()
                    self.assertEqual(fetched, f.archive)

    def test_existing_anchor_same_target_cannot_omit_create(self):
        f = self.prepare()
        f.set_ref(ANCHOR, f.archive)
        before = f.refs()
        result = self.push(f)
        self.assert_blocked(f, before, result)
        self.assertFalse(f.receipt.exists())
        self.assertIn(b'Pinned pre-push guard rejected', result.stderr)

    def test_existing_anchor_other_target_blocks(self):
        f = self.prepare()
        f.set_ref(ANCHOR, f.other)
        self.assert_blocked(f, f.refs(), self.push(f))

    def test_anchor_appears_after_preliminary_read_before_advertisement(self):
        for same in (True, False):
            with self.subTest(same_target=same):
                f = self.prepare(22)
                self.assertNotIn(ANCHOR, f.refs())
                f.set_ref(ANCHOR, f.archive if same else f.other)
                self.assert_blocked(f, f.refs(), self.push(f))

    def test_changed_or_missing_candidate_before_advertisement(self):
        for value in ('changed', 'missing'):
            with self.subTest(candidate=value):
                f = self.prepare(22)
                ref = next(iter(f.candidates))
                f.set_ref(ref, f.other if value == 'changed' else None)
                self.assert_blocked(f, f.refs(), self.push(f))

    def test_anchor_appears_after_advertisement_same_or_other_oid(self):
        for same in (True, False):
            with self.subTest(same_target=same):
                f = self.prepare(22)
                oid = f.archive if same else f.other
                f.race_after_advertisement(ANCHOR, oid)
                result = self.push(f)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(f.refs(), {**f.initial, ANCHOR: oid})
                # The guard did run with the earlier ZERO advertisement; receive-pack
                # must reject the later racing write under old-ID CAS atomically.
                self.assertTrue(f.receipt.exists())
                self.assertEqual(f.pushes, 1)

    def test_candidate_changes_after_advertisement_rolls_back_every_update(self):
        f = self.prepare()
        ref = next(iter(f.candidates))
        f.race_after_advertisement(ref, f.other)
        result = self.push(f)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(f.refs(), {**f.initial, ref: f.other})
        self.assertTrue(f.receipt.exists())
        self.assertEqual(f.pushes, 1)

    def test_failing_prepush_hook_blocks_entire_transaction(self):
        f = self.prepare(22)
        (f.hooks / 'pre-push').write_text('#!/usr/bin/python3\nraise SystemExit(1)\n')
        (f.hooks / 'pre-push').chmod(0o700)
        self.assert_blocked(f, f.initial, self.push(f))
        self.assertFalse(f.receipt.exists())

    def test_receiver_hook_rejection_rolls_back_entire_transaction(self):
        f = self.prepare()
        f.set_receiver_hook('pre-receive', 'cat >/dev/null\nexit 1')
        self.assert_blocked(f, f.initial, self.push(f))

    def test_receiver_update_hook_rejection_rolls_back_entire_transaction(self):
        f = self.prepare(22)
        f.set_receiver_hook('update', 'exit 1')
        self.assert_blocked(f, f.initial, self.push(f))

    def test_missing_atomic_capability_does_not_fallback(self):
        f = self.prepare(22)
        run_git('-C', f.remote, 'config', 'receive.advertiseAtomic', 'false')
        result = self.push(f)
        self.assert_blocked(f, f.initial, result)
        self.assertIn(b'does not support --atomic push', result.stderr)

    def test_receiver_denies_deletions_without_partial_anchor_creation(self):
        f = self.prepare(22)
        run_git('-C', f.remote, 'config', 'receive.denyDeletes', 'true')
        self.assert_blocked(f, f.initial, self.push(f))

    def test_missing_delete_refs_advertisement_does_not_fallback(self):
        f = self.prepare(22)
        wrapper = f.root / 'without-delete-refs'
        # Offline server fixture only. It launches native receive-pack unchanged,
        # removes one advertised capability and forwards all remaining bytes.
        # No production transport implements or interprets this protocol.
        wrapper.write_text("""#!/usr/bin/python3
import subprocess, sys, threading
p = subprocess.Popen(['git-receive-pack', *sys.argv[1:]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def forward_stdin():
    try:
        while True:
            data = sys.stdin.buffer.read1(65536)
            if not data:
                p.stdin.close()
                return
            p.stdin.write(data)
            p.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
threading.Thread(target=forward_stdin, daemon=True).start()
header = p.stdout.read(4)
size = int(header, 16)
data = p.stdout.read(size - 4)
assert b'\\x00' in data and b'delete-refs' in data
head, capabilities = data.split(b'\\x00', 1)
capabilities = b' '.join(c for c in capabilities.split(b' ') if c != b'delete-refs')
data = head + b'\\x00' + capabilities
sys.stdout.buffer.write(('%04x' % (len(data) + 4)).encode('ascii') + data)
sys.stdout.buffer.flush()
while True:
    data = p.stdout.read1(65536)
    if not data:
        break
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
sys.exit(p.wait())
""", encoding='ascii')
        wrapper.chmod(0o700)
        result = self.push(f, ['--receive-pack=' + str(wrapper)])
        self.assert_blocked(f, f.initial, result)
        self.assertTrue(b'does not support deleting refs' in result.stderr or
                        b'remote does not support deleting refs' in result.stdout,
                        (result.stdout + result.stderr).decode())


class TransportSingleAttemptTests(unittest.TestCase):
    def setUp(self):
        self.f = GitFixture(22)
        self.addCleanup(self.f.close)
        self.repo = {'full_name': 'jiying2007/kws-data', 'archive_commit_oid': self.f.archive,
                     'delete_refs': fixture_binding(self.f)['delete_refs']}
        self.askpass = ROOT / 'git_atomic_askpass_20261008.py'
        if not self.askpass.exists():
            self.askpass = ROOT.parent / 'tools/git_atomic_askpass_20261008.py'
        self.transport = T.GitAtomicTransport(
            GUARD_PATH, hashlib.sha256(GUARD_PATH.read_bytes()).hexdigest(),
            self.askpass, hashlib.sha256(self.askpass.read_bytes()).hexdigest())
        self.calls = []
        self.token = 'OFFLINE-SYNTHETIC-NOT-A-CREDENTIAL'

    def fake_runner(self, outcome):
        def run(argv, **kwargs):
            self.calls.append((argv, kwargs))
            if 'config' in argv:
                self.assertNotIn('KWS_GIT_TOKEN', kwargs['env'])
                return subprocess.CompletedProcess(argv, 0,
                    b'core.repositoryformatversion=0\ncore.filemode=true\ncore.bare=true\n', b'')
            if 'update-ref' in argv:
                self.assertNotIn('KWS_GIT_TOKEN', kwargs['env'])
                return subprocess.CompletedProcess(argv, 0, b'', b'')
            self.assertIn('push', argv)
            env = kwargs['env']
            self.assertEqual(env['KWS_GIT_TOKEN'], self.token)
            self.assertFalse(any(self.token in x for x in argv))
            self.assertEqual([x for x in argv if x.startswith('core.hooksPath=')],
                             ['core.hooksPath=' + str(Path(env['KWS_GIT_HOOK_RECEIPT']).parent / 'hooks')])
            self.assertNotIn('--no-verify', argv)
            for key in ('GIT_CONFIG_COUNT', 'GIT_CONFIG_KEY_0', 'GIT_CONFIG_VALUE_0',
                        'GIT_TRACE', 'GIT_TRACE_CURL', 'GIT_CURL_VERBOSE', 'HTTPS_PROXY'):
                self.assertNotIn(key, env)
            if isinstance(outcome, BaseException):
                raise outcome
            b = json.loads(Path(env['KWS_GIT_BINDING']).read_text())
            if outcome != 'no-receipt':
                Path(env['KWS_GIT_HOOK_RECEIPT']).write_text(json.dumps(
                    {'anchor_old_oid': ZERO, 'outgoing_rows': 23}))
            if outcome == 'bad-porcelain':
                return subprocess.CompletedProcess(argv, 0, b'partial response', b'')
            if outcome == 'failure':
                return subprocess.CompletedProcess(argv, 1, b'', b'receiver rejected')
            return subprocess.CompletedProcess(argv, 0, successful_porcelain(b), b'')
        return run

    @contextmanager
    def patched(self, outcome):
        runner = self.fake_runner(outcome)
        exception_runner = self.fake_runner('success')
        processes = []
        def popen(argv, **kwargs):
            self.assertTrue(kwargs.get('start_new_session'))
            self.assertEqual(kwargs['stdout'], subprocess.PIPE)
            self.assertEqual(kwargs['stderr'], subprocess.PIPE)
            completed = (exception_runner if isinstance(outcome, BaseException) else runner)(argv, **kwargs)
            class Process:
                pid = 987654321
                returncode = completed.returncode
                communicates = 0
                def communicate(process, timeout=None):
                    process.communicates += 1
                    if isinstance(outcome, BaseException) and process.communicates == 1:
                        raise outcome
                    return completed.stdout, completed.stderr
            process = Process()
            processes.append(process)
            return process
        with patch.object(T.subprocess, 'run', side_effect=runner), \
             patch.object(T.subprocess, 'Popen', side_effect=popen), \
             patch.object(T.os, 'killpg') as kill:
            yield kill, processes

    def push_calls(self):
        return [c for c in self.calls if 'push' in c[0]]

    def test_success_once_and_no_same_transport_retry(self):
        with self.patched('success'):
            result = self.transport.push(self.f.local, self.repo, self.token)
            self.assertEqual(result['deleted_refs'], 22)
            with self.assertRaises(T.Stop):
                self.transport.push(self.f.local, self.repo, self.token)
        self.assertEqual(len(self.push_calls()), 1)

    def test_timeout_disconnect_interruption_bad_response_no_receipt_never_retry(self):
        cases = [subprocess.TimeoutExpired('git', 180), OSError('simulated disconnect'),
                 subprocess.SubprocessError('interrupted'), 'bad-porcelain', 'no-receipt', 'failure']
        for outcome in cases:
            with self.subTest(outcome=str(outcome)):
                self.calls.clear()
                self.transport.invoked = False
                with self.patched(outcome) as (kill, processes):
                    with self.assertRaises(T.Stop) as caught:
                        self.transport.push(self.f.local, self.repo, self.token)
                    self.assertIn('reconcile', str(caught.exception))
                    with self.assertRaises(T.Stop):
                        self.transport.push(self.f.local, self.repo, self.token)
                self.assertEqual(len(self.push_calls()), 1)
                if isinstance(outcome, BaseException):
                    kill.assert_called_once()
                    self.assertEqual(processes[0].communicates, 2)

    def test_keyboard_interrupt_kills_process_group_and_consumes_transport(self):
        with self.patched(KeyboardInterrupt()) as (kill, processes):
            with self.assertRaises(KeyboardInterrupt):
                self.transport.push(self.f.local, self.repo, self.token)
            with self.assertRaises(T.Stop):
                self.transport.push(self.f.local, self.repo, self.token)
        kill.assert_called_once()
        self.assertEqual(processes[0].communicates, 2)
        self.assertEqual(len(self.push_calls()), 1)

    def test_failed_process_group_cleanup_remains_unknown(self):
        with self.patched(subprocess.TimeoutExpired('git', 180)) as (kill, processes):
            kill.side_effect = PermissionError('fixture group cleanup failed')
            with self.assertRaises(T.Stop):
                self.transport.push(self.f.local, self.repo, self.token)
            self.assertTrue(self.transport.invoked)
            self.assertFalse(self.transport.acknowledged)
            self.assertFalse(self.transport.cleanup_complete)
            self.assertEqual(processes[0].communicates, 1)
            with self.assertRaises(T.Stop):
                self.transport.push(self.f.local, self.repo, self.token)
        self.assertEqual(len(self.push_calls()), 1)

    def test_inherited_hook_override_and_noverify_environment_are_removed(self):
        dirty = {'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.hooksPath',
                 'GIT_CONFIG_VALUE_0': '/dev/null', 'GIT_TRACE': '1',
                 'GIT_TRACE_CURL': '1', 'GIT_CURL_VERBOSE': '1', 'HTTPS_PROXY': 'http://bad.invalid'}
        with patch.dict(os.environ, dirty), self.patched('success'):
            self.transport.push(self.f.local, self.repo, self.token)
        self.assertEqual(len(self.push_calls()), 1)

    def test_bad_hook_pin_or_askpass_pin_stops_before_any_push(self):
        for attribute in ('hook_sha256', 'askpass_sha256'):
            with self.subTest(attribute=attribute):
                previous = getattr(self.transport, attribute)
                setattr(self.transport, attribute, 'f' * 64)
                self.calls.clear()
                with self.patched('success'):
                    with self.assertRaises(T.Stop):
                        self.transport.push(self.f.local, self.repo, self.token)
                self.assertEqual(self.push_calls(), [])
                setattr(self.transport, attribute, previous)

    def test_missing_overridden_hookspath_or_noverify_prevents_push(self):
        original = T.build_command
        def altered(kind):
            def build(bare, hooks, binding):
                argv = original(bare, hooks, binding)
                index = next(n for n, arg in enumerate(argv) if arg.startswith('core.hooksPath='))
                if kind == 'missing':
                    del argv[index - 1:index + 1]
                elif kind == 'overridden':
                    argv[index] = 'core.hooksPath=/dev/null'
                elif kind == 'duplicated':
                    argv[1:1] = ['-c', 'core.hooksPath=/dev/null']
                else:
                    argv.insert(argv.index('push') + 1, '--no-verify')
                return argv
            return build
        for kind in ('missing', 'overridden', 'duplicated', 'no-verify'):
            self.calls.clear()
            self.transport.invoked = False
            with self.subTest(kind=kind), patch.object(T, 'build_command', side_effect=altered(kind)), \
                 self.patched('success'), patch.object(T.subprocess, 'Popen') as popen:
                with self.assertRaises(T.Stop):
                    self.transport.push(self.f.local, self.repo, self.token)
                popen.assert_not_called()
            self.assertEqual(self.push_calls(), [])

    def test_process_creation_failure_consumes_attempt_and_sanitizes_error(self):
        with self.patched('success'), patch.object(T.subprocess, 'Popen', side_effect=OSError(self.token)) as popen:
            with self.assertRaises(T.Stop) as caught:
                self.transport.push(self.f.local, self.repo, self.token)
            self.assertNotIn(self.token, str(caught.exception))
            self.assertIn('reconcile', str(caught.exception))
            with self.assertRaises(T.Stop):
                self.transport.push(self.f.local, self.repo, self.token)
            self.assertEqual(popen.call_count, 1)

    def test_wrong_repo_count_missing_token_or_dirty_config_prevents_push(self):
        changes = [('full_name', 'jiying2007/other'), ('delete_refs', self.repo['delete_refs'][:-1])]
        for key, value in changes:
            repo = {**self.repo, key: value}
            with self.subTest(key=key), self.patched('success'):
                with self.assertRaises(T.Stop):
                    self.transport.push(self.f.local, repo, self.token)
            self.assertEqual(self.push_calls(), [])
        with patch.object(T.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'core.hooksPath=/dev/null\n', b'')) as runner:
            with self.assertRaises(T.Stop):
                self.transport.push(self.f.local, self.repo, self.token)
            self.assertFalse(any('push' in call.args[0] for call in runner.call_args_list))
        with patch.object(T.subprocess, 'run') as runner:
            with self.assertRaises(T.Stop):
                self.transport.push(self.f.local, self.repo, '')
            runner.assert_not_called()




class AskpassTests(unittest.TestCase):
    def setUp(self):
        self.f = GitFixture(22)
        self.addCleanup(self.f.close)
        source = ROOT / 'git_atomic_askpass_20261008.py'
        if not source.exists():
            source = ROOT.parent / 'tools/git_atomic_askpass_20261008.py'
        self.askpass = self.f.root / 'askpass'
        self.askpass.write_bytes(source.read_bytes())
        self.askpass.chmod(0o700)
        self.token = 'OFFLINE-SYNTHETIC-NOT-A-CREDENTIAL'
        self.env = {'KWS_GIT_REPOSITORY': 'jiying2007/kws-data', 'KWS_GIT_TOKEN': self.token,
                    'GIT_ASKPASS': str(self.askpass), 'GIT_TERMINAL_PROMPT': '0', 'LC_ALL': 'C'}

    def invoke(self, prompt, env=None):
        return subprocess.run([str(self.askpass), prompt], env=self.env if env is None else env,
                              capture_output=True, check=False, timeout=10)

    def test_exact_own_repository_username_and_password_prompts(self):
        for repo in ('jiying2007/kws-pipeline', 'jiying2007/kws-data'):
            env = {**self.env, 'KWS_GIT_REPOSITORY': repo}
            username = self.invoke(f"Username for 'https://github.com/{repo}.git': ", env)
            password = self.invoke(f"Password for 'https://x-access-token@github.com/{repo}.git': ", env)
            self.assertEqual(username.returncode, 0)
            self.assertEqual(username.stdout, b'x-access-token\n')
            self.assertEqual(password.returncode, 0)
            self.assertTrue(password.stdout == (self.token + '\n').encode())
            self.assertEqual(password.stderr, b'')

    def test_foreign_host_repo_user_protocol_and_unexpected_prompts_rejected(self):
        prompts = ["Password for 'https://x-access-token@github.com/jiying2007/kws-pipeline.git': ",
                   "Password for 'https://x-access-token@example.invalid/jiying2007/kws-data.git': ",
                   "Password for 'http://x-access-token@github.com/jiying2007/kws-data.git': ",
                   "Password for 'https://other@github.com/jiying2007/kws-data.git': ",
                   "Password for 'https://github.com/jiying2007/kws-data.git': ",
                   "Password for 'https://x-access-token@github.com': ",
                   "Password for 'https://x-access-token@github.com/jiying2007/kws-data': ",
                   "Username for 'https://github.com/jiying2007/other.git': ", 'password', '',
                   "Password for 'https://x-access-token@github.com/jiying2007/kws-data.git': \n"]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = self.invoke(prompt)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b'')

    def test_bad_repository_missing_or_multiline_token_rejected(self):
        prompt = "Password for 'https://x-access-token@github.com/jiying2007/kws-data.git': "
        for env in ({**self.env, 'KWS_GIT_REPOSITORY': 'jiying2007/other'},
                    {**self.env, 'KWS_GIT_TOKEN': ''}, {**self.env, 'KWS_GIT_TOKEN': 'bad\nvalue'},
                    {**self.env, 'KWS_GIT_TOKEN': 'bad\rvalue'}):
            result = self.invoke(prompt, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b'')

    def test_real_git_credential_fill_own_repo_prompt_contract_offline(self):
        # `credential fill` exercises installed Git's real askpass wording only.
        # This command does not open any network connection or save credentials.
        result = run_git('-c', 'credential.helper=', '-c', 'credential.useHttpPath=true',
                         'credential', 'fill', env=self.env,
                         input=b'protocol=https\nhost=github.com\npath=jiying2007/kws-data.git\n\n', check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        fields = dict(line.split(b'=', 1) for line in result.stdout.splitlines())
        self.assertEqual(fields[b'username'], b'x-access-token')
        self.assertTrue(fields[b'password'] == self.token.encode())
        self.assertEqual(result.stderr, b'')

    def test_real_git_credential_fill_foreign_repo_rejected_offline(self):
        result = run_git('-c', 'credential.helper=', '-c', 'credential.useHttpPath=true',
                         'credential', 'fill', env=self.env,
                         input=b'protocol=https\nhost=github.com\npath=jiying2007/other.git\n\n', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(self.token.encode(), result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
