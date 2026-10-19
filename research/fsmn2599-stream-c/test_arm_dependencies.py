"""Dependency failures and durable evidence, without ARM execution."""
import json
import io
from contextlib import redirect_stderr, contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_arm_dependencies as dep


def info(needed=(), definitions=(), requirements=None):
    return {'needed': list(needed), 'interpreter': None, 'version_needs': requirements or {},
            'version_definitions': list(definitions), 'build_ids': []}


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root, self.art = self.base / 'sysroot', self.base / 'artifact'
        (self.root / 'lib').mkdir(parents=True)
        self.art.mkdir()
        self.graph = {}
        self.c = dep.Closure(self.root, self.art, lambda p: self.graph[p.name])

    def lib(self, name, metadata=None):
        p = self.root / 'lib' / name
        p.write_bytes(b'public fake ELF')
        self.graph[name] = metadata or info()
        return p

    def test_missing_and_unsafe_names(self):
        for name in ('absent.so', '../outside', '/lib/x', '.'):
            with self.assertRaises(RuntimeError): self.c.resolve(name)

    def test_ambiguous_even_if_bytes_identical(self):
        p = self.lib('a.so')
        (self.art / p.name).write_bytes(p.read_bytes())
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'): self.c.resolve(p.name)

    def test_escape_and_dangling_symlink(self):
        outside = self.base / 'outside'; outside.write_bytes(b'x')
        alias = self.root / 'lib/a.so'; alias.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, 'escape'): self.c.resolve('a.so')
        outside.unlink()
        with self.assertRaises((RuntimeError, FileNotFoundError)): self.c.resolve('a.so')

    def test_symlink_alias_same_target_and_identity_mutation(self):
        p = self.lib('real.so')
        alias = self.root / 'lib/a.so'; alias.symlink_to(p.name)
        self.assertEqual(self.c.resolve('a.so'), p)
        p.write_bytes(b'mutated')
        with self.assertRaisesRegex(RuntimeError, 'identity changed'): self.c.verify()

    def test_symlink_retargeting(self):
        a, b = self.lib('a.so'), self.lib('b.so')
        alias = self.root / 'lib/alias.so'; alias.symlink_to(a.name)
        self.c.resolve('alias.so')
        alias.unlink(); alias.symlink_to(b.name)
        with self.assertRaisesRegex(RuntimeError, 'identity changed'): self.c.verify()

    def test_dag_repeat_is_not_cycle(self):
        a = self.lib('a.so', info(['b.so', 'c.so']))
        self.lib('b.so', info(['c.so'])); self.lib('c.so')
        self.c.visit(a)
        self.assertEqual(len(self.c.nodes), 3)
        self.assertEqual(len(self.c.edges), 3)
        self.c.verify()

    def test_cycle_fails(self):
        a = self.lib('a.so', info(['b.so']))
        self.lib('b.so', info(['a.so']))
        with self.assertRaisesRegex(RuntimeError, 'cycle'): self.c.visit(a)

    def test_symbol_version_resolution(self):
        a = self.lib('a.so', info(['b.so'], requirements={'b.so': ['GLIBC_2.99']}))
        self.lib('b.so', info(definitions=['GLIBC_2.4']))
        with self.assertRaisesRegex(RuntimeError, 'symbol version'): self.c.visit(a)

    def test_object_limit(self):
        a = self.lib('a.so', info(['b.so'])); self.lib('b.so')
        with patch.object(dep, 'MAX_OBJECTS', 1):
            with self.assertRaisesRegex(RuntimeError, 'object limit'): self.c.visit(a)

    def test_new_ambiguous_candidate_fails_final_verification(self):
        p = self.lib('a.so'); self.c.resolve(p.name)
        (self.art / p.name).write_bytes(p.read_bytes())
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'): self.c.verify()

    def test_retention_overflow_keeps_fail_identity_receipt(self):
        output = self.base / 'output'; output.mkdir()
        (output / 'partial.elf').write_bytes(b'x' * 2048)
        report = {'status': 'NEW_SNAPSHOT_PASS', 'commands': []}
        with patch.object(dep, 'ARCHIVE_BYTES', 1024): dep.finalize(output, report)
        retained = json.loads((output / 'report.json').read_text())
        self.assertEqual(retained['status'], 'FAIL')
        self.assertEqual(retained['omitted_artifacts'][0]['bytes'], 2048)
        self.assertFalse((output / 'partial.elf').exists())

    def test_oversized_metadata_has_small_fail_receipt(self):
        output = self.base / 'output'; output.mkdir()
        report = {'status': 'NEW_SNAPSHOT_PASS', 'nodes': {'large': 'x' * 8192}}
        with patch.object(dep.arm, 'OUTPUT_BYTES', 4096): dep.finalize(output, report)
        self.assertEqual(report['status'], 'FAIL')
        self.assertIn('full_report_sha256', json.loads((output / 'report.json').read_text()))

    def test_parser_rejects_loader_directives_and_duplicates(self):
        for tag in ('RPATH', 'RUNPATH', 'AUDIT', 'DEPAUDIT', 'FILTER', 'AUXILIARY'):
            with self.assertRaises(RuntimeError): dep.parse_elf('('+tag+') [x]', '', '', '')
        with self.assertRaises(RuntimeError):
            dep.parse_elf('(NEEDED) [a.so]\n(NEEDED) [a.so]', '', '', '')

    def test_parser_versions(self):
        parsed = dep.parse_elf('(NEEDED) Shared library: [libc.so.6]',
                              '[Requesting program interpreter: /lib/ld-linux-armhf.so.3]',
                              'Version needs section\n File: libc.so.6 Cnt: 1\n Name: GLIBC_2.4\n',
                              'Build ID: abc123')
        self.assertEqual(parsed['version_needs'], {'libc.so.6': ['GLIBC_2.4']})
        self.assertEqual(parsed['build_ids'], ['abc123'])

    def test_atomic_report_pre_replace_failures_preserve_previous(self):
        for stage in ('write', 'fsync', 'replace'):
            with self.subTest(stage=stage):
                output = self.base / stage; output.mkdir()
                previous = output / 'report.json'; previous.write_text('{"status":"INCOMPLETE"}\n')
                before = previous.read_bytes()
                report = {'status': 'NEW_SNAPSHOT_PASS'}
                original = dep.tempfile.NamedTemporaryFile
                @contextmanager
                def partial_write(**kwargs):
                    with original(**kwargs) as stream:
                        write = stream.write
                        def fail(data):
                            write(data[:7]); stream.flush()
                            raise OSError('synthetic partial write')
                        stream.write = fail
                        yield stream
                failure = (patch.object(dep.tempfile, 'NamedTemporaryFile', partial_write)
                           if stage == 'write' else patch.object(dep.os, stage, side_effect=OSError('synthetic I/O failure')))
                with failure:
                    with self.assertRaises(OSError): dep.finalize(output, report)
                self.assertEqual(previous.read_bytes(), before)
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(sorted(p.name for p in output.iterdir()), ['report.json'])

    def test_atomic_report_directory_sync_failure_is_not_success(self):
        output = self.base / 'dirsync'; output.mkdir()
        report = {'status': 'NEW_SNAPSHOT_PASS'}
        with patch.object(dep.os, 'fsync', side_effect=[None, OSError('directory sync failed')]):
            with self.assertRaises(OSError): dep.finalize(output, report)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(json.loads((output / 'report.json').read_text())['status'], 'NEW_SNAPSHOT_PASS')
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['report.json'])

    def test_atomic_report_success_and_peak_charge(self):
        output = self.base / 'atomic'; output.mkdir()
        dep.save(output, {'status': 'INCOMPLETE'})
        old_size = (output / 'report.json').stat().st_size
        new = {'status': 'FAIL'}
        new_size = len((json.dumps(new, indent=2)+'\n').encode())
        with patch.object(dep, 'ARCHIVE_BYTES', old_size + new_size - 1):
            with self.assertRaisesRegex(RuntimeError, 'retained'): dep.save(output, new)
        self.assertEqual(json.loads((output / 'report.json').read_text())['status'], 'INCOMPLETE')
        dep.save(output, new)
        self.assertEqual(json.loads((output / 'report.json').read_text()), new)

    def test_io_failure_main_has_bounded_stderr_and_nonzero_exit(self):
        err = io.StringIO()
        with patch.object(dep, 'execute', side_effect=OSError('/private/path')), patch.object(dep.sys, 'argv', ['driver', '--execute', '--output', '/unused', '--emulator', '/unused']), redirect_stderr(err):
            self.assertEqual(dep.main(), 1)
        receipt = json.loads(err.getvalue())
        self.assertEqual(receipt['status'], 'FAIL'); self.assertFalse(receipt['qualification_allowed'])
        self.assertNotIn('/private', err.getvalue()); self.assertLess(len(err.getvalue()), 1024)

    def test_early_failure_retained_and_no_child_retry(self):
        output = self.base / 'output'
        with patch.dict(dep.os.environ, {'LD_AUDIT': 'blocked'}), patch.object(dep.arm.guard, 'run_bounded') as run:
            self.assertFalse(dep.execute(output, self.root, self.base / 'qemu'))
            run.assert_not_called()
        report = json.loads((output / 'report.json').read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['probe_attempts'], 0)
        self.assertEqual(report['historical_616'], 'MISSING_ORIGINAL_ELF_SYSROOT')
        self.assertNotIn(str(self.base), json.dumps(report))
        with self.assertRaises(FileExistsError): dep.execute(output, self.root, self.base / 'qemu')


if __name__ == '__main__':
    unittest.main()
