"""Exercise shared-Python startup and preserved loader guards without models."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import research_python_entry as entry
import run_research_fsmn_stream_arm_build as full_build


class PythonEntryTests(unittest.TestCase):
    def test_environment_without_loader_override_is_unchanged(self):
        env = {'PATH': '/usr/bin:/bin', 'UNCHANGED': 'value'}
        before = dict(env)
        self.assertIsNone(entry.clear_interpreter_library_path(env))
        self.assertEqual(env, before)

    def test_only_current_interpreter_library_path_is_removed(self):
        lib = str(Path(sys.base_prefix) / 'lib')
        env = {'LD_LIBRARY_PATH': lib, 'UNCHANGED': 'value'}
        self.assertEqual(entry.clear_interpreter_library_path(env), lib)
        self.assertEqual(env, {'UNCHANGED': 'value'})

    def test_preload_and_audit_are_rejected_without_mutation(self):
        for name in ('LD_PRELOAD', 'LD_AUDIT'):
            for value in ('', '/untrusted/loader.so'):
                with self.subTest(name=name, value=value):
                    env = {'LD_LIBRARY_PATH': str(Path(sys.base_prefix) / 'lib'), name: value}
                    before = dict(env)
                    with self.assertRaisesRegex(RuntimeError, 'loader overrides'):
                        entry.clear_interpreter_library_path(env)
                    self.assertEqual(env, before)

    def test_foreign_empty_and_multiple_library_paths_are_rejected(self):
        lib = str(Path(sys.base_prefix) / 'lib')
        for value in ('', '.', 'lib', '/tmp', lib + ':', ':' + lib,
                      lib + ':/tmp', '/tmp:' + lib, lib + ':' + lib):
            with self.subTest(value=value):
                env = {'LD_LIBRARY_PATH': value}
                with self.assertRaisesRegex(RuntimeError, 'loader overrides'):
                    entry.clear_interpreter_library_path(env)
                self.assertEqual(env, {'LD_LIBRARY_PATH': value})

    def test_library_must_be_the_installed_shared_interpreter(self):
        for soname in (None, 'libpython-missing.so', '../libpython3.12.so'):
            with self.subTest(soname=soname), patch.object(entry.sysconfig, 'get_config_var', return_value=soname):
                env = {'LD_LIBRARY_PATH': str(Path(sys.base_prefix) / 'lib')}
                with self.assertRaisesRegex(RuntimeError, 'loader overrides'):
                    entry.clear_interpreter_library_path(env)
                self.assertIn('LD_LIBRARY_PATH', env)

    def test_cli_starts_shared_python_then_preserves_script_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sibling.py').write_text('VALUE = 37\n')
            script = root / 'probe.py'
            script.write_text(
                'import json, os, sys, sibling\n'
                'print(json.dumps({"loader": os.environ.get("LD_LIBRARY_PATH"), '
                '"argv": sys.argv, "sibling": sibling.VALUE, "optimize": sys.flags.optimize, '
                '"no_site": sys.flags.no_site, "no_bytecode": sys.dont_write_bytecode}))\n')
            env = dict(os.environ)
            for name in ('LD_PRELOAD', 'LD_AUDIT'):
                env.pop(name, None)
            env['LD_LIBRARY_PATH'] = str(Path(sys.base_prefix) / 'lib')
            mode = ['-' + 'O' * sys.flags.optimize] if sys.flags.optimize else []
            result = subprocess.run(
                [sys.executable, '-S', '-B', *mode, str(ROOT / 'tools/research_python_entry.py'),
                 str(script), '--fixture', 'literal argument'],
                env=env, text=True, capture_output=True, timeout=10, check=True)
            self.assertEqual(json.loads(result.stdout), {
                'loader': None, 'argv': [str(script), '--fixture', 'literal argument'],
                'sibling': 37, 'optimize': sys.flags.optimize, 'no_site': 1, 'no_bytecode': True})
            self.assertEqual(env['LD_LIBRARY_PATH'], str(Path(sys.base_prefix) / 'lib'))

    def test_preserved_driver_still_rejects_all_loader_overrides(self):
        source = ROOT / 'research/fsmn2599-stream-c/check_arm_synthetic.py'
        spec = importlib.util.spec_from_file_location('preserved_arm_guard', source)
        arm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arm)
        for name in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH'):
            with self.subTest(name=name), patch.dict(os.environ, {name: '/untrusted'}, clear=True):
                with patch.object(arm.guard, 'validate_sources') as validate:
                    with self.assertRaisesRegex(RuntimeError, 'loader overrides'):
                        arm.execute('/missing')
                    validate.assert_not_called()

    def test_active_arm_entry_reaches_missing_tool_after_python_path(self):
        env = {'LD_LIBRARY_PATH': str(Path(sys.base_prefix) / 'lib')}
        with patch.dict(os.environ, env, clear=True), patch.object(full_build.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, r'ARM hard-float gcc, g\+\+ and readelf required'):
                full_build.execute(ROOT)
            self.assertNotIn('LD_LIBRARY_PATH', os.environ)

    def test_active_arm_entry_rejects_unsafe_loader_before_source_reads(self):
        for name in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH'):
            with self.subTest(name=name), patch.dict(os.environ, {name: '/untrusted'}, clear=True):
                with self.assertRaisesRegex(RuntimeError, 'loader overrides'):
                    full_build.execute(Path('/missing-source-tree'))


if __name__ == '__main__':
    unittest.main()
