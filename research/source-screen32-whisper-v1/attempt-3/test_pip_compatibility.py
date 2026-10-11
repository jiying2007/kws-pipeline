"""Offline tests only: synthetic help and local text, never invoke/import pip."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import json

spec = importlib.util.spec_from_file_location('compat', Path(__file__).with_name('pip_compatibility.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class PolicyTests(unittest.TestCase):
    def test_source_manifest_matches_enforced_hashes(self):
        path = Path(__file__).parents[1] / 'pip-compatibility' / 'upstream-source-manifest.json'
        manifest = json.loads(path.read_bytes())
        self.assertEqual(manifest['tag'], c.AUDITED_VERSION)
        self.assertEqual(manifest['commit'], c.UPSTREAM_COMMIT)
        self.assertEqual(len(manifest['files']), 11)
        mapping = {}
        for row in manifest['files']:
            self.assertTrue(row['path'].startswith('src/pip/'))
            relative = row['path'][len('src/pip/'):]
            self.assertNotIn(relative, mapping)
            mapping[relative] = row['sha256']
            self.assertRegex(row['git_blob_sha1'], r'^[0-9a-f]{40}$')
            self.assertEqual(row['url'], 'https://github.com/pypa/pip/blob/' +
                             c.UPSTREAM_COMMIT + '/' + row['path'])
        self.assertEqual(mapping, c.SOURCE_SHA256)

    def help(self, flags):
        return ('\n'.join('  '+x+' <value>  fixture' for x in flags)).encode()

    def test_exact_audited_version_omits_unavailable_resume(self):
        with patch.object(c, '_source_matches', return_value=True):
            result = c.select_contract('25.0.1', 0, self.help(c.OPTIONS[1:]), '/fixture')
        self.assertEqual(result['retry_argv'], ['--retries', '0'])
        self.assertEqual(result['evidence']['missing_options'], ['--resume-retries'])
        self.assertTrue(result['evidence']['reviewed_source_matches'])

    def test_exact_version_requires_matching_sources(self):
        with patch.object(c, '_source_matches', return_value=False):
            with self.assertRaises(c.CompatibilityFailure) as error:
                c.select_contract('25.0.1', 0, self.help(c.OPTIONS[1:]), '/fixture')
        self.assertFalse(error.exception.evidence['reviewed_source_matches'])

    def test_other_versions_do_not_inherit_exemption(self):
        for version in ('24.3.1', '25.0', '25.1', '25.2', '99.0'):
            with self.assertRaises(c.CompatibilityFailure) as error:
                c.select_contract(version, 0, self.help(c.OPTIONS[1:]))
            self.assertEqual(error.exception.evidence['missing_options'], ['--resume-retries'])
            result = c.select_contract(version, 0, self.help(c.OPTIONS))
            self.assertEqual(result['retry_argv'], ['--retries', '0', '--resume-retries', '0'])

    def test_missing_each_required_flag_is_named(self):
        for option in c.OPTIONS:
            flags = [x for x in c.OPTIONS if x != option]
            with self.assertRaises(c.CompatibilityFailure) as error:
                c.select_contract('99.0', 0, self.help(flags))
            self.assertEqual(error.exception.evidence['missing_options'], [option])

    def test_failed_help_exit_is_retained(self):
        with self.assertRaises(c.CompatibilityFailure) as error:
            c.select_contract('99.0', 2, self.help(c.OPTIONS))
        self.assertEqual(error.exception.evidence['help_exit_code'], 2)
        self.assertEqual(error.exception.evidence['missing_options'], [])

    def test_malformed_bounded_inputs_fail_closed(self):
        for version, code, text in [('25.0.1+local', 0, b''), ('99.0', True, b''),
                                    ('99.0', 0, b'x'*(128*1024+1)), ('99.0', 0, 'raw')]:
            with self.assertRaises(c.CompatibilityFailure):
                c.select_contract(version, code, text)

    def test_option_substrings_do_not_pass(self):
        flags = [x if x != '--retries' else '--retries-extra' for x in c.OPTIONS]
        with self.assertRaises(c.CompatibilityFailure) as error:
            c.select_contract('99.0', 0, self.help(flags))
        self.assertEqual(error.exception.evidence['missing_options'], ['--retries'])

    def test_file_hash_missing_changed_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file = root/'source.py'; file.write_bytes(b'a\n')
            with patch.object(c, 'SOURCE_SHA256', {'source.py': hashlib.sha256(b'a\n').hexdigest()}):
                self.assertTrue(c._source_matches(root))
                file.write_bytes(b'b\n'); self.assertFalse(c._source_matches(root))
                file.unlink(); self.assertFalse(c._source_matches(root))
                other = root/'other.py'; other.write_bytes(b'a\n'); file.symlink_to(other)
                self.assertFalse(c._source_matches(root))

if __name__ == '__main__':
    unittest.main()
