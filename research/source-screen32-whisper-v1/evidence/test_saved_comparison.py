#!/usr/bin/env python3
"""Recompute saved real comparison from the retained ZIP, entirely offline.

Install beside verify_fourth_archive.py, outside the comparison package's
unchanged 11-file manifest. Run with python -S -B (also supported under -O/-OO).
No archive member is extracted, imported, or executed. No model is loaded.
"""
import copy
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import unicodedata

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_fourth_archive as archive

PACKAGE = archive.RUN / 'comparison'
PACKAGE_MANIFEST_SHA256 = '7f78b924f19cc6a04bfbf9a922012d7dad19705b3ca3fe729c7730c330844718'
TERMINAL_SHA256 = '115773be9a64ee04cb0d5bb5002268528e9a4573ef094d754565187f95942a10'
VALIDATION_SHA256 = '4e40bff753892fef4cd5754f786fa35a759f6f1dbfdb59243ceacf0dbe09845e'
INPUT_SHA256 = 'e45b6620c8491b204234ddb9e506ec93b947807e43e1b7fc1cf17ecb95bee51e'
NATIVE_VALIDATION_LABEL = 'PERFORMED_SEPARATELY_BY_FROZEN_HOST_VALIDATOR_AND_VERIFIER; see validation file and its hash'
EXPECTED_FILES = {
    'README.md', 'SHA256SUMS.json', 'authentication.json', 'compare.py',
    'compare_verified_run.py', 'comparison-input.json', 'comparison.json',
    'report.zh.txt', 'test_compare.py', 'test_compare_verified_run.py',
    'verification.json',
}


def checked_package(directory):
    """Authenticate the complete public closure before importing its code."""
    manifest_raw = archive.file_bytes(directory, 'SHA256SUMS.json', 16384)
    archive.require(archive.sha(manifest_raw) == PACKAGE_MANIFEST_SHA256, 'comparison manifest identity')
    manifest = archive.read_json(manifest_raw)
    archive.require(set(manifest) == {'schema', 'files'} and manifest['schema'] == 'public-three-asr-comparison-files-v1', 'comparison manifest schema')
    archive.require(set(manifest['files']) == EXPECTED_FILES - {'SHA256SUMS.json'}, 'comparison manifest closed file set')
    entries = list(directory.iterdir())
    archive.require({entry.name for entry in entries} == EXPECTED_FILES and
                    all(not entry.is_symlink() and entry.is_file() for entry in entries), 'comparison directory closed regular-file set')
    files = {'SHA256SUMS.json': manifest_raw}
    for name, expected in manifest['files'].items():
        archive.require(set(expected) == {'bytes', 'sha256'}, 'comparison identity fields')
        archive.valid_identity(expected, 1024 * 1024)
        raw = archive.file_bytes(directory, name, 1024 * 1024)
        archive.require(archive.identity(raw) == expected, 'comparison file identity: ' + name)
        files[name] = raw
        if name.endswith('.json'):
            archive.read_json(raw)
    return files


def load_comparator():
    for name, file in [('compare', 'compare.py'), ('saved_comparison_implementation', 'compare_verified_run.py')]:
        spec = importlib.util.spec_from_file_location(name, PACKAGE / file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def recomputed_payload(comparator, data, terminal, reauthentication):
    result = comparator.detailed_compare(data, terminal)
    result['native_artifact_validation'] = NATIVE_VALIDATION_LABEL
    result['retained_reauthentication'] = reauthentication
    return result


def check_saved_payload(saved, expected):
    actual = copy.deepcopy(saved)
    archive.require('provenance' in actual, 'comparison provenance missing')
    actual.pop('provenance')
    archive.require(actual == expected, 'saved comparison differs from real-data recomputation')


class SavedComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = checked_package(PACKAGE)
        # This reader authenticates the pinned carrier and all 431 members in
        # memory; no archive Python/model/package bodies are executed.
        cls.carrier, cls.manifest, cls.members = archive.read_archive()
        cls.native = archive.verify_records(cls.members)
        archive.require(archive.sha(cls.members['terminal-outcomes.json']) == TERMINAL_SHA256, 'terminal identity')
        cls.terminal = archive.read_json(cls.members['terminal-outcomes.json'])
        archive.require(archive.sha(cls.files['comparison-input.json']) == INPUT_SHA256, 'comparison input identity')
        cls.data = archive.read_json(cls.files['comparison-input.json'])
        cls.saved = archive.read_json(cls.files['comparison.json'])
        cls.authentication = archive.read_json(cls.files['authentication.json'])
        cls.comparator = load_comparator()
        cls.reauthentication = cls.comparator.authenticate_retained(cls.data, cls.authentication, archive.ROOT)
        cls.expected = recomputed_payload(cls.comparator, cls.data, cls.terminal, cls.reauthentication)

    def copied_package(self):
        temporary = tempfile.TemporaryDirectory(prefix='saved-comparison-test-')
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / 'comparison'
        shutil.copytree(PACKAGE, destination)
        return destination

    def test_manifest_and_saved_actual_payload(self):
        self.assertEqual(set(self.files), EXPECTED_FILES)
        check_saved_payload(self.saved, self.expected)
        self.assertEqual(self.native['completed_native_eot'], 16)
        self.assertEqual(self.reauthentication['retained_files_rechecked'], 170)
        self.assertEqual(self.reauthentication['archive_members_rechecked'], 335)
        self.assertEqual(self.reauthentication['wav_pcm_identities_rechecked'], 16)
        self.assertEqual(self.reauthentication['successful_receipt_literal_bindings_rechecked'], 32)

    def test_provenance_exact_source_and_terminal_bindings(self):
        raw = archive.file_bytes(archive.RUN, 'verification-receipt.json', 256 * 1024)
        self.assertEqual(archive.sha(raw), VALIDATION_SHA256)
        validation = archive.read_json(raw)
        base = 'research/source-screen32-whisper-v1/evidence/run-38126849922'
        expected = {
            'repository': self.data['repository'],
            'retained_source_commit': self.data['source_commit'],
            'retained_reviewed_tree': self.data['reviewed_tree'],
            'source_run_urls': {model: f'https://github.com/{self.data["repository"]}/actions/runs/{run}' for model, run in self.data['source_runs'].items()},
            'comparison_input_file': base + '/comparison/comparison-input.json',
            'comparison_input_sha256': INPUT_SHA256,
            'retained_authentication_file': base + '/comparison/authentication.json',
            'retained_authentication_sha256': archive.sha(self.files['authentication.json']),
            'whisper_run_url': 'https://github.com/jiying2007/kws-pipeline/actions/runs/38126849922',
            'whisper_run_id': '38126849922',
            'whisper_head_sha': '1986acbeb790833a92f5014fccb88a931c0f3e7f',
            'whisper_executed_tree': validation['executed_tree'],
            'whisper_artifact_id': validation['artifact_id'],
            'whisper_artifact_zip_sha256': archive.sha(self.carrier),
            'whisper_terminal_sha256': TERMINAL_SHA256,
            'whisper_validation_file': base + '/verification-receipt.json',
            'whisper_validation_sha256': VALIDATION_SHA256,
            'comparison_script_sha256': archive.sha(self.files['compare_verified_run.py']),
            'inherited_comparator_sha256': archive.sha(self.files['compare.py']),
            'whisper_terminal_archive_member': 'terminal-outcomes.json',
            'whisper_terminal_archive_manifest': base + '/ARCHIVE.json',
            'path_convention': 'Repository-relative public files; terminal-outcomes.json is a logical member of the exact retained ZIP, not an expanded repository file.',
        }
        self.assertEqual(self.saved['provenance'], expected)
        self.assertEqual(validation['run_id'], 38126849922)
        self.assertEqual(validation['head_sha'], expected['whisper_head_sha'])
        self.assertEqual(validation['terminal_validation_errors'], 0)
        for row, verified in zip(self.terminal, validation['per_clip']):
            for key in ['audio_id', 'wav_sha256', 'raw_text', 'raw_output_sha256', 'attempts']:
                self.assertEqual(row[key], verified[key])
        self.assertEqual(len(self.terminal), len(validation['per_clip']))

    def test_every_literal_row_and_both_normalizations_independently(self):
        punctuation = lambda text: ''.join(c for c in text if not unicodedata.category(c).startswith('P'))
        whitespace = lambda text: ''.join(c for c in punctuation(text) if not c.isspace())
        strict_ids, secondary_ids = [], []
        self.assertEqual(len(self.saved['rows']), 16)
        for index, (saved, source, terminal) in enumerate(zip(self.saved['rows'], self.data['rows'], self.terminal), 1):
            audio_id = f'clip-{index:06d}'
            self.assertEqual(saved['audio_id'], audio_id)
            self.assertEqual(source['audio_id'], audio_id)
            self.assertEqual(terminal['audio_id'], audio_id)
            self.assertEqual(saved['wav_sha256'], source['wav_sha256'])
            self.assertEqual(saved['wav_sha256'], terminal['wav_sha256'])
            self.assertEqual(saved['pcm_sha256'], source['pcm_sha256'])
            votes = {'qwen06': source['qwen06']['raw_text'], 'sensevoice': source['sensevoice']['raw_text'], 'whisper': terminal['raw_text']}
            self.assertEqual(saved['raw_texts'], votes)
            self.assertEqual(saved['valid_literal_votes'], votes)
            self.assertEqual(saved['target_text'], source['target_text'])
            self.assertEqual(saved['normalized_valid_votes'], {model: punctuation(text) for model, text in votes.items()})
            self.assertEqual(saved['punctuation_whitespace_valid_votes'], {model: whitespace(text) for model, text in votes.items()})
            strict = len({punctuation(text) for text in votes.values()}) == 1
            secondary = len({whitespace(text) for text in votes.values()}) == 1
            self.assertEqual(saved['normalized_three_way_agreement'], strict)
            self.assertEqual(saved['punctuation_whitespace_three_way_agreement'], secondary)
            self.assertFalse(saved['literal_three_way_agreement'])
            if strict:
                strict_ids.append(audio_id)
            if secondary:
                secondary_ids.append(audio_id)
            for model, text in votes.items():
                self.assertEqual(saved['per_asr_vs_target'][model], {'literal_match': text == source['target_text'], 'normalized_match': punctuation(text) == punctuation(source['target_text'])})
        self.assertEqual(strict_ids, [f'clip-{i:06d}' for i in (1, 2, 6, 8, 9, 14)])
        self.assertEqual(secondary_ids, [f'clip-{i:06d}' for i in (1, 2, 5, 6, 8, 9, 14)])
        self.assertEqual(self.saved['counts']['fixed_denominator'], 16)
        self.assertEqual(self.saved['observed_valid_triple_denominator'], 16)
        self.assertEqual(self.saved['agreement_rates_observed_valid_triples']['normalized_three_way_agreement'], 6 / 16)
        self.assertEqual(self.saved['punctuation_whitespace_comparison']['agreement_rates_observed_valid_triples']['normalized_three_way_agreement'], 7 / 16)
        self.assertEqual(self.saved['categories']['unanimous_differs_from_target'], ['clip-000006', 'clip-000014'])

    def test_chinese_report_and_audit_bind_exact_saved_data(self):
        self.assertEqual(self.files['report.zh.txt'].decode(), self.comparator.chinese_report(self.saved))
        audit = archive.read_json(self.files['verification.json'])
        for key, file in [('input_sha256', 'comparison-input.json'), ('comparison_sha256', 'comparison.json'), ('chinese_report_sha256', 'report.zh.txt')]:
            self.assertEqual(audit[key], archive.sha(self.files[file]))
        self.assertEqual(audit['terminal_sha256'], TERMINAL_SHA256)
        self.assertEqual((audit['rows_checked'], audit['fixed_denominator'], audit['observed_valid_triple_denominator']), (16, 16, 16))
        self.assertEqual((audit['literal_agreement'], audit['punctuation_only_agreement'], audit['punctuation_and_whitespace_agreement']), (0, 6, 7))

    def test_changed_counts_rows_rates_or_boundaries_are_rejected(self):
        for section, key, value in [
            ('counts', 'normalized_three_way_agreement', 7),
            ('agreement_rates_observed_valid_triples', 'normalized_three_way_agreement', 0.5),
            ('punctuation_whitespace_comparison', 'observed_valid_triple_denominator', 15),
        ]:
            changed = copy.deepcopy(self.saved)
            changed[section][key] = value
            with self.assertRaisesRegex(ValueError, 'real-data recomputation'):
                check_saved_payload(changed, self.expected)
        for key, value in [('target_text', '小屋'), ('wav_sha256', '0' * 64), ('training_admitted', True)]:
            changed = copy.deepcopy(self.saved)
            changed['rows'][0][key] = value
            with self.assertRaisesRegex(ValueError, 'real-data recomputation'):
                check_saved_payload(changed, self.expected)

    def test_changed_manifest_is_rejected(self):
        directory = self.copied_package()
        (directory / 'SHA256SUMS.json').write_bytes(self.files['SHA256SUMS.json'] + b'\n')
        with self.assertRaisesRegex(ValueError, 'manifest identity'):
            checked_package(directory)

    def test_changed_report_is_rejected(self):
        directory = self.copied_package()
        (directory / 'report.zh.txt').write_bytes(b'changed result')
        with self.assertRaisesRegex(ValueError, 'file identity'):
            checked_package(directory)

    def test_unlisted_file_or_missing_file_is_rejected(self):
        for extra in [True, False]:
            directory = self.copied_package()
            if extra:
                (directory / 'unlisted.txt').write_bytes(b'extra')
            else:
                (directory / 'README.md').unlink()
            with self.assertRaisesRegex(ValueError, 'closed regular-file set'):
                checked_package(directory)

    def test_symlink_and_unlisted_directory_are_rejected(self):
        for symlink in [True, False]:
            directory = self.copied_package()
            if symlink:
                (directory / 'report.zh.txt').unlink()
                (directory / 'report.zh.txt').symlink_to(PACKAGE / 'report.zh.txt')
            else:
                (directory / 'extra').mkdir()
            with self.assertRaisesRegex(ValueError, 'closed regular-file set'):
                checked_package(directory)

    def test_no_model_runtime_or_external_packages_loaded(self):
        for module in ['torch', 'whisper', 'numpy', 'transformers', 'funasr']:
            self.assertNotIn(module, sys.modules)


if __name__ == '__main__':
    unittest.main()
