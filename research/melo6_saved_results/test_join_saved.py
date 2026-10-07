"""Four offline regressions against caller-supplied published Melo6 artifacts.

Tampered JSON exists only in temporary copies. No model or audio is generated.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import join_saved as join


def hashes(root, exclude=()):
    return {p.relative_to(root).as_posix(): join.digest(p.read_bytes())
            for p in root.rglob('*') if p.is_file() and p.name not in exclude}


def write(path, value):
    path.write_bytes(join.canonical(value))


def refreeze(root, model_freezes=True):
    if model_freezes:
        for model in join.MODELS:
            directory = root / 'raw' / model
            write(directory / 'model-raw-freeze.json', {
                'files': hashes(directory, {'model-raw-freeze.json'}), 'labels_joined': False})
    write(root / 'raw-freeze.json', {
        'schema': 'asr6-raw-freeze-v1', 'labels_joined': False,
        'files': {'raw/' + k: v for k, v in hashes(root / 'raw').items()}})
    write(root / 'artifact-freeze.json', {
        'schema': 'asr6-artifact-freeze-v1', 'private_labels_joined': False,
        'files': hashes(root, {'artifact-freeze.json'})})


class SavedJoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = hashes(Path(ARGS.asr_root))
        cls.expected = Path(ARGS.expected_comparison).read_bytes()
        if join.digest(cls.expected) != '7af81332103cf8174d8cbda9deda00e988027493e84f4643518993f20cdf280e':
            raise ValueError('Expected original Melo6 comparison bytes')

    @classmethod
    def tearDownClass(cls):
        if hashes(Path(ARGS.asr_root)) != cls.before:
            raise AssertionError('Original ASR artifact changed')

    def compare(self, root):
        return join.compare_saved(root, ARGS.source_root, ARGS.release_sha256,
                                  ARGS.source_head, ARGS.tts_root, ARGS.plan, ARGS.tts_audit)

    def test_saved_result_reproduces_exact_original_comparison(self):
        result = self.compare(ARGS.asr_root)
        self.assertEqual(join.audio.encode_json(result), self.expected)
        self.assertEqual(result['counts']['both_ASR_exact_intent_matches'], 2)
        self.assertFalse(result['all_six_weak_lexical_gate_passed'])
        self.assertEqual(result['counts']['training_admitted'], 0)
        self.assertTrue(all(r['ctc_target'] is None and r['human_transcript'] == 'UNKNOWN'
                            for r in result['clips']))

    def test_refrozen_aggregate_text_cannot_override_completion_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'tampered-artifact'
            shutil.copytree(ARGS.asr_root, root)
            receipt = root / 'raw/qwen06/clip-000003.receipt.json'
            receipt_before = receipt.read_bytes()
            aggregate = root / 'raw/qwen06/outcomes.json'
            outcomes = json.loads(aggregate.read_bytes())
            original = outcomes[2]['raw_text']
            self.assertEqual(original, json.loads(receipt_before)['raw_text'])
            # Keep the final-status length projection coherent while contradicting
            # the retained completion receipt and its unchanged valid digest.
            outcomes[2]['raw_text'] = 'X' + original[1:]
            write(aggregate, outcomes)
            refreeze(root)
            self.assertEqual(receipt.read_bytes(), receipt_before)
            with self.assertRaisesRegex(ValueError, 'Completion evidence identity'):
                self.compare(root)

    def test_second_raw_freeze_failure_precedes_intent_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'tampered-artifact'
            shutil.copytree(ARGS.asr_root, root)
            path = root / 'raw/qwen06/model-raw-freeze.json'
            value = json.loads(path.read_bytes())
            value['files']['outcomes.json'] = '0' * 64
            write(path, value)
            refreeze(root, model_freezes=False)
            with patch.object(join, 'verify_generation', side_effect=AssertionError('Intent read')) as intent:
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    self.compare(root)
                intent.assert_not_called()

    def test_cli_writes_exact_result_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'comparison.json'
            argv = []
            for name in ('asr-root', 'source-root', 'release-sha256', 'source-head',
                         'tts-root', 'plan', 'tts-audit'):
                argv.extend(['--' + name, getattr(ARGS, name.replace('-', '_'))])
            argv.extend(['--out', str(output)])
            self.assertEqual(join.main(argv), 0)
            self.assertEqual(output.read_bytes(), self.expected)
            with self.assertRaisesRegex(ValueError, 'Output must be new'):
                join.main(argv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('asr-root', 'source-root', 'release-sha256', 'source-head',
                 'tts-root', 'plan', 'tts-audit', 'expected-comparison'):
        parser.add_argument('--' + name, required=True)
    ARGS, unittest_args = parser.parse_known_args()
    unittest.main(argv=[__file__, *unittest_args])
