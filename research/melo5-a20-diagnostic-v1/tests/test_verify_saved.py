"""Offline archive and malformed-trace tests; no native code is loaded."""
from pathlib import Path
import argparse
import copy
import json
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import verify_saved
DATA = None

class SavedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding, cls.scorer = verify_saved.source_and_scorer()
        cls.raw = (DATA / 'original_A20.raw.jsonl').read_bytes()
        cls.records = [json.loads(line) for line in cls.raw.splitlines()]

    def score(self, records):
        return self.scorer.score(records, ROOT / 'metadata/manifest.json',
                                 ROOT / 'metadata/geometry.json',
                                 self.binding['acquisition_bindings'])

    def reject(self, mutate):
        records = copy.deepcopy(self.records)
        mutate(records)
        with self.assertRaises(self.scorer.ValidationError):
            self.score(records)

    def test_complete_archive_exact_reproduction(self):
        result = verify_saved.verify(DATA)
        self.assertEqual(result['observations_bytes_reproduced'], 10763)
        self.assertEqual(result['descriptive_counts']['expected_keyword_hits'], 0)
        self.assertEqual(result['descriptive_counts']['human_word_positive_clips'], 2)
        self.assertEqual(result['descriptive_counts']['nonwake_clips_with_events'], 0)
        self.assertEqual(result['descriptive_counts']['human_word_nonwake_clips'], 3)

    def test_missing_run_end(self):
        self.reject(lambda r: r.pop())

    def test_missing_callback(self):
        self.reject(lambda r: r.pop(3))

    def test_duplicate_callback(self):
        self.reject(lambda r: r.insert(4, copy.deepcopy(r[3])))

    def test_nonfinite_logit(self):
        self.reject(lambda r: r[3]['logits'][0].__setitem__(0, float('nan')))

    def test_wrong_class_count(self):
        self.reject(lambda r: r[3]['logits'][0].pop())

    def test_missing_feed(self):
        self.reject(lambda r: r.pop(next(i for i, v in enumerate(r) if v['kind'] == 'feed')))

    def test_wrong_identity(self):
        self.reject(lambda r: r[0].__setitem__('model_sha256', '1' * 64))

    def test_wrong_label(self):
        self.reject(lambda r: r[2].__setitem__('label', 0))

    def test_wrong_geometry(self):
        self.reject(lambda r: r[3]['centers'].__setitem__(0, 1))

    def test_truncated_final_newline(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'raw.jsonl'
            path.write_bytes(self.raw[:-1])
            with self.assertRaises(self.scorer.ValidationError):
                self.score(path)

    def test_changed_missing_extra_or_symlink_data_rejected(self):
        for case in ('changed', 'missing', 'extra', 'symlink'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as d:
                target = Path(d) / 'data'
                shutil.copytree(DATA, target)
                raw = target / 'original_A20.raw.jsonl'
                if case == 'changed':
                    raw.write_bytes(self.raw.replace(b'11.2697487', b'11.2697488', 1))
                elif case == 'missing':
                    raw.unlink()
                elif case == 'extra':
                    (target / 'unexpected.json').write_text('{}\n')
                else:
                    raw.unlink()
                    raw.symlink_to(DATA / 'original_A20.raw.jsonl')
                with self.assertRaises(verify_saved.VerificationError):
                    verify_saved.verify(target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args, unittest_args = parser.parse_known_args()
    DATA = args.data_dir.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_args])
