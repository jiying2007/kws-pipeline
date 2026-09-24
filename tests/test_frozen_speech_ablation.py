#!/usr/bin/env python3
from __future__ import annotations
import copy
import json
import math
import pathlib
import shutil
import struct
import sys
import tempfile
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import frozen_speech_ablation as f

KEYWORDS = {1: (1, 2, 3, 4), 2: (3, 4, 3, 4)}


def fixture(root):
    rows = []
    for s, split in enumerate((*f.SPLITS, 'qualification')):
        for k, targets in enumerate([list(KEYWORDS[1]), list(KEYWORDS[2]), [1, 2, 3], []]):
            path = root / f'{split}-{k}.wav'
            frequency = 190 + 31 * (s * 4 + k)
            samples = [int(6000 * math.sin(2 * math.pi * frequency * i / 16000)) for i in range(6400)]
            with wave.open(str(path), 'wb') as w:
                w.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                w.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
            rows.append({'split': split, 'path': str(path), 'wav_sha256': f.sha(path),
                         'kind': 'positive' if k < 2 else 'negative',
                         'keyword_id': k + 1 if k < 2 else None, 'target_ids': targets,
                         'event_start_frame': 200, 'event_end_frame': 6000,
                         'speech_like_provenance': {'fixture_only': True}})
    return rows


class FrozenSpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.rows = fixture(self.root)
        self.pool = self.root / 'pool'; self.pool.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_treatments_differ_only_in_four_weights(self):
        current, ctc = f.loss_settings('current'), f.loss_settings('ctc-only')
        differences = {k for k in current if current[k] != ctc[k]}
        from objective_config import AUXILIARY_LOSS_WEIGHT_NAMES
        self.assertEqual(differences, set(AUXILIARY_LOSS_WEIGHT_NAMES))
        self.assertEqual(len(differences), 4)
        self.assertTrue(all(ctc[k] == 0.0 for k in differences))
        self.assertTrue(all(current[k] > 0 for k in differences))
        self.assertEqual(current['path_purity_loss_weight'], 0)
        self.assertEqual(current['sequence_margin_negative_policy'], ctc['sequence_margin_negative_policy'])
        with self.assertRaises(ValueError): f.loss_settings('arbitrary')

    def test_explicit_current_weights_match_trainer_defaults(self):
        import ast
        tree = ast.parse((ROOT / 'training/train_ctc.py').read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try: constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError): pass
        for name, value in f.auxiliary_loss_weights(f.loss_settings('current')).items():
            self.assertEqual(value, constants[name.upper()])

    def test_qualification_never_enters_retained_pool(self):
        kept = f.materialize_pool(self.rows, self.pool, KEYWORDS)
        self.assertEqual(len(kept), 12)
        self.assertEqual({r['split'] for r in kept}, set(f.SPLITS))
        self.assertFalse((self.pool / 'qualification.tsv').exists())
        for r in kept:
            self.assertFalse(pathlib.Path(r['path']).is_absolute())
            self.assertEqual(f.sha(self.pool / r['path']), r['wav_sha256'])

    def test_mutated_wav_rejected(self):
        pathlib.Path(self.rows[0]['path']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'bytes mismatch'):
            f.materialize_pool(self.rows, self.pool, KEYWORDS)

    def test_duplicate_or_cross_split_audio_rejected(self):
        self.rows[4]['path'] = self.rows[0]['path']
        self.rows[4]['wav_sha256'] = self.rows[0]['wav_sha256']
        with self.assertRaisesRegex(ValueError, 'duplicate or cross-split'):
            f.materialize_pool(self.rows, self.pool, KEYWORDS)

    def test_positive_target_mismatch_rejected(self):
        self.rows[0]['target_ids'] = [4, 3, 2, 1]
        with self.assertRaisesRegex(ValueError, 'positive target'):
            f.materialize_pool(self.rows, self.pool, KEYWORDS)

    def test_negative_wake_contradiction_rejected(self):
        self.rows[2]['target_ids'] = list(KEYWORDS[1])
        with self.assertRaisesRegex(ValueError, 'negative contains'):
            f.materialize_pool(self.rows, self.pool, KEYWORDS)

    def test_invalid_bounds_rejected(self):
        self.rows[0]['event_end_frame'] = 6401
        with self.assertRaisesRegex(ValueError, 'bounds'):
            f.materialize_pool(self.rows, self.pool, KEYWORDS)

    def test_missing_keyword_support_rejected(self):
        rows = [r for r in self.rows if not (r['split'] == 'test' and r['keyword_id'] == 2)]
        with self.assertRaisesRegex(ValueError, 'missing configured keyword'):
            f.materialize_pool(rows, self.pool, KEYWORDS)

    def test_missing_negative_support_rejected(self):
        rows = [r for r in self.rows if not (r['split'] == 'test' and r['kind'] == 'negative')]
        with self.assertRaisesRegex(ValueError, 'negative support'):
            f.materialize_pool(rows, self.pool, KEYWORDS)

    def test_portable_pool_digest_and_byte_checks(self):
        kept = f.materialize_pool(self.rows, self.pool, KEYWORDS)
        files = {str(p.relative_to(self.pool)): f.sha(p) for p in self.pool.rglob('*') if p.is_file()}
        receipt = {'policy': f.POLICY, 'development_only': True, 'release_authority': False,
                   'files': files, 'rows': kept}
        f.write(self.pool / 'pool.json', receipt)
        digest = f.sha(self.pool / 'pool.json')
        moved = self.root / 'moved'; shutil.copytree(self.pool, moved)
        self.assertEqual(f.verify_pool(moved, digest), receipt)
        with self.assertRaisesRegex(ValueError, 'receipt hash'):
            f.verify_pool(moved, '0' * 64)
        (moved / 'train.tsv').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            f.verify_pool(moved, digest)

    def test_unsafe_paths_rejected(self):
        for path in ('../x', '/etc/passwd'):
            with self.subTest(path=path), self.assertRaises(ValueError): f.resolve(self.pool, path)

    def test_archive_hash_rejected_before_extraction(self):
        fake = self.root / 'bad.tar.gz'; fake.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'archive hash'):
            f.prepare(fake, self.root / 'new')
        self.assertFalse((self.root / 'new').exists())


if __name__ == '__main__':
    unittest.main()
