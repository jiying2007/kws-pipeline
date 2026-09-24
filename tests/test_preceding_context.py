#!/usr/bin/env python3
from __future__ import annotations
import copy
import json
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import preceding_context_evaluation as ctx

WORDS = {1: (1, 2, 3, 4), 2: (3, 4, 3, 4)}


def row(target, split='train', wake=False):
    return {'split': split, 'target_ids': list(target), 'expected': [{'keyword_id': 2}] if wake else [],
            'path': 'fixture.wav', 'wav_sha256': 'a' * 64}


class JoinTests(unittest.TestCase):
    def test_repeated_short_phrase_forms_new_wake(self):
        self.assertTrue(ctx.crosses_keyword_join([3, 4], [3, 4], WORDS))
        self.assertTrue(ctx.crosses_keyword_join([1, 2], [3, 4], WORDS))

    def test_all_possible_join_positions(self):
        for word in WORDS.values():
            for i in range(1, len(word)):
                with self.subTest(word=word, i=i):
                    self.assertTrue(ctx.crosses_keyword_join(word[:i], word[i:], WORDS))

    def test_existing_wake_is_not_new_cross_boundary_wake(self):
        self.assertFalse(ctx.crosses_keyword_join([1, 2, 4], WORDS[1], WORDS))
        self.assertFalse(ctx.crosses_keyword_join(WORDS[1], [2, 2], WORDS))

    def test_embedded_keyword_is_detected_in_prior(self):
        self.assertTrue(ctx.contains_keyword([4, 1, 2, 3, 4, 2], WORDS))
        self.assertFalse(ctx.contains_keyword([1, 2, 4], WORDS))

    def test_transcripts_fail_closed(self):
        for bad in ([], [0], [True], ['1'], '12', [1.0], None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ctx.crosses_keyword_join(bad, [3, 4], WORDS)
        with self.assertRaises(ValueError):
            ctx.crosses_keyword_join([1], [2], {})

    def test_selection_excludes_cross_join_and_other_splits(self):
        unsafe = row([3, 4]); other = row([1, 2, 4], 'test'); safe = row([1, 2, 4])
        target = row([3, 4], wake=False)
        self.assertIs(ctx.select_prior([unsafe, other, safe], target, 0, WORDS), safe)

    def test_selection_excludes_wakes_even_with_bad_negative_label(self):
        with self.assertRaisesRegex(ValueError, 'no same-split'):
            ctx.select_prior([row([1, 2, 3, 4])], row([1, 2, 4]), 0, WORDS)

    def test_no_safe_context_never_falls_back_to_bad_labels(self):
        with self.assertRaisesRegex(ValueError, 'no same-split'):
            ctx.select_prior([row([3, 4])], row([3, 4]), 0, WORDS)
        with self.assertRaises(ValueError):
            ctx.select_prior([], row([1], 'qualification'), 0, WORDS)

    def test_selection_is_deterministic_and_does_not_mutate(self):
        rows = [row([1, 2, 4]), row([3, 1, 2, 4])]; before = copy.deepcopy(rows)
        self.assertIs(ctx.select_prior(rows, row([3, 4]), 3, WORDS), rows[1])
        self.assertEqual(rows, before)
        for bad in (-1, True, 0.1):
            with self.assertRaises(ValueError):ctx.select_prior(rows, row([3, 4]), bad, WORDS)


class PcmTests(unittest.TestCase):
    def test_noise_is_predeclared_deterministic_and_ends_in_silence(self):
        low = ctx.noise_prefix(1); high = ctx.noise_prefix(10)
        self.assertEqual(len(low), 32000); self.assertEqual(low, ctx.noise_prefix(1))
        self.assertEqual(low[-6400:], b'\0' * 6400)
        a = struct.unpack('<16000h', low); b = struct.unpack('<16000h', high)
        self.assertEqual(tuple(10 * x for x in a), b)
        for bad in (2, True, 1.0):
            with self.assertRaises(ValueError):ctx.noise_prefix(bad)

    def test_speech_padding_preserves_pcm_and_aligns_hop(self):
        raw = b'\1\0' * 337
        value = ctx.speech_prefix(raw)
        self.assertEqual(value[:len(raw)], raw);self.assertEqual(len(value) // 2 % 320, 0)
        self.assertTrue(value.endswith(b'\0\0' * 3200))
        for bad in (b'', b'\1', None):
            with self.assertRaises(ValueError):ctx.speech_prefix(bad)


class EvaluationTests(unittest.TestCase):
    def fixture(self, root):
        for name in ('model.kwm', 'runner'):(root / name).write_bytes(b'fixture-' + name.encode())
        rows = []
        for split in ('train', 'calibration', 'test'):
            for kid, target in ((1, [1, 2, 3, 4]), (2, [3, 4, 3, 4]), (None, [1, 2, 4])):
                r = row(target, split);r['expected'] = [] if kid is None else [{'keyword_id': kid, 'start_s': .01, 'end_s': .03}]
                rows.append(r)
        return {'rows': rows}

    def run_fixture(self, root, output, mutate_runner=False):
        pool = self.fixture(root)
        def compile_pack(*args, **kwargs):
            (output / 'keywords.kwk').write_bytes(b'pack')
        def collect(*args):
            if mutate_runner:(root / 'runner').write_bytes(b'changed')
            return []
        with mock.patch.object(ctx, 'verify_pool', return_value=pool), \
             mock.patch.object(ctx, 'keyword_target_sequences', return_value=WORDS), \
             mock.patch.object(ctx, 'resolve', return_value=root / 'fixture.wav'), \
             mock.patch.object(ctx, 'pcm', return_value=b'\1\0' * 640), \
             mock.patch.object(ctx.subprocess, 'run', side_effect=compile_pack), \
             mock.patch.object(ctx, 'collect', side_effect=collect):
            return ctx.evaluate(root, 'p' * 64, root / 'model.kwm', root / 'runner', output)

    def test_real_control_flow_retains_paired_inputs_without_claiming_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp);out = root / 'result';result = self.run_fixture(root, out)
            self.assertTrue(result['completed']); self.assertTrue(result['report_only'])
            self.assertFalse(result['release_authority']);self.assertEqual(len(result['rows']), 9)
            self.assertFalse(result['contexts_overlap_target_audio'])
            for r in result['rows']:
                self.assertEqual(r['context']['matched'], 0)
                self.assertEqual(r['context']['expected'], 2)
            records = json.loads((out / 'test-prior-nonwake.json').read_text())['records']
            for r in records:
                self.assertFalse(ctx.crosses_keyword_join(r['prior_target_ids'], r['target_ids'], WORDS))
                self.assertFalse(r['cross_boundary_keyword'])

    def test_partial_evidence_is_not_complete_when_runtime_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp);out = root / 'result'
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.run_fixture(root, out, mutate_runner=True)
            self.assertFalse(json.loads((out / 'summary.json').read_text())['completed'])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp);out = root / 'result';out.mkdir()
            with self.assertRaises(FileExistsError):self.run_fixture(root, out)


if __name__ == '__main__':unittest.main()
