#!/usr/bin/env python3
"""Invented saved-record tests only: no C binary, PCM, model or network calls."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('en63_state_saved_records', HERE / 'state_replay.py')
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def invented_trace():
    raw = b'EN63BND1' + struct.pack('<II', 1, 180)
    counts = (399, 399, 399, 399, 201, 399, 399, 0, 396, 396, 1, 396, 396, 201, 196, 196)
    chunks = [raw]
    for group, count in enumerate(counts):
        cut = {8: 399, 11: 400, 14: 32401}.get(group, 0)
        frame_base = 0 if cut < 400 else 1 + (cut - 400) // 160
        instance = 1 if group in (1, 3, 9, 12, 15) else 0
        for i in range(1, count + 1):
            # These are fabricated binary fields, not the output of any network.
            values = [0.0] * 35 + [1.0, 0.0, 0.0]
            chunks.append(S.TRACE_RECORD.pack(group, instance, frame_base + i,
                                             cut + 400 + (i - 1) * 160, *values, 0))
    return b''.join(chunks)


def invented_native():
    result = {'observations': {'schema': 'english63-state-native-case-observations-v1', 'kind': 'gru',
            'forward_frames': 4773, 'supplied_samples_including_replays': 769475, 'finish_calls': 34,
            'reentry_rejections': 4, 'finished_rejections': 34, 'argument_rejections': 6,
            'overflow_rejections': 3, 'fault_rejections': 3, 'model_rejections': 6,
            'live_instances': 2, 'input_buffer_bytes': 256492, 'clock_overflow_fixture_only': True,
            'fault_flag_fixture_only': True, 'eof_padding_samples': 0, 'eof_events': 0,
            'gap_padding_samples': 0, 'gap_callbacks': 0, 'native_checks_passed': True,
            'state_bytes': 38192, 'inference_end_high_water_rss_kib': 1000},
            'lifetime': {'schema': 'english63-state-native-child-lifetime-v1', 'child_exit_code': 0,
            'child_signaled': False, 'measurement': 'linux-wait4-fresh-native-child-after-exit',
            'includes_test_harness_and_trace_io': True, 'ssc305_measured': False,
            'process_lifetime_peak_rss_kib': 1200}}
    result['raw_stdout'] = '\n'.join(json.dumps(result[key]) for key in ('observations', 'lifetime')) + '\n'
    return result


class SavedRecordContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = invented_trace()

    def test_protocol_counts_only_existing_supplied_pcm_as_diagnostic_replays(self):
        p = S.protocol()
        self.assertEqual(p['test_unique_pcm_samples'], 128246)
        self.assertEqual(p['forward_frames_per_model'], 4773)
        self.assertEqual(p['supplied_samples_including_replays_per_model'], 769475)
        self.assertEqual(p['new_independent_audio_samples'], 0)
        self.assertEqual(p['cases_per_model'], list(S.CASES))
        self.assertEqual(len(set(p['cases_per_model'])), 10)
        self.assertTrue(p['baseline_bit_identity_required'])
        self.assertFalse(p['formal_training_admitted'])

    def test_non_actions_real_entry_is_rejected_before_any_child(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(S.subprocess, 'Popen') as child:
            with self.assertRaisesRegex(ValueError, 'PUBLIC_ACTIONS_FIRST_ATTEMPT_ONLY'):
                S.require_public_actions()
            child.assert_not_called()

    def test_saved_trace_accepts_session_clock_and_segment_counter_difference(self):
        grouped = S.read_trace(self.raw)
        self.assertEqual(sum(map(len, grouped.values())), 4773)
        self.assertEqual(grouped[14][0]['frame'], 202)
        self.assertEqual(grouped[14][0]['sample'], 32801)
        self.assertTrue(S.identical(grouped[14], grouped[15], 201, 32401))

    def test_gap_cannot_silently_reset_supplied_sample_clock(self):
        raw = bytearray(self.raw)
        for offset in range(16, len(raw), 180):
            if struct.unpack_from('<I', raw, offset)[0] == 14:
                struct.pack_into('<Q', raw, offset + 16, 400)
                break
        with self.assertRaisesRegex(ValueError, 'RAW_SESSION_CLOCK'):
            S.read_trace(bytes(raw))

    def test_missing_frame_and_eof_extra_frame_are_rejected(self):
        for raw in (self.raw[:-180], self.raw + self.raw[-180:]):
            with self.subTest(length=len(raw)):
                with self.assertRaisesRegex(ValueError, 'EXACT_GROUP_FRAME_COUNTS'):
                    S.read_trace(raw)

    def test_nonfinite_saved_score_is_rejected(self):
        raw = bytearray(self.raw)
        struct.pack_into('<f', raw, 16 + 24, float('nan'))
        with self.assertRaisesRegex(ValueError, 'TRACE_VALUES'):
            S.read_trace(bytes(raw))

    def test_post_gap_proposal_during_new_warmup_is_rejected(self):
        raw = bytearray(self.raw)
        for offset in range(16, len(raw), 180):
            if struct.unpack_from('<I', raw, offset)[0] == 14:
                struct.pack_into('<I', raw, offset + 176, 1)
                break
        with self.assertRaisesRegex(ValueError, 'RAW_FRESH_WARMUP'):
            S.read_trace(bytes(raw))

    def test_small_numeric_difference_does_not_waive_fixed_baseline_bit_gate(self):
        rows = S.read_trace(self.raw)[0]
        old = b'EN63TRC1' + struct.pack('<II', 1, 168)
        old += b''.join(struct.pack('<IQ', 0, row['sample']) + row['payload'] for row in rows)
        changed = copy.deepcopy(rows)
        value = list(changed[0]['values']); value[0] = 0.00001
        changed[0]['values'] = tuple(value)
        changed[0]['payload'] = struct.pack('<38fI', *value, 0)
        with patch.dict(S.PREFIX_SHA, {'gru': S.c.digest(old)}):
            result = S.compare_saved_prefix('gru', changed, old)
        self.assertTrue(result['original_tolerance_diagnostic_passed'])
        self.assertFalse(result['baseline_trace_bit_identical'])

    def test_bool_is_not_a_zero_native_exit_code(self):
        native = invented_native()
        native['lifetime']['child_exit_code'] = False
        native['raw_stdout'] = '\n'.join(json.dumps(native[key]) for key in ('observations', 'lifetime')) + '\n'
        with self.assertRaisesRegex(ValueError, 'ACTUAL_NATIVE_LIFETIME_RSS_SCOPE'):
            S.case_observations('gru', self.raw, b'', native)

    def test_parsed_native_receipt_must_match_actual_saved_stdout(self):
        native = invented_native()
        native['observations']['native_checks_passed'] = False
        with self.assertRaisesRegex(ValueError, 'RAW_NATIVE_STDOUT_BINDING'):
            S.case_observations('gru', self.raw, b'', native)

    def test_raw_false_cannot_bind_to_parsed_integer_zero(self):
        native = invented_native()
        raw_parts = [json.loads(line) for line in native['raw_stdout'].splitlines()]
        raw_parts[1]['child_exit_code'] = False
        native['raw_stdout'] = '\n'.join(json.dumps(row) for row in raw_parts) + '\n'
        with self.assertRaisesRegex(ValueError, 'RAW_NATIVE_STDOUT_BINDING'):
            S.case_observations('gru', self.raw, b'', native)

    def test_summary_and_receipt_alone_are_not_a_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'SUMMARY.json').write_text('{}')
            (root / 'REPLAY.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'EXACT_ARTIFACT_CLOSURE'):
                S.artifact_manifest(root)


if __name__ == '__main__':
    unittest.main()
