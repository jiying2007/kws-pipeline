"""Invented PCM/callbacks and source contracts only; never load ONNX/Torch."""
import hashlib
import io
from pathlib import Path
import struct
import sys
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weak_embedding_contract as c
import causal_embedding as stream
import causal_head as head
import weak_audio_cases as audio


def invented_recipe_and_receipts():
    cells = []
    for voice, role in c.VOICE_ROLES.items():
        for label in range(3):
            phrase = next((k for k, v in c.CLASSES.items() if v == label), None) if label else None
            for index in range(8):
                cells.append({'cell_id': 'invented-%s-%d-%d' % (voice, label, index),
                    'voice_id': voice, 'role': role, 'source_family_id': 'qwen3-preset-' + voice,
                    'intent': 'wake-candidate' if label else 'ordinary-negative-candidate',
                    'canonical_phrase': phrase, 'requested_keyword_index': label if label else None,
                    'target_ipa': c.TARGET_IPAS[phrase] if label else None, **dict.fromkeys(c.FALSE_FLAGS, False)})
    receipts = [{'cell_id': cell['cell_id'], 'status': 'GENERATED_UNADMITTED', 'frames': 16000,
                 'wav_sha256': hashlib.sha256(('wav/' + cell['cell_id']).encode()).hexdigest(),
                 'pcm_sha256': hashlib.sha256(('pcm/' + cell['cell_id']).encode()).hexdigest()}
                for cell in cells]
    observations = {r['cell_id']: {'source_pcm_sha256': r['pcm_sha256'],
        'observer_artifact_sha256': c.digest(b'invented-observer-artifact'),
        'observation_sha256': c.digest(('invented-observation/' + r['cell_id']).encode()),
        'evidence_status': 'USEFUL_FOR_INTENT_RESEARCH',
        'phone_review_status': 'SEGMENTAL_USEFUL_QUANTITY_STRESS_UNVERIFIED'} for r in receipts}
    return {'cells': cells}, receipts, observations


class SourceOnlyTests(unittest.TestCase):
    def test_source_ci_cannot_repeat_consumed_asset_audit(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/english-embedding-backbone-assets-once.yml').read_text()
        inspected = workflow.split('\n  inspect-assets:\n', 1)[1]
        self.assertIn("github.sha == '511ccee46dc2fc01483b0d598cfd7fefb286bf31'", inspected)
        self.assertIn('github.run_attempt == 1', inspected)
        self.assertEqual(workflow.count('      - research/english_final_embedding/**\n'), 2)
        for name in ('test_prepared_embedding.py', 'test_portable_head.py'):
            for mode in ('', ' -O', ' -OO'):
                self.assertIn('python3 -I -S -B' + mode + ' research/english_final_embedding/' + name + ' -v', workflow)
        self.assertNotIn('workflow_dispatch:', workflow)
        self.assertNotIn('schedule:', workflow)

    def test_fixed_head_dimensions_and_no_runtime_imports(self):
        spec = head.head_spec()
        self.assertEqual(spec['head_parameters'], (16 * 96 + 1) * 32 + (32 + 1) * 3)
        self.assertEqual(spec['optimizer_updates'], 96)
        self.assertFalse(spec['checkpoint_selection'])
        self.assertFalse(any(x in sys.modules for x in ('torch', 'onnx', 'onnxruntime', 'numpy')))

    def test_prefix_is_observed_background_only_and_phase_aligns_original_eof(self):
        for count in (400, 11520, 16640, 20480, 31999, 35200):
            prefix = c.positive_prefix_samples(count)
            end = prefix + count
            self.assertGreaterEqual(prefix, 35200)
            self.assertLess(prefix, 35200 + 1280)
            self.assertEqual(c.clip_envelope_instance_ends(prefix, count, end), [end])
        for invalid in (0, 399, 35201, True):
            with self.assertRaises(ValueError):
                c.positive_prefix_samples(invalid)

    def test_input_envelope_is_not_mel_support_and_real_context_is_needed(self):
        # Invented metadata fixture models the expected geometry, not evidence
        # that a real model has been inspected or executed in these tests.
        geometry = {'status': 'FIXED_ONNX_CONV_GEOMETRY_VERIFIED',
                    'head_support_start_before_available': 31840,
                    'support_end_before_available': 128}
        frames = 16640
        start = c.positive_prefix_samples(frames)
        eof = start + frames
        self.assertEqual(c.clip_envelope_instance_ends(start, frames, eof), [eof])
        self.assertEqual(c.mel_supported_clip_ends(start, frames, eof, geometry), [])
        self.assertEqual(c.mel_supported_clip_ends(start, frames, eof + 5120, geometry),
                         [eof + i * 1280 for i in range(1, 5)])
        with self.assertRaises(ValueError):
            c.mel_supported_clip_ends(start, frames, eof + 5120, {})
        self.assertEqual(c.mel_supported_clip_ends(35200, 32000, 80000, geometry), [])

    def test_chunk_invariance_with_invented_stateless_embedding(self):
        pcm = [i % 2000 - 1000 for i in range(40123)]
        def invented(values):
            self.assertEqual(len(values), 16000)
            return [float(sum(values) % 997), float(values[-1])] + [0.0] * 94
        expected = stream.CausalFeatureStream(invented).accept(pcm)
        for chunk in (1, 160, 997, 1280, 3199):
            processor = stream.CausalFeatureStream(invented)
            actual = []
            for start in range(0, len(pcm), chunk):
                actual += processor.accept(pcm[start:start + chunk])
            self.assertEqual(actual, expected)
            self.assertEqual(processor.finish(), [])
            self.assertEqual(processor.embedding_calls, len(expected))
        self.assertEqual(expected[-1]['available_sample'], 39040)

    def test_future_pcm_cannot_change_already_emitted_embeddings(self):
        def invented(values):
            return [float(values[-1])] * 96
        a = stream.CausalFeatureStream(invented)
        b = stream.CausalFeatureStream(invented)
        prefix = [3] * 35200
        emitted = a.accept(prefix)
        self.assertEqual(emitted, b.accept(prefix))
        a.accept([7] * 1280)
        b.accept([-9] * 1280)
        self.assertEqual(emitted[-1]['embedding'], (3.0,) * 96)

    def test_head_windows_have_ordered_complete_history(self):
        records = [{'available_sample': 16000 + i * 1280, 'embedding': (float(i),) * 96}
                   for i in range(17)]
        windows = stream.windows_at_ends(records, [35200, 36480])
        self.assertEqual([r[0] for r in windows[0]], list(range(16)))
        self.assertEqual([r[0] for r in windows[1]], list(range(1, 17)))
        with self.assertRaises(ValueError):
            stream.windows_at_ends(records, [16000])
        records[10]['available_sample'] += 1
        with self.assertRaises(ValueError):
            stream.windows_at_ends(records, [35200])

    def test_background_split_is_speaker_disjoint_and_order_independent(self):
        speakers = [str(i) for i in range(40)]
        mapping = c.speaker_roles(speakers)
        self.assertEqual(mapping, c.speaker_roles(list(reversed(speakers))))
        self.assertEqual(sum(v == 'train' for v in mapping.values()), 30)
        self.assertEqual(sum(v == 'development' for v in mapping.values()), 10)
        with self.assertRaises(ValueError):
            c.speaker_roles(speakers[:-1] + ['0'])

    def test_real_prefix_and_context_use_same_role_exact_source_bytes(self):
        invented_sources = []
        for index in range(517):
            pcm = struct.pack('<h', index + 1) * 4000
            invented_sources.append({'id': 'invented-%04d.flac' % index, 'speaker': str(index % 40),
                                     'frames': 4000, 'pcm16le': pcm, 'pcm_sha256': c.digest(pcm)})
        tracks = audio.background_tracks(invented_sources)
        self.assertEqual(len(tracks), 40)
        by_id = {track['id']: track for track in tracks}
        rows, wavs = [], {}
        for number, (role, label) in enumerate((('train', 0), ('train', 1), ('train', 2),
                                                ('development', 0), ('development', 1), ('development', 2))):
            pcm = struct.pack('<h', 1000 + number) * 16640
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(pcm)
            cell_id = 'invented-cell-%d' % number
            wavs[cell_id] = buffer.getvalue()
            rows.append({'cell_id': cell_id, 'role': role, 'weak_class': label, 'frames': 16640,
                         'wav_sha256': c.digest(wavs[cell_id]), 'pcm_sha256': c.digest(pcm),
                         'target_basis': 'requested-intent-not-acoustic-truth',
                         'phone_review_status': 'UNCERTAIN_OR_CONFLICTING'})
        cases = audio.assemble_audio_cases(rows, wavs, tracks)
        self.assertEqual(len(cases), 46)
        for case, row in zip(cases[40:], rows):
            evidence = case['evidence']
            track = by_id[evidence['prefix_track']]
            self.assertEqual(case['role'], evidence['prefix_role'])
            self.assertEqual(case['role'], evidence['context_role'])
            a = 2 * evidence['prefix_start_sample']
            n = 2 * evidence['prefix_samples']
            z = 2 * evidence['context_start_sample']
            k = 2 * evidence['context_samples']
            expected = track['pcm16le'][a:a + n] + c.verified_pcm(wavs[row['cell_id']], row) + track['pcm16le'][z:z + k]
            self.assertEqual(case['pcm16le'], expected)
            self.assertEqual(evidence['silence_or_invented_samples'], 0)
            self.assertEqual(evidence['context_samples'], 5120)
            self.assertLess(evidence['original_clip_eof_sample'], evidence['observed_samples'])
        tracks[0]['role'] = 'development' if tracks[0]['role'] == 'train' else 'train'
        with self.assertRaises(ValueError):
            audio.assemble_audio_cases(rows, wavs, tracks)

    def test_144_recipe_has_only_weak_roles_and_no_acoustic_admission(self):
        recipe, receipts, observations = invented_recipe_and_receipts()
        selected = [r['cell_id'] for r in receipts]
        excluded = []
        rows = c.weak_rows(recipe, receipts, eligible_cell_ids=selected, observations=observations, excluded=excluded)
        self.assertEqual(len(rows), 144)
        self.assertEqual(sum(r['role'] == 'train' for r in rows), 96)
        self.assertTrue(all(all(r[k] is False for k in c.FALSE_FLAGS) for r in rows))
        self.assertTrue(all(r['phone_review_status'] == observations[r['cell_id']]['phone_review_status'] for r in rows))
        self.assertEqual(excluded, [])
        self.assertEqual(len({r['research_selection_sha256'] for r in rows}), 1)
        recipe['cells'][0]['role'] = 'development'
        with self.assertRaises(ValueError):
            c.weak_rows(recipe, receipts, eligible_cell_ids=selected, observations=observations, excluded=[])

    def test_generated_text_does_not_select_or_override_conflict_unknown(self):
        recipe, receipts, observations = invented_recipe_and_receipts()
        selected = [r['cell_id'] for r in receipts]
        with self.assertRaises(TypeError):
            c.weak_rows(recipe, receipts)
        observations[selected[0]]['evidence_status'] = 'STRONG_CONFLICT'
        observations[selected[1]]['evidence_status'] = 'UNKNOWN'
        observations[selected[0]]['phone_review_status'] = 'OBSERVED_VOWEL_CONFLICT'
        observations[selected[1]]['phone_review_status'] = 'OBSERVER_UNINFORMATIVE'
        excluded = []
        rows = c.weak_rows(recipe, receipts, eligible_cell_ids=selected, observations=observations, excluded=excluded)
        self.assertEqual(len(rows), 142)
        self.assertEqual([x['reason'] for x in excluded], ['STRONG_CONFLICT', 'UNKNOWN'])
        self.assertTrue(all(r['status'] == 'GENERATED_UNADMITTED' for r in receipts))
        self.assertFalse(set(selected[:2]) & {r['cell_id'] for r in rows})
        observations[selected[2]]['source_pcm_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'PCM_BOUND_OBSERVATION'):
            c.weak_rows(recipe, receipts, eligible_cell_ids=selected, observations=observations, excluded=[])

    def test_explicit_selection_requires_all_six_role_intent_groups(self):
        recipe, receipts, observations = invented_recipe_and_receipts()
        selected = [c['cell_id'] for c in recipe['cells'] if c['role'] == 'train']
        with self.assertRaisesRegex(ValueError, 'MISSING_WEAK_CLASS_OR_ROLE'):
            c.weak_rows(recipe, receipts, eligible_cell_ids=selected, observations=observations, excluded=[])


if __name__ == '__main__':
    unittest.main()
