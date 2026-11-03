"""Stdlib-only contracts. Fixtures are invented, never model measurements."""
import copy
import hashlib
import io
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
import wave
import struct
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/model_comparison'))
import runner
from compare import compare, validate_split
from runner import (GreedyKeywordDecoder, projected_identity, SOURCE_PATHS, source_hashes,
                    verify_source_hashes, sample_order, bounded_integer, MAX_CHUNK_FRAMES, check_wav_bound,
                    iter_wav_features)


def fixture():
    rows = [dict(split=s, speaker_id=s, session_id=s, source_id=s, pcm_sha256=str(i)*64)
            for i, s in enumerate(('train','calibration','selection'))]
    plan = dict(development_only=True, protected_evidence_used=False,
                split_inventory=rows, seed=7, max_far_per_hour=0.5,
                budget=dict(max_training_seconds=100, max_peak_rss_bytes=1000,
                            max_model_bytes=1000, max_macs_per_frame=1000, sample_exposures=10))
    control = dict(arm='control', status='MEASURED', split_sha256=validate_split(rows), seed=7,
                   sample_exposures=10, threshold_source='calibration-only',
                   selection_used_for_tuning=False, thresholds_sha256='a'*64,
                   negative_hours=8, false_accept_events=2, training_seconds=10,
                   peak_rss_bytes=100, model_bytes=100, macs_per_frame=100,
                   per_keyword={w: dict(positives=10,false_rejects=2) for w in ('你好小窝','小窝小窝')})
    for k in ('source_sha','runner_id','frontend_sha256','tokens_sha256','optimizer_sha256',
              'decoder_sha256','selection_stream_sha256','calibration_stream_sha256'):
        control[k] = 'fixture'
    control['calibration_stream_sha256'] = 'b' * 64
    control['selection_stream_sha256'] = 'c' * 64
    treatment = copy.deepcopy(control)
    treatment['arm'] = 'treatment'
    treatment['per_keyword']['你好小窝']['false_rejects'] = 1
    return plan, [control, treatment]


class ComparisonTests(unittest.TestCase):
    def test_fixture_arithmetic(self):
        p, r = fixture()
        result = compare(p,r)
        self.assertTrue(result['treatment_per_word_dominates_at_shared_gate'])
        self.assertFalse(result['shipping_approved'])
    def test_reject_same_calibration_and_selection_stream(self):
        p, receipts = fixture()
        for receipt in receipts:
            receipt['selection_stream_sha256'] = receipt['calibration_stream_sha256']
        with self.assertRaisesRegex(ValueError, 'calibration and selection streams must differ'):
            compare(p, receipts)

    def test_stream_digests_are_canonical_sha256(self):
        for key in ('calibration_stream_sha256', 'selection_stream_sha256'):
            for value in ('fixture', '', ' ', 'a' * 63, 'a' * 65,
                          'g' * 64, 'A' * 64, ' ' + 'a' * 64, None, 123):
                with self.subTest(key=key, value=value):
                    p, receipts = fixture()
                    for receipt in receipts:
                        receipt[key] = value
                    with self.assertRaises(ValueError):
                        compare(p, receipts)

    def test_reject_unmatched_stream_digests(self):
        for key in ('calibration_stream_sha256', 'selection_stream_sha256'):
            p, receipts = fixture()
            receipts[1][key] = 'd' * 64
            with self.assertRaisesRegex(ValueError, 'unmatched ' + key):
                compare(p, receipts)

    def test_each_leakage_dimension(self):
        for key in ('speaker_id','session_id','source_id','pcm_sha256'):
            p,_=fixture()
            p['split_inventory'][1][key]=p['split_inventory'][0][key]
            with self.assertRaisesRegex(ValueError,'leakage'): validate_split(p['split_inventory'])
    def test_protected_split(self):
        p,_=fixture();p['split_inventory'][0]['split']='qualification'
        with self.assertRaises(ValueError):validate_split(p['split_inventory'])
    def test_reject_unmatched_and_tuned(self):
        for key,value in [('seed',8),('sample_exposures',9),('source_sha','different'),
                          ('selection_used_for_tuning',True),('status','NOT_RUN'),('negative_hours',4)]:
            p,r=fixture();r[1][key]=value
            with self.assertRaises(ValueError):compare(p,r)
    def test_far_and_resource_failure_block_winner(self):
        for key,value in [('false_accept_events',5),('training_seconds',101)]:
            p,r=fixture();r[1][key]=value
            self.assertFalse(compare(p,r)['treatment_per_word_dominates_at_shared_gate'])
    def test_one_word_regression_blocks_winner(self):
        p,r=fixture();r[1]['per_keyword']['小窝小窝']['false_rejects']=3
        self.assertFalse(compare(p,r)['treatment_per_word_dominates_at_shared_gate'])
    def test_nan_and_zero_denominator(self):
        p,r=fixture();r[1]['negative_hours']=float('nan')
        with self.assertRaises(ValueError):compare(p,r)
        p,r=fixture();r[1]['per_keyword']['你好小窝']['positives']=0
        with self.assertRaises(ValueError):compare(p,r)


class DecoderTests(unittest.TestCase):
    def decoder(self, **kw):
        return GreedyKeywordDecoder([dict(id=2,tokens=[3,4,3,4],threshold=0.5)], **kw)
    def test_repeat_and_blank(self):
        d=self.decoder();hits=[]
        for x in [3,3,0,4,4,0,3,0,4]:hits.extend(d.accept(x,.8))
        self.assertEqual(len(hits),1)
        self.assertAlmostEqual(hits[0]['time_s'],(400+8*320)/16000)
    def test_no_stale_prefix(self):
        d=self.decoder(max_frames=4)
        hits=[]
        for x in [3,4,0,0,0,3,4]:hits.extend(d.accept(x,.8))
        self.assertEqual(hits,[])
    def test_threshold_and_reset(self):
        d=self.decoder()
        self.assertEqual([h for x in [3,4,3,4] for h in d.accept(x,.4)],[])
        self.assertEqual(self.decoder().history,[])
    def test_cooldown(self):
        d=self.decoder(cooldown_frames=50)
        hits=[h for x in [3,4,3,4,3,4] for h in d.accept(x,.8)]
        self.assertEqual(len(hits),1)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):self.decoder().accept(1,float('nan'))


class HardeningTests(unittest.TestCase):
    def test_preset_voice_leakage_survives_projection(self):
        for location in ('top', 'provenance', 'links'):
            p,_=fixture(); rows=[]
            for i,row in enumerate(p['split_inventory']):
                voice={'preset_voice_id':'shared-voice' if i < 2 else 'other-voice'}
                raw=(voice if location == 'top' else
                     {'speech_like_provenance':voice} if location == 'provenance' else
                     {'lineage_links':[voice]})
                rows.append(projected_identity(row,raw,row['split']))
            with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_direct_preset_voice_leakage(self):
        p,_=fixture();rows=p['split_inventory']
        rows[0]['preset_voice_id']='shared-voice'
        rows[1]['preset_voice_id']='shared-voice'
        with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_preset_voice_same_split_and_distinct_voices_allowed(self):
        p,_=fixture();rows=p['split_inventory']
        for row in rows:row['preset_voice_id']='voice-'+row['split']
        rows.append(dict(rows[0],source_id='another-train-source',pcm_sha256='f'*64))
        validate_split(rows)
    def test_preset_voice_placeholder_rejected(self):
        for value in ('unknown',' pending_001 ',None,''):
            p,_=fixture();rows=p['split_inventory']
            rows[0]=projected_identity(rows[0],{'preset_voice_id':value},'train')
            with self.assertRaises(ValueError):validate_split(rows)
    def test_preset_voice_namespace_not_source_identity(self):
        p,_=fixture();rows=p['split_inventory']
        rows[0]['preset_voice_id']=rows[1]['source_id']
        validate_split(rows)
    def test_unknown_identities_rejected(self):
        for key in ('speaker_id','session_id','source_id'):
            for value in ('unknown',' NONE ','pending','null','n/a','Unknown-speaker','pending_001'):
                p,_=fixture(); p['split_inventory'][0][key]=value
                with self.assertRaises(ValueError):validate_split(p['split_inventory'])
    def test_augmented_source_family_leakage(self):
        for key in ('source_family','source_family_id','family_id','derivation_family','derivation_family_id',
                    'source_pcm','source_pcm_sha256','original_pcm_sha256','reference_audio_sha256'):
            p,_=fixture(); rows=p['split_inventory']
            shared='f'*64 if 'pcm' in key or key.endswith('sha256') else 'original-family'
            projected=[]
            for i,row in enumerate(rows):
                raw={key:shared} if i < 2 else {}
                projected.append(projected_identity(row,raw,row['split']))
            with self.assertRaisesRegex(ValueError,'leakage'):validate_split(projected)
    def test_file_and_pcm_hash_namespaces_are_distinct(self):
        p,_=fixture();rows=p['split_inventory']
        rows[1]=projected_identity(rows[1],{'reference_audio_sha256':rows[0]['pcm_sha256']},'calibration')
        validate_split(rows)
    def test_family_aliases_cross_connect(self):
        p,_=fixture();rows=p['split_inventory']
        rows[0]=projected_identity(rows[0],{'source_family_id':'shared'},'train')
        rows[1]=projected_identity(rows[1],{'derivation_family':'shared'},'calibration')
        with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_source_id_to_derivation_family_leakage(self):
        p,_=fixture();rows=p['split_inventory']
        rows[1]=projected_identity(rows[1],{'derivation_family':rows[0]['source_id']},'calibration')
        with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_original_to_augmented_pcm_leakage(self):
        p,_=fixture();rows=p['split_inventory']
        rows[1]=projected_identity(rows[1],{'source_pcm_sha256':rows[0]['pcm_sha256']},'calibration')
        with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_transitive_family_alias_and_nested_provenance(self):
        p,_=fixture();rows=p['split_inventory']
        rows[0]=projected_identity(rows[0],{'speech_like_provenance':{'source_family':'family-A'}},'train')
        bridge=dict(rows[0],source_id='bridge-source',pcm_sha256='e'*64,
                    lineage_links=[{'source_family':'family-A','derivation_family':'family-B'}])
        rows.append(bridge)
        rows[1]=projected_identity(rows[1],{'derivation_family':'family-B'},'calibration')
        with self.assertRaisesRegex(ValueError,'leakage'):validate_split(rows)
    def test_pcm_projection_uses_measured_and_checks_claim(self):
        p,_=fixture();row=p['split_inventory'][0]
        with self.assertRaises(ValueError):projected_identity(row,{'pcm_sha256':'a'*64},'train')
        self.assertEqual(projected_identity(row,{},'train')['pcm_sha256'],row['pcm_sha256'])
    def test_exact_source_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for path in SOURCE_PATHS:
                target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(path)
            valid=source_hashes(root)
            for path in SOURCE_PATHS:
                self.assertEqual(valid[path], hashlib.sha256(path.encode()).hexdigest())
            verify_source_hashes(valid,root)
            missing=dict(valid);missing.pop(SOURCE_PATHS[0])
            extra=dict(valid,unexpected='a'*64)
            bad=dict(valid);bad[SOURCE_PATHS[0]]='not-a-digest'
            upper=dict(valid);upper[SOURCE_PATHS[0]]='A'*64
            wrong=dict(valid);wrong[SOURCE_PATHS[0]]='0'*64
            for invalid in ({},None,missing,extra,bad,upper,wrong):
                with self.assertRaises(ValueError):verify_source_hashes(invalid,root)
    def test_schedule_bounded_lazy_and_repeatable(self):
        a=sample_order(5,3,7)
        self.assertIs(iter(a),a)
        self.assertEqual(list(a),list(sample_order(5,3,7)))
        for size,epochs in ((1,101),(10001,1),(10000,11)):
            with self.assertRaises(ValueError):list(sample_order(size,epochs,1))
    def test_audio_duration_bound(self):
        import wave
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'too-long.wav'
            with wave.open(str(path),'wb') as wf:
                wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(16000)
                wf.writeframes(b'\0\0'*16001)
            with self.assertRaises(ValueError):check_wav_bound(path,1)
    def test_chunk_bounds(self):
        for value in (0,-1,MAX_CHUNK_FRAMES+1,True,1.5):
            with self.assertRaises(ValueError):bounded_integer(value,'chunk',1,MAX_CHUNK_FRAMES)

class StreamCompletenessTests(unittest.TestCase):
    """Invented PCM and shape-only doubles; no Torch or model execution."""
    @staticmethod
    def wav(path, samples):
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            wf.writeframes(b'\0\0' * samples)

    @staticmethod
    def runtime():
        buffers = []
        class PCM:
            def __init__(self, size): self.size = size
            def float(self): return self
            def __truediv__(self, value): return self
        def frombuffer(raw, dtype):
            buffers.append(len(raw))
            return PCM(len(raw) // 2)
        def features(pcm, bins, frontend):
            return SimpleNamespace(shape=(max(0, (pcm.size - 400) // 320 + 1), bins))
        torch = SimpleNamespace(frombuffer=frombuffer, int16='fixture-int16')
        return (torch, None, None, features, 'fixture', None, None), buffers

    def test_complete_short_tail_and_chunk_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.wav'
            for samples in (0, 1, 399, 400, 401, 720, 721, 9999):
                self.wav(path, samples)
                for chunk in (1, 2, 25, MAX_CHUNK_FRAMES):
                    with self.subTest(samples=samples, chunk=chunk):
                        rt, buffers = self.runtime()
                        with wave.open(str(path), 'rb') as wf:
                            frames = sum(x.shape[0] for x in iter_wav_features(rt, wf, chunk))
                        self.assertEqual(frames, max(0, (samples - 400) // 320 + 1))
                        self.assertTrue(all(size <= (chunk * 320 + 399) * 2 for size in buffers))

    def test_truncated_payload_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.wav'
            for samples in (400, 4000):
                self.wav(path, samples)
                complete = path.read_bytes()
                for removed in (1, 2, samples * 2):
                    path.write_bytes(complete[:-removed])
                    for chunk in (1, 2, 25, MAX_CHUNK_FRAMES):
                        with self.subTest(samples=samples, removed=removed, chunk=chunk):
                            rt, _ = self.runtime()
                            with wave.open(str(path), 'rb') as wf:
                                with self.assertRaisesRegex(ValueError, 'truncated PCM payload'):
                                    list(iter_wav_features(rt, wf, chunk))

    def test_odd_declared_data_size_rejected_before_frame_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.wav'
            self.wav(path, 400)
            header = path.read_bytes()[:40]
            for actual in (800, 801, 802):
                with self.subTest(actual_bytes=actual):
                    raw = bytearray(header + struct.pack('<I', 801) + b'\0' * actual)
                    raw[4:8] = struct.pack('<I', len(raw) - 8)
                    path.write_bytes(raw)
                    with self.assertRaisesRegex(ValueError, 'PCM16 data size'):
                        check_wav_bound(path, 1)

    def test_preflight_allows_unknown_chunks_and_riff_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.wav'
            self.wav(path, 399)
            original = path.read_bytes()
            def chunk(tag, payload):
                return tag + struct.pack('<I', len(payload)) + payload + b'\0' * (len(payload) % 2)
            for prefix in (b'', chunk(b'JUNK', b'abc'),
                           chunk(b'JUNK', b'abc') + chunk(b'LIST', b'abcd')):
                body = b'WAVE' + prefix + original[12:36] + chunk(b'JUNK', b'x') + original[36:]
                path.write_bytes(b'RIFF' + struct.pack('<I', len(body)) + body)
                check_wav_bound(path, 1)
                with wave.open(str(path), 'rb') as wf:
                    self.assertEqual(list(iter_wav_features((None,) * 7, wf, 25)), [])

    def test_preflight_rejects_data_outside_file_or_riff_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.wav'
            self.wav(path, 400)
            complete = path.read_bytes()
            short_riff = bytearray(complete)
            short_riff[4:8] = struct.pack('<I', len(complete) - 10)
            for raw in (complete[:-2], short_riff):
                path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'truncated PCM payload'):
                    check_wav_bound(path, 1)

    def test_truncated_inference_does_not_emit_completion_receipt(self):
        # Stub checkpoint/model setup only. Truncation happens before any feature
        # or model forward call; this checks orchestration, not model numerics.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, checkpoint, keywords, output = (root / name for name in
                ('fixture.wav', 'fixture.pt', 'keywords.json', 'events.jsonl'))
            self.wav(audio, 400)
            checkpoint.write_bytes(b'invented checkpoint')
            keywords.write_text('[{"id":1,"tokens":[1],"threshold":0.5}]')
            rt, _ = self.runtime()
            rt[0].float32 = 'fixture-float32'
            rt[0].no_grad = nullcontext
            rt[0].load = lambda *a, **kw: dict(schema='research-paired-ctc-v1',
                frontend='logmel', source_hashes={}, spec={}, state_dict={},
                synthetic_diagnostic_only=True)
            model = unittest.mock.Mock()
            args = SimpleNamespace(chunk_frames=25, wav=audio, checkpoint=checkpoint,
                keywords=keywords, output=output, repo=ROOT, recording='fixture')
            def truncate_after_preflight(repo):
                audio.write_bytes(audio.read_bytes()[:-2])
                return rt
            with patch.object(runner, 'runtime', side_effect=truncate_after_preflight), \
                 patch.object(runner, 'build', return_value=model), \
                 patch.object(runner, 'verify_source_hashes'):
                with self.assertRaisesRegex(ValueError, 'truncated PCM payload'):
                    runner.infer(args)
            model.forward_chunk.assert_not_called()
            self.assertFalse(Path(str(output) + '.receipt.json').exists())



class StreamingHashTests(unittest.TestCase):
    LIMIT = 64 * 1024

    def test_digest_matches_whole_file_at_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bytes.bin'
            for size in (0, 1, self.LIMIT - 1, self.LIMIT,
                         self.LIMIT + 1, 3 * self.LIMIT + 17):
                with self.subTest(size=size):
                    payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
                    path.write_bytes(payload)
                    for argument in (path, str(path)):
                        self.assertEqual(runner.sha(argument), hashlib.sha256(payload).hexdigest())

    def test_reads_are_bounded_and_short_reads_are_not_eof(self):
        payload = bytes(range(256)) * 800
        calls = []
        limit = self.LIMIT

        class ShortReader(io.BytesIO):
            def read(self, size=-1):
                calls.append(size)
                if not 0 < size <= limit:
                    raise AssertionError('hash must use bounded positive reads')
                return super().read(min(size, 997))

        stream = ShortReader(payload)
        with patch.object(Path, 'open', return_value=stream) as opened:
            self.assertEqual(runner.sha('fixture.bin'), hashlib.sha256(payload).hexdigest())
        opened.assert_called_once_with('rb')
        self.assertGreater(len(calls), 2)
        self.assertTrue(stream.closed)

    def test_read_failure_propagates_and_closes_stream(self):
        class BrokenReader(io.BytesIO):
            calls = 0
            def read(self, size=-1):
                self.calls += 1
                if self.calls == 2:
                    raise OSError('injected read failure')
                return super().read(min(size, 3) if size > 0 else 3)

        stream = BrokenReader(b'abcdefgh')
        with patch.object(Path, 'open', return_value=stream):
            with self.assertRaisesRegex(OSError, 'injected read failure'):
                runner.sha('fixture.bin')
        self.assertTrue(stream.closed)

    def test_missing_file_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                runner.sha(Path(tmp) / 'missing.bin')


class DataPairTests(unittest.TestCase):
    """Invented WAVs and fake tensor/optimizer; never imports Torch."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for i in range(7):
            path = self.root / ('audio%d.wav' % i)
            with wave.open(str(path), 'wb') as out:
                out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                out.writeframes(struct.pack('<h', i + 1) * (16 + i))
            self.rows.append({'path': path.name, 'speaker_id': 'speaker%d' % i,
                              'session_id': 'session%d' % i, 'source_id': 'source%d' % i,
                              'tokens': [1, 2]})
        self.paths = {key: self.root / (key + '.jsonl')
                      for key in ('control', 'treatment', 'calibration', 'selection')}
        self.write_rows('control', self.rows[:2])
        self.write_rows('treatment', self.rows[:5])
        self.write_rows('calibration', self.rows[5:6])
        self.write_rows('selection', self.rows[6:])
        (self.root / 'tokens.json').write_text('{}')
        self.config = {'schema': 'research-data-pair-v1', 'development_only': True,
            'protected_evidence_used': False, 'synthetic_diagnostic_only': True,
            'tokens': str(self.root / 'tokens.json'),
            'train_manifests': {key: str(self.paths[key]) for key in ('control', 'treatment')},
            'evaluation_manifests': {key: str(self.paths[key]) for key in ('calibration', 'selection')},
            'spec': {'architecture': 'rnn64_control'}, 'optimizer_steps': 7, 'seed': 7,
            'learning_rate': 0.001, 'max_training_seconds': 10}
        self.config_path = self.root / 'config.json'
        self.output = self.root / 'result'
        self.admission_calls, self.step_counts, self.models = [], [], []
        self.reject_treatment = False
        self.mutate_on_step = None
        self.init_drift = False

    def write_rows(self, split, rows):
        import json
        self.paths[split].write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def inspector(self, path):
        with wave.open(str(path), 'rb') as wav:
            pcm = wav.readframes(wav.getnframes())
        return {'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
                'file_sha256': runner.sha(path)}

    def fake_admission(self, paths, tokens, diagnostic):
        self.admission_calls.append((str(paths[0]), diagnostic))
        if self.reject_treatment and paths[0] == self.paths['treatment']:
            raise ValueError('TEST_ADMISSION_REJECTED')
        return {'test_double': True, 'training_admitted': False}

    def context(self):
        from contextlib import ExitStack
        import json
        test = self
        class Tensor:
            dtype, shape = 'float32', (3,)
            def __init__(self, raw=b'initial'):
                self.raw = raw
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return self
            def tobytes(self): return self.raw
            def unsqueeze(self, dim): return self
            def log_softmax(self, dim): return self
            def numel(self): return 2
            def backward(self): pass
            def __float__(self): return 1.0
        class Model:
            def __init__(self):
                self.index = len(test.models)
                self.tensor = Tensor(b'different' if test.init_drift and self.index else b'initial')
                test.models.append(self)
            def state_dict(self): return {'weight': self.tensor}
            def parameters(self): return []
            def train(self): pass
            def __call__(self, x): return Tensor()
        class Optimizer:
            def __init__(self, *args, **kwargs):
                self.index = len(test.step_counts)
                test.step_counts.append(0)
            def zero_grad(self, **kwargs): pass
            def step(self):
                test.step_counts[self.index] += 1
                if test.mutate_on_step:
                    test.mutate_on_step()
        class Manifest:
            def __init__(self, paths, *args):
                self.rows = [json.loads(l) for l in paths[0].read_text().splitlines() if l.strip()]
                self.identity_rows = [
                    {**test.inspector(paths[0].parent/r['path']),
                     **{k: r[k] for k in ('speaker_id','session_id','source_id')}}
                    for r in self.rows]
            def __len__(self): return len(self.rows)
            def __getitem__(self, index): return Tensor(), Tensor()
        torch = SimpleNamespace(__version__='fake-not-torch', manual_seed=lambda seed: None,
            optim=SimpleNamespace(Adam=Optimizer),
            nn=SimpleNamespace(CTCLoss=lambda **kw: lambda *a: Tensor(),
                               utils=SimpleNamespace(clip_grad_norm_=lambda *a: None)),
            tensor=lambda x: Tensor(), isfinite=lambda x: True,
            save=lambda value,path: Path(path).write_bytes(b'fake-checkpoint'))
        rt = (torch, None, Manifest, None, 'fake-logmel', None, None)
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {
            'kws_vocab': SimpleNamespace(load_tokens=lambda p:{str(i):i for i in range(5)}),
            'training_admission': SimpleNamespace(verify_training_manifests=self.fake_admission),
            'corpus_identity': SimpleNamespace(canonical_audio_path=lambda row,path:row['path'],
                                              inspect_pcm16_wav=self.inspector)}))
        stack.enter_context(patch.object(runner, 'runtime', return_value=rt))
        stack.enter_context(patch.object(runner, 'build', side_effect=lambda *a: Model()))
        stack.enter_context(patch.object(runner, 'source_hashes', return_value={'fixture':'a'*64}))
        return stack

    def execute(self):
        import json
        self.config_path.write_text(json.dumps(self.config))
        return runner.train_data(SimpleNamespace(config=self.config_path, repo=ROOT, output=self.output))

    def test_success_actual_orchestration_two_sizes_same_steps(self):
        with self.context():
            receipt = self.execute()
        self.assertEqual(self.step_counts, [7, 7])
        self.assertEqual([p for p,d in self.admission_calls[:2]],
                         [str(self.paths['control']), str(self.paths['treatment'])])
        self.assertTrue(all(d is True for p,d in self.admission_calls))
        a,b = receipt['arms']
        self.assertEqual(a['initial_state_sha256'], b['initial_state_sha256'])
        self.assertNotEqual(a['schedule_sha256'], b['schedule_sha256'])
        self.assertEqual(receipt['status'], 'TRAINED_PAIR')
        self.assertFalse(receipt['effect_measured'])
        self.assertFalse(receipt['shipping_approved'])
        self.assertTrue((self.output/'data-pair-training-receipt.json').exists())

    def test_second_admission_refusal_before_any_model(self):
        self.reject_treatment = True
        with self.context(), self.assertRaisesRegex(ValueError, 'TEST_ADMISSION_REJECTED'):
            self.execute()
        self.assertEqual(self.models, [])
        self.assertEqual(self.step_counts, [])
        self.assertFalse(self.output.exists())

    def test_changed_control_label_rejected(self):
        changed = copy.deepcopy(self.rows[:5])
        changed[0]['tokens'] = [3]
        self.write_rows('treatment', changed)
        with self.context(), self.assertRaisesRegex(ValueError, 'CONTROL_ROWS_CHANGED'):
            self.execute()
        self.assertEqual(self.models, [])

    def test_duplicate_is_not_data_addition(self):
        self.write_rows('treatment', self.rows[:2] + self.rows[:1])
        with self.context(), self.assertRaisesRegex(ValueError, 'NO_UNIQUE_ADDITION'):
            self.execute()

    def test_common_eval_only_and_no_architecture_factor(self):
        for key, value in [('specs', {'control':{}, 'treatment':{}}),
                           ('evaluation_manifests', {'control':{}, 'treatment':{}})]:
            config = copy.deepcopy(self.config); config[key] = value
            with self.assertRaises(ValueError):
                runner.data_pair_config(config)
        config = copy.deepcopy(self.config); config['spec']['architecture'] = 'other'
        with self.assertRaisesRegex(ValueError, 'SINGLE_RNN64_SPEC'):
            runner.data_pair_config(config)

    def test_train_and_eval_alias_rejected(self):
        self.config['evaluation_manifests']['selection'] = self.config['evaluation_manifests']['calibration']
        with self.assertRaisesRegex(ValueError, 'INPUT_ALIAS'):
            runner.data_pair_config(self.config)

    def test_real_inventory_declared_parent_leakage(self):
        changed = copy.deepcopy(self.rows[:5])
        changed[4]['parent_source_identity_kind'] = 'file-sha256'
        changed[4]['parent_source_sha256'] = runner.sha(self.root/self.rows[6]['path'])
        self.write_rows('treatment', changed)
        with self.context(), self.assertRaisesRegex(ValueError, 'cross-split leakage'):
            self.execute()
        self.assertEqual(self.models, [])

    def test_parent_kind_missing_or_wrong_rejected(self):
        for kind in (None, 'pcm-sha256', 'unknown'):
            row = {'parent_source_sha256': 'a'*64}
            if kind is not None: row['parent_source_identity_kind'] = kind
            with self.assertRaisesRegex(ValueError, 'PARENT_IDENTITY_KIND'):
                runner.data_pair_parent(row)
        self.assertEqual(runner.data_pair_parent({'path':'x'}), {'path':'x'})
        with self.assertRaisesRegex(ValueError, 'PARENT_IDENTITY_KIND'):
            runner.data_pair_parent({'parent_source_sha256': None})

    def test_union_connected_component_rejected(self):
        # Pure union rejection, not a claim that both partial fixture
        # inventories individually satisfy all three required splits.
        def row(split, who, pcm, links=()):
            return {'split':split, 'speaker_id':who, 'session_id':who,
                    'source_id':who, 'pcm_sha256':pcm*64, 'lineage_links':list(links)}
        cal=row('calibration','cal','c'); sel=row('selection','sel','d')
        atrain=row('train','a','a',[{'source_audio_sha256':'e'*64}])
        btrain=row('train','b','b',[{'source_audio_sha256':'e'*64}, {'source_id':'sel'}])
        details={'control':{'train_rows':[{'path':'a'}], 'inventory':[atrain,cal,sel]},
                 'treatment':{'train_rows':[{'path':'a'},{'path':'b'}], 'inventory':[btrain]}}
        with self.assertRaisesRegex(ValueError, 'cross-split leakage'):
            runner.validate_data_pair_inventories(details)

    def test_initial_state_drift_rejected(self):
        self.init_drift=True
        with self.context(), self.assertRaisesRegex(ValueError, 'INITIAL_STATE_DRIFT'):
            self.execute()
        self.assertEqual(self.step_counts, [7])
        self.assertFalse((self.output/'data-pair-training-receipt.json').exists())

    def test_config_drift_rejected(self):
        self.mutate_on_step=lambda: self.config_path.write_text('{}')
        with self.context(), self.assertRaisesRegex(ValueError, 'INPUT_DRIFT'):
            self.execute()
        self.assertFalse((self.output/'data-pair-training-receipt.json').exists())

    def test_input_audio_drift_rejected(self):
        audio = self.root/self.rows[0]['path']
        original = audio.read_bytes()
        # Keep the RIFF envelope valid; change one PCM byte deterministically.
        changed = original[:-1] + bytes([original[-1] ^ 1])
        self.mutate_on_step=lambda: audio.write_bytes(changed)
        with self.context(), self.assertRaisesRegex(ValueError, 'DATA_PAIR_AUDIO_DRIFT'):
            self.execute()
        self.assertFalse((self.output/'data-pair-training-receipt.json').exists())

    def test_timeout_is_incomplete_not_success(self):
        with self.context(), patch.object(runner.time, 'monotonic', side_effect=[0, 11, 12]), \
                self.assertRaisesRegex(ValueError, 'DATA_PAIR_INCOMPLETE'):
            self.execute()
        import json
        receipt=json.loads((self.output/'data-pair-training-receipt.json').read_text())
        self.assertEqual(receipt['status'], 'INCOMPLETE_PAIR')
        self.assertEqual(receipt['arms'][0]['optimizer_steps'], 0)
        self.assertEqual(len(self.models), 1)

    def test_fixed_step_schedule_bounds_and_repeatability(self):
        self.assertEqual(len(list(runner.fixed_step_order(2,7,7))),7)
        self.assertEqual(list(runner.fixed_step_order(5,7,7)),list(runner.fixed_step_order(5,7,7)))
        for size,steps,seed in [(0,7,7),(2,True,7),(2,7,-1)]:
            with self.assertRaises(ValueError): list(runner.fixed_step_order(size,steps,seed))

    def test_strict_json_and_insufficient_steps(self):
        for raw in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError): runner.data_pair_json(raw)
        self.config['optimizer_steps'] = 4
        with self.context(), self.assertRaisesRegex(ValueError, 'STEPS_MUST_COVER_TREATMENT'):
            self.execute()
        self.assertEqual(self.models, [])

    def test_added_relabelled_same_pcm_rejected(self):
        changed = copy.deepcopy(self.rows[:2])
        extra = dict(self.rows[0]); extra['tokens'] = [3]
        changed.append(extra)
        self.write_rows('treatment', changed)
        with self.context(), self.assertRaisesRegex(ValueError, 'DUPLICATE_TRAIN_PCM'):
            self.execute()
        self.assertEqual(self.models, [])


if __name__=='__main__':unittest.main()
