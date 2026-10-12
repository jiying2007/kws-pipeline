"""Invented WAV/native metadata in temporary directories; no runtime/model calls."""
import copy
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for folder in (ROOT / 'tools', ROOT / 'tests', ROOT / 'research/qwen6_asr',
               ROOT / 'research/qwen6_asr/core', HERE):
    sys.path.insert(0, str(folder))
import campaign_contract as c
import worker_receipts as w
from asr_native_fixtures import native_fixture, rebind_native_fixture
from asr_stage.decoding import qwen_transcript
from sense_parser import corrected_sense_extract


class ReceiptFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='k1-asr18-invented-only-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wavs, self.audio = {}, []
        for number in range(1, 4):
            stream = io.BytesIO()
            pcm = number.to_bytes(2, 'little', signed=True) * 1280
            with wave.open(stream, 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(pcm)
            raw = stream.getvalue(); wav_sha = c.digest(raw)
            path = self.root / (wav_sha + '.wav'); path.write_bytes(raw); path.chmod(0o444)
            self.wavs[wav_sha] = path
            self.audio.append({'wav_sha256': wav_sha, 'pcm_sha256': c.digest(pcm), 'audio_frames': 1280})
        self.audio.sort(key=lambda row: row['wav_sha256'])
        clips = [{'audio_id': f'clip-{index:06d}', 'audio_path': 'audio/' + row['wav_sha256'] + '.wav',
                  'wav_sha256': row['wav_sha256']} for index, row in enumerate(self.audio, 1)]
        self.job = {'schema': 'blind-source-screen32-job-v1', 'clips': clips}
        self.plan = {'schema': 'k1-asr18-anonymous-order-metadata-proposal-v1', 'execution_enabled': False,
            'job_sha256': c.digest(c.canonical(self.job)), 'family_order': list(c.FAMILIES),
            'audio_order': ['clip-000003', 'clip-000001', 'clip-000002']}
        by_id = {clip['audio_id']: {**row, 'audio_id': clip['audio_id']} for clip, row in zip(clips, self.audio)}
        self.input = {'schema': 'k1-asr18-worker-input-binding-v1', 'job_sha256': self.plan['job_sha256'],
            'execution_plan_sha256': c.digest(c.canonical(self.plan)), 'screening_receipt_sha256': 'a' * 64,
            'tts_authority_sha256': 'b' * 64, 'audio': [by_id[key] for key in self.plan['audio_order']]}
        self.prepared, self.current, self.captures, self.configs = {}, {}, {}, {}
        for index, family in enumerate(c.FAMILIES):
            frozen, current = self.root / ('frozen-' + family), self.root / ('current-' + family)
            frozen.mkdir(); current.mkdir()
            self.prepared[family], self.current[family] = frozen, current
            source = {'schema': 'retained-asr-source-lock-v1', 'executed_head_sha': 'c' * 40,
                'workflow_run_id': '123', 'files': [{'path': 'invented-' + family + '.py', 'bytes': 1,
                                                  'sha256': c.digest(family.encode())}]}
            assets = [{'path': 'invented-model-' + family, 'bytes': 1, 'sha256': c.digest(('model-' + family).encode())}]
            config, capture = native_fixture(w.ADAPTERS[family], audio_frames=1280)
            model = {'schema': 'retained-asr-model-lock-v1', 'assets': assets,
                'model_sha256': c.digest(c.canonical(assets)), 'provider_id': 'invented-' + family,
                'model_id': 'invented-model-' + family, 'model_revision': 'invented-v1',
                'model_family': config['model_family']}
            config['source_lock_sha256'] = c.digest(c.canonical(source))
            config['asset_lock_sha256'] = c.digest(c.canonical(model))
            capture['source_lock_sha256'] = config['source_lock_sha256']
            capture['asset_lock_sha256'] = config['asset_lock_sha256']
            rebind_native_fixture(config, capture)
            self.captures[family], self.configs[family] = capture, config
            for name, value in zip(w.FILES, (source, model, config)):
                for folder in (frozen, current):
                    (folder / name).write_bytes(c.canonical(value))
                (frozen / name).chmod(0o444)
        self.policy = w.build_policy(self.prepared)
        self.policy_path = self.root / 'policy.json'
        self.policy_path.write_bytes(c.canonical(self.policy)); self.policy_path.chmod(0o444)
        self.bindings = {family: self.bind(family) for family in c.FAMILIES}

    def bind(self, family):
        return w.bind_predecode(recognizer_id=family, policy_path=self.policy_path,
            expected_policy_sha256=c.digest(c.canonical(self.policy)), frozen_dir=self.prepared[family],
            current_dir=self.current[family], run_id='123', executed_head_sha='c' * 40,
            job=self.job, execution_plan=self.plan, input_binding=self.input)

    def start(self, family='qwen06', audio_id=None, directory=None):
        binding = self.bindings[family]
        audio_id = audio_id or self.plan['audio_order'][binding._state['next']]
        row = next(row for row in self.input['audio'] if row['audio_id'] == audio_id)
        return w.emit_start(binding, directory or self.current[family] / audio_id,
            wav_path=self.wavs[row['wav_sha256']], expected_file_sha256=row['wav_sha256'],
            audio_id=audio_id, request_id='k1asr18-123-' + family + '-' + audio_id)

    def result(self, family):
        capture = copy.deepcopy(self.captures[family]); termination = capture['termination']
        text = '合成夹具'
        if family == 'qwen06':
            decoded = 'language Chinese<asr_text>' + text
            row, metadata = qwen_transcript(decoded, termination['generated_token_ids'])
            raw = {'decoded_skip_special_tokens': decoded, 'token_ids': termination['generated_token_ids'],
                'metadata': metadata, 'native_v2_validated': True, 'native_capture': capture}
        elif family == 'sensevoice':
            decoded = '<|zh|><|NEUTRAL|><|Speech|><|woitn|>' + text
            row = corrected_sense_extract(decoded, termination['lexical_token_ids']); metadata = row.pop('metadata')
            raw = {'raw_result': ([{'text': decoded}], {}), 'metadata': metadata, 'native_v2_validated': True,
                'native_capture': capture, 'token_evidence': {'decoded': decoded, 'token_ids': termination['raw_token_ids']},
                'token_binding': {key: copy.deepcopy(termination[key]) for key in ('raw_token_ids', 'metadata_token_ids',
                    'lexical_token_ids', 'metadata_boundary_verified', 'raw_redecode_matches_capture')}}
            raw['token_binding'].update(raw_decoded_sha256=c.digest(decoded.encode()),
                parser_contract_sha256=self.configs[family]['effective_config']['parser_contract_sha256'])
        else:
            row = None
            raw = {'raw_text': text, 'native_capture': capture, 'decoder_completion_validated': True,
                'full_input_processed': True, 'returned_token_ids': termination['generated_token_ids'][:-1]}
        return row, raw

    def returned(self, attempt):
        w.mark_native_enter(attempt)
        row, raw = self.result(attempt.binding.recognizer_id)
        w.mark_native_return(attempt)
        if attempt.binding.recognizer_id == 'whisper':
            (attempt.directory / 'raw-output.json').write_bytes(c.canonical(raw))
        return row, raw

    def finish(self, attempt):
        row, raw = self.returned(attempt)
        return w.emit_completion(attempt, row=row, raw=raw)


class ReceiptTests(ReceiptFixture):
    def test_all_three_native_semantics_same_validators(self):
        for family in c.FAMILIES:
            for audio_id in self.plan['audio_order']:
                attempt = self.start(family, audio_id)
                receipt = self.finish(attempt)
                self.assertEqual(receipt['status'], 'success')
                self.assertEqual(receipt['started_sha256'], c.digest(attempt.started_bytes))
                self.assertEqual(receipt['worker_input_binding_sha256'], c.digest(c.canonical(self.input)))
                self.assertEqual(receipt['execution_plan_sha256'], c.digest(c.canonical(self.plan)))
                self.assertEqual(receipt['screening_receipt_sha256'], self.input['screening_receipt_sha256'])
                self.assertEqual(receipt['acoustic_verbatim_completeness'], 'unknown')
                self.assertFalse((attempt.directory / 'completed.json').stat().st_mode & 0o222)
            with self.assertRaises(ValueError):
                self.bind(family)

    def test_full_canonical_file_and_both_fsync_precede_native(self):
        events = []; original_write, original_sync = os.write, os.fsync
        def write(fd, body):
            events.append(('write', len(body))); return original_write(fd, body[:17])
        def sync(fd):
            events.append(('fsync_file' if stat.S_ISREG(os.fstat(fd).st_mode) else 'fsync_dir', None))
            return original_sync(fd)
        with patch.object(os, 'write', write), patch.object(os, 'fsync', sync):
            attempt = self.start()
            w.mark_native_enter(attempt); events.append(('native', None))
        self.assertEqual(events[0][0], 'fsync_dir')  # New slot directory's parent.
        self.assertEqual([item[0] for item in events[-3:]], ['fsync_file', 'fsync_dir', 'native'])
        self.assertEqual((attempt.directory / 'started.json').read_bytes(), attempt.started_bytes)
        self.assertGreater(sum(item[0] == 'write' for item in events), 1)

    def test_failed_claim_write_consumed_and_restart_rejected(self):
        for fault in ('partial_write', 'zero_write', 'file_fsync', 'parent_fsync'):
            with self.subTest(fault=fault):
                # Fresh independent fixture family output for each injected failure.
                self.setUp()
                before = len(list(Path('/proc/self/fd').iterdir()))
                real_write, real_sync = os.write, os.fsync
                writes, syncs = [0], [0]
                def write(fd, body):
                    writes[0] += 1
                    if fault == 'zero_write': return 0
                    if fault == 'partial_write':
                        if writes[0] == 1: return real_write(fd, body[:13])
                        raise OSError('invented partial write fault')
                    return real_write(fd, body)
                def sync(fd):
                    syncs[0] += 1
                    if ((fault == 'file_fsync' and stat.S_ISREG(os.fstat(fd).st_mode)) or
                        (fault == 'parent_fsync' and syncs[0] == 3)):
                        raise OSError('invented fsync fault')
                    return real_sync(fd)
                with patch.object(os, 'write', write), patch.object(os, 'fsync', sync):
                    with self.assertRaises((ValueError, OSError)):
                        self.start()
                first = self.plan['audio_order'][0]
                self.assertIn(first, self.bindings['qwen06']._state['consumed'])
                self.assertTrue(self.bindings['qwen06']._state['stopped'])
                self.assertTrue((self.current['qwen06'] / first / 'started.json').exists())
                with self.assertRaises(ValueError): self.start(audio_id=self.plan['audio_order'][1])
                with self.assertRaises(ValueError): self.bind('qwen06')
                self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)

    def test_new_directory_parent_fsync_failure_stops_before_start(self):
        with patch.object(os, 'fsync', side_effect=OSError('invented grandparent fault')):
            with self.assertRaises(OSError): self.start()
        self.assertTrue(self.bindings['qwen06']._state['stopped'])
        first = self.plan['audio_order'][0]
        self.assertTrue((self.current['qwen06'] / first).is_dir())
        self.assertFalse((self.current['qwen06'] / first / 'started.json').exists())
        with self.assertRaises(ValueError): self.bind('qwen06')

    def test_claim_symlink_directory_and_alternate_path_rejected(self):
        first = self.plan['audio_order'][0]
        (self.current['qwen06'] / first).symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(FileExistsError): self.start()
        self.assertFalse((self.root / 'started.json').exists())
        with self.assertRaises(ValueError): self.start('sensevoice', directory=self.root / 'alternate')

    def test_native_single_boundary_cannot_enter_twice(self):
        attempt = self.start(); w.mark_native_enter(attempt)
        with self.assertRaises(ValueError): w.mark_native_enter(attempt)
        self.assertTrue(attempt.binding._state['stopped'])
        self.assertFalse((attempt.directory / 'completed.json').exists())

    def test_completion_without_native_return_or_without_start_rejected(self):
        attempt = self.start(); row, raw = self.result('qwen06')
        with self.assertRaises(ValueError): w.emit_completion(attempt, row=row, raw=raw)
        self.assertFalse((attempt.directory / 'completed.json').exists())
        with self.assertRaises(ValueError): w.emit_completion(None, row=row, raw=raw)

    def test_bad_capture_stops_all_later_eligible(self):
        attempt = self.start(); row, raw = self.returned(attempt)
        raw['native_capture']['input']['processed_end_frame'] -= 1
        with self.assertRaises(ValueError): w.emit_completion(attempt, row=row, raw=raw)
        self.assertTrue((attempt.directory / 'raw-native-output.json').exists())
        self.assertFalse((attempt.directory / 'completed.json').exists())
        with self.assertRaises(ValueError): self.start(audio_id=self.plan['audio_order'][1])

    def test_failed_completion_retention_stops_and_consumes(self):
        attempt = self.start(); row, raw = self.returned(attempt)
        real_sync = os.fsync
        def sync(fd):
            if os.readlink('/proc/self/fd/' + str(fd)).endswith('/completed.json'):
                raise OSError('invented completion persistence fault')
            return real_sync(fd)
        with patch.object(os, 'fsync', sync):
            with self.assertRaises(OSError): w.emit_completion(attempt, row=row, raw=raw)
        self.assertTrue(attempt.binding._state['stopped'])
        self.assertFalse(attempt._state['completed'])
        with self.assertRaises(ValueError): w.emit_completion(attempt, row=row, raw=raw)
        with self.assertRaises(ValueError): self.start(audio_id=self.plan['audio_order'][1])

    def test_reordering_and_different_request_forbidden(self):
        with self.assertRaises(ValueError): self.start(audio_id=self.plan['audio_order'][1])
        self.assertTrue(self.bindings['qwen06']._state['stopped'])
        binding = self.bindings['sensevoice']; aid = self.plan['audio_order'][0]; row = self.input['audio'][0]
        with self.assertRaises(ValueError):
            w.emit_start(binding, self.current['sensevoice'] / aid, wav_path=self.wavs[row['wav_sha256']],
                expected_file_sha256=row['wav_sha256'], audio_id=aid, request_id='different-retry-id')

    def test_mutated_wav_between_start_and_native_never_calls(self):
        attempt = self.start(); path = attempt.wav_path
        path.chmod(0o600); path.write_bytes(path.read_bytes()[:-2] + b'xx'); path.chmod(0o444)
        with self.assertRaises(ValueError): w.mark_native_enter(attempt)
        self.assertFalse(attempt._state['native_entered'])
        self.assertTrue(attempt.binding._state['stopped'])

    def test_prior_active_start_prevents_next_and_new_binding(self):
        attempt = self.start()
        with self.assertRaises(ValueError): self.bind('qwen06')
        with self.assertRaises(ValueError): self.start(audio_id=self.plan['audio_order'][1])
        self.assertTrue(attempt.binding._state['stopped'])

    def test_prepared_load_not_recognition(self):
        self.assertTrue(all(binding._state['next'] == 0 for binding in self.bindings.values()))
        self.assertFalse(any(self.root.rglob('started.json')))
        self.assertFalse(any(self.root.rglob('completed.json')))

    def test_uncertain_exclusive_open_conservatively_consumed(self):
        real_open = os.open
        def failure(path, *args, **kwargs):
            if path == 'started.json': raise OSError('invented uncertain exclusive create')
            return real_open(path, *args, **kwargs)
        before = len(list(Path('/proc/self/fd').iterdir()))
        with patch.object(os, 'open', failure):
            with self.assertRaises(OSError): self.start()
        first = self.plan['audio_order'][0]
        self.assertIn(first, self.bindings['qwen06']._state['consumed'])
        self.assertTrue((self.current['qwen06'] / first).exists())
        self.assertTrue(self.bindings['qwen06']._state['stopped'])
        self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)
        with self.assertRaises(ValueError): self.bind('qwen06')

    def test_close_uncertainty_after_durable_write_never_allows_native(self):
        real_close = os.close
        fired = [False]
        def failure(fd):
            path = os.readlink('/proc/self/fd/' + str(fd))
            real_close(fd)
            if not fired[0] and path.endswith('/started.json'):
                fired[0] = True
                raise OSError('invented close uncertainty')
        before = len(list(Path('/proc/self/fd').iterdir()))
        with patch.object(os, 'close', failure):
            with self.assertRaises(OSError): self.start()
        self.assertTrue(fired[0])
        self.assertTrue(self.bindings['qwen06']._state['stopped'])
        self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)
        self.assertTrue((self.current['qwen06'] / self.plan['audio_order'][0] / 'started.json').exists())

    def test_decode_budget_proof_has_exact_caps_and_actual_bindings(self):
        proof = {'schema': 'k1-asr18-predecode-package-budget-v1',
            'status': 'passed_before_any_audio_forward', 'new_native_calls': 0,
            'eligible_clip_count': len(self.job['clips']), 'whisper_full_prefix_steps_per_clip': 224,
            'transcript_reserve_characters_each_copy': 32768, 'transcript_copies_per_clip': 2,
            'extra_json_reserve_bytes': c.MIB, 'recognizer_policy_sha256': c.digest(c.canonical(self.policy)),
            'worker_input_binding_sha256': c.digest(c.canonical(self.input)),
            'execution_plan_sha256': c.digest(c.canonical(self.plan)), 'job_sha256': c.digest(c.canonical(self.job)),
            'compressed_bytes': 12 * c.MIB, 'expanded_member_bytes': 32 * c.MIB,
            'actual_prepared_members': 1, 'worst_fixture_members': 4096}
        def validate(value):
            return c.validate_decode_proof(value, job=self.job, execution_plan=self.plan,
                input_binding=self.input, policy_path=self.policy_path, family='qwen06')
        self.assertEqual(validate(proof), proof)
        for key, bad in (('compressed_bytes', 12 * c.MIB + 1), ('expanded_member_bytes', 32 * c.MIB + 1),
                         ('new_native_calls', True), ('eligible_clip_count', 6),
                         ('actual_prepared_members', 0), ('worst_fixture_members', 4097),
                         ('recognizer_policy_sha256', 'f' * 64), ('whisper_full_prefix_steps_per_clip', 223)):
            mutated = {**proof, key: bad}
            with self.subTest(key=key), self.assertRaises(ValueError): validate(mutated)

    def test_exact_local_imports_without_campaign_history(self):
        self.assertEqual(Path(c.__file__).resolve(), HERE / 'campaign_contract.py')
        self.assertEqual(Path(w.__file__).resolve(), HERE / 'worker_receipts.py')
        # Test the receipt import closure in a genuinely fresh interpreter.
        # Other tests legitimately import the trusted host into this test process.
        program = '''import pathlib,sys
sys.path.insert(0,sys.argv[1])
import test_asr18_receipts as tests
root=pathlib.Path(sys.argv[1])
if pathlib.Path(tests.c.__file__).resolve()!=root/'campaign_contract.py':raise SystemExit(2)
if pathlib.Path(tests.w.__file__).resolve()!=root/'worker_receipts.py':raise SystemExit(3)
forbidden=('remaining35','qwen12-native-admission-v1/campaign_contract','hosted_run.py')
if any(any(word in str(getattr(module,'__file__','')) for word in forbidden) for module in list(sys.modules.values())):raise SystemExit(4)
'''
        optimize = ['-' + 'O' * sys.flags.optimize] if sys.flags.optimize else []
        observed = subprocess.run([sys.executable, '-I', '-B', *optimize, '-c', program, str(HERE)],
            capture_output=True, timeout=20, env={'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent'})
        self.assertEqual(observed.returncode, 0, observed.stderr.decode())
        self.assertEqual(observed.stdout, b'')


if __name__ == '__main__':
    unittest.main()
