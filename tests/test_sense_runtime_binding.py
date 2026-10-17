"""Stdlib-only synthetic fixtures. No model, acoustic data, weights or downloads."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research/qwen6_asr'
sys.path[:0] = [str(STAGE / 'core'), str(STAGE)]
from asr_stage import adapters as a, execution as e
import sense_binding as b

IDS = [1, 2, 3, 4, 5]
PIECES = {0: '<unk>', 1: '<|zh|>', 2: '<|NEUTRAL|>', 3: '<|Speech|>',
          4: '<|woitn|>', 5: 'fixture', 25009: '<|EMO_UNKNOWN|>'}
TEXT = ''.join(PIECES[i] for i in IDS)


class FakeSP:
    def vocab_size(self): return 25055
    def unk_id(self): return 0
    def id_to_piece(self, i): return PIECES.get(i, 'fixture')
    def is_unknown(self, i): return i == 0
    def decode(self, ids): return ''.join(self.id_to_piece(i) for i in ids)


class FakeTokenizer:
    def __init__(self): self.sp = FakeSP()
    def get_vocab_size(self): return self.sp.vocab_size()
    def decode(self, ids): return self.sp.decode(ids)


class FakeModel:
    def __init__(self): self.calls = 0; self.after_decode = lambda: None
    def eval(self): pass
    def inference(self, **kw):
        self.calls += 1
        text = kw['tokenizer'].decode(list(IDS))
        self.after_decode()
        return ([{'key': kw['key'][0], 'text': text}], {})


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / 'nondefault-runtime/models/sensevoice' / b.TOKENIZER_NAME
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b'synthetic tokenizer bytes; not a model')
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(b, 'TOKENIZER_SHA256', hashlib.sha256(self.file.read_bytes()).hexdigest()))
        self.default = self.root / 'missing-default' / b.TOKENIZER_NAME
        self.stack.enter_context(patch.object(b, 'tokenizer_path', return_value=self.default))
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.wrapper = types.SimpleNamespace(model=self.model, kwargs={'tokenizer': self.tokenizer, 'frontend': object()})
        descriptor = {'wav': {'sha256': 'a'*64, 'bytes': 46},
                      'raw_pcm16': {'frame_count': 1}, 'float32_pcm': {'fixture': True}}
        raw = json.dumps(descriptor).encode()
        self.bound = types.SimpleNamespace(opaque_id='clip-000001', descriptor=descriptor,
            descriptor_json=raw, descriptor_sha256=hashlib.sha256(raw).hexdigest())
        self.inputs = {self.bound.opaque_id: {'descriptor': descriptor, 'binding_sha256': self.bound.descriptor_sha256}}
        self.runtime = a.PreparedRuntime(a.SENSE, self.wrapper, {}, {'decoder_manifest_sha256': 'b'*64}, set(), self.inputs, self.file)
        self.stack.enter_context(patch.object(a, 'require_verified_receipt'))
        self.stack.enter_context(patch.object(a, 'make_pcm_array', return_value=[0.0]))
        self.stack.enter_context(patch.object(a, 'verify_pcm_array'))
        self.stack.enter_context(patch.dict(sys.modules, {'torch': types.SimpleNamespace(inference_mode=contextlib.nullcontext)}))

    def infer(self):
        return a.infer_sense_once_after_approval(self.runtime, self.bound, 'b'*64)

    def test_nondefault_runtime_without_default_directory(self):
        self.assertFalse(self.default.parent.exists())
        row, evidence = self.infer()
        self.assertEqual(row['raw_text'], 'fixture')
        self.assertTrue(evidence['token_binding']['raw_redecode_matches_capture'])
        self.assertEqual(self.model.calls, 1)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate candidate'):
            self.infer()
        self.assertEqual(self.model.calls, 1)

    def test_runtime_path_survives_working_directory_change(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            row, _ = self.infer()
        finally:
            os.chdir(previous)
        self.assertEqual(row['raw_text'], 'fixture')
        self.assertEqual(self.model.calls, 1)

    def test_changed_tokenizer_before_decode_rejected_without_call(self):
        self.file.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'tokenizer bytes changed'): self.infer()
        self.assertEqual(self.model.calls, 0)

    def test_changed_tokenizer_during_decode_is_failed_unbound_and_no_retry(self):
        self.model.after_decode = lambda: self.file.write_bytes(b'changed')
        with self.assertRaises(a.SensePostprocessError) as caught: self.infer()
        failure = caught.exception.evidence
        self.assertEqual(failure['status'], 'failed_unbound')
        self.assertFalse(failure['binding_verified'])
        self.assertEqual(failure['raw_decoded_output']['text'], TEXT)
        self.assertEqual(failure['token_evidence']['token_ids'], IDS)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.file.write_bytes(b'synthetic tokenizer bytes; not a model')
        with self.assertRaisesRegex(RuntimeError, 'Duplicate candidate'): self.infer()
        self.assertEqual(self.model.calls, 1)

    def test_missing_runtime_path_never_falls_back(self):
        self.runtime.tokenizer_file = None
        with self.assertRaisesRegex(RuntimeError, 'absolute verified tokenizer'): self.infer()
        self.assertEqual(self.model.calls, 0)

    def test_relative_runtime_path_rejected(self):
        self.runtime.tokenizer_file = Path('relative.model')
        with self.assertRaisesRegex(RuntimeError, 'absolute verified tokenizer'): self.infer()
        self.assertEqual(self.model.calls, 0)

    def test_symlink_tokenizer_rejected(self):
        link = self.root / 'link.model'; link.symlink_to(self.file)
        self.runtime.tokenizer_file = link
        with self.assertRaisesRegex(ValueError, 'tokenizer bytes changed'): self.infer()
        self.assertEqual(self.model.calls, 0)

    def test_vocab_and_unknown_semantics_still_checked(self):
        for method, value in [('vocab_size', 25054), ('unk_id', 1), ('is_unknown', False)]:
            with self.subTest(method=method), patch.object(self.tokenizer.sp, method, return_value=value):
                with self.assertRaisesRegex(ValueError, 'semantics mismatch'): self.infer()
        self.assertEqual(self.model.calls, 0)

    def test_token_redecode_mismatch_stays_failure(self):
        with patch.object(b, 'bound_sense_transcript', wraps=b.bound_sense_transcript):
            self.model.after_decode = lambda: setattr(self.tokenizer.sp, 'decode', lambda ids: 'mismatch')
            with self.assertRaises(a.SensePostprocessError) as caught: self.infer()
        self.assertRegex(str(caught.exception.__cause__), 're-decode mismatch')
        self.assertEqual(self.model.calls, 1)

    def test_legacy_binding_call_keeps_default_and_still_checks_bytes(self):
        capture = {'token_ids': IDS, 'decoded': TEXT, 'token_scope': b.TOKEN_SCOPE}
        with self.assertRaisesRegex(ValueError, 'tokenizer bytes changed'):
            b.bound_sense_transcript(TEXT, capture, self.tokenizer.sp)
        with patch.object(b, 'tokenizer_path', return_value=self.file):
            row, _, _ = b.bound_sense_transcript(TEXT, capture, self.tokenizer.sp)
        self.assertEqual(row['raw_text'], 'fixture')

    def test_capture_association_failure_keeps_bounded_unbound_text(self):
        with patch.object(a.CapturingTokenizer, 'evidence', side_effect=ValueError('fixture capture mismatch')):
            with self.assertRaises(a.SensePostprocessError) as caught: self.infer()
        failure = caught.exception.evidence
        self.assertEqual(failure['raw_decoded_output']['text'], TEXT)
        self.assertNotIn('token_evidence', failure)
        self.assertFalse(failure['binding_verified'])
        self.assertEqual(self.model.calls, 1)

    def test_pcm_validation_failure_keeps_unbound_evidence(self):
        with patch.object(a, 'verify_pcm_array', side_effect=ValueError('fixture PCM mutation')):
            with self.assertRaises(a.SensePostprocessError) as caught: self.infer()
        self.assertFalse(caught.exception.evidence['binding_verified'])
        self.assertEqual(self.model.calls, 1)

    def test_loader_retains_absolute_path_used_for_construction(self):
        captured = {}
        def model_factory(**kw): captured.update(kw); return self.wrapper
        torch = types.SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None,
                                      load=lambda *a, **kw: {})
        config = {'tokenizer': 'SentencepiecesTokenizer', 'tokenizer_conf': {}, 'frontend': 'WavFrontend', 'frontend_conf': {}}
        modules = {'torch': torch, 'yaml': types.SimpleNamespace(safe_load=lambda raw: config),
                   'funasr': types.SimpleNamespace(AutoModel=model_factory),
                   'funasr.utils': types.SimpleNamespace(fbank=types.SimpleNamespace(_HAS_TORCHAUDIO=False, _knf=object()))}
        with patch.dict(sys.modules, modules), patch.object(a, '_preload', return_value=({}, 'schema', {}, {}, {}, {})), \
             patch.object(a, 'validate_asset_lock', return_value={}), patch.object(a, 'claim_fresh_model_process'), \
             patch.object(a, 'read_small_locked', return_value=b'fixture'), patch.object(a, 'validate_sense_geometry'), \
             patch.object(a, 'exact_apply', return_value={}):
            runtime = a.load_sense_after_approval(self.file.parent, {}, '', {}, '', {}, '', b'')
        self.assertTrue(runtime.tokenizer_file.is_absolute())
        self.assertEqual(runtime.tokenizer_file, self.file)
        self.assertEqual(captured['tokenizer_conf']['bpemodel'], str(self.file))
        self.assertEqual(runtime.receipt['tokenizer_identity']['sha256'], b.TOKENIZER_SHA256)

    def test_execution_persists_failure_evidence_and_never_success_label(self):
        self._execute_failure('error')

    def test_execution_preserves_postprocess_deadline_status(self):
        self._execute_failure('timeout')

    def test_execution_accepts_nondefault_bound_success(self):
        self._execute_failure('success')

    def _execute_failure(self, expected_status):
        contract = {'model_id': a.SENSE, 'decoder_manifest_sha256': 'b'*64,
            'candidate_count': 1, 'experiment_id': 'fixture', 'source_manifest_sha256': 'c'*64,
            'input_freeze_sha256': 'd'*64, 'run_id': 'fixture'}
        manifest = json.dumps({k:contract[k] for k in ('experiment_id','source_manifest_sha256','input_freeze_sha256')}).encode()
        plan = {'schema': 'asr6-run-plan-v1', 'model_id': a.SENSE, 'model_root': 'fixture', 'pcm_root': 'fixture',
            'asset_lock': {}, 'asset_lock_sha256': 'e'*64, 'source_lock': {}, 'source_lock_sha256': 'f'*64,
            'contract': contract, 'contract_sha256': e.canonical_sha(contract),
            'decoder_manifest_sha256': 'b'*64, 'clip_seconds': 120}
        if expected_status == 'timeout':
            self.stack.enter_context(patch.object(a, 'verify_pcm_array', side_effect=e.ClipDeadline('fixture deadline')))
        elif expected_status == 'error':
            self.model.after_decode = lambda: self.file.write_bytes(b'changed')
        directory = self.root / 'retained'
        with patch.object(e, 'validate_decoder_manifest', return_value=self.inputs):
            summary = e.execute_primary(plan, directory, manifest, loader=lambda *args:self.runtime,
                binder=lambda *args:self.bound, check_environment=False)
        self.assertEqual(summary['status'], 'complete' if expected_status == 'success' else 'decode_stopped')
        self.assertEqual(summary['success'], 1 if expected_status == 'success' else 0)
        self.assertEqual(summary['attempted'], 1)
        self.assertEqual(summary['retries'], 0)
        sidecar = json.loads((directory/'clip-000001.decoder.json').read_text())
        self.assertEqual(sidecar['outcome']['status'], expected_status)
        if expected_status == 'success':
            self.assertEqual(sidecar['outcome']['raw_text'], 'fixture')
            self.assertTrue(sidecar['evidence']['token_binding']['raw_redecode_matches_capture'])
        else:
            self.assertIsNone(sidecar['outcome']['raw_text'])
            self.assertEqual(sidecar['evidence']['status'], 'failed_unbound')
            self.assertEqual(sidecar['evidence']['raw_decoded_output']['text'], TEXT)
        receipt = json.loads((directory/'clip-000001.receipt.json').read_text())
        self.assertTrue(receipt['model_forward_completed'])
        self.assertEqual(receipt['decoder_evidence_sha256'], hashlib.sha256((directory/'clip-000001.decoder.json').read_bytes()).hexdigest())
        self.assertEqual(self.model.calls, 1)


if __name__ == '__main__':
    unittest.main()
