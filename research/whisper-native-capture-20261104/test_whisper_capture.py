"""Invented stdlib-only fixtures; no downloads, ML imports or model execution.

These tests demonstrate schema/state-machine consistency, not authenticity.
"""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import wave

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(HERE))
import whisper_capture as w

CONTRACT_PATH = WORKSPACE/'kws-three-asr-pilot-20261104/tools/asr_native_contracts.py'
TOOLS = WORKSPACE/'kws-audit-streaming-20261104/tools'
SINK = WORKSPACE/'kws-native-capture-20261104/research/qwen-native-capture-20261104/native_sink.py'
contracts = w.import_exact('asr_native_contracts', CONTRACT_PATH)
w.import_exact('speech_label_admission', TOOLS/'speech_label_admission.py')
w.import_exact('speech_asr_consensus', TOOLS/'speech_asr_consensus.py')
w.import_exact('generation_capture', SINK.parent/'generation_capture.py')
sink_module = w.import_exact('_whisper_test_shared_sink', SINK)


def effective():
    controls = dict(transcribe=10, translate=11, sot=12, sot_prev=13,
                    sot_lm=14, no_speech=15, language=16, no_timestamps=17)
    return dict(execution=dict(contracts.EXECUTION), native_request=dict(contracts.WHISPER_REQUEST),
        decoder='greedy', n_text_ctx=16, effective_sample_len=8,
        tokenizer_sha256=w.sha(b'invented tokenizer'), non_speech_token_ids=[6],
        control_token_ids=controls, resolved_suppress_token_ids=[6, 10, 11, 12, 13, 14, 15],
        space_token_ids=[20], initial_blank_suppress_token_ids=[19, 20], eot_token_id=19,
        initial_token_ids=[12, 16, 10, 17])


def config(eff, source_raw, model_raw, family='whisper'):
    adapter = {'whisper': contracts.WHISPER_ADAPTER, 'sensevoice': contracts.SENSE_ADAPTER,
               'qwen3-asr': contracts.QWEN_ADAPTER}[family]
    return dict(schema_version=2, evidence_class=contracts.CONFIG_CLASS, adapter_id=adapter,
        adapter_version=1, model_family=family, source_lock_sha256=w.sha(source_raw),
        asset_lock_sha256=w.sha(model_raw), conditioning_policy=contracts.CONDITIONING_POLICY,
        execution=dict(contracts.EXECUTION), input_policy=dict(sample_rate=16000, channels=1,
            max_audio_frames=480000, pcm_format='float32_pcm16_scale_32768',
            preprocessing='zero_pad_only_480000' if family == 'whisper' else 'full_input'),
        native_request=copy.deepcopy(contracts.REQUESTS[adapter]), effective_config=eff,
        effective_config_sha256=w.sha(w.canonical(eff)))


def policy(ident):
    q = dict(execution=dict(contracts.EXECUTION), native_request=dict(contracts.QWEN_REQUEST),
        generation_mode='greedy_search', num_beams=1, num_return_sequences=1,
        forced_bos_token_id=None, forced_eos_token_id=None, suppress_tokens=None,
        begin_suppress_tokens=None, bad_words_ids=None, force_words_ids=None,
        sequence_bias=None, stop_strings=None, custom_logits_processors=[],
        custom_stopping_criteria=[], resolved_generation_config_sha256='a'*64)
    s = dict(execution=dict(contracts.EXECUTION), native_request=dict(contracts.SENSE_REQUEST),
        decoding='full_input_greedy_ctc', vad_used=False, rich_postprocess_applied=False,
        vocabulary_size=25055, lexical_unknown_id=0, metadata_prefix_tokens=4,
        tokenizer_sha256='b'*64, parser_contract_sha256='c'*64, frontend_config_sha256='d'*64)
    rows = [copy.deepcopy(ident)]
    for index, (family, eff) in enumerate((('qwen3-asr', q), ('sensevoice', s))):
        cfg = config(eff, b'fixture source', b'fixture model', family)
        rows.append(dict(recognizer_id='fixture-'+family, provider_id='fixture', model_id=family,
            model_revision='invented', model_family=family, model_sha256=str(index+1)*64,
            native_decode_config=cfg, native_decode_config_sha256=w.sha(w.canonical(cfg))))
    return dict(schema_version=2, evidence_class='speech-three-asr-recognizer-policy-v2',
        normalization='fixed-punctuation-removal-only-v1', frozen_before_recognition=True,
        target_conditioning_allowed=False, recognizers=rows)


def termination(steps, eff):
    # Invented decoder emits the observed pre-finalize sequence; the production
    # adapter is separately pinned and supplies its original termination().
    ids = steps[-1]['token_ids'][len(eff['initial_token_ids']):]
    return dict(kind='whisper_greedy_pre_finalize', capture_stage='decoder_update_before_finalize',
        initial_token_ids=eff['initial_token_ids'], generated_token_ids=ids,
        step_count=len(steps), pre_finalize_token_count=len(steps[-1]['token_ids']),
        completed=steps[-1]['completed'], stop_reason='eot', truncated=False)


class FakeLoaded:
    def __init__(self, events):
        self.events = events
        self.effective = effective()
        self.config = None
        self.torch, self.model = object(), object()
        self.raise_during_decode = False
        self.mutate_after_decode = False
        self.corrupt_trace = False
        self.calls = 0
    def make_task(self):
        self.events.append('make_task_no_forward')
        return object()
    def decode(self, pcm, frames, directory):
        self.events.append('decode')
        self.calls += 1
        if not (directory.parent/'started.json').is_file():
            raise RuntimeError('start was not persisted before decode')
        if self.raise_during_decode:
            raise RuntimeError('invented decoder failure')
        ids = self.effective['initial_token_ids']
        steps = [dict(step=1, token_ids=ids+[1], completed=False),
                 dict(step=2, token_ids=ids+[1, 19], completed=True)]
        inp = dict(sample_rate=16000, channels=1, audio_frames=frames,
            processed_start_frame=0, processed_end_frame=frames,
            padding_frames=480000-frames, trimmed_frames=0,
            preprocessing='zero_pad_only_480000', pcm_format='float32_pcm16_scale_32768')
        cfg = self.config
        cap = dict(schema_version=2, evidence_class=contracts.CAPTURE_CLASS,
            native_config_sha256=w.sha(w.canonical(cfg)), source_lock_sha256=cfg['source_lock_sha256'],
            asset_lock_sha256=cfg['asset_lock_sha256'], effective_config=copy.deepcopy(self.effective),
            effective_config_sha256=cfg['effective_config_sha256'], input=inp,
            termination=termination(steps, self.effective), transcript_postcorrection_applied=False,
            target_context_received=False, peer_transcript_received=False,
            acoustic_verbatim_completeness='unknown')
        result = dict(raw_text='invented fixture', native_capture=cap, trace_files=2,
            decoder_completion_validated=True, full_input_processed=True)
        for index, step in enumerate(steps, 1):
            if self.corrupt_trace and index == 2:
                step = dict(step, token_ids=ids+[2, 19])
            w.persist(directory/('update-%03d.json' % index), w.canonical(step))
        w.persist(directory/'pre-finalize.json', w.canonical(dict(token_ids=ids+[1, 19],
            last_update_completed=True, trace_count=2)))
        w.persist(directory/'input-capture.json', w.canonical(dict(native_input=inp, pcm16_sha256=w.sha(pcm))))
        w.persist(directory/'raw-output.json', w.canonical(result))
        if self.mutate_after_decode:
            self.effective['n_text_ctx'] = 32
        return result


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = []
        self.loaded = FakeLoaded(self.events)
        self.tokenizer = self.root/'tokenizer'
        self.tokenizer.write_bytes(b'invented tokenizer')
        self.checkpoint = self.root/'small.pt'
        self.checkpoint.write_bytes(b'invented checkpoint, not model weights')
        self.paths = {w.TOKENIZER_MEMBER: self.tokenizer}
        self.bodies = {w.TOKENIZER_MEMBER: self.tokenizer.read_bytes()}
        asset = dict(path='checkpoint/small.pt', **w.identity(self.checkpoint))
        self.model = dict(schema='retained-asr-model-lock-v1', provider_id='openai',
            model_id='openai/whisper-small', model_revision='invented', model_family='whisper',
            assets=[asset], model_sha256=w.sha(w.canonical([asset])))
        self.source = dict(schema='retained-asr-source-lock-v1', executed_head_sha='f'*40,
            files=[dict(path=w.TOKENIZER_MEMBER, **w.identity(self.tokenizer))])
        self.model_raw, self.source_raw = w.canonical(self.model), w.canonical(self.source)
        self.cfg = config(effective(), self.source_raw, self.model_raw)
        self.loaded.config = copy.deepcopy(self.cfg)
        self.ident = {key: self.model[key] for key in ('provider_id', 'model_id', 'model_revision', 'model_family', 'model_sha256')}
        self.ident.update(recognizer_id='whisper-small', native_decode_config=self.cfg,
                          native_decode_config_sha256=w.sha(w.canonical(self.cfg)))
        self.adapter = NS(observed_effective=lambda *args: copy.deepcopy(self.loaded.effective), termination=termination)
        self.scope = NS(verify_runtime_scope=lambda: self.events.append('scope'))
        self.p = w._PreparedWhisper(self.root, self.loaded, self.scope, self.adapter, contracts,
            sink_module, self.paths, self.bodies, self.checkpoint, self.model_raw,
            self.source_raw, self.ident, 2)
        self.wav = self.root/'fixture.wav'
        with wave.open(str(self.wav), 'wb') as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            f.writeframes(b'\x00\x00\x01\x00')
        self.wav_sha = w.sha(self.wav.read_bytes())
    def bind(self, p=None, value=None):
        p = p or self.p
        value = value or policy(p.identity)
        p.bind_policy(w.canonical(value), w.sha(w.canonical(value)))
    def infer(self, p=None, opaque='clip-000001', request='request-1'):
        return (p or self.p).infer(self.wav, opaque, self.wav_sha, 'source-1', request)
    def test_python38_syntax(self):
        ast.parse((HERE/'whisper_capture.py').read_text(), feature_version=(3, 8))
        ast.parse(Path(__file__).read_text(), feature_version=(3, 8))
    def test_no_ml_imports(self):
        self.assertFalse(any(name in sys.modules for name in ('torch', 'numpy', 'whisper')))
    def test_requires_policy_before_decode(self):
        with self.assertRaises(ValueError): self.infer()
        self.assertEqual(self.loaded.calls, 0)
    def test_complete_strict_chain_and_dependencies(self):
        self.bind()
        self.assertEqual(self.loaded.calls, 0)
        receipt = self.infer()
        base = self.root/'capture/clip-000001'
        start = w.parse(w.read(base/'started.json'))
        output = w.parse(w.read(base/'raw-output.json'))
        self.assertEqual(start['schema'], 'retained-asr-started-v1')
        self.assertEqual(receipt['schema'], 'retained-asr-completed-v1')
        self.assertEqual(output['schema'], 'retained-asr-raw-transcript-v1')
        self.assertEqual(receipt['started_sha256'], w.sha(w.read(base/'started.json')))
        self.assertEqual(receipt['raw_output_sha256'], w.sha(w.read(base/'raw-output.json')))
        self.assertEqual(receipt['native_capture']['acoustic_verbatim_completeness'], 'unknown')
        contracts.validate_native_capture(self.cfg, receipt['native_capture'])
        for key in ('source_id', 'request_id', 'file_sha256', 'pcm_sha256', 'audio_frames',
                    'recognizer_id', 'recognizer_identity_sha256', 'recognizer_policy_sha256',
                    'model_lock_sha256', 'source_lock_sha256', 'native_decode_config_sha256'):
            self.assertEqual(start[key], receipt[key])
        self.assertEqual(receipt['loaded_model_assets'], self.model['assets'])
        self.assertEqual(receipt['executed_source_files'], self.source['files'])
        self.assertEqual(output['actual_text'], 'invented fixture')
        self.assertEqual(len(output['original_native_decoder_evidence']['native_update_trace']), 2)
        self.assertEqual(w.read(self.root/'capture'/w.TOKENIZER_MEMBER), b'invented tokenizer')
    def test_policy_identity_mutation_rejected(self):
        value = policy(self.ident)
        value['recognizers'][0]['model_revision'] = 'different'
        with self.assertRaises(ValueError): self.bind(value=value)
        self.assertTrue(self.p.failed)
        self.assertEqual(self.loaded.calls, 0)
    def test_source_mutation_before_start(self):
        self.bind()
        self.tokenizer.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.infer()
        self.assertTrue(self.p.failed)
        self.assertFalse((self.root/'capture/clip-000001/started.json').exists())
    def test_checkpoint_mutation_rejected(self):
        self.checkpoint.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.bind()
    def test_duplicate_attempt_poisoned(self):
        self.bind(); self.infer()
        with self.assertRaises(ValueError): self.infer(request='request-2')
        self.assertTrue(self.p.failed)
        self.assertEqual(self.loaded.calls, 1)
    def test_path_traversal_rejected(self):
        self.bind()
        with self.assertRaises(ValueError): self.infer(opaque='../escape')
        self.assertEqual(self.loaded.calls, 0)
    def test_decode_exception_preserves_start_no_completion_no_retry(self):
        self.bind(); self.loaded.raise_during_decode = True
        with self.assertRaises(RuntimeError): self.infer()
        self.assertTrue((self.root/'capture/clip-000001/started.json').is_file())
        self.assertFalse((self.root/'capture/clip-000001/completed.json').exists())
        with self.assertRaises(ValueError): self.infer(opaque='clip-000002', request='request-2')
        self.assertEqual(self.loaded.calls, 1)
    def test_post_decode_effective_change_rejected(self):
        self.bind(); self.loaded.mutate_after_decode = True
        with self.assertRaises(ValueError): self.infer()
        self.assertFalse((self.root/'capture/clip-000001/completed.json').exists())
    def test_corrupted_native_trace_rejected(self):
        self.bind(); self.loaded.corrupt_trace = True
        with self.assertRaises(ValueError): self.infer()
        self.assertFalse((self.root/'capture/clip-000001/completed.json').exists())
    def test_wrong_wav_hash_no_start(self):
        self.bind(); self.wav_sha = '0'*64
        with self.assertRaises(ValueError): self.infer()
        self.assertEqual(self.loaded.calls, 0)
    def test_duplicate_json_rejected(self):
        with self.assertRaises(ValueError): w.parse(b'{"a":1,"a":2}')
    def test_verified_imports_check_source_install_assets_before_package(self):
        repo = self.root/'repository'; repo.mkdir()
        pinned = repo/'fixture-pin'; pinned.write_bytes(b'invented pinned source')
        runtime = self.root/'runtime'
        expected_module = NS(__file__=str(runtime/'source/whisper/__init__.py'))
        events = []
        asset = dict(bytes=17, sha256='a'*64)
        def read_json(path):
            if Path(path).name == 'source-lock.json':
                return {'files': [dict(path='whisper/__init__.py', size_bytes=17, sha256='b'*64)]}
            if Path(path).name == 'installed-inventory.json':
                return {'files': {'invented.py': dict(bytes=17, sha256='c'*64)}}
            return {'files': {'small.pt': asset}}
        common = NS(read_json=read_json,
            verify_inventory=lambda path, expected: events.append(('inventory', str(path))),
            identity=lambda path: events.append(('asset', str(path))) or asset)
        scope = NS(verify_runtime_scope=lambda: events.append(('scope',)))
        with patch.object(w, 'PINS', {'fixture-pin': w.sha(pinned.read_bytes())}), \
             patch.object(w, 'MODEL_SHA', 'a'*64), \
             patch.object(w, 'import_exact', side_effect=lambda name, path: scope if name == '_whisper_native_scope' else common), \
             patch.dict(sys.modules, {'whisper': expected_module}), \
             patch.object(sys, 'path', list(sys.path)):
            self.assertIs(w.verify_prepare_imports(repo, runtime), expected_module)
        self.assertEqual([item[0] for item in events], ['scope', 'inventory', 'inventory', 'asset'])
        self.assertEqual(self.loaded.calls, 0)

    def test_optional_integration_source_closure_is_required_as_a_pair(self):
        repo = self.root/'repository'; integration = repo/'research/native-asr-integration-20261104'
        integration.mkdir(parents=True)
        pin = 'research/source-screen32-whisper-v1/source-lock.json'
        pinned = w.canonical(dict(source_revision='86098128c0b4f24f0e2aa2994de830614b474227', files=[]))
        def mock_read(path, maximum=32*1024**2):
            if str(path).endswith(pin):
                return pinned
            if Path(path).name in ('prepare_worker.py', 'zero_forward.py'):
                return Path(path).read_bytes()
            return b'invented source bytes'
        with patch.object(w, 'PINS', {pin: w.sha(pinned)}), patch.object(w, 'read', mock_read):
            with self.assertRaises(FileNotFoundError):
                w._source_snapshot(repo, self.root/'runtime', self.root/'native_sink.py', self.root/'tools')
            (integration/'prepare_worker.py').write_bytes(b'invented preparation entry')
            (integration/'zero_forward.py').write_bytes(b'invented guard')
            _, bodies = w._source_snapshot(repo, self.root/'runtime', self.root/'native_sink.py', self.root/'tools')
        self.assertEqual(bodies['bindings/capture/prepare_worker.py'], b'invented preparation entry')
        self.assertEqual(bodies['bindings/capture/zero_forward.py'], b'invented guard')

    def test_prepare_has_no_audio_forward_and_diagnostics_not_upgraded(self):
        runtime = self.root/'runtime'; (runtime/'model').mkdir(parents=True)
        checkpoint = runtime/'model/small.pt'; checkpoint.write_bytes(self.checkpoint.read_bytes())
        def loader(_runtime, output):
            self.events.append('load_without_audio')
            w.persist(output/'load-receipt.json', w.canonical(dict(audio_calls=0, status='invented original loader')))
            w.persist(output/'native-config.json', w.canonical(dict(invented_original_diagnostic=True)))
            return self.loaded
        self.adapter.FixedWhisper = loader
        target = self.root/'prepared'
        with patch.object(w, '_source_snapshot', return_value=(self.paths, self.bodies)), \
             patch.object(w, '_modules', return_value=(self.scope, NS(), contracts, self.adapter, sink_module)), \
             patch.object(w, 'MODEL_SHA', w.sha(checkpoint.read_bytes())):
            p = w.prepare(self.root, runtime, target, 'f'*40, SINK, TOOLS)
        self.assertEqual(self.loaded.calls, 0)
        self.assertEqual(w.parse(w.read(target/'prepare-receipt.json'))['audio_calls'], 0)
        self.assertEqual(w.parse(w.read(target/'original-loader-diagnostics/native-config.json')),
                         dict(invented_original_diagnostic=True))
        self.assertEqual(self.loaded.config, p.identity['native_decode_config'])
        self.bind(p)
        self.infer(p)
        self.assertEqual(self.loaded.calls, 1)


if __name__ == '__main__':
    unittest.main()
