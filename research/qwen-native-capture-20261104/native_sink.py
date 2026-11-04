"""Prospective Qwen native-v2 capture, never retrospective receipt promotion.

Caller supplies all three identities frozen before recognition. The actual
native resolver hook must match their exact snapshot before writing started.
No source-root registration, policy manufacture or model execution occurs here.
"""
import hashlib
import json
import math
import os
import re
from pathlib import Path

from asr_native_contracts import canonical_sha256, validate_native_capture
from speech_asr_consensus import validate_recognizer_policy
from generation_capture import canonical, require

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read_json(raw):
    require(type(raw) is bytes and 0 < len(raw) <= 8*1024**2, 'bounded original JSON bytes')
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,'duplicate JSON member')
            result[key]=value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON value')
    def finite(text):
        value=float(text)
        require(math.isfinite(value),'nonfinite JSON numeric overflow')
        return value
    return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid,parse_float=finite)

def persist(path, raw):
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())

class NativeEvidenceSink:
    def __init__(self, root, policy_raw, model_raw, source_raw, recognizer_id,
                 source_bodies, verified_assets):
        self.policy = read_json(policy_raw)
        identities = validate_recognizer_policy(self.policy)
        require(recognizer_id in identities, 'prefrozen recognizer identity')
        self.identity = identities[recognizer_id]
        self.config = self.identity['native_decode_config']
        self.model = read_json(model_raw); self.source = read_json(source_raw)
        require(self.model.get('schema') == 'retained-asr-model-lock-v1' and
                self.source.get('schema') == 'retained-asr-source-lock-v1', 'native retained lock schemas')
        require(re.fullmatch('[0-9a-f]{40}',self.source.get('executed_head_sha','')) is not None,
                'actual executed source head')
        for lock, key in ((self.model,'assets'),(self.source,'files')):
            items=lock.get(key)
            require(type(items) is list and items, 'nonempty retained lock inventory')
            paths=set()
            for item in items:
                path=Path(item['path'])
                require(not path.is_absolute() and '..' not in path.parts and item['path'] not in paths
                        and type(item['bytes']) is int and item['bytes'] > 0
                        and re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,
                        'bounded unique retained lock item')
                paths.add(item['path'])
        require(self.config['asset_lock_sha256'] == sha(model_raw) and
                self.config['source_lock_sha256'] == sha(source_raw), 'exact config lock bytes')
        for key in ('provider_id','model_id','model_revision','model_family','model_sha256'):
            require(self.model.get(key) == self.identity[key], 'model identity: '+key)
        require(canonical_sha256(self.model['assets']) == self.identity['model_sha256'] and
                canonical_sha256(verified_assets) == canonical_sha256(self.model['assets']),
                'actual verified loaded asset inventory')
        require(type(source_bodies) is dict and
                set(source_bodies) == {item['path'] for item in self.source['files']},
                'exact executed source byte inventory')
        for item in self.source['files']:
            body = source_bodies[item['path']]
            require(type(body) is bytes and len(body) == item['bytes'] and sha(body) == item['sha256'],
                    'executed source bytes changed')
        self.root = Path(root); self.root.mkdir(exist_ok=False)
        persist(self.root/'recognizer-policy.json',policy_raw)
        persist(self.root/'model-lock.json',model_raw)
        persist(self.root/'source-lock.json',source_raw)
        persist(self.root/'native-config.json',canonical(self.config))
        self.base = dict(recognizer_id=recognizer_id,
            recognizer_identity_sha256=canonical_sha256(self.identity),
            recognizer_policy_sha256=canonical_sha256(self.policy),
            model_lock_sha256=sha(model_raw), source_lock_sha256=sha(source_raw),
            native_decode_config_sha256=canonical_sha256(self.config))
        self.active = None
        self.seen = set()
        self.failed = False

    def begin(self, bound, source_id, request_id):
        require(type(bound.opaque_id) is str and
                re.fullmatch(r'clip-[0-9]{6}',bound.opaque_id) is not None,
                'safe exact single-segment opaque identity')
        require(self.active is None and not self.failed and bound.opaque_id not in self.seen,
                'single prospective native attempt')
        require(type(source_id) is str and source_id and type(request_id) is str and request_id,
                'source and request IDs must be supplied from frozen audio-only identity map')
        self.seen.add(bound.opaque_id)
        d = bound.descriptor
        require(d['operations'] == {'mix':False,'resample':False,'trim':False} and
                d['raw_pcm16']['sample_rate_hz'] == 16000 and d['raw_pcm16']['channels'] == 1,
                'complete bound native input')
        require(type(d['raw_pcm16']['frame_count']) is int and
                0 < d['raw_pcm16']['frame_count'] <= self.config['input_policy']['max_audio_frames'],
                'bound frame limit before recognition')
        directory = self.root/bound.opaque_id
        directory.mkdir(exist_ok=False)
        self.active = dict(directory=directory, started=None, snapshot=None, descriptor=d,
            common=dict(self.base,source_id=source_id,request_id=request_id,
                file_sha256=d['wav']['sha256'],pcm_sha256=d['raw_pcm16']['sha256'],
                audio_frames=d['raw_pcm16']['frame_count']))

    def before_continue(self, snapshot_raw, observed_execution, observed_controls):
        """Invoke only after actual native resolver returns, before generation."""
        require(self.active is not None and not self.failed and self.active['started'] is None,
                'exactly one prospective native start')
        expected = self.config['effective_config']
        require(self.config['adapter_id'] == 'qwen3-asr-transformers-greedy', 'Qwen native adapter')
        require(observed_execution == self.config['execution'] and
                observed_controls == {'custom_logits_processors':[], 'custom_stopping_criteria':[]},
                'actual execution/custom generation controls')
        require(sha(snapshot_raw) == expected['resolved_generation_config_sha256'],
                'actual resolver snapshot differs from pre-recognition policy')
        generation = read_json(snapshot_raw)
        for key in ('num_beams','num_return_sequences','forced_bos_token_id','forced_eos_token_id',
                    'suppress_tokens','begin_suppress_tokens','bad_words_ids','force_words_ids',
                    'sequence_bias','stop_strings'):
            require(key in generation and canonical(generation[key]) == canonical(expected[key]),
                    'actual resolved control: '+key)
        for key in ('do_sample','max_new_tokens','eos_token_id'):
            require(key in generation and canonical(generation[key]) == canonical(self.config['native_request'][key]),
                    'actual generation request: '+key)
        directory = self.active['directory']
        persist(directory/'resolved-generation-config.json',snapshot_raw)
        self.start_native(self.config)
        self.active['snapshot'] = snapshot_raw

    def start_native(self, actual_config):
        """After actual config verification, before any recognition forward."""
        require(self.active is not None and self.active['started'] is None and not self.failed,
                'one prospective start before recognition')
        require(canonical(actual_config) == canonical(self.config), 'actual config differs from prefrozen policy')
        raw = canonical(dict(self.active['common'],schema='retained-asr-started-v1'))
        persist(self.active['directory']/'started.json',raw)
        self.active['started'] = raw

    def complete_native(self, actual_text, capture, original_evidence):
        """Persist actual complete native result; never synthesize its capture."""
        require(self.active is not None and self.active['started'] is not None and not self.failed,
                'real prospective start required')
        require(type(actual_text) is str and actual_text and len(actual_text) <= 32768,
                'bounded actual transcript')
        validate_native_capture(self.config,capture)
        a = self.active
        frames = a['common']['audio_frames']
        require(capture['input']['audio_frames'] == frames and
                capture['input']['processed_start_frame'] == 0 and
                capture['input']['processed_end_frame'] == frames, 'complete bound input range')
        output = dict(schema='retained-asr-raw-transcript-v1',request_id=a['common']['request_id'],
            actual_text=actual_text,output_scope='complete-decoder-transcript',truncated=False,
            original_native_decoder_evidence=original_evidence)
        raw = canonical(output)
        persist(a['directory']/'raw-output.json',raw)
        receipt = dict(a['common'],schema='retained-asr-completed-v1',
            started_sha256=sha(a['started']),raw_output_sha256=sha(raw),status='success',
            decoder_completed=True,processed_start_frame=0,processed_end_frame=frames,
            loaded_model_assets=self.model['assets'],executed_source_files=self.source['files'],
            native_capture=capture)
        persist(a['directory']/'completed.json',canonical(receipt))
        self.active = None
        return receipt

    def complete(self, row, evidence):
        require(self.active is not None and self.active['started'] is not None and not self.failed,
                'real pre-generation start required')
        a = self.active
        require(row['completeness'] == 'complete' and row['quality_flags'] == [] and
                type(row['raw_text']) is str and row['raw_text'], 'successful complete transcript')
        require(evidence['input_binding'] == a['descriptor'] and
                evidence['prompt_context'] == '' and evidence['force_language'] is None and
                evidence['transcript_repair_applied'] is False and
                evidence['metadata']['eos_terminated'] is True,
                'actual input/context/termination capture')
        frames = a['common']['audio_frames']
        config = self.config
        capture = dict(schema_version=2,evidence_class='speech-asr-native-capture-v2',
            native_config_sha256=canonical_sha256(config),source_lock_sha256=config['source_lock_sha256'],
            asset_lock_sha256=config['asset_lock_sha256'],effective_config=config['effective_config'],
            effective_config_sha256=config['effective_config_sha256'],
            input=dict(sample_rate=16000,channels=1,audio_frames=frames,processed_start_frame=0,
                       processed_end_frame=frames,padding_frames=0,trimmed_frames=0,
                       preprocessing='full_input',pcm_format='float32_pcm16_scale_32768'),
            termination=dict(kind='autoregressive_eos',capture_stage='returned_generation_sequence',
                generated_token_ids=evidence['token_ids'],generated_tokens=evidence['metadata']['generated_tokens'],
                stop_reason='eos_token',truncated=False),
            transcript_postcorrection_applied=False,target_context_received=False,
            peer_transcript_received=False,acoustic_verbatim_completeness='unknown')
        return self.complete_native(row['raw_text'],capture,evidence)

    def fail(self):
        self.failed = True
        # Preserve any actual start/snapshot/raw bytes, never create completion.

NativeQwenSink = NativeEvidenceSink
