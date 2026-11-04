"""Prospective wrapper around the unchanged, pinned primary worker.

No model is imported by this module. The old worker owns loading, PCM guards,
claims, deadlines and terminal receipts. New hooks add actual native capture.
"""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from generation_capture import ResolvedConfigRecorder, canonical, require, EXPECTED, snapshot

HERE = Path(__file__).resolve().parent

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

class BoundCapture:
    def __init__(self, loaded, root, maximum, native_sink=None, request_map=None):
        self.loaded = loaded
        self.thinker = loaded.wrapper.model.thinker
        self.original_generate = self.thinker.generate
        self.signature = inspect.signature(self.original_generate)
        self.recorder = ResolvedConfigRecorder(self.thinker, root, maximum)
        self.active = None
        self.seen = set()
        self.dead = False
        self.generate_calls = 0
        self.resolved_object = None
        self.native_sink = native_sink
        self.request_map = request_map
        self.native_execution = None
        original_resolver = self.recorder.original
        self.original_resolver = original_resolver
        def remember(*args, **kwargs):
            value = original_resolver(*args, **kwargs)
            self.resolved_object = value[0]
            if self.native_sink is not None:
                self.native_sink.before_continue(snapshot(value[0]),self.native_execution,
                    {'custom_logits_processors':[], 'custom_stopping_criteria':[]})
            return value
        self.recorder.original = remember
        self.recorder.install()
        self.thinker.generate = self.generate

    def restore(self):
        self.thinker.generate = self.original_generate
        self.recorder.restore()
        # recorder.original was wrapped only to retain the observed config
        # object. Restore the actual pre-install bound native resolver.
        self.thinker._prepare_generation_config = self.original_resolver

    def generate(self, *args, **kwargs):
        require(self.active is not None and not self.dead and self.generate_calls == 0,
                'exactly one native generate per claimed clip')
        self.generate_calls += 1
        actual = self.signature.bind(*args, **kwargs)
        actual.apply_defaults()
        for key in ('logits_processor', 'stopping_criteria', 'prefix_allowed_tokens_fn',
                    'assistant_model', 'custom_generate', 'negative_prompt_ids',
                    'negative_prompt_attention_mask'):
            require(key in actual.arguments and actual.arguments[key] is None,
                    'custom native generation control forbidden: ' + key)
        if self.native_sink is not None:
            values = actual.arguments.get('kwargs',{})
            ids = values.get('input_ids', actual.arguments.get('inputs'))
            require(ids is not None and len(ids.shape) == 2 and ids.shape[0] == 1,
                    'actual single native input batch')
            self.native_execution = measured_execution(self.loaded)
        return self.original_generate(*args, **kwargs)

    def infer(self, original, runtime, bound, decoder_sha):
        require(runtime is self.loaded and not self.dead and self.active is None,
                'fresh verified runtime required')
        require(bound.opaque_id not in self.seen, 'clip already consumed')
        self.seen.add(bound.opaque_id)
        self.active = bound.opaque_id
        self.generate_calls = 0
        before = self.recorder.completed
        try:
            if self.native_sink is not None:
                require(type(self.request_map) is dict and bound.opaque_id in self.request_map,
                        'prefrozen audio-only source/request map required')
                identity = self.request_map[bound.opaque_id]
                require(type(identity) is dict and set(identity) == {'source_id','request_id'},
                        'exact audio-only source/request identity')
                self.native_sink.begin(bound,identity['source_id'],identity['request_id'])
            claim = dict(schema='prospective-native-capture-claim-v1',
                opaque_id=bound.opaque_id, wav_sha256=bound.descriptor['wav']['sha256'],
                pcm_sha256=bound.descriptor['raw_pcm16']['sha256'],
                input_binding_sha256=bound.descriptor_sha256,
                decoder_manifest_sha256=decoder_sha, expected_resolution_ordinal=before+1,
                training_admitted=False)
            with (self.recorder.output/('claim-%04d.json' % (before+1))).open('xb') as stream:
                stream.write(canonical(claim)); stream.flush(); os.fsync(stream.fileno())
            result, evidence = original(runtime, bound, decoder_sha)
            require(self.generate_calls == 1 and self.recorder.completed == before + 1
                    and self.recorder.calls == before + 1 and not self.recorder.poisoned,
                    'successful inference requires exactly one actual capture')
            # Later native code adds special-token tensors and resolves lengths.
            # Only compare the required public scalar/list controls here.
            for key, expected in EXPECTED.items():
                require(canonical(getattr(self.resolved_object, key)) == canonical(expected),
                        'required control changed after resolver: ' + key)
            number = self.recorder.calls
            path = self.recorder.output / ('resolved-generation-config-%04d.json' % number)
            binding = dict(schema='bound-native-generation-observation-v1',
                opaque_id=bound.opaque_id, wav_sha256=bound.descriptor['wav']['sha256'],
                pcm_sha256=bound.descriptor['raw_pcm16']['sha256'],
                input_binding_sha256=bound.descriptor_sha256, decoder_manifest_sha256=decoder_sha,
                resolved_generation_config_sha256=sha(path.read_bytes()),
                resolution_ordinal=number, snapshot_file=path.name,
                actual_native_generate_calls=1, custom_logits_processors=[],
                custom_stopping_criteria=[], required_controls_unchanged_after_generation=True,
                snapshot_stage='native_resolver_return', final_full_config_claimed=False,
                training_admitted=False)
            # This object is inside the legacy decoder evidence. Its hash is in
            # the legacy completion receipt, preserving the existing chain.
            if self.native_sink is not None:
                strict = self.native_sink.complete(result,evidence)
                binding['strict_completion_sha256'] = sha(canonical(strict))
            return result, dict(evidence, prospective_generation_capture=binding)
        except BaseException:
            self.dead = True
            self.recorder.poisoned = True
            if self.native_sink is not None:
                self.native_sink.fail()
            raise
        finally:
            self.active = None

def measured_execution(loaded):
    model = loaded.wrapper.model.thinker
    require(model.training is False, 'actual thinker eval mode')
    seen = False
    for parameter in model.parameters():
        seen = True
        require(str(parameter.device) == 'cpu' and str(parameter.dtype) == 'torch.float32',
                'actual thinker CPU FP32')
    require(seen, 'observed thinker parameters')
    return {'device':'cpu','dtype':'float32','batch_size':1}

def prepare_native_qwen(loaded, output, model_raw, source_raw):
    """Zero forward: resolve actual config before the three-model policy freeze.

    Same explicit kwargs as the locked Qwen top-level generate dispatch. The
    actual inference hook must later match this snapshot byte-for-byte.
    """
    from asr_native_contracts import validate_native_config, canonical_sha256
    verified_native_source(loaded)
    execution = measured_execution(loaded)
    result = loaded.wrapper.model.thinker._prepare_generation_config(None,None,
        max_new_tokens=256,do_sample=False,eos_token_id=[151645,151643],return_dict_in_generate=True)
    require(type(result) is tuple and len(result) == 2, 'actual native preparation result')
    raw = snapshot(result[0]); actual = json.loads(raw)
    effective = {key:actual[key] for key in ('num_beams','num_return_sequences','forced_bos_token_id',
        'forced_eos_token_id','suppress_tokens','begin_suppress_tokens','bad_words_ids','force_words_ids',
        'sequence_bias','stop_strings')}
    request = {'context':'','force_language':None,'max_new_tokens':actual['max_new_tokens'],
               'do_sample':actual['do_sample'],'eos_token_id':actual['eos_token_id']}
    mode = result[0].get_generation_mode(None)
    require(getattr(mode,'value',None) == 'greedy_search', 'actual native generation mode')
    effective.update(execution=execution,native_request=request,generation_mode=mode.value,
        custom_logits_processors=[],custom_stopping_criteria=[],resolved_generation_config_sha256=sha(raw))
    config = dict(schema_version=2,evidence_class='speech-asr-native-config-v2',
        adapter_id='qwen3-asr-transformers-greedy',adapter_version=1,model_family='qwen3-asr',
        source_lock_sha256=sha(source_raw),asset_lock_sha256=sha(model_raw),
        conditioning_policy='blind-no-target-or-peer-context-v1',execution=execution,
        input_policy=dict(sample_rate=16000,channels=1,max_audio_frames=192000,
                          pcm_format='float32_pcm16_scale_32768',preprocessing='full_input'),
        native_request=request,effective_config=effective,effective_config_sha256=canonical_sha256(effective))
    validate_native_config(config)
    output=Path(output);output.mkdir(exist_ok=False)
    for name,value in [('resolved-generation-config.json',raw),('native-config.json',canonical(config))]:
        with (output/name).open('xb') as stream:
            stream.write(value);stream.flush();os.fsync(stream.fileno())
    return config

def prepare_strict_identity(loaded, output, executed_head, source_bodies, recognizer_id='qwen06'):
    """Wrap actual verified loader inventory; source bytes are supplied/measured.

    Source inventory must include this adapter and pinned native dependency code.
    The independently reviewed host owns completeness of that closed inventory.
    """
    import re
    from asr_native_contracts import canonical_sha256
    require(re.fullmatch('[0-9a-f]{40}',executed_head) is not None, 'executed head')
    require(type(source_bodies) is dict and source_bodies, 'actual source closure bytes')
    files=[]
    for path,body in sorted(source_bodies.items()):
        require(type(body) is bytes and body and not Path(path).is_absolute()
                and '..' not in Path(path).parts, 'nonempty relative source body')
        files.append(dict(path=path,bytes=len(body),sha256=sha(body)))
    assets=loaded.receipt['assets']
    require(assets['model_id']=='Qwen/Qwen3-ASR-0.6B', 'verified Qwen asset identity')
    inventory=[dict(path=row['filename'],bytes=row['bytes'],sha256=row['sha256'])
               for row in assets['verified_files']]
    identity=dict(recognizer_id=recognizer_id,provider_id='Qwen',model_id=assets['model_id'],
        model_revision=assets['revision'],model_family='qwen3-asr',model_sha256=canonical_sha256(inventory))
    model=dict(schema='retained-asr-model-lock-v1',assets=inventory,
               **{key:value for key,value in identity.items() if key!='recognizer_id'})
    source=dict(schema='retained-asr-source-lock-v1',executed_head_sha=executed_head,files=files)
    model_raw=canonical(model);source_raw=canonical(source)
    config=prepare_native_qwen(loaded,output,model_raw,source_raw)
    identity.update(native_decode_config=config,native_decode_config_sha256=canonical_sha256(config))
    for name,value in [('model-lock.json',model_raw),('source-lock.json',source_raw),('identity.json',canonical(identity))]:
        with (Path(output)/name).open('xb') as stream:
            stream.write(value);stream.flush();os.fsync(stream.fileno())
    return dict(identity=identity,model_raw=model_raw,source_raw=source_raw,verified_assets=inventory,
                source_bodies=source_bodies)

def verified_native_source(loaded):
    review = json.loads((HERE / 'transformers-review.json').read_text())
    generate = inspect.unwrap(loaded.wrapper.model.thinker.generate)
    source = Path(inspect.getsourcefile(generate))
    require(sha(source.read_bytes()) == review['sha256'], 'pinned native GenerationMixin source')

def run(legacy, model, input_root, output_root, runtime_root, native_sink_factory=None, request_map=None):
    require(model != 'qwen06' or callable(native_sink_factory),
            'strict prefrozen native sink factory required before Qwen launch')
    original_loader = legacy.load_qwen_after_approval
    original_import = legacy.import_helpers
    state = {}
    def loader(*args, **kwargs):
        loaded = original_loader(*args, **kwargs)
        verified_native_source(loaded)
        native_sink=native_sink_factory(loaded,Path(output_root)/model/'strict-native')
        state['capture'] = BoundCapture(loaded, Path(output_root)/model/'native-capture', len(loaded.decoder_inputs),
                                       native_sink=native_sink,request_map=request_map)
        return loaded
    def helpers():
        original, evidence = original_import()
        proxy = SimpleNamespace(**vars(original))
        def infer(*args):
            return state['capture'].infer(original.infer_qwen_once_after_approval, *args)
        proxy.infer_qwen_once_after_approval = infer
        return proxy, evidence
    require(model in ('qwen06', 'sensevoice'), 'primary model only')
    legacy.load_qwen_after_approval = loader
    legacy.import_helpers = helpers
    try:
        return legacy.run(model, input_root, output_root, runtime_root)
    finally:
        legacy.load_qwen_after_approval = original_loader
        legacy.import_helpers = original_import
        if 'capture' in state:
            cap = state['capture']
            cap.restore()

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=('qwen06','sensevoice'))
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--runtime', required=True)
    args = p.parse_args()
    retained = HERE.parent/'source-screen32-v1'
    sys.path.insert(0, str(retained))
    spec = importlib.util.spec_from_file_location('retained_primary_worker', retained/'asr_worker.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    run(legacy, args.model, args.input, args.output, args.runtime)

if __name__ == '__main__':
    main()
