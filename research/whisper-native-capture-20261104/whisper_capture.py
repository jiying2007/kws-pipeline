"""Prospective native-v2 Whisper capture; import is standard-library only.

prepare() is a real model-load operation for an independently authorized,
network-disabled runner. It does not accept audio and performs no forward.
The returned live object must receive the frozen three-model policy before
infer(). No old output, config, receipt or checkpoint is upgraded in place.

This producer records observed execution. It cannot authenticate itself, add a
trusted source root, authorize source rights, or grant training admission.
"""
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from types import SimpleNamespace
import wave

HERE = Path(__file__).resolve().parent
MODEL_SHA = '9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794'
PINS = {
    'research/source-screen32-whisper-v1/adapter.py': 'd142605a62ff990633194b43992b698673780b065e5f3599d7e27eccc74d693b',
    'research/source-screen32-whisper-v1/common.py': '21cc72171d071cc9ac0113307af751315b69ce70fbf522a1b2586a7617900861',
    'research/source-screen32-whisper-v1/source-lock.json': '0635fb8759fb56439f59db6956612f78cc8903c239f81f3eb7f9b0c47eb99649',
    'research/source-screen32-whisper-v1/runtime-lock.json': '881443c6e66acb99223137bd8342788468f814d0a1c3e73ba33a27da3ad1fe6d',
    'research/source-screen32-v1/runtime_scope.py': 'c08bd46d34bea011a63d32867386b26637d5a02753769654e08e4c32c31cb927',
    'tools/asr_native_contracts.py': '752f192fb125fb31291a8246845c3015dc97348fc27822c389e14b52da3ef674',
}
TOKENIZER_MEMBER = 'bindings/upstream/whisper/assets/multilingual.tiktoken'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError('nonfinite JSON')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def relative(value):
    require(type(value) is str and value and '\\' not in value, 'canonical relative member')
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and
            not any(part in ('.', '..') for part in path.parts), 'unsafe member')
    return value


def regular(path, maximum):
    path = Path(path).absolute()
    require(path.resolve() == path and not path.is_symlink(), 'aliased file path')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
            0 < info.st_size <= maximum, 'bounded regular file required')
    return path, info


def identity(path, maximum=1024**3):
    path, before = regular(path, maximum)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, key) == getattr(after, key) for key in fields), 'file changed while hashing')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def read(path, maximum=32*1024**2):
    expected = identity(path, maximum)
    raw = Path(path).read_bytes()
    require(len(raw) == expected['bytes'] and sha(raw) == expected['sha256'], 'file changed while reading')
    return raw


def persist(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def import_exact(name, path):
    """Do not silently re-use another model worker's module named common."""
    path = Path(path).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        require(getattr(module, '__file__', None) and Path(module.__file__).resolve() == path,
                'conflicting loaded module: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, 'module unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _source_snapshot(retained_repo, runtime, sink_source, consensus_tools):
    repo, runtime = Path(retained_repo), Path(runtime)
    paths, bodies = {}, {}
    def add(member, path):
        relative(member)
        require(member not in bodies, 'duplicate source member')
        paths[member] = Path(path).absolute()
        bodies[member] = read(paths[member])
    for name, expected in sorted(PINS.items()):
        add('bindings/retained/' + name, repo/name)
        require(sha(bodies['bindings/retained/' + name]) == expected, 'retained source pin changed: ' + name)
    upstream = parse(bodies['bindings/retained/research/source-screen32-whisper-v1/source-lock.json'])
    require(upstream['source_revision'] == '86098128c0b4f24f0e2aa2994de830614b474227', 'upstream revision changed')
    for row in upstream['files']:
        name = relative(row['path'])
        add('bindings/upstream/' + name, runtime/'source'/name)
        body = bodies['bindings/upstream/' + name]
        require(sha(body) == row['sha256'] and len(body) == row['size_bytes'], 'upstream source pin changed')
    add('bindings/capture/whisper_capture.py', __file__)
    add('bindings/capture/worker.py', HERE/'worker.py')
    integration = repo/'research/native-asr-integration-20261104'
    if integration.exists():
        require(integration.is_dir() and integration.resolve() == integration.absolute(), 'unaliased integration source closure')
        for name in ('prepare_worker.py', 'zero_forward.py'):
            add('bindings/capture/' + name, integration/name)
    add('bindings/capture/native_sink.py', sink_source)
    add('bindings/capture/generation_capture.py', Path(sink_source).parent/'generation_capture.py')
    for name in ('speech_asr_consensus.py', 'speech_label_admission.py'):
        add('bindings/consensus/' + name, Path(consensus_tools)/name)
    for name in ('installed-inventory.json', 'asset-lock.json'):
        add('bindings/runtime/' + name, runtime/name)
    return paths, bodies


def verify_prepare_imports(retained_repo, runtime):
    """Verify the unchanged loader's source/install/assets before ML import.

    For an independently authorized zero-audio preparation only. This imports
    official Whisper definitions so a caller can install a forward fence before
    the actual FixedWhisper constructor. It does not call load_model or decode.
    The actual constructor repeats its original verification without changes.
    """
    repo, runtime = Path(retained_repo).absolute(), Path(runtime).absolute()
    for name, digest in PINS.items():
        require(sha(read(repo/name)) == digest, 'retained source pin changed: ' + name)
    scope = import_exact('_whisper_native_scope', repo/'research/source-screen32-v1/runtime_scope.py')
    scope.verify_runtime_scope()
    common = import_exact('common', repo/'research/source-screen32-whisper-v1/common.py')
    source = runtime/'source'
    source_lock = common.read_json(repo/'research/source-screen32-whisper-v1/source-lock.json')
    expected = {row['path']: {'sha256': row['sha256'], 'bytes': row['size_bytes']}
                for row in source_lock['files']}
    common.verify_inventory(source, expected)
    installed = runtime/'venv/lib/python3.12/site-packages'
    common.verify_inventory(installed, common.read_json(runtime/'installed-inventory.json')['files'])
    assets = common.read_json(runtime/'asset-lock.json')
    for name, item in assets['files'].items():
        path = runtime/'model'/name if name == 'small.pt' else source/name
        require(common.identity(path) == item, 'loaded asset byte mismatch before import')
    require(assets['files']['small.pt']['sha256'] == MODEL_SHA, 'unchanged small checkpoint only')
    if 'whisper' in sys.modules:
        module = sys.modules['whisper']
        require(getattr(module, '__file__', None) and
                Path(module.__file__).resolve() == source/'whisper/__init__.py', 'conflicting Whisper import')
    sys.path.insert(0, str(source))
    # Verified imports may initialize package definitions, not a model.
    import whisper
    require(Path(whisper.__file__).resolve() == source/'whisper/__init__.py', 'trusted Whisper source origin')
    return whisper


def _modules(retained_repo, sink_source, consensus_tools):
    repo = Path(retained_repo)
    scope = import_exact('_whisper_native_scope', repo/'research/source-screen32-v1/runtime_scope.py')
    # This checks actual kernel isolation before importing any model dependency.
    scope.verify_runtime_scope()
    common = import_exact('common', repo/'research/source-screen32-whisper-v1/common.py')
    contracts = import_exact('asr_native_contracts', repo/'tools/asr_native_contracts.py')
    adapter = import_exact('_retained_whisper_adapter', repo/'research/source-screen32-whisper-v1/adapter.py')
    import_exact('speech_label_admission', Path(consensus_tools)/'speech_label_admission.py')
    import_exact('speech_asr_consensus', Path(consensus_tools)/'speech_asr_consensus.py')
    import_exact('generation_capture', Path(sink_source).parent/'generation_capture.py')
    sink = import_exact('_shared_native_sink', sink_source)
    return scope, common, contracts, adapter, sink


def _config(loaded, adapter, contracts, source_raw, model_raw, tokenizer_raw):
    # Native task construction resolves real tokenizer/suppression without run().
    effective = adapter.observed_effective(loaded.make_task(), loaded.torch,
                                          loaded.model, sha(tokenizer_raw))
    require(canonical(effective) == canonical(loaded.effective), 'loaded effective config changed')
    config = dict(schema_version=2, evidence_class='speech-asr-native-config-v2',
        adapter_id='openai-whisper-decode-short', adapter_version=1, model_family='whisper',
        source_lock_sha256=sha(source_raw), asset_lock_sha256=sha(model_raw),
        conditioning_policy=contracts.CONDITIONING_POLICY,
        execution=effective['execution'],
        input_policy=dict(sample_rate=16000, channels=1, max_audio_frames=480000,
            pcm_format='float32_pcm16_scale_32768', preprocessing='zero_pad_only_480000'),
        native_request=effective['native_request'], effective_config=effective,
        effective_config_sha256=sha(canonical(effective)))
    contracts.validate_native_config(config)
    return config


def prepare(retained_repo, runtime, output, executed_head_sha, sink_source,
            consensus_tools, recognizer_id='whisper-small', maximum=64):
    """Load once, observe native settings, freeze identity; no audio or forward.

    Caller supplies the executed repository commit for later independent
    provenance verification. It is not made trustworthy by writing it here.
    A fresh, empty output is mandatory. This does not download/install anything.
    """
    require(type(executed_head_sha) is str and re.fullmatch('[a-f0-9]{40}', executed_head_sha), 'explicit executed source commit required')
    require(type(recognizer_id) is str and re.fullmatch('[A-Za-z0-9_-]{1,80}', recognizer_id), 'recognizer ID')
    require(type(maximum) is int and 1 <= maximum <= 64, 'bounded one-attempt count')
    root = Path(output).absolute()
    require(root.resolve() == root, 'aliased output path')
    paths, bodies = _source_snapshot(retained_repo, runtime, sink_source, consensus_tools)
    scope, common, contracts, adapter, sink = _modules(retained_repo, sink_source, consensus_tools)
    checkpoint = Path(runtime).absolute()/'model/small.pt'
    asset = dict(path='checkpoint/small.pt', **identity(checkpoint))
    require(asset['sha256'] == MODEL_SHA, 'unchanged Whisper-small checkpoint required')
    model = dict(schema='retained-asr-model-lock-v1', provider_id='openai',
        model_id='openai/whisper-small', model_revision=MODEL_SHA, model_family='whisper',
        assets=[asset], model_sha256=sha(canonical([asset])))
    files = [dict(path=name, bytes=len(body), sha256=sha(body)) for name, body in sorted(bodies.items())]
    source = dict(schema='retained-asr-source-lock-v1', executed_head_sha=executed_head_sha,
        files=files, source_inventory_scope='Pinned upstream tree and executed capture/validator sources; runtime package bytes are checked by the unchanged loader against retained installed inventory.')
    source_raw, model_raw = canonical(source), canonical(model)
    root.mkdir(exist_ok=False)
    diagnostics = root/'original-loader-diagnostics'
    diagnostics.mkdir()
    # The retained loader keeps its own pins, inventory checks, CPU load and
    # diagnostic schemas unchanged. Its constructor has no audio argument.
    loaded = adapter.FixedWhisper(runtime, diagnostics)
    require(parse(read(diagnostics/'load-receipt.json'))['audio_calls'] == 0, 'prepare must precede all audio')
    for member, path in paths.items():
        require(read(path) == bodies[member], 'source changed during load')
    require(dict(path='checkpoint/small.pt', **identity(checkpoint)) == asset, 'checkpoint changed during load')
    config = _config(loaded, adapter, contracts, source_raw, model_raw, bodies[TOKENIZER_MEMBER])
    # This is a NEW prospective in-memory configuration before the FIRST decode.
    # The retained decoder uses it directly when producing its native_capture.
    loaded.config = parse(canonical(config))
    ident = {key: model[key] for key in ('provider_id', 'model_id', 'model_revision', 'model_family', 'model_sha256')}
    ident.update(recognizer_id=recognizer_id, native_decode_config=config,
                 native_decode_config_sha256=sha(canonical(config)))
    for name, value in (('model-lock.json', model), ('source-lock.json', source),
                        ('native-config.json', config), ('identity.json', ident)):
        persist(root/name, canonical(value))
    for member, body in bodies.items():
        persist(root/member, body)
    dependencies = {'tokenizer_sha256': TOKENIZER_MEMBER}
    persist(root/'native-dependencies.json', canonical(dependencies))
    persist(root/'prepare-receipt.json', canonical(dict(schema='prospective-whisper-prepare-v1',
        observation_stage='loaded_native_task_before_first_audio', audio_calls=0,
        identity_sha256=sha(canonical(ident)), native_config_sha256=sha(canonical(config)),
        original_loader_receipt_sha256=sha(read(diagnostics/'load-receipt.json')),
        training_admitted=False, authenticity='requires-independent-source-resolution')))
    return _PreparedWhisper(root, loaded, scope, adapter, contracts, sink,
        paths, bodies, checkpoint, model_raw, source_raw, ident, maximum)


class _PreparedWhisper:
    def __init__(self, root, loaded, scope, adapter, contracts, sink_module,
                 paths, bodies, checkpoint, model_raw, source_raw, ident, maximum):
        self.root, self.loaded, self.scope = root, loaded, scope
        self.adapter, self.contracts, self.sink_module = adapter, contracts, sink_module
        self.paths, self.bodies, self.checkpoint = paths, bodies, checkpoint
        self.model_raw, self.source_raw = model_raw, source_raw
        self.identity = parse(canonical(ident))
        self.config = self.identity['native_decode_config']
        self.maximum, self.seen, self.request_ids, self.failed = maximum, set(), set(), False
        self.sink = None

    def _verify(self):
        self.scope.verify_runtime_scope()
        for member, path in self.paths.items():
            require(read(path) == self.bodies[member], 'executed source/dependency changed')
        assets = [dict(path='checkpoint/small.pt', **identity(self.checkpoint))]
        require(canonical(assets) == canonical(parse(self.model_raw)['assets']), 'loaded checkpoint bytes changed')
        actual = _config(self.loaded, self.adapter, self.contracts, self.source_raw,
                         self.model_raw, self.bodies[TOKENIZER_MEMBER])
        require(canonical(actual) == canonical(self.config) and
                canonical(self.loaded.config) == canonical(self.config), 'actual config differs from prepared identity')
        return actual, assets

    def bind_policy(self, policy_raw, expected_policy_sha256):
        """Freeze an externally coordinated three-model policy, once, no forward."""
        require(not self.failed and self.sink is None and not self.seen, 'one policy freeze before audio')
        try:
            require(type(policy_raw) is bytes, 'original policy bytes required')
            policy = parse(policy_raw)
            require(sha(canonical(policy)) == expected_policy_sha256, 'frozen policy canonical hash')
            actual, assets = self._verify()
            matches = [row for row in policy['recognizers'] if row['recognizer_id'] == self.identity['recognizer_id']]
            require(len(matches) == 1 and canonical(matches[0]) == canonical(self.identity), 'prepared identity differs from frozen policy')
            self.sink = self.sink_module.NativeEvidenceSink(self.root/'capture', policy_raw,
                self.model_raw, self.source_raw, self.identity['recognizer_id'], self.bodies, assets)
            for member, body in self.bodies.items():
                persist(self.sink.root/member, body)
            persist(self.sink.root/'native-dependencies.json', canonical({'tokenizer_sha256': TOKENIZER_MEMBER}))
        except BaseException:
            self.failed = True
            if self.sink is not None:
                self.sink.fail()
            raise

    def infer(self, audio_path, opaque_id, expected_wav_sha256, source_id, request_id):
        """One complete PCM16 WAV, one attempt, no target or peer text input."""
        require(not self.failed and self.sink is not None, 'policy must be frozen before recognition')
        try:
            require(type(opaque_id) is str and re.fullmatch('clip-[0-9]{6}', opaque_id), 'safe opaque ID')
            require(opaque_id not in self.seen and len(self.seen) < self.maximum, 'attempt already consumed or budget exhausted')
            self.seen.add(opaque_id)
            require(type(request_id) is str and request_id not in self.request_ids, 'unique request ID')
            for value in (source_id, request_id):
                require(type(value) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', value), 'opaque source/request binding')
            self.request_ids.add(request_id)
            raw_wav = read(audio_path, 960044)
            require(type(expected_wav_sha256) is str and sha(raw_wav) == expected_wav_sha256, 'original WAV hash')
            with wave.open(io.BytesIO(raw_wav), 'rb') as wav:
                frames = wav.getnframes()
                require((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) ==
                        (16000, 1, 2, 'NONE') and 0 < frames <= 480000, 'bounded full PCM16 WAV')
                pcm = wav.readframes(frames + 1)
            require(len(pcm) == frames*2, 'complete PCM input')
            actual, _ = self._verify()
            descriptor = dict(wav=dict(sha256=sha(raw_wav)),
                raw_pcm16=dict(sha256=sha(pcm), frame_count=frames, sample_rate_hz=16000, channels=1),
                operations=dict(mix=False, resample=False, trim=False))
            self.sink.begin(SimpleNamespace(opaque_id=opaque_id, descriptor=descriptor), source_id, request_id)
            self.sink.start_native(actual)
            native = self.sink.root/opaque_id/'native'
            native.mkdir()
            result = self.loaded.decode(pcm, frames, native)
            require(parse(read(native/'raw-output.json')) == result, 'actual native output differs from persisted bytes')
            self._verify()
            require(result['decoder_completion_validated'] is True and result['full_input_processed'] is True,
                    'natural pre-finalize EOT and full input required')
            capture = result['native_capture']
            self.contracts.validate_native_capture(self.config, capture)
            inp = parse(read(native/'input-capture.json'))
            require(inp['pcm16_sha256'] == sha(pcm) and inp['native_input'] == capture['input'], 'actual frontend input binding')
            steps = [parse(read(native/('update-%03d.json' % index))) for index in range(1, result['trace_files']+1)]
            require(self.adapter.termination(steps, self.config['effective_config']) == capture['termination'], 'persisted pre-finalize trace binding')
            pre = parse(read(native/'pre-finalize.json'))
            require(pre == dict(token_ids=steps[-1]['token_ids'], last_update_completed=True, trace_count=len(steps)), 'actual pre-finalize boundary')
            expected = {'raw-output.json', 'input-capture.json', 'pre-finalize.json'} | {
                'update-%03d.json' % index for index in range(1, len(steps)+1)}
            require({path.name for path in native.iterdir()} == expected, 'exact native output membership')
            artifacts = [dict(path=path.relative_to(self.sink.root).as_posix(), **identity(path, 8*1024**2))
                         for path in sorted(native.iterdir())]
            evidence = dict(decoder_result=result, input_capture=inp, native_update_trace=steps,
                            pre_finalize=pre, private_artifacts=artifacts)
            return self.sink.complete_native(result['raw_text'], capture, evidence)
        except BaseException:
            self.failed = True
            self.sink.fail()
            raise
