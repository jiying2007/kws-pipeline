#!/usr/bin/env python3
"""Rebuilt one-shot KW1 native TTS worker; imports no inference packages at import.

This is newly reconstructed source, not a claim of recovering the lost original.
Durable exclusive receipts precede every load/call. No resume/retry path exists.
The standalone mount contains this file and only pinned retained loader sources.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import resource
import signal
import stat
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1]
MAX_BYTES = 8 * 1024**2
SHA = re.compile(r'[0-9a-f]{64}\Z')
ENGINES = ('qwen06', 'qwen17')
SEEDS = (2026101401, 2026101402)
INSTRUCTION = '请用标准普通话清晰自然地朗读提供的文字，最后一个字读第一声；不要改字、增字或漏字。'
MODELS = {
    'qwen06': ('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', '85e237c12c027371202489a0ec509ded67b5e4b5', '0b6'),
    'qwen17': ('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', '0c0e3051f131929182e2c023b9537f8b1c68adfe', '1b7'),
}
PRESET_IDS = {'Uncle_Fu': 3010, 'Eric': 2875, 'Dylan': 2878}
DIALECTS = {'uncle_fu': False, 'eric': 'sichuan_dialect', 'dylan': 'beijing_dialect'}
GENERATION = dict(do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
    repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
    subtalker_top_p=1.0, subtalker_temperature=0.9, max_new_tokens=120,
    non_streaming_mode=True)
LOAD_SECONDS, CALL_SECONDS = 1200, 300
ERRORS = {'exception', 'timeout', 'interrupted', 'technical_quality', 'duplicate_native'}
NATIVE_KEYS = {'sample_rate_hz', 'channels', 'frames', 'subtype', 'source_values_sha256'}
METRIC_KEYS = {'duration_seconds', 'peak', 'rms', 'clipped_fraction', 'quality_flags', 'technically_valid'}
TERMINATION_KEYS = {'iterations', 'last_token_ids', 'eos', 'ended_with_eos', 'hit_token_cap_without_eos'}
OUTCOME_EXTRA = {'status', 'attempts', 'error_type', 'file', 'raw_wav_sha256', 'native', 'metrics', 'termination'}
# Populated from recovered retained main source bytes, never downloaded at runtime.
RETAINED_PINS = {'research/qwen6_tts/setup_locked.py': 'aac8d8e66504444f2965aa8d95dcfef9e8c6fa63d74f2eade15ca1290bc6b4b3', 'research/qwen6_tts/runtime-lock.json': '6a707c2fa2c2ae8f842176a7f9c8301186d51fddef167c413160c7f2816613de', 'research/qwen6_tts/model-locks.json': 'c206556a633b7d04c334494fc5a82934a19d572c125982c689f18f8b2d472110', 'research/qwen6_tts/setup_diagnostics.py': '7c583ba0ed6f0e90f72ab3b5c107e3666a32d25359ce13ffc284d545f7d1dd5f', 'research/qwen6_tts/wheel_identity.py': '623ee07a0d8f85ba1d7218a56fcd0e270df8c6e2c3e353dd52b3184ca915d8d5', 'research/qwen6_tts/import_preflight.py': '30cfe60347ccc6074d98a21406fccee3e3be5c54822aa8d874aae3655693ae1c', 'research/qwen6_tts/runtime_gate.py': 'f4333b694bb1bc261e4bdeea6d586dc9f76b1afce6ce0af9dfcdb74d57d12815', 'research/qwen6_tts/generate_six.py': '25530b89fa3f58f44ceb8d3c2a8f26f3b37651c7173f62414dc96142919ea03e', 'research/source-screen32-v1/setup_adapter.py': 'b7f6c3493ad0da909c7e6856d27f3247f5e1716ce6c981cb396a03687aa7e547', 'research/source-screen32-v1/runtime_scope.py': 'c08bd46d34bea011a63d32867386b26637d5a02753769654e08e4c32c31cb927', 'research/tts-quality-pilot-v1/setup_adapter.py': '889b2b2ee33e940dff00a51a090f1ec5fad83c8343f6aa21ac4392602e22faf6', 'research/tts-quality-pilot-v1/runtime_scope.py': 'c08bd46d34bea011a63d32867386b26637d5a02753769654e08e4c32c31cb927'}
MODEL_LOCK_PINS = {'qwen06': '9da27e3dea80033174141613458b03901460d0523ca2be7323e12071bad2ac90', 'qwen17': '24267f7e4a63f3f4f2a13bf6c87459ae21be61c56bc5b55504381569fdd9f3f3'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    require(type(raw) is bytes and len(raw) <= MAX_BYTES, 'bounded JSON required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def unaliased(path):
    path = Path(path).absolute()
    require(path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)), 'unaliased path required')
    return path


def read_regular(path, maximum=MAX_BYTES):
    path = unaliased(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum, 'bounded unaliased regular file required')
        with os.fdopen(os.dup(descriptor), 'rb') as stream:
            raw = stream.read(maximum + 1)
        after, entry = os.fstat(descriptor), path.lstat()
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(stamp(before) == stamp(after) == stamp(entry) and len(raw) == before.st_size, 'file changed during read')
        return raw
    finally:
        os.close(descriptor)


def fsync_directory(path):
    descriptor = os.open(unaliased(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_once(path, raw):
    path = unaliased(path)
    require(type(raw) is bytes and len(raw) <= MAX_BYTES, 'bounded receipt required')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        fsync_directory(path.parent)  # Empty exclusive claim survives interruption.
        with os.fdopen(os.dup(descriptor), 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)
    return sha(raw)


def write_once(path, value):
    return write_bytes_once(path, canonical(value))


def frozen_plan():
    slots = []
    for engine, speakers, instructions in (
        ('qwen06', ('Uncle_Fu', 'Eric'), ('',)),
        ('qwen17', ('Uncle_Fu', 'Eric', 'Dylan'), ('', INSTRUCTION))):
        for speaker in speakers:
            for instruction in instructions:
                for seed in SEEDS:
                    slots.append(dict(slot_id=f'kw1s-{len(slots):02d}', engine=engine,
                        speaker=speaker, group_id=speaker, target_text='你好小窝', language='Chinese',
                        seed=seed, instruction=instruction, generation_parameters=dict(GENERATION)))
    return dict(schema='kw1-preset-supplement-plan-v1', campaign_id='kw1-preset-supplement-20261014-v1',
        status='PREPARED_NOT_EXECUTED', execution_enabled=False, slots=slots,
        models={key: dict(model_id=value[0], revision=value[1], tts_model_size=value[2]) for key, value in MODELS.items()},
        preset_ids=dict(PRESET_IDS), runtime=dict(device='cpu', dtype='float32', attention='eager',
            evaluation=True, inference_mode=True, threads=4, interop_threads=1, qwen_tts_version='0.1.1'),
        native_audio=dict(sample_rate_hz=24000, channels=1, subtype='FLOAT', max_seconds=12),
        budget=dict(tts_calls=16, asr_calls=48, max_attempts_per_slot=1, retries=0, money_usd=0,
            load_seconds=1200, setup_seconds=1200, call_seconds=300, global_seconds=18000),
        admission=dict(human_gold=False, training_admitted=False, calibration_gold=False, selection_gold=False))


def validate_plan(plan):
    require(type(plan) is dict and canonical(plan) == canonical(frozen_plan()), 'frozen sixteen-slot plan changed')
    return plan['slots']


def initial_outcome(cell):
    return dict(cell, status='NOT_RUN', attempts=0, error_type=None, file=None,
        raw_wav_sha256=None, native=None, metrics=None, termination=None)


def validate_termination(value):
    require(type(value) is dict and set(value) == TERMINATION_KEYS, 'termination schema')
    require(type(value['iterations']) is int and 1 <= value['iterations'] <= 120 and
        type(value['eos']) is int and value['eos'] == 2150 and
        type(value['last_token_ids']) is list and len(value['last_token_ids']) == 1 and
        all(type(x) is int and 0 <= x <= 2**31-1 for x in value['last_token_ids']), 'termination values')
    ended = value['last_token_ids'] == [value['eos']]
    require(type(value['ended_with_eos']) is bool and value['ended_with_eos'] == ended and
        type(value['hit_token_cap_without_eos']) is bool and
        value['hit_token_cap_without_eos'] == (value['iterations'] >= 120 and not ended), 'termination flags')
    return ended and value['iterations'] < 120


def signal_metrics(samples, termination):
    require(type(samples) in (list, tuple) and 0 < len(samples) <= 288000 and all(math.isfinite(x) for x in samples), 'native finite geometry')
    peak = max(abs(x) for x in samples)
    rms = math.sqrt(math.fsum(x*x for x in samples) / len(samples))
    flags = []
    if peak >= 1: flags.append('source_peak_ge_one')
    if rms < 1e-4: flags.append('near_silent')
    if termination is None or not validate_termination(termination): flags.append('termination_unverified')
    return dict(duration_seconds=len(samples)/24000, peak=peak, rms=rms,
        clipped_fraction=sum(abs(x) >= 1 for x in samples)/len(samples),
        quality_flags=flags, technically_valid=not flags)


def float_wav(values_raw):
    require(type(values_raw) is bytes and 0 < len(values_raw) <= 288000*4 and len(values_raw)%4 == 0, 'FLOAT sample bytes')
    fmt = b'fmt ' + struct.pack('<IHHIIHH', 16, 3, 1, 24000, 96000, 4, 32)
    fact = b'fact' + struct.pack('<II', 4, len(values_raw)//4)
    data = b'data' + struct.pack('<I', len(values_raw)) + values_raw
    return b'RIFF' + struct.pack('<I', 4+len(fmt)+len(fact)+len(data)) + b'WAVE' + fmt + fact + data


def native_samples(raw):
    require(type(raw) is bytes and 44 <= len(raw) <= MAX_BYTES and raw[:4] == b'RIFF' and raw[8:12] == b'WAVE' and
        struct.unpack_from('<I', raw, 4)[0] == len(raw)-8, 'native RIFF container')
    chunks = {}; offset = 12
    while offset < len(raw):
        require(offset+8 <= len(raw), 'native truncated chunk')
        name, length = raw[offset:offset+4], struct.unpack_from('<I', raw, offset+4)[0]
        offset += 8
        require(name not in chunks and name in (b'fmt ', b'fact', b'data') and offset+length <= len(raw), 'native chunk inventory')
        chunks[name] = raw[offset:offset+length]; offset += length + (length%2)
    require(offset == len(raw) and set(chunks) == {b'fmt ', b'fact', b'data'} and
        chunks[b'fmt '] == struct.pack('<HHIIHH', 3, 1, 24000, 96000, 4, 32), 'native exact FLOAT mono format')
    values = chunks[b'data']
    require(0 < len(values) <= 288000*4 and len(values)%4 == 0 and chunks[b'fact'] == struct.pack('<I', len(values)//4), 'native frames')
    samples = [x[0] for x in struct.iter_unpack('<f', values)]
    require(all(math.isfinite(x) for x in samples), 'nonfinite native audio')
    return values, samples


def verify_native(raw, native):
    values, samples = native_samples(raw)
    expected = dict(sample_rate_hz=24000, channels=1, frames=len(samples), subtype='FLOAT', source_values_sha256=sha(values))
    require(type(native) is dict and canonical(native) == canonical(expected), 'native metadata differs from actual FLOAT values')
    return samples


def validate_metrics(value):
    require(type(value) is dict and set(value) == METRIC_KEYS, 'native metrics schema')
    require(all(type(value[k]) in (int, float) and math.isfinite(value[k]) for k in
        ('duration_seconds', 'peak', 'rms', 'clipped_fraction')), 'finite native metrics')
    require(0 < value['duration_seconds'] <= 12 and value['peak'] >= 0 and value['rms'] >= 0 and
        0 <= value['clipped_fraction'] <= 1 and type(value['quality_flags']) is list and
        value['quality_flags'] == [f for f in ('source_peak_ge_one', 'near_silent', 'termination_unverified') if f in value['quality_flags']] and
        type(value['technically_valid']) is bool and value['technically_valid'] == (not value['quality_flags']), 'native metric values')


def validate_outcome(outcome, cell, status):
    require(type(outcome) is dict and set(outcome) == set(cell) | OUTCOME_EXTRA and
        all(canonical(outcome[k]) == canonical(v) for k, v in cell.items()), 'native cell identity')
    require(status in ('completed', 'FAILED', 'NOT_RUN') and outcome['status'] == status and
        type(outcome['attempts']) is int and outcome['attempts'] == (0 if status == 'NOT_RUN' else 1), 'native attempt state')
    require(outcome['error_type'] is None if status != 'FAILED' else outcome['error_type'] in ERRORS, 'native error classification')
    if status == 'NOT_RUN':
        require(outcome == initial_outcome(cell), 'unattempted native evidence')
        return outcome
    if outcome['termination'] is not None:
        validate_termination(outcome['termination'])
    evidence = ('file', 'raw_wav_sha256', 'native', 'metrics')
    if outcome['native'] is None:
        require(status == 'FAILED' and all(outcome[k] is None for k in evidence), 'incomplete native evidence')
    else:
        native = outcome['native']
        require(outcome['file'] == cell['slot_id']+'.raw-float.wav' and type(outcome['raw_wav_sha256']) is str and
            SHA.fullmatch(outcome['raw_wav_sha256']) and type(native) is dict and set(native) == NATIVE_KEYS and
            type(native['sample_rate_hz']) is int and native['sample_rate_hz'] == 24000 and
            type(native['channels']) is int and native['channels'] == 1 and type(native['frames']) is int and
            0 < native['frames'] <= 288000 and native['subtype'] == 'FLOAT' and
            type(native['source_values_sha256']) is str and SHA.fullmatch(native['source_values_sha256']), 'native identity fields')
        validate_metrics(outcome['metrics'])
    if status == 'completed':
        require(outcome['native'] is not None and outcome['metrics']['technically_valid'] and
            outcome['termination'] is not None and validate_termination(outcome['termination']), 'native technical completion')
    return outcome


def verify_audio_outcome(raw, outcome):
    require(sha(raw) == outcome['raw_wav_sha256'], 'native WAV bytes identity')
    samples = verify_native(raw, outcome['native'])
    require(canonical(signal_metrics(samples, outcome['termination'])) == canonical(outcome['metrics']), 'native metrics differ from actual samples')
    return True


def stage_claim(engine, plan):
    return dict(schema='kw1-tts-stage-claim-v1', engine=engine, plan_sha256=sha(canonical(plan)), attempt=1)


def started_record(cell, plan, load_sha):
    return dict(schema='kw1-tts-cell-started-v1', engine=cell['engine'], slot_id=cell['slot_id'],
        plan_sha256=sha(canonical(plan)), load_completed_sha256=load_sha, cell_sha256=sha(canonical(cell)), attempt=1)


@contextmanager
def deadline(seconds):
    def alarm(*_): raise TimeoutError('single attempt deadline')
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def error_kind(error):
    return 'timeout' if isinstance(error, TimeoutError) else 'exception'


def progress(output, stage, index):
    # Non-authoritative supervisor heartbeat. Never used for recovery/accounting.
    path = output/'progress.json'; temporary = output/'.progress.tmp'
    write_once(temporary, dict(stage=stage, index=index, started_monotonic=time.monotonic()))
    os.replace(temporary, path); fsync_directory(output)


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'loader source unavailable')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_sources(engine, root=SOURCE_ROOT):
    root = unaliased(root)
    require(engine in ENGINES and bool(RETAINED_PINS) and set(MODEL_LOCK_PINS) == set(ENGINES), 'retained source pins unavailable')
    selected = 'research/tts-quality-pilot-v1/'+engine+'-model-lock.json'
    alias = 'research/tts-quality-pilot-v1/qwen-model-lock.json'
    own = 'research/kw1-preset-supplement-v1/'
    expected = set(RETAINED_PINS) | {selected, alias, own+'worker.py', own+'plan.json'}
    observed = set()
    for path in root.rglob('*'):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), 'source aliases/special files forbidden')
        if path.is_file(): observed.add(path.relative_to(root).as_posix())
    require(observed == expected, 'isolated TTS source inventory differs')
    pins = dict(RETAINED_PINS, **{selected: MODEL_LOCK_PINS[engine], alias: MODEL_LOCK_PINS[engine]})
    for name, digest in pins.items():
        require(sha(read_regular(root/name)) == digest, 'retained TTS source drift')
    validate_plan(decode(read_regular(root/(own+'plan.json'))))
    read_regular(root/(own+'worker.py'))
    return pins


def verify_scope():
    return module_from_file('_kw1_runtime_scope', SOURCE_ROOT/'research/source-screen32-v1/runtime_scope.py').verify_runtime_scope()


class RetainedBackend:
    """Reuse the unmodified retained runtime verifier and guarded local loader."""
    def __init__(self, engine, runtime, model_lock):
        self.engine, self.runtime, self.model_lock = engine, Path(runtime), Path(model_lock)
        self.stack = ExitStack()

    def __enter__(self):
        try:
            self.source_pins = verify_sources(self.engine)
            verify_scope()
            require(unaliased(self.runtime) == Path('/runtime') and
                self.model_lock == SOURCE_ROOT/'research/tts-quality-pilot-v1'/(self.engine+'-model-lock.json'), 'fixed runtime/model lock mount')
            require(platform.python_implementation() == 'CPython' and platform.python_version() == '3.12.14' and
                platform.machine() == 'x86_64' and sys.platform == 'linux' and sys.byteorder == 'little' and
                Path(sys.executable).absolute() == Path('/runtime/venv/bin/python'), 'exact runtime host')
            adapter = module_from_file('_kw1_setup_adapter', SOURCE_ROOT/'research/tts-quality-pilot-v1/setup_adapter.py')
            self.runtime_identity = adapter.verify_runtime(self.runtime, 'tts')
            gate = sys.modules['runtime_gate']
            self.torch, self.transformers, events = gate.prepare()
            import numpy as np
            from qwen_tts import Qwen3TTSModel
            self.np = np
            modeldir = self.runtime/'models/qwen_tts'
            sidecars = gate.inspect_model_sidecars(modeldir); calls = []
            config = decode(read_regular(modeldir/'config.json'))
            preset_map = config.get('talker_config', {}).get('spk_id', {})
            require(all(type(preset_map.get(name.lower())) is int and preset_map[name.lower()] == value for name, value in PRESET_IDS.items()), 'frozen preset config IDs')
            self.stack.enter_context(gate.local_loader_policy(modeldir, self.torch, self.transformers, calls))
            self.model = Qwen3TTSModel.from_pretrained(str(modeldir), device_map='cpu', dtype=self.torch.float32,
                attn_implementation='eager', use_safetensors=True, weights_only=True, local_files_only=True, trust_remote_code=False)
            require(self.model.model.tts_model_type == 'custom_voice' and
                self.model.model.tts_model_size == MODELS[self.engine][2], 'loaded CustomVoice model identity')
            speakers = self.model.get_supported_speakers()
            require(type(speakers) is list and all(type(x) is str for x in speakers) and
                {x.lower() for x in speakers} == set(preset_map), 'actual supported preset list')
            self._check_model()
            self.receipt = dict(model_id=MODELS[self.engine][0], revision=MODELS[self.engine][1],
                sidecars=sidecars, loader_calls=calls, network_events=events,
                platform=dict(python=platform.python_version(), machine=platform.machine(), system=sys.platform),
                versions=dict(torch=self.torch.__version__, numpy=np.__version__, transformers=self.transformers.__version__))
            return self
        except BaseException:
            self.stack.close(); raise

    def _check_model(self):
        inner = self.model.model
        require(inner.tts_model_type == 'custom_voice' and inner.tts_model_size == MODELS[self.engine][2], 'loaded model identity')
        talker = inner.config.talker_config
        require(type(talker.codec_eos_token_id) is int and talker.codec_eos_token_id == 2150 and
            all(type(talker.spk_id.get(name.lower())) is int and talker.spk_id[name.lower()] == value for name, value in PRESET_IDS.items()), 'loaded preset/EOS identities')
        require(all(canonical(talker.spk_is_dialect.get(name)) == canonical(value) for name, value in DIALECTS.items()), 'loaded preset dialect routing')
        speakers = self.model.get_supported_speakers()
        require(type(speakers) is list and all(type(x) is str for x in speakers) and
            len(speakers) == len(set(speakers)) and {x.lower() for x in speakers} == set(talker.spk_id), 'loaded supported speakers')
        for owner in (self.model.model, self.model.model.speech_tokenizer.model):
            parameters = list(owner.parameters())
            require(parameters and not owner.training and all(str(p.device) == 'cpu' and str(p.dtype) == 'torch.float32' for p in parameters), 'CPU FP32 eval required')
            modules = list(owner.named_modules())
            require(modules and all(not module.training for _, module in modules), 'all modules eval required')
            attention = [m for _, m in modules if all(hasattr(m, k) for k in ('q_proj', 'k_proj', 'v_proj'))]
            require(attention and all(getattr(m.config, '_attn_implementation', None) == 'eager' for m in attention), 'eager attention required')

    def generate(self, cell):
        verify_scope(); self._check_model()
        random.seed(cell['seed']); self.np.random.seed(cell['seed']); self.torch.manual_seed(cell['seed'])
        termination = {}; original = self.model.model.talker.generate
        def capture(*args, **kwargs):
            require(not termination, 'one acoustic generation per slot')
            result = original(*args, **kwargs)
            eos = int(self.model.model.config.talker_config.codec_eos_token_id)
            last = result.sequences[:, -1].detach().cpu().tolist()
            iterations = len(result.hidden_states)
            termination.update(iterations=iterations, last_token_ids=last, eos=eos,
                ended_with_eos=bool(last) and all(x == eos for x in last),
                hit_token_cap_without_eos=iterations >= 120 and not (bool(last) and all(x == eos for x in last)))
            return result
        self.model.model.talker.generate = capture
        try:
            with self.torch.inference_mode():
                wavs, rate = self.model.generate_custom_voice(text=cell['target_text'], speaker=cell['speaker'],
                    language=cell['language'], instruct=cell['instruction'], **cell['generation_parameters'])
            require(type(rate) is int and rate == 24000 and len(wavs) == 1, 'native output geometry')
            values = self.np.asarray(wavs[0])
            require(values.ndim == 1 and values.dtype == self.np.float32 and 0 < values.size <= 288000 and
                self.np.isfinite(values).all(), 'unaltered native FLOAT mono required')
            validate_termination(termination)
            return values.astype('<f4', copy=False).tobytes(), dict(termination)
        finally:
            self.model.model.talker.generate = original

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def run(engine, plan, output, runtime, model_lock, *, backend_factory=RetainedBackend):
    validate_plan(plan); require(engine in ENGINES, 'engine choice')
    output = unaliased(output)
    require(output.is_dir() and not any(output.iterdir()), 'fresh empty output required; retry/resume prohibited')
    claim = stage_claim(engine, plan); write_once(output/'stage-claim.json', claim)
    cells = [c for c in plan['slots'] if c['engine'] == engine]
    outcomes = [initial_outcome(c) for c in cells]
    cells_root = output/'cells'; cells_root.mkdir(mode=0o700); fsync_directory(output)
    failure = None; load_started_sha = None; load_sha = None
    seen_values, seen_bytes = set(), set()
    try:
        progress(output, 'load', 0)
        load_started_sha = write_once(output/'model-load-started.json', dict(claim, load_seconds=LOAD_SECONDS))
        with ExitStack() as stack:
            with deadline(LOAD_SECONDS):
                model = stack.enter_context(backend_factory(engine, runtime, model_lock))
                receipt = dict(schema='kw1-public-tts-load-identity-v1', engine=engine,
                    scientific_load_sha256=sha(canonical(model.receipt)), runtime_identity_sha256=sha(canonical(model.runtime_identity)),
                    source_pins_sha256=sha(canonical(model.source_pins)))
                load_sha = write_once(output/'model-load-completed.json', dict(claim, started_sha256=load_started_sha, receipt=receipt))
            for index, (cell, outcome) in enumerate(zip(cells, outcomes), 1):
                progress(output, 'cell', index)
                directory = cells_root/cell['slot_id']; directory.mkdir(mode=0o700); fsync_directory(cells_root)
                started_sha = write_once(directory/'started.json', started_record(cell, plan, load_sha))
                outcome.update(status='FAILED', attempts=1, error_type='interrupted')
                try:
                    with deadline(CALL_SECONDS):
                        values_raw, termination = model.generate(cell)
                        validate_termination(termination); outcome['termination'] = termination
                        raw = float_wav(values_raw); values, samples = native_samples(raw)
                        native = dict(sample_rate_hz=24000, channels=1, frames=len(samples), subtype='FLOAT', source_values_sha256=sha(values))
                        metrics = signal_metrics(samples, termination)
                        filename = cell['slot_id']+'.raw-float.wav'
                        write_bytes_once(output/filename, raw)
                        outcome.update(file=filename, raw_wav_sha256=sha(raw), native=native, metrics=metrics)
                        if not metrics['technically_valid']: failure = 'technical_quality'
                        elif sha(values) in seen_values or sha(raw) in seen_bytes: failure = 'duplicate_native'
                        if failure:
                            outcome['error_type'] = failure
                            validate_outcome(outcome, cell, 'FAILED')
                            write_once(directory/'failed.json', dict(schema='kw1-tts-cell-failed-v1', started_sha256=started_sha, reason=failure, outcome=outcome))
                            break
                        outcome.update(status='completed', error_type=None)
                        validate_outcome(outcome, cell, 'completed'); verify_audio_outcome(raw, outcome)
                        write_once(directory/'completed.json', dict(schema='kw1-tts-cell-completed-v1', started_sha256=started_sha, outcome=outcome))
                        seen_values.add(sha(values)); seen_bytes.add(sha(raw))
                except BaseException as error:
                    failure = error_kind(error); outcome.update(status='FAILED', attempts=1, error_type=failure)
                    # Never overwrite or append a contradictory terminal record.
                    if not any(os.path.lexists(directory/name) for name in ('completed.json', 'failed.json')):
                        write_once(directory/'failed.json', dict(schema='kw1-tts-cell-failed-v1', started_sha256=started_sha, reason=failure, outcome=outcome))
                    break
    except BaseException as error:
        failure = error_kind(error)
        if load_started_sha is not None and load_sha is None and not os.path.lexists(output/'model-load-failed.json'):
            write_once(output/'model-load-failed.json', dict(claim, started_sha256=load_started_sha, reason=failure))
    if failure:
        write_once(output/'stage-failed.json', dict(reason=failure, retries=0))
    status = 'complete' if failure is None and all(r['status'] == 'completed' for r in outcomes) else 'stopped'
    write_once(output/'outcomes.json', outcomes)
    summary = dict(schema='kw1-tts-summary-v1', engine=engine, status=status, requested=len(cells),
        attempted=sum(r['attempts'] for r in outcomes), completed=sum(r['status'] == 'completed' for r in outcomes),
        not_run=sum(r['status'] == 'NOT_RUN' for r in outcomes), retries=0, plan_sha256=sha(canonical(plan)),
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        process_peak_rss_scope='worker process lifetime high-water RSS; not container-tree peak', cgroup_memory_peak_bytes=None)
    write_once(output/'summary.json', summary)
    return summary


def recover(output, engine, plan):
    """Strict read-only recovery; malformed evidence raises and reserves the stage.

    No arbitrary body is returned. A valid start with no terminal consumes one
    attempt. Truncated/malformed TTS receipt bodies are UNVERIFIED at the caller.
    """
    validate_plan(plan); require(engine in ENGINES, 'engine choice')
    output = unaliased(output); cells = [c for c in plan['slots'] if c['engine'] == engine]
    outcomes = [initial_outcome(c) for c in cells]
    if not output.exists() or not any(output.iterdir()): return outcomes
    def document(path): return decode(read_regular(path))
    claim = stage_claim(engine, plan)
    root_names = {'stage-claim.json','model-load-started.json','model-load-completed.json','model-load-failed.json','stage-failed.json','outcomes.json','summary.json','progress.json','.progress.tmp','cells'} | {c['slot_id']+'.raw-float.wav' for c in cells}
    require({p.name for p in output.iterdir()} <= root_names, 'unexpected TTS output entry')
    require(canonical(document(output/'stage-claim.json')) == canonical(claim), 'stage claim identity')
    load_start = output/'model-load-started.json'; load_done = output/'model-load-completed.json'; load_fail = output/'model-load-failed.json'
    require(not (load_done.exists() and load_fail.exists()), 'contradictory model load evidence')
    if load_start.exists():
        require(canonical(document(load_start)) == canonical(dict(claim, load_seconds=LOAD_SECONDS)), 'model load start')
    else:
        require(not load_done.exists() and not load_fail.exists(), 'model load parent missing')
    load_sha = None
    if load_done.exists():
        raw = read_regular(load_done); completed = decode(raw)
        expected = dict(claim, started_sha256=sha(read_regular(load_start)))
        require(type(completed) is dict and set(completed) == set(expected)|{'receipt'} and
            all(canonical(completed[k]) == canonical(v) for k, v in expected.items()), 'model load completion chain')
        receipt = completed['receipt']
        require(type(receipt) is dict and set(receipt) == {'schema','engine','scientific_load_sha256','runtime_identity_sha256','source_pins_sha256'} and
            receipt['schema'] == 'kw1-public-tts-load-identity-v1' and receipt['engine'] == engine and
            all(type(receipt[k]) is str and SHA.fullmatch(receipt[k]) for k in ('scientific_load_sha256','runtime_identity_sha256','source_pins_sha256')), 'model load identity projection')
        load_sha = sha(raw)
    if load_fail.exists():
        failed = document(load_fail)
        require(type(failed) is dict and set(failed) == set(claim)|{'started_sha256','reason'} and
            all(canonical(failed[k]) == canonical(v) for k, v in claim.items()) and failed['started_sha256'] == sha(read_regular(load_start)) and
            failed['reason'] in ('exception','timeout'), 'model load failed chain')
    if (output/'stage-failed.json').exists():
        stage_failed = document(output/'stage-failed.json')
        require(type(stage_failed) is dict and set(stage_failed) == {'reason','retries'} and
            stage_failed['reason'] in ERRORS and type(stage_failed['retries']) is int and stage_failed['retries'] == 0, 'stage failure schema')
    root = output/'cells'
    for cell in cells:
        if os.path.lexists(output/(cell['slot_id']+'.raw-float.wav')):
            require(os.path.lexists(root/cell['slot_id']/'started.json'), 'native audio without durable started receipt')
    if not root.exists(): return outcomes
    require(root.is_dir() and not root.is_symlink() and {p.name for p in root.iterdir()} <= {c['slot_id'] for c in cells}, 'cell directory inventory')
    stopped = False; seen_values = set(); seen_bytes = set()
    for cell, outcome in zip(cells, outcomes):
        directory = root/cell['slot_id']
        if not directory.exists():
            stopped = True; continue
        require(directory.is_dir() and not directory.is_symlink() and not stopped, 'nonserial native receipt directory')
        names = {p.name for p in directory.iterdir()}
        require(names <= {'started.json','completed.json','failed.json'}, 'native receipt inventory')
        if not names:
            stopped = True; continue
        require('started.json' in names and load_sha is not None, 'native start parent missing')
        started_raw = read_regular(directory/'started.json')
        require(canonical(decode(started_raw)) == canonical(started_record(cell, plan, load_sha)), 'native started identity')
        require(not {'completed.json','failed.json'} <= names, 'conflicting native terminals')
        outcome.update(status='FAILED', attempts=1, error_type='interrupted')
        terminal_name = 'completed.json' if 'completed.json' in names else ('failed.json' if 'failed.json' in names else None)
        if terminal_name is None:
            stopped = True; continue
        terminal = document(directory/terminal_name); success = terminal_name == 'completed.json'
        keys = {'schema','started_sha256','outcome'} | (set() if success else {'reason'})
        require(type(terminal) is dict and set(terminal) == keys and terminal['schema'] ==
            ('kw1-tts-cell-completed-v1' if success else 'kw1-tts-cell-failed-v1') and terminal['started_sha256'] == sha(started_raw), 'native terminal chain')
        accepted = validate_outcome(terminal['outcome'], cell, 'completed' if success else 'FAILED')
        if not success: require(terminal['reason'] == accepted['error_type'], 'native failure reason')
        if accepted['native'] is not None:
            verify_audio_outcome(read_regular(output/accepted['file']), accepted)
        if success:
            require(accepted['native']['source_values_sha256'] not in seen_values and accepted['raw_wav_sha256'] not in seen_bytes, 'duplicate native completion')
            seen_values.add(accepted['native']['source_values_sha256']); seen_bytes.add(accepted['raw_wav_sha256'])
        else: stopped = True
        outcome.update(accepted)
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=ENGINES, required=True)
    for name in ('plan', 'runtime', 'model-lock', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.plan == HERE/'plan.json' and args.runtime == Path('/runtime') and args.output == Path('/output') and
        args.model_lock == SOURCE_ROOT/'research/tts-quality-pilot-v1'/(args.engine+'-model-lock.json'), 'fixed mounted paths')
    summary = run(args.engine, decode(read_regular(args.plan)), args.output, args.runtime, args.model_lock)
    return 0 if summary['status'] == 'complete' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        print(canonical(dict(status='blocked', reason=error_kind(error))).decode(), file=sys.stderr)
        raise SystemExit(2)
