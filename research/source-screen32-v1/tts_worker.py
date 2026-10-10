"""One fixed sixteen-cell worker. Launched only by the reviewed container runner."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import time
import wave

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runtime_scope import verify_runtime_scope
from contract import expected_plan, validate_generation_ledger, blind_job, require, canonical
from qwen_adapter import FixedQwenAdapter


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as target:
        json.dump(value, target, ensure_ascii=False, sort_keys=True, allow_nan=False)
        target.flush(); os.fsync(target.fileno())
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_helper(name, filename):
    path = HERE.parent / 'qwen6_tts' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def initial_ledger():
    return [{'cell_id': cell['cell_id'], 'status': 'NOT_RUN', 'attempts': 0, 'audio': None}
            for cell in expected_plan()['cells']]


def capture_eos(wrapper):
    """Same talker termination observation as retained Qwen6, with no extra call."""
    original = wrapper.model.talker.generate
    termination = {}
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        last = result.sequences[:, -1].detach().cpu().tolist()
        eos = int(wrapper.model.config.talker_config.codec_eos_token_id)
        iterations = len(result.hidden_states)
        ended = bool(last) and all(value == eos for value in last)
        termination.update(iterations=iterations, last_token_ids=last, eos=eos,
                           ended_with_eos=ended, hit_token_cap_without_eos=iterations >= 120 and not ended)
        return result
    wrapper.model.talker.generate = capture
    return termination


def write_audio(array, native, output, cell_id, np, sf, resample_poly, audio_helpers):
    """Preserve raw float32, then fixed 2:3 resampling, no gain or trimming repair."""
    raw = output / (cell_id + '.raw-float.wav')
    derived = output / (cell_id + '.wav')
    sf.write(raw, array, 24000, subtype='FLOAT')
    restored, rate = sf.read(raw, dtype='float32')
    require(rate == 24000 and restored.ndim == 1 and
            hashlib.sha256(np.asarray(restored, dtype='<f4').tobytes()).hexdigest() ==
            native['native_float_values_sha256'], 'raw FLOAT file/value binding')
    normalized = resample_poly(array.astype(np.float64), 2, 3)
    require(normalized.ndim == 1, 'derived mono waveform required')
    peak, flags = audio_helpers.derivative_signal(normalized, math.ceil(len(array) * 2 / 3))
    sf.write(derived, normalized, 16000, subtype='PCM_16')
    with wave.open(str(derived), 'rb') as handle:
        frames = handle.getnframes()
        require((handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (16000, 1, 2),
                'derived PCM geometry')
        require(frames == len(normalized), 'derived frame count')
        pcm = handle.readframes(frames)
    require(len(pcm) == 2 * frames, 'truncated derived PCM')
    audio = {'wav_sha256': file_hash(derived), 'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
             'frames': frames, 'sample_rate_hz': 16000, 'channels': 1, 'sample_width_bytes': 2}
    return audio, {'raw_wav_sha256': file_hash(raw), 'derived_peak': peak,
                   'derived_quality_flags': flags, 'raw_file': raw.name, 'derived_file': derived.name}


def run_cells(adapter, termination, output, writer):
    """Durable claims precede calls. Any exception stops the source, with no resume."""
    output = Path(output)
    ledger = initial_ledger()
    receipt = {'schema': 'screen32-tts-receipt-v1', 'status': 'started', 'ledger': ledger,
               'outcomes': [], 'human_review': 'PENDING', 'human_gold': False, 'training_admitted': False}
    target = output / 'tts-receipt.json'
    require(not target.exists(), 'TTS source already attempted; no resume')
    atomic(target, receipt)
    try:
        for index, row in enumerate(ledger[:16]):
            verify_runtime_scope()
            cell_id = row['cell_id']
            # Exclusive durable claim is deliberately separate from replaceable receipts.
            with (output / (cell_id + '.claim')).open('x') as claim:
                claim.write(cell_id + '\n'); claim.flush(); os.fsync(claim.fileno())
            row.update(status='FAILED_NO_RETRY', attempts=1)
            outcome = {'cell_id': cell_id, 'status': 'attempt_started', 'attempts': 1}
            receipt['outcomes'].append(outcome); atomic(target, receipt)
            termination.clear()
            atomic(output / 'progress.json', {'stage': 'cell', 'index': index + 1,
                                              'started_monotonic': time.monotonic()})
            signal.alarm(300)
            array, observation = adapter.generate_cell(cell_id)
            audio, files = writer(array, observation['native'], cell_id)
            observation['native']['termination_verified'] = termination.get('ended_with_eos') is True
            flags = observation['native']['quality_flags']
            if not termination.get('ended_with_eos'): flags.append('generation_end_unverified')
            if termination.get('hit_token_cap_without_eos'): flags.append('token_cap_without_eos')
            row.update(status='GENERATED', audio=audio)
            validate_generation_ledger(expected_plan(), ledger)
            outcome.update(status='generated_candidate', observation=observation,
                           termination=dict(termination), files=files)
            atomic(target, receipt); signal.alarm(0)
        receipt['status'] = 'completed_16_candidates'
    except BaseException as error:
        receipt['status'] = 'stopped_no_retry'
        receipt['error_type'] = type(error).__name__
        if receipt['outcomes'] and receipt['outcomes'][-1]['status'] == 'attempt_started':
            receipt['outcomes'][-1]['status'] = 'failed_no_retry'
            ledger[len(receipt['outcomes']) - 1].update(status='FAILED_NO_RETRY', audio=None)
        raise
    finally:
        signal.alarm(0)
        atomic(target, receipt)
    return receipt


def run(runtime, output):
    scope = verify_runtime_scope()  # before setup verification and all third-party imports
    runtime, output = Path(runtime), Path(output)
    require(runtime == Path('/runtime') and output == Path('/output'), 'fixed container paths required')
    output.mkdir(exist_ok=True)
    require(not any(output.iterdir()), 'fresh output directory required')
    atomic(output / 'progress.json', {'stage': 'load', 'index': 0, 'started_monotonic': time.monotonic()})
    atomic(output / 'kernel-scope.json', scope)
    from setup_adapter import verify_runtime
    verify_runtime(runtime, 'tts')
    sys.path.insert(0, str(HERE.parent / 'qwen6_tts'))
    gate = load_helper('screen32_tts_gate', 'runtime_gate.py')
    cache_keys = tuple(gate.cache_paths())
    gate.cache_paths = lambda: {key: Path('/scratch') / key.lower() for key in cache_keys}
    torch, transformers, network_events = gate.prepare()
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from qwen_tts import Qwen3TTSModel
    preflight = load_helper('screen32_native_io', 'import_preflight.py')
    native_io = preflight.native_io(np, sf, 'tts', native_root=runtime / 'venv')
    audio_helpers = load_helper('screen32_audio_helpers', 'generate_six.py')
    model_dir = runtime / 'models/qwen_tts'
    sidecars = gate.inspect_model_sidecars(model_dir)
    loader_calls = []
    def alarm(_signal, _frame):
        raise TimeoutError('fixed per-cell deadline')
    signal.signal(signal.SIGALRM, alarm)
    with gate.local_loader_policy(model_dir, torch, transformers, loader_calls):
        verify_runtime_scope()
        model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map='cpu', dtype=torch.float32,
                    attn_implementation='eager', use_safetensors=True, weights_only=True,
                    local_files_only=True, trust_remote_code=False)
        adapter = FixedQwenAdapter(model, torch=torch, numpy=np)
        termination = capture_eos(model)
        atomic(output / 'runtime-observations.json', {'native_io': native_io, 'sidecars': sidecars,
                  'loader_calls': loader_calls, 'network_events': network_events})
        writer = lambda array, native, cell: write_audio(array, native, output, cell, np, sf,
                                                         resample_poly, audio_helpers)
        return run_cells(adapter, termination, output, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        run(args.runtime, args.output)
    except BaseException as error:
        # Container logs retain no raw prompt, paths, package stderr or tracebacks.
        print(json.dumps({'status': 'stopped_no_retry', 'error_type': type(error).__name__}))
        raise SystemExit(1)
