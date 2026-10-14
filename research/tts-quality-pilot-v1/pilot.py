"""New experiment adapter. No downloader, model loader or workflow dispatch.

Root launcher supplies a verified, offline loaded CustomVoice model inside the
existing source-screen32 containment. Probe is cell 0, not an extra generation.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import time

HERE = Path(__file__).resolve().parent
VOICES = ('Vivian', 'Serena', 'Uncle_Fu', 'Dylan')
TEXTS = ('你好小窝', '小窝小窝')
MODELS = {
    'qwen06': ('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', '85e237c12c027371202489a0ec509ded67b5e4b5'),
    'qwen17': ('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', '0c0e3051f131929182e2c023b9537f8b1c68adfe'),
}

def plan(engine):
    if engine not in MODELS:
        raise ValueError('CosyVoice3 is blocked pending four verified reference voices and runtime lock')
    return [dict(cell_id=f'tqp-{engine}-{i:02d}', engine=engine, speaker=voice,
                 text=text, seed=20261014 + repeat, repeat=repeat, resource_probe=(i == 0))
            for i, (voice, text, repeat) in enumerate(
                (v, t, r) for v in VOICES for t in TEXTS for r in range(2))]

def scope():
    path = HERE.parent / 'source-screen32-v1/runtime_scope.py'
    spec = importlib.util.spec_from_file_location('tqp_scope', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_runtime_scope()

def save(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def kwargs(cell):
    # No instruction differential: 0.6B lacks instruction control. Exact Hanzi
    # in both lanes; punctuation/pinyin experiments would be separate factors.
    return dict(text=cell['text'], language='Chinese', speaker=cell['speaker'],
                instruct='', do_sample=True, top_k=50, top_p=1.0,
                temperature=0.9, repetition_penalty=1.05,
                subtalker_dosample=True, subtalker_top_k=50,
                subtalker_top_p=1.0, subtalker_temperature=0.9,
                max_new_tokens=120, non_streaming_mode=True)

def summarize(values, rate, ended_with_eos):
    samples = [float(x) for x in values]
    if rate != 24000 or not samples or len(samples) > 12 * rate or not all(map(math.isfinite, samples)):
        raise ValueError('invalid native audio geometry or values')
    peak = max(map(abs, samples))
    rms = math.sqrt(sum(x*x for x in samples)/len(samples))
    flags = []
    if peak >= 1: flags.append('source_peak_ge_one')
    if rms < 1e-4: flags.append('near_silent')
    if not ended_with_eos: flags.append('termination_unverified')
    return dict(duration_seconds=len(samples)/rate, peak=peak, rms=rms,
                clipped_fraction=sum(abs(x)>=1 for x in samples)/len(samples),
                quality_flags=flags, technically_valid=not flags,
                human_phonetic_review='PENDING', naturalness='PENDING',
                pronunciation_target='PENDING', qualified=False, training_admitted=False)

def qualified(metrics, review):
    # ASR consensus is evidence, never the ground-truth acceptance rule.
    required = ('exact_target', 'no_missing_or_added_syllables', 'natural', 'no_tail_artifact')
    return metrics.get('technically_valid') is True and all(review.get(k) is True for k in required)

def run_loaded(model, torch, np, sf, engine, output, phase):
    """Call in reviewed container; phase probe or remaining; no retries/resampling.

    Loader and model-file checksum verification belong to the existing launcher.
    Terminal failures leave a durable claim and block the remaining lane.
    """
    scope()
    if phase not in ('probe', 'remaining'): raise ValueError('invalid phase')
    cells = plan(engine)
    if model.model.tts_model_type != 'custom_voice': raise ValueError('wrong model type')
    expected_size = {'qwen06': '0b6', 'qwen17': '1b7'}[engine]
    if model.model.tts_model_size != expected_size: raise ValueError('wrong model size')
    speakers = {x.lower() for x in model.get_supported_speakers()}
    if not all(v.lower() in speakers for v in VOICES): raise ValueError('missing preset')
    for owner in (model.model, model.model.speech_tokenizer.model):
        if owner.training or any(str(p.device) != 'cpu' or str(p.dtype) != 'torch.float32' for p in owner.parameters()):
            raise ValueError('CPU FP32 eval required')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / (engine + '-receipt.json')
    if phase == 'probe':
        if receipt_path.exists(): raise ValueError('probe already attempted')
        receipt = dict(engine=engine, model_id=MODELS[engine][0], revision=MODELS[engine][1],
                       status='probe_started', outcomes=[], qualified_count=0)
        chosen = cells[:1]
    else:
        receipt = json.loads(receipt_path.read_text())
        if receipt['status'] != 'probe_passed' or len(receipt['outcomes']) != 1:
            raise ValueError('successful single-cell probe required')
        chosen = cells[1:]
    save(receipt_path, receipt)
    termination = {}
    original = model.model.talker.generate
    def capture(*args, **kw):
        result = original(*args, **kw)
        eos = int(model.model.config.talker_config.codec_eos_token_id)
        last = result.sequences[:, -1].detach().cpu().tolist()
        termination['eos'] = bool(last) and all(x == eos for x in last)
        return result
    def timeout(*_): raise TimeoutError('per-cell 300 second limit')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    model.model.talker.generate = capture
    try:
        for cell in chosen:
            scope()
            with (output / (cell['cell_id']+'.claim')).open('x') as f:
                f.write(cell['cell_id']); f.flush(); os.fsync(f.fileno())
            row = dict(cell, status='attempt_started', attempts=1)
            receipt['outcomes'].append(row); save(receipt_path, receipt)
            random.seed(cell['seed']); np.random.seed(cell['seed']); torch.manual_seed(cell['seed'])
            termination.clear(); started = time.monotonic(); signal.alarm(300)
            save(output / 'progress.json', dict(stage='cell', index=cells.index(cell)+1, started_monotonic=started))
            with torch.inference_mode(): wavs, rate = model.generate_custom_voice(**kwargs(cell))
            if len(wavs) != 1 or wavs[0].ndim != 1: raise ValueError('one mono output required')
            metrics = summarize(wavs[0], rate, termination.get('eos') is True)
            path = output / (cell['cell_id']+'.raw-float.wav')
            sf.write(path, wavs[0], rate, subtype='FLOAT')
            signal.alarm(0)
            row.update(status='generated', seconds=time.monotonic()-started, metrics=metrics,
                       raw_wav_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       native=dict(sample_rate_hz=rate,channels=1,frames=len(wavs[0]),subtype='FLOAT',
                           source_values_sha256=hashlib.sha256(np.asarray(wavs[0],dtype='<f4').tobytes()).hexdigest()))
            if not metrics['technically_valid']: raise ValueError('technical audio gate failed; lane stopped')
            save(receipt_path, receipt)
        receipt['status'] = 'probe_passed' if phase == 'probe' else 'generated_16_pending_review'
    except BaseException as e:
        receipt['status'] = 'stopped_no_retry'; receipt['error_type'] = type(e).__name__
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        model.model.talker.generate = original; save(receipt_path, receipt)
    return receipt

if __name__ == '__main__':
    print(json.dumps({e: plan(e) for e in MODELS}, ensure_ascii=False, indent=2))
