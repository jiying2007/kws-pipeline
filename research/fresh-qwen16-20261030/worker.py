"""Fixed CPU synthesis worker. Called only inside the retained offline scope."""
from pathlib import Path
import hashlib
import json
import math
import random
import signal
import sys
import time
import wave

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contract import (CAMPAIGN, MODEL_ID, MODEL_REVISION, RECIPE, RETAINED_COMMIT,
                      VOICES, digest, file_hash, plan, require, save, signal_metrics)


def main():
    output = Path('/output')
    receipt = dict(schema='fresh-qwen16-generation-v1', campaign=CAMPAIGN,
                   model_id=MODEL_ID, model_revision=MODEL_REVISION,
                   retained_commit=RETAINED_COMMIT, plan_sha256=digest(plan()),
                   recipe=RECIPE, recipe_sha256=digest(RECIPE), status='started',
                   stage='scope', planned=16, outcomes=[], asr_calls=0,
                   kws_training_calls=0, training_admitted=False,
                   qualification_allowed=False, human_gold=False)
    target = output / 'generation-receipt.json'
    require(not target.exists(), 'no resume or regeneration')
    save(target, receipt)
    original = None
    talker = None
    previous_handler = None
    try:
        sys.path.insert(0, str(HERE.parent / 'tts-quality-pilot-v1'))
        from setup_adapter import verify_runtime
        receipt['stage'] = 'verify_runtime'
        save(target, receipt)
        save(output / 'progress.json', dict(stage='load', index=0,
                                           started_monotonic=time.monotonic()))
        verify_runtime(Path('/runtime'), 'tts')
        gate = sys.modules['runtime_gate']
        receipt['stage'] = 'import_runtime'
        save(target, receipt)
        torch, transformers, network_events = gate.prepare()
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        from qwen_tts import Qwen3TTSModel
        model_dir = Path('/runtime/models/qwen_tts')
        receipt['rights_evidence_sha256'] = file_hash(model_dir / 'README.md')
        receipt['rights_evidence_kind'] = 'pinned-public-model-card-presets-no-personal-reference'
        receipt['license_id'] = 'Apache-2.0'
        sidecars = gate.inspect_model_sidecars(model_dir)
        loader_calls = []
        receipt['stage'] = 'load_model'
        save(target, receipt)
        with gate.local_loader_policy(model_dir, torch, transformers, loader_calls):
            model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map='cpu',
                dtype=torch.float32, attn_implementation='eager', use_safetensors=True,
                weights_only=True, local_files_only=True, trust_remote_code=False)
            require(model.model.tts_model_type == 'custom_voice' and
                    model.model.tts_model_size == '0b6', 'exact loaded model type required')
            require(set(v.lower() for v in VOICES) <=
                    set(v.lower() for v in model.get_supported_speakers()), 'preset unavailable')
            for owner in (model.model, model.model.speech_tokenizer.model):
                require(not owner.training and all(str(p.device) == 'cpu' and
                        str(p.dtype) == 'torch.float32' for p in owner.parameters()),
                        'CPU FP32 evaluation model required')
            save(output / 'runtime-observations.json', dict(sidecars=sidecars,
                 loader_calls=loader_calls, network_events=network_events))
            termination = {}
            talker = model.model.talker
            original = talker.generate

            def capture(*args, **kwargs):
                result = original(*args, **kwargs)
                eos = int(model.model.config.talker_config.codec_eos_token_id)
                last = result.sequences[:, -1].detach().cpu().tolist()
                termination.update(codec_eos_token_id=eos, last_token_ids=last,
                    ended_with_eos=bool(last) and all(token == eos for token in last))
                return result

            def timeout(*_):
                raise TimeoutError('fixed per-cell wall budget')

            talker.generate = capture
            previous_handler = signal.signal(signal.SIGALRM, timeout)
            for index, cell in enumerate(plan()):
                row = dict(cell, status='attempt_started', attempts=1)
                with (output / (cell['cell_id'] + '.claim')).open('x') as claim:
                    claim.write(cell['cell_id'] + '\n')
                    claim.flush()
                    import os
                    os.fsync(claim.fileno())
                receipt['outcomes'].append(row)
                receipt['stage'] = 'generation'
                save(target, receipt)
                save(output / 'progress.json', dict(stage='cell', index=index + 1,
                                                   started_monotonic=time.monotonic()))
                random.seed(cell['seed'])
                np.random.seed(cell['seed'])
                torch.manual_seed(cell['seed'])
                termination.clear()
                started = time.monotonic()
                signal.alarm(300)
                try:
                    with torch.inference_mode():
                        wavs, rate = model.generate_custom_voice(text=cell['intended_text'],
                                                                speaker=cell['speaker'], **RECIPE)
                    require(len(wavs) == 1 and wavs[0].ndim == 1, 'one mono output required')
                    metrics = signal_metrics(wavs[0], rate, termination.get('ended_with_eos') is True)
                    raw_path = output / (cell['cell_id'] + '.raw-float.wav')
                    sf.write(raw_path, wavs[0], rate, subtype='FLOAT')
                    row.update(native=metrics, termination=dict(termination),
                        raw_wav_sha256=file_hash(raw_path),
                        source_values_sha256=hashlib.sha256(np.asarray(wavs[0], dtype='<f4').tobytes()).hexdigest())
                    save(target, receipt)
                    require(metrics['technically_valid'], 'native technical gate failed')
                    derived = resample_poly(np.asarray(wavs[0], dtype=np.float64), 2, 3)
                    require(derived.ndim == 1 and len(derived) == math.ceil(len(wavs[0]) * 2 / 3)
                            and np.all(np.isfinite(derived)) and np.max(np.abs(derived)) < 1.0,
                            'invalid or clipping resampled signal; no gain repair')
                    path = output / (cell['cell_id'] + '.wav')
                    sf.write(path, derived, 16000, subtype='PCM_16')
                    with wave.open(str(path), 'rb') as reader:
                        require((reader.getframerate(), reader.getnchannels(), reader.getsampwidth(),
                                 reader.getnframes()) == (16000, 1, 2, len(derived)), 'PCM geometry mismatch')
                        pcm = reader.readframes(reader.getnframes())
                    require(len(pcm) == 2 * len(derived), 'incomplete PCM')
                    row.update(status='generated_pending_label_review', wav_sha256=file_hash(path),
                        pcm_sha256=hashlib.sha256(pcm).hexdigest(), frames=len(derived),
                        sample_rate_hz=16000, wall_seconds=time.monotonic() - started,
                        derivative=dict(method='scipy.signal.resample_poly', up=2, down=3,
                                        input_dtype='float64', peak=float(np.max(np.abs(derived)))))
                    save(target, receipt)
                except BaseException as error:
                    row.update(status='failed_no_retry', error_type=type(error).__name__,
                               wall_seconds=time.monotonic() - started)
                    save(target, receipt)
                    raise
                finally:
                    signal.alarm(0)
            receipt['status'] = 'generated_16_pending_label_review'
    except BaseException as error:
        receipt.update(status='stopped_no_retry', error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        if previous_handler is not None:
            signal.signal(signal.SIGALRM, previous_handler)
        if talker is not None and original is not None:
            talker.generate = original
        receipt['attempted'] = len(receipt['outcomes'])
        receipt['generated'] = sum(row['status'] == 'generated_pending_label_review'
                                   for row in receipt['outcomes'])
        save(target, receipt)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print(json.dumps(dict(status='stopped_no_retry', error_type=type(error).__name__)), file=sys.stderr)
        raise SystemExit(1)
