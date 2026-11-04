"""Sixteen fixed per-lane non-clone preset calls, under the retained offline CPU runtime gate."""
from pathlib import Path
import hashlib
import json
import os
import random
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from campaign import CAMPAIGN, VOICES, POLICY, lane_plan, require
sys.path.insert(0, str(HERE.parent/'tts-quality-pilot-v1'))
from pilot import kwargs, summarize, save, scope


def generate(model, torch, np, sf, output, lane):
    cells = lane_plan(lane)
    require(model.model.tts_model_type == 'custom_voice' and
            model.model.tts_model_size == '0b6', 'EXACT_CUSTOMVOICE_REQUIRED')
    speakers = model.get_supported_speakers()
    require(type(speakers) in (list, tuple) and all(type(s) is str for s in speakers), 'PRESET_SHAPE')
    require({v.lower() for v in VOICES} <= {s.lower() for s in speakers}, 'ALL_FOUR_PRESETS_REQUIRED')
    for owner in (model.model, model.model.speech_tokenizer.model):
        require(not owner.training and all(str(p.device)=='cpu' and str(p.dtype)=='torch.float32'
                for p in owner.parameters()), 'CPU_FP32_EVAL_REQUIRED')
    receipt_path = output/'generation.json'
    require(not receipt_path.exists() and not list(output.glob('*.claim')), 'ATTEMPT_ALREADY_STARTED')
    receipt = {'campaign':CAMPAIGN,'source_kind':'synthetic-public-preset','planned':cells,
               'lane':lane,'expected_speakers':list(VOICES),'supported_speakers':sorted(speakers),
               'policy':POLICY,'outcomes':[],'status':'started','training_admitted':False,'human_gold':False}
    save(receipt_path,receipt)
    termination = {}
    original = model.model.talker.generate
    def capture(*args, **kw):
        result = original(*args, **kw)
        eos = int(model.model.config.talker_config.codec_eos_token_id)
        last = result.sequences[:, -1].detach().cpu().tolist()
        termination['eos'] = bool(last) and all(x == eos for x in last)
        return result
    def timeout(*unused):
        raise TimeoutError('CELL_300_SECOND_LIMIT')
    previous = signal.signal(signal.SIGALRM, timeout)
    model.model.talker.generate = capture
    try:
        for index, cell in enumerate(cells):
            scope()
            with (output/(cell['cell_id']+'.claim')).open('x') as claim:
                claim.write(cell['cell_id']); claim.flush(); os.fsync(claim.fileno())
            row = dict(cell,status='attempt_started')
            receipt['outcomes'].append(row); save(receipt_path,receipt)
            random.seed(cell['seed']); np.random.seed(cell['seed']); torch.manual_seed(cell['seed'])
            termination.clear(); started=time.monotonic(); signal.alarm(300)
            save(output/'progress.json',dict(stage='cell',index=index+1,started_monotonic=started))
            with torch.inference_mode():
                wavs,rate=model.generate_custom_voice(**kwargs(cell))
            require(len(wavs)==1 and wavs[0].ndim==1, 'ONE_MONO_OUTPUT_REQUIRED')
            metrics=summarize(wavs[0],rate,termination.get('eos') is True)
            path=output/(cell['cell_id']+'.raw-float.wav')
            sf.write(path,wavs[0],rate,subtype='FLOAT')
            signal.alarm(0)
            row.update(status='generated',seconds=time.monotonic()-started,metrics=metrics,
                       raw_wav_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       native={'sample_rate_hz':rate,'channels':1,'frames':len(wavs[0]),'subtype':'FLOAT',
                       'source_values_sha256':hashlib.sha256(np.asarray(wavs[0],dtype='<f4').tobytes()).hexdigest()})
            save(receipt_path,receipt)
            require(metrics['technically_valid'], 'AUDIO_TECHNICAL_GATE_FAILED')
        receipt['status']='lane_generated_pending_independent_screen'
    except BaseException as exc:
        receipt.update(status='stopped_no_retry',error_type=type(exc).__name__)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous)
        model.model.talker.generate=original
        save(receipt_path,receipt)
    return receipt


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('0','1','2','3'), 'FIXED_LANE_ARGUMENT')
    lane=int(sys.argv[1])
    output=Path('/output')
    status={'campaign':CAMPAIGN,'status':'started','stage':'scope','training_admitted':False}
    sidecars=None; calls=[]; events=[]
    try:
        scope()
        save(output/'progress.json',dict(stage='load',index=0,started_monotonic=time.monotonic()))
        from setup_adapter import verify_runtime,engine
        require(engine=='qwen06','ENGINE_CHANGED')
        verified=verify_runtime(Path('/runtime'),'tts')
        verification_json=json.dumps(verified,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        save(output/'runtime-verification.json',{
            'campaign':CAMPAIGN,'status':'completed_before_model_import',
            'verification_sha256':hashlib.sha256(verification_json).hexdigest(),
            'installed':[{'name':r.get('name'),'version':r.get('version'),
                          'metadata_sha256':r.get('metadata_sha256')} for r in verified['installed']],
            'source_members':verified['source_members'],'model_assets':verified['model_assets'],
            'model_id':'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
            'model_revision':'85e237c12c027371202489a0ec509ded67b5e4b5',
            'runtime_lock_sha256':hashlib.sha256((HERE.parent/'qwen6_tts/runtime-lock.json').read_bytes()).hexdigest(),
            'training_admitted':False})
        gate=sys.modules['runtime_gate']
        status['stage']='runtime_import'
        torch,transformers,events=gate.prepare()
        import numpy as np
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
        model_dir=Path('/runtime/models/qwen_tts')
        sidecars=gate.inspect_model_sidecars(model_dir); calls=[]
        status['stage']='load'
        with gate.local_loader_policy(model_dir,torch,transformers,calls):
            model=Qwen3TTSModel.from_pretrained(str(model_dir),device_map='cpu',dtype=torch.float32,
                attn_implementation='eager',use_safetensors=True,weights_only=True,
                local_files_only=True,trust_remote_code=False)
            save(output/'runtime-observations.json',dict(sidecars=sidecars,loader_calls=calls,network_events=events))
            status['stage']='generation'
            generate(model,torch,np,sf,output,lane)
        status['status']='lane_generated_pending_independent_screen'
    except BaseException as exc:
        status.update(status='stopped_no_retry',error_type=type(exc).__name__)
        raise
    finally:
        save(output/'runtime-observations.json',dict(sidecars=sidecars,loader_calls=calls,network_events=events))
        save(output/'worker-status.json',status)

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        print(json.dumps({'status':'stopped_no_retry','error_type':type(exc).__name__}))
        raise SystemExit(1)
