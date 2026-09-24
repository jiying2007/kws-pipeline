#!/usr/bin/env python3
"""Exact C startup/continuous regression on a frozen development pool.

No retraining, decoder override, threshold search or protected data. Edited-prefix
counterfactuals are kept separate from authentic C outputs and have no authority.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import subprocess
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'training'), str(ROOT/'eval'), str(ROOT/'tools')]
from frozen_speech_ablation import verify_pool, resolve, sha, write, acoustic_diagnostics
from score_events import validate_recordings, validate_detections, score
from diagnose_sequence_margin_runtime_gap import read_trace_logits

PREFIX_SAMPLES = (0, 3200, 16000, 32000, 80000)
POLICY = 'exact-c-startup-context-regression-v1'


def pcm(path: pathlib.Path) -> bytes:
    with wave.open(str(path), 'rb') as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('expected mono PCM16 16kHz')
        return f.readframes(f.getnframes())


def wav(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(raw)


def shifted(events: list[dict], samples: int) -> list[dict]:
    if type(samples) is not int or samples < 0 or samples % 320:
        raise ValueError('prefix must be nonnegative hop-aligned samples')
    return [{**e, 'start_s': e['start_s'] + samples/16000,
                   'end_s': e['end_s'] + samples/16000} for e in events]


def measure(refs: list[dict], dets: list[dict]) -> dict:
    recordings = validate_recordings(refs)
    summary, _, _ = score(recordings, validate_detections(dets, recordings), 0.15, 0.5)
    return summary


def collect(runner: pathlib.Path, model: pathlib.Path, pack: pathlib.Path,
            audio: pathlib.Path, recording: str) -> list[dict]:
    p = subprocess.run([str(runner), str(model), str(pack), str(audio), recording],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=30)
    return [json.loads(s) for s in p.stdout.splitlines() if s.strip()]


def blank_prefix(trace: pathlib.Path, output: pathlib.Path, prefix_samples: int) -> int:
    """Only full frames within the *added* silence, never speech, are edited."""
    read_trace_logits(trace)  # Validates header geometry and finite original data.
    data = bytearray(trace.read_bytes()); vocab = struct.unpack_from('<H', data, 12)[0]
    count = struct.unpack_from('<Q', data, 40)[0]; stride = 16 + vocab * 4; edited = 0
    for i in range(count):
        offset = 120 + i*stride
        if struct.unpack_from('<Q', data, offset)[0] <= prefix_samples:
            struct.pack_into('<'+'f'*vocab, data, offset+16, 0.0, *([-80.0]*(vocab-1))); edited += 1
    output.write_bytes(data)
    write(output.with_suffix('.counterfactual.json'), {'counterfactual': True, 'authentic_model_output': False,
        'release_authority': False, 'source_trace_sha256': sha(trace), 'edited_trace_sha256': sha(output),
        'added_silence_samples': prefix_samples, 'edited_frames': edited})
    return edited


def evaluate(pool_root: pathlib.Path, pool_sha: str, model: pathlib.Path, output: pathlib.Path,
             runner: pathlib.Path, dump: pathlib.Path, replay: pathlib.Path) -> dict:
    pool = verify_pool(pool_root, pool_sha)
    output.mkdir(parents=True, exist_ok=False)
    pack = output/'keywords.kwk'
    subprocess.run([sys.executable,str(ROOT/'tools/compile_keywords.py'),'--tokens',str(pool_root/'tokens.example.txt'),
                    '--keywords',str(pool_root/'zh_cn_example.tsv'),'--out-pack',str(pack)],
                    stdout=subprocess.DEVNULL, check=True, timeout=30)
    model_sha = sha(model)
    trace_root=output/'traces';trace_root.mkdir()
    margin=float(json.loads((ROOT/'configs/parameter-contract.json').read_text())
                 ['algorithm_constants']['KWS_ROOT_START_LOGIT_MARGIN']['default'])
    result = {'policy': POLICY, 'development_only': True, 'release_authority': False,
        'pool_sha256':pool_sha,'model_sha256':model_sha,'pack_sha256':sha(pack),
        'runner_sha256':sha(runner),'dump_sha256':sha(dump),'replay_sha256':sha(replay),
        'per_clip':{},'continuous':{},'root_in_activity':{},'counterfactual':{},
        'limitations':['Clean synthetic development data; no product accuracy or long-FAR authority.',
                      'Activity envelopes are not phoneme timestamps; no validation selection or threshold tuning.']}
    # WAV payloads are unchanged; prefix offsets and all later stream offsets are hop aligned.
    for split in ('train','calibration','test'):
        rows = [r for r in pool['rows'] if r['split']==split]
        raw_rows = [pcm(resolve(pool_root,r['path'])) for r in rows]
        result['per_clip'][split] = {}
        counter_refs, counter_dets, authentic_dets, pairs = [],[],[],[]
        roots = {1:{'expected':0,'admissible_in_activity':0},2:{'expected':0,'admissible_in_activity':0}}
        for prefix in PREFIX_SAMPLES:
            refs, dets = [],[]
            for i,(row,raw) in enumerate(zip(rows,raw_rows)):
                audio = output/'scratch.wav';wav(audio,b'\0\0'*prefix+raw)
                name=f'{split}-{i}';events=shifted(row['expected'],prefix)
                ref={'recording':name,'duration_s':(len(raw)//2+prefix)/16000,'expected':events}
                found=collect(runner,model,pack,audio,name);refs.append(ref);dets.extend(found)
                if row['expected'] and prefix in (3200,16000):
                    trace=trace_root/f'{split}-{i}-prefix-{prefix}.kwtr'
                    subprocess.run([str(dump),str(model),str(audio),str(trace)],check=True,stdout=subprocess.DEVNULL,timeout=30)
                    if prefix==16000:
                        logits=read_trace_logits(trace); e=events[0];kid=e['keyword_id'];token=1 if kid==1 else 3
                        # Mirrors root start admissibility; report only, activity is not phoneme ground truth.
                        ok=False
                        for t,lp in enumerate(logits):
                            end=(t*320+400)/16000
                            if not e['start_s'] <= end <= e['end_s']:continue
                            top=max(range(len(lp)),key=lp.__getitem__)
                            if top==token or (top in (0,1,3) and lp[top]-lp[token]<=margin): ok=True
                        roots[kid]['expected']+=1;roots[kid]['admissible_in_activity']+=int(ok)
                    else:
                        edited=trace_root/f'{split}-{i}-counterfactual.kwtr'
                        n=blank_prefix(trace,edited,prefix)
                        changed=collect(replay,model,pack,edited,name)
                        counter_refs.append(ref);authentic_dets.extend(found);counter_dets.extend(changed)
                        pairs.append({'reference':ref,'source_wav_sha256':row['wav_sha256'],
                            'authentic_trace_sha256':sha(trace),'edited_trace_sha256':sha(edited),
                            'edited_frames':n,'authentic_detections':found,'edited_detections':changed,
                            'lost_matches':max(0,measure([ref],found)['matched']-measure([ref],changed)['matched'])})
            result['per_clip'][split][str(prefix/16000)] = measure(refs,dets)
            write(output/f'{split}-prefix-{prefix}.json',{'references':refs,'detections':dets})
        result['root_in_activity'][split] = roots
        result['counterfactual'][split] = {'counterfactual':True,'release_authority':False,
            'authentic':measure(counter_refs,authentic_dets),'blanked_startup':measure(counter_refs,counter_dets),
            'lost_match_count':sum(p['lost_matches'] for p in pairs),'pairs':pairs}
        # Single engine invocation, two-second lead-in and gaps, includes original nonwake rows.
        stream=bytearray(b'\0\0'*32000);events=[]
        for row,raw in zip(rows,raw_rows):
            offset=len(stream)//2;events.extend(shifted(row['expected'],offset));stream.extend(raw)
            padding=(-(len(stream)//2))%320
            stream.extend(b'\0\0'*(padding+32000))
        path=output/f'{split}-continuous.wav';wav(path,bytes(stream))
        ref={'recording':split+'-continuous','duration_s':len(stream)/32000,'expected':events}
        found=collect(runner,model,pack,path,ref['recording'])
        result['continuous'][split] = measure([ref],found)
        write(output/f'{split}-continuous.json',{'references':[ref],'detections':found})
        # Regenerable from content-bound inputs; do not retain duplicated large waveforms.
        path.unlink()
    negative_refs,negative_dets=[],[]
    for seconds in (1,5,10):
        path=output/'scratch.wav';wav(path,b'\0\0'*(seconds*16000));name=f'silence-{seconds}'
        negative_refs.append({'recording':name,'duration_s':seconds,'expected':[]})
        negative_dets.extend(collect(runner,model,pack,path,name))
    result['silence_only'] = measure(negative_refs,negative_dets)
    train_rows=list(result['per_clip']['train'].values())+[result['continuous']['train']]
    result['train_fit_gate'] = all(r['matched']==r['expected'] and r['false_accepts']==0 for r in train_rows)
    result['startup_independence_gate'] = all(
        v['lost_match_count']==0 and all(v['authentic']['per_keyword'].get(str(k),{}).get('matched',0)>0 for k in (1,2))
        for v in result['counterfactual'].values())
    result['basic_train_learnability_pass'] = (result['train_fit_gate']
        and result['counterfactual']['train']['lost_match_count']==0
        and result['silence_only']['false_accepts']==0
        and all(v['admissible_in_activity']==v['expected'] for v in result['root_in_activity']['train'].values()))
    checkpoint=model.with_suffix('.pt')
    if checkpoint.is_file():
        result['acoustic']=acoustic_diagnostics(pool,pool_root,checkpoint,dump,output)
        result['checkpoint_sha256']=sha(checkpoint)
    verify_pool(pool_root,pool_sha)
    if sha(model)!=model_sha:raise ValueError('model changed during evaluation')
    write(output/'summary.json',result)
    print(json.dumps({'model':model_sha,'train_fit_gate':result['train_fit_gate'],
        'per_clip_matches':{s:{p:[v['matched'],v['expected'],v['false_accepts']] for p,v in vs.items()} for s,vs in result['per_clip'].items()},
        'continuous':{s:[v['matched'],v['expected'],v['false_accepts']] for s,v in result['continuous'].items()}}),flush=True)
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pool','model','output','runner','dump','replay'):p.add_argument('--'+key,type=pathlib.Path,required=True)
    p.add_argument('--pool-sha',required=True);a=p.parse_args()
    evaluate(a.pool.resolve(),a.pool_sha,a.model.resolve(),a.output.resolve(),a.runner.resolve(),a.dump.resolve(),a.replay.resolve())

if __name__=='__main__':main()
