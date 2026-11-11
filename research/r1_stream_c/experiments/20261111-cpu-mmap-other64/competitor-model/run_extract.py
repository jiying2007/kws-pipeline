#!/usr/bin/env python3
"""One bounded 134-input teacher pass, only after parent inspection/authorization.

Uses the existing model once per original PCM. No retries, regeneration,
training or full-logit persistence. Every F/E must match retained par4 bytes.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
IDS=(0,1027,1401,1462,2093)
HEADER=struct.pack('<8s6I5IIQ',b'R1HD140\0',1,64,1200,140,5,2601,*IDS,1,0)
ROW=struct.Struct('<3Q2I146d')

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
def validate_hidden(path,record,reference):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=64+512*1200,'hidden output bounds')
    data=path.read_bytes()
    need(data[:64]==HEADER and len(data)>64 and (len(data)-64)%1200==0,'hidden header/length')
    frame_rows=[];samples=record['samples'];count=0
    for index,values in enumerate(ROW.iter_unpack(data[64:])):
        frame,source,delivery,best,other_id,*outputs=values
        need(frame==index and source==720+480*index and
             delivery==min(((source+319)//320)*320,samples),'hidden clock grid')
        need(all(math.isfinite(x) for x in outputs),'nonfinite hidden/teacher outputs')
        need(best<2601 and other_id<2601 and other_id not in IDS,'invalid original token IDs')
        exact=outputs[140:145];other=outputs[145]
        winner=min(range(5),key=lambda k:(-exact[k],IDS[k]))
        best_exact=IDS[winner];value=exact[winner]
        original=(best_exact if value>other else other_id if value<other else min(best_exact,other_id))
        need(best==original,'original teacher winner/tie reconstruction mismatch')
        frame_rows.append(f'F,{frame},{source},{delivery},{best}\n'.encode())
        count+=1
    need(count==max(0,(samples-720)//480+1),'hidden frame count')
    ref_frames=b''.join(line for line in reference.splitlines(keepends=True) if line.startswith(b'F,'))
    need(b''.join(frame_rows)==ref_frames,'hidden rows differ from original F trace')
    return count

def main():
    os.umask(0o077)
    predeclared=HERE/'PLAN.json';plan=load(predeclared)
    need(plan['max_teacher_clip_calls']==134 and plan['per_process_seconds']==30 and
         plan['total_seconds']==900 and plan['retry'] is False,'changed extraction budget')
    for name,digest in plan['source_sha256'].items():
        need(sha(HERE/name)==digest,'extraction source changed: '+name)
    runtime=plan['runtime']
    need(sys.version.split()[0]==runtime['python_version'] and
         sha(Path(sys.executable).resolve())==runtime['python_sha256'],'Python runtime changed')
    compiled=load(HERE/'build/BUILD_RECEIPT.json')
    need(compiled['status']=='BUILT_NO_INFERENCE' and compiled['predeclared_plan_sha256']==sha(predeclared),
         'unmatched extraction build')
    binary=Path(compiled['binary']['path']);need(sha(binary)==compiled['binary']['sha256'],'extractor binary changed')
    model=Path(plan['model']['path']);frontend=Path(plan['frontend']['path'])
    inputs=Path(plan['input_root']);saved=Path(plan['saved_root'])
    def inputs_unchanged():
        need(sha(model)==plan['model']['sha256'] and sha(frontend)==plan['frontend']['sha256'],
             'model/frontend identity changed')
        need(sha(inputs/'INPUT_MANIFEST.json')==plan['input_manifest_sha256'] and
             sha(saved/'FILE_MANIFEST.json')==plan['saved_trace_manifest_sha256'],
             'input/baseline manifest changed')
        need(sha(inputs/'original-observations.jsonl')==plan['observation_labels_sha256'],
             'original source labels changed')
        records=load(inputs/'INPUT_MANIFEST.json')['records']
        need(len(records)==134 and [r['index'] for r in records]==list(range(134)),'input denominator changed')
        for record in records:
            path=inputs/'pcm'/f"{record['index']:03d}.pcm"
            need(path.stat().st_size==record['pcm_bytes'] and sha(path)==record['pcm_sha256'],'PCM changed')
        return records
    records=inputs_unchanged();index=load(saved/'FILE_MANIFEST.json');references={}
    for record in records:
        i=record['index'];name=f'private-traces/{i:03d}-arm_par4.txt';p=saved/name
        need(p.stat().st_size==index[name]['bytes'] and sha(p)==index[name]['sha256'],'saved trace changed')
        references[i]=p.read_bytes()
    out=HERE/'extraction';out.mkdir(exist_ok=False)
    for name in ('hidden','traces','receipts','claims','logs'):(out/name).mkdir()
    actual=dict(plan,predeclared_plan_sha256=sha(predeclared),binary=compiled['binary'],
                build_receipt_sha256=sha(HERE/'build/BUILD_RECEIPT.json'),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(out/'PLAN.json',actual)
    result={'schema':'r1-hidden140-teacher-extraction-v1','status':'RUNNING',
            'source_head':plan['source_head'],'model_sha256':plan['model']['sha256'],
            'frontend_sha256':plan['frontend']['sha256'],
            'input_manifest_sha256':plan['input_manifest_sha256'],
            'saved_trace_manifest_sha256':plan['saved_trace_manifest_sha256'],
            'plan_sha256':sha(out/'PLAN.json'),'claimed':[],'records':[],
            'unknown_indices_excluded_from_fit':[96,97,98,99],
            'full_logits_persisted':False,'training':0,'model_promotion_authorized':False,
            'physical_ssc305':'NOT_RUN','independent_far_frr':'NOT_RUN'}
    start=time.monotonic();deadline=start+900;affinity=min(os.sched_getaffinity(0))
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))
        resource.setrlimit(resource.RLIMIT_CPU,(31,31));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        os.sched_setaffinity(0,{affinity})
    try:
        for record in records:
            i=record['index'];remaining=deadline-time.monotonic();need(remaining>0,'total deadline exceeded')
            hidden=out/'hidden'/f'{i:03d}.r1h';trace=out/'traces'/f'{i:03d}.txt'
            argv=[str(x) for x in (binary,model,frontend,inputs/'pcm'/f'{i:03d}.pcm',hidden,trace)]
            save(out/'claims'/f'{i:03d}.json',{'index':i,'attempt':1,'argv':argv,'pcm_sha256':record['pcm_sha256']})
            result['claimed'].append(i)
            env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('R1_TRACE_PATH',None)
            with (out/'receipts'/f'{i:03d}.json').open('xb') as so,(out/'logs'/f'{i:03d}.stderr').open('xb') as se:
                process=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=so,stderr=se,env=env,
                                       preexec_fn=limits,timeout=min(30,remaining),check=False)
            need(process.returncode==0,'teacher extraction process failed; no retry')
            need((out/'logs'/f'{i:03d}.stderr').stat().st_size==0,'nonempty teacher extraction stderr')
            need(trace.read_bytes()==references[i],'teacher F/E differs from retained baseline')
            frames=validate_hidden(hidden,record,references[i]);receipt=load(out/'receipts'/f'{i:03d}.json')
            need(receipt['schema']=='r1-hidden140-teacher-clip-v1' and
                 receipt['pcm_bytes']==record['pcm_bytes'] and receipt['frames']==frames and
                 receipt['teacher_steps']==frames and receipt['full_logits_persisted'] is False,
                 'teacher receipt mismatch')
            event_lines=[line.split(b',') for line in references[i].splitlines() if line.startswith(b'E,')]
            need(receipt['events']==len(event_lines) and
                 receipt['keyword1_events']==sum(x[2]==b'1' for x in event_lines) and
                 receipt['keyword2_events']==sum(x[2]==b'2' for x in event_lines),'teacher event count mismatch')
            result['records'].append({'index':i,'path':f'hidden/{i:03d}.r1h',
                'bytes':hidden.stat().st_size,'sha256':sha(hidden),'frames':frames,
                'trace_sha256':sha(trace),'pcm_sha256':record['pcm_sha256'],
                'original_split':record['split'],'fit_excluded_unknown':i in (96,97,98,99),
                'fe_trace_equal':True,'returncode':process.returncode,
                'receipt_sha256':sha(out/'receipts'/f'{i:03d}.json')})
            if (i+1)%16==0:print(json.dumps({'teacher_clips_completed':i+1,'max':134}),flush=True)
        need(len(result['records'])==134 and sum(r['frames'] for r in result['records'])==12534,
             'teacher output denominator mismatch')
        result['status']='PASS_FIXED_TEACHER_EXTRACTION'
    except BaseException as exc:
        result.update(status='STOPPED_NO_RETRY',failure=type(exc).__name__+': '+str(exc));raise
    finally:
        try:
            inputs_unchanged();result['post_input_model_frontend_hashes_unchanged']=True
        except BaseException as exc:
            result.update(status='STOPPED_NO_RETRY',post_input_model_frontend_hashes_unchanged=False,
                          identity_failure=type(exc).__name__+': '+str(exc))
        result['extraction_count']=len(result['records'])
        result['actual_frames']=sum(r['frames'] for r in result['records'])
        result['wall_seconds']=time.monotonic()-start
        save(out/'RESULT.json',result)
    need(result['status']=='PASS_FIXED_TEACHER_EXTRACTION','failed final identity check')
    print(json.dumps({'status':result['status'],'extraction_count':result['extraction_count'],
                      'actual_frames':result['actual_frames'],'wall_seconds':result['wall_seconds']}),flush=True)
if __name__=='__main__':main()
