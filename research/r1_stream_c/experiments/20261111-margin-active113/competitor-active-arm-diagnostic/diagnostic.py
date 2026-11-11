#!/usr/bin/env python3
"""One separately authorized ARM baseline capture, never a candidate rerun."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import subprocess
import time

HERE=Path(__file__).resolve().parent


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    need(path.is_file() and not path.is_symlink(),'regular input required: '+path.name)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):return json.loads(path.read_text())


def save(path,data):
    with path.open('x') as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def declared(expected):
    path=HERE/'PLAN.json';need(sha(path)==expected,'diagnostic plan changed');p=load(path)
    need(p['max_pcm_calls']==1 and p['process_timeout_seconds']==30 and p['retry'] is False,'diagnostic budget changed')
    for name,digest in p['source_sha256'].items():need(sha(HERE/name)==digest,'diagnostic source drift')
    for name,value in p['bindings'].items():need(sha(Path(value['path']))==value['sha256'],'retained binding changed: '+name)
    return p


def build(expected):
    p=declared(expected);out=HERE/'build';out.mkdir(exist_ok=False)
    env=dict(os.environ);env['PATH']=p['toolchain_bin']+os.pathsep+env['PATH'];env['LD_LIBRARY_PATH']=p['toolchain_host_lib']
    argv=[p['compiler']['path'],*p['compile_flags'],'-I'+p['sdk_include'],'-I'+p['sdk_core'],
          str(HERE/'extract_teacher.c'),p['bindings']['baseline_arm_library']['path'],
          '-Wl,--wrap=r1dfsmn_step','-lm','-o',str(out/'baseline_arm_extract')]
    need(sha(Path(p['compiler']['path']).resolve())==p['compiler']['sha256'],'compiler changed')
    receipt={'schema':'active113-single-arm-diagnostic-build-v1','status':'BUILDING','plan_sha256':expected,
             'command':argv,'pcm_calls':0,'model_calls':0}
    try:
        with (out/'stdout.log').open('xb') as so,(out/'stderr.log').open('xb') as se:
            child=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,timeout=90,check=False)
        need(child.returncode==0 and (out/'stderr.log').stat().st_size==0,'diagnostic compile failed')
        binary=out/'baseline_arm_extract';data=binary.read_bytes()
        need(data[:6]==b'\x7fELF\x01\x01' and struct.unpack_from('<H',data,18)[0]==40 and
             struct.unpack_from('<I',data,36)[0]&0x400,'ARM32 little-endian hard-float ELF required')
        receipt['binary']={'path':str(binary),'bytes':len(data),'sha256':sha(binary)}
        receipt['status']='BUILT_SINGLE_DIAGNOSTIC_NO_PCM';declared(expected)
    except BaseException as e:
        receipt.update(status='FAILED_NO_PCM',failure=type(e).__name__+': '+str(e));raise
    finally:save(out/'BUILD_RECEIPT.json',receipt)
    print(json.dumps({'status':receipt['status'],'build_receipt_sha256':sha(out/'BUILD_RECEIPT.json'),'binary':receipt['binary']}))


def score_rows(path,kind):
    data=path.read_bytes();need(len(data)>64 and (len(data)-64)%1200==0,'numeric capture length')
    h=struct.unpack('<8s6I5IIQ',data[:64]);ids=(0,1027,1401,1462,2093)
    expect=(b'R1HD140\0',1,64,1200,140,5,2601,*ids,1,0) if kind=='teacher' else (b'R1A113H\0',1,64,1200,140,6,2601,*ids,113,0)
    need(h==expect,'numeric capture schema')
    rows=[data[i:i+1200] for i in range(64,len(data),1200)]
    for i,row in enumerate(rows):
        need(struct.unpack_from('<Q',row)[0]==i,'frame sequence')
        need(all(math.isfinite(x[0]) for x in struct.iter_unpack('<d',row[32:])), 'nonfinite saved numeric data')
    return rows


def ordered(bits):return ((1<<64)-1-bits) if bits>>63 else bits+(1<<63)


def compare(a,b,offset,count):
    need(len(a)==len(b),'comparison frame denominator')
    differences=0;frames=0;maximum=0.0;maximum_ulp=0;total=0.0
    for aa,bb in zip(a,b):
        need(aa[:24]==bb[:24],'frame clocks differ')
        changed=False
        for j in range(count):
            x=aa[offset+j*8:offset+(j+1)*8];y=bb[offset+j*8:offset+(j+1)*8]
            if x!=y:
                changed=True;differences+=1
                error=abs(struct.unpack('<d',x)[0]-struct.unpack('<d',y)[0]);maximum=max(maximum,error);total+=error
                maximum_ulp=max(maximum_ulp,abs(ordered(int.from_bytes(x,'little'))-ordered(int.from_bytes(y,'little'))))
        frames+=changed
    return {'frames':len(a),'values':len(a)*count,'different_values':differences,'different_frames':frames,
            'byte_exact':differences==0,'max_absolute_error':maximum,'mean_absolute_error':total/(len(a)*count),
            'max_ulp_distance':maximum_ulp}


def run(expected,build_sha):
    p=declared(expected);out=HERE/'execution';need(not out.exists(),'single diagnostic already claimed; no retry')
    receipt=HERE/'build/BUILD_RECEIPT.json';need(sha(receipt)==build_sha,'diagnostic build receipt changed');b=load(receipt)
    need(b['status']=='BUILT_SINGLE_DIAGNOSTIC_NO_PCM' and b['plan_sha256']==expected,'diagnostic build binding')
    need(sha(Path(b['binary']['path']))==b['binary']['sha256'],'diagnostic executable changed')
    need(sha(Path(p['pcm']['path']))==p['pcm']['sha256'] and Path(p['pcm']['path']).stat().st_size==p['pcm']['bytes'],'fixed diagnostic PCM identity')
    need(sha(Path(p['qemu']['path']))==p['qemu']['sha256'],'QEMU identity changed')
    os.umask(0o077);out.mkdir()
    hidden=out/'baseline_arm.r1h';trace=out/'baseline_arm.txt'
    argv=[p['qemu']['path'],'-L',p['sysroot'],b['binary']['path'],p['bindings']['original_model']['path'],
          p['bindings']['frontend']['path'],p['pcm']['path'],str(hidden),str(trace)]
    save(out/'CLAIM.json',{'schema':'active113-single-arm-diagnostic-claim-v1','attempt':1,'max_pcm_calls':1,
        'plan_sha256':expected,'build_receipt_sha256':build_sha,'argv':argv,'pcm_sha256':p['pcm']['sha256'],
        'kind':'new same-architecture baseline numeric reference; no candidate rerun','resource_measurement':False})
    terminal={'schema':'active113-single-arm-diagnostic-terminal-v1','status':'CLAIMED','pcm_calls_claimed':1,
              'pcm_calls_completed':0,'plan_sha256':expected,'build_receipt_sha256':build_sha,
              'candidate_reruns':0,'resource_calls':0,'strict_cross_arch_gate':'FAILED_UNCHANGED'}
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576));resource.setrlimit(resource.RLIMIT_CPU,(31,31))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    start=time.monotonic()
    try:
        env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
        with (out/'receipt.json').open('xb') as so,(out/'stderr.log').open('xb') as se:
            child=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,
                preexec_fn=limits,timeout=30,check=False)
        terminal['returncode']=child.returncode
        need(child.returncode==0 and (out/'stderr.log').stat().st_size==0,'baseline diagnostic capture failed')
        terminal['pcm_calls_completed']=1
        raw=load(out/'receipt.json');need(raw['frames']==raw['teacher_steps']==p['frames'] and raw['pcm_bytes']==p['pcm']['bytes'],'baseline diagnostic frame/PCM count')
        baseline=score_rows(hidden,'teacher')
        candidate_arm=score_rows(Path(p['bindings']['candidate_arm_scores']['path']),'candidate')
        host_teacher=score_rows(Path(p['bindings']['host_teacher_hidden']['path']),'teacher')
        candidate_host=score_rows(Path(p['bindings']['candidate_host_scores']['path']),'candidate')
        need(len(baseline)==p['frames'],'diagnostic frame denominator')
        comparisons={
            'baseline_arm_vs_candidate_arm':{'hidden140':compare(baseline,candidate_arm,32,140),'precise5':compare(baseline,candidate_arm,1152,5)},
            'baseline_arm_vs_host_teacher':{'hidden140':compare(baseline,host_teacher,32,140),'precise5':compare(baseline,host_teacher,1152,5)},
            'candidate_host_vs_host_teacher':{'hidden140':compare(candidate_host,host_teacher,32,140),'precise5':compare(candidate_host,host_teacher,1152,5)}}
        exact=all(x['byte_exact'] for x in comparisons['baseline_arm_vs_candidate_arm'].values())
        cross=any(not x['byte_exact'] for x in comparisons['baseline_arm_vs_host_teacher'].values())
        text=trace.read_bytes();need(text==Path(p['bindings']['candidate_arm_trace']['path']).read_bytes()==
              Path(p['bindings']['saved_original_trace']['path']).read_bytes(),'diagnostic F/E differs')
        original_ids_equal=all(a[24:28]==c[24:28] for a,c in zip(baseline,candidate_arm))
        need(original_ids_equal,'diagnostic original argmax changed')
        terminal.update(status='SAME_ARCH_PARITY_WITH_CROSS_ARCH_DIFFERENCE' if exact and cross else 'DIAGNOSTIC_COMPLETED_OTHER_OUTCOME',
            comparisons=comparisons,same_arch_hidden_and_precise_byte_exact=exact,
            baseline_cross_arch_numeric_difference_observed=cross,fe_trace_byte_exact=True,
            global_winner_ids_byte_exact=original_ids_equal,
            evidence_scope='one newly captured baseline ARM recording only; no conclusion for unexecuted remaining ARM recordings',
            baseline_hidden_sha256=sha(hidden),baseline_trace_sha256=sha(trace),receipt_sha256=sha(out/'receipt.json'))
        declared(expected)
    except BaseException as e:
        terminal.update(status='STOPPED_SINGLE_DIAGNOSTIC_NO_RETRY',failure=type(e).__name__+': '+str(e));raise
    finally:
        terminal['wall_seconds']=time.monotonic()-start;save(out/'TERMINAL.json',terminal)
    result={'schema':'active113-single-same-arm-diagnostic-result-v1',
        'status':'PASS_SAME_ARM_BASELINE_HIDDEN_PRECISE' if terminal['same_arch_hidden_and_precise_byte_exact'] else 'FAIL_SAME_ARM_BASELINE_HIDDEN_PRECISE',
        'actual_pcm_calls':1,'frames':p['frames'],'all_hidden140_bits_equal':comparisons['baseline_arm_vs_candidate_arm']['hidden140']['byte_exact'],
        'all_precise5_bits_equal':comparisons['baseline_arm_vs_candidate_arm']['precise5']['byte_exact'],
        'fe_byte_exact':True,'global_winner_ids_byte_exact':True,
        'baseline_c_source_sha256':p['source_sha256']['extract_teacher.c'],
        'baseline_model_sha256':p['bindings']['original_model']['sha256'],
        'baseline_arm_library_sha256':p['bindings']['baseline_arm_library']['sha256'],
        'baseline_arm_extractor_sha256':b['binary']['sha256'],
        'retained_candidate_score_sha256':p['bindings']['candidate_arm_scores']['sha256'],
        'retained_candidate_trace_sha256':p['bindings']['candidate_arm_trace']['sha256'],
        'host_teacher_cache_sha256':p['bindings']['host_teacher_hidden']['sha256'],
        'plan_sha256':expected,'build_receipt_sha256':build_sha,'claim_sha256':sha(out/'CLAIM.json'),
        'terminal_sha256':sha(out/'TERMINAL.json'),'comparisons':comparisons,
        'candidate_reruns':0,'resource_calls':0,'training':0,'strict_cross_arch_gate':'FAILED_UNCHANGED',
        'model_promotion_authorized':False,'full_other_equivalence':'NOT_ASSERTED',
        'scope':'single newly captured same-architecture baseline reference; remaining ARM inputs unexecuted by this diagnostic'}
    save(out/'RESULT.json',result);print(json.dumps(result,sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['build','run']);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--build-sha256');a=p.parse_args()
    if a.mode=='build':build(a.plan_sha256)
    else:
        need(a.build_sha256 is not None,'run requires exact build receipt hash');run(a.plan_sha256,a.build_sha256)


if __name__=='__main__':main()
