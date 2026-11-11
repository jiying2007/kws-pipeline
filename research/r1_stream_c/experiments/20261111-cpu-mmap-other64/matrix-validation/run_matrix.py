#!/usr/bin/env python3
"""One-shot mmap matrix against saved traces, optionally followed by host-only timing.

No baseline reruns, training, ASR, TTS, retries or threshold/model selection.
Capture and QEMU timing are retained only as diagnostics, never performance evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time

SOURCE_HEAD='13661a38064247d38bca88287e48db85bf2ccd79'
OLD=Path('/workspace/scratch/1e1fe64498b2')
PRIOR_PLAN=OLD/'r1-candidate/execution/PLAN.json'
INPUTS=OLD/'r1-local/evidence-original/KWS_FSMN_R1_EVIDENCE_20261110/inputs'
SAVED=OLD/'r1-local/evidence-qemu'
INPUT_SHA='63d2f75e5e93e454bdf23624277fa728bba0cef6eee9cce311f037612198d6ab'
SAVED_SHA='8a3d0fbd2822c383b40db2159785419a3b85ab926f06a14c3b3da2a056dd1e44'
MODEL_SHA='63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4'
FRONTEND_SHA='49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49'
KERNELS=('par4','block2','block4')
ORDERS=(KERNELS,('block2','block4','par4'),('block4','par4','block2'))
LIMIT=65536

def require(ok,message):
    if not ok: raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())

def load(path):
    def bad(value): raise ValueError('nonfinite JSON: '+value)
    return json.loads(path.read_text(),parse_constant=bad)

def counts(data):
    require(len(data)<=LIMIT and data.endswith(b'\n'),'invalid trace size/termination')
    frames=events=k1=k2=0
    for line in data.decode('ascii').splitlines():
        parts=line.split(',')
        require((parts[0]=='F' and len(parts)==5) or
                (parts[0]=='E' and len(parts)==7),'invalid F/E trace row')
        values=[int(v) for v in parts[1:]]
        require(all(v>=0 for v in values),'negative trace value')
        if parts[0]=='F':
            require(values[0]==frames and values[1]<=values[2],'invalid frame clock')
            frames+=1
        else:
            require(values[1] in (1,2) and values[2]<frames and values[4]<=values[5],
                    'invalid event identity/clock')
            events+=1;k1+=values[1]==1;k2+=values[1]==2
    return {'sdk_model_frames':frames,'keyword_events':events,
            'keyword1_events':k1,'keyword2_events':k2}

def validate_receipt(row,pcm_bytes,expected,storage='read_only_mmap'):
    require(row.get('schema')=='r1-strong-fsmn-offline-board-resource-v1','wrong receipt schema')
    require(storage in ('read_only_mmap','sdk_owned_copy'),'unsupported expected storage')
    require(row.get('model_storage')==storage and
            row.get('model_caller_heap_bytes')==(0 if storage=='read_only_mmap' else 1960140),
            'requested storage mode was not actually reported')
    require(row.get('pcm_bytes')==pcm_bytes and row.get('feed_chunk_bytes')==640 and
            row.get('feed_calls')==(pcm_bytes+639)//640,'incomplete or changed chunk input')
    require(all(row.get(k)==v for k,v in expected.items()),'frame/event totals differ from saved trace')
    require(all(row.get(k) is False for k in ('device_qualified','live_afe','independent_far_frr')),
            'qualification boundary changed')
    for k in ('offline_feed_cpu_seconds','offline_feed_wall_seconds','cpu_seconds_per_audio_second'):
        require(type(row.get(k)) in (float,int) and math.isfinite(row[k]) and row[k]>0,
                'invalid timing receipt')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--host-perf',action='store_true')
    ap.add_argument('--host-memory',action='store_true')
    args=ap.parse_args();build=args.build.resolve();out=args.output.resolve()
    os.umask(0o077)
    require(not out.exists(),'fresh execution directory required; no restart/retry')
    prior=load(PRIOR_PLAN);compiled=load(build/'BUILD_RECEIPT.json')
    require(compiled['status']=='BUILT_NO_MODEL_EXECUTION' and compiled['source_head']==SOURCE_HEAD,
            'unexpected source/build identity')
    require(sha(INPUTS/'INPUT_MANIFEST.json')==INPUT_SHA,'input manifest drift')
    require(sha(SAVED/'FILE_MANIFEST.json')==SAVED_SHA,'saved trace manifest drift')
    input_manifest=load(INPUTS/'INPUT_MANIFEST.json');trace_manifest=load(SAVED/'FILE_MANIFEST.json')
    records=input_manifest['records']
    require(len(records)==134 and [r['index'] for r in records]==list(range(134)),
            'input denominator changed')
    model=Path(prior['identities']['model']['path']);frontend=Path(prior['identities']['frontend']['path'])
    qemu=Path(prior['identities']['qemu']['path']);sysroot=Path(prior['sysroot'])
    require(sha(model)==MODEL_SHA and sha(frontend)==FRONTEND_SHA,'model/frontend drift')
    require(sha(qemu)==prior['identities']['qemu']['sha256'],'QEMU drift')
    bins={}
    for kernel in KERNELS:
        for arch in ('host','arm'):
            for kind in ('plain','capture'):
                item=compiled['builds'][kernel+'_'+arch]['binaries'][kind]
                binary=Path(item['path']);require(sha(binary)==item['sha256'],'binary drift')
                bins[kernel,arch,kind]=binary
    saved={}
    expected={}
    for record in records:
        i=record['index'];pcm=INPUTS/'pcm'/f'{i:03d}.pcm'
        require(pcm.stat().st_size==record['pcm_bytes'] and sha(pcm)==record['pcm_sha256'],'PCM drift')
        name=f'private-traces/{i:03d}-arm_par4.txt';path=SAVED/name;descriptor=trace_manifest[name]
        require(path.stat().st_size==descriptor['bytes'] and sha(path)==descriptor['sha256'],
                'saved baseline trace drift')
        saved[i]=path.read_bytes();expected[i]=counts(saved[i])
    require(records[49]['pcm_bytes']==266240 and max(r['pcm_bytes'] for r in records)==266240,
            'fixed longest record changed')
    affinity=min(os.sched_getaffinity(0))
    out.mkdir(parents=True,exist_ok=False)
    for name in ('claims','traces','logs','receipts'): (out/name).mkdir()
    plan={'schema':'r1-mmap-fixed-matrix-plan-v1','source_head':SOURCE_HEAD,
          'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'build_receipt_sha256':sha(build/'BUILD_RECEIPT.json'),
          'run_script_sha256':sha(Path(__file__)),'input_manifest_sha256':INPUT_SHA,
          'saved_trace_manifest_sha256':SAVED_SHA,'model_sha256':MODEL_SHA,'frontend_sha256':FRONTEND_SHA,
          'model_path':str(model),'frontend_path':str(frontend),'qemu':prior['identities']['qemu'],
          'sysroot':str(sysroot),'kernels':list(KERNELS),'architectures':['host','arm_qemu'],
          'input_rows':134,'feed_chunk_bytes':640,'max_capture_invocations':804,
          'optional_host_plain_invocations':27 if args.host_perf else 0,
          'optional_host_memory_invocations':12 if args.host_memory else 0,
          'max_total_model_invocations':804+(27 if args.host_perf else 0)+(12 if args.host_memory else 0),
          'host_perf_pcm_index':49,'host_perf_pcm_bytes':266240,'host_perf_orders':ORDERS,
          'host_memory_pcm_index':49,'host_memory_kernel':'par4','host_memory_pairs':6 if args.host_memory else 0,
          'host_memory_orders':[['copy','mmap'],['mmap','copy']],
          'memory_scope':'own-process PSS/Private_Dirty after actual streaming; no peak-RSS or speedup inference',
          'process_timeout_seconds':45,'total_deadline_seconds':2400,'per_output_limit_bytes':LIMIT,
          'retry':False,'old_model_inference_invocations':0,'training':0,'tts':0,'asr':0,
          'trace_scope':'byte-exact F/E argmax, event and source/delivery clocks; full logits NOT_TESTED',
          'performance_scope':'uninstrumented host-only; capture and QEMU timings excluded',
          'physical_ssc305':'NOT_RUN','independent_far_frr':'NOT_RUN','model_promotion_authorized':False,
          'host':{'machine':platform.machine(),'platform':platform.platform(),'affinity_cpu':affinity}}
    save(out/'PLAN.json',plan)
    start=time.monotonic();deadline=start+2400
    terminal={'schema':'r1-mmap-fixed-matrix-result-v1','status':'RUNNING',
              'plan_sha256':sha(out/'PLAN.json'),'claimed':[],'completed':[],
              'completed_invocations':[],
              'physical_ssc305':'NOT_RUN','independent_far_frr':'NOT_RUN',
              'full_logit_parity':'NOT_TESTED','original_fp32_numeric_gate':'FAIL_UNCHANGED',
              'model_promotion_authorized':False,'host_perf':None,'host_memory':None}
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE,(LIMIT,LIMIT))
        resource.setrlimit(resource.RLIMIT_CPU,(46,46))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        os.sched_setaffinity(0,{affinity})
    def run(name,command,trace=None):
        remaining=deadline-time.monotonic()
        require(remaining>0,'total deadline exceeded; no further invocation')
        require(len(terminal['claimed'])<plan['max_total_model_invocations'],'invocation budget exceeded')
        command=[str(x) for x in command]
        save(out/'claims'/(name+'.json'),{'name':name,'argv':command,
             'trace':str(trace) if trace else None,'attempt':1,
             'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        terminal['claimed'].append(name)
        env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('R1_TRACE_PATH',None)
        if trace:env['R1_TRACE_PATH']=str(trace)
        stdout=out/'receipts'/(name+'.json');stderr=out/'logs'/(name+'.stderr')
        with stdout.open('xb') as so,stderr.open('xb') as se:
            result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=so,stderr=se,env=env,
                                  timeout=min(45,remaining),preexec_fn=limits,check=False)
        require(result.returncode==0,'nonzero process exit for '+name)
        require(stdout.stat().st_size<=LIMIT and stderr.stat().st_size<=LIMIT,'output bound exceeded')
        return load(stdout)
    def completed(name,index,row,scope):
        terminal['completed_invocations'].append({'name':name,'scope':scope,'index':index,
            'model_storage':row['model_storage'],'counts_match_saved':True,
            'expected_counts':expected[index],
            'receipt_sha256':sha(out/'receipts'/(name+'.json'))})
    try:
        for arch in ('host','arm'):
            for kernel in KERNELS:
                for record in records:
                    i=record['index'];name=f'parity-{arch}-{kernel}-{i:03d}'
                    trace=out/'traces'/(name+'.txt')
                    command=[bins[kernel,arch,'capture'],model,frontend,INPUTS/'pcm'/f'{i:03d}.pcm','--mmap']
                    if arch=='arm':command=[qemu,'-L',sysroot,*command]
                    row=run(name,command,trace)
                    require(trace.read_bytes()==saved[i],'F/E trace differs: '+name)
                    validate_receipt(row,record['pcm_bytes'],expected[i])
                    terminal['completed'].append({'name':name,'index':i,'arch':arch,'kernel':kernel,
                        'trace_sha256':sha(trace),'receipt_sha256':sha(out/'receipts'/(name+'.json')),
                        'fe_trace_equal':True,'counts_match_saved':True,**expected[i]})
                    completed(name,i,row,'captured_fe_parity_no_performance_claim')
                    if len(terminal['completed'])%25==0:
                        print(json.dumps({'completed':len(terminal['completed']),'expected':804}),flush=True)
        require(len(terminal['completed'])==804,'incomplete parity matrix')
        if args.host_perf:
            measurements={k:[] for k in KERNELS};p95={k:[] for k in KERNELS}
            for round_number in range(9):
                for kernel in ORDERS[round_number%3]:
                    name=f'host-perf-{round_number:02d}-{kernel}'
                    row=run(name,[bins[kernel,'host','plain'],model,frontend,INPUTS/'pcm'/'049.pcm','--mmap'])
                    validate_receipt(row,266240,expected[49])
                    completed(name,49,row,'uninstrumented_host_cpu_screen')
                    measurements[kernel].append(row['cpu_seconds_per_audio_second'])
                    p95[kernel].append(row['feed_p95_histogram_upper_us'])
            reference=measurements['par4']
            terminal['host_perf']={'scope':'host-only single-core offline mapped model',
                'instrumented_or_qemu_measurements_used':0,'rounds':9,'warmup_invocations':0,
                'pcm_index':49,'cpu_seconds_per_audio_second':measurements,'p95_upper_us':p95,
                'median_cpu_seconds_per_audio_second':{k:statistics.median(v) for k,v in measurements.items()},
                'paired_median_cpu_savings_pct':{k:statistics.median([100*(1-x/b) for x,b in
                      zip(measurements[k],reference)]) for k in ('block2','block4')},
                'physical_ssc305':'NOT_RUN','decision':'HOST_SCREEN_ONLY_NO_DEFAULT_CHANGE'}
        if args.host_memory:
            metrics=('proc_self_pss_kib','proc_self_private_dirty_kib')
            observed={mode:{metric:[] for metric in metrics} for mode in ('copy','mmap')}
            for pair in range(6):
                order=('copy','mmap') if pair%2==0 else ('mmap','copy')
                for mode in order:
                    name=f'host-memory-{pair:02d}-{mode}'
                    command=[bins['par4','host','plain'],model,frontend,INPUTS/'pcm'/'049.pcm']
                    if mode=='mmap':command+=['--mmap']
                    row=run(name,command)
                    storage='read_only_mmap' if mode=='mmap' else 'sdk_owned_copy'
                    validate_receipt(row,266240,expected[49],storage)
                    for metric in metrics:
                        require(metric in row and (row[metric] is None or
                                (type(row[metric]) is int and row[metric]>=0)),
                                'invalid own-process memory measurement')
                        observed[mode][metric].append(row[metric])
                    completed(name,49,row,'uninstrumented_host_memory_only_no_speed_claim')
            medians={mode:{metric:(statistics.median(values) if all(v is not None for v in values) else None)
                          for metric,values in by_metric.items()} for mode,by_metric in observed.items()}
            deltas={metric:([c-m for c,m in zip(observed['copy'][metric],observed['mmap'][metric])]
                     if all(v is not None for mode in observed for v in observed[mode][metric]) else None)
                    for metric in metrics}
            terminal['host_memory']={'scope':'host-only actual streaming copy/mmap memory comparison',
                'kernel':'par4','pcm_index':49,'pairs':6,'invocations':12,
                'mode_order_alternated':True,'counts_match_saved_for_every_call':True,
                'own_process_observations_kib':observed,'own_process_medians_kib':medians,
                'paired_copy_minus_mmap_kib':deltas,
                'median_copy_minus_mmap_kib':{metric:statistics.median(values) if values is not None else None
                                           for metric,values in deltas.items()},
                'peak_rss_used_for_savings':False,'memory_pair_timing_used_for_speedup':False,
                'physical_ssc305':'NOT_RUN','decision':'HOST_MEMORY_SCREEN_ONLY_NO_DEFAULT_CHANGE'}
        completed_names=[row['name'] for row in terminal['completed_invocations']]
        require(terminal['claimed']==completed_names and
                len(completed_names)==plan['max_total_model_invocations'],
                'incomplete claimed invocation denominator')
        terminal['all_claims_completed']=True
        terminal['status']='PASS_FIXED_FE_MATRIX_HOST_SCREEN_ONLY'
    except BaseException as exc:
        terminal['status']='STOPPED_NO_RETRY';terminal['failure']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        terminal['all_claims_completed']=(terminal['claimed']==
            [row['name'] for row in terminal['completed_invocations']] and
            len(terminal['claimed'])==plan['max_total_model_invocations'])
        try:
            after={'model_sha256':sha(model),'frontend_sha256':sha(frontend),
                   'input_manifest_sha256':sha(INPUTS/'INPUT_MANIFEST.json'),
                   'saved_trace_manifest_sha256':sha(SAVED/'FILE_MANIFEST.json'),
                   'verified_pcm_rows':0}
            for record in records:
                pcm=INPUTS/'pcm'/f"{record['index']:03d}.pcm"
                require(pcm.stat().st_size==record['pcm_bytes'] and sha(pcm)==record['pcm_sha256'],
                        'post-run PCM drift')
                after['verified_pcm_rows']+=1
            require(after['model_sha256']==MODEL_SHA and after['frontend_sha256']==FRONTEND_SHA and
                    after['input_manifest_sha256']==INPUT_SHA and after['saved_trace_manifest_sha256']==SAVED_SHA,
                    'post-run model/frontend/manifest drift')
            after['all_unchanged']=True;terminal['post_run_inputs']=after
        except BaseException as exc:
            terminal['post_run_inputs']={'all_unchanged':False,'failure':type(exc).__name__+': '+str(exc)}
            terminal['status']='STOPPED_NO_RETRY'
            terminal.setdefault('failure','post-run immutable input recheck failed')
        terminal['wall_seconds']=time.monotonic()-start
        terminal['capture_completed']=len(terminal['completed'])
        save(out/'TERMINAL.json',terminal)
    require(terminal['status']=='PASS_FIXED_FE_MATRIX_HOST_SCREEN_ONLY','execution or post-run identity failure')
    print(json.dumps({'status':terminal['status'],'capture_completed':len(terminal['completed']),
                      'claimed':len(terminal['claimed']),'wall_seconds':terminal['wall_seconds']}),flush=True)

if __name__=='__main__': main()
