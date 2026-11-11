#!/usr/bin/env python3
"""One bounded ACTIVE113 PCM validation, then paired host resource observations.

260 candidate capture calls + 18 plain CPU calls + 12 plain memory calls.
No baseline functional reruns, unknown calls, training, retries, or promotion.
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
import signal
import stat
import statistics
import struct
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SDKROOT=ROOT/'competitor-active-sdk-v1'
SDK=SDKROOT/'sdk-v2'
PACK=SDKROOT/'pack-v2'
SDK_BUILD=SDKROOT/'build-v2'
OLD=Path('/workspace/scratch/1e1fe64498b2')
INPUTS=OLD/'r1-local/evidence-original/KWS_FSMN_R1_EVIDENCE_20261110/inputs'
SAVED=OLD/'r1-local/evidence-qemu'
HIDDEN=ROOT/'competitor-model/extraction'
SCREEN=ROOT/'competitor-active-v1/execution'
PRECISE=(0,1027,1401,1462,2093)
SCORE_HEADER=struct.pack('<8s6I5IIQ',b'R1A113H\0',1,64,1200,140,6,2601,*PRECISE,113,0)
HIDDEN_HEADER=struct.pack('<8s6I5IIQ',b'R1HD140\0',1,64,1200,140,5,2601,*PRECISE,1,0)
LIMIT=1024*1024

def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):
    need(stat.S_ISREG(path.lstat().st_mode),'regular non-symlink evidence required: '+str(path))
    return path.read_bytes()
def sha(path):return hashlib.sha256(read(path)).hexdigest()
def identity(path):return {'bytes':path.stat().st_size,'sha256':sha(path)}
def load(path):
    def bad(value):raise ValueError('nonfinite JSON constant '+value)
    return json.loads(read(path),parse_constant=bad)
def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def counts(data):
    need(len(data)<=65536 and data.endswith(b'\n'),'invalid saved F/E trace')
    frames=events=k1=k2=0
    for line in data.decode('ascii').splitlines():
        parts=line.split(',');need((parts[0]=='F' and len(parts)==5) or (parts[0]=='E' and len(parts)==7),'bad trace row')
        v=[int(x) for x in parts[1:]];need(all(x>=0 for x in v),'negative trace integer')
        if parts[0]=='F':need(v[0]==frames and v[1]<=v[2] and v[3]<2601,'bad frame');frames+=1
        else:
            need(v[0]==0 and v[1] in (1,2) and v[2]<frames and v[3]<=v[2] and v[4]<=v[5],'bad event')
            events+=1;k1+=v[1]==1;k2+=v[1]==2
    return {'sdk_model_frames':frames,'keyword_events':events,'keyword1_events':k1,'keyword2_events':k2}
def timed_out(_signum,_frame):raise TimeoutError('fixed900secondtotaldeadline')
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--preflight-sha256');ap.add_argument('--declaration-sha256',required=True)
    ap.add_argument('--probe-build-sha256',required=True);args=ap.parse_args()
    need(args.execute!=args.preflight_only,'choose exactly one of --execute or --preflight-only')
    os.umask(0o077);start=time.monotonic();deadline=start+900
    signal.signal(signal.SIGALRM,timed_out);signal.setitimer(signal.ITIMER_REAL,900)
    out=HERE/'execution';need(not out.exists(),'fresh execution directory required; no overwrite/retry')
    need(sha(HERE/'DECLARED_PLAN.json')==args.declaration_sha256,'declaration changed')
    declared=load(HERE/'DECLARED_PLAN.json')
    need(declared['limits']=={'total_seconds':900,'child_seconds':30,'cpu_threads':1,'capture_calls':260,
        'host_cpu_calls':18,'host_memory_calls':12,'total_pcm_calls':290,'retry':False},'declared budget changed')
    watched={str(HERE/'DECLARED_PLAN.json'):args.declaration_sha256}
    for name,expected in declared['source_files'].items():
        need(identity(Path(name))==expected,'validation source changed');watched[name]=expected['sha256']
    for name,expected in declared['data_pins'].items():need(sha(Path(name))==expected,'fixed evidence changed');watched[name]=expected
    need(sha(HERE/'build/BUILD_RECEIPT.json')==args.probe_build_sha256,'probe build changed')
    compiled=load(HERE/'build/BUILD_RECEIPT.json');watched[str(HERE/'build/BUILD_RECEIPT.json')]=args.probe_build_sha256
    need(compiled['status']=='BUILT_PROBES_NO_PCM_OR_NEURAL_EXECUTION' and
         compiled['declaration_sha256']==args.declaration_sha256 and compiled['source_files']==declared['source_files'],'unbound probe build')
    need(all(compiled[k]==declared['sdk_bindings'][k] for k in
         ('sdk_plan_sha256','sdk_build_receipt_sha256','pack_receipt_sha256')),'final v2 SDK bindings changed')
    for name,key in [('PLAN.json','sdk_plan_sha256'),('build-v2/BUILD_RECEIPT.json','sdk_build_receipt_sha256'),('pack-v2/PACK.json','pack_receipt_sha256')]:
        p=SDKROOT/name;need(sha(p)==compiled[key],'SDK plan/build/pack changed');watched[str(p)]=compiled[key]
    for name,expected in compiled['generated_sdk_sources'].items():
        p=SDK/name;need(identity(p)==expected,'generated SDK source changed');watched[str(p)]=expected['sha256']
    for arch,expected in compiled['input_libraries'].items():
        p=SDK_BUILD/arch/'libactive113_sdk_static.a';need(identity(p)==expected,'SDK library changed');watched[str(p)]=expected['sha256']
    pack=load(PACK/'PACK.json');need(pack==compiled['pack'],'pack receipt byte/semantic drift')
    model=PACK/pack['model']['path'];frontend=PACK/pack['frontend']['path']
    for field,path in [('model',model),('frontend',frontend)]:
        need(identity(path)=={k:pack[field][k] for k in ('bytes','sha256')},'active model/frontend changed');watched[str(path)]=pack[field]['sha256']
    need(pack['model']['bytes']==1582624 and pack['selection_sha256']==declared['selection_sha256'] and
         pack['model']['sha256']==declared['sdk_bindings']['compact_model_sha256'] and
         pack['screen_result_sha256']==declared['cached_screen_sha256'],'compact model not bound to admitted candidate')
    baseline_binary=Path(declared['baseline_probe']['path']);baseline_model=Path(declared['baseline_model']['path'])
    for descriptor,path in [(declared['baseline_probe'],baseline_binary),(declared['baseline_model'],baseline_model)]:
        need(identity(path)=={k:descriptor[k] for k in ('bytes','sha256')},'baseline resource input changed');watched[str(path)]=descriptor['sha256']
    qemu=Path(declared['qemu']['path']);need(identity(qemu)=={k:declared['qemu'][k] for k in ('bytes','sha256')},'QEMU changed')
    watched[str(qemu)]=declared['qemu']['sha256'];sysroot=Path(declared['sysroot'])
    for name,expected in declared['sysroot_runtime_files'].items():need(identity(Path(name))==expected,'ARM runtime changed');watched[name]=expected['sha256']
    bins={}
    for arch in ('host','arm'):
        for kind in ('plain','capture'):
            d=compiled['builds'][arch]['binaries'][kind];p=Path(d['path'])
            need(identity(p)=={k:d[k] for k in ('bytes','sha256')},'active probe changed');bins[arch,kind]=p;watched[str(p)]=d['sha256']
    aggregate=load(SCREEN/'AGGREGATE.json');audit=load(ROOT/'competitor-active-v1/INDEPENDENT_AUDIT.json')
    need(aggregate['status']=='PASS_RETAINED_EVENT_SCREEN_NOT_PROMOTED' and aggregate['selection_sha256']==declared['selection_sha256'] and
         aggregate['training_max_other_bit_equal'] and aggregate['train_no_new_event'] and aggregate['development_complete_event_retention'] and
         aggregate['groups']['train']['original_token_id_changes']==aggregate['groups']['development']['original_token_id_changes']==0 and
         audit['status']=='PASS_SAVED_ACTIVE113_SCREEN_AUDIT' and audit['aggregate_sha256']==declared['cached_screen_sha256'],
         'cached candidate screen/audit does not authorize baseline F/E equivalence comparison')
    predictions=load(SCREEN/'PREDICTIONS.json');need(sha(SCREEN/'PREDICTIONS.json')==aggregate['predictions_sha256'] and
        predictions['all_predictions_persisted_before_scoring'] is True,'cached prediction manifest changed')
    predrows={r['index']:r for r in predictions['records']};hiddenrows={r['index']:r for r in load(HIDDEN/'RESULT.json')['records']}
    original=load(INPUTS/'INPUT_MANIFEST.json');trace_manifest=load(SAVED/'FILE_MANIFEST.json')
    labels={r['source_id']:r for r in (json.loads(line) for line in read(INPUTS/'original-observations.jsonl').splitlines())}
    records=[];saved={};expected={};hidden={};predicted={}
    for record in original['records']:
        i=record['index'];label=labels[record['source_id']]
        need(label['split']==record['split'] and label['pcm_sha256']==record['pcm_sha256'],'input label join changed')
        if label['ctc_weight']==0:continue
        need(label['ctc_weight'] in (0.5,1.0) and i in predrows,'unknown or unscored input')
        pcm=INPUTS/'pcm'/f'{i:03d}.pcm';need(pcm.stat().st_size==record['pcm_bytes'] and sha(pcm)==record['pcm_sha256'],'PCM changed')
        trace_name=f'private-traces/{i:03d}-arm_par4.txt';trace=SAVED/trace_name;need(identity(trace)==trace_manifest[trace_name],'baseline F/E changed')
        h=HIDDEN/hiddenrows[i]['path'];p=SCREEN/predrows[i]['path']
        need(h.name==f'{i:03d}.r1h' and p.name==f'{i:03d}.r1a' and sha(h)==hiddenrows[i]['sha256'] and
             identity(p)=={k:predrows[i][k] for k in ('bytes','sha256')},'cached numerical reference changed')
        saved[i]=read(trace);expected[i]=counts(saved[i]);hidden[i]=read(h);predicted[i]=read(p)
        frames=expected[i]['sdk_model_frames']
        need(hidden[i][:64]==HIDDEN_HEADER and len(hidden[i])==64+1200*frames and len(predicted[i])==64+40*frames and
             frames==hiddenrows[i]['frames']==predrows[i]['frames'],'reference shape changed')
        pheader=struct.pack('<8s6I5IIQ',b'R1ACTV1\0',1,64,40,140,5,2601,*PRECISE,113,frames)
        need(predicted[i][:64]==pheader,'cached prediction schema changed')
        for path,digest in [(pcm,record['pcm_sha256']),(trace,trace_manifest[trace_name]['sha256']),(h,hiddenrows[i]['sha256']),(p,predrows[i]['sha256'])]:watched[str(path)]=digest
        records.append({'index':i,'split':record['split'],'pcm_path':str(pcm),'pcm_bytes':record['pcm_bytes'],
            'pcm_sha256':record['pcm_sha256'],'expected_counts':expected[i],'hidden_sha256':hiddenrows[i]['sha256'],
            'cached_prediction_sha256':predrows[i]['sha256'],'baseline_trace_sha256':trace_manifest[trace_name]['sha256']})
    records.sort(key=lambda r:r['index'])
    need([r['index'] for r in records]==list(range(96))+list(range(100,134)),'known130denominator changed')
    need(sum(r['expected_counts']['sdk_model_frames'] for r in records)==12255 and
         sum(r['expected_counts']['keyword_events'] for r in records)==86,'known frame/event totals changed')
    longest=next(r for r in records if r['index']==49);need(longest['pcm_bytes']==266240 and
         max(r['pcm_bytes'] for r in records)==266240,'fixed performance input changed')
    affinity=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity});need(len(os.listdir('/proc/self/task'))==1,'single runner thread required')
    execution_plan={'schema':'active113-pcm-validation-execution-plan-v1','declaration_sha256':args.declaration_sha256,
        'probe_build_receipt_sha256':args.probe_build_sha256,'sdk_plan_sha256':compiled['sdk_plan_sha256'],
        'sdk_build_receipt_sha256':compiled['sdk_build_receipt_sha256'],'pack_receipt_sha256':compiled['pack_receipt_sha256'],
        'generated_sdk_sources':compiled['generated_sdk_sources'],'libraries':compiled['input_libraries'],'active_probes':compiled['builds'],
        'active_model':{'path':str(model),**identity(model)},'frontend':{'path':str(frontend),**identity(frontend)},
        'baseline_model':declared['baseline_model'],'baseline_probe':declared['baseline_probe'],'input_records':records,
        'selection_sha256':declared['selection_sha256'],'cached_screen_sha256':declared['cached_screen_sha256'],
        'qemu':declared['qemu'],'sysroot':str(sysroot),'sysroot_runtime_files':declared['sysroot_runtime_files'],
        'limits':declared['limits'],'resource_orders':declared['resource_orders'],'cpu_affinity':[affinity],
        'host_machine':platform.machine(),'host_platform':platform.platform(),'score_capture_format':declared['score_capture_format'],
        'unknown_pcm_calls':0,'baseline_functional_reruns':0,'training':0,'model_promotion_authorized':False}
    preflight_path=HERE/'PREFLIGHT.json'
    if args.preflight_only:
        save(preflight_path,{'schema':'active113-validation-preflight-v1','status':'READY_NO_PCM_EXECUTION',
            'execution_plan_preview':execution_plan,'validated_file_sha256':watched,
            'source_binaries_models_references_verified':True,'expected_capture_calls':260,'expected_cpu_calls':18,
            'expected_memory_calls':12,'total_pcm_budget':290,'pcm_calls':0,'neural_calls':0})
        signal.setitimer(signal.ITIMER_REAL,0)
        print(json.dumps({'status':'READY_NO_PCM_EXECUTION','preflight_sha256':sha(preflight_path),'pcm_calls':0},sort_keys=True));return
    need(args.preflight_sha256 is not None and sha(preflight_path)==args.preflight_sha256,'reviewed preflight receipt required')
    preflight=load(preflight_path)
    need(preflight['status']=='READY_NO_PCM_EXECUTION' and preflight['execution_plan_preview']==execution_plan and
         preflight['validated_file_sha256']==watched,'preflight identities no longer match')
    watched[str(preflight_path)]=args.preflight_sha256;execution_plan['preflight_sha256']=args.preflight_sha256
    out.mkdir(mode=0o700)
    for name in ('claims','receipts','traces','scores','logs','comparisons'):(out/name).mkdir()
    save(out/'PLAN.json',execution_plan)
    terminal={'schema':'active113-pcm-validation-terminal-v1','status':'RUNNING','stage':'FUNCTIONAL',
        'execution_plan_sha256':sha(out/'PLAN.json'),'claimed':[],'completed_invocations':[],'functional':[],
        'host_cpu':None,'host_memory':None,'unknown_pcm_calls':0,'baseline_functional_reruns':0,'training':0,
        'default_model_changed':False,'model_promotion_authorized':False,'physical_ssc305':'NOT_RUN',
        'independent_far_frr':'NOT_RUN','original_fp32_numeric_gate':'FAIL_UNCHANGED','full_2601_logits':'NOT_TESTED'}
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE,(LIMIT,LIMIT));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.sched_setaffinity(0,{affinity})
    def invoke(name,command,record,kind,capture=False,arch='host'):
        need(len(terminal['claimed'])<290 and time.monotonic()<deadline,'call/time budget exhausted')
        command=[str(x) for x in command];trace=out/'traces'/(name+'.txt') if capture else None
        score=out/'scores'/(name+'.r1s') if capture else None
        save(out/'claims'/(name+'.json'),{'name':name,'argv':command,'attempt':1,'scope':kind,'index':record['index'],
            'architecture':arch,'pcm_sha256':record['pcm_sha256'],'expected_counts':record['expected_counts'],
            'trace':str(trace) if trace else None,'scores':str(score) if score else None,
            'execution_plan_sha256':terminal['execution_plan_sha256'],'timeout_seconds':30})
        terminal['claimed'].append(name)
        env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
        for key in ('R1_TRACE_PATH','ACTIVE113_TRACE_PATH','ACTIVE113_SCORE_PATH'):env.pop(key,None)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        if capture:env['ACTIVE113_TRACE_PATH']=str(trace);env['ACTIVE113_SCORE_PATH']=str(score)
        stdout=out/'receipts'/(name+'.json');stderr=out/'logs'/(name+'.stderr')
        with stdout.open('xb') as so,stderr.open('xb') as se:
            child=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=so,stderr=se,env=env,
                timeout=min(30,deadline-time.monotonic()),preexec_fn=limits,check=False)
        need(child.returncode==0 and stderr.stat().st_size==0 and stdout.stat().st_size<=65536,'child failure; no retry: '+name)
        receipt=load(stdout);active=kind!='baseline_cpu' and kind!='baseline_memory'
        schema='r1-active113-offline-resource-v1' if active else 'r1-strong-fsmn-offline-board-resource-v1'
        need(receipt.get('schema')==schema and receipt.get('model_storage')=='read_only_mmap' and
             receipt.get('model_caller_heap_bytes')==0,'wrong schema/storage: '+name)
        need(receipt.get('pcm_bytes')==record['pcm_bytes'] and receipt.get('feed_chunk_bytes')==640 and
             receipt.get('feed_calls')==(record['pcm_bytes']+639)//640 and
             all(receipt.get(k)==v for k,v in record['expected_counts'].items()),'input/count mismatch: '+name)
        need(all(receipt.get(k) is False for k in ('device_qualified','live_afe','independent_far_frr')),'qualification boundary changed')
        for k in ('offline_feed_cpu_seconds','offline_feed_wall_seconds','cpu_seconds_per_audio_second'):
            need(type(receipt.get(k)) in (int,float) and math.isfinite(receipt[k]) and receipt[k]>0,'invalid timing receipt')
        if active:
            need(receipt.get('model_sha256')==pack['model']['sha256'] and receipt.get('frontend_sha256')==pack['frontend']['sha256'] and
                 receipt.get('selection_sha256')==declared['selection_sha256'] and receipt.get('model_weight_bytes')==1582624 and
                 receipt.get('capture_instrumented') is capture,'active identity/capture mismatch')
        completed={'name':name,'index':record['index'],'scope':kind,'architecture':arch,
            'receipt_sha256':sha(stdout),'counts_match_saved':True,'expected_counts':record['expected_counts']}
        if capture:
            need(read(trace)==saved[record['index']],'F/E byte mismatch: '+name)
            raw=read(score);h=hidden[record['index']];p=predicted[record['index']];frames=record['expected_counts']['sdk_model_frames']
            need(raw[:64]==SCORE_HEADER and len(raw)==64+frames*1200,'capture score shape: '+name)
            for frame in range(frames):
                a=64+frame*1200;b=64+frame*40
                need(raw[a:a+24]==h[a:a+24]==p[b:b+24],'frame/clock bits changed: '+name)
                need(raw[a+24:a+28]==h[a+24:a+28],'original argmax ID changed: '+name)
                need(raw[a+28:a+32]==p[b+24:b+28],'selected OTHER ID changed: '+name)
                need(raw[a+32:a+1152]==h[a+32:a+1152],'hidden140 bits changed: '+name)
                need(raw[a+1152:a+1192]==h[a+1152:a+1192],'five precise logits bits changed: '+name)
                need(raw[a+1192:a+1200]==p[b+32:b+40],'selected OTHER score bits changed: '+name)
                scores=struct.unpack_from('<6d',raw,a+1152);other=struct.unpack_from('<I',raw,a+28)[0]
                need(all(math.isfinite(x) for x in scores),'nonfinite actual score')
                ids=(*PRECISE,other);winner=min(range(6),key=lambda j:(-scores[j],ids[j]))
                need(ids[winner]==struct.unpack_from('<I',raw,a+24)[0],'score/original-ID winner mismatch')
            evidence={'name':name,'index':record['index'],'architecture':arch,'frames':frames,
                'fe_trace_byte_equal':True,'hidden140_bit_equal':True,'five_precise_logits_bit_equal':True,
                'selected_other_score_bit_equal':True,'selected_other_id_equal':True,'argmax_ids_equal':True,
                'trace_sha256':sha(trace),'score_trace_sha256':sha(score),'receipt_sha256':sha(stdout),
                'hidden_reference_sha256':record['hidden_sha256'],'prediction_reference_sha256':record['cached_prediction_sha256'],
                'expected_counts':record['expected_counts']}
            save(out/'comparisons'/(name+'.json'),evidence);terminal['functional'].append(evidence)
            completed.update(trace_sha256=evidence['trace_sha256'],score_trace_sha256=evidence['score_trace_sha256'])
        terminal['completed_invocations'].append(completed);return receipt
    try:
        for arch in ('host','arm'):
            for record in records:
                name=f'capture-{arch}-{record["index"]:03d}'
                argv=[bins[arch,'capture'],model,frontend,record['pcm_path'],'--mmap']
                if arch=='arm':argv=[qemu,'-L',sysroot,*argv]
                invoke(name,argv,record,'candidate_functional_capture',True,arch)
                if len(terminal['functional'])%20==0:print(json.dumps({'functional_completed':len(terminal['functional']),'expected':260}),flush=True)
        need(len(terminal['functional'])==260,'functional matrix incomplete; resource calls forbidden')
        save(out/'FUNCTIONAL_PASS.json',{'status':'PASS260_FE_HIDDEN140_PRECISE5_SELECTED_OTHER_BITS',
            'invocations':260,'unique_known_inputs':130,'frames_per_architecture':12255,'events_per_architecture':86,
            'functional_records':terminal['functional'],'execution_plan_sha256':terminal['execution_plan_sha256']})
        terminal['stage']='HOST_CPU';cpu={'baseline':[],'active113':[]};cpu_p95={'baseline':[],'active113':[]}
        for pair,order in enumerate(declared['resource_orders']['cpu']):
            for variant in order:
                binary,weights=(baseline_binary,baseline_model) if variant=='baseline' else (bins['host','plain'],model)
                row=invoke(f'cpu-{pair:02d}-{variant}',[binary,weights,frontend,longest['pcm_path'],'--mmap'],longest,
                    'baseline_cpu' if variant=='baseline' else 'active113_cpu')
                cpu[variant].append(row['cpu_seconds_per_audio_second']);cpu_p95[variant].append(row['feed_p95_histogram_upper_us'])
        savings=[100*(1-a/b) for b,a in zip(cpu['baseline'],cpu['active113'])]
        terminal['host_cpu']={'scope':'host only, uninstrumented plain probes, full offline feed, both read-only mmap',
            'pairs':9,'invocations':18,'pcm_index':49,'cpu_seconds_per_audio_second':cpu,'p95_histogram_upper_us':cpu_p95,
            'median_cpu_seconds_per_audio_second':{k:statistics.median(v) for k,v in cpu.items()},
            'paired_cpu_savings_pct':savings,'paired_median_cpu_savings_pct':statistics.median(savings),
            'active_wins':sum(a<b for b,a in zip(cpu['baseline'],cpu['active113'])),
            'warmup_calls':0,'capture_or_qemu_measurements_used':0,'physical_ssc305':'NOT_RUN'}
        terminal['stage']='HOST_MEMORY';metrics=('proc_self_pss_kib','proc_self_private_dirty_kib')
        memory={v:{k:[] for k in metrics} for v in ('baseline','active113')}
        for pair,order in enumerate(declared['resource_orders']['memory']):
            for variant in order:
                binary,weights=(baseline_binary,baseline_model) if variant=='baseline' else (bins['host','plain'],model)
                row=invoke(f'memory-{pair:02d}-{variant}',[binary,weights,frontend,longest['pcm_path'],'--mmap'],longest,
                    'baseline_memory' if variant=='baseline' else 'active113_memory')
                for metric in metrics:
                    value=row.get(metric);need(value is None or (type(value) is int and value>=0),'invalid own-process memory metric')
                    memory[variant][metric].append(value)
        medians={v:{k:statistics.median(vals) if all(x is not None for x in vals) else None for k,vals in by.items()} for v,by in memory.items()}
        deltas={k:[b-a for b,a in zip(memory['baseline'][k],memory['active113'][k])] if
                all(x is not None for v in memory for x in memory[v][k]) else None for k in metrics}
        terminal['host_memory']={'scope':'own-process metrics after actual streaming before SDK destroy; both models mmap',
            'pairs':6,'invocations':12,'pcm_index':49,'own_process_observations_kib':memory,'own_process_medians_kib':medians,
            'paired_baseline_minus_active113_kib':deltas,
            'median_baseline_minus_active113_kib':{k:statistics.median(v) if v is not None else None for k,v in deltas.items()},
            'peak_rss_used_for_savings':False,'memory_pair_timing_used_for_speedup':False,'physical_ssc305':'NOT_RUN'}
        names=[r['name'] for r in terminal['completed_invocations']]
        need(len(names)==len(set(names))==len(terminal['claimed'])==290 and set(names)==set(terminal['claimed']),'claim completion denominator')
        for name,expected_hash in watched.items():need(sha(Path(name))==expected_hash,'post-execution identity drift: '+name)
        need(os.sched_getaffinity(0)=={affinity} and len(os.listdir('/proc/self/task'))==1,'CPU/thread contract changed')
        terminal.update(status='PASS_ACTIVE113_FIXED_PCM_PARITY_HOST_RESOURCE_ONLY',stage='COMPLETED',
            all_claims_completed=True,completed_pcm_calls=290,candidate_capture_calls=260,host_cpu_calls=18,host_memory_calls=12,
            hidden140_bit_parity='PASS',precise5_bit_parity='PASS',selected_other_bit_parity='PASS',
            all_inputs_models_binaries_sources_posthash_unchanged=True)
    except BaseException as exc:
        terminal.update(status='STOPPED_ACTIVE113_PCM_VALIDATION',failure=type(exc).__name__+': '+str(exc));raise
    finally:
        terminal['wall_seconds']=time.monotonic()-start;signal.setitimer(signal.ITIMER_REAL,0);save(out/'TERMINAL.json',terminal)
    save(out/'AGGREGATE.json',{'schema':'active113-pcm-validation-aggregate-v1','status':terminal['status'],
        'execution_plan_sha256':terminal['execution_plan_sha256'],'terminal_sha256':sha(out/'TERMINAL.json'),
        'functional_pass_sha256':sha(out/'FUNCTIONAL_PASS.json'),'selection_sha256':declared['selection_sha256'],
        'cached_screen_sha256':declared['cached_screen_sha256'],'model_sha256':pack['model']['sha256'],
        'frontend_sha256':pack['frontend']['sha256'],'model_bytes':1582624,'original_model_bytes':1960140,
        'known_inputs':130,'frames_per_architecture':12255,'events_per_architecture':86,
        'completed_pcm_calls':290,'candidate_capture_calls':260,'host_cpu_calls':18,'host_memory_calls':12,
        'all_claims_completed':True,'all_inputs_models_binaries_sources_posthash_unchanged':True,
        'hidden140_bit_parity':'PASS','precise5_bit_parity':'PASS','selected_other_bit_parity':'PASS',
        'fe_trace_byte_parity':'PASS','full_2601_logits':'NOT_TESTED','host_cpu':terminal['host_cpu'],
        'host_memory':terminal['host_memory'],'wall_seconds':terminal['wall_seconds'],
        'baseline_functional_reruns':0,'unknown_pcm_calls':0,'training':0,'default_model_changed':False,
        'model_promotion_authorized':False,'original_fp32_numeric_gate':'FAIL_UNCHANGED',
        'physical_ssc305':'NOT_RUN','independent_far_frr':'NOT_RUN'})
    print(json.dumps({'status':terminal['status'],'completed_pcm_calls':290,'aggregate_sha256':sha(out/'AGGREGATE.json'),
        'host_cpu':terminal['host_cpu'],'host_memory':terminal['host_memory']},sort_keys=True))
if __name__=='__main__':main()
