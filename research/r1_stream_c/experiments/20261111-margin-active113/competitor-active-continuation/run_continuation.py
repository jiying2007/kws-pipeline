#!/usr/bin/env python3
"""Complementary execution of125unattemptedARM inputs, then30resource calls.

Keeps the original strict bit gate FAILED. Reuses135captured candidate calls;
six-score cross-architecture acceptance uses the pre-existing logit contract.
Hidden140 differences are diagnostics only. No binary/model/selection changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import statistics
import struct
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
STRICT=ROOT/'competitor-active-validation'
OLD=STRICT/'execution'
DIAGNOSTIC=ROOT/'competitor-active-arm-diagnostic'
PRECISE=(0,1027,1401,1462,2093)
KNOWN=tuple(range(96))+tuple(range(100,134))
NEW_ARM=tuple(range(5,96))+tuple(range(100,134))
HEADER=struct.pack('<8s6I5IIQ',b'R1A113H\0',1,64,1200,140,6,2601,*PRECISE,113,0)
LIMIT=1024*1024

def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):
    need(stat.S_ISREG(path.lstat().st_mode),'regular non-symlink file required: '+str(path))
    return path.read_bytes()
def sha(path):return hashlib.sha256(read(path)).hexdigest()
def ident(path):return {'bytes':path.stat().st_size,'sha256':sha(path)}
def load(path):
    def bad(value):raise ValueError('nonfinite JSON constant')
    return json.loads(read(path),parse_constant=bad)
def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def timed_out(_signum,_frame):raise TimeoutError('cumulative executed-phase900seconddeadline')
def numeric_metrics(actual,reference,offset,count,frames,enforce_logits=False):
    different_values=different_frames=0;maximum=0.0;maximum_ratio=0.0;square_sum=0.0
    for frame in range(frames):
        base=64+frame*1200+offset;row_different=False
        for j in range(count):
            pos=base+j*8;a=struct.unpack_from('<d',actual,pos)[0];b=struct.unpack_from('<d',reference,pos)[0]
            need(math.isfinite(a) and math.isfinite(b),'nonfinite saved/captured numerical value')
            difference=abs(a-b);maximum=max(maximum,difference);square_sum+=difference*difference
            if actual[pos:pos+8]!=reference[pos:pos+8]:different_values+=1;row_different=True
            if enforce_logits:
                bound=1e-8+1e-8*abs(b);maximum_ratio=max(maximum_ratio,difference/bound)
                need(difference<=bound and difference<=1e-6,'pre-existing six-score logit contract exceeded')
        different_frames+=row_different
    result={'values':frames*count,'bit_different_values':different_values,'bit_different_frames':different_frames,
            'maximum_absolute_difference':maximum,'rms_difference':math.sqrt(square_sum/(frames*count))}
    if enforce_logits:result.update(atol=1e-8,rtol=1e-8,maximum_absolute_bound=1e-6,maximum_tolerance_ratio=maximum_ratio,contract_passed=True)
    else:result['acceptance_bound_applied']=False
    return result
def compare_capture(name,index,score_path,trace_path,host_scores,baseline_trace,record):
    actual=read(score_path);reference=host_scores;frames=record['expected_counts']['sdk_model_frames']
    need(actual[:64]==reference[:64]==HEADER and len(actual)==len(reference)==64+frames*1200,'score capture shape/header')
    need(read(trace_path)==baseline_trace,'complete F/E byte stream differs from saved original baseline')
    for frame in range(frames):
        start=64+frame*1200
        need(actual[start:start+32]==reference[start:start+32],
             'frame/clocks/globalID/selectedOTHER-ID differ from corresponding HOST ACTIVE113')
        scores=struct.unpack_from('<6d',actual,start+1152);other=struct.unpack_from('<I',actual,start+28)[0]
        need(other<2601 and other not in PRECISE,'invalid actual OTHER metadata')
        ids=(*PRECISE,other);winner=min(range(6),key=lambda j:(-scores[j],ids[j]))
        need(ids[winner]==struct.unpack_from('<I',actual,start+24)[0],'actual score/original-ID argmax inconsistent')
    logits=numeric_metrics(actual,reference,1152,6,frames,True)
    hidden=numeric_metrics(actual,reference,32,140,frames,False)
    return {'name':name,'index':index,'frames':frames,'fe_byte_equal':True,'clocks_and_original_ids_exact':True,
        'six_scores_to_host_active113':logits,'hidden140_diagnostic_to_host_active113':hidden,
        'score_sha256':sha(score_path),'trace_sha256':sha(trace_path),'host_score_sha256':hashlib.sha256(reference).hexdigest(),
        'original_strict_bit_gate':'FAIL_UNCHANGED','hidden140_acceptance_bound_applied':False}
def validate_receipt(row,record,active,model_sha,frontend_sha,selection_sha,capture=False):
    schema='r1-active113-offline-resource-v1' if active else 'r1-strong-fsmn-offline-board-resource-v1'
    need(row.get('schema')==schema and row.get('model_storage')=='read_only_mmap' and row.get('model_caller_heap_bytes')==0,'wrong resource identity/storage')
    need(row.get('pcm_bytes')==record['pcm_bytes'] and row.get('feed_chunk_bytes')==640 and
         row.get('feed_calls')==(record['pcm_bytes']+639)//640 and
         all(row.get(k)==v for k,v in record['expected_counts'].items()),'PCM/frame/event denominator changed')
    need(all(row.get(k) is False for k in ('device_qualified','live_afe','independent_far_frr')),'qualification boundary changed')
    if active:need(row.get('model_sha256')==model_sha and row.get('frontend_sha256')==frontend_sha and
        row.get('selection_sha256')==selection_sha and row.get('model_weight_bytes')==1582624 and
        row.get('capture_instrumented') is capture,'active identity or capture mode changed')
    for key in ('offline_feed_cpu_seconds','offline_feed_wall_seconds','cpu_seconds_per_audio_second'):
        need(type(row.get(key)) in (int,float) and math.isfinite(row[key]) and row[key]>0,'invalid timing receipt')
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-sha256',required=True);ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--execute',action='store_true');ap.add_argument('--preflight-sha256');args=ap.parse_args()
    need(args.execute!=args.preflight_only,'choose one of --preflight-only or --execute')
    os.umask(0o077);started=time.monotonic();out=HERE/'execution'
    need(not out.exists(),'fresh continuation execution required; no retry')
    need(sha(HERE/'PLAN.json')==args.plan_sha256,'frozen continuation plan changed')
    plan=load(HERE/'PLAN.json');watched={str(HERE/'PLAN.json'):args.plan_sha256}
    need(plan['new_arm_indices']==list(NEW_ARM) and plan['limits']=={'maximum_new_arm_calls':125,'host_cpu_calls':18,
        'host_memory_calls':12,'maximum_new_pcm_calls':155,'cumulative_pcm_calls':291,'cumulative_executed_wall_seconds':900,
        'child_seconds':30,'cpu_threads':1,'retry':False},'continuation membership/budget changed')
    need(plan['six_score_contract']['atol']==1e-8 and plan['six_score_contract']['rtol']==1e-8 and
         plan['six_score_contract']['maximum_absolute_difference']==1e-6 and
         plan['six_score_contract']['reference']=='corresponding retained HOST ACTIVE113 six-score capture' and
         plan['six_score_contract']['hidden140_acceptance_bound_applied'] is False,'fixed numerical contract changed')
    for name,value in plan['source_files'].items():need(ident(Path(name))==value,'continuation source changed');watched[name]=value['sha256']
    for name,value in plan['pins'].items():need(sha(Path(name))==value,'frozen evidence changed');watched[name]=value
    strict=load(OLD/'TERMINAL.json');audit=load(STRICT/'INDEPENDENT_AUDIT.json');original_plan=load(OLD/'PLAN.json')
    old_preflight=load(STRICT/'PREFLIGHT.json');original_declared=load(STRICT/'DECLARED_PLAN.json')
    stopped_name=strict['claimed'][-1]
    need(strict['status']=='STOPPED_ACTIVE113_PCM_VALIDATION' and strict['failure']=='ValueError: hidden140 bits changed: '+stopped_name and
         len(strict['claimed'])==135 and len(strict['completed_invocations'])==134 and not strict['host_cpu'] and not strict['host_memory'],
         'original failed phase changed')
    need(audit['status']=='PASS_AUDIT_OF_STOPPED_VALIDATION' and audit['original_strict_bit_gate']=='FAIL_UNCHANGED' and
         audit['evidence_sha256']['execution/TERMINAL.json']==sha(OLD/'TERMINAL.json') and
         audit['public_summary']['executed_pcm_calls']==135,'stopped-phase audit mismatch')
    for name,value in old_preflight['validated_file_sha256'].items():need(sha(Path(name))==value,'protected original input/source/model changed');watched[name]=value
    expected_old={f'capture-host-{i:03d}' for i in KNOWN}|{f'capture-arm-{i:03d}' for i in range(5)}
    need(set(strict['claimed'])==set(audit['private_artifact_inventory'])==expected_old,'retained135capture identities changed')
    for name,entry in audit['private_artifact_inventory'].items():
        for sub,suffix,key in [('claims','.json','claim_sha256'),('receipts','.json','receipt_sha256'),('scores','.r1s','score_sha256'),('traces','.txt','trace_sha256')]:
            path=OLD/sub/(name+suffix);need(sha(path)==entry[key],'retained candidate artifact drift');watched[str(path)]=entry[key]
        stderr=OLD/'logs'/(name+'.stderr');need(read(stderr)==b'','retained child stderr changed');watched[str(stderr)]=sha(stderr)
    stopped_claim=load(OLD/'claims'/(stopped_name+'.json'))
    records={r['index']:r for r in original_plan['input_records']};need(set(records)==set(KNOWN),'known130membership changed')
    need(stopped_claim['name']==stopped_name and stopped_claim['architecture']=='arm' and
         stopped_claim['expected_counts']==records[stopped_claim['index']]['expected_counts'],'stopped capture identity changed')
    stopped_frames=records[stopped_claim['index']]['expected_counts']['sdk_model_frames']
    diagnostic=load(DIAGNOSTIC/'execution/RESULT.json');diagnostic_terminal=load(DIAGNOSTIC/'execution/TERMINAL.json')
    diagnostic_audit=load(DIAGNOSTIC/'INDEPENDENT_AUDIT.json')
    need(diagnostic_audit['status']=='PASS_SAVED_SINGLE_ARM_DIAGNOSTIC_AUDIT' and
         diagnostic_audit['evidence_sha256']['execution/RESULT.json']==sha(DIAGNOSTIC/'execution/RESULT.json') and
         diagnostic_audit['all_bound_sources_models_saved_outputs_unchanged'] is True,'independent diagnostic audit required')
    need(diagnostic['schema']=='active113-single-same-arm-diagnostic-result-v1' and
         diagnostic['status']=='PASS_SAME_ARM_BASELINE_HIDDEN_PRECISE' and diagnostic['actual_pcm_calls']==1 and diagnostic['frames']==stopped_frames and
         all(diagnostic[k] is True for k in ('all_hidden140_bits_equal','all_precise5_bits_equal','fe_byte_exact','global_winner_ids_byte_exact')),
         'one-call same-ARM baseline proof required')
    need(diagnostic['terminal_sha256']==sha(DIAGNOSTIC/'execution/TERMINAL.json') and
         diagnostic['baseline_model_sha256']==original_plan['baseline_model']['sha256'] and
         diagnostic['retained_candidate_score_sha256']==audit['private_artifact_inventory'][stopped_name]['score_sha256'] and
         diagnostic['retained_candidate_trace_sha256']==audit['private_artifact_inventory'][stopped_name]['trace_sha256'],
         'same-architecture diagnostic provenance mismatch')
    prior_wall=float(strict['wall_seconds']);diagnostic_wall=float(diagnostic_terminal['wall_seconds'])
    need(math.isfinite(prior_wall) and math.isfinite(diagnostic_wall) and prior_wall>0 and diagnostic_wall>0,'invalid prior duration')
    remaining_limit=900-prior_wall-diagnostic_wall;need(remaining_limit>0,'cumulative wall budget exhausted')
    deadline=started+remaining_limit;signal.signal(signal.SIGALRM,timed_out)
    signal.setitimer(signal.ITIMER_REAL,max(0.001,deadline-time.monotonic()))
    need(sum(r['expected_counts']['sdk_model_frames'] for r in records.values())==12255 and
         sum(r['expected_counts']['keyword_events'] for r in records.values())==86,'known frame/event totals changed')
    model=Path(original_plan['active_model']['path']);frontend=Path(original_plan['frontend']['path'])
    base_model=Path(original_plan['baseline_model']['path']);base_probe=Path(original_plan['baseline_probe']['path'])
    qemu=Path(original_plan['qemu']['path']);sysroot=Path(original_plan['sysroot'])
    arm_probe=Path(original_plan['active_probes']['arm']['binaries']['capture']['path'])
    host_probe=Path(original_plan['active_probes']['host']['binaries']['plain']['path'])
    model_sha=original_plan['active_model']['sha256'];frontend_sha=original_plan['frontend']['sha256'];selection_sha=original_plan['selection_sha256']
    affinity=original_plan['cpu_affinity'];need(len(affinity)==1 and affinity[0] in os.sched_getaffinity(0),'original CPU affinity unavailable')
    os.sched_setaffinity(0,set(affinity));need(len(os.listdir('/proc/self/task'))==1,'single runner thread required')
    host_scores={};baseline_traces={};retained_arm=[]
    savedroot=Path('/workspace/scratch/1e1fe64498b2/r1-local/evidence-qemu')
    for i in KNOWN:
        host_scores[i]=read(OLD/'scores'/f'capture-host-{i:03d}.r1s')
        baseline_traces[i]=read(savedroot/'private-traces'/f'{i:03d}-arm_par4.txt')
        name=f'capture-host-{i:03d}';row=load(OLD/'receipts'/(name+'.json'))
        validate_receipt(row,records[i],True,model_sha,frontend_sha,selection_sha,True)
        need(read(OLD/'traces'/(name+'.txt'))==baseline_traces[i],'retained host F/E changed')
        need(host_scores[i][:64]==HEADER and len(host_scores[i])==64+records[i]['expected_counts']['sdk_model_frames']*1200,'retained host score shape')
    for i in range(5):
        name=f'capture-arm-{i:03d}';validate_receipt(load(OLD/'receipts'/(name+'.json')),records[i],True,model_sha,frontend_sha,selection_sha,True)
        retained_arm.append(compare_capture(name,i,OLD/'scores'/(name+'.r1s'),OLD/'traces'/(name+'.txt'),host_scores[i],baseline_traces[i],records[i]))
    preview={'schema':'active113-complementary-execution-plan-v1','declared_plan_sha256':args.plan_sha256,
        'original_failed_terminal_sha256':sha(OLD/'TERMINAL.json'),'stopped_audit_sha256':sha(STRICT/'INDEPENDENT_AUDIT.json'),
        'diagnostic_result_sha256':sha(DIAGNOSTIC/'execution/RESULT.json'),'diagnostic_terminal_sha256':sha(DIAGNOSTIC/'execution/TERMINAL.json'),
        'diagnostic_independent_audit_sha256':sha(DIAGNOSTIC/'INDEPENDENT_AUDIT.json'),
        'retained_candidate_calls':135,'diagnostic_baseline_calls':1,'new_arm_indices':list(NEW_ARM),
        'new_candidate_arm_calls':125,'host_cpu_calls':18,'host_memory_calls':12,'new_pcm_budget':155,'cumulative_pcm_budget':291,
        'remaining_executed_wall_budget_seconds':remaining_limit,'child_seconds':30,'cpu_affinity':affinity,
        'resource_orders':original_plan['resource_orders'],'unchanged_probe_model_source_plan':original_plan,
        'retained_arm_comparisons':retained_arm,'six_score_contract':plan['six_score_contract'],
        'hidden140_acceptance_bound_applied':False,'original_strict_bit_gate':'FAIL_UNCHANGED','full_2601_logits':'NOT_TESTED',
        'posterior_parity':'NOT_TESTED','model_promotion_authorized':False}
    preflight_path=HERE/'PREFLIGHT.json'
    if args.preflight_only:
        save(preflight_path,{'schema':'active113-complementary-preflight-v1','status':'READY_NO_NEW_PCM',
            'execution_plan_preview':preview,'validated_file_sha256':watched,'new_pcm_calls':0,'new_head_or_backbone_calls':0})
        signal.setitimer(signal.ITIMER_REAL,0);print(json.dumps({'status':'READY_NO_NEW_PCM','preflight_sha256':sha(preflight_path)}));return
    need(args.preflight_sha256 and sha(preflight_path)==args.preflight_sha256,'reviewed preflight required')
    preflight=load(preflight_path);need(preflight['execution_plan_preview']==preview and preflight['validated_file_sha256']==watched,'preflight identity mismatch')
    watched[str(preflight_path)]=args.preflight_sha256;preview['preflight_sha256']=args.preflight_sha256
    out.mkdir(mode=0o700)
    for name in ('claims','receipts','scores','traces','logs','comparisons'):(out/name).mkdir()
    save(out/'PLAN.json',preview)
    terminal={'schema':'active113-complementary-terminal-v1','status':'RUNNING','stage':'UNATTEMPTED_ARM',
        'claimed':[],'completed_invocations':[],'new_arm_comparisons':[],'retained_arm_comparisons':retained_arm,
        'host_cpu':None,'host_memory':None,'original_strict_bit_gate':'FAIL_UNCHANGED','default_model_changed':False,
        'model_promotion_authorized':False,'full_2601_logits':'NOT_TESTED','posterior_parity':'NOT_TESTED','training':0,'unknown_calls':0}
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE,(LIMIT,LIMIT));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.sched_setaffinity(0,set(affinity))
    def invoke(name,argv,record,scope,active,capture=False):
        need(len(terminal['claimed'])<155 and time.monotonic()<deadline,'new call/wall budget exhausted')
        need(name not in expected_old,'retained candidate invocation must not be repeated')
        trace=out/'traces'/(name+'.txt') if capture else None;score=out/'scores'/(name+'.r1s') if capture else None
        argv=[str(x) for x in argv]
        save(out/'claims'/(name+'.json'),{'name':name,'attempt':1,'argv':argv,'scope':scope,'index':record['index'],
            'pcm_sha256':record['pcm_sha256'],'expected_counts':record['expected_counts'],
            'trace':str(trace) if trace else None,'scores':str(score) if score else None,'execution_plan_sha256':sha(out/'PLAN.json')})
        terminal['claimed'].append(name)
        env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
        for key in ('R1_TRACE_PATH','ACTIVE113_TRACE_PATH','ACTIVE113_SCORE_PATH'):env.pop(key,None)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        if capture:env['ACTIVE113_TRACE_PATH']=str(trace);env['ACTIVE113_SCORE_PATH']=str(score)
        stdout=out/'receipts'/(name+'.json');stderr=out/'logs'/(name+'.stderr')
        with stdout.open('xb') as so,stderr.open('xb') as se:
            result=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=so,stderr=se,env=env,
                timeout=min(30,deadline-time.monotonic()),preexec_fn=limits,check=False)
        need(result.returncode==0 and stderr.stat().st_size==0 and stdout.stat().st_size<=65536,'child failed; no retry: '+name)
        receipt=load(stdout);validate_receipt(receipt,record,active,model_sha,frontend_sha,selection_sha,capture)
        completed={'name':name,'scope':scope,'index':record['index'],'receipt_sha256':sha(stdout),
                   'counts_match_saved':True,'expected_counts':record['expected_counts']}
        if capture:
            comparison=compare_capture(name,record['index'],score,trace,host_scores[record['index']],baseline_traces[record['index']],record)
            save(out/'comparisons'/(name+'.json'),comparison);terminal['new_arm_comparisons'].append(comparison)
            completed.update(score_sha256=sha(score),trace_sha256=sha(trace))
        terminal['completed_invocations'].append(completed);return receipt
    try:
        for i in NEW_ARM:
            record=records[i];invoke(f'new-arm-{i:03d}',[qemu,'-L',sysroot,arm_probe,model,frontend,record['pcm_path'],'--mmap'],
                record,'previously_unattempted_arm_capture',True,True)
            if len(terminal['new_arm_comparisons'])%20==0:print(json.dumps({'new_arm_completed':len(terminal['new_arm_comparisons']),'expected':125}),flush=True)
        combined=retained_arm+terminal['new_arm_comparisons']
        need(len(combined)==130 and {r['index'] for r in combined}==set(KNOWN),'all130ARMcoverage required before resources')
        save(out/'NUMERIC_PASS.json',{'status':'PASS130_ARM_SIX_SCORE_LOGIT_CONTRACT_AND_EXACT_FE_IDS',
            'arm_comparisons':combined,'retained_arm_calls':5,'new_arm_calls':125,'retained_host_calls':130,
            'six_score_contract':plan['six_score_contract'],'hidden140_acceptance_bound_applied':False,
            'original_strict_bit_gate':'FAIL_UNCHANGED','full_2601_logits':'NOT_TESTED','posterior_parity':'NOT_TESTED'})
        terminal['stage']='HOST_CPU';record=records[49];cpu={'baseline':[],'active113':[]};p95={'baseline':[],'active113':[]}
        for pair,order in enumerate(original_plan['resource_orders']['cpu']):
            for variant in order:
                active=variant=='active113';binary=host_probe if active else base_probe;weights=model if active else base_model
                row=invoke(f'cpu-{pair:02d}-{variant}',[binary,weights,frontend,record['pcm_path'],'--mmap'],record,'host_plain_cpu',active)
                cpu[variant].append(row['cpu_seconds_per_audio_second']);p95[variant].append(row['feed_p95_histogram_upper_us'])
        savings=[100*(1-a/b) for b,a in zip(cpu['baseline'],cpu['active113'])]
        terminal['host_cpu']={'scope':'host uninstrumented full offline feed; both mmap; QEMU/capture timings excluded',
            'pairs':9,'invocations':18,'pcm_index':49,'cpu_seconds_per_audio_second':cpu,'feed_p95_histogram_upper_us':p95,
            'median_cpu_seconds_per_audio_second':{k:statistics.median(v) for k,v in cpu.items()},
            'paired_cpu_savings_pct':savings,'paired_median_cpu_savings_pct':statistics.median(savings),
            'active_wins':sum(a<b for b,a in zip(cpu['baseline'],cpu['active113'])),'warmup_calls':0,'physical_ssc305':'NOT_RUN'}
        terminal['stage']='HOST_MEMORY';metrics=('proc_self_pss_kib','proc_self_private_dirty_kib')
        memory={v:{m:[] for m in metrics} for v in ('baseline','active113')}
        for pair,order in enumerate(original_plan['resource_orders']['memory']):
            for variant in order:
                active=variant=='active113';binary=host_probe if active else base_probe;weights=model if active else base_model
                row=invoke(f'memory-{pair:02d}-{variant}',[binary,weights,frontend,record['pcm_path'],'--mmap'],record,'host_plain_memory',active)
                for metric in metrics:
                    value=row.get(metric);need(value is None or (type(value) is int and value>=0),'invalid PSS/Private_Dirty')
                    memory[variant][metric].append(value)
        medians={v:{m:statistics.median(xs) if all(x is not None for x in xs) else None for m,xs in by.items()} for v,by in memory.items()}
        deltas={m:[b-a for b,a in zip(memory['baseline'][m],memory['active113'][m])] if
            all(x is not None for v in memory for x in memory[v][m]) else None for m in metrics}
        terminal['host_memory']={'scope':'own-process after streaming before SDK destroy; both mmap','pairs':6,'invocations':12,
            'pcm_index':49,'own_process_observations_kib':memory,'own_process_medians_kib':medians,
            'paired_baseline_minus_active113_kib':deltas,'median_baseline_minus_active113_kib':{m:statistics.median(v) if v is not None else None for m,v in deltas.items()},
            'peak_rss_used_for_savings':False,'memory_pair_timing_used_for_speedup':False,'physical_ssc305':'NOT_RUN'}
        names=[r['name'] for r in terminal['completed_invocations']]
        need(len(names)==len(set(names))==len(terminal['claimed'])==155 and set(names)==set(terminal['claimed']),'all155uniqueclaims must complete')
        need({r['index'] for r in terminal['new_arm_comparisons']}==set(NEW_ARM),'new ARM membership changed')
        for name,value in watched.items():need(sha(Path(name))==value,'post-run protected identity drift')
        need(os.sched_getaffinity(0)==set(affinity) and len(os.listdir('/proc/self/task'))==1,'CPU/thread state changed')
        terminal.update(status='PASS_COMPLEMENTARY_SIX_SCORE_CONTRACT_HOST_RESOURCES_STRICT_BIT_FAIL_RETAINED',stage='COMPLETED',
            all_claims_completed=True,new_pcm_calls=155,new_arm_calls=125,host_cpu_calls=18,host_memory_calls=12,
            cumulative_pcm_calls=291,retained_candidate_calls=135,diagnostic_baseline_calls=1,
            all_protected_identities_posthash_unchanged=True,arm_known_inputs=130,host_known_inputs=130,
            frames_per_architecture=12255,events_per_architecture=86)
    except BaseException as exc:
        terminal.update(status='STOPPED_COMPLEMENTARY_PROTOCOL_STRICT_BIT_FAIL_RETAINED',failure=type(exc).__name__+': '+str(exc));raise
    finally:
        terminal['wall_seconds']=time.monotonic()-started
        terminal['cumulative_executed_phase_wall_seconds']=prior_wall+diagnostic_wall+terminal['wall_seconds']
        signal.setitimer(signal.ITIMER_REAL,0);save(out/'TERMINAL.json',terminal)
    combined=terminal['retained_arm_comparisons']+terminal['new_arm_comparisons']
    save(out/'AGGREGATE.json',{'schema':'active113-complementary-aggregate-v1','status':terminal['status'],
        'declared_plan_sha256':args.plan_sha256,'execution_plan_sha256':sha(out/'PLAN.json'),'terminal_sha256':sha(out/'TERMINAL.json'),
        'numeric_pass_sha256':sha(out/'NUMERIC_PASS.json'),'original_failed_terminal_sha256':sha(OLD/'TERMINAL.json'),
        'stopped_phase_audit_sha256':sha(STRICT/'INDEPENDENT_AUDIT.json'),'diagnostic_result_sha256':sha(DIAGNOSTIC/'execution/RESULT.json'),
        'diagnostic_independent_audit_sha256':sha(DIAGNOSTIC/'INDEPENDENT_AUDIT.json'),
        'selection_sha256':selection_sha,'model_sha256':model_sha,'frontend_sha256':frontend_sha,
        'model_bytes':1582624,'original_model_bytes':1960140,'retained_candidate_calls':135,'diagnostic_baseline_calls':1,
        'new_arm_calls':125,'host_cpu_calls':18,'host_memory_calls':12,'new_pcm_calls':155,'cumulative_pcm_calls':291,
        'known_inputs_per_architecture':130,'frames_per_architecture':12255,'events_per_architecture':86,
        'all_claims_completed':True,'all_protected_identities_posthash_unchanged':True,'fe_clocks_argmax_other_ids_exact':True,
        'six_score_contract':plan['six_score_contract'],'six_score_max_abs_difference':max(x['six_scores_to_host_active113']['maximum_absolute_difference'] for x in combined),
        'six_score_bit_different_values':sum(x['six_scores_to_host_active113']['bit_different_values'] for x in combined),
        'hidden140_diagnostic_max_abs_difference':max(x['hidden140_diagnostic_to_host_active113']['maximum_absolute_difference'] for x in combined),
        'hidden140_bit_different_values':sum(x['hidden140_diagnostic_to_host_active113']['bit_different_values'] for x in combined),
        'hidden140_acceptance_bound_applied':False,'original_strict_bit_gate':'FAIL_UNCHANGED','full_2601_logits':'NOT_TESTED',
        'posterior_parity':'NOT_TESTED','host_cpu':terminal['host_cpu'],'host_memory':terminal['host_memory'],
        'wall_seconds':terminal['wall_seconds'],'cumulative_executed_phase_wall_seconds':terminal['cumulative_executed_phase_wall_seconds'],
        'unknown_calls':0,'training':0,'candidate_reruns':0,'default_model_changed':False,'model_promotion_authorized':False,
        'physical_ssc305':'NOT_RUN','independent_far_frr':'NOT_RUN','original_fp32_numeric_gate':'FAIL_UNCHANGED'})
    print(json.dumps({'status':terminal['status'],'new_pcm_calls':155,'cumulative_pcm_calls':291,
        'aggregate_sha256':sha(out/'AGGREGATE.json'),'host_cpu':terminal['host_cpu'],'host_memory':terminal['host_memory']},sort_keys=True))
if __name__=='__main__':main()
