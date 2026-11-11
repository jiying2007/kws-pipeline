#!/usr/bin/env python3
"""One frozen ACTIVE113 cached-head pass, followed by saved-prediction scoring.

No PCM, backbone, frontend, training, tuning, or retry. All predictions are
durable before the unchanged C detector is called. Requires --execute.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import struct
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PRECISE=(0,1027,1401,1462,2093)
INPUT_HEADER=struct.pack('<8s6I5IIQ',b'R1HD140\0',1,64,1200,140,5,2601,*PRECISE,1,0)

def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):
    need(stat.S_ISREG(path.lstat().st_mode),'regular non-symlink file required')
    return path.read_bytes()
def sha(path):return hashlib.sha256(read(path)).hexdigest()
def load(path):return json.loads(read(path))
def identity(path):return {'bytes':path.stat().st_size,'sha256':sha(path)}
def save(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def write(path,data):
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def limits():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(10,10))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
def timed_out(_signum,_frame):raise TimeoutError('fixed 600 second total deadline')
def category(token):return 'blank' if token==0 else 'precise_'+str(token) if token in PRECISE else 'OTHER'
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    need(args.execute,'explicit --execute required')
    os.umask(0o077)
    started=time.monotonic();deadline=started+600
    signal.signal(signal.SIGALRM,timed_out);signal.setitimer(signal.ITIMER_REAL,600)
    out=HERE/'execution';need(not out.exists(),'fresh execution directory required; no retry/overwrite')
    declared=load(HERE/'PLAN.json');declared_sha=sha(HERE/'PLAN.json')
    need(declared['limits']=={'total_seconds':600,'child_seconds':10,'cpu_threads':1,'maximum_head_recordings':130,
         'maximum_head_frames':12255,'maximum_detector_replays':130,'retry':False},'execution limits changed')
    for name,expected in declared['source_files'].items():
        need(identity(Path(name))==expected,'frozen source changed')
    selection=load(HERE/'SELECTION.json')
    need(sha(HERE/'SELECTION.json')==declared['selection_sha256'],'selection changed')
    selected=set(selection['selected_other_ids']);need(len(selected)==113,'fixed 113 rows required')
    build=load(HERE/'build/BUILD_RECEIPT.json')
    need(build['status']=='BUILT_NOT_EXECUTED' and build['declared_plan_sha256']==declared_sha and
         build['source_files']==declared['source_files'] and build['compiler']==declared['compiler'],'build not source-bound')
    binary=Path(build['binary']['path'])
    need(identity(binary)=={k:build['binary'][k] for k in ('bytes','sha256')},'predictor binary changed')
    detector=Path(declared['detector_binary']['path']);need(identity(detector)=={k:declared['detector_binary'][k] for k in ('bytes','sha256')},'detector changed')
    model=Path(declared['model']['path']);need(identity(model)=={k:declared['model'][k] for k in ('bytes','sha256')},'model changed')
    inputs=Path(declared['inputs_root']);extract=Path(declared['extraction_root']);saved=Path(declared['saved_trace_root'])
    watched={str(HERE/'PLAN.json'):declared_sha,str(HERE/'SELECTION.json'):declared['selection_sha256'],str(model):declared['model']['sha256'],
             str(binary):build['binary']['sha256'],str(detector):declared['detector_binary']['sha256']}
    for path,expected in declared['data_pins'].items():need(sha(Path(path))==expected,'data pin changed');watched[path]=expected
    manifest=load(inputs/'INPUT_MANIFEST.json');labels={r['source_id']:r for r in (json.loads(line) for line in read(inputs/'original-observations.jsonl').splitlines())}
    extraction=load(extract/'RESULT.json');cache={r['index']:r for r in extraction['records']};baseline=load(saved/'FILE_MANIFEST.json')
    need(extraction['status']=='PASS_FIXED_TEACHER_EXTRACTION','uncompleted extraction')
    records=[]
    for row in manifest['records']:
        label=labels[row['source_id']]
        need(label['split']==row['split'] and label['pcm_sha256']==row['pcm_sha256'],'label join changed')
        if label['ctc_weight']==0:continue
        need(label['ctc_weight'] in (0.5,1.0) and label['evaluation_status']=='evaluated','unknown labels')
        refs={k['keyword_id']:k['reference'] for k in label['keywords'] if k['known'] is True}
        need(set(refs)=={1,2} and set(refs.values())<={0,1},'known weak references required')
        index=row['index'];c=cache[index];path=extract/f'hidden/{index:03d}.r1h'
        need(c['path']==f'hidden/{index:03d}.r1h' and c['original_split']==row['split'] and c['fit_excluded_unknown'] is False and
             path.stat().st_size==c['bytes'] and sha(path)==c['sha256'] and c['fe_trace_equal'] is True,'cache identity changed')
        trace_name=f'private-traces/{index:03d}-arm_par4.txt';expected=baseline[trace_name];trace=saved/trace_name
        need(identity(trace)==expected,'saved trace changed')
        watched[str(path)]=c['sha256'];watched[str(trace)]=expected['sha256']
        records.append({'index':index,'split':row['split'],'frames':c['frames'],'hidden_path':str(path),'hidden_sha256':c['sha256'],
                        'baseline_path':str(trace),'baseline_sha256':expected['sha256'],'references':refs,'pcm_samples':row['samples']})
    records.sort(key=lambda r:r['index'])
    need([r['index'] for r in records]==list(range(96))+list(range(100,134)),'known 130 membership changed')
    need(sum(r['frames'] for r in records if r['split']=='train')==9079 and
         sum(r['frames'] for r in records if r['split']=='development')==3176,'frame denominator changed')
    for source in selection['training_sources']:
        need(watched[source['cache_path']]==source['sha256'],'training selection cache changed')
    affinity=sorted(os.sched_getaffinity(0));need(affinity,'no available CPU');os.sched_setaffinity(0,{affinity[0]})
    need(len(os.listdir('/proc/self/task'))==1,'single thread required')
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    env['LC_ALL']='C'
    out.mkdir(mode=0o700)
    for name in ('claims','receipts','predictions','tokens','events','logs'):(out/name).mkdir()
    save(out/'PLAN.json',{'schema':'active113-cached-head-execution-plan-v1','declared_plan_sha256':declared_sha,
        'selection_sha256':declared['selection_sha256'],'build_receipt_sha256':sha(HERE/'build/BUILD_RECEIPT.json'),
        'predictor':build['binary'],'detector':declared['detector_binary'],'records':records,
        'cpu_affinity':[affinity[0]],'python':sys.version,'limits':declared['limits'],'policy':declared['policy']})
    terminal={'schema':'active113-cached-head-terminal-v1','status':'RUNNING','stage':'PREDICTING',
              'head_calls_completed':0,'head_frames_completed':0,'detector_calls_completed':0,'claims_created':0,
              'completed':[],'training_updates':0,'backbone_calls':0,'pcm_calls':0,'unknown_records_opened':0,
              'default_r1_changed':False,'model_promotion_authorized':False}
    predictions=[];errors={'train':[],'development':[]}
    def call(kind,index,argv,stdout,stderr,binding):
        remaining=deadline-time.monotonic();need(remaining>0,'total time exhausted')
        save(out/'claims'/f'{kind}-{index:03d}.json',{'kind':kind,'index':index,'argv':argv,'attempt':1,
            'selection_sha256':declared['selection_sha256'],'binding':binding,'child_limit_seconds':min(10,remaining)})
        terminal['claims_created']+=1
        with stdout.open('xb') as so,stderr.open('xb') as se:
            result=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=so,stderr=se,env=env,
                timeout=min(10,remaining),check=False,preexec_fn=limits)
        need(result.returncode==0 and stderr.stat().st_size==0 and stdout.stat().st_size<65536,'child failed; no retry')
    try:
        for row in records:
            index=row['index'];prediction=out/'predictions'/f'{index:03d}.r1a'
            receipt=out/'receipts'/f'{index:03d}.json'
            call('head',index,[str(binary),str(model),row['hidden_path'],row['hidden_sha256'],str(prediction),row['split']],
                 receipt,out/'logs'/f'head-{index:03d}.stderr',{'hidden_sha256':row['hidden_sha256']})
            observed=load(receipt)
            need(observed['status']=='PREDICTED_SELECTED_ORIGINAL_HEAD' and observed['frames']==row['frames'] and
                 observed['precise_logits_bit_equal']==row['frames']*5 and observed['selection_sha256']==declared['selection_sha256'] and
                 observed['model_sha256']==declared['model']['sha256'] and observed['hidden_sha256']==row['hidden_sha256'],
                 'prediction receipt mismatch')
            if row['split']=='train':need(observed['max_other_bit_equal_frames']==row['frames'] and
                 observed['true_other_winner_covered_frames']==row['frames'],'training max-OTHER not exact')
            need(prediction.stat().st_size==64+row['frames']*40,'prediction byte count')
            predictions.append({'index':index,'split':row['split'],'path':f'predictions/{index:03d}.r1a',
                **identity(prediction),'frames':row['frames'],'hidden_sha256':row['hidden_sha256'],'receipt_sha256':sha(receipt)})
            terminal['head_calls_completed']+=1;terminal['head_frames_completed']+=row['frames']
        need(len(predictions)==130 and terminal['head_frames_completed']==12255,'prediction denominator')
        save(out/'PREDICTIONS.json',{'schema':'active113-saved-predictions-v1','selection_sha256':declared['selection_sha256'],
            'declared_plan_sha256':declared_sha,'predictor_sha256':build['binary']['sha256'],
            'all_predictions_persisted_before_scoring':True,'records':predictions,'frames':12255,'head_recordings':130})
        prediction_manifest_sha=sha(out/'PREDICTIONS.json');watched[str(out/'PREDICTIONS.json')]=prediction_manifest_sha
        terminal['stage']='SCORING_SAVED_PREDICTIONS'
        for row,descriptor in zip(records,predictions):
            index=row['index'];raw=read(Path(row['hidden_path']));predpath=out/descriptor['path'];pred=read(predpath)
            need(sha(predpath)==descriptor['sha256'] and raw[:64]==INPUT_HEADER,'saved prediction/cache changed')
            expected_header=struct.pack('<8s6I5IIQ',b'R1ACTV1\0',1,64,40,140,5,2601,*PRECISE,113,row['frames'])
            need(pred[:64]==expected_header,'prediction header')
            token_lines=[];category_changes=Counter();coarse_changes=Counter();token_id_changes=0
            ties=teacher_ties=covered=equal=teacher_other=covered_teacher_other=0
            for frame in range(row['frames']):
                start=64+frame*1200;f,source,delivery=struct.unpack_from('<QQQ',raw,start)
                original,original_other=struct.unpack_from('<II',raw,start+24)
                precise=struct.unpack_from('<5d',raw,start+1152);full_other=struct.unpack_from('<d',raw,start+1192)[0]
                pf,ps,pd,other_id,reserved,other=struct.unpack_from('<QQQIId',pred,64+frame*40)
                need((pf,ps,pd)==(f,source,delivery) and f==frame and reserved==0 and other_id in selected and
                     all(math.isfinite(x) for x in (*precise,full_other,other)) and other<=full_other,'prediction row invariant')
                best=max(range(5),key=lambda j:precise[j]);maximum=precise[best];precise_id=PRECISE[best]
                reference=original_other if full_other>maximum or (full_other==maximum and original_other<precise_id) else precise_id
                need(reference==original,'teacher argmax/tie reconstruction')
                candidate=other_id if other>maximum or (other==maximum and other_id<precise_id) else precise_id
                bit_equal=pred[64+frame*40+32:64+frame*40+40]==raw[start+1192:start+1200]
                has=original_other in selected
                need(not has or (bit_equal and other_id==original_other),'covered maximum is not exact')
                if row['split']=='train':need(bit_equal and has and candidate==original,'training original argmax lost')
                equal+=bit_equal;covered+=has;ties+=other==maximum;teacher_ties+=full_other==maximum
                if original not in PRECISE:teacher_other+=1;covered_teacher_other+=has
                if original!=candidate:token_id_changes+=1
                before,after=category(original),category(candidate)
                if before!=after:category_changes[before+'->'+after]+=1
                cb='KEYWORD' if original in PRECISE[1:] else 'BLANK' if original==0 else 'OTHER'
                ca='KEYWORD' if candidate in PRECISE[1:] else 'BLANK' if candidate==0 else 'OTHER'
                if cb!=ca:coarse_changes[cb+'->'+ca]+=1
                errors[row['split']].append(full_other-other)
                token_lines.append(f'F,{frame},{source},{delivery},{candidate}\n'.encode())
            tokens=out/'tokens'/f'{index:03d}.txt';write(tokens,b''.join(token_lines))
            events=out/'events'/f'{index:03d}.txt'
            call('detector',index,[str(detector),'original',str(tokens)],events,out/'logs'/f'detector-{index:03d}.stderr',
                 {'prediction_sha256':descriptor['sha256'],'predictions_manifest_sha256':prediction_manifest_sha})
            terminal['detector_calls_completed']+=1
            baseline_lines=read(Path(row['baseline_path'])).splitlines(keepends=True)
            expected=b''.join(line for line in baseline_lines if line.startswith(b'E,'));actual=read(events)
            original_events=Counter(expected.splitlines());candidate_events=Counter(actual.splitlines())
            added=list((candidate_events-original_events).elements());missing=list((original_events-candidate_events).elements())
            words=[int(line.split(b',')[2]) for line in actual.splitlines()]
            wrong=sum(row['references'][word]==0 for word in words)
            added_wrong=sum(row['references'][int(line.split(b',')[2])]==0 for line in added)
            scored={'index':index,'split':row['split'],'frames':row['frames'],'events_exact':actual==expected,
                'actual_events':len(words),'reference_events':sum(original_events.values()),'new_events':len(added),
                'missing_events':len(missing),'wrong_keyword_events_machine_weak':wrong,
                'added_wrong_keyword_events_machine_weak':added_wrong,'max_other_bit_equal_frames':equal,
                'true_other_winner_covered_frames':covered,'teacher_overall_other_frames':teacher_other,
                'covered_teacher_overall_other_frames':covered_teacher_other,'original_token_id_changes':token_id_changes,
                'frame_category_changes':sum(category_changes.values()),'changed_categories':dict(category_changes),
                'coarse_category_changes':dict(coarse_changes),'precise_other_tie_frames':ties,
                'teacher_precise_other_tie_frames':teacher_ties,'prediction_sha256':descriptor['sha256'],'events_sha256':sha(events)}
            save(out/'receipts'/f'score-{index:03d}.json',scored);terminal['completed'].append(scored)
            watched[str(predpath)]=descriptor['sha256']
        need(len(terminal['completed'])==130 and terminal['claims_created']==260 and
             terminal['detector_calls_completed']==130,'incomplete call denominator')
        for path,expected in watched.items():need(sha(Path(path))==expected,'post-execution identity drift')
        for name,expected in declared['source_files'].items():need(identity(Path(name))==expected,'source changed during execution')
        need(os.sched_getaffinity(0)=={affinity[0]} and len(os.listdir('/proc/self/task'))==1,'runtime thread/CPU changed')
        groups={}
        for split in ('train','development'):
            group=[r for r in terminal['completed'] if r['split']==split];values=errors[split]
            aggregate_keys=('frames','actual_events','reference_events','new_events','missing_events','wrong_keyword_events_machine_weak',
                'added_wrong_keyword_events_machine_weak','max_other_bit_equal_frames','true_other_winner_covered_frames',
                'teacher_overall_other_frames','covered_teacher_overall_other_frames','original_token_id_changes','frame_category_changes',
                'precise_other_tie_frames','teacher_precise_other_tie_frames')
            summary={k:sum(r[k] for r in group) for k in aggregate_keys}
            summary.update(records=len(group),event_exact_records=sum(r['events_exact'] for r in group),
                max_other_underestimate_mean=sum(values)/len(values),max_other_underestimate_max=max(values))
            for key in ('changed_categories','coarse_category_changes'):
                combined=Counter()
                for row in group:combined.update(row[key])
                summary[key]=dict(combined)
            summary['true_other_winner_coverage_fraction']=summary['true_other_winner_covered_frames']/summary['frames']
            groups[split]=summary
        train=groups['train'];dev=groups['development']
        train_pass=train['records']==96 and train['max_other_bit_equal_frames']==9079 and train['original_token_id_changes']==0 and train['new_events']==0 and train['added_wrong_keyword_events_machine_weak']==0
        dev_pass=dev['records']==34 and dev['event_exact_records']==34
        terminal.update(status='PASS_RETAINED_EVENT_SCREEN_NOT_PROMOTED' if train_pass and dev_pass else 'REJECTED_FIXED_RESEARCH_SCREEN',
            stage='COMPLETED',groups=groups,training_max_other_bit_equal=train['max_other_bit_equal_frames']==9079,
            train_no_new_event=train_pass,development_complete_event_retention=dev_pass,
            inputs_model_sources_posthash_unchanged=True,all_final_predictions_saved_before_scoring=True)
    except BaseException as exc:
        terminal.update(status='STOPPED_FIXED_ACTIVE_SCREEN',failure=type(exc).__name__+': '+str(exc));raise
    finally:
        terminal['wall_seconds']=time.monotonic()-started;signal.setitimer(signal.ITIMER_REAL,0)
        save(out/'TERMINAL.json',terminal)
    save(out/'AGGREGATE.json',{'schema':'active113-cached-head-aggregate-v1','status':terminal['status'],
        'selection_sha256':declared['selection_sha256'],'declared_plan_sha256':declared_sha,
        'execution_plan_sha256':sha(out/'PLAN.json'),'terminal_sha256':sha(out/'TERMINAL.json'),
        'predictions_sha256':sha(out/'PREDICTIONS.json'),'groups':terminal['groups'],'policy':declared['policy'],
        'selected_other_count':113,'selected_output_count':118,
        'source_identity':declared['source_identity'],'predictor_sha256':build['binary']['sha256'],
        'score_script_sha256':sha(Path(__file__)),'original_model_sha256':declared['model']['sha256'],
        'training_max_other_bit_equal':terminal['training_max_other_bit_equal'],
        'train_no_new_event':terminal['train_no_new_event'],
        'development_complete_event_retention':terminal['development_complete_event_retention'],
        'all_final_predictions_saved_before_scoring':True,'all_claims_completed':terminal['claims_created']==260,
        'head_recordings':130,'head_frames':12255,'detector_replays':130,'wall_seconds':terminal['wall_seconds'],
        'training_updates':0,'backbone_calls':0,'pcm_calls':0,'unknown_records_opened':0,
        'default_r1_changed':False,'model_promotion_authorized':False,'original_fp32_numeric_gate':'FAIL_UNCHANGED',
        'independent_far_frr':'NOT_RUN','physical_ssc305':'NOT_RUN',
        'scope':'Fixed train-selected original-row competition on exposed cached teacher embeddings; actual compact SDK/PCM execution is a separate later gate.'})
    print(json.dumps({'status':terminal['status'],'groups':terminal['groups'],'head_recordings':130,'detector_replays':130,
        'aggregate_sha256':sha(out/'AGGREGATE.json'),'training_updates':0,'backbone_calls':0,'pcm_calls':0},sort_keys=True))
if __name__=='__main__':main()
