#!/usr/bin/env python3
"""Independent saved-artifact audit; never trains or executes neural/detector code."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np

IDS = np.array([0,1027,1401,1462,2093],dtype=np.uint32)
DTYPE = np.dtype([('frame','<u8'),('source','<u8'),('delivery','<u8'),
    ('original','<u4'),('other_id','<u4'),('hidden','<f8',(140,)),
    ('logits','<f8',(5,)),('other','<f8')])
HEADER = struct.pack('<8s6I5IIQ',b'R1HD140\0',1,64,1200,140,5,2601,*map(int,IDS),1,0)

def sha(path):
    if not path.is_file() or path.is_symlink(): raise ValueError('regular unlinked file required')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def metric(error):
    error = np.abs(error)
    return {'mae':float(error.mean()),'p95_absolute_error':float(np.quantile(error,.95)),
            'max_absolute_error':float(error.max()),'frames':len(error)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','teacher','inputs','baseline','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    root, teacher, inputs, baseline = args.candidate,args.teacher,args.inputs,args.baseline
    run, score = root/'fit',root/'fit/score'
    if args.output.exists(): raise ValueError('fresh audit output required')
    checks = []
    def need(value,reason):
        if not value: raise ValueError(reason)
        checks.append(reason)
    pre = read(root/'PLAN.json'); plan = read(run/'PLAN.json')
    terminal = read(run/'FIT_TERMINAL.json'); predictions = read(run/'PREDICTIONS.json')
    setup = read(run/'TRAIN_SETUP.json'); aggregate = read(score/'AGGREGATE.json')
    score_plan = read(score/'PLAN.json'); score_terminal = read(score/'TERMINAL.json')
    extraction = read(teacher/'extraction/RESULT.json')
    need(sha(root/'PLAN.json') == plan['preregistered_plan_sha256'] == aggregate['preregistered_plan_sha256'] == score_plan['preregistered_plan_sha256'],'preregistration identity')
    need(pre['config'] == plan['config'] == setup['config'],'fixed config unchanged')
    need(sha(root/'fit_margin.py') == pre['candidate_source_sha256'] == plan['script_sha256'] == sha(run/'source/fit_margin.py'),'source identity')
    need(sha(run/'source/PREREGISTRATION.json') == sha(root/'PLAN.json'),'retained preregistration')
    need(sha(run/'PLAN.json') == terminal['plan_sha256'] == aggregate['fit_plan_sha256'] == score_plan['training_plan_sha256'],'execution plan identity')
    need(sha(teacher/'extraction/RESULT.json') == pre['extraction_result_sha256'] == plan['extraction_result_sha256'] == predictions['extraction_result_sha256'],'fixed teacher cache')
    need(sha(teacher/'extraction/PLAN.json') == pre['extraction_plan_sha256'],'teacher extraction plan')
    need(sha(run/'PREDICTIONS.json') == aggregate['predictions_sha256'] == score_plan['predictions_sha256'],'final predictions manifest')
    need(sha(run/'checkpoint.npz') == terminal['checkpoint']['sha256'] == predictions['checkpoint']['sha256'] == aggregate['checkpoint_sha256'],'checkpoint identity')
    need(sha(score/'TERMINAL.json') == aggregate['score_terminal_sha256'],'score terminal identity')
    need(terminal['status'] == 'PASS_FIXED_FIT_AND_PREDICTIONS_NOT_PROMOTED' and terminal['optimizer_updates_completed'] == 2000 and terminal['final_predictions_completed'] == 130 and terminal['fit_seconds'] <= 600,'fixed successful single fit')
    need(score_plan['new_head_or_backbone_forwards'] == 0 and score_plan['maximum_detector_replays'] == 130 and not score_plan['calibration_or_threshold_changes'],'fixed saved scoring')
    for relative,digest in pre['prior_rejected_run_sha256'].items():
        need(sha(teacher/relative) == digest,'prior rejected evidence unchanged: '+relative)
    need(read(teacher/'fit/score/AGGREGATE.json')['status'] == 'REJECTED_FIXED_RESEARCH_SCREEN','prior rejection retained')
    gradient = read(run/'GRADIENT_CHECK.json')
    need(gradient['status'] == 'PASS_INVENTED_ARRAY_GRADIENT_CHECK' and gradient['parameters_checked'] == 11 and gradient['teacher_constants_held_fixed'] and not gradient['real_model_or_audio_used'],'recorded artificial gradient check')
    original = read(inputs/'INPUT_MANIFEST.json')['records']
    observations = {x['source_id']:x for x in map(json.loads,(inputs/'original-observations.jsonl').read_text().splitlines())}
    need(sha(inputs/'INPUT_MANIFEST.json') == plan['input_manifest_sha256'] and sha(inputs/'original-observations.jsonl') == plan['original_observations_sha256'],'eligibility sources')
    baseline_index = read(baseline/'FILE_MANIFEST.json')
    need(sha(baseline/'FILE_MANIFEST.json') == plan['saved_trace_manifest_sha256'],'baseline index')
    descriptors = {r['index']:r for r in extraction['records']}
    rows, train, dev, unknown = {},[],[],[]
    for source in original:
        index = source['index']; observation = observations[source['source_id']]
        split = 'unknown' if observation['ctc_weight'] == 0 else observation['split']
        if split == 'unknown': unknown.append(index); continue
        (train if split == 'train' else dev).append(index)
        desc = descriptors[index]; path = teacher/'extraction'/desc['path']; raw = path.read_bytes()
        need(sha(path) == desc['sha256'] and len(raw) == desc['bytes'] and raw[:64] == HEADER and DTYPE.itemsize == 1200,'cached record '+str(index))
        array = np.frombuffer(raw,dtype=DTYPE,offset=64)
        need(len(array) == source['feature_shape'][0] == desc['frames'],'frame denominator '+str(index))
        need(np.isfinite(array['hidden']).all() and np.isfinite(array['logits']).all() and np.isfinite(array['other']).all(),'finite teacher values '+str(index))
        trace_name = f'private-traces/{index:03d}-arm_par4.txt'; trace = baseline/trace_name
        need(sha(trace) == baseline_index[trace_name]['sha256'],'baseline trace '+str(index))
        rows[index] = {'split':split,'array':array,'observation':observation,
            'baseline':b''.join(line for line in trace.read_bytes().splitlines(keepends=True) if line.startswith(b'E,')),
            'hidden_sha256':desc['sha256']}
    need(train == list(range(96)) and dev == list(range(100,134)) and unknown == [96,97,98,99],'original eligibility memberships')
    x = np.concatenate([rows[i]['array']['hidden'] for i in train])
    raw_y = np.concatenate([rows[i]['array']['other'] for i in train])
    raw_p = np.concatenate([rows[i]['array']['logits'].max(axis=1) for i in train])
    winner = np.concatenate([rows[i]['array']['original'] for i in train])
    lengths = np.array([len(rows[i]['array']) for i in train],dtype=np.int64)
    starts = np.concatenate(([0],np.cumsum(lengths)[:-1]))
    category = np.where(winner == 0,0,np.where(np.isin(winner,IDS[1:]),1,2)).astype(np.int8)
    sign = np.where(np.isin(winner,IDS),1.,-1.)
    mean_y = float(raw_y.mean()); scale_y = max(float(raw_y.std()),1e-6)
    constants = {'x_mean':x.mean(axis=0),'x_scale':np.maximum(x.std(axis=0),1e-6),
        'y_mean':np.array([mean_y]),'y_scale':np.array([scale_y]),
        'teacher_other_standardized':(raw_y-mean_y)/scale_y,
        'teacher_max_precise_standardized':(raw_p-mean_y)/scale_y,
        'winner_sign':sign,'required_margin_standardized':np.minimum(np.abs(raw_p-raw_y)/scale_y,1.),
        'frame_category':category,'frame_clip_index':np.repeat(np.arange(96,dtype=np.int64),lengths)}
    need(x.shape == (9079,140) and np.bincount(category,minlength=3).tolist() == [7904,616,559],'train-only category denominator')
    need(np.count_nonzero(raw_p == raw_y) == 0 and np.all(sign*(raw_p-raw_y)>0),'teacher sign and tie policy')
    need(sha(run/'TRAIN_CONSTANTS.npz') == terminal['train_constants']['sha256'] == setup['train_constants']['sha256'],'train constants identity')
    with np.load(run/'TRAIN_CONSTANTS.npz',allow_pickle=False) as saved:
        need(set(saved.files) == set(constants),'train constant inventory')
        for key,value in constants.items(): need(np.array_equal(saved[key],value),key+' recomputed from train only')
    pools = []
    descriptions = {}
    for kind,name in enumerate(['blank','keyword','other']):
        clips, frame_lists = [],[]
        for clip,index in enumerate(train):
            c = category[starts[clip]:starts[clip]+lengths[clip]]
            indices = np.flatnonzero(c == kind)+starts[clip]
            if len(indices): clips.append(clip); frame_lists.append(indices)
        counts = np.array([len(v) for v in frame_lists],dtype=np.int64)
        pool = {'clips':np.array(clips,dtype=np.int64),'counts':counts,
                'starts':np.concatenate(([0],np.cumsum(counts)[:-1])),'frames':np.concatenate(frame_lists)}
        pools.append(pool); descriptions[name] = {'eligible_train_clips':len(clips),'eligible_train_frames':len(pool['frames'])}
    need(descriptions == setup['category_pools'],'category-clip pools')
    with np.load(run/'checkpoint.npz',allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint['metadata_json']))
        need(metadata['config'] == plan['config'] and metadata['plan_sha256'] == sha(run/'PLAN.json') and metadata['preregistered_plan_sha256'] == sha(root/'PLAN.json') and metadata['train_constants_sha256'] == sha(run/'TRAIN_CONSTANTS.npz') and metadata['train_setup_sha256'] == sha(run/'TRAIN_SETUP.json'),'checkpoint metadata bindings')
        for key in ('x_mean','x_scale','y_mean','y_scale'): need(np.array_equal(checkpoint[key],constants[key]),'checkpoint '+key)
        for key,shape in [('w1',(140,64)),('b1',(64,)),('w2',(64,)),('b2',(1,))]:
            for prefix in ('','adam_m_','adam_v_'):
                value = checkpoint[prefix+key]
                need(value.shape == shape and np.isfinite(value).all(),prefix+key+' retained')
        expected_clip = checkpoint['train_clip_sample_counts'].copy()
        expected_category = checkpoint['train_category_sample_counts'].copy()
        expected_joint = checkpoint['train_category_clip_sample_counts'].copy()
        need(np.array_equal(checkpoint['precise_ids'],IDS),'precise output identities')
    logs = [json.loads(line) for line in (run/'loss.jsonl').read_text().splitlines()]
    need(len(logs) == 2000 and [v['update'] for v in logs] == list(range(1,2001)),'2000 contiguous optimizer ledger rows')
    rng = np.random.default_rng(20261111)
    rng.normal(0,math.sqrt(2/140),(140,64)); rng.normal(0,math.sqrt(1/64),64)
    exposure_clip = np.zeros(96,dtype=np.int64); exposure_category = np.zeros(3,dtype=np.int64)
    exposure_joint = np.zeros((3,96),dtype=np.int64)
    for value in logs:
        chosen_category = rng.integers(0,3,size=256)
        selected = np.empty(256,dtype=np.int64); chosen_clip = np.empty(256,dtype=np.int64)
        for kind,pool in enumerate(pools):
            locations = np.flatnonzero(chosen_category == kind)
            if not len(locations): continue
            choices = rng.integers(0,len(pool['clips']),size=len(locations))
            offsets = rng.integers(0,pool['counts'][choices])
            selected[locations] = pool['frames'][pool['starts'][choices]+offsets]
            chosen_clip[locations] = pool['clips'][choices]
        need(hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest() == value['selected_train_frame_indices_sha256'] and hashlib.sha256(chosen_category.astype('<i8').tobytes()).hexdigest() == value['selected_categories_sha256'],'deterministic category/clip/frame sampling '+str(value['update']))
        need(np.array_equal(category[selected],chosen_category) and np.bincount(chosen_category,minlength=3).tolist() == value['batch_category_counts'],'batch category membership '+str(value['update']))
        need(all(math.isfinite(value[k]) and value[k] >= 0 for k in ['total_loss_standardized','smooth_l1_standardized','margin_penalty_standardized']) and abs(value['total_loss_standardized']-value['smooth_l1_standardized']-value['margin_penalty_standardized']) < 1e-10 and 0 <= value['margin_violating_frames'] <= 256,'finite loss components '+str(value['update']))
        exposure_clip += np.bincount(chosen_clip,minlength=96)
        exposure_category += np.bincount(chosen_category,minlength=3)
        for kind in range(3): exposure_joint[kind] += np.bincount(chosen_clip[chosen_category == kind],minlength=96)
    need(np.array_equal(exposure_clip,expected_clip) and np.array_equal(exposure_category,expected_category) and np.array_equal(exposure_joint,expected_joint) and int(exposure_clip.sum()) == 512000,'checkpoint sample exposure ledger')
    need(rng.bit_generator.state == metadata['rng_state'],'final RNG state')
    known = train+dev
    need([r['index'] for r in predictions['records']] == known and len(list((run/'predictions').glob('*.npy'))) == 130 and len(score_terminal['completed']) == 130,'130 known predictions and scores')
    need((run/'checkpoint.npz').stat().st_mtime_ns < (run/'PREDICTIONS.json').stat().st_mtime_ns < (score/'PLAN.json').stat().st_mtime_ns,'local checkpoint/prediction/score chronology')
    scored = {r['index']:r for r in score_terminal['completed']}; groups = {}; changes = []
    for descriptor in predictions['records']:
        index = descriptor['index']; row = rows[index]; a = row['array']; path = run/descriptor['path']
        need(sha(path) == descriptor['sha256'] and path.stat().st_size == descriptor['bytes'] and descriptor['hidden_sha256'] == row['hidden_sha256'],'prediction bytes '+str(index))
        other = np.load(path,allow_pickle=False)
        need(other.dtype == np.float64 and other.shape == (len(a),) and np.isfinite(other).all(),'prediction shape '+str(index))
        precise = IDS[np.argmax(a['logits'],axis=1)]; maximum = a['logits'].max(axis=1)
        candidate = np.where(maximum>=other,precise,1)
        reference = np.where(np.isin(a['original'],IDS),a['original'],1)
        token_text = ''.join(f"F,{int(v['frame'])},{int(v['source'])},{int(v['delivery'])},{int(c)}\n" for v,c in zip(a,candidate))
        need((score/'tokens'/f'{index:03d}.txt').read_text() == token_text,'saved category decision '+str(index))
        events_path = score/'events'/f'{index:03d}.txt'; actual = events_path.read_bytes(); expected = row['baseline']
        new = Counter(actual.splitlines())-Counter(expected.splitlines()); missing = Counter(expected.splitlines())-Counter(actual.splitlines())
        labels = {k['keyword_id']:k['reference'] for k in row['observation']['keywords'] if k['known']}
        wrong = sum(labels[int(line.split(b',')[2])] == 0 for line in actual.splitlines())
        added_wrong = sum(count for line,count in new.items() if labels[int(line.split(b',')[2])] == 0)
        result = scored[index]; error = other-a['other']; flips = int(np.count_nonzero(candidate!=reference))
        need(result['events_exact'] == (actual==expected) and result['new_events'] == sum(new.values()) and result['missing_events'] == sum(missing.values()) and result['wrong_keyword_events_machine_weak'] == wrong and result['added_wrong_keyword_events_machine_weak'] == added_wrong and result['events_sha256'] == sha(events_path),'event comparison '+str(index))
        need(result['frame_category_changes'] == flips and result['max_other_error'] == metric(error),'saved scalar/category metrics '+str(index))
        need((score/'logs'/f'{index:03d}.stderr').stat().st_size == 0 and read(score/'claims'/f'{index:03d}.json')['attempt'] == 1,'single detector receipt '+str(index))
        group = groups.setdefault(row['split'],{'records':0,'frames':0,'event_exact_records':0,'new_events':0,'missing_events':0,'wrong_keyword_events_machine_weak':0,'added_wrong_keyword_events_machine_weak':0,'frame_category_changes':0,'actual_events':0,'reference_events':0,'errors':[]})
        group['records'] += 1; group['frames'] += len(a); group['event_exact_records'] += actual==expected
        group['new_events'] += sum(new.values()); group['missing_events'] += sum(missing.values())
        group['wrong_keyword_events_machine_weak'] += wrong; group['added_wrong_keyword_events_machine_weak'] += added_wrong
        group['frame_category_changes'] += flips; group['actual_events'] += len(actual.splitlines()); group['reference_events'] += len(expected.splitlines()); group['errors'].append(error)
        if actual != expected:
            changes.append({'index':index,'split':row['split'],'added_events':[{'line':k.decode(),'count':v} for k,v in new.items()],'missing_events':[{'line':k.decode(),'count':v} for k,v in missing.items()]})
    for split,group in groups.items():
        group['max_other_error'] = metric(np.concatenate(group.pop('errors')))
        for key in ['records','frames','event_exact_records','new_events','missing_events','wrong_keyword_events_machine_weak','added_wrong_keyword_events_machine_weak','frame_category_changes','max_other_error']:
            need(group[key] == aggregate['groups'][split][key],split+' aggregate '+key)
    dev_pass = groups['development']['event_exact_records'] == 34
    train_pass = groups['train']['new_events'] == 0 and groups['train']['added_wrong_keyword_events_machine_weak'] == 0
    expected_status = 'PASS_EXPOSED_DEVELOPMENT_SCREEN_NOT_PROMOTED' if dev_pass and train_pass else 'REJECTED_FIXED_RESEARCH_SCREEN'
    need(aggregate['status'] == score_terminal['status'] == expected_status and aggregate['development_exact_retention'] == dev_pass and aggregate['train_no_new_wrong_events'] == train_pass,'unchanged preregistered acceptance')
    need(not aggregate['model_promotion_authorized'] and not aggregate['default_r1_changed'] and aggregate['prior_uniform_clip_mlp64'] == 'REJECTED_FIXED_RESEARCH_SCREEN_UNCHANGED','no promotion or rewritten rejection')
    report = {'schema':'r1-margin-candidate-independent-audit-v1','status':'PASS_SAVED_MARGIN_CANDIDATE_AUDIT',
        'candidate_screen_status':expected_status,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'audit_source_sha256':sha(Path(__file__)),'fit_script_sha256':sha(root/'fit_margin.py'),
        'preregistered_plan_sha256':sha(root/'PLAN.json'),'fit_plan_sha256':sha(run/'PLAN.json'),
        'checkpoint_sha256':sha(run/'checkpoint.npz'),'train_constants_sha256':sha(run/'TRAIN_CONSTANTS.npz'),
        'predictions_manifest_sha256':sha(run/'PREDICTIONS.json'),'score_aggregate_sha256':sha(score/'AGGREGATE.json'),
        'individual_assertions_passed':len(checks),'new_training_or_neural_forwards':0,'new_detector_replays':0,
        'train_only_normalization_and_margin_constants_exact':True,'sampling_ledger_all2000_updates_verified':True,
        'sampled_train_frames':512000,'sampled_category_counts':exposure_category.tolist(),
        'saved_prediction_files_individually_verified':130,'saved_event_files_compared':130,
        'unknown_in_training_or_scoring':0,'prior_rejected_evidence_unchanged':True,
        'groups':groups,'private_changed_event_records':changes,'development_exact_retention':dev_pass,
        'train_no_added_or_wrong_events':train_pass,'model_promotion_authorized':False,
        'interpretation':'A pass establishes only this fixed exposed-development retention screen; a failure remains rejected. No independent FAR/FRR, full probability/CTC equivalence, physical SSC305 or component-isolated ablation claim.'}
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'path':str(args.output),'sha256':sha(args.output),**report},sort_keys=True))

if __name__ == '__main__': main()
