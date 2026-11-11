#!/usr/bin/env python3
"""Audit the saved ACTIVE113 screen, without any head/backbone/detector call."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import time

PRECISE = (0,1027,1401,1462,2093)
SELECTION_SHA = '8289f2ca46e40717747c1ed8d08152d8db2fbadcac5f4a84fdbb20d06375dc42'
DECLARED_SHA = '86cd4dccfb35bce57739a68a8b6170b7ff52e732cf7ef36d9715eba36e5753bf'

def sha(path):
    if not path.is_file() or path.is_symlink(): raise ValueError('regular unlinked file required')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path): return json.loads(path.read_text())

def category(token):
    return 'blank' if token == 0 else 'precise_'+str(token) if token in PRECISE else 'OTHER'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); root = args.root; run = root/'execution'
    if args.output.exists(): raise ValueError('fresh audit output required')
    checks = []
    def need(value,reason):
        if not value: raise ValueError(reason)
        checks.append(reason)
    declared = load(root/'PLAN.json'); selection = load(root/'SELECTION.json')
    plan = load(run/'PLAN.json'); pred = load(run/'PREDICTIONS.json')
    terminal = load(run/'TERMINAL.json'); aggregate = load(run/'AGGREGATE.json')
    build = load(root/'build/BUILD_RECEIPT.json')
    need(sha(root/'SELECTION.json') == SELECTION_SHA == declared['selection_sha256'] == pred['selection_sha256'] == aggregate['selection_sha256'],'immutable train-only selection')
    need(sha(root/'PLAN.json') == DECLARED_SHA == plan['declared_plan_sha256'] == pred['declared_plan_sha256'] == aggregate['declared_plan_sha256'] == build['declared_plan_sha256'],'fixed declared plan')
    need(sha(run/'PLAN.json') == aggregate['execution_plan_sha256'] and sha(run/'PREDICTIONS.json') == aggregate['predictions_sha256'] and sha(run/'TERMINAL.json') == aggregate['terminal_sha256'],'execution evidence bindings')
    need(sha(root/'build/BUILD_RECEIPT.json') == plan['build_receipt_sha256'],'build receipt binding')
    for name,descriptor in declared['source_files'].items():
        path = Path(name)
        need(sha(path) == descriptor['sha256'] and path.stat().st_size == descriptor['bytes'],'frozen source '+path.name)
    for name,digest in declared['data_pins'].items(): need(sha(Path(name)) == digest,'frozen data metadata')
    for descriptor in (declared['model'],declared['detector_binary'],build['binary']):
        path = Path(descriptor['path'])
        need(sha(path) == descriptor['sha256'] and path.stat().st_size == descriptor['bytes'],'model or executable identity')
    need(build['source_files'] == declared['source_files'] and build['compiler'] == declared['compiler'],'build source/compiler declaration')
    need(sha(root/'run_screen.py') == aggregate['score_script_sha256'],'executed runner identity')
    need(aggregate['policy'] == plan['policy'] == declared['policy'] and declared['policy']['between_precise_and_other_tie'] == 'lowest_original_id' and declared['policy']['within_other_tie'] == 'lowest_original_id' and declared['policy']['detector_other_token'] == 'actual_original_winner_id' and not declared['policy']['sentinel_mapping'],'original-ID semantic policy')
    need(terminal['head_calls_completed'] == 130 and terminal['head_frames_completed'] == 12255 and terminal['detector_calls_completed'] == 130 and terminal['claims_created'] == 260 and terminal['wall_seconds'] <= 600,'bounded completed call denominator')
    need(pred['all_predictions_persisted_before_scoring'] and pred['head_recordings'] == 130 and pred['frames'] == 12255,'all saved predictions precede scoring')
    inputs = Path(declared['inputs_root']); baseline = Path(declared['saved_trace_root'])
    observations = {r['source_id']:r for r in map(json.loads,(inputs/'original-observations.jsonl').read_text().splitlines())}
    manifest = load(inputs/'INPUT_MANIFEST.json')['records']
    original = {r['index']:r for r in manifest}; eligibility = {}
    for source in manifest:
        row = observations[source['source_id']]
        need(row['split'] == source['split'] and row['pcm_sha256'] == source['pcm_sha256'],'original label join')
        if row['ctc_weight'] == 0: continue
        need(row['ctc_weight'] in (0.5,1.0) and row['evaluation_status'] == 'evaluated','known input eligibility')
        eligibility[source['index']] = row
    known = list(range(96))+list(range(100,134))
    need(sorted(eligibility) == known and [r['index'] for r in plan['records']] == known and [r['index'] for r in pred['records']] == known,'known130 membership and unknown exclusion')
    selected = set(selection['selected_other_ids'])
    need(len(selected) == 113 and selection['selected_other_ids'] == sorted(selected) and not selected.intersection(PRECISE),'fixed113 candidate classes')
    claims = sorted((run/'claims').glob('*.json'))
    need(len(claims) == 260 and {p.name for p in claims} == {f'{kind}-{i:03d}.json' for i in known for kind in ['head','detector']},'unique260 call claims')
    need(len(list((run/'predictions').glob('*.r1a'))) == 130 and len(list((run/'events').glob('*.txt'))) == 130,'retained prediction/event inventory')
    need(all((run/'logs'/f'{kind}-{i:03d}.stderr').stat().st_size == 0 for i in known for kind in ['head','detector']),'empty child stderr')
    need(max((run/r['path']).stat().st_mtime_ns for r in pred['records']) < (run/'PREDICTIONS.json').stat().st_mtime_ns < min((run/'claims'/f'detector-{i:03d}.json').stat().st_mtime_ns for i in known),'local all-predictions-before-first-detector chronology')
    baseline_index = load(baseline/'FILE_MANIFEST.json'); by_index = {r['index']:r for r in terminal['completed']}
    groups = {}; private_differences = []
    input_header = struct.pack('<8s6I5IIQ',b'R1HD140\0',1,64,1200,140,5,2601,*PRECISE,1,0)
    for record,descriptor in zip(plan['records'],pred['records']):
        index = record['index']; split = record['split']; frames = record['frames']
        need(split == eligibility[index]['split'] and frames == descriptor['frames'] and descriptor['split'] == split,'record roles '+str(index))
        hidden_path = Path(record['hidden_path']); hidden = hidden_path.read_bytes()
        path = run/descriptor['path']; payload = path.read_bytes()
        need(sha(hidden_path) == record['hidden_sha256'] == descriptor['hidden_sha256'] and len(hidden) == 64+frames*1200 and hidden[:64] == input_header,'cached teacher identity '+str(index))
        need(sha(path) == descriptor['sha256'] and len(payload) == descriptor['bytes'] == 64+frames*40,'saved candidate identity '+str(index))
        expected_header = struct.pack('<8s6I5IIQ',b'R1ACTV1\0',1,64,40,140,5,2601,*PRECISE,113,frames)
        need(payload[:64] == expected_header,'saved candidate layout '+str(index))
        head_receipt_path = run/'receipts'/f'{index:03d}.json'; head = load(head_receipt_path)
        need(sha(head_receipt_path) == descriptor['receipt_sha256'] and head['status'] == 'PREDICTED_SELECTED_ORIGINAL_HEAD' and head['frames'] == frames and head['precise_logits_bit_equal'] == frames*5 and head['selection_sha256'] == SELECTION_SHA and head['hidden_sha256'] == descriptor['hidden_sha256'],'head receipt '+str(index))
        for kind in ('head','detector'):
            claim = load(run/'claims'/f'{kind}-{index:03d}.json')
            need(claim['kind'] == kind and claim['index'] == index and claim['attempt'] == 1 and claim['selection_sha256'] == SELECTION_SHA and 0 < claim['child_limit_seconds'] <= 10,'single fixed claim '+kind+str(index))
            if kind == 'head':
                need(claim['argv'] == [build['binary']['path'],declared['model']['path'],record['hidden_path'],record['hidden_sha256'],str(path),split],'head arguments '+str(index))
            else:
                need(claim['argv'] == [declared['detector_binary']['path'],'original',str(run/'tokens'/f'{index:03d}.txt')] and claim['binding']['prediction_sha256'] == descriptor['sha256'] and claim['binding']['predictions_manifest_sha256'] == sha(run/'PREDICTIONS.json'),'detector arguments '+str(index))
        changed = Counter(); coarse = Counter(); tokens = []
        equal = covered = original_changes = ties = teacher_ties = teacher_other = covered_teacher_other = 0
        errors = []
        for frame in range(frames):
            at = 64+1200*frame
            f,source,delivery,original_id,full_other_id = struct.unpack_from('<QQQII',hidden,at)
            exact = struct.unpack_from('<5d',hidden,at+1152); full_other = struct.unpack_from('<d',hidden,at+1192)[0]
            pf,ps,pd,other_id,reserved,other = struct.unpack_from('<QQQIId',payload,64+40*frame)
            need((pf,ps,pd) == (f,source,delivery) and f == frame and source == 720+480*frame and delivery == min(((source+319)//320)*320,original[index]['samples']) and other_id in selected and reserved == 0 and all(math.isfinite(x) for x in (*exact,full_other,other)) and other <= full_other,'saved frame invariant '+str(index)+':'+str(frame))
            k = max(range(5),key=lambda n:exact[n]); precise_id = PRECISE[k]; maximum = exact[k]
            full_winner = full_other_id if full_other > maximum or (full_other == maximum and full_other_id < precise_id) else precise_id
            winner = other_id if other > maximum or (other == maximum and other_id < precise_id) else precise_id
            need(full_winner == original_id,'original winner reconstruction '+str(index)+':'+str(frame))
            bit_equal = payload[64+40*frame+32:64+40*frame+40] == hidden[at+1192:at+1200]
            has = full_other_id in selected
            need(not has or (bit_equal and other_id == full_other_id),'covered teacher maximum '+str(index)+':'+str(frame))
            if split == 'train': need(has and bit_equal and winner == original_id,'train coverage and exact argmax '+str(index)+':'+str(frame))
            equal += bit_equal; covered += has; original_changes += winner != original_id
            ties += other == maximum; teacher_ties += full_other == maximum
            if original_id not in PRECISE: teacher_other += 1; covered_teacher_other += has
            before,after = category(original_id),category(winner)
            if before != after: changed[before+'->'+after] += 1
            before = 'KEYWORD' if original_id in PRECISE[1:] else 'BLANK' if original_id == 0 else 'OTHER'
            after = 'KEYWORD' if winner in PRECISE[1:] else 'BLANK' if winner == 0 else 'OTHER'
            if before != after: coarse[before+'->'+after] += 1
            errors.append(full_other-other); tokens.append(f'F,{frame},{source},{delivery},{winner}\n'.encode())
        need((run/'tokens'/f'{index:03d}.txt').read_bytes() == b''.join(tokens),'saved detector tokens '+str(index))
        need(head['max_other_bit_equal_frames'] == equal and head['true_other_winner_covered_frames'] == covered,'head equality receipt '+str(index))
        trace_name = f'private-traces/{index:03d}-arm_par4.txt'; base = baseline/trace_name
        need(sha(base) == record['baseline_sha256'] == baseline_index[trace_name]['sha256'],'baseline trace identity '+str(index))
        baseline_bytes = base.read_bytes(); baseline_frames = b''.join(line for line in baseline_bytes.splitlines(keepends=True) if line.startswith(b'F,'))
        expected = b''.join(line for line in baseline_bytes.splitlines(keepends=True) if line.startswith(b'E,'))
        actual_path = run/'events'/f'{index:03d}.txt'; actual = actual_path.read_bytes()
        added = Counter(actual.splitlines())-Counter(expected.splitlines()); missing = Counter(expected.splitlines())-Counter(actual.splitlines())
        refs = {x['keyword_id']:x['reference'] for x in eligibility[index]['keywords'] if x['known']}
        wrong = sum(refs[int(line.split(b',')[2])] == 0 for line in actual.splitlines())
        added_wrong = sum(count for line,count in added.items() if refs[int(line.split(b',')[2])] == 0)
        result = {'index':index,'split':split,'frames':frames,'events_exact':actual == expected,'actual_events':len(actual.splitlines()),'reference_events':len(expected.splitlines()),'new_events':sum(added.values()),'missing_events':sum(missing.values()),'wrong_keyword_events_machine_weak':wrong,'added_wrong_keyword_events_machine_weak':added_wrong,'max_other_bit_equal_frames':equal,'true_other_winner_covered_frames':covered,'teacher_overall_other_frames':teacher_other,'covered_teacher_overall_other_frames':covered_teacher_other,'original_token_id_changes':original_changes,'frame_category_changes':sum(changed.values()),'changed_categories':dict(changed),'coarse_category_changes':dict(coarse),'precise_other_tie_frames':ties,'teacher_precise_other_tie_frames':teacher_ties,'prediction_sha256':descriptor['sha256'],'events_sha256':sha(actual_path)}
        need(result == by_index[index] == load(run/'receipts'/f'score-{index:03d}.json'),'complete saved score row '+str(index))
        if original_changes == 0: need(baseline_frames == b''.join(tokens),'full F-byte parity '+str(index))
        if actual != expected: private_differences.append({'index':index,'added':sum(added.values()),'missing':sum(missing.values())})
        group = groups.setdefault(split,{'rows':[],'errors':[]}); group['rows'].append(result);group['errors'].extend(errors)
    summaries = {}
    summable = ('frames','actual_events','reference_events','new_events','missing_events','wrong_keyword_events_machine_weak','added_wrong_keyword_events_machine_weak','max_other_bit_equal_frames','true_other_winner_covered_frames','teacher_overall_other_frames','covered_teacher_overall_other_frames','original_token_id_changes','frame_category_changes','precise_other_tie_frames','teacher_precise_other_tie_frames')
    for split,group in groups.items():
        rows,errors = group['rows'],group['errors']; summary = {k:sum(r[k] for r in rows) for k in summable}
        summary.update(records=len(rows),event_exact_records=sum(r['events_exact'] for r in rows),max_other_underestimate_mean=sum(errors)/len(errors),max_other_underestimate_max=max(errors))
        for name in ('changed_categories','coarse_category_changes'):
            counter = Counter()
            for row in rows: counter.update(row[name])
            summary[name] = dict(counter)
        summary['true_other_winner_coverage_fraction'] = summary['true_other_winner_covered_frames']/summary['frames']
        need(summary == aggregate['groups'][split] == terminal['groups'][split],split+' aggregate identity')
        summaries[split] = summary
    train,dev = summaries['train'],summaries['development']
    train_pass = train['records'] == 96 and train['max_other_bit_equal_frames'] == 9079 and train['original_token_id_changes'] == 0 and train['new_events'] == 0 and train['added_wrong_keyword_events_machine_weak'] == 0
    dev_pass = dev['records'] == 34 and dev['event_exact_records'] == 34
    expected_status = 'PASS_RETAINED_EVENT_SCREEN_NOT_PROMOTED' if train_pass and dev_pass else 'REJECTED_FIXED_RESEARCH_SCREEN'
    need(terminal['status'] == aggregate['status'] == expected_status and aggregate['development_complete_event_retention'] == dev_pass and aggregate['train_no_new_event'] == train_pass,'predeclared outcome gate')
    need(not aggregate['default_r1_changed'] and not aggregate['model_promotion_authorized'] and aggregate['original_fp32_numeric_gate'] == 'FAIL_UNCHANGED' and aggregate['independent_far_frr'] == aggregate['physical_ssc305'] == 'NOT_RUN','qualification limits unchanged')
    report = {'schema':'active113-independent-saved-screen-audit-v1','status':'PASS_SAVED_ACTIVE113_SCREEN_AUDIT',
        'candidate_screen_status':expected_status,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'audit_source_sha256':sha(Path(__file__)),'selection_sha256':SELECTION_SHA,'declared_plan_sha256':DECLARED_SHA,
        'execution_plan_sha256':sha(run/'PLAN.json'),'build_receipt_sha256':sha(root/'build/BUILD_RECEIPT.json'),
        'predictions_manifest_sha256':sha(run/'PREDICTIONS.json'),'terminal_sha256':sha(run/'TERMINAL.json'),
        'aggregate_sha256':sha(run/'AGGREGATE.json'),'individual_assertions_passed':len(checks),
        'prediction_files_verified':130,'frame_records_verified':12255,'event_files_verified':130,'unique_claims_verified':260,
        'all_saved_predictions_precede_scoring':True,'groups':summaries,'private_event_differences':private_differences,
        'new_head_or_backbone_calls':0,'new_detector_calls':0,'new_training_updates':0,'pcm_reads':0,
        'default_r1_changed':False,'model_promotion_authorized':False,
        'limits':['Audit recomputes saved decisions/metrics and verifies source/byte/claim bindings; it does not repeat head arithmetic.','Cached exposed synthetic screen is not independent FAR/FRR or full probability/CTC equivalence.','The17 development max-OTHER underestimates must remain reported even with unchanged original argmax/event traces.','No compact SDK PCM, physical SSC305 performance or deployment evidence is created by this audit.']}
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'path':str(args.output),'sha256':sha(args.output),**report},sort_keys=True))

if __name__ == '__main__':main()
