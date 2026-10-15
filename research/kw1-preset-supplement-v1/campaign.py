"""Host-only fixed-denominator orchestration. No model imports or retry path."""
from pathlib import Path
import time
from execution_common import require,write,sha,read
from plan_contract import validate_plan,normalize,coverage_only
RECOGNIZERS=('qwen_asr','sensevoice','whisper_small')

def report(plan,tts,mapping,asr,stopped):
    # Reconstructed revision: Python bool/int equality is not evidence typing.
    validate_plan(plan)
    require(type(tts) is list and type(mapping) is list and type(asr) is dict, 'typed report input collections')
    require(stopped is None or type(stopped) is str, 'typed terminal reason')
    require(len({r['slot_id'] for r in tts}) == len(tts), 'duplicate native slot')
    require(len({r['slot_id'] for r in mapping}) == len(mapping), 'duplicate mapping slot')
    require(set(asr) <= set(RECOGNIZERS), 'known recognizers only')
    planned_ids = {r['slot_id'] for r in plan['slots']}
    require(all(r['slot_id'] in planned_ids for r in tts + mapping), 'only planned native/mapping identities')
    bound_ids = {r['derived']['opaque_id'] for r in mapping}
    require(len(bound_ids) == len(mapping), 'unique mapped PCM identities')
    for native in tts:
        require(type(native) is dict and type(native.get('attempts')) is int and native['attempts'] in (0,1), 'strict native attempt integer')
        if type(native.get('metrics')) is dict and 'technically_valid' in native['metrics']:
            require(type(native['metrics']['technically_valid']) is bool, 'strict native validity boolean')
    for outcomes in asr.values():
        require(type(outcomes) is list, 'typed ASR collection')
        for outcome in outcomes:
            require(type(outcome) is dict and type(outcome.get('attempts')) is int and outcome['attempts'] in (0,1), 'strict ASR attempt integer')
            require(outcome.get('opaque_id') in bound_ids, 'ASR outcome must bind host PCM identity')
            if 'raw_text' in outcome:
                require(type(outcome['raw_text']) is str or (outcome['raw_text'] is None and outcome.get('status') not in ('success','SUCCESS')), 'ASR transcript text')
            if 'quality_flags' in outcome:
                require(type(outcome['quality_flags']) is list and all(type(x) is str for x in outcome['quality_flags']), 'ASR flags string list')
    by_id={r['slot_id']:r for r in tts}; by_map={r['slot_id']:r for r in mapping}; rows=[]
    for cell in plan['slots']:
        sid=cell['slot_id']; native=by_id.get(sid,dict(status='NOT_RUN',attempts=0)); bind=by_map.get(sid)
        recognized=[]
        for recognizer in RECOGNIZERS:
            matches=[r for r in asr.get(recognizer,[]) if bind and r.get('opaque_id')==bind['derived']['opaque_id']]
            require(len(matches)<=1,'duplicate ASR identity')
            row=matches[0] if matches else dict(status='NOT_RUN',attempts=0)
            recognized.append(dict(recognizer=recognizer,**{k:row[k] for k in ('status','attempts','opaque_id','pcm_sha256','wav_sha256','raw_text','completeness','quality_flags','error_type') if k in row}))
        native_valid=native.get('status')=='completed' and native.get('attempts')==1 and native.get('metrics',{}).get('technically_valid') is True
        eligible=bool(bind) and native_valid
        text_consistent=all(r.get('status')=='success' and r.get('completeness')=='complete' and r.get('quality_flags')==[] for r in recognized) and len({normalize(r.get('raw_text','')) for r in recognized})==1
        consensus=eligible and all(r.get('status')=='success' and r.get('attempts')==1 and r.get('completeness')=='complete' and r.get('quality_flags')==[] and normalize(r.get('raw_text',''))==normalize(cell['target_text']) for r in recognized)
        rows.append(dict(slot_id=sid,engine=cell['engine'],speaker=cell['speaker'],group_id=cell['group_id'],target_text=cell['target_text'],native_status=native['status'],native_attempts=native.get('attempts',0),native_sha256=native.get('raw_wav_sha256'),native_metadata=native.get('native'),native_metrics=native.get('metrics'),native_termination=native.get('termination'),generation_parameters=cell['generation_parameters'],seed=cell['seed'],instruction=cell['instruction'],audio_quality_pass=native_valid,derived_audio_ready=bool(bind),text_three_model_consistent=text_consistent,identity=bind,recognitions=recognized,
            label_tier='three-model-consensus' if consensus else 'review',machine_train_only_candidate=consensus,human_gold=False,training_admitted=False,calibration_gold=False,selection_gold=False))
    consumed_tts=sum(r['native_attempts'] for r in rows);consumed_asr=sum(r.get('attempts',0) for s in rows for r in s['recognitions'])
    require(consumed_tts<=16 and consumed_asr<=48,'campaign call budget')
    candidates=[r['slot_id'] for r in rows if r['machine_train_only_candidate']]
    return dict(schema='kw1-supplement-report-v1',status='stopped_no_retry' if stopped else 'completed',stopped_reason=stopped,
        denominator=dict(design_slots=16,authorized_slots=16,tts_call_slots=16,asr_call_slots=48),
        consumed_calls=dict(tts=consumed_tts,asr=consumed_asr),
        audio_quality_pass=sum(r['audio_quality_pass'] for r in rows),derived_audio_ready=sum(r['derived_audio_ready'] for r in rows),text_three_model_consistent=sum(r['text_three_model_consistent'] for r in rows),target_qualified=len(candidates),target_qualified_denominator=16,
        candidate_ids=candidates,group_coverage=coverage_only(candidates),
        admission_status='BLOCKED_NO_CALIBRATION_SELECTION_GOLD',training_admitted=False,human_gold=False,shipping=False,
        note='Three recognizer agreement yields machine train-only candidates. Group coverage is necessary, not sufficient; no calibration/selection gold or fresh heldout is created.',slots=rows)

def execute(plan,work,backend,derive_fn=None,recover_tts=None,recover_asr=None):
    validate_plan(plan);work=Path(work)
    require((work/'campaign.claim.json').is_file(),'admission claim required')
    if derive_fn is None:
        from derive_native import derive as derive_fn
    if recover_tts is None:
        from worker import recover as recover_tts
    if recover_asr is None:
        from blind_asr_worker import recover_outcomes as recover_asr
    tts=[];mapping=[];asr={};stopped=None;roots={};errors=[]
    try:
        for engine in ('qwen06','qwen17'):
            output=work/'tts'/engine/'output';roots[engine]=output
            try: backend.tts(engine,plan,output)
            finally:
                if output.exists():
                    try:
                        from public_evidence import validate_tts_archive,active
                        recovered=validate_tts_archive(output,engine,plan) if active(output) else recover_tts(output,engine,plan)
                    except Exception:
                        errors.append('unverified_tts_'+engine);raise
                    tts.extend(recovered['outcomes'] if isinstance(recovered,dict) else recovered)
            expected=[s for s in plan['slots'] if s['engine']==engine]
            got=[r for r in tts if r.get('engine')==engine]
            require(len(got)==len(expected) and all(r.get('status')=='completed' for r in got),'TTS stage incomplete')
            identities=[r['native']['source_values_sha256'] for r in tts]
            require(len(identities)==len(set(identities)),'duplicate native output across lanes')
        mapping=derive_fn(plan['slots'],tts,roots,work/'blind-input');write(work/'identity-map.json',mapping)
        expected_ids={r['derived']['opaque_id'] for r in mapping};require(len(expected_ids)==16,'fixed unique derived inputs')
        for recognizer in RECOGNIZERS:
            output=work/'asr'/recognizer/'output'
            try: backend.asr(recognizer,work/'blind-input',output)
            finally:
                if output.exists():
                    try:
                        from public_evidence import validate_asr_archive,active
                        recovered=validate_asr_archive(work/'blind-input',output,recognizer)[1] if active(output) else recover_asr(work/'blind-input',output,recognizer)
                    except Exception:
                        errors.append('unverified_asr_'+recognizer);raise
                    asr[recognizer]=recovered['outcomes'] if isinstance(recovered,dict) else recovered
                    asr[recognizer]=[dict(r,status='success' if r.get('status')=='SUCCESS' else r.get('status')) for r in asr[recognizer]]
            rows=asr.get(recognizer,[])
            require(len(rows)==16 and {r.get('opaque_id') for r in rows}==expected_ids,'ASR full identity set')
            require(all(r.get('status')=='success' and r.get('attempts')==1 and r.get('completeness')=='complete' and r.get('quality_flags')==[] for r in rows),'ASR incomplete/flagged stops campaign')
    except BaseException as error:
        stopped=type(error).__name__
    result=report(plan,tts,mapping,asr,stopped)
    mark_unknown(result,errors)
    result['resources']=backend.public_summary() if hasattr(backend,'public_summary') else {'schema':'mock-resource-summary','no_models_executed':True}
    if errors:result['resources']['unverified_stages']=errors
    write(work/'report.json',result)
    return result

def collect_only(plan,work,persist=True):
    """Rebuild derived state only; no claim, derivation, or model invocation."""
    from public_evidence import assess_tts_archive,assess_asr_archive,assess_host_inputs
    validate_plan(plan);work=Path(work);tts=[];asr={};errors=[]
    host=assess_host_inputs(work,plan)
    mapping=host['mapping'] if host['verified'] else []
    for engine in ('qwen06','qwen17'):
        assessment=assess_tts_archive(work/'tts'/engine/'output',engine,plan)
        if not assessment['verified']:errors.append('unverified_tts_'+engine)
        else:tts.extend(assessment['outcomes'])
    for recognizer in RECOGNIZERS:
        assessment=assess_asr_archive(work/'blind-input',work/'asr'/recognizer/'output',recognizer,host)
        if not assessment['verified']:errors.append('unverified_asr_'+recognizer)
        else:asr[recognizer]=[dict(r,status='success' if r['status']=='SUCCESS' else r['status']) for r in assessment['outcomes']]
    result=report(plan,tts,mapping,asr,'interrupted_or_preexecution_failure')
    mark_unknown(result,errors)
    result['resources']=dict(schema='kw1-recovery-only-v1',model_calls_made=0,unverified_stages=errors,
        counts_are_lower_bounds=bool(errors),all_unverified_budget_slots_reserved=True)
    if persist:
        # Derived reports may be rebuilt after interruption. Never reuse the
        # exclusive fixed .tmp filename used by immutable execution evidence.
        import tempfile,os
        from execution_common import canonical
        fd,name=tempfile.mkstemp(prefix='.kw1-report-',suffix='.tmp',dir=work)
        with os.fdopen(fd,'wb') as handle:
            handle.write(canonical(result)+b'\n');handle.flush();os.fsync(handle.fileno())
        os.replace(name,work/'report.json')
    return result


def mark_unknown(result,errors):
    """Unverifiable evidence never becomes zero calls or a free retry."""
    if not errors:return result
    known=dict(result['consumed_calls'])
    for label in errors:
        if label.startswith('unverified_tts_'):
            engine=label[len('unverified_tts_'):];result['consumed_calls']['tts']=None
            for row in result['slots']:
                if row['engine']==engine:
                    row.update(native_status='UNVERIFIED',native_attempts=None,audio_quality_pass=False,machine_train_only_candidate=False,label_tier='review')
        else:
            name=label[len('unverified_asr_'):];result['consumed_calls']['asr']=None
            for row in result['slots']:
                row.update(machine_train_only_candidate=False,label_tier='review',text_three_model_consistent=False)
                for recognition in row['recognitions']:
                    if recognition['recognizer']==name:recognition.update(status='UNVERIFIED',attempts=None)
    result['consumed_calls']['verified_lower_bound']=known
    result['consumed_calls']['unverifiable_attempts_reserved_no_retry']=True
    result['audio_quality_pass']=sum(r['audio_quality_pass'] for r in result['slots'])
    result['text_three_model_consistent']=sum(r['text_three_model_consistent'] for r in result['slots'])
    result['candidate_ids']=[r['slot_id'] for r in result['slots'] if r['machine_train_only_candidate']]
    result['target_qualified']=len(result['candidate_ids']);result['group_coverage']=coverage_only(result['candidate_ids'])
    return result
