#!/usr/bin/env python3
"""Small stdlib-only LR-stability evidence adapter. Never executes model or optimizer code."""
import argparse,hashlib,json,os,stat,types
from pathlib import Path
BASE_SHA='cf974939751107b8e6e6c15d34bff1e5830e173a1a45cb3c34d1642681d36bcc'
BASE=Path(__file__).absolute().parent.parent/'grouped-balance-result-verification/verify_result.py'
if BASE.resolve(strict=True)!=BASE or not stat.S_ISREG(BASE.stat().st_mode) or BASE.stat().st_nlink!=1:raise ValueError('unsafe shared validator')
raw=BASE.read_bytes()
if hashlib.sha256(raw).hexdigest()!=BASE_SHA:raise ValueError('shared validator pin mismatch')
v=types.ModuleType('frozen_grouped_verifier');v.__file__=str(BASE);exec(compile(raw,str(BASE),'exec'),v.__dict__)
CAMPAIGN='kws-lr-stability-20261016-v1'
HISTORY_SHA='4260c936d8249c59f2d5b270d40274b5f2b29bb45d930dd31c338b240d666b25'
HISTORY_COMMIT='f86ffb57234c1083c03d25c4ae89bfdbac2f733c'
EVENTS=v.forward_events()[:608]
ALLOWED={'learning-rate-transition.json','intent.json','binding.json','terminal.json','failure.json','A-initialization.json'}|{f'A-{u:04d}.json' for u in v.CHECKPOINTS}|{f'{k}-{i:04d}.json' for k in ('call','step') for i in range(1,601)}|{f'{k}-{i:04d}.json' for k in ('forward','forward-return','payload') for i in range(1,609)}

def historical_payload(actual,expected):
    return v.exact({k:x for k,x in actual.items() if k!='schema'},{k:x for k,x in expected.items() if k!='schema'})

def historical_step(actual,expected):
    ignored={'payload_sha256','learning_rate','historical_replay_matched'}
    return v.exact({k:x for k,x in actual.items() if k not in ignored},{k:x for k,x in expected.items() if k not in ignored})

def load_history(base):
    manifest=v.strict_json(v.pinned(Path(base)/'grouped-balance-result-acceptance/hash-manifest.json',HISTORY_SHA))
    needed={'A-initialization.json','A-0600.json'}|{f'payload-{i:04d}.json' for i,(a,k,u) in enumerate(EVENTS,1) if u<=500}|{f'step-{i:04d}.json' for i in range(1,501)}|{f'A-{u:04d}.json' for u in v.CHECKPOINTS if u<=500}
    history={n:v.strict_json(v.pinned(Path(base)/'grouped-balance-result'/n,manifest['result_files'][n])) for n in needed}
    for i,(arm,kind,u) in enumerate(EVENTS,1):
        if kind=='observation' and u<=500:v.same(history[f'A-{u:04d}.json'],history[f'payload-{i:04d}.json'],'historical_observation_payload_binding')
    return history

def verify_objects(data,plan,train,history,claim_raw,result_path,process,plan_sha,raws=None):
    v.require(type(data) is dict and set(data)<=ALLOWED,'unexpected_result_file')
    v.same(data.get('intent.json'),dict(schema='lr-stability-intent-v1',campaign_id=CAMPAIGN,max_optimizer_calls=600,max_model_forward_calls=608,no_resume=True,**v.DENIALS),'intent')
    v.same(v.strict_json(claim_raw),dict(schema='lr-stability-exclusive-claim-v1',campaign_id=CAMPAIGN,output=str(Path(result_path).absolute()),max_optimizer_calls=600,max_model_forward_calls=608,**v.DENIALS),'claim')
    def indices(prefix):return sorted(int(n[len(prefix)+1:-5]) for n in data if v.re.fullmatch(prefix+r'-[0-9]{4}\.json',n))
    calls=indices('call');steps=indices('step');forwards=indices('forward');returns=indices('forward-return');payloads=indices('payload')
    for label,seq in [('call',calls),('step',steps),('forward',forwards),('return',returns),('payload',payloads)]:v.same(seq,list(range(1,len(seq)+1)),label+'_gap')
    v.require(len(steps)<=len(calls)<=len(steps)+1,'optimizer_prefix')
    v.require(len(payloads)<=len(returns)<=len(forwards)<=len(returns)+1 and len(forwards)<=len(payloads)+1,'forward_prefix')
    covered=set();observations={};training_index={};mismatches=[];cache={};statistics={}
    for i in forwards:
        arm,kind,u=EVENTS[i-1];forward=dict(status='forward_reserved',stage='A',update=u,kind=kind,model_forward_reserved=i)
        v.same(data[f'forward-{i:04d}.json'],forward,'forward_event')
        if i in returns:v.same(data[f'forward-return-{i:04d}.json'],{**forward,'status':'forward_returned'},'return_event')
        receipt=f'step-{u:04d}.json' if kind=='training' else f'A-{u:04d}.json'
        if receipt in data:v.require(i in returns and i in payloads,'receipt_without_return_or_payload');covered.add(receipt)
        else:v.require(i==len(forwards),'continued_after_unfinished_event')
        if kind=='training':training_index[u]=i;v.require(u<=len(calls)+1,'forward_ahead_of_optimizer')
        else:v.require(u<=len(steps),'observation_before_update')
        if i in payloads:
            name=f'payload-{i:04d}.json';p=data[name];v.same(p.get('schema'),'lr-stability-forward-payload-v1','payload_schema')
            # The numerical core remains exactly the independently reviewed validator.
            summary=v.payload({**p,'schema':'grouped-forward-payload-v1'},'A',kind,u,i,train,[],cache)
            if kind=='observation':
                statistics[f'A-{u:04d}']=summary
                if receipt in data:v.same(data[receipt],p,'observation_payload');observations[receipt]=i
            if u<=500 and not historical_payload(p,history[name]):
                v.require(i==len(forwards) and receipt not in data and (kind!='training' or u not in calls),'continued_after_historical_payload_mismatch');mismatches.append(name)
    obs_names={n for n in data if v.re.fullmatch(r'A-[0-9]{4}\.json',n)};v.same(obs_names,set(observations),'observation_without_forward')
    for u in calls:
        v.same(data[f'call-{u:04d}.json'],dict(status='step_reserved',stage='A',update=u,optimizer_call_reserved=u),'call_event')
        v.require(u in training_index and training_index[u] in returns and training_index[u] in payloads,'optimizer_without_return')
    for u in steps:
        name=f'step-{u:04d}.json';v.require(name in covered,'step_without_forward');s=data[name];v.keys(s,v.STEP_KEYS|{'learning_rate','historical_replay_matched'})
        pn=f'payload-{training_index[u]:04d}.json'
        expected=dict(status='step_succeeded',stage='A',update=u,payload_file=pn,payload_sha256=v.sha(raws[pn] if raws else v.encoded(data[pn])),blank_lambda=1,updated_weights_finite=True,weights_changed=True,learning_rate=.003 if u<=500 else .0003,historical_replay_matched=u<=500)
        for k,value in expected.items():v.same(s[k],value,'step_'+k)
        for k in ('finite_loss','ctc_loss','anchor_loss'):v.same(s[k],data[pn][k],'step_payload_'+k)
        v.number(s['finite_grad_norm'])
        if u<=500:v.require(historical_step(s,history[name]),'historical_step_mismatch')
    if 'binding.json' in data:
        initial=history['A-initialization.json']['initial_weight_sha256'];b=data['binding.json']
        expected=dict(schema='lr-stability-input-binding-v1',plan_sha256=plan_sha,initial_weight_sha256=initial,registry_sha256=plan['registry_sha256'],cohort_sha256=plan['cohort_sha256'],runtime_lock_sha256=plan['runtime_lock_sha256'],historical_manifest_sha256=HISTORY_SHA,seed=20261015,threads={'intraop':1,'interop':1},campaign_id=CAMPAIGN,campaign_claim_path=plan['campaign_claim_path'],campaign_claim_sha256=v.sha(claim_raw),max_model_forward_calls=608,**v.DENIALS)
        v.same(b,expected,'binding')
    if 'A-initialization.json' in data:v.require('binding.json' in data,'init_without_binding');v.same(data['A-initialization.json'],history['A-initialization.json'],'same_initialization')
    if forwards:v.require('A-initialization.json' in data,'forward_without_init')
    if 'learning-rate-transition.json' in data:
        v.same(data['learning-rate-transition.json'],dict(update=501,previous_lr=.003,new_lr=.0003,optimizer_recreated=False,historical_replay_through_step=500),'lr_transition')
        v.require(len(steps)>=500 and 'A-0500.json' in observations and not mismatches,'lr_transition_before_replay_gate')
    if len(forwards)>=508:v.require('learning-rate-transition.json' in data,'missing_lr_transition')
    status='incomplete'
    if 'terminal.json' in data:
        v.require(len(calls)==len(steps)==600 and len(forwards)==len(returns)==len(payloads)==608 and len(observations)==8 and not mismatches and 'failure.json' not in data,'terminal_completeness')
        v.same(data['terminal.json'],dict(schema='lr-stability-terminal-v1',status='completed',primary_endpoint_step=600,train_endpoints={'A':statistics['A-0600']},historical_replay_through_step=500,historical_replay_observations=[0,50,100,200,300,400,500],optimizer_calls_reserved=600,model_forward_calls_reserved=608,probe_used_for_selection=False,**v.DENIALS),'terminal')
        status='completed'
    if 'failure.json' in data:
        f=data['failure.json'];v.keys(f,{'schema','status','stage','update','optimizer_calls_reserved','model_forward_calls_reserved','error_type','context','error'}|set(v.DENIALS));v.denials(f)
        for k,value in dict(schema='lr-stability-failure-v1',status='failed_no_retry',optimizer_calls_reserved=len(calls),model_forward_calls_reserved=len(forwards)).items():v.same(f[k],value,'failure_'+k)
        v.require(type(f['error_type']) is str and type(f['error']) is str and type(f['context']) is dict,'failure_types');v.integer(f['update'],0,600)
        if f['stage']=='preparation':v.require(not forwards and not calls and f['update']==0,'preparation_failure')
        else:
            v.same(f['stage'],'A','failure_arm');eligible=[EVENTS[len(forwards)-1]] if forwards else [EVENTS[0]]
            if forwards:
                a,k,u=eligible[0];receipt=f'step-{u:04d}.json' if k=='training' else f'A-{u:04d}.json'
                if receipt in covered and len(forwards)<608:eligible.append(EVENTS[len(forwards)])
            v.require(f['update'] in {u for a,k,u in eligible},'failure_position')
        ctx=f['context'];phase=ctx.get('phase')
        if phase=='historical_payload_comparison':
            v.keys(ctx,{'phase','historical_file','update'})
            v.same(ctx['update'],f['update'],'historical_failure_update')
            v.require(ctx['historical_file'] in mismatches,'false_historical_payload_failure')
        elif phase=='historical_gradient_comparison':
            v.keys(ctx,{'phase','historical_file','update','observed','expected'});u=f['update']
            v.same(ctx['update'],u,'historical_gradient_update');v.same(ctx['historical_file'],f'step-{u:04d}.json','historical_gradient_file')
            v.require(1<=u<=500 and u not in calls and training_index.get(u)==len(forwards) and len(payloads)==len(forwards),'historical_gradient_position')
            fields={'finite_grad_norm','finite_loss','ctc_loss','anchor_loss','blank_lambda'}
            v.keys(ctx['observed'],fields);v.keys(ctx['expected'],fields)
            expected={k:history[ctx['historical_file']][k] for k in fields};v.same(ctx['expected'],expected,'historical_expected_gradient')
            for k,x in ctx['observed'].items():v.number(x)
            for k in fields-{'finite_grad_norm'}:v.same(ctx['observed'][k],data[f'payload-{len(forwards):04d}.json'][k],'historical_observed_loss')
            v.require(not v.exact(ctx['observed'],expected),'false_historical_gradient_failure')
        elif phase=='forward':
            v.keys(ctx,{'phase','kind','record_ids'});v.same(ctx['record_ids'],[r['record_id'] for r in train],'failure_records')
            v.require(f['stage']=='A' and ('A',ctx['kind'],f['update']) in eligible,'failure_forward_event')
        elif phase=='loss':
            v.keys(ctx,{'phase','nonfinite_log_probs','ctc_nll'});v.integer(ctx['nonfinite_log_probs'],0,1000000)
            v.require(type(ctx['ctc_nll']) is list and len(ctx['ctc_nll'])==12,'failure_nll_count')
            for n in ctx['ctc_nll']:
                if n is not None:v.number(n,-1e100,1e100)
        elif phase in ('backward','optimizer_step','evaluation','optimizer_reservation'):v.keys(ctx,{'phase'})
        else:v.same(ctx,{},'failure_context')
        v.require(not mismatches or phase=='historical_payload_comparison','missing_historical_failure_context')
        v.same(f['error'],'Stopped on numerical, evidence, budget or output invariant; no retry or resume','failure_error')
        status='failed_partial'
    v.require(not mismatches or 'failure.json' in data,'unexplained_historical_mismatch')
    v.require(type(process.get('exit_code')) is int and type(process.get('wall_timeout')) is bool,'process_types')
    if process['exit_code']!=0 or process['wall_timeout']:status='failed_partial' if 'failure.json' in data else 'incomplete'
    summary=dict(schema='lr-stability-independent-verification-v1',evidence_validation='PASS',scientific_validation='DESCRIPTIVE_ONLY' if status=='completed' else 'NOT_EVALUATED',execution_status=status,optimizer_calls_reserved=len(calls),successful_step_receipts=len(steps),model_forward_calls_reserved=len(forwards),model_forward_calls_returned=len(returns),payloads_verified=len(payloads),observations_verified=len(observations),historical_visible_trajectory_through500_matched=not mismatches and len(steps)>=500 and 'A-0500.json' in observations,statistics=statistics,**v.DENIALS)
    if status=='completed':
        old_endpoint=v.payload(history['A-0600.json'],'A','observation',600,608,train,[],{})
        current=statistics['A-0600'];summary['historical_A600_endpoint']=old_endpoint
        summary['fixed_600_new_minus_historical_A']={'macro_complete_rate':current['macro_complete_rate']-old_endpoint['macro_complete_rate'],'per_class_complete_rate':{c:current['per_class_complete_rate'][c]-old_endpoint['per_class_complete_rate'][c] for c in v.TARGETS}}
    return summary

FIXED_PLAN={'schema': 'kws-lr-stability-plan-v1', 'purpose': 'posthoc-proposed-fixed-lr-stability-diagnostic', 'production_qualified': False, 'seed': 20261015, 'model': {'feature_dim': 32, 'hidden_dim': 48, 'vocab_size': 5}, 'targets_by_text': {'你好小窝': [1, 2, 3, 4], '小窝小窝': [3, 4, 3, 4]}, 'optimizer': {'name': 'Adam', 'lr': 0.003, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'eps': 1e-08, 'gradient_clip': 5.0}, 'ctc': {'blank': 0, 'reduction': 'none', 'zero_infinity': False, 'normalization': 'per-sample-input-frame-count'}, 'max_optimizer_calls': 600, 'retries': 0, 'new_tts_calls': 0, 'new_asr_calls': 0, 'warm_start': False, 'augmentation': False, 'vad': False, 'export_weights': False, 'registry_sha256': '28033edbbc1e444eae27c33af2e060d1654c4b7a072e65401b055fc27fc3b1b0', 'source_verifier_sha256': '9d37fb852cdd5048a4d5c88f9e21efe2b3c1e18f25244babb2d811675ea9fbae', 'runtime_lock_sha256': 'b132c2c202e5ff00b6cbd1b2b630fa92dfe9ba46fcf0423dc0633381bbe80984', 'code_sha256': {'training/model.py': '768800e25537367da770ef44178249208956d7c3c33f1eb7d60b1b9c98fe59c5', 'training/frontend.py': 'd8ae687303d5c9c6bfdf6539c94f4ae463d910a4837ed77633d7656ff02a582e', 'training/frontend_spec.py': 'fbd17236e1449101bed6df8f105baad476edb0b005e6e45cb83ebc7971d166e6'}, 'campaign_id': 'kws-lr-stability-20261016-v1', 'campaign_claim_path': 'research-budget/kws-lr-stability-20261016-v1.claim.json', 'input_pins_sha256': '64c77a270c7f13d8252d10282191aaeb207f425c10fb89a2f01e390fa994e37c', 'frontend': {'frontend': 'logmel', 'feature_dim': 32, 'frame_len': 400, 'hop': 320}, 'stages': [{'name': 'A', 'updates': 600, 'checkpoints': [0, 50, 100, 200, 300, 400, 500, 600], 'blank_lambda': 1}], 'observation_timing': 'post-step; step 0 before any optimizer call', 'loss_timing': 'pre-step', 'max_model_forward_calls': 608, 'cohort_sha256': 'fe23e633925573e2b02ac8530e066214879a11cbccc2e4ef5875aa2c38006e87', 'prefix_rule': {'window_samples': 400, 'hop_samples': 320, 'threshold': 0.0001, 'rule': 'maximal initial run of complete unwindowed PCM RMS strictly below threshold; zero allowed'}, 'train_weights': {'A': {'KW1': 0.08333333333333333, 'KW2': 0.08333333333333333}}, 'early_stop': False, 'extension': False, 'resume': False, 'probe_selection': False, 'train_record_ids': ['kw1-preset-supplement-20261014-v1:kw1s-05', 'kw1-preset-supplement-20261014-v1:kw1s-06', 'kw1-preset-supplement-20261014-v1:kw1s-07', 'tts-quality-remaining-20261014-v1:tqp-qwen06-05', 'tts-quality-remaining-20261014-v1:tqp-qwen06-06', 'tts-quality-remaining-20261014-v1:tqp-qwen06-10', 'tts-quality-remaining-20261014-v1:tqp-qwen06-11', 'tts-quality-remaining-20261014-v1:tqp-qwen06-14', 'tts-quality-remaining-20261014-v1:tqp-qwen06-15', 'tts-quality-remaining-20261014-v1:tqp-qwen17-06', 'tts-quality-remaining-20261014-v1:tqp-qwen17-14', 'tts-quality-remaining-20261014-v1:tqp-qwen17-15'], 'probe_record_ids': [], 'max_probe_forwards': 0, 'historical_input_pins_sha256': '849d7b768b61d7c8238ffea3f5bd69328b6cfba70d7eb25088338b20b33929cd', 'historical_manifest_sha256': '4260c936d8249c59f2d5b270d40274b5f2b29bb45d930dd31c338b240d666b25', 'initial_weight_sha256': '0eafd98dd20f8c799d1c4d733a47314018ecfdd942c2769fed49cf8ae4fc7b2d', 'historical_replay_through_step': 500, 'learning_rate_schedule': [{'first_update': 1, 'last_update': 500, 'lr': 0.003}, {'first_update': 501, 'last_update': 600, 'lr': 0.0003}], 'shared_source_sha256': {'contract.py': '6dca1cad9dba10f50e439f775b6d04d90f246544cbd5a634e30490589490bc74', 'run_grouped_balance.py': 'f57e0fae61b314769cb1d3f3365b77f9ccb9400071be425b3ce7d04e66d53b6e', 'cohort-prefix-freeze.json': 'fe23e633925573e2b02ac8530e066214879a11cbccc2e4ef5875aa2c38006e87', 'runtime-candidate.lock.json': 'b132c2c202e5ff00b6cbd1b2b630fa92dfe9ba46fcf0423dc0633381bbe80984', 'requirements.research.lock': '7aa51b6eebd2e2db290338345c30b8c7a8e197531746ee77efb7ef8678124dee'}, 'shared_module': 'research/kws-grouped-balance-v1'}

def load_sources(repo,freeze):
    repo=Path(repo);module=repo/'research/kws-lr-stability-v1';shared=repo/'research/kws-grouped-balance-v1'
    plan=v.strict_json(v.pinned(module/'plan.json',freeze['plan_sha256']))
    v.keys(plan,set(FIXED_PLAN)|{'execution_enabled','diagnostic_code_sha256'})
    for k,value in FIXED_PLAN.items():v.same(plan[k],value,'fixed_plan_'+k)
    v.same(plan['execution_enabled'],True,'execution_enabled');v.keys(plan['diagnostic_code_sha256'],{'contract.py','run_lr_stability.py','numeric_preflight.py'})
    for name,h in plan['diagnostic_code_sha256'].items():v.pinned(module/name,h)
    for name,h in plan['shared_source_sha256'].items():v.pinned(shared/name,h)
    for name,h in plan['code_sha256'].items():v.pinned(repo/name,h)
    registry_dir=repo/'research/kws-candidate-registry-v1'
    registry=v.strict_json(v.pinned(registry_dir/'candidates.json',plan['registry_sha256']));v.pinned(registry_dir/'build_registry.py',plan['source_verifier_sha256'])
    inputs=v.strict_json(v.pinned(registry_dir/'input-pins.json',plan['input_pins_sha256']));v.same(inputs,registry['repository_inputs'],'registry_input_pins')
    for name,h in inputs.items():v.pinned(repo/name,h)
    cohort=v.strict_json(v.pinned(shared/'cohort-prefix-freeze.json',plan['cohort_sha256']));train,_=v.check_cohort(cohort,v.FIXED_PLAN,registry)
    public=repo/'research/kws-grouped-balance-v1/result-20261016'
    archive=v.strict_json(v.pinned(public/'archive-manifest.json','cb854f2e697313f1895e730b3a5586155d4a93b1d1e9b64d4c546e3ce8708997'))
    v.same(archive['files']['result-hash-manifest.json'],HISTORY_SHA,'historical_public_binding')
    manifest=v.strict_json(v.pinned(public/'result-hash-manifest.json',HISTORY_SHA))
    history_pins=v.strict_json(v.pinned(module/'historical-input-pins.json',plan['historical_input_pins_sha256']))
    v.same(history_pins['accepted_manifest_sha256'],HISTORY_SHA,'historical_manifest_pin')
    for name,h in history_pins['files'].items():v.same(h,manifest['result_files'].get(name),'historical_file_pin')
    return plan,train

def verify_outer(process_path, verified_source_commit, repo, freeze):
    PLAN_SHA=freeze["plan_sha256"];LAUNCHER_SHA=freeze["launcher_sha256"]
    v.require(type(verified_source_commit) is str and v.re.fullmatch('[0-9a-f]{40}',verified_source_commit) is not None,'source_commit_format')
    parent=Path(process_path).parent
    process=v.strict_json(v.safe_read(process_path))
    intent=v.strict_json(v.safe_read(parent/'launch-intent.json'))
    started=v.strict_json(v.safe_read(parent/'process-start.json'))
    v.keys(intent,{'schema','campaign_id','parent_verified_source_commit','plan_sha256','runner_sha256','launcher_sha256','command_sha256','max_optimizer_updates','new_tts_calls','new_asr_calls','max_probe_forwards','retry','actual_optimizer_updates','wall_timeout_seconds','cpu_limit_seconds','virtual_address_space_limit_bytes','per_file_limit_bytes','cpu_affinity','started_at_utc','production_qualified','reusable_weights_exported','reviewed_source_pins','independent_new_budget','previous_campaign_budget_reused','claim_relative','core_dump_limit_bytes','max_model_forward_calls','runner_evidence_directory_limit_bytes'})
    expected=dict(schema='research-lr-stability-local-launch-intent-v1',campaign_id=CAMPAIGN,parent_verified_source_commit=verified_source_commit,plan_sha256=PLAN_SHA,runner_sha256=freeze['runner_sha256'],launcher_sha256=LAUNCHER_SHA,max_optimizer_updates=600,max_model_forward_calls=608,new_tts_calls=0,new_asr_calls=0,max_probe_forwards=0,retry=False,actual_optimizer_updates='UNKNOWN_UNTIL_RESULT_VERIFICATION',wall_timeout_seconds=3600,cpu_limit_seconds=3600,virtual_address_space_limit_bytes=8*1024**3,per_file_limit_bytes=16*1024**2,runner_evidence_directory_limit_bytes=512*1024**2,production_qualified=False,reusable_weights_exported=False,reviewed_source_pins=freeze['reviewed_source_pins'],independent_new_budget=True,previous_campaign_budget_reused=False,claim_relative='research-budget/kws-lr-stability-20261016-v1.claim.json',core_dump_limit_bytes=0)
    for k,val in expected.items():v.same(intent[k],val,'launch_binding_'+k)
    v.digest(intent['command_sha256'])
    base=Path(repo).absolute().parent
    v.pinned(base/'launch_reviewed_lr_stability.py',LAUNCHER_SHA)
    command=[str(base/'fit-venv/bin/python'),'-I','-B',str(base/'lr-stability-integrated/research/kws-lr-stability-v1/run_lr_stability.py'),'--repo',str(base/'lr-stability-integrated'),'--artifacts',str(base.parent/'recovery-kws-20261015/artifacts'),'--wheelhouse',str(base/'runtime-work/wheels'),'--output',str(base/'lr-stability-result'),'--claim-root',str(base/'lr-stability-campaign-ledger'),'--plan-sha256',PLAN_SHA,'--historical-result',str(base/'grouped-balance-result'),'--historical-manifest',str(base/'grouped-balance-result-acceptance/hash-manifest.json')]
    v.same(intent['command_sha256'],v.sha(json.dumps(command,separators=(',',':')).encode()),'fixed_launch_command')
    v.require(type(intent['cpu_affinity']) is list and len(intent['cpu_affinity'])==1,'cpu_affinity');v.integer(intent['cpu_affinity'][0],0,2**20)
    v.require(type(intent['started_at_utc']) is str and v.re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00',intent['started_at_utc']) is not None,'start_time')
    v.keys(started,{'pid','command_sha256'});v.integer(started['pid'],1,2**31);v.same(started['command_sha256'],intent['command_sha256'],'process_command')
    v.keys(process,{'schema','exit_code','wall_timeout','wall_seconds','user_cpu_seconds','system_cpu_seconds','child_max_rss_kib','max_rss_scope','stdout_sha256','stderr_sha256','launcher_sha256','command_sha256','actual_optimizer_updates','production_qualified'})
    expected_process=dict(schema='research-lr-stability-local-process-result-v1',max_rss_scope='Linux RUSAGE_CHILDREN, not cgroup or SSC305',launcher_sha256=LAUNCHER_SHA,command_sha256=intent['command_sha256'],actual_optimizer_updates='VERIFY_CALL_AND_STEP_LEDGER',production_qualified=False)
    for k,val in expected_process.items():v.same(process[k],val,'process_binding_'+k)
    v.integer(process['exit_code'],-255,255);v.require(type(process['wall_timeout']) is bool,'timeout_bool')
    for k in ('wall_seconds','user_cpu_seconds','system_cpu_seconds','child_max_rss_kib'):v.number(process[k])
    for name in ('stdout','stderr'):v.same(v.sha(v.safe_read(parent/(name+'.log'))),process[name+'_sha256'],'process_log_hash')
    # These raw receipts and logs stay local. Only the supplied public commit is summarized.
    return process


def verify(repo,result,claim_path,process_path,freeze_path):
    freeze_raw=v.safe_read(freeze_path);freeze=v.strict_json(freeze_raw)
    v.keys(freeze,{'schema','plan_sha256','launcher_sha256','runner_sha256','reviewed_source_commit','reviewed_source_pins','tolerances_sha256','verifier_sha256'})
    v.same(freeze['schema'],'lr-stability-verifier-trusted-freeze-v1','freeze_schema')
    for k in ('plan_sha256','launcher_sha256','runner_sha256','tolerances_sha256','verifier_sha256'):v.digest(freeze[k])
    v.pinned(__file__,freeze['verifier_sha256']);v.pinned(BASE,BASE_SHA)
    v.same(freeze['tolerances_sha256'],'b701856f6bae2c93aa377cece1549a4afcab03921940dbc49ece6b3ddc01cf9b','unchanged_tolerances')
    v.pinned(BASE.with_name('tolerances.json'),freeze['tolerances_sha256'])
    v.require(type(freeze['reviewed_source_pins']) is dict and freeze['reviewed_source_pins'],'freeze_source_pins')
    for name,h in freeze['reviewed_source_pins'].items():
        v.require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts,'source_path');v.pinned(Path(repo)/name,h)
    prefix='research/kws-lr-stability-v1/'
    v.same(freeze['reviewed_source_pins'].get(prefix+'plan.json'),freeze['plan_sha256'],'freeze_plan');v.same(freeze['reviewed_source_pins'].get(prefix+'run_lr_stability.py'),freeze['runner_sha256'],'freeze_runner')
    base=Path(repo).absolute().parent
    for actual,expected in ((repo,base/'lr-stability-integrated'),(result,base/'lr-stability-result'),(claim_path,base/'lr-stability-campaign-ledger/research-budget/kws-lr-stability-20261016-v1.claim.json'),(process_path,base/'lr-stability-execution-evidence/process-result.json')):v.same(Path(actual).absolute(),expected,'fixed_path')
    plan,train=load_sources(repo,freeze);history=load_history(base)
    v.same(history['A-initialization.json']['initial_weight_sha256'],plan['initial_weight_sha256'],'historical_initial_hash')
    fd=v.safe_dir(result)
    try:names=sorted(os.listdir(fd))
    finally:os.close(fd)
    v.require(set(names)<=ALLOWED,'result_file_set');raws={};size=0
    for n in names:
        raw=v.safe_read(Path(result)/n);size+=len(raw);v.require(size<=512*1024**2,'evidence_size');raws[n]=raw
    data={n:v.strict_json(raw) for n,raw in raws.items()};process=verify_outer(process_path,freeze['reviewed_source_commit'],repo,freeze)
    summary=verify_objects(data,plan,train,history,v.safe_read(claim_path),result,process,freeze['plan_sha256'],raws)
    summary.update(externally_supplied_source_commit=freeze['reviewed_source_commit'],historical_source_commit=HISTORY_COMMIT,historical_manifest_sha256=HISTORY_SHA,shared_verifier_sha256=BASE_SHA,tolerances=v.TOLERANCES,limitations=['First500 equality concerns the saved visible numerical trajectory, not independent proof of unsaved Adam or parameter states.','Only the predeclared step600 endpoint is evaluated; no best-checkpoint selection.','This posthoc-proposed diagnostic uses 12 exposed machine-labelled training samples and no probe; it establishes neither generalization, recall, FA/h nor production readiness.','All LPs receive independent float64 CTC checks under unchanged pre-execution tolerances; reviewed source and receipts are not hardware attestation.'])
    fd=v.safe_dir(result)
    try:v.same(sorted(os.listdir(fd)),names,'directory_changed')
    finally:os.close(fd)
    for n,raw in raws.items():v.same(v.sha(v.safe_read(Path(result)/n)),v.sha(raw),'evidence_changed')
    manifest=dict(schema='lr-stability-evidence-hash-manifest-v1',trusted_freeze_sha256=v.sha(freeze_raw),plan_sha256=freeze['plan_sha256'],shared_verifier_sha256=BASE_SHA,historical_manifest_sha256=HISTORY_SHA,verifier_sha256=v.sha(v.safe_read(__file__)),result_files={n:v.sha(raw) for n,raw in raws.items()})
    return summary,manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('repo','result','campaign-claim','process-result','trusted-freeze','report-dir'):parser.add_argument('--'+flag,type=Path,required=True)
    a=parser.parse_args()
    try:
        summary,manifest=verify(a.repo,a.result,a.campaign_claim,a.process_result,a.trusted_freeze)
        v.write_reports(a.report_dir,a.result,summary,manifest)
    except Exception:
        print(json.dumps(dict(schema='lr-stability-independent-verification-v1',evidence_validation='FAIL',scientific_validation='NOT_EVALUATED',execution_status='incomplete',retry_authorized=False)));return 2
    print(json.dumps({k:x for k,x in summary.items() if k not in ('statistics','limitations')},sort_keys=True))
    return 0 if summary['execution_status']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
