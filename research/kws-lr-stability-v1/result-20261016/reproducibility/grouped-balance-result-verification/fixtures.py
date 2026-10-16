"""Complete synthetic ledgers; no numeric runtime, model, audio, or weights."""
import copy,json,math
import verify_result as v

def sample(meta,weight,success,training=False):
    c=meta['target_class'];n=meta['input_frames'];p=meta['prefix_frames'];path=[0]*n
    if success:path[p:p+4]=v.TARGETS[c]
    rows=[[math.log(.96 if i==t else .01) for i in range(5)] for t in path]
    nll={k:v.ctc_nll(rows,t) for k,t in {**v.TARGETS,'empty':[]}.items()};blank=[math.exp(r[0]) for r in rows];anchor=-sum(r[0] for r in rows[:p]);decoded=v.collapse(path)
    return dict(record_id=meta['record_id'],target_class=c,preset_voice_id=meta['preset_voice_id'],pcm_sha256=meta['pcm_sha256'],input_frames=n,target_length=4,raw_argmax_tokens=path,target_tokens=v.TARGETS[c],decode_tokens=decoded,complete_match=decoded==v.TARGETS[c],ctc_nll=nll[c],ctc_nll_per_input_frame=nll[c]/n,blank_argmax_fraction=path.count(0)/n,blank_mean_probability=sum(blank)/n,log_probs=rows,ctc_nll_by_target={c:nll[c]} if training else nll,prefix_frames=p,prefix_nonblank_argmax_frames=0,prefix_blank_probability_min=min(blank[:p]) if p else None,prefix_blank_probability_max=max(blank[:p]) if p else None,anchor_nll=anchor,anchor_nll_per_input_frame=anchor/n,first_nonblank_frame=next((i for i,t in enumerate(path) if t),None),loss_weight=weight,total_loss_per_input_frame=(nll[c]+anchor)/n,weighted_total_loss=(nll[c]+anchor)/n*weight)

def fixture(success=False):
    plan=copy.deepcopy(v.FIXED_PLAN);plan.update(execution_enabled=True,diagnostic_code_sha256={n:'a'*64 for n in ('run_grouped_balance.py','contract.py','numeric_preflight.py','requirements.research.lock')})
    train=[dict(record_id=r,target_class='KW1' if i<4 else 'KW2',preset_voice_id=('qwen-preset:serena','qwen-preset:uncle_fu','qwen-preset:dylan')[i%3],pcm_sha256=f'{i:064x}',input_frames=9,prefix_frames=i%2) for i,r in enumerate(plan['train_record_ids'])]
    probe=[dict(record_id=r,target_class='KW1' if i==0 else 'KW2',preset_voice_id='qwen-preset:vivian',pcm_sha256=f'{i+20:064x}',input_frames=9,prefix_frames=1) for i,r in enumerate(plan['probe_record_ids'])]
    runtime={};out='/tmp/synthetic-grouped-balance-result';plan_sha='e'*64
    claim=v.encoded(dict(schema='grouped-exclusive-claim-v1',campaign_id=v.CAMPAIGN,output=out,max_optimizer_calls=1200,max_model_forward_calls=1218,**v.DENIALS))
    data={'intent.json':dict(schema='grouped-intent-v1',campaign_id=v.CAMPAIGN,max_optimizer_calls=1200,max_model_forward_calls=1218,no_resume=True,**v.DENIALS)}
    data['binding.json']=dict(schema='grouped-input-binding-v1',plan_sha256=plan_sha,initial_weight_sha256='d'*64,registry_sha256=plan['registry_sha256'],cohort_sha256=plan['cohort_sha256'],runtime_lock_sha256=plan['runtime_lock_sha256'],seed=20261015,threads={'intraop':1,'interop':1},campaign_id=v.CAMPAIGN,campaign_claim_path=plan['campaign_claim_path'],campaign_claim_sha256=v.sha(claim),max_model_forward_calls=1218,**v.DENIALS)
    for a in ('A','B'):data[a+'-initialization.json']=dict(initial_weight_sha256='d'*64,seed=20261015)
    cache={};endpoints={};probe_endpoints={}
    for i,(a,kind,u) in enumerate(v.forward_events(),1):
        total=(0 if a=='A' else 600)+u
        rows=[]
        for meta in probe if kind=='probe' else train:
            c=meta['target_class'];weight=(1/2 if c=='KW1' else 1/6) if kind=='probe' else 1/12 if a=='A' else 1/8 if c=='KW1' else 1/16
            key=(meta['record_id'],weight,kind=='training',success and u==600)
            if key not in cache:cache[key]=sample(meta,weight,success and u==600,kind=='training')
            rows.append(cache[key])
        per_class={c:sum(r['complete_match'] for r in rows if r['target_class']==c)/sum(r['target_class']==c for r in rows) for c in v.TARGETS};macro=sum(per_class.values())/2
        ctc=sum(r['ctc_nll_per_input_frame']*r['loss_weight'] for r in rows);anchor=sum(r['anchor_nll_per_input_frame']*r['loss_weight'] for r in rows)
        payload=dict(schema='grouped-forward-payload-v1',kind=kind,stage=a,update=u,role='probe' if kind=='probe' else 'train',timing='pre-step' if kind=='training' else 'post-step' if u else 'initial',model_forward_reserved=i,optimizer_calls_reserved=1200 if kind=='probe' else total-(kind=='training'),samples=rows,ctc_loss=ctc,anchor_loss=anchor,finite_loss=ctc+anchor,blank_lambda=1,per_class_complete_rate=per_class,macro_complete_rate=macro,**v.DENIALS)
        pn=f'payload-{i:04d}.json';data[pn]=payload
        event=dict(status='forward_reserved',stage=a,update=u,kind=kind,model_forward_reserved=i)
        data[f'forward-{i:04d}.json']=event;data[f'forward-return-{i:04d}.json']={**event,'status':'forward_returned'}
        if kind=='training':
            data[f'call-{total:04d}.json']=dict(status='step_reserved',stage=a,update=u,optimizer_call_reserved=total)
            data[f'step-{total:04d}.json']=dict(status='step_succeeded',stage=a,update=u,payload_file=pn,payload_sha256=v.sha(v.encoded(payload)),finite_loss=ctc+anchor,ctc_loss=ctc,anchor_loss=anchor,blank_lambda=1,finite_grad_norm=1.0,updated_weights_finite=True,weights_changed=True)
        else:
            name=f'{a}-{u:04d}.json' if kind=='observation' else f'probe-{a}-0600.json';data[name]=payload
            if u==600:(endpoints if kind=='observation' else probe_endpoints)[a]=dict(fit_success=success,prefix_success=True,macro_complete_rate=macro,per_class_complete_rate=per_class)
    data['terminal.json']=dict(schema='grouped-terminal-v1',status='completed',primary_endpoint_step=600,train_endpoints=endpoints,probe_endpoints=probe_endpoints,optimizer_calls_reserved=1200,model_forward_calls_reserved=1218,probe_used_for_selection=False,**v.DENIALS)
    return [data,plan,train,probe,runtime,claim,out,dict(exit_code=0,wall_timeout=False),plan_sha]

def partial(args,forward_count,returned=True,payload_present=True,call_present=False,step_present=False,failure=False,phase='forward'):
    args=copy.deepcopy(args);data=args[0];events=v.forward_events();arm,kind,u=events[forward_count-1];total=(0 if arm=='A' else 600)+u
    keep={'intent.json','binding.json','A-initialization.json'}
    if arm=='B':keep.add('B-initialization.json')
    for i,(a,k,step) in enumerate(events[:forward_count],1):
        n=(0 if a=='A' else 600)+step;keep.add(f'forward-{i:04d}.json')
        if i<forward_count or returned:keep.add(f'forward-return-{i:04d}.json')
        if i<forward_count or payload_present:keep.add(f'payload-{i:04d}.json')
        if k=='training':
            if i<forward_count or call_present:keep.add(f'call-{n:04d}.json')
            if i<forward_count or step_present:keep.add(f'step-{n:04d}.json')
        elif i<forward_count:keep.add(f'{a}-{step:04d}.json' if k=='observation' else f'probe-{a}-0600.json')
    args[0]={k:val for k,val in data.items() if k in keep};data=args[0]
    if failure:
        context={'phase':phase}
        if phase=='forward':context.update(kind=kind,record_ids=[r['record_id'] for r in (args[3] if kind=='probe' else args[2])])
        data['failure.json']=dict(schema='grouped-failure-v1',status='failed_no_retry',stage=arm,update=u,optimizer_calls_reserved=sum(k.startswith('call-') for k in data),model_forward_calls_reserved=forward_count,error_type='RuntimeError',context=context,error='Stopped on numerical, evidence, budget or output invariant; no retry or resume',**v.DENIALS)
    args[7]=dict(exit_code=1,wall_timeout=False)
    return args
