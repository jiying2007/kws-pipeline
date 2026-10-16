"""Synthetic only. Reuses the reviewed grouped fixture generator, never model code."""
import copy,types
from pathlib import Path
import verify_result as a
v=a.v
helper=Path(__file__).absolute().parent.parent/'grouped-balance-result-verification/fixtures.py'
raw=v.safe_read(helper)
if v.sha(raw)!='d499d321d5093815a024bdab0843140a596eaf323dbb02163fcb430a834b68f4':raise ValueError('fixture pin mismatch')
ns={'FROZEN':v};exec(compile(raw.decode().replace('import verify_result as v','v=FROZEN'),str(helper),'exec'),ns)

def fixture(success=False):
    base=ns['fixture'](success);old=base[0];data={k:x for k,x in old.items() if k in a.ALLOWED and not k.startswith(('call-','step-','forward-','forward-return-','payload-'))}
    for i in range(1,609):
        for k in ('forward','forward-return','payload'):data[f'{k}-{i:04d}.json']=old[f'{k}-{i:04d}.json']
    for u in range(1,601):
        for k in ('call','step'):data[f'{k}-{u:04d}.json']=old[f'{k}-{u:04d}.json']
    history=copy.deepcopy({k:x for k,x in data.items() if k=='A-initialization.json' or k.startswith(('payload-','step-','A-'))})
    for k,obj in data.items():
        if 'schema' in obj:obj['schema']=obj['schema'].replace('grouped-','lr-stability-')
    plan=copy.deepcopy(a.FIXED_PLAN);plan.update(execution_enabled=True,diagnostic_code_sha256={'contract.py':'a'*64,'run_lr_stability.py':'b'*64})
    out='/tmp/synthetic-lr-stability-result';claim=v.encoded(dict(schema='lr-stability-exclusive-claim-v1',campaign_id=a.CAMPAIGN,output=out,max_optimizer_calls=600,max_model_forward_calls=608,**v.DENIALS))
    data['intent.json'].update(campaign_id=a.CAMPAIGN,max_optimizer_calls=600,max_model_forward_calls=608)
    data['binding.json'].update(campaign_id=a.CAMPAIGN,campaign_claim_path=plan['campaign_claim_path'],campaign_claim_sha256=v.sha(claim),max_model_forward_calls=608,historical_manifest_sha256=a.HISTORY_SHA)
    for u in range(1,601):
        s=data[f'step-{u:04d}.json'];s.update(learning_rate=.003 if u<=500 else .0003,historical_replay_matched=u<=500,payload_sha256=v.sha(v.encoded(data[s['payload_file']])))
    data['learning-rate-transition.json']=dict(update=501,previous_lr=.003,new_lr=.0003,optimizer_recreated=False,historical_replay_through_step=500)
    data['terminal.json']=dict(schema='lr-stability-terminal-v1',status='completed',primary_endpoint_step=600,train_endpoints={'A':old['terminal.json']['train_endpoints']['A']},historical_replay_through_step=500,historical_replay_observations=[0,50,100,200,300,400,500],optimizer_calls_reserved=600,model_forward_calls_reserved=608,probe_used_for_selection=False,**v.DENIALS)
    return [data,plan,base[2],history,claim,out,dict(exit_code=0,wall_timeout=False),'e'*64]

def partial(args,index,returned=True,payload=True,call=False,step=False,phase='forward'):
    args=copy.deepcopy(args);old=args[0];keep={'intent.json','binding.json','A-initialization.json'}
    for i,(arm,kind,u) in enumerate(a.EVENTS[:index],1):
        keep.add(f'forward-{i:04d}.json')
        if i<index or returned:keep.add(f'forward-return-{i:04d}.json')
        if i<index or payload:keep.add(f'payload-{i:04d}.json')
        if kind=='training':
            if i<index or call:keep.add(f'call-{u:04d}.json')
            if i<index or step:keep.add(f'step-{u:04d}.json')
        elif i<index:keep.add(f'A-{u:04d}.json')
    if index>=508:keep.add('learning-rate-transition.json')
    data={k:x for k,x in old.items() if k in keep};args[0]=data;arm,kind,u=a.EVENTS[index-1]
    ctx={'phase':phase}
    if phase=='forward':ctx.update(kind=kind,record_ids=[r['record_id'] for r in args[2]])
    data['failure.json']=dict(schema='lr-stability-failure-v1',status='failed_no_retry',stage='A',update=u,optimizer_calls_reserved=sum(k.startswith('call-') for k in data),model_forward_calls_reserved=index,error_type='RuntimeError',context=ctx,error='Stopped on numerical, evidence, budget or output invariant; no retry or resume',**v.DENIALS)
    args[6]=dict(exit_code=1,wall_timeout=False);return args
