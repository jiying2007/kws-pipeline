#!/usr/bin/env python3
"""Single bounded in-memory research fit. No training receipts or model exports."""
from __future__ import annotations
import argparse
from array import array
from collections import Counter
import importlib.metadata
import importlib.util
import json
import math
import os
import stat
import sysconfig
import zipfile
import io
import types
import platform
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
_contract_spec=importlib.util.spec_from_file_location('fit_contract',HERE/'contract.py')
_contract=importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(_contract)
Budget,DENIALS,atomic_json,collapse,select_pair,sha,weights=(_contract.Budget,_contract.DENIALS,_contract.atomic_json,_contract.collapse,_contract.select_pair,_contract.sha,_contract.weights)

HERE = Path(__file__).resolve().parent

def load_module(name, path, raw):
    module=types.ModuleType(name)
    module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module

def read_checked(path, expected):
    path=Path(path).absolute()
    if path.resolve()!=path or '..' in path.parts:
        raise ValueError('input aliases/symlinks are forbidden')
    directory=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    fd=None
    try:
        for part in path.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
            os.close(directory);directory=nxt
        fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:
            raise ValueError('input must be an unaliased regular file')
        with os.fdopen(fd,'rb') as stream:
            fd=None;raw=stream.read();after=os.fstat(stream.fileno())
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('input changed during verification')
    finally:
        if fd is not None:os.close(fd)
        os.close(directory)
    if not expected or sha(raw)!=expected:
        raise ValueError(f'unreviewed or changed input: {path.name}')
    return raw

def validate_runtime(lock, wheelhouse):
    if platform.python_version()!=lock['python'] or platform.machine()!='x86_64' or sys.platform!='linux':
        raise ValueError('research runtime mismatch; production training qualification not implied')
    installed={x.metadata['Name'].lower().replace('_','-') for x in importlib.metadata.distributions()}
    allowed={r['name'].lower().replace('_','-') for r in lock['files']}|{'pip'}
    if installed-allowed:raise ValueError('unexpected distributions in research runtime')
    site=Path(sysconfig.get_paths()['purelib'])
    for row in lock['files']:
        raw=read_checked(wheelhouse/row['filename'],row['sha256'])
        if len(raw)!=row['size_bytes'] or importlib.metadata.version(row['name'])!=row['version']:
            raise ValueError('runtime wheel/installed version mismatch')
        with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
            for name in wheel.namelist():
                if name.endswith('/') or name.endswith('.dist-info/RECORD'):continue
                parts=Path(name).parts
                if any(x in ('..','.') for x in parts) or Path(name).is_absolute():raise ValueError('unsafe wheel member')
                if '.data' in parts[0]:
                    if parts[1] not in ('purelib','platlib'):continue
                    parts=parts[2:]
                payload=wheel.read(name)
                read_checked(site.joinpath(*parts),sha(payload))

def run(args):
    plan=json.loads(read_checked(HERE/'plan.json',args.plan_sha256))
    if plan['execution_enabled'] is not True:
        raise ValueError('execution disabled pending independent review and root approval')
    if [(x['name'],x['updates'],x['checkpoints']) for x in plan['stages']] != [('pair',300,[0,50,100,200,300]),('A',600,[0,50,100,200,300,400,500,600]),('B',600,[0,50,100,200,300,400,500,600])]:
        raise ValueError('fixed stage budget/checkpoints changed')
    if plan['max_optimizer_calls']!=1500 or any(plan[k] for k in ('production_qualified','warm_start','augmentation','vad','auxiliary_losses','export_weights')):
        raise ValueError('research-only contract changed')
    for name,digest in plan['diagnostic_code_sha256'].items():read_checked(HERE/name,digest)
    registry_path=args.repo/'research/kws-candidate-registry-v1/candidates.json'
    verifier_path=args.repo/'research/kws-candidate-registry-v1/build_registry.py'
    registry=json.loads(read_checked(registry_path,plan['registry_sha256']))
    verifier_raw=read_checked(verifier_path,plan['source_verifier_sha256'])
    source_bytes={name:read_checked(args.repo/name,digest) for name,digest in registry['repository_inputs'].items()}
    pins=json.loads(read_checked(verifier_path.with_name('input-pins.json'),plan['input_pins_sha256']))
    if pins!=registry['repository_inputs']:raise ValueError('source input pins differ from reviewed registry')
    for path,digest in plan['code_sha256'].items():read_checked(args.repo/path,digest)
    runtime=json.loads(read_checked(HERE/'runtime-candidate.lock.json',plan['runtime_lock_sha256']))
    validate_runtime(runtime,args.wheelhouse)
    adapter=load_module('candidate_registry',verifier_path,verifier_raw)
    records=adapter.verify_inputs(registry,{'repo':args.repo,'artifacts':args.artifacts})
    if Counter(r['target_class'] for r in records)!=Counter(KW1=5,KW2=11):raise ValueError('requires exact 5/11 cohort')
    records.sort(key=lambda r:r['record_id'])
    if len(set(r['record_id'] for r in records))!=16:raise ValueError('duplicate IDs')
    if select_pair(registry['records'])!=plan['pair_record_ids']:raise ValueError('pair selection changed')
    for r in records:
        if sha(r['pcm_bytes'])!=r['pcm_sha256'] or len(r['pcm_bytes'])!=r['frames']*2 or r['sample_rate_hz']!=16000 or r['channels']!=1:
            raise ValueError('PCM binding mismatch')
        if r['target_text'] not in plan['targets_by_text']:raise ValueError('unknown target')
    claim_relative=Path(plan['campaign_claim_path'])
    if claim_relative.is_absolute() or '..' in claim_relative.parts or str(claim_relative)!=plan['campaign_claim_path']:
        raise ValueError('reviewed fixed relative campaign claim path required')
    claim_path=args.claim_root/claim_relative
    budget=Budget(args.output,claim_path,plan['campaign_id'])
    failure_context={}
    stage_name='model_and_feature_preparation'; update=0
    try:
        # Must run in a fresh interpreter: model.py pins both Torch CPU thread pools.
        for name in ('torch','model','frontend','frontend_spec'):
            if name in sys.modules:raise ValueError('numeric module was preloaded')
        for name in ('frontend_spec','model','frontend'):
            relative='training/'+name+'.py'
            load_module(name,args.repo/relative,read_checked(args.repo/relative,plan['code_sha256'][relative]))
        from model import TinyStreamingRNN
        from frontend import features
        import torch
        if Path(torch.__file__).resolve()!=Path(sysconfig.get_paths()['purelib'])/'torch/__init__.py':raise ValueError('unexpected Torch import root')
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(plan['seed'])
        model=TinyStreamingRNN(**plan['model'])
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        initial_sha=sha(b''.join(bytes(v.contiguous().view(torch.uint8).reshape(-1).tolist()) for v in initial.values()))
        batches={}
        for name in ('pair','A','B'):
            selected=[r for r in records if name!='pair' or r['record_id'] in plan['pair_record_ids']]
            tensors=[]
            for row in selected:
                pcm=array('h');pcm.frombytes(row['pcm_bytes'])
                if sys.byteorder!='little':pcm.byteswap()
                f=features(torch.tensor(pcm,dtype=torch.float32)/32768.0,**plan['frontend'])
                if not torch.isfinite(f).all() or len(f)<7:
                    context={'record_id':row['record_id'],'input_frames':len(f),'feature_shape':list(f.shape),'feature_nonfinite_count':int((~torch.isfinite(f)).sum()),'finite_min':float(f[torch.isfinite(f)].min()) if torch.isfinite(f).any() else 'NO_FINITE_VALUES','finite_max':float(f[torch.isfinite(f)].max()) if torch.isfinite(f).any() else 'NO_FINITE_VALUES'}
                    stage_name='feature_preparation'; failure_context=context
                    raise ValueError('invalid/too-short features')
                tensors.append(f)
            lengths=torch.tensor([len(f) for f in tensors],dtype=torch.long)
            x=torch.nn.utils.rnn.pad_sequence(tensors,batch_first=True)
            targets=[plan['targets_by_text'][r['target_text']] for r in selected]
            y=torch.tensor([v for row in targets for v in row],dtype=torch.long)
            yl=torch.tensor([len(t) for t in targets],dtype=torch.long)
            w=torch.tensor(weights([r['target_class'] for r in selected],name))
            batches[name]=(selected,x,lengths,targets,y,yl,w)
        atomic_json(args.output/'binding.json',{'schema':'research-fit-input-binding-v1','plan_sha256':sha((HERE/'plan.json').read_bytes()),'initial_weight_sha256':initial_sha,'registry_sha256':plan['registry_sha256'],'runtime':runtime,'runtime_lock_sha256':plan['runtime_lock_sha256'],'reviewed_source_pins':{'model_frontend':plan['code_sha256'],'diagnostic':plan['diagnostic_code_sha256'],'source_verifier':plan['source_verifier_sha256']},'seed':plan['seed'],'threads':{'intraop':torch.get_num_threads(),'interop':torch.get_num_interop_threads()},'campaign_id':plan['campaign_id'],'campaign_claim_path':plan['campaign_claim_path'],'campaign_claim_sha256':sha(claim_path.read_bytes()),'source_capture_limit':'13 historical ID joins; 3 same-PCM capture chains; no formal production ASR admission',**DENIALS})
        loss_fn=torch.nn.CTCLoss(blank=0,reduction='none',zero_infinity=False)
        stage_name='not_started'; update=0
        def sample_finiteness(selected,nll,lengths):
            return [{'record_id':r['record_id'],'input_frames':int(lengths[i]),'target_length':4,'finite_ctc_nll':float(nll[i]) if torch.isfinite(nll[i]) else None,'ctc_nll_status':('FINITE' if torch.isfinite(nll[i]) else 'NAN' if torch.isnan(nll[i]) else 'POS_INF' if nll[i]>0 else 'NEG_INF')} for i,r in enumerate(selected)]
        def parameter_finiteness(gradient=False):
            results=[]
            for name,p in model.named_parameters():
                value=p.grad if gradient else p
                finite=value[torch.isfinite(value)] if value is not None else None
                results.append({'parameter_name':name,'shape':list(p.shape),'missing_gradient':value is None,'nonfinite_count':int((~torch.isfinite(value)).sum()) if value is not None else None,'finite_min':float(finite.min()) if finite is not None and finite.numel() else 'NO_FINITE_VALUES','finite_max':float(finite.max()) if finite is not None and finite.numel() else 'NO_FINITE_VALUES'})
            return results
        def evaluate(stage,step,selected,x,lengths,targets,y,yl):
            nonlocal failure_context
            with torch.no_grad():
                failure_context={'phase':'evaluation_forward','record_ids':[r['record_id'] for r in selected]}
                lp=model(x).log_softmax(2)
                failure_context={'phase':'evaluation_ctc','record_ids':[r['record_id'] for r in selected]}
                nll=loss_fn(lp,y,lengths,yl)
                failure_context={'phase':'evaluation','samples':sample_finiteness(selected,nll,lengths),'log_probability_nonfinite_count':int((~torch.isfinite(lp)).sum())}
                if not torch.isfinite(lp).all() or not torch.isfinite(nll).all():raise ValueError('nonfinite evaluation')
                predictions=lp.argmax(2)
                rows=[]
                for i,r in enumerate(selected):
                    path=predictions[:int(lengths[i]),i].tolist(); decoded=collapse(path)
                    rows.append({'record_id':r['record_id'],'target_class':r['target_class'],'input_frames':int(lengths[i]),'target_length':len(targets[i]),'raw_argmax_tokens':path,'target_tokens':targets[i],'decode_tokens':decoded,'complete_match':decoded==targets[i],'ctc_nll':float(nll[i]),'ctc_nll_per_input_frame':float(nll[i]/lengths[i]),'blank_argmax_fraction':path.count(0)/len(path),'blank_mean_probability':float(lp[:int(lengths[i]),i,0].exp().mean())})
                per_class={c:sum(r['complete_match'] for r in rows if r['target_class']==c)/sum(r['target_class']==c for r in rows) for c in ('KW1','KW2')}
                result={'schema':'research-fit-observation-v1','stage':stage,'update':step,'optimizer_calls_reserved':budget.count,'samples':rows,'per_class_complete_rate':per_class,'macro_complete_rate':sum(per_class.values())/2,**DENIALS}
                atomic_json(args.output/f'{stage}-{step:04d}.json',result)
                return all(r['complete_match'] for r in rows)
        try:
            for stage in plan['stages']:
                stage_name=stage['name'];update=0
                selected,x,lengths,targets,y,yl,w=batches[stage_name]
                model.load_state_dict(initial)
                torch.manual_seed(plan['seed'])
                op=plan['optimizer']
                optimizer=torch.optim.Adam(model.parameters(),lr=op['lr'],betas=tuple(op['betas']),eps=op['eps'],weight_decay=0.0)
                evaluate(stage_name,0,selected,x,lengths,targets,y,yl)
                passed=False
                for update in range(1,stage['updates']+1):
                    optimizer.zero_grad(set_to_none=True)
                    failure_context={'phase':'training_forward','record_ids':[r['record_id'] for r in selected]}
                    lp=model(x).log_softmax(2)
                    failure_context={'phase':'training_ctc','record_ids':[r['record_id'] for r in selected]}
                    nll=loss_fn(lp,y,lengths,yl)
                    loss=((nll/lengths)*w).sum()
                    failure_context={'phase':'loss','samples':sample_finiteness(selected,nll,lengths)}
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    failure_context['phase']='backward'
                    loss.backward()
                    failure_context={'phase':'gradient','parameters':parameter_finiteness(True),'samples':sample_finiteness(selected,nll,lengths)}
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('missing/nonfinite gradients')
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),op['gradient_clip'],error_if_nonfinite=True)
                    before=[p.detach().clone() for p in model.parameters()]
                    budget.before_step(stage_name,update)
                    failure_context['phase']='optimizer_step'
                    optimizer.step()
                    failure_context={'phase':'updated_weights','parameters':parameter_finiteness(),'samples':sample_finiteness(selected,nll,lengths)}
                    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite updated weights')
                    if not any(not torch.equal(a,b) for a,b in zip(before,model.parameters())):raise ValueError('no updated weights')
                    atomic_json(args.output/f'step-{budget.count:04d}.json',{'stage':stage_name,'update':update,'finite_loss':float(loss.detach()),'finite_grad_norm':float(norm),'updated_weights_finite':True,'weights_changed':True})
                    if update in stage['checkpoints']:
                        passed=evaluate(stage_name,update,selected,x,lengths,targets,y,yl)
                if stage_name=='pair' and not passed:
                    atomic_json(args.output/'terminal.json',{'schema':'research-fit-terminal-v1','status':'pair_gate_failed','optimizer_calls_reserved':budget.count,'later_stages_skipped':['A','B'],**DENIALS});return
            atomic_json(args.output/'terminal.json',{'schema':'research-fit-terminal-v1','status':'completed','optimizer_calls_reserved':budget.count,**DENIALS})
        except BaseException:
            raise
    except BaseException as exc:
        if not (args.output/'failure.json').exists():
            atomic_json(args.output/'failure.json',{'schema':'research-fit-failure-v1','stage':stage_name,'update':update,'optimizer_calls_reserved':budget.count,'error_type':type(exc).__name__,'context':failure_context,'error':'Stopped on failed numerical, evidence, budget or output invariant; inspect stage/update and exception type',**DENIALS})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('repo','artifacts','wheelhouse','output','claim-root'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True,help='Externally reviewed enabled-plan identity')
    run(p.parse_args())
