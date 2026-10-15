#!/usr/bin/env python3
"""Single bounded in-memory horizon diagnostic. Research receipts; no model exports."""
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
_contract_spec=importlib.util.spec_from_file_location('grouped_contract',HERE/'contract.py')
_contract=importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(_contract)
Budget,DENIALS,atomic_json,collapse,sha,weights=(_contract.Budget,_contract.DENIALS,_contract.atomic_json,_contract.collapse,_contract.sha,_contract.weights)

validate_plan=_contract.validate_plan

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

def prefix_from_pcm(raw):
    pcm=array('h');pcm.frombytes(raw)
    if sys.byteorder!='little':pcm.byteswap()
    rms=[math.sqrt(sum((v/32768.0)**2 for v in pcm[s:s+400])/400) for s in range(0,len(pcm)-399,320)]
    return next((i for i,v in enumerate(rms) if v>=1e-4),len(rms)),rms

def run(args):
    # All identity checks precede any numerical import or experimental reservation.
    plan=_contract.strict_json(read_checked(HERE/'plan.json',args.plan_sha256));validate_plan(plan)
    for name,digest in plan['diagnostic_code_sha256'].items():read_checked(HERE/name,digest)
    freeze=_contract.strict_json(read_checked(HERE/'cohort-prefix-freeze.json',plan['cohort_sha256']))
    registry_path=args.repo/'research/kws-candidate-registry-v1/candidates.json'
    verifier_path=registry_path.with_name('build_registry.py')
    registry=_contract.strict_json(read_checked(registry_path,plan['registry_sha256']))
    verifier_raw=read_checked(verifier_path,plan['source_verifier_sha256'])
    for name,digest in registry['repository_inputs'].items():read_checked(args.repo/name,digest)
    pins=_contract.strict_json(read_checked(verifier_path.with_name('input-pins.json'),plan['input_pins_sha256']))
    if pins!=registry['repository_inputs'] or freeze['input_archives']!=registry['input_archives']:raise ValueError('source pins mismatch')
    for path,digest in plan['code_sha256'].items():read_checked(args.repo/path,digest)
    runtime=_contract.strict_json(read_checked(HERE/'runtime-candidate.lock.json',plan['runtime_lock_sha256']));validate_runtime(runtime,args.wheelhouse)
    adapter=load_module('candidate_registry',verifier_path,verifier_raw)
    records=sorted(adapter.verify_inputs(registry,{'repo':args.repo,'artifacts':args.artifacts}),key=lambda r:r['record_id'])
    if len(records)!=16 or len(freeze['samples'])!=16:raise ValueError('exact cohort required')
    for r,f in zip(records,freeze['samples']):
        for key in ('record_id','slot_id','target_class','target_text','preset_voice_id','pcm_sha256','frames'):
            if r[key]!=f[key]:raise ValueError('cohort identity mismatch')
        prefix,rms=prefix_from_pcm(r['pcm_bytes'])
        if not _contract.strict_equal(rms,f['rms']) or prefix!=f['prefix_frames'] or len(rms)!=f['input_frames']:raise ValueError('frozen PCM prefix mismatch')
        if sha(r['pcm_bytes'])!=r['pcm_sha256'] or len(r['pcm_bytes'])!=r['frames']*2 or r['sample_rate_hz']!=16000 or r['channels']!=1:raise ValueError('PCM binding mismatch')
        r.update(prefix_frames=prefix,input_frames=len(rms),role=f['role'])
    for role in ('train','probe'):
        selected=[r for r in records if r['role']==role]
        if [r['record_id'] for r in selected]!=plan[role+'_record_ids'] or Counter(r['target_class'] for r in selected)!=(Counter(KW1=4,KW2=8) if role=='train' else Counter(KW1=1,KW2=3)):raise ValueError('split changed')
    if {r['preset_voice_id'] for r in records if r['role']=='train'}!={'qwen-preset:serena','qwen-preset:uncle_fu','qwen-preset:dylan'} or {r['preset_voice_id'] for r in records if r['role']=='probe'}!={'qwen-preset:vivian'}:raise ValueError('group split changed')
    claim_path=args.claim_root/plan['campaign_claim_path']
    budget=Budget(args.output,claim_path,plan['campaign_id'])
    stage_name='preparation';update=0;failure_context={}
    try:
        for name in ('torch','model','frontend','frontend_spec'):
            if name in sys.modules:raise ValueError('numeric module preloaded')
        for name in ('frontend_spec','model','frontend'):
            relative='training/'+name+'.py';load_module(name,args.repo/relative,read_checked(args.repo/relative,plan['code_sha256'][relative]))
        from model import TinyStreamingRNN
        from frontend import features
        import torch
        if Path(torch.__file__).resolve()!=Path(sysconfig.get_paths()['purelib'])/'torch/__init__.py':raise ValueError('unexpected Torch import root')
        torch.use_deterministic_algorithms(True);torch.manual_seed(plan['seed'])
        model=TinyStreamingRNN(**plan['model'])
        state_hash=lambda state:sha(b''.join(bytes(v.detach().contiguous().view(torch.uint8).reshape(-1).tolist()) for v in state.values()))
        initial={k:v.detach().clone() for k,v in model.state_dict().items()};initial_sha=state_hash(initial)
        batches={}
        for role in ('train','probe'):
            selected=[r for r in records if r['role']==role];tensors=[]
            for row in selected:
                pcm=array('h');pcm.frombytes(row['pcm_bytes'])
                if sys.byteorder!='little':pcm.byteswap()
                f=features(torch.tensor(pcm,dtype=torch.float32)/32768.0,**plan['frontend'])
                if not torch.isfinite(f).all() or len(f)!=row['input_frames'] or len(f)<7:raise ValueError('invalid features')
                tensors.append(f)
            lengths=torch.tensor([len(f) for f in tensors],dtype=torch.long);x=torch.nn.utils.rnn.pad_sequence(tensors,batch_first=True)
            targets=[plan['targets_by_text'][r['target_text']] for r in selected]
            y=torch.tensor([v for t in targets for v in t],dtype=torch.long);yl=torch.tensor([len(t) for t in targets],dtype=torch.long)
            batches[role]=(selected,x,lengths,targets,y,yl)
        atomic_json(args.output/'binding.json',dict(schema='grouped-input-binding-v1',plan_sha256=sha((HERE/'plan.json').read_bytes()),initial_weight_sha256=initial_sha,registry_sha256=plan['registry_sha256'],cohort_sha256=plan['cohort_sha256'],runtime_lock_sha256=plan['runtime_lock_sha256'],seed=plan['seed'],threads={'intraop':torch.get_num_threads(),'interop':torch.get_num_interop_threads()},campaign_id=plan['campaign_id'],campaign_claim_path=plan['campaign_claim_path'],campaign_claim_sha256=sha(claim_path.read_bytes()),max_model_forward_calls=1218,**DENIALS))
        loss_fn=torch.nn.CTCLoss(blank=0,reduction='none',zero_infinity=False)
        endpoints={};final_states={}
        def forward(kind,stage,step,role):
            nonlocal failure_context
            selected,x,lengths,targets,y,yl=batches[role]
            failure_context={'phase':'forward','kind':kind,'record_ids':[r['record_id'] for r in selected]}
            budget.before_forward(kind,stage,step)
            lp=model(x).log_softmax(2)
            budget.after_forward(kind,stage,step)
            nll=loss_fn(lp,y,lengths,yl)
            anchors=torch.stack([-lp[:r['prefix_frames'],i,0].sum() for i,r in enumerate(selected)])
            # Probe weights are descriptive macro class weights, never optimized.
            w=torch.tensor(weights([r['target_class'] for r in selected],stage if role=='train' else 'B'))
            ctc_loss=(nll/lengths*w).sum();anchor_loss=(anchors/lengths*w).sum();loss=ctc_loss+anchor_loss
            failure_context={'phase':'loss','nonfinite_log_probs':int((~torch.isfinite(lp)).sum()),'ctc_nll':[float(v) if torch.isfinite(v) else None for v in nll]}
            if not torch.isfinite(lp).all() or not torch.isfinite(nll).all() or not torch.isfinite(anchors).all() or not torch.isfinite(loss):raise ValueError('nonfinite numerical output')
            rows=[]
            with torch.no_grad():
                for i,r in enumerate(selected):
                    length=int(lengths[i]);prefix=r['prefix_frames'];p=lp[:length,i,:].detach();path=p.argmax(1).tolist();decoded=collapse(path)
                    alt={r['target_class']:float(nll[i].detach())} if kind=='training' else {}
                    alternatives=[] if kind=='training' else [('KW1',[1,2,3,4]),('KW2',[3,4,3,4]),('empty',[])]
                    for label,tokens in alternatives:
                        v=loss_fn(p[:,None,:],torch.tensor(tokens,dtype=torch.long),torch.tensor([length]),torch.tensor([len(tokens)]))[0]
                        if not torch.isfinite(v):raise ValueError('nonfinite alternate CTC')
                        alt[label]=float(v)
                    rows.append(dict(record_id=r['record_id'],target_class=r['target_class'],preset_voice_id=r['preset_voice_id'],pcm_sha256=r['pcm_sha256'],input_frames=length,target_length=len(targets[i]),target_tokens=targets[i],raw_argmax_tokens=path,decode_tokens=decoded,complete_match=decoded==targets[i],ctc_nll=float(nll[i].detach()),ctc_nll_per_input_frame=float((nll[i]/lengths[i]).detach()),anchor_nll=float(anchors[i].detach()),anchor_nll_per_input_frame=float((anchors[i]/lengths[i]).detach()),total_loss_per_input_frame=float(((nll[i]+anchors[i])/lengths[i]).detach()),weighted_total_loss=float(((nll[i]+anchors[i])/lengths[i]*w[i]).detach()),loss_weight=float(w[i]),blank_argmax_fraction=path.count(0)/length,blank_mean_probability=float(p[:,0].exp().mean()),log_probs=p.tolist(),ctc_nll_by_target=alt,prefix_frames=prefix,prefix_nonblank_argmax_frames=sum(t!=0 for t in path[:prefix]),prefix_blank_probability_min=float(p[:prefix,0].exp().min()) if prefix else None,prefix_blank_probability_max=float(p[:prefix,0].exp().max()) if prefix else None,first_nonblank_frame=next((j for j,t in enumerate(path) if t!=0),None)))
            rates={c:sum(r['complete_match'] for r in rows if r['target_class']==c)/sum(r['target_class']==c for r in rows) for c in ('KW1','KW2')}
            evidence=dict(schema='grouped-forward-payload-v1',kind=kind,stage=stage,update=step,role=role,timing='pre-step' if kind=='training' else 'post-step' if step else 'initial',model_forward_reserved=budget.forward_count,optimizer_calls_reserved=budget.count,samples=rows,ctc_loss=float(ctc_loss.detach()),anchor_loss=float(anchor_loss.detach()),finite_loss=float(loss.detach()),blank_lambda=1,per_class_complete_rate=rates,macro_complete_rate=sum(rates.values())/2,**DENIALS)
            atomic_json(args.output/f'payload-{budget.forward_count:04d}.json',evidence)
            return loss,evidence
        def evaluate(kind,stage,step,role):
            model.eval();rng=torch.get_rng_state().clone();before=state_hash(model.state_dict())
            with torch.no_grad():loss,evidence=forward(kind,stage,step,role)
            if before!=state_hash(model.state_dict()) or not torch.equal(rng,torch.get_rng_state()):raise ValueError('observation mutated state')
            name=f'{stage}-{step:04d}.json' if kind=='observation' else f'probe-{stage}-{step:04d}.json'
            atomic_json(args.output/name,evidence)
            return {'fit_success':all(r['complete_match'] for r in evidence['samples']),'prefix_success':all(r['prefix_nonblank_argmax_frames']==0 for r in evidence['samples']),'macro_complete_rate':evidence['macro_complete_rate'],'per_class_complete_rate':evidence['per_class_complete_rate']}
        for stage in plan['stages']:
            stage_name=stage['name'];update=0;model.load_state_dict(initial);torch.manual_seed(plan['seed'])
            if state_hash(model.state_dict())!=initial_sha:raise ValueError('initialization mismatch')
            atomic_json(args.output/f'{stage_name}-initialization.json',dict(initial_weight_sha256=initial_sha,seed=plan['seed']))
            op=plan['optimizer'];optimizer=torch.optim.Adam(model.parameters(),lr=op['lr'],betas=tuple(op['betas']),eps=op['eps'],weight_decay=op['weight_decay'])
            evaluate('observation',stage_name,0,'train')
            for update in range(1,601):
                model.train();optimizer.zero_grad(set_to_none=True)
                loss,evidence=forward('training',stage_name,update,'train');failure_context={'phase':'backward'};loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('invalid gradients')
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),op['gradient_clip'],error_if_nonfinite=True)
                before=state_hash(model.state_dict());budget.before_step(stage_name,update);failure_context={'phase':'optimizer_step'};optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()) or before==state_hash(model.state_dict()):raise ValueError('invalid updated weights')
                # LP stays in one payload file; receipt binds it by digest, avoiding duplication.
                payload=f'payload-{budget.forward_count:04d}.json'
                budget.after_step(stage_name,update,dict(status='step_succeeded',stage=stage_name,update=update,payload_file=payload,payload_sha256=sha((args.output/payload).read_bytes()),finite_loss=evidence['finite_loss'],ctc_loss=evidence['ctc_loss'],anchor_loss=evidence['anchor_loss'],blank_lambda=1,finite_grad_norm=float(norm),updated_weights_finite=True,weights_changed=True))
                if update in stage['checkpoints']:endpoints[stage_name]=evaluate('observation',stage_name,update,'train')
            final_states[stage_name]={k:v.detach().clone() for k,v in model.state_dict().items()}
        probe_endpoints={}
        for stage_name in ('A','B'):
            update=600;model.load_state_dict(final_states[stage_name]);probe_endpoints[stage_name]=evaluate('probe',stage_name,600,'probe')
        budget.complete()
        atomic_json(args.output/'terminal.json',dict(schema='grouped-terminal-v1',status='completed',primary_endpoint_step=600,train_endpoints=endpoints,probe_endpoints=probe_endpoints,optimizer_calls_reserved=budget.count,model_forward_calls_reserved=budget.forward_count,probe_used_for_selection=False,**DENIALS))
    except BaseException as exc:
        if not (args.output/'failure.json').exists():atomic_json(args.output/'failure.json',dict(schema='grouped-failure-v1',status='failed_no_retry',stage=stage_name,update=update,optimizer_calls_reserved=budget.count,model_forward_calls_reserved=budget.forward_count,error_type=type(exc).__name__,context=failure_context,error='Stopped on numerical, evidence, budget or output invariant; no retry or resume',**DENIALS))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('repo','artifacts','wheelhouse','output','claim-root'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True,help='Externally reviewed enabled-plan identity')
    args=p.parse_args()
    try:run(args)
    except BaseException as exc:
        # Preflight failures also get durable evidence, without borrowing an old claim.
        # Never write into an existing output, which may belong to an earlier run.
        if not args.output.exists() and args.output.absolute().resolve()==args.output.absolute():
            args.output.mkdir(exist_ok=False)
            atomic_json(args.output/'failure.json',dict(schema='grouped-failure-v1',status='failed_no_retry',stage='preflight',update=0,optimizer_calls_reserved=0,model_forward_calls_reserved=0,error_type=type(exc).__name__,context={},error='Preflight failed before numerical execution',**DENIALS))
        raise
