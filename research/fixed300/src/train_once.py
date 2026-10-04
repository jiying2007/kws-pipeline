"""One fixed300 attempt. Imports Torch only after a released child context."""
import argparse,hashlib,json,math,os,random,struct
from pathlib import Path
from common import check_context,read_json,require,safe,write_json,write_bytes,digest
from contracts import CMVN,HEAD,TOKENS,STATE_SHA,cohort_weights,collapse,minimum_frames
from control import configure_cpu,load_a20,tensor_ctc_loss
ROOT=Path(__file__).resolve().parents[1]

def fresh_optimizer(model):
    import torch
    params=dict(model.named_parameters());head=[params[n]for n in HEAD];encoder=[p for n,p in params.items()if n not in HEAD]
    require(sum(p.numel()for p in head)==846 and sum(p.numel()for p in encoder)==389674,'exact disjoint parameter groups')
    optimizer=torch.optim.AdamW([{'params':head,'lr':1e-4},{'params':encoder,'lr':1e-5}],betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,maximize=False,capturable=False,differentiable=False,fused=False)
    require(not optimizer.state,'fresh empty optimizer');return optimizer

def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def diagnostics(logits,rows,per,total,origin):
    result=[]
    for i,r in enumerate(rows):
        greedy=collapse(logits[i,:r['model_rows']].argmax(-1).tolist());target=r['target_ids']
        result.append(dict(recording=r['recording'],cohort=r['cohort'],label=r['label'],target_ids=target,normalized_ctc=float(per[i]),greedy_ids=greedy,edit_distance=edit_distance(greedy,target),greedy_exact=greedy==target))
    return dict(schema='a20-fixed300-train-diagnostic-v1',origin=origin,rows=result,cohort_weighted_ctc=float(total),D20_mean=math.fsum(x['normalized_ctc']for x in result[:20])/20,Qwen12_mean=math.fsum(x['normalized_ctc']for x in result[20:])/12,selection_allowed=False,wake_events_not_scored=True)

def tensor_identity(model,schema,check_initial=False):
    import torch
    state=model.state_dict();require(list(state)==[e['name']for e in schema['tensors']],'exact state order')
    canonical=hashlib.sha256();payload=bytearray();entries=[]
    for e in schema['tensors']:
        t=state[e['name']];require(type(t)is torch.Tensor and t.device.type=='cpu'and t.dtype==torch.float32 and list(t.shape)==e['shape']and torch.isfinite(t).all().item(),'finite tensor schema')
        raw=t.detach().contiguous().numpy().tobytes();h=hashlib.sha256(raw).hexdigest();require(len(raw)==e['bytes'],'tensor byte count')
        if check_initial or e['name']in CMVN:require(h==e['sha256'],'original tensor or CMVN identity')
        canonical.update(e['name'].encode());canonical.update(raw);payload.extend(raw);entries.append(dict(name=e['name'],shape=e['shape'],bytes=len(raw),sha256=h))
    require(len(payload)==1565280,'exact tensor payload size')
    return dict(state_sha256=canonical.hexdigest(),payload_sha256=hashlib.sha256(payload).hexdigest(),tensors=entries),bytes(payload)

def optimizer_identity(model,optimizer):
    import torch
    params=dict(model.named_parameters());order=list(HEAD)+[n for n in params if n not in HEAD]
    require(len(optimizer.state)==len(params)==28,'optimizer state for all28 parameters')
    expected=(dict(lr=1e-4,params=[params[n]for n in HEAD]),dict(lr=1e-5,params=[p for n,p in params.items()if n not in HEAD]))
    for group,want in zip(optimizer.param_groups,expected):
        require(group['lr']==want['lr']and len(group['params'])==len(want['params'])and all(x is y for x,y in zip(group['params'],want['params'])),'optimizer group order')
        require(group['betas']==(.9,.999)and group['eps']==1e-8 and group['weight_decay']==0,'optimizer hyperparameters')
        require(all(group[k]is False for k in ('amsgrad','maximize','capturable','differentiable','foreach','fused')),'optimizer options')
    entries=[];h=hashlib.sha256()
    for n in order:
        p=params[n];state=optimizer.state[p];require(set(state)=={'step','exp_avg','exp_avg_sq'},'AdamW state keys')
        require(state['step'].device.type=='cpu'and state['step'].dtype==torch.float32 and state['step'].ndim==0 and state['step'].item()==300,'all optimizer step300')
        tensors=[]
        for key in ('step','exp_avg','exp_avg_sq'):
            t=state[key];require(t.device.type=='cpu'and t.dtype==torch.float32 and torch.isfinite(t).all().item(),'finite optimizer state')
            if key!='step':require(t.shape==p.shape,'optimizer moment shape')
            raw=t.detach().contiguous().numpy().tobytes();h.update(n.encode());h.update(key.encode());h.update(raw)
            tensors.append(dict(key=key,shape=list(t.shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        entries.append(dict(parameter=n,tensors=tensors))
    return dict(schema='a20-fixed300-optimizer-state-v1',canonical_sha256=h.hexdigest(),parameter_order=order,entries=entries,step=300,loaded_historical_optimizer=False)

class Journal:
    def __init__(self,path):self.f=Path(path).open('xb');self.started=dict(forward=0,backward=0,update=0);self.completed=dict(forward=0,backward=0,update=0)
    def emit(self,kind,status,step):
        require(kind in self.started and status in ('started','completed'),'journal event vocabulary')
        target=self.started if status=='started'else self.completed;target[kind]+=1
        require(self.completed[kind]<=self.started[kind],'journal ordering')
        raw=(json.dumps(dict(kind=kind,status=status,step=step),sort_keys=True)+'\n').encode();self.f.write(raw);self.f.flush();os.fsync(self.f.fileno())
    def close(self):self.f.close()

def load_batch():
    import numpy as np,torch
    rows=read_json(ROOT/'metadata/TRAIN32.json')['rows'];require(len(rows)==32 and len({r['recording']for r in rows})==32 and all(r['role']=='train'for r in rows),'exact train32')
    cohort_weights([r['cohort']for r in rows]);require(sum(r['model_rows']for r in rows)==1812 and max(r['model_rows']for r in rows)==95,'training geometry')
    prep=ROOT/'work/prepared';manifest=read_json(prep/'native-features.json');features=manifest['rows'];require(len(features)==32,'32 features')
    x=torch.zeros((32,95,400),dtype=torch.float32)
    for i,(r,f)in enumerate(zip(rows,features)):
        require(f['recording']==r['recording']and f['pcm_sha256']==r['pcm_sha256']and f['wav_sha256']==r['wav_sha256'],'feature source identity and order')
        require(f['shape']==[r['model_rows'],400]and f['bytes']==r['model_rows']*1600 and f['normalization']=='PRE_CMVN','feature geometry and normalization')
        require(f['file']=='native/'+r['recording']+'.f32le','feature path')
        require([TOKENS.index(c)for c in r['text']]==r['target_ids']and len(r['target_ids'])in (3,4)and minimum_frames(r['target_ids'])<=r['model_rows'],'full CTC target')
        p=safe(prep,f['file']);require(p.stat().st_size==f['bytes']and digest(p)==f['sha256'],'feature raw identity')
        arr=np.frombuffer(p.read_bytes(),dtype='<f4').copy().reshape(f['shape']);require(np.isfinite(arr).all(),'finite features');x[i,:len(arr)]=torch.from_numpy(arr)
    initial=read_json(prep/'initial-control.json');raw=prep/'a20-initial-logits.f32le'
    require(raw.stat().st_size==72960 and digest(raw)==initial['raw_sha256']=='91568a8c774c0811f7c3262cad009d6a5d6e62333f02fe8137757f2062c94c99','saved initial logits')
    require(initial['source_order']==[r['recording']for r in rows]and initial['lengths']==[r['model_rows']for r in rows]and initial['native_feature_hashes']==[f['sha256']for f in features]and initial['state_sha256']==STATE_SHA,'saved initial binding')
    initial_logits=torch.from_numpy(np.frombuffer(raw.read_bytes(),dtype='<f4').copy().reshape(32,95,6))
    return x,rows,initial,initial_logits

def save_checkpoint(model,optimizer,schema,rows,context,out):
    import torch,numpy as np
    identity,payload=tensor_identity(model,schema);opt=optimizer_identity(model,optimizer)
    write_bytes(out/'terminal-state-payload.f32le',payload,2*1024**2);write_json(out/'terminal-tensors.json',identity);write_json(out/'optimizer-state.json',opt)
    ns=np.random.get_state();obj=dict(schema='a20-fixed300-checkpoint-v1',model=model.state_dict(),optimizer=optimizer.state_dict(),step=300,seed=610104,symbols=list(TOKENS),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=dict(name=ns[0],keys=ns[1].tolist(),position=ns[2],has_gauss=ns[3],cached_gaussian=ns[4]),source_freeze_sha256=context['source_freeze_sha256'],release_sha256=context['release_sha256'],protocol_sha256=digest(ROOT/'PROTOCOL.json'),prepared_input_manifest_sha256=digest(ROOT/'metadata/PREPARED-INPUTS.json'),model_input_manifest_sha256=digest(ROOT/'metadata/MODEL-INPUTS.json'),train32_sha256=digest(ROOT/'metadata/TRAIN32.json'),initial_state_sha256=STATE_SHA,terminal_state_sha256=identity['state_sha256'],optimizer_canonical_sha256=opt['canonical_sha256'],resume_allowed=False)
    p=out/'terminal-step300.pt'
    with p.open('xb')as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    require(p.stat().st_size<=6*1024**2,'checkpoint byte cap')
    restored=torch.load(p,map_location='cpu',weights_only=True);require(set(restored)==set(obj)and restored['step']==300 and restored['resume_allowed']is False,'checkpoint readback envelope')
    require(list(restored['model'])==list(obj['model']),'checkpoint ordered model keys')
    for n,t in model.state_dict().items():require(restored['model'][n].dtype==t.dtype and restored['model'][n].shape==t.shape and torch.equal(restored['model'][n],t),'checkpoint model readback')
    require(restored['optimizer']['param_groups']==obj['optimizer']['param_groups'],'checkpoint optimizer group readback')
    require(torch.equal(restored['torch_rng'],obj['torch_rng']),'checkpoint RNG readback')
    for k in set(obj)-{'model','optimizer','torch_rng'}:require(restored[k]==obj[k],'checkpoint metadata readback')
    for k,v in optimizer.state_dict()['state'].items():
        for n,t in v.items():require(torch.equal(restored['optimizer']['state'][k][n],t),'checkpoint optimizer readback')
    write_json(out/'checkpoint.json',dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p),state_sha256=identity['state_sha256'],optimizer_sha256=opt['canonical_sha256'],strict_weights_only_readback=True,endpoint=300,selection=False))
    return identity

def run(context):
    check_context(ROOT,context,'training');configure_cpu()
    import torch
    x,rows,initial,initial_logits=load_batch();schema=read_json(ROOT/'metadata/A20-TENSOR-SCHEMA.json');bindings=read_json(ROOT/'metadata/CONTROL-BINDINGS.json')
    model=load_a20(ROOT/'work/inputs',bindings);identity,_=tensor_identity(model,schema,True);require(identity['state_sha256']==STATE_SHA,'original A20 state')
    out=ROOT/'work/artifact';write_json(out/'initial-train-diagnostics.json',diagnostics(initial_logits,rows,initial['source_normalized_ctc'],initial['cohort_weighted_ctc'],'REUSED_VERIFIED_SAVED_CONTROL_NO_FORWARD'))
    optimizer=fresh_optimizer(model);model.train();require(all(m.training for m in model.modules()),'frozen training mode')
    journal=Journal(out/'call-journal.jsonl');history=[];cmvn_calls=[];dropout_calls=[]
    def cmvn_hook(module,args,output):
        require(len(args)==1 and args[0]is x and tuple(output.shape)==tuple(x.shape)and torch.isfinite(output).all().item(),'one unchanged PRE-CMVN normalization');cmvn_calls.append(1)
    hook=model.global_cmvn.register_forward_hook(cmvn_hook)
    dhooks=[m.register_forward_hook(lambda *unused:dropout_calls.append(1))for m in model.modules()if isinstance(m,torch.nn.Dropout)]
    def forward(step,grad):
        before=len(cmvn_calls);journal.emit('forward','started',step)
        if grad:logits,_=model(x,None)
        else:
            with torch.no_grad():logits,_=model(x,None)
        journal.emit('forward','completed',step);require(len(cmvn_calls)==before+1 and not dropout_calls,'CMVN once and no invoked dropout')
        return logits
    try:
        for step in range(1,301):
            optimizer.zero_grad(set_to_none=True);logits=forward(step,True)
            loss,per=tensor_ctc_loss(logits,[r['model_rows']for r in rows],[r['target_ids']for r in rows],[r['cohort']for r in rows])
            if step==1:
                # Host-dependent FP32 deltas are retained, never a new bitwise gate.
                delta=(logits.detach()-initial_logits).abs();base=initial_logits.abs().clamp_min(1e-6)
                write_json(out/'first-forward-binding.json',dict(state_sha256=identity['state_sha256'],train32_sha256=digest(ROOT/'metadata/TRAIN32.json'),feature_manifest_sha256=digest(ROOT/'work/prepared/native-features.json'),cmvn_calls=1,source_order=[r['recording']for r in rows],same_state_input_order_bound=True,saved_logits_sha256=initial['raw_sha256'],current_logits_sha256=hashlib.sha256(logits.detach().contiguous().numpy().tobytes()).hexdigest(),max_abs_logit_delta=float(delta.max()),max_relative_logit_delta_denominator_floor_1e_6=float((delta/base).max()),ctc_delta=float(loss.detach())-initial['cohort_weighted_ctc'],ctc_per_source_delta=[float(v.detach())-u for v,u in zip(per,initial['source_normalized_ctc'])],numerical_deltas='DIAGNOSTIC_ONLY_NO_CROSS_HOST_BITWISE_GATE'))
            journal.emit('backward','started',step);loss.backward();journal.emit('backward','completed',step)
            require(all(p.grad is not None and torch.isfinite(p.grad).all().item()for p in model.parameters()),'finite full gradients')
            norm=torch.nn.utils.clip_grad_norm_(list(model.parameters()),1.0,error_if_nonfinite=True,foreach=False)
            journal.emit('update','started',step);optimizer.step();journal.emit('update','completed',step)
            require(all(torch.isfinite(p).all().item()for p in model.parameters()),'finite updated parameters')
            for e in schema['tensors']:
                if e['name']in CMVN:require(hashlib.sha256(model.state_dict()[e['name']].numpy().tobytes()).hexdigest()==e['sha256'],'CMVN unchanged')
            record=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm_before_clip=float(norm));history.append(record)
            with (out/'updates.jsonl').open('ab')as f:f.write((json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
        terminal=save_checkpoint(model,optimizer,schema,rows,context,out)
        # The endpoint remains train-mode, as the frozen original control; no dropout is invoked.
        logits=forward(300,False);loss,per=tensor_ctc_loss(logits,[r['model_rows']for r in rows],[r['target_ids']for r in rows],[r['cohort']for r in rows])
        write_bytes(out/'terminal-train-logits.f32le',logits.detach().contiguous().numpy().tobytes(),72960)
        write_json(out/'terminal-train-diagnostics.json',diagnostics(logits,rows,per,float(loss),'FIXED_STEP300_NO_SELECTION'))
        after,_=tensor_identity(model,schema);require(after==terminal,'endpoint diagnostic state unchanged')
        require(journal.started==journal.completed==dict(forward=301,backward=300,update=300)and len(cmvn_calls)==301,'exact complete call budget')
        write_json(out/'training-result.json',dict(schema='a20-fixed300-result-v1',status='SUCCESS',calls=journal.completed,cmvn_calls=301,optimizer_constructions=1,seed=610104,endpoint=300,training_source_count=32,feature_extractions=0,dev_calls=0,decoder_calls=0,native_parity='NOT_EVALUATED',acoustic_endpoint_evaluation='NOT_RUN',initial_control_reused=True,terminal_state_sha256=terminal['state_sha256']))
    finally:
        hook.remove()
        for h in dhooks:h.remove()
        journal.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-context',type=Path);a=p.parse_args()
    if a.execute_context is None:raise SystemExit('PREPARATION_ONLY: no exact training context')
    from safe_failure import save
    try:run(read_json(a.execute_context))
    except BaseException as e:save(ROOT,e,'training','FIXED300_ATTEMPT');raise SystemExit(1)
