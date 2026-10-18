"""Prepared numerical probe. Run only through approved launch_probe.py.
Synthetic feature tensors only; does not read PCM or any private tensor cache.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ACTIVE_REPORT = None
ACTIVE_OUT = None

def main():
    global ACTIVE_REPORT, ACTIVE_OUT
    if os.environ.get('MDTC_PROBE_AUTHORIZED') != '1':
        raise SystemExit('Use reviewed, parent-authorized launch_probe.py')
    out = Path(sys.argv[1]).resolve()
    counts = {'forward_reserved':0,'backward_reserved':0,'optimizer_steps':0,'audio_reads':0}
    report = {'status':'running','counts':counts,'checks':[], 'tolerance':{'atol':1e-5,'rtol':1e-5,'scope':'same-runtime synthetic FP32; not old C conformance'}}
    ACTIVE_REPORT, ACTIVE_OUT = report, out
    def persist(): (out/'probe-result.json').write_text(json.dumps(report,indent=2)+'\n')
    persist()
    import torch
    from model import build_model, event_loss
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    if torch.__version__ != '2.11.0+cpu': raise RuntimeError('runtime version mismatch')
    model = build_model([0.]*80,[1.]*80,cmvn_identity={'scope':'synthetic_fixture_only'}).eval()
    assert sum(p.numel() for p in model.parameters()) == 42594
    def call(x,c=None,train_lengths=None):
        if counts['forward_reserved'] >= 600: raise RuntimeError('forward budget')
        counts['forward_reserved'] += 1; persist()
        return model(x,c) if train_lengths is None else model.train_padded(x,train_lengths)
    def equal(a,b,name):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise AssertionError(name+' nonfinite')
        diff=(a-b).abs(); over=diff > (1e-5+1e-5*b.abs())
        indices=over.nonzero()
        item={'name':name,'max_abs':float(diff.max()),'max_rel_floor1e_12':float((diff/b.abs().clamp_min(1e-12)).max()),'over_tolerance':int(over.sum()),'first_over_index':None if not bool(over.any()) else indices[0].tolist()}
        report['checks'].append(item);persist()
        if bool(over.any()): raise AssertionError(name+' exceeds fixed tolerance')
    x=torch.randn(1,401,80,dtype=torch.float32)
    z,c=call(x)
    assert not z.requires_grad and not c.requires_grad and tuple(c.shape)==(1,32,376)
    for chunk in (1,7,31):
        pieces=[]; state=None
        for start in range(0,401,chunk):
            y,state=call(x[:,start:start+chunk],state); pieces.append(y)
        equal(torch.cat(pieces,1),z,f'chunk{chunk}_logits'); equal(state,c,f'chunk{chunk}_cache')
    future=x.clone();future[:,200:]+=3
    changed,_=call(future);equal(changed[:,:200],z[:,:200],'future_cannot_change_past')
    # Independent stream histories, interleaved calls, then a fresh session reset.
    other=torch.randn_like(x)
    oz,oc=call(other);a,ac=call(x[:,:199]);b,bc=call(other[:,:199])
    a2,ac=call(x[:,199:],ac);b2,bc=call(other[:,199:],bc)
    equal(torch.cat([a,a2],1),z,'stream_a');equal(torch.cat([b,b2],1),oz,'stream_b')
    equal(ac,c,'stream_a_cache');equal(bc,oc,'stream_b_cache')
    reset,rc=call(x);equal(reset,z,'session_reset');equal(rc,c,'reset_cache')
    invalid=[torch.empty(0,1,80),torch.empty(1,0,80),x.double(),torch.full_like(x,float('nan'))]
    for item in invalid:
        try: call(item)
        except ValueError: pass
        else: raise AssertionError('invalid feature accepted')
    for cache in (c.double(),torch.full_like(c,float('inf')),c[:,:,:1]):
        try:call(x[:,:1],cache)
        except ValueError:pass
        else:raise AssertionError('invalid cache accepted')
    model.train()
    assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm1d))
    lengths=torch.tensor([401,257],dtype=torch.int64)
    batch=torch.cat([x,other],0); padded=batch.clone();padded[1,257:]=1000
    y=call(batch,train_lengths=lengths)
    yp=call(padded,train_lengths=lengths)
    equal(y[0],z[0],'batch_vs_single_stream_a');equal(y[1,:257],oz[0,:257],'batch_vs_single_stream_b')
    equal(y[0],yp[0],'padding_sample0');equal(y[1,:257],yp[1,:257],'padding_sample1')
    labels=torch.tensor([[1.,0.],[0.,1.]])
    weights=torch.tensor([[1.,0.],[.25,.25]])
    loss=event_loss(y,lengths,labels,weights);other_loss=event_loss(yp,lengths,labels,weights)
    equal(loss.detach(),other_loss.detach(),'padding_loss')
    changed_labels=labels.clone();changed_labels[0,1]=1.
    equal(loss.detach(),event_loss(y,lengths,changed_labels,weights).detach(),'unknown_label_abstention')
    try:event_loss(y,lengths,labels,torch.zeros_like(weights))
    except ValueError:pass
    else:raise AssertionError('all unknown accepted')
    toy=torch.tensor([[[-9.,-9.],[9.,-9.],[-9.,9.],[999.,999.]]])
    tl=torch.tensor([3],dtype=torch.int64); tw=torch.ones(1,2); ty=torch.ones(1,2)
    assert float(event_loss(toy,tl,ty,tw))<0.001
    for badlen in (torch.tensor([0]),torch.tensor([5]),torch.tensor([1.5])):
        try:event_loss(toy,badlen,ty,tw)
        except ValueError:pass
        else:raise AssertionError('illegal loss lengths accepted')
    y.retain_grad();counts['backward_reserved']+=1;persist();loss.backward()
    assert y.grad[0,:,1].abs().sum()==0 and y.grad[1,257:].abs().sum()==0
    norms={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert norms['project.0.weight']>0 and norms['head.weight']>0
    # Nested model class is intentionally never pickled; persist state_dict/config.
    torch.save({'state_dict':model.state_dict(),'config':json.loads((ROOT/'config.json').read_text()),'cmvn_identity':model.cmvn_identity,'synthetic_fixture_only':True,'untrained':True,'config_execution_counts_scope':'preparation snapshot; current probe counters are in probe-result.json'},out/'synthetic-state.pt')
    report.update(status='passed',gradient_norms=norms,loss=float(loss.detach()),torch=torch.__version__,threads=1,seed=0)
    persist()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if ACTIVE_REPORT is not None:
            ACTIVE_REPORT.update(status='failed',error=type(exc).__name__+': '+str(exc))
            (ACTIVE_OUT/'probe-result.json').write_text(json.dumps(ACTIVE_REPORT,indent=2)+'\n')
        raise
