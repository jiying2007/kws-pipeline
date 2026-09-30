"""Freeze synthetic full-donor reference BEFORE C evaluation. Local weights only."""
import hashlib, importlib.util, json, pathlib, sys
import numpy as np
import torch
BASE=pathlib.Path('/workspace/shared/kws-cfsmn-baseline')
OUT=pathlib.Path(sys.argv[1]);OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):
 p=OUT/name
 if p.exists():raise RuntimeError('refuse overwrite '+str(p))
 p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
spec=importlib.util.spec_from_file_location('baseline',BASE/'run_baseline.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.audited_sources();torch.set_num_threads(1);torch.set_num_interop_threads(1)
m=b.Donor().eval();state=m.state_dict();assert len(state)==30
manifest=[];payload=bytearray()
for name,t in state.items():
 a=t.detach().numpy().astype('<f4');assert np.isfinite(a).all()
 raw=a.tobytes();manifest.append(dict(name=name,shape=list(a.shape),offset=len(payload),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));payload.extend(raw)
assert len(payload)==3027732
p=OUT/'donor-local.f32';assert not p.exists();p.write_bytes(payload)
save('weights-manifest.json',dict(version=1,source_sha256=sha(BASE/'resources/base.pt'),payload_sha256=sha(p),payload_bytes=len(payload),tensors=manifest))
arrays={};reports=[]
inputs={'zero':np.zeros((9,400),np.float32),'impulse':np.eye(1,400,37,dtype=np.float32).repeat(17,0),'random':np.random.Generator(np.random.PCG64(734)).uniform(-1,1,(37,400)).astype(np.float32)}
inputs['impulse'][1:]=0
for name,a in inputs.items():
 arrays[name+'_input']=a
 for route,parts in [('whole',[len(a)]),('split',([1,3,9,8,16] if len(a)==37 else [1,len(a)-1]))]:
  cache=None;offset=0
  for ci,n in enumerate(parts):
   x=torch.from_numpy(a[offset:offset+n]).unsqueeze(0);prefix=f'{name}_{route}_{ci}'
   with torch.inference_mode():
    x=m.global_cmvn(x);arrays[prefix+'_cmvn']=x.numpy().copy()
    c=[None]*4 if cache is None else [cache[:,:,:,i:i+1] for i in range(4)]
    stages=[]
    for module in [m.backbone.in_linear1,m.backbone.in_linear2,m.backbone.relu]:
     x,_=module(x);stages.append(x.numpy().copy())
    for i,block in enumerate(m.backbone.fsmn):
     x,_=block[0](x);stages.append(x.numpy().copy())
     x,c[i]=block[1]((x,c[i]));stages.append(x.numpy().copy())
     x,_=block[2](x);stages.append(x.numpy().copy())
     x,_=block[3](x);stages.append(x.numpy().copy())
    for module in [m.backbone.out_linear1,m.backbone.out_linear2]:
     x,_=module(x);stages.append(x.numpy().copy())
    cache=torch.cat(c,dim=-1)
   for i,s in enumerate(stages):arrays[prefix+f'_stage{i}']=s
   arrays[prefix+'_cache']=cache.numpy().copy();reports.append(dict(prefix=prefix,input=name,offset=offset,frames=n,reset=ci==0))
   offset+=n
np.savez(OUT/'goldens.npz',**arrays)
save('reference-receipt.json',dict(schema=1,generator_sha256=sha(pathlib.Path(__file__)),driver_sha256=sha(BASE/'run_baseline.py'),goldens_sha256=sha(OUT/'goldens.npz'),weights_manifest_sha256=sha(OUT/'weights-manifest.json'),torch=torch.__version__,numpy=np.__version__,seed=734,range=[-1,1],cases=reports,gate=dict(atol=1e-4,rtol=1e-5),scope='synthetic-only; no corpus inference; weights local-only'))
print('FROZEN',sha(OUT/'reference-receipt.json'),len(reports),'calls')
