"""Freeze exact original Python Hamming 42x3 PCM calls before native output."""
import hashlib,importlib.util,json,pathlib,resource,sys,wave
import numpy as np
import torch
BASE=pathlib.Path('/workspace/shared/kws-cfsmn-baseline');OUT=pathlib.Path('/workspace/shared/kws-fsmn-pcm-diagnostic-v1');R=OUT/'reference'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 if p.exists():raise ValueError('refuse overwrite '+str(p))
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
s=importlib.util.spec_from_file_location('baseline',BASE/'run_baseline.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
def setup():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 frontend,ns,hashes=b.audited_sources();conditions=b.inputs();tokens=(BASE/'resources/tokens_2599.txt').read_text().splitlines()
 return frontend,ns,hashes,conditions,tokens

def main():
 frontend,ns,hashes,conditions,tokens=setup()
 spec=dict(historical_readbacks={c:sha(BASE/'results'/f'hamming-{c}.json') for c in conditions},schema=1,script_sha256=sha(pathlib.Path(__file__)),driver_sha256=sha(BASE/'run_baseline.py'),sources=hashes,conditions={c:[dict(recording=r['recording'],file_sha256=r['file_sha256'],pcm_sha256=r['pcm_sha256'],frames=r['frames']) for r in rows] for c,(root,rows) in conditions.items()},window='hamming',calls_samples=4800,scope='126 unchanged original-reference runs only, no C/no training',python=sys.version,torch=torch.__version__,numpy=np.__version__,CPU_seconds_limit=600,output_limit_bytes=1500000000)
 if sys.argv[1]=='prepare':save(OUT/'reference-preflight.json',spec);print('FROZEN',sha(OUT/'reference-preflight.json'));return
 assert sys.argv[1]=='run' and json.loads((OUT/'reference-preflight.json').read_text())==spec
 resource.setrlimit(resource.RLIMIT_CPU,(600,600));R.mkdir(exist_ok=False)
 model=b.Donor().eval();trace={};handles=[]
 modules=[model.backbone.in_linear1,model.backbone.in_linear2,model.backbone.relu]+[part for block in model.backbone.fsmn for part in block]+[model.backbone.out_linear1,model.backbone.out_linear2]
 def hook(name):
  def fn(module,inp,out):
   trace[name]=(out[0] if isinstance(out,tuple) else out).detach().numpy().copy()
   if name=='cmvn':trace['splice']=inp[0].detach().numpy().copy()
  return fn
 for i,m in enumerate(modules):handles.append(m.register_forward_hook(hook(f'stage{i}')))
 handles.append(model.global_cmvn.register_forward_hook(hook('cmvn')))
 # Capture actual fbank return without altering features or operational state.
 class Frontend:
  def fbank(self,*a,**kw):
   out=frontend.fbank(*a,**kw);trace['fbank']=out.numpy().copy();return out
 reports=[];totalbytes=0
 with torch.inference_mode():
  for condition,(root,rows) in conditions.items():
   historical=json.loads((BASE/'results'/f'hamming-{condition}.json').read_text());old={r['recording']:r for r in historical['recordings']}
   for ri,row in enumerate(rows):
    k=b.make_spotter('hamming',model,Frontend(),ns,tokens);model.logs=[];arrays={};calls=[];events=[];fbank_count=0
    with wave.open(str(root/row['path'])) as f:pcm=f.readframes(f.getnframes())
    assert len(pcm)==row['frames']*2 and hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
    assert sha(root/row['path'])==row['file_sha256']
    for ci,start in enumerate(range(0,len(pcm),9600)):
     trace.clear();end=min(len(pcm),start+9600);old_count=0 if k.feature_remained is None else len(k.feature_remained);offset=k.feats_ctx_offset;first=k.feature_remained is None
     result=k.forward(pcm[start:end]);n=trace['fbank'].shape[0] if 'fbank' in trace else 0;centers=[]
     if n:
      ctx_rows=n-2 if first else n+old_count-4;base=0 if first else fbank_count-old_count+2
      centers=[base+j for j in range(offset,ctx_rows,3)];fbank_count+=n
     key=f'call{ci}'
     for name,a in trace.items():arrays[key+'_'+name]=a
     if 'stage20' in trace:
      logits=trace['stage20'];arrays[key+'_logits']=logits;arrays[key+'_probabilities']=torch.from_numpy(logits).softmax(2).numpy();assert logits.shape[1]==len(centers)
      arrays[key+'_cache']=k.in_cache.numpy().copy()
     if result.get('state')==1:events.append(dict(result,keyword_id=b.PHRASES[result['keyword']],available_audio_samples=end//2,eof_flush=False))
     calls.append(dict(index=ci,input_samples=(end-start)//2,available_audio_samples=end//2,fbank_rows=n,centers=centers,upstream_total_frames=k.total_frames,wave_remained=len(k.wave_remained),feature_remained=0 if k.feature_remained is None else len(k.feature_remained),offset=k.feats_ctx_offset,result=dict(result)))
    logits=torch.cat(model.logs,1)[0].numpy();assert hashlib.sha256(logits.tobytes()).hexdigest()==old[row['recording']]['logits_sha256'];assert events==old[row['recording']]['events']
    assert all(np.isfinite(a).all() for a in arrays.values())
    assert totalbytes+sum(a.nbytes for a in arrays.values())+4096*(len(arrays)+1)<1500000000
    filename=f'{condition}-{ri:02d}.npz';np.savez(R/filename,**arrays);totalbytes+=(R/filename).stat().st_size;assert totalbytes<1500000000
    reports.append(dict(condition=condition,recording=row['recording'],kind=row['kind'],keyword_id=row['keyword_id'],source_sha256=row['file_sha256'],pcm_sha256=row['pcm_sha256'],arrays_file=filename,arrays_sha256=sha(R/filename),calls=calls,events=events,logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest()))
 b.inputs();b.audited_sources()
 assert sha(pathlib.Path(__file__))==spec['script_sha256'] and sha(BASE/'run_baseline.py')==spec['driver_sha256']
 assert all(sha(BASE/'results'/f'hamming-{c}.json')==h for c,h in spec['historical_readbacks'].items())
 save(R/'receipt.json',dict(preflight_sha256=sha(OUT/'reference-preflight.json'),recordings=reports,clips=len(reports),array_bytes=totalbytes,historical_logits_and_events_exact=True,scope='original Python126reference frozen before native output'))
 print('REFERENCE COMPLETE',len(reports),totalbytes,sha(R/'receipt.json'))
if __name__=='__main__':main()
