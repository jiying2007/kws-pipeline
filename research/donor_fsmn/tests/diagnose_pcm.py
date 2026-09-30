"""One fixed126 PCM composition/behavior diagnosis. Old B remains failed."""
import ctypes as C,hashlib,importlib.util,json,pathlib,resource,shutil,subprocess,sys,traceback,wave
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=pathlib.Path('/workspace/shared/kws-cfsmn-baseline');E=pathlib.Path('/workspace/shared/kws-fsmn-pcm-diagnostic-v1');REF=E/'reference';OUT=E/'native'
F=C.POINTER(C.c_float)
class Model(C.Structure):_fields_=[('weights',F),('cache',C.c_float*5632),('fault',C.c_int)]
class Batch(C.Structure):
 _fields_=[('call_index',C.c_uint64),('available_samples',C.c_uint64),('call_samples',C.c_size_t),('waveform_samples',C.c_size_t),('fbank_rows',C.c_size_t),('splice_rows',C.c_size_t),('selected_rows',C.c_size_t),('is_final_short',C.c_uint32),('fbank',F),('rows',F),('centers',C.POINTER(C.c_uint64)),('wave_samples',C.c_uint32),('feature_count',C.c_uint32),('offset',C.c_uint32)]
CALLBACK=C.CFUNCTYPE(None,C.c_void_p,C.POINTER(Batch))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 if p.exists():raise ValueError('refuse overwrite '+str(p))
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=module('baseline',BASE/'run_baseline.py');manifest=module('manifest',ROOT/'manifest.py')
def ptr(a):return a.ctypes.data_as(F)
def setup():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 frontend,ns,hashes=b.audited_sources();conditions=b.inputs();tokens=(BASE/'resources/tokens_2599.txt').read_text().splitlines()
 return frontend,ns,hashes,conditions,tokens

def identity(hashes):
 cc=pathlib.Path(shutil.which('cc')).resolve();r=json.loads((REF/'receipt.json').read_text())
 files=[ROOT/n for n in ['pcm.c','pcm.h','splice.c','splice.h','fsmn.c','fsmn.h','manifest.py']]+[ROOT.parent/'donor_fbank'/n for n in ['donor_fbank.c','donor_fbank.h','frontend_tables.h','fft_twiddles.h','v2-contract.json']]
 return dict(schema=1,reference_sha256=sha(REF/'receipt.json'),reference_arrays={x['arrays_file']:x['arrays_sha256'] for x in r['recordings']},script_sha256=sha(pathlib.Path(__file__)),sources={str(p.relative_to(ROOT.parent)):sha(p) for p in files},upstream_sources=hashes,baseline_driver_sha256=sha(BASE/'run_baseline.py'),payload_sha256=manifest.PAYLOAD_SHA256,compiler=dict(path=str(cc),sha256=sha(cc),version=subprocess.check_output([str(cc),'--version'],text=True)),flags=['-std=c11','-O2','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off'],python=sys.version,numpy=np.__version__,torch=torch.__version__,scope='one fixed42x3Hamming PCM run; C frontend/splice/full2599 model with unchanged Python softmax/decoder; no training or threshold change',limits=dict(cpu_seconds=600,total_native_array_bytes=1500000000),gates=dict(logits=dict(atol=1e-4,rtol=1e-5),probabilities=dict(atol=1e-5,rtol=1e-5,row_sum_error=1e-5),event_fields_exact_including_score=True,score_tolerance_report_only=1e-5),old_B_still_failed=True)

def run(frontend,ns,hashes,conditions,tokens):
 pre=json.loads((E/'native-preflight.json').read_text());assert pre==identity(hashes)
 OUT.mkdir(exist_ok=False);resource.setrlimit(resource.RLIMIT_CPU,(600,600))
 cmd=[pre['compiler']['path'],*pre['flags'],'-shared','-fPIC',str(ROOT/'pcm.c'),str(ROOT/'splice.c'),str(ROOT/'fsmn.c'),str(ROOT.parent/'donor_fbank/donor_fbank.c'),'-lm','-o',str(OUT/'native.so')];subprocess.run(cmd,check=True)
 lib=C.CDLL(str(OUT/'native.so'));lib.donor_pcm_state_bytes.restype=C.c_size_t
 lib.donor_pcm_init.argtypes=[C.c_void_p];lib.donor_pcm_feed.argtypes=[C.c_void_p,C.POINTER(C.c_int16),C.c_size_t,CALLBACK,C.c_void_p];lib.donor_pcm_finish.argtypes=[C.c_void_p,CALLBACK,C.c_void_p]
 lib.df_init.argtypes=[C.POINTER(Model),F,C.c_size_t];lib.df_step.argtypes=[C.POINTER(Model),F,F,F];lib.df_cmvn.argtypes=[C.POINTER(Model),F,F]
 weights=np.frombuffer(manifest.load(pathlib.Path('/workspace/shared/kws-fsmn-ab-evidence/weights-manifest.json'),pathlib.Path('/workspace/shared/kws-fsmn-ab-evidence/donor-local.f32')),dtype='<f4').copy()
 reference=json.loads((REF/'receipt.json').read_text());refs={(x['condition'],x['recording']):x for x in reference['recordings']};assert len(refs)==reference['clips']==126
 assert set(refs)=={(c,r['recording']) for c,(_,rows) in conditions.items() for r in rows}
 reports=[];totalbytes=0
 dims=[140,250,250]+[128,128,250,250]*4+[140,2599]
 class NativeModel(torch.nn.Module):
  def __init__(self):
   super().__init__();self.state=Model();assert lib.df_init(C.byref(self.state),ptr(weights),len(weights))==0;self.trace={}
  def forward(self,features,unused_cache):
   a=features[0].numpy();logits=np.empty((len(a),2599),np.float32);traces=np.empty((len(a),21,2599),np.float32);norm=np.empty_like(a)
   for i,x in enumerate(a):
    assert lib.df_cmvn(C.byref(self.state),ptr(x),ptr(norm[i]))==0
    assert lib.df_step(C.byref(self.state),ptr(x),ptr(logits[i]),ptr(traces[i]))==0
   self.trace={'cmvn':norm[None],**{f'stage{s}':traces[:,s,:d][None].copy() for s,d in enumerate(dims)},'cache':np.ctypeslib.as_array(self.state.cache).reshape(1,128,11,4).copy()}
   assert all(np.isfinite(v).all() for v in self.trace.values())
   return torch.from_numpy(logits[None]),torch.from_numpy(self.trace['cache'])
 class Adapter:
  def __init__(self):
   self.storage=(C.c_uint64*((lib.donor_pcm_state_bytes()+7)//8))();assert lib.donor_pcm_init(self.storage)==0;self.last=False;self.pending=[];self.error=None;self.batch=None
   def cb(user,p):
    try:
     q=p.contents
     self.pending.append(dict(wave_samples=int(q.wave_samples),feature_count=int(q.feature_count),offset=int(q.offset),call_index=int(q.call_index),available_samples=int(q.available_samples),call_samples=int(q.call_samples),is_final_short=int(q.is_final_short),fbank=np.ctypeslib.as_array(q.fbank,shape=(q.fbank_rows*80,)).copy().reshape(q.fbank_rows,80),rows=np.ctypeslib.as_array(q.rows,shape=(q.selected_rows*400,)).copy().reshape(q.selected_rows,400),centers=np.ctypeslib.as_array(q.centers,shape=(q.selected_rows,)).copy()))
    except Exception as exc:self.error=exc
   self.callback=CALLBACK(cb)
  def accept(self,pcm):
   a=np.frombuffer(pcm,dtype='<i2').copy();assert lib.donor_pcm_feed(self.storage,a.ctypes.data_as(C.POINTER(C.c_int16)),len(a),self.callback,None)==0
   if self.last:assert lib.donor_pcm_finish(self.storage,self.callback,None)==0
   if self.error:raise self.error
   assert len(self.pending)==1;self.batch=self.pending.pop();assert np.isfinite(self.batch['rows']).all() and np.isfinite(self.batch['fbank']).all()
   return torch.from_numpy(self.batch['rows'])
 def metric(a,r,atol,rtol):
  assert a.shape==r.shape and np.isfinite(a).all() and np.isfinite(r).all()
  d=np.abs(a.astype(float)-r.astype(float));bound=atol+rtol*np.abs(r.astype(float))
  return dict(max_abs=float(d.max()) if d.size else 0.,fail_elements=int((d>bound).sum()),elements=d.size)
 with torch.inference_mode():
  for condition,(root,rows) in conditions.items():
   for ri,row in enumerate(rows):
    ref=refs[(condition,row['recording'])];assert row['file_sha256']==ref['source_sha256'] and row['pcm_sha256']==ref['pcm_sha256'];assert sha(REF/ref['arrays_file'])==ref['arrays_sha256'];g=np.load(REF/ref['arrays_file'],allow_pickle=False)
    assert sha(root/row['path'])==row['file_sha256']
    with wave.open(str(root/row['path'])) as f:pcm=f.readframes(f.getnframes())
    assert len(pcm)==row['frames']*2 and hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
    model=NativeModel();adapter=Adapter();k=b.make_spotter('hamming',model,frontend,ns,tokens);k.accept_wave=adapter.accept;events=[];calls=[];arrays={}
    for ci,start in enumerate(range(0,len(pcm),9600)):
     end=min(len(pcm),start+9600);adapter.last=end==len(pcm);model.trace={};result=k.forward(pcm[start:end]);batch=adapter.batch;rc=ref['calls'][ci];key=f'call{ci}'
     assert batch['call_index']==ci and batch['call_samples']==rc['input_samples'] and batch['available_samples']==rc['available_audio_samples']
     assert batch['wave_samples']==rc['wave_remained'] and batch['feature_count']==rc['feature_remained'] and batch['offset']==rc['offset']
     assert batch['centers'].tolist()==rc['centers'] and len(batch['fbank'])==rc['fbank_rows'] and k.total_frames==rc['upstream_total_frames']
     assert (key+'_logits' in g)==bool(model.trace)
     arrays[key+'_fbank']=batch['fbank'];arrays[key+'_splice']=batch['rows'][None];metrics={}
     if batch['fbank'].size:metrics['fbank']=metric(batch['fbank'],g[key+'_fbank'],1e-3,0)
     if model.trace:
      for n,v in model.trace.items():arrays[key+'_'+n]=v
      arrays[key+'_logits']=model.trace['stage20'];probs=torch.from_numpy(model.trace['stage20']).softmax(2).numpy();arrays[key+'_probabilities']=probs
      assert np.isfinite(probs).all() and np.all(probs>=0)
      metrics['splice']=metric(arrays[key+'_splice'],g[key+'_splice'],1e-3,0);metrics['cmvn']=metric(model.trace['cmvn'],g[key+'_cmvn'],2e-4,0)
      for s in range(21):metrics[f'stage{s}']=metric(model.trace[f'stage{s}'],g[key+f'_stage{s}'],1e-4,1e-5)
      metrics['cache']=metric(model.trace['cache'],g[key+'_cache'],1e-4,1e-5);metrics['logits']=metric(arrays[key+'_logits'],g[key+'_logits'],1e-4,1e-5);metrics['probabilities']=metric(probs,g[key+'_probabilities'],1e-5,1e-5)
      metrics['probability_row_sum_max_error']=float(np.abs(probs.astype(float).sum(axis=2)-1).max())
     if result.get('state')==1:events.append(dict(result,keyword_id=b.PHRASES[result['keyword']],available_audio_samples=end//2,eof_flush=False))
     calls.append(dict(index=ci,has_acoustic_output=bool(model.trace),wave_samples=batch['wave_samples'],feature_count=batch['feature_count'],offset=batch['offset'],centers=batch['centers'].tolist(),available_audio_samples=end//2,result=dict(result),metrics=metrics))
    assert len(calls)==len(ref['calls'])
    events_exact=events==ref['events'];without_score=lambda es:[{a:b for a,b in x.items() if a!='score'} for x in es]
    events_non_score_exact=without_score(events)==without_score(ref['events']);score_diffs=[abs(a['score']-b['score']) for a,b in zip(events,ref['events'])] if events_non_score_exact else []
    assert all(np.isfinite(a).all() for a in arrays.values());assert totalbytes+sum(a.nbytes for a in arrays.values())+4096*(len(arrays)+1)<1500000000
    filename=f'{condition}-{ri:02d}.npz';np.savez(OUT/filename,**arrays);totalbytes+=(OUT/filename).stat().st_size;assert totalbytes<1500000000
    reports.append(dict(condition=condition,recording=row['recording'],kind=row['kind'],keyword_id=row['keyword_id'],events=events,events_exact=events_exact,events_non_score_exact=events_non_score_exact,score_max_difference=max(score_diffs,default=0) if events_non_score_exact else None,score_within_1e_minus_5=events_non_score_exact and all(d<=1e-5 for d in score_diffs),calls=calls,arrays_file=filename,arrays_sha256=sha(OUT/filename)))
    save(OUT/f'{condition}-{ri:02d}.json',reports[-1])
 b.inputs();b.audited_sources();assert identity(hashes)==pre
 assert len(reports)==126 and {(r['condition'],r['recording']) for r in reports}==set(refs)
 final=dict(preflight_sha256=sha(E/'native-preflight.json'),clips=len(reports),array_bytes=totalbytes,recordings=reports,old_B_still_failed=True,full_fields_events_exact=all(r['events_exact'] for r in reports),non_score_event_fields_exact=all(r['events_non_score_exact'] for r in reports),event_scores_within_1e_minus_5=all(r['score_within_1e_minus_5'] for r in reports),logit_gate_passed=all(c['metrics']['logits']['fail_elements']==0 for r in reports for c in r['calls'] if c['has_acoustic_output']),probability_gate_passed=all(c['metrics']['probabilities']['fail_elements']==0 and c['metrics']['probability_row_sum_max_error']<=1e-5 for r in reports for c in r['calls'] if c['has_acoustic_output']),scope='fixed PCM diagnostic only; event agreement cannot overwrite old B failure')
 save(OUT/'result.json',final);print(json.dumps({a:b for a,b in final.items() if a!='recordings'},indent=2))
if __name__=='__main__':
 frontend,ns,hashes,conditions,tokens=setup()
 if sys.argv[1]=='prepare':save(E/'native-preflight.json',identity(hashes));print('FROZEN',sha(E/'native-preflight.json'))
 elif sys.argv[1]=='run':
  try:run(frontend,ns,hashes,conditions,tokens)
  except Exception as exc:
   if not (E/'native-failure.json').exists():save(E/'native-failure.json',dict(exception=repr(exc),traceback=traceback.format_exc(),old_B_still_failed=True))
   raise
 else:raise SystemExit('prepare or run')
