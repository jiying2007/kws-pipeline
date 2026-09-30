"""Single fixed x86 profile of failed numerical candidate; not all-C/SSC305."""
import ast,ctypes as C,hashlib,importlib.util,json,pathlib,resource,sys,time,types,wave,os
START=(time.process_time_ns(),time.perf_counter_ns())
def io():
 return {k:int(v) for k,v in (line.split(':') for line in pathlib.Path('/proc/self/io').read_text().splitlines())}
START_IO=io()
import numpy as np
import torch
IMPORT_END=(time.process_time_ns(),time.perf_counter_ns())
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=pathlib.Path('/workspace/shared/kws-cfsmn-baseline');DIAG=pathlib.Path('/workspace/shared/kws-fsmn-pcm-diagnostic-v1');OUT=pathlib.Path('/workspace/shared/kws-full-fsmn-resource-profile')
F=C.POINTER(C.c_float)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=module('baseline',BASE/'run_baseline.py');manifest=module('manifest',ROOT/'manifest.py');diag=module('diagnose_pcm',ROOT/'tests/diagnose_pcm.py')
def save(p,x):
 if p.exists():raise ValueError('refuse overwrite '+str(p))
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def ptr(a):return a.ctypes.data_as(F)
def clock():return time.process_time_ns(),time.perf_counter_ns()
def elapsed(a,z):return dict(cpu_ns=z[0]-a[0],wall_ns=z[1]-a[1])
def rss():
 for line in pathlib.Path('/proc/self/status').read_text().splitlines():
  if line.startswith('VmRSS:'):return int(line.split()[1])*1024

def thread_count():
 for line in pathlib.Path('/proc/self/status').read_text().splitlines():
  if line.startswith('Threads:'):return int(line.split()[1])
def prepare():
 thread_env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}
 assert all(v=='1' for v in thread_env.values())
 return dict(thread_environment=thread_env,script_sha256=sha(pathlib.Path(__file__)),plan_sha256=sha(pathlib.Path('/workspace/shared/kws-fsmn-native-spec/RESOURCE_PROFILE_PROPOSAL.md')),library_sha256=sha(DIAG/'native/native.so'),native_result_sha256=sha(DIAG/'native/result.json'),native_preflight_sha256=sha(DIAG/'native-preflight.json'),baseline_driver_sha256=sha(BASE/'run_baseline.py'),manifest_loader_sha256=sha(ROOT/'manifest.py'),pcm_driver_sha256=sha(ROOT/'tests/diagnose_pcm.py'),kernel_sha256=sha(ROOT/'fsmn.c'),payload_sha256=manifest.PAYLOAD_SHA256,threads=1,warmup='first3rawclips in original receipt order',passes=3,clips_per_pass=42,feed_samples=4800,CMVN='inside model timed calls; standalone microcall outside pipeline, no double count/subtraction',trace_pointer='NULL, no diagnostic copies; model arithmetic unchanged',python=sys.version,numpy=np.__version__,torch=torch.__version__,CPU_limit_seconds=600,scope='fixed numerically failed candidate; x86 Python+Torch+ctypes research process; no SSC305 extrapolation')

def run():
 assert json.loads((OUT/'preflight.json').read_text())==prepare()
 if (OUT/'result.json').exists():raise ValueError('existing result')
 resource.setrlimit(resource.RLIMIT_CPU,(600,600));torch.set_num_threads(1);torch.set_num_interop_threads(1)
 a=clock();frontend,ns,_=b.audited_sources();conditions=b.inputs();tokens=(BASE/'resources/tokens_2599.txt').read_text().splitlines();startup_audit=elapsed(a,clock())
 a=clock();lib=C.CDLL(str(DIAG/'native/native.so'));lib.donor_pcm_state_bytes.restype=C.c_size_t;lib.donor_pcm_init.argtypes=[C.c_void_p];lib.donor_pcm_feed.argtypes=[C.c_void_p,C.POINTER(C.c_int16),C.c_size_t,diag.CALLBACK,C.c_void_p];lib.donor_pcm_finish.argtypes=[C.c_void_p,diag.CALLBACK,C.c_void_p];lib.df_init.argtypes=[C.POINTER(diag.Model),F,C.c_size_t];lib.df_step.argtypes=[C.POINTER(diag.Model),F,F,F];lib.df_cmvn.argtypes=[C.POINTER(diag.Model),F,F]
 raw=manifest.load(pathlib.Path('/workspace/shared/kws-fsmn-ab-evidence/weights-manifest.json'),pathlib.Path('/workspace/shared/kws-fsmn-ab-evidence/donor-local.f32'));weights=np.frombuffer(raw,dtype='<f4').copy();initial_model=diag.Model();assert lib.df_init(C.byref(initial_model),ptr(weights),len(weights))==0;startup_model=elapsed(a,clock())
 a=clock();root,rows=conditions['raw'];clips=[]
 for row in rows:
  with wave.open(str(root/row['path'])) as f:pcm=f.readframes(f.getnframes())
  assert len(pcm)==row['frames']*2 and hashlib.sha256(pcm).hexdigest()==row['pcm_sha256'];clips.append((row,pcm))
 startup_pcm=elapsed(a,clock())
 a=clock();nr=json.loads((DIAG/'native/result.json').read_text());expected={}
 for r in nr['recordings']:
  if r['condition']!='raw':continue
  path=DIAG/'native'/r['arrays_file'];assert sha(path)==r['arrays_sha256'];g=np.load(path)
  full=np.concatenate([g[f"call{c['index']}_logits"][0] for c in r['calls'] if c['has_acoustic_output']])
  expected[r['recording']]=dict(logits_sha256=hashlib.sha256(full.tobytes()).hexdigest(),events=r['events']);g.close()
 assert len(expected)==len(clips)==42;startup_reference=elapsed(a,clock())
 # Instrument only the exact softmax call in original forward; decoder methods
 # are wrapped for timing, preserving original body/order/reset behavior.
 tree=ast.parse((BASE/'upstream/wekws/bin/stream_kws_ctc.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='KeyWordSpotter');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
 class TimerCall(ast.NodeTransformer):
  count=0
  def visit_Call(self,node):
   node=self.generic_visit(node)
   if isinstance(node.func,ast.Attribute) and node.func.attr=='softmax' and isinstance(node.func.value,ast.Name) and node.func.value.id=='logits':
    self.count+=1;return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_timed_softmax',ctx=ast.Load()),args=[ast.Name(id='logits',ctx=ast.Load()),*node.args],keywords=node.keywords),node)
   return node
 transform=TimerCall();fn=transform.visit(fn);assert transform.count==1;code=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope=dict(ns);exec(compile(code,'timing-only-original-forward','exec'),scope)
 startup=dict(sampled_threads=thread_count(),imports=elapsed(START,IMPORT_END),source_input_audit=startup_audit,model_validation_load=startup_model,PCM_preload=startup_pcm,diagnostic_reference_audit=startup_reference,whole=elapsed(START,clock()),io_before=START_IO,io_after=io(),rss_bytes=rss(),scope='after stdlib import through Torch/imports/audits/model/PCM/reference preloading; warm host cache')
 class Native(torch.nn.Module):
  def __init__(self,records):
   super().__init__();self.state=diag.Model();assert lib.df_init(C.byref(self.state),ptr(weights),len(weights))==0;self.records=records;self.logits=[];self.features=[]
  def forward(self,features,cache):
   x=features[0].numpy();y=np.empty((len(x),2599),np.float32)
   for i,row in enumerate(x):
    t=clock();rc=lib.df_step(C.byref(self.state),ptr(row),ptr(y[i]),None);z=clock();self.records['model_including_cmvn'].append(elapsed(t,z));assert rc==0
   self.logits.append(y);self.features.append(x.copy())
   return torch.from_numpy(y[None]),torch.from_numpy(np.ctypeslib.as_array(self.state.cache).reshape(1,128,11,4))
 class Adapter:
  def __init__(self,records):
   self.storage=(C.c_uint64*((lib.donor_pcm_state_bytes()+7)//8))();assert lib.donor_pcm_init(self.storage)==0;self.records=records;self.pending=[];self.error=None;self.last=False
   def cb(user,p):
    t=clock()
    try:q=p.contents;self.pending.append(np.ctypeslib.as_array(q.rows,shape=(q.selected_rows*400,)).copy().reshape(q.selected_rows,400))
    except Exception as exc:self.error=exc
    self.records['callback_copy'].append(elapsed(t,clock()))
   self.callback=diag.CALLBACK(cb)
  def accept(self,pcm):
   x=np.frombuffer(pcm,dtype='<i2');t=clock();rc=lib.donor_pcm_feed(self.storage,x.ctypes.data_as(C.POINTER(C.c_int16)),len(x),self.callback,None);z=clock();self.records['frontend_splice_inclusive'].append(elapsed(t,z));assert rc==0
   if self.last:
    t=clock();rc=lib.donor_pcm_finish(self.storage,self.callback,None);z=clock();self.records['frontend_splice_inclusive'].append(elapsed(t,z));assert rc==0
   if self.error:raise self.error
   assert len(self.pending)==1;return torch.from_numpy(self.pending.pop())
 def one(row,pcm,do_micro):
  setup_start=clock()
  rec={k:[] for k in ['model_including_cmvn','frontend_splice_inclusive','callback_copy','torch_softmax','python_decoder','standalone_cmvn_micro']};model=Native(rec);adapter=Adapter(rec);k=b.make_spotter('hamming',model,frontend,ns,tokens);k.accept_wave=adapter.accept;k.forward=types.MethodType(scope['forward'],k)
  def sm(logits,dim):
   t=clock();y=logits.softmax(dim);rec['torch_softmax'].append(elapsed(t,clock()));return y
  k._timed_softmax=sm
  for method in ['decode_keywords','execute_detection']:
   original=getattr(k,method)
   def wrapped(*args,_original=original):
    t=clock();y=_original(*args);rec['python_decoder'].append(elapsed(t,clock()));return y
   setattr(k,method,wrapped)
  clip_initialization=elapsed(setup_start,clock());events=[];t=clock()
  with torch.inference_mode():
   for start in range(0,len(pcm),9600):
    end=min(len(pcm),start+9600);adapter.last=end==len(pcm);result=k.forward(pcm[start:end])
    if result.get('state')==1:events.append(dict(result,keyword_id=b.PHRASES[result['keyword']],available_audio_samples=end//2,eof_flush=False))
  whole=elapsed(t,clock())
  assert hashlib.sha256(np.concatenate(model.logits).tobytes()).hexdigest()==expected[row['recording']]['logits_sha256'];assert events==expected[row['recording']]['events']
  if do_micro:
   output=np.empty(400,np.float32)
   for features in model.features:
    for x in features:
     t=clock();rc=lib.df_cmvn(C.byref(model.state),ptr(x),ptr(output));rec['standalone_cmvn_micro'].append(elapsed(t,clock()));assert rc==0
  return dict(clip_initialization=clip_initialization,recording=row['recording'],audio_samples=len(pcm)//2,whole=whole,components=rec,logits_and_events_exact=True,retained_feature_bytes=sum(x.nbytes for x in model.features),retained_logit_bytes=sum(x.nbytes for x in model.logits))
 for row,pcm in clips[:3]:one(row,pcm,False)
 passes=[]
 for n in range(3):
  threads_before=thread_count();before=io();t=clock();recordings=[one(row,pcm,True) for row,pcm in clips];all_elapsed=elapsed(t,clock());after=io();threads_after=thread_count()
  passes.append(dict(threads_before=threads_before,threads_after=threads_after,index=n,recordings=recordings,outer_including_validation_micro=all_elapsed,io_before=before,io_after=after,rss_bytes=rss()))
 assert prepare()==json.loads((OUT/'preflight.json').read_text())
 cpu=sum(r['whole']['cpu_ns'] for p in passes for r in p['recordings']);audio=sum(r['audio_samples']/16000 for p in passes for r in p['recordings'])
 result=dict(preflight_sha256=sha(OUT/'preflight.json'),startup=startup,passes=passes,audio_seconds=audio,pipeline_cpu_seconds=cpu/1e9,pipeline_cpu_RTF=cpu/1e9/audio,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,pcm_C_state_bytes=lib.donor_pcm_state_bytes(),model_C_ctypes_mirror_bytes=C.sizeof(diag.Model),parameter_payload_bytes=weights.nbytes,declared_df_step_scratch_float_bytes=(400+250+250+128+128)*4,scope='Python+Torch+ctypes process on x86; numerical candidate failed; no SSC305 extrapolation; no all-C pipeline claim',measurement_notes=['frontend_splice_inclusive already includes callback_copy; never add both','pass I/O includes validation, CMVN microcalls and observers, not only timed pipeline','warmup is only first3 raw recordings in receipt order','whole includes feature/logit retention copies and Python timing-record allocations','Torch intra/inter-op and OMP/MKL/OPENBLAS fixed1; sampled Threads does not prove unsampled peak','ctypes and timer boundary overhead remains in component calls'],host_CPU=next(line.split(':',1)[1].strip() for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),old_B_still_failed=True,PCM_logit_gate_still_failed=True)
 save(OUT/'result.json',result);print(json.dumps({a:b for a,b in result.items() if a not in ['startup','passes']},indent=2))
if __name__=='__main__':
 if sys.argv[1]=='prepare':save(OUT/'preflight.json',prepare());print('FROZEN',sha(OUT/'preflight.json'))
 elif sys.argv[1]=='run':
  try:run()
  except Exception as exc:
   import traceback
   if not (OUT/'failure.json').exists():save(OUT/'failure.json',dict(exception=repr(exc),traceback=traceback.format_exc()))
   raise
 else:raise SystemExit('prepare or run')
