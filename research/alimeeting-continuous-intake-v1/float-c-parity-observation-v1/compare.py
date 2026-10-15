"""One fixed CPU float inference per original clip; never loads unsafe pickle."""
import os,sys,pathlib,json,hashlib,struct,wave,math,platform,time,resource
D=pathlib.Path(__file__).resolve().parent;B=D.parent;H=B/'historical-parity-source-review';C=B/'positive-trace-observation-v1';O=B/'machine-positive-control-v1/run-p4-11b93abe';R=D/'run';SRC=B.parent/'kws-asr-registration-p4-20261014/candidate'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,v):pathlib.Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check(v,m):
 if not v:raise RuntimeError(m)
p=SRC/'models/registry/model-749187ec1d66';prov=json.loads((p/'xiaowo-model-provenance.json').read_text());cp=p/'xiaowo-model.pt';check(sha(cp)==prov['checkpoint']['sha256'],'checkpoint hash')
checks={str(cp):sha(cp),str(p/'xiaowo-model.kwm'):sha(p/'xiaowo-model.kwm'),str(p/'xiaowo-keywords.kwk'):sha(p/'xiaowo-keywords.kwk'),str(D/'requirements.lock'):sha(D/'requirements.lock'),str(D/'wheels.json'):sha(D/'wheels.json'),str(__file__):sha(__file__)}
for f in ['frontend.py','frontend_spec.py','model.py']:
 check(sha(H/f)==prov['training']['environment']['training_code_sha256']['training/'+f],'source hash');checks[str(H/f)]=sha(H/f)
inputs=json.loads((O/'inputs.json').read_text());check(len(inputs['selected'])==13,'selection');checks[str(O/'inputs.json')]=sha(O/'inputs.json')
for row in inputs['selected']:
 f=O/'audio'/(row['sample_id']+'.wav');check(sha(f)==row['wav_sha256'],'wav hash');checks[str(f)]=sha(f);f=C/'run'/(row['sample_id']+'.kwtr');checks[str(f)]=sha(f)
R.mkdir(exist_ok=False);rec={'status':'running','inputs':checks,'checkpoint_loading':'torch.load(map_location=cpu,weights_only=True), no added safe globals','no_new_c_dump':True,'no_tts_asr_training_or_tuning':True,'calls':[]};save(R/'receipt.json',rec)
try:
 sys.path.insert(0,str(H));from model import TinyStreamingRNN
 import torch
 from frontend import features
 check(torch.__version__=='2.13.0+cpu' and torch.version.cuda is None,'CPU version');torch.use_deterministic_algorithms(True)
 rec['environment']={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':__import__('numpy').__version__,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'mkldnn':torch.backends.mkldnn.enabled,'torch_config':torch.__config__.show(),'python_sha256':sha(sys.executable),'cpu_env':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','MKL_CBWR','ATEN_CPU_CAPABILITY','OPENBLAS_NUM_THREADS']}}
 ck=torch.load(cp,map_location='cpu',weights_only=True)
 check(isinstance(ck,dict),'checkpoint object')
 check(tuple(ck[k] for k in ['feature_dim','hidden_dim','vocab_size','frame_length_samples','frame_hop_samples'])==(32,64,5,400,320),'geometry')
 check(ck['frontend_name']=='logmel' and ck['frontend_kind']==0 and ck['frontend_spec_version']==2,'frontend')
 state=ck['state_dict'];expected={'in_proj.weight':(64,32),'in_proj.bias':(64,),'rec_proj.weight':(64,64),'out_proj.weight':(5,64),'out_proj.bias':(5,)};check(set(state)==set(expected),'unknown state keys')
 for k,shape in expected.items():check(type(state[k]) is torch.Tensor and tuple(state[k].shape)==shape and state[k].dtype==torch.float32 and bool(torch.isfinite(state[k]).all()),'tensor contract')
 model=TinyStreamingRNN(32,64,5);model.load_state_dict(state,strict=True);model.eval();summary=[]
 for row in inputs['selected']:
  sid=row['sample_id'];save(R/(sid+'.claim.json'),{'attempt':1,'sample_id':sid,'operation':'frozen float forward'})
  with wave.open(str(O/'audio'/(sid+'.wav')),'rb') as w:
   check((w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,16000),'wav');pcm=w.readframes(w.getnframes())
  wav=torch.frombuffer(bytearray(pcm),dtype=torch.int16).to(torch.float32)/32768.0
  start=time.monotonic()
  with torch.inference_mode():ft=features(wav,32,400,320,'logmel');out=model(ft.unsqueeze(0))[:,0,:].contiguous()
  c=(C/'run'/(sid+'.kwtr')).read_bytes();n=struct.unpack_from('<Q',c,40)[0];check(tuple(out.shape)==(n,5),'aligned frames');raw=out.numpy().tobytes();(R/(sid+'.float-logits.f32')).write_bytes(raw)
  vals=out.tolist();diff=[];tops={'float':[0]*5,'c':[0]*5};agree=0;float_run=[];c_run=[];maxp3=0;mingap3=1e30
  for i,v in enumerate(vals):
   cv=struct.unpack_from('<5f',c,120+36*i+16);diff.extend(a-b for a,b in zip(v,cv));t=max(range(5),key=lambda k:v[k]);ct=max(range(5),key=lambda k:cv[k]);tops['float'][t]+=1;tops['c'][ct]+=1;agree+=t==ct
   if not float_run or float_run[-1]!=t:float_run.append(t)
   if not c_run or c_run[-1]!=ct:c_run.append(ct)
   mx=max(v);es=[math.exp(x-mx) for x in v];maxp3=max(maxp3,es[3]/sum(es));mingap3=min(mingap3,mx-v[3])
  item={'sample_id':sid,'target_text':row['target_text'],'frames':n,'top1_counts':tops,'top1_agreement_frames':agree,'float_greedy':[t for t in float_run if t],'c_greedy':[t for t in c_run if t],'logit_max_abs_difference':max(abs(x) for x in diff),'logit_rmse':math.sqrt(sum(x*x for x in diff)/len(diff)),'float_xiao3_max_exact_softmax':maxp3,'float_xiao3_min_top_gap':mingap3,'float_logits_sha256':hashlib.sha256(raw).hexdigest(),'features_sha256':hashlib.sha256(ft.numpy().tobytes()).hexdigest()};summary.append(item);rec['calls'].append({'sample_id':sid,'attempt':1,'elapsed_s':time.monotonic()-start,'status':'success'});save(R/'receipt.json',rec)
 for f,h in checks.items():check(sha(f)==h,'input mutated')
 save(R/'analysis.json',summary);rec['status']='pass';rec['inputs_unchanged']=True;rec['float_forward_calls']=13
except BaseException as e:
 rec['status']='failed';rec['error_type']=type(e).__name__;raise
finally:
 rec['process_ru_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;rec['outputs']={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in R.iterdir() if f.is_file() and f.name!='receipt.json'};save(R/'receipt.json',rec)
print('FLOAT COMPARISON PASS')
