import json,pathlib,hashlib,subprocess,struct,math,time,collections
D=pathlib.Path(__file__).resolve().parent;B=D.parent;O=B/'machine-positive-control-v1/run-p4-11b93abe';R=D/'run'; sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,v):pathlib.Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check(v,msg):
 if not v:raise RuntimeError(msg)
r=json.loads((B/'diagnostic-build/build-receipt.json').read_text());br=json.loads((D/'build-receipt.json').read_text()); inputs=json.loads((O/'inputs.json').read_text());check(br['status']=='pass','build')
checks=dict(br['source_inputs']);checks.update(inputs['sha256_by_path'])
for f in ['libkws_pipeline.a','libkws_tool_io.a']:check(sha(D/'build'/f)==sha(B/'diagnostic-build/build'/f),'core differs')
for f in ['kws_posterior_dump','kws_decoder_replay','kws_decoder_path_replay']:checks[str(D/'build'/f)]=br['outputs']['build/'+f]['sha256']
for p in O.iterdir():
 if p.is_file(): checks[str(p)]=sha(p)
for p,h in checks.items():check(sha(p)==h,p)
R.mkdir(exist_ok=False)
rec={'schema':'positive-trace-observation-v1','source_revision':r['source_revision'],'source_tree':r['source_tree'],'original_control_summary_sha256':sha(O/'summary.json'),'build_receipt_sha256':sha(D/'build-receipt.json'),'script_sha256':sha(__file__),'inputs':checks,'calls':[],'scope':'exact 13 machine development positives; no padding, overrides, tuning or training','status':'running'}
save(R/'receipt.json',rec)
def call(kind,sid,argv):
 claim=R/(sid+'.'+kind+'.claim.json');check(not claim.exists(),'already attempted');save(claim,{'attempt':1,'argv':argv});t=time.monotonic()
 p=subprocess.run(argv,env=r['environment'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
 (R/(sid+'.'+kind+'.stdout')).write_bytes(p.stdout);(R/(sid+'.'+kind+'.stderr')).write_bytes(p.stderr)
 rec['calls'].append({'sample_id':sid,'kind':kind,'attempt':1,'argv':argv,'exit_code':p.returncode,'elapsed_s':time.monotonic()-t});save(R/'receipt.json',rec);check(p.returncode==0,'command failed');return p.stdout
try:
 rows=[]
 for row in inputs['selected']:
  sid=row['sample_id'];wav=O/'audio'/(sid+'.wav');check(sha(wav)==row['wav_sha256'],'wav');tr=R/(sid+'.kwtr');m=r['model']['path'];p=r['keyword_pack']['path']
  call('dump',sid,[str(D/'build/kws_posterior_dump'),m,str(wav),str(tr)])
  events=call('replay',sid,[str(D/'build/kws_decoder_replay'),m,p,str(tr),sid]);check(events==(O/(sid+'.events.jsonl')).read_bytes(),'events differ')
  paths=json.loads(call('path',sid,[str(D/'build/kws_decoder_path_replay'),m,p,str(tr),sid]))
  data=tr.read_bytes();check(data[:8]==b'KWTRACE1','magic');n=struct.unpack_from('<Q',data,40)[0];check(len(data)==120+n*36,'trace bytes');check(data[48:112].decode()==r['model']['sha256'],'model binding')
  frames=[];hist=collections.Counter()
  for i in range(n):
   off=120+i*36;end=struct.unpack_from('<Q',data,off)[0];speech=data[off+8];logits=list(struct.unpack_from('<5f',data,off+16));top=max(range(5),key=lambda x:logits[x]);hist[str(top)]+=1;mx=max(logits);ex=[math.exp(x-mx) for x in logits];probs=[x/sum(ex) for x in ex]
   frames.append({'frame':i+1,'end_sample':end,'speech_active':speech,'top1':top,'logits':logits,'exact_softmax':probs,'root_gap_ni3':mx-logits[1],'root_gap_xiao3':mx-logits[3]})
  stats=json.loads((O/(sid+'.stats.json')).read_text());check(n==stats['processed_frames'] and hist['0']==stats['blank_top1_frames'] and sum(f['speech_active'] for f in frames)==stats['speech_frames'],'frame stats differ');check(all(f['end_sample']==400+i*320 for i,f in enumerate(frames)),'frame geometry')
  save(R/(sid+'.frames.json'),frames)
  collapsed=[]
  for f in frames:
   if not collapsed or collapsed[-1]!=f['top1']:collapsed.append(f['top1'])
  rows.append({'sample_id':sid,'target_text':row['target_text'],'keyword_id':row['keyword_id'],'frames':n,'top1_histogram':dict(hist),'top1_run_tokens':collapsed,'ctc_greedy_tokens':[x for x in collapsed if x],'speech_frames':stats['speech_frames'],'paths':paths,'events_byte_identical':True,'trace_sha256':sha(tr)})
 for p,h in checks.items():check(sha(p)==h,p)
 save(R/'analysis.json',rows);rec['status']='pass';rec['completed_samples']=len(rows);rec['acoustic_dump_calls']=13;rec['decoder_only_calls']=26;rec['all_original_inputs_unchanged']=True
except BaseException as e:
 rec['status']='failed';rec['error_type']=type(e).__name__;raise
finally:
 rec['outputs']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in R.iterdir() if p.is_file() and p.name!='receipt.json'};save(R/'receipt.json',rec)
print('OBSERVATION PASS')
