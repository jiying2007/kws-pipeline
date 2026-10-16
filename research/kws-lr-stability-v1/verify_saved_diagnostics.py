"""Offline saved-diagnostic integrity and report semantics; never replay a decoder."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
DIRECTORY='saved-diagnostics-20261016'
PREFIX='lr-stability-trigger-audit-20261016'
DECODER='lr-stability-decoder-replay-20261016'
COMMIT='e80527ad4b1d4cf299ab986441f3c3e60be32747'
def need(value,label):
 if not value:raise ValueError(label)
def exact(a,b):
 if type(a) is not type(b):return False
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
def decode(raw):
 def pairs(rows):
  out={}
  for k,v in rows:need(k not in out,'duplicate key');out[k]=v
  return out
 def bad(x):raise ValueError('nonfinite JSON')
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(root,name):
 p=Path(name);need(not p.is_absolute() and '..' not in p.parts,'unsafe path')
 for i in range(1,len(p.parts)+1):need(not (root/Path(*p.parts[:i])).is_symlink(),'symlink')
 return (root/p).read_bytes()
def load(root,name):return decode(read(root,name))
def fields(obj,expected,label):
 for k,v in expected.items():need(exact(obj.get(k),v),label+' '+k)
def suppression(hit,end,deadline):
 need(type(hit) is bool and type(end) is int and type(deadline) is int and end>=0 and deadline>=0,'synthetic state types')
 if not hit:return 0,deadline
 return (1,end+19200) if end>=deadline else (2,deadline)
def prefix_semantics(report,plan):
 fields(report,{'schema':'saved-prefix-audit-result-v1','status':'POSTHOC_DESCRIPTIVE_ONLY','new_model_forwards':0,'new_optimizer_updates':0,'runtime_replay':False},'prefix report')
 fields(plan,{'runtime_source_commit':COMMIT,'observation_update':600,'sample_count':12,'class_counts':{'KW1':4,'KW2':8},'beam_width':32,'top_k':1,'runtime_replay':False,'model_forwards':0,'optimizer_updates':0,'tts_calls':0,'asr_calls':0,'checkpoint_selection':False,'decoder_selection':False,'parameter_selection':False,'new_negative_evaluation':False,'threshold_sweep':None},'prefix plan')
 need(exact(report['input_hashes'],plan['inputs']),'prefix input identity')
 samples=report['samples'];need(len(samples)==12 and len({s['record_id'] for s in samples})==12,'prefix roster');need(sum(s['input_frames'] for s in samples)==1276,'prefix frame total')
 for c,n in [('KW1',4),('KW2',8)]:
  selected=[s for s in samples if s['class']==c];need(len(selected)==n,'prefix class counts')
  for d in ['greedy','beam32']:
   summary=report['summary'][d][c];need(summary['samples']==n,'prefix summary size');need(summary['final_exact_count']==sum(s['decoders'][d]['final_exact_match'] is True for s in selected),'prefix final counts')
   need(summary['ever_full_exact_count']==sum(s['decoders'][d]['matches'][c]['episode_count']>0 for s in selected),'prefix episode counts')
 for s in samples:
  need(type(s['input_frames']) is int and s['input_frames']>0,'prefix input frames');need(len(s['trace'])==s['input_frames'],'prefix trace length')
  for i,row in enumerate(s['trace']):need(row['frame']==i and row['nominal_available_ms']==(400+320*i)/16,'prefix nominal coordinates')
  for d in ['greedy','beam32']:
   target=plan['targets'][s['class']];entry=s['decoders'][d];need(entry['final_exact_match'] is (entry['final_tokens']==target),'prefix final tokens')
 need(len(report['limitations'])>=6,'prefix scope limitations')
def decoder_semantics(report,plan,claim,complete):
 fields(report,{'schema':'controlled-native-decoder-replay-result-v1','status':'COMPLETED_SYNTHETIC_GATE_UNCALIBRATED_TRAINING_ONLY','source_commit':COMMIT,'sample_count':12,'frame_count':1276,'saved_data_native_passes':1,'speech_gate':'synthetic-constant-one','threshold':0.55,'threshold_calibrated_for_research_model':False,'production_engine_executed':False,'suppression_mirror_refractory_ms':1200,'new_model_forwards':0,'new_optimizer_updates':0,'new_TTS_ASR_calls':0,'audio_or_weight_reads':0,'LP_transport_bitwise_verified':True,'suppression_recurrence_independently_verified':True},'decoder report')
 fields(plan,{'source_commit':COMMIT,'sample_count':12,'saved_frames':1276,'native_saved_data_passes':1,'new_model_forwards':0,'new_optimizer_updates':0,'new_tts_calls':0,'new_asr_calls':0,'audio_reads':0,'weight_reads':0,'official_model_bound_replay':False,'remote_publication':False,'no_parameter_search':True,'no_padding':True,'no_concatenation':True},'decoder plan')
 fields(claim,{'native_saved_data_passes_reserved':1,'frames_reserved':1276,'no_retry':True,'new_model_forwards_reserved':0},'decoder claim')
 fields(complete,{'native_returncode':0,'saved_data_passes_completed':1,'frames_completed':1276,'original_inputs_unchanged':True,'next_action':'STOP_NO_ADDITIONAL_EXPERIMENTS'},'decoder complete')
 need(plan['speech_active']=={'value':1,'origin':'synthetic-control-constant-not-observed-VAD'},'synthetic speech gate');need(plan['suppression_mirror']['refractory_samples']==19200 and plan['suppression_mirror']['suppressed_extends_until'] is False,'suppression definition')
 samples=report['samples'];need(len(samples)==12 and len({s['record_id'] for s in samples})==12,'decoder roster');need(sum(s['input_frames'] for s in samples)==1276,'decoder frame count')
 totals={k:0 for k in ['native_decoder_hits','mirror_accepted','mirror_suppressed']}
 for s in samples:
  deadline=0;counts=dict.fromkeys(totals,0)
  for event in s['events']:
   f=event['frame'];need(type(f) is int and 0<=f<s['input_frames'],'event bounds');end=400+320*f;need(event['end_sample']==end and event['nominal_clip_coordinate_ms']==end/16,'decoder nominal coordinates');need(event['keyword_id'] in [1,2] and .55-1e-7<=event['confidence']<=1,'event identity');need(event['intended_keyword'] is (event['keyword_id']==(1 if s['target_class']=='KW1' else 2)),'event intended identity')
   outcome,deadline=suppression(True,end,deadline);label='accepted' if outcome==1 else 'suppressed';need(event['mirror_outcome']==label,'suppression semantics');counts['native_decoder_hits']+=1;counts['mirror_'+label]+=1
  need(exact(counts,s['counts']),'event counts')
  for k in counts:totals[k]+=counts[k]
 need(exact(totals,report['totals']),'decoder totals')
 for c,n in [('KW1',4),('KW2',8)]:
  selected=[s for s in samples if s['target_class']==c];need(len(selected)==n and report['per_class'][c]['sample_count']==n,'decoder class counts')
  for k in totals:need(report['per_class'][c][k]==sum(s['counts'][k] for s in selected),'decoder class totals')
 need(len(report['limitations'])>=7,'decoder scope limitations')
def verify(root=HERE):
 module_root=Path(root);root=module_root/DIRECTORY;m=load(root,'archive-manifest.json')
 need(m['source_commit']==COMMIT and m['execution_authority']=='none; completed evidence only','archive scope')
 for name,pin in m['files'].items():
  raw=read(root,name);need(len(raw)==pin['bytes'] and sha(raw)==pin['sha256'],'archive hash '+name)
 p=root/PREFIX;d=root/DECODER;pr=load(p,'report.json');pp=load(p,'analysis-plan.json');dr=load(d,'report.json');dp=load(d,'analysis-plan.json');freeze=load(d,'preexecution-freeze.json');claim=load(d,'execution-claim.json');complete=load(d,'execution-complete.json')
 prefix_semantics(pr,pp);decoder_semantics(dr,dp,claim,complete)
 for key,name in [('analysis_plan_sha256','analysis-plan.json'),('definitions_sha256','DEFINITIONS.md'),('analyzer_sha256','analyze.py'),('source_manifest_sha256','source-manifest.json')]:need(pr[key]==sha(read(p,name)),'prefix binding '+key)
 need(pp['source_manifest_sha256']==pr['source_manifest_sha256'],'prefix source manifest binding')
 add=load(p,'source-addendum.json');need(add['analysis_plan_sha256']==pr['analysis_plan_sha256'] and add['experiment_plan_sha256']==pp['inputs']['plan']['sha256'],'writer plan binding');need(add['original_observation_writer_input']==pp['inputs']['observation_writer'],'writer helper identity');need(add['actual_observation_writer']==dp['additional_original_inputs']['actual_lr_observation_writer'],'actual LR writer identity');need(add['verifier_sha256']==sha(read(p,'verify_source_addendum.py')),'addendum verifier binding')
 fields(add,{'analysis_definitions_changed':False,'numeric_report_changed':False,'model_calls':0,'source_code_executed':False},'writer correction scope')
 need(dr['plan_sha256']==freeze['plan_sha256']==sha(read(d,'analysis-plan.json')),'decoder plan binding');need(dr['preexecution_freeze_sha256']==claim['preexecution_freeze_sha256']==sha(read(d,'preexecution-freeze.json')),'decoder freeze binding');need(complete['report_sha256']==sha(read(d,'report.json')),'completion report binding');need(dr['native_output_sha256']==sha(read(d,'native-output.jsonl')),'native numerical body binding');need(freeze['source_manifest_sha256']==sha(read(d,'source-manifest.json')),'decoder source binding')
 need(dp['original_inputs']==pp['inputs'],'shared original inputs');need(dp['prior_feasibility_sha256']==sha(read(p,'NEXT-REPLAY-FEASIBILITY.md')),'prior feasibility identity')
 references={(r['archive_directory'],r['original_snapshot_path']):r for r in m['source_references']};need(len(references)==len(m['source_references'])==24,'reference roster')
 for directory,manifest_name,snapshot_prefix in [(p,'source-manifest.json','source-contract/'),(p,'feasibility-source-manifest.json','feasibility-source/'),(d,'source-manifest.json','source/')]:
  source=load(directory,manifest_name);need(source['commit']==COMMIT and source['repository']=='jiying2007/kws-pipeline','source commit')
  for path,pin in source['files'].items():
   need(pin['url']=='https://github.com/jiying2007/kws-pipeline/blob/'+COMMIT+'/'+path,'immutable source URL');ref=references.get((directory.name,snapshot_prefix+path));need(ref is not None,'missing source reference')
   need(ref['repository_path']==path and ref['source_commit']==COMMIT and ref['source_url']==pin['url'] and ref['sha256']==pin['sha256'] and ref['git_blob_sha1']==pin['git_blob_sha1'],'source reference binding')
 for path,digest in freeze['file_sha256'].items():
  if (d/path).is_file():need(sha(read(d,path))==digest,'frozen source '+path)
  else:need(path in m['omitted_frozen_files'] and m['omitted_frozen_files'][path]['sha256']==digest and bool(m['omitted_frozen_files'][path]['reason']),'unexpected frozen omission')
 need(set(m['omitted_frozen_files'])=={name for name in freeze['file_sha256'] if name.startswith('source/')}|{'build-1/build-and-synthetic-tests.json','build-1/replay_harness','build-1/test_decoder','build-1/saved-lp-input.txt'},'omission roster')
 previous=module_root/'result-20261016';original=load(previous,'result-hash-manifest.json');need(pp['inputs']['accepted_manifest']['sha256']==sha(read(previous,'result-hash-manifest.json')),'original accepted manifest');need(pp['inputs']['observation']['sha256']==original['result_files']['A-0600.json'],'original observation hash');need(pp['inputs']['plan']['sha256']==sha(read(previous,'executed-plan.json')),'original enabled plan')
 return {'integrity':'PASS','scope':'hashes and saved report semantics; no saved analysis or decoder replay','saved_data_native_passes_started':0,'model_forwards':0,'optimizer_updates':0}
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
