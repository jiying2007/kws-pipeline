#!/usr/bin/env python3
"""Offline metadata only: no inference, audio output, fit, or trust-root admission."""
import argparse
import hashlib
import types
import io
import os
import sys
import json
from pathlib import Path, PurePosixPath
import stat
import struct
import wave
import zipfile

ARCHIVES = {
 'asr-11899008382.zip':'f8bf70a0539d6ca7bd11dfed9229a087c0c7d27b943550f73136cd765f54ef0c',
 'qwen06-11895552988.zip':'fbd39b79838dc91202fcf8e6c2af22f6486e5d11a9daf48aab15aabb05a6fba9',
 'qwen17-11895324049.zip':'2bc9ceb32ea64039c04982d88dc6247a8a43879a0c2b04ccb91ef3bb54a76307',
 'kw1-supplement-11942325284.zip':'e9cb187fbc0c0233f95e95efa7ca064359d7b00e3e6205041c7da06b3dedc5fe'}
OLD_IDS = ['tqp-qwen06-'+s for s in ('03','05','06','10','11','14','15')] + ['tqp-qwen17-'+s for s in ('01','02','03','06','14','15')]
NEW_IDS = ['kw1s-05','kw1s-06','kw1s-07']
EVIDENCE = 'research/tts-pilot-screening-v1/evidence/asr-run-38573117922/'
INPUTS = [EVIDENCE+'thirteen-candidate-evidence.json', 'research/tts-pilot-screening-v1/evidence/tts-input-20261014/artifact-inventory.json', 'research/consolidation/fit-history-recovery-2026-10-15.json', 'research/kw1-preset-supplement-observation-v1/observation.json', 'research/tts-pilot-screening-v1/screen.py', 'research/kw1-preset-supplement-v1/blind_pcm.py', 'research/kw1-preset-supplement-v1/worker.py']

def require(ok, why):
 if not ok: raise ValueError(why)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value, ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
def decode(raw):
 def unique(pairs):
  d={}
  for k,v in pairs:
   require(k not in d,'duplicate JSON key'); d[k]=v
  return d
 def invalid(_): raise ValueError('nonfinite JSON')
 return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid)

def archive(path, expected):
 path=Path(path).absolute()
 require(not any(p.is_symlink() for p in (path,*path.parents)) and path.resolve()==path,'unaliased ZIP path')
 before=path.stat();require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size <= 64*1024**2,'bounded regular ZIP')
 with path.open('rb') as f: raw=f.read(64*1024**2+1)
 after=path.stat();require(before==after and len(raw)==before.st_size,'ZIP changed during bounded read')
 require(sha(raw)==expected,'ZIP SHA256 mismatch')
 z=zipfile.ZipFile(io.BytesIO(raw)); infos=z.infolist()
 require(0<len(infos)<=512,'ZIP member count')
 names=set(); total=0
 for i in infos:
  p=PurePosixPath(i.filename); mode=i.external_attr>>16
  require(i.filename not in names and not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename and str(p)==i.filename,'unsafe or duplicate ZIP member')
  require(not i.is_dir() and stat.S_IFMT(mode) in (0,stat.S_IFREG) and not (i.flag_bits&1),'nonregular or encrypted ZIP member')
  require(0<=i.file_size<=8*1024**2,'ZIP member cap');total+=i.file_size;names.add(i.filename)
 require(total<=64*1024**2,'ZIP aggregate cap')
 return z

INPUT_PINS_SHA256='64c77a270c7f13d8252d10282191aaeb207f425c10fb89a2f01e390fa994e37c'

def read_bounded(path, maximum=8*1024**2):
 path=Path(path).absolute()
 require(not any(p.is_symlink() for p in (path,*path.parents)) and path.resolve()==path,'unaliased input path')
 before=path.stat();require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=maximum,'bounded regular input')
 with path.open('rb') as f:raw=f.read(maximum+1)
 require(path.stat()==before and len(raw)==before.st_size,'input changed during read')
 return raw

def verified_sources(repo, pin_path=None):
 raw=read_bounded(pin_path or Path(__file__).parent/'input-pins.json')
 require(sha(raw)==INPUT_PINS_SHA256,'fixed input-pins SHA256 mismatch')
 pins=decode(raw);require(set(pins)==set(INPUTS),'exact source pin membership')
 sources={name:read_bounded(repo/name) for name in INPUTS}
 for name,digest in pins.items():require(sha(sources[name])==digest,'repository input SHA256: '+name)
 return pins,sources

def module(raw,path,name):
 # Execute the exact bytes verified above, never reopen a checked path.
 m=types.ModuleType(name);m.__file__=str(path)
 # These pinned module call paths import only stdlib. Remove repository and
 # working-directory search paths during execution to prevent local shadows.
 old_path=sys.path[:];stdlib=Path(os.__file__).resolve().parent
 try:
  sys.path[:]=[str(stdlib),str(stdlib/'lib-dynload')]
  exec(compile(raw,str(path),'exec'),m.__dict__)
 finally:sys.path[:]=old_path
 return m

def native_identity(raw, worker):
 # Historical libsndfile WAVs contain a PEAK ancillary chunk. Remove only this
 # metadata in memory to reuse the exact existing native FLOAT sample decoder.
 require(raw[:4]==b'RIFF' and raw[8:12]==b'WAVE' and len(raw)<=8*1024**2 and struct.unpack_from('<I',raw,4)[0]==len(raw)-8,'native RIFF')
 chunks=[];seen=set();off=12
 while off<len(raw):
  require(off+8<=len(raw),'truncated chunk');tag=raw[off:off+4];n=struct.unpack_from('<I',raw,off+4)[0];end=off+8+n+(n%2)
  require(tag not in seen and tag in (b'fmt ',b'fact',b'PEAK',b'data') and end<=len(raw),'native chunk')
  seen.add(tag)
  if tag!=b'PEAK': chunks.append(raw[off:end])
  off=end
 body=b'WAVE'+b''.join(chunks);values,samples=worker.native_samples(b'RIFF'+struct.pack('<I',len(body))+body)
 return {'wav_sha256':sha(raw),'values_sha256':sha(values),'sample_format':'f32le','sample_rate_hz':24000,'channels':1,'frames':len(samples),'bytes':len(values)}

def pcm_identity(raw):
 with wave.open(io.BytesIO(raw),'rb') as w:
  require((w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getcomptype())==(1,2,16000,'NONE'),'derived PCM format')
  frames=w.getnframes();require(0<frames<=192000,'derived frame bound');pcm=w.readframes(frames+1)
 require(len(pcm)==frames*2,'derived PCM byte count')
 return {'sha256':sha(pcm),'bytes':len(pcm),'frames':frames,'sample_rate_hz':16000,'channels':1,'sample_format':'s16le'}

def base(slot, campaign, source):
 voice=slot.get('speaker');target=slot.get('target_text',slot.get('text'));sid=slot.get('slot_id',slot.get('cell_id'))
 return {'record_id':campaign+':'+sid,'slot_id':sid,'source':source,'target_text':target,'target_class':{'你好小窝':'KW1','小窝小窝':'KW2'}[target], 'tts_engine':slot['engine'],'seed':slot['seed'],'preset_voice_id':'qwen-preset:'+voice.casefold(),'preset_voice_name':voice,'speaker_id':None,'conservative_group_id':'qwen-preset:'+voice.casefold(),'group_independence_proven':False,'candidate':True,'training_admitted':False,'human_gold':False,'evaluation_gold':False,'label_origin':'machine-consensus','source_trust_root_admitted':False,'role_eligibility':{'calibration':False,'selection':False,'sealed_test':False},'role_blockers':['machine_consensus_is_not_human_gold','exposed_screening_candidates','source_not_admitted_to_training'],'history_complete':False,'exposure_history':[{'role':'screening','source':source}],'evidence_tier':'machine_three_asr_training_candidate'}

def build(repo, artifacts):
 pins,sources=verified_sources(repo)
 screen=module(sources[INPUTS[4]],repo/INPUTS[4],'registry_screen');blind=module(sources[INPUTS[5]],repo/INPUTS[5],'registry_blind');worker=module(sources[INPUTS[6]],repo/INPUTS[6],'registry_worker')
 zs={n:archive(artifacts/n,h) for n,h in ARCHIVES.items()}
 old=zs['asr-11899008382.zip']; get=lambda z,n:decode(z.read(n))
 candidates=get_repo=decode(sources[INPUTS[0]]);require([x['sample_id'] for x in candidates]==OLD_IDS,'exact old thirteen')
 generations={x['sample_id']:x for x in get(old,'generation-receipts.json')};results=screen.result_index(get(old,'results.json'))
 inventory=decode(sources[INPUTS[1]])['input_artifacts']; rows=[]
 for e in candidates:
  sid=e['sample_id'];engine=e['tts_engine'];name=next(n for n in ARCHIVES if n.startswith(engine+'-'));z=zs[name]
  receipt=get(z,engine+'-receipt.json');slot=next(x for x in receipt['outcomes'] if x['cell_id']==sid);g=generations[sid]
  inv=next(x for x in inventory if x['zip_sha256']==ARCHIVES[name]);source={'campaign_id':'tts-quality-remaining-20261014-v1','tts_run_id':inv['run_id'],'asr_run_id':38573117922,'native_archive':name,'asr_archive':'asr-11899008382.zip'}
  r=base(slot,'tts-quality-remaining-20261014-v1',source);native=native_identity(z.read(sid+'.raw-float.wav'),worker)
  require(native['wav_sha256']==slot['raw_wav_sha256']==g['native_wav_sha256'] and native['values_sha256']==slot['native']['source_values_sha256'],'native identity')
  wav=old.read('audio/'+sid+'.wav');require(sha(wav)==g['wav_sha256'],'derived WAV identity')
  pcm=pcm_identity(wav);r.update(native_audio=native,derived_pcm=pcm,derived_wav={'sha256':sha(wav),'member':'audio/'+sid+'.wav','status':'archived_bytes_verified'},asr=[])
  for family in screen.FAMILIES:
   a=results[(sid,family)]; summary=next(x for x in e['asr'] if x['family']==family)
   require(screen.is_ok(a) and a['completeness']=='complete' and a['quality_flags']==[] and a['attempt']==1 and a['raw_text']==summary['raw_text'] and screen.normalize(a['raw_text'])==r['target_text'],'old ASR exact agreement')
   r['asr'].append({'family':family,'raw_text':a['raw_text'],'normalized_text':screen.normalize(a['raw_text']),'source_member':'results.json','source_member_sha256':sha(old.read('results.json')),'sample_id':sid,'bound_pcm_sha256':None,'binding_status':'historical_id_join_only'})
  r['capture_scope']='historical_screening_summary_not_production_capture';r['same_pcm_asr_binding_verified']=False;r['evidence_gaps']=['Historical ASR rows contain sample_id only; no per-recognizer PCM hash receipt. WAV-to-PCM identity verified now does not retroactively prove decoder input.','Native-to-derived resampling not rerun.']
  r['exposure_history'].append({'role':'development_fit','source':INPUTS[2],'scope':'public summary confirms fixed thirteen; original source/result overlay not recovered; no receipts reconstructed'})
  rows.append(r)
 z=zs['kw1-supplement-11942325284.zip'];plan=get(z,'plan.json');mapping={x['slot_id']:x for x in get(z,'evidence/identity-map.json')};report=get(z,'report.json');require(report['candidate_ids']==NEW_IDS,'exact new three')
 job_raw=z.read('evidence/blind-input/job.json');job=blind.validate_job(job_raw)
 for sid in NEW_IDS:
  slot=next(x for x in plan['slots'] if x['slot_id']==sid);m=mapping[sid];d=m['derived'];h=d['sha256'];pcm=z.read('evidence/blind-input/'+d['file'])
  require(sha(pcm)==h and len(pcm)==d['bytes'] and d in job['clips'],'new PCM identity')
  native=native_identity(z.read('evidence/tts/'+slot['engine']+'/output/'+sid+'.raw-float.wav'),worker)
  require(native['wav_sha256']==m['native_sha256'] and native['values_sha256']==m['native_values_sha256'],'new native identity')
  source={'campaign_id':plan['campaign_id'],'tts_run_id':38664865348,'asr_run_id':38664865348,'native_archive':'kw1-supplement-11942325284.zip','asr_archive':'kw1-supplement-11942325284.zip'}
  r=base(slot,plan['campaign_id'],source);wavhash=sha(blind.canonical_wav(pcm));r.update(native_audio=native,derived_pcm={k:d[k] for k in ('sha256','bytes','frames','sample_rate_hz','channels','sample_format')},derived_wav={'sha256':wavhash,'status':'canonical_header_hash_verified_in_memory_not_archived_WAV'},derivation=m['derivation'],asr=[])
  for family in ('qwen_asr','sensevoice','whisper_small'):
   prefix='evidence/asr/'+family+'/output/';stem=prefix+'cells/'+h+'/';completed=get(z,stem+'completed.json');started=get(z,stem+'started.json');decoder=get(z,stem+'decoder-evidence.json');a=completed['outcome']
   require(sha(z.read(stem+'started.json'))==completed['started_sha256'] and sha(z.read(stem+'decoder-evidence.json'))==completed['decoder_evidence_sha256'],'ASR receipt chain')
   require(started['job_sha256']==sha(job_raw) and started['identity_map_sha256']==sha(z.read(prefix+'identity-map.json')),'ASR job binding')
   require(all(x['pcm_sha256']==h and x['wav_sha256']==wavhash and x['recognizer']==family for x in (started,a)),'same PCM binding')
   require(a['status']=='SUCCESS' and a['attempts']==1 and a['completeness']=='complete' and a['quality_flags']==[] and decoder['result']['raw_text']==a['raw_text'] and screen.normalize(a['raw_text'])==slot['target_text'],'new ASR exact agreement')
   r['asr'].append({'family':family,'raw_text':a['raw_text'],'normalized_text':screen.normalize(a['raw_text']),'source_member':stem+'completed.json','source_member_sha256':sha(z.read(stem+'completed.json')),'bound_pcm_sha256':h,'binding_status':'three_recognizer_pcm_receipts_verified'})
  r['capture_scope']='kw1_blind_receipts_not_production_retained_asr_schema';r['same_pcm_asr_binding_verified']=True;r['evidence_gaps']=['Native-to-derived resampling not rerun.'];rows.append(r)
 out={'schema':'kws-candidate-registry-v1','normalization':{'implementation':INPUTS[4]+':normalize','rule':'Unicode NFKC, remove whitespace and Unicode punctuation only; no homophone or target rewriting'},'input_archives':ARCHIVES,'repository_inputs':pins,'records':rows,'counts':{'total':16,'KW1':5,'KW2':11,'live_native_and_pcm_bytes_verified':16,'same_pcm_three_asr_receipts_verified':3,'historical_asr_id_join_only':13},'training_admitted':False,'human_gold':False,'source_trust_root_admitted':False,'new_model_calls':0,'new_training_calls':0}
 validate(out);return out

def verify_inputs(registry, sources):
 """Research-only, in-memory bytes; does not authorize training or emit receipts.

 sources = {'repo': Path(...), 'artifacts': Path(...)}. The full registry must
 equal offline regeneration, including every hash, evidence gap and role block.
 """
 require(type(sources) is dict and set(sources)=={'repo','artifacts'},'exact source locations')
 repo, artifacts=Path(sources['repo']),Path(sources['artifacts'])
 validate(registry)
 require(canonical(registry)==canonical(build(repo,artifacts)),'registry regeneration mismatch')
 archives={n:archive(artifacts/n,h) for n,h in ARCHIVES.items()};out=[]
 for r in registry['records']:
  if r['slot_id'] in OLD_IDS:
   wav=archives['asr-11899008382.zip'].read('audio/'+r['slot_id']+'.wav')
   with wave.open(io.BytesIO(wav),'rb') as w: pcm=w.readframes(w.getnframes())
  else:
   pcm=archives['kw1-supplement-11942325284.zip'].read('evidence/blind-input/'+r['derived_pcm']['sha256']+'.pcm')
  require(sha(pcm)==r['derived_pcm']['sha256'],'consumer PCM mismatch')
  out.append({**{k:r[k] for k in ('record_id','slot_id','target_class','target_text','preset_voice_id')},'pcm_bytes':pcm,'pcm_sha256':sha(pcm),'sample_rate_hz':16000,'channels':1,'frames':len(pcm)//2})
 return out


def validate(out):
 require(all(out[k] is False for k in ('training_admitted','human_gold','source_trust_root_admitted')),'top-level no admission')
 rows=out['records'];require(len(rows)==16 and len({r['record_id'] for r in rows})==16,'exact unique sixteen')
 require(sorted(r['slot_id'] for r in rows)==sorted(OLD_IDS+NEW_IDS),'exact candidate membership')
 require(sum(r['target_class']=='KW1' for r in rows)==5 and sum(r['target_class']=='KW2' for r in rows)==11,'5/11 distribution')
 for r in rows:
  require(r['candidate'] is True and r['training_admitted'] is False and r['human_gold'] is False and r['evaluation_gold'] is False and r['label_origin']=='machine-consensus' and r['source_trust_root_admitted'] is False,'no admission escalation')
  require(r['speaker_id'] is None and r['conservative_group_id']==r['preset_voice_id'],'preset is not a human speaker')
  require(set(r['role_eligibility'])=={'calibration','selection','sealed_test'} and all(v is False for v in r['role_eligibility'].values()),'evaluation blocked')
  p=r['derived_pcm'];require(p['bytes']==2*p['frames'] and p['sha256']!=r['derived_wav']['sha256'],'PCM is not WAV')
  require(len(r['asr'])==3 and len({a['family'] for a in r['asr']})==3,'three recognizers')
  require(r['same_pcm_asr_binding_verified'] is (r['slot_id'] in NEW_IDS),'historical binding cannot be promoted')
  for a in r['asr']: require(a['bound_pcm_sha256']==(p['sha256'] if r['slot_id'] in NEW_IDS else None),'ASR PCM binding')
 return True

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args();out=build(a.repo,a.artifacts)
 if a.verify: require(canonical(decode(a.verify.read_bytes()))==canonical(out),'registry differs from offline regeneration')
 if a.output: a.output.write_text(canonical(out),encoding='utf-8')
 print('Verified 16 candidate identities (5 KW1 / 11 KW2); 3 strong same-PCM ASR chains; 13 historical ID joins; no admission.')
