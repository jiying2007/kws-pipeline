#!/usr/bin/env python3
"""Read-only, stdlib-only validator for closed saved evidence. No native execution.

Never reads audio/feature/model files, starts a subprocess, fetches a URL, imports
a model, computes softmax, or replays the decoder. Oracle uses invented rational
probability tables and exhaustive CTC arithmetic only.
"""
import argparse, collections, hashlib, json, math, re, struct, sys, unicodedata
from fractions import Fraction as Q
from itertools import product, groupby
from pathlib import Path

RAW_PINS = {
 'demand/raw.jsonl': (1537908, '4605f7e8e581a2556db9bd2797bbadf29c741228011357be3b1595b06c7f5882'),
 'fleurs20/raw.jsonl': (1271113, 'a3abea95149320e48f0b2a8063f7d8b5f73791fc92fd4496da5f3f9fc00ca3db'),
 'qwen3case/raw.jsonl': (166648, '556a01a6a7d1a1f9ac0262310e24f2aa82608843e2549d93a0f76ac37770f236'),
 'path-witness/raw.jsonl': (23574, 'f256fe8301180fae43c9bca9259a2d59e2d12907c6559d764a0090e5c93e0b38')
}
KEYWORDS={1:(1,2,3,4), 2:(3,4,3,4)}
def need(ok, why):
 if not ok: raise ValueError(why)
def sha(b): return hashlib.sha256(b).hexdigest()
def no_duplicates(pairs):
 d={}
 for k,v in pairs:
  need(k not in d,'duplicate JSON key'); d[k]=v
 return d
def reject_constant(s): raise ValueError('nonfinite JSON number')
def parse(text, lexical=False):
 return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant,
                   **({'parse_float':str,'parse_int':str} if lexical else {}))
def read(p): return parse(p.read_text())
def evidence(p):
 o=read(p)
 if 'saved_evidence' in o:
  need(o['record_type']=='PUBLIC_PROJECTION_OF_SAVED_EVIDENCE','projection label')
  need(o['projection_was_executed_as_experiment'] is False,'projection executed claim')
  return o['saved_evidence']
 return o
def lines(p): return [parse(x) for x in p.read_text().splitlines()]
def bitfloat(s, width):
 need(type(s) is str and re.fullmatch('[0-9a-f]{'+str(width//4)+'}',s),'IEEE bit string')
 v=struct.unpack('>f' if width==32 else '>d',bytes.fromhex(s))[0]
 need(math.isfinite(v),'nonfinite IEEE bits'); return v
def f64bits(v): return struct.pack('>d',v).hex()
def safe_relative(s):
 p=Path(s)
 need(type(s) is str and not p.is_absolute() and p.parts and '..' not in p.parts and '.' not in p.parts and '\\' not in s,'unsafe relative path')
 return p
def check_manifest(root):
 m=read(root/'MANIFEST.json'); need(m['schema']=='a20-domain-public-payload-manifest-v1','manifest schema')
 listed=set()
 for e in m['files']:
  rel=safe_relative(e['path']);need(str(rel) not in listed,'duplicate manifest path');listed.add(str(rel))
  need(rel.suffix in ('.json','.jsonl','.md'),'nontext evidence extension')
  p=root/rel;need(p.is_file() and not p.is_symlink(),'missing file or symlink')
  need(all(not x.is_symlink() for x in [p,*p.parents] if x!=root.parent),'symlink parent')
  b=p.read_bytes();b.decode('utf-8');need(not b.startswith((b'RIFF',b'\x7fELF')),'binary payload')
  need(len(b)==e['bytes'] and sha(b)==e['sha256'],'payload identity: '+str(rel))
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
 need(actual==listed|{'MANIFEST.json'},'unlisted or missing payload')
 for rel,(n,h) in RAW_PINS.items():
  b=(root/rel).read_bytes();need(len(b)==n and sha(b)==h,'independent immutable raw pin: '+rel)
 return {'files':len(listed)+1,'bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}

def publication_pin(reference):
 """Pure fail-closed admission of an immutable, separately reviewed data pin."""
 need(reference.get('repository')=='jiying2007/kws-data','data repository binding')
 need(reference.get('candidate_path')=='research/2026-10-04-domain-decoder-diagnostics','data record path binding')
 commit=reference.get('new_record_commit')
 need(reference.get('publication_ready') is True,'data reference NOT_PUBLISHED or publication gate closed')
 need(type(commit) is str and re.fullmatch('[0-9a-f]{40}',commit),'require immutable published data commit')
 need(type(reference.get('manifest_sha256')) is str and re.fullmatch('[0-9a-f]{64}',reference['manifest_sha256']),'require manifest SHA256')
 need(type(reference.get('manifest_bytes')) is int and reference['manifest_bytes']>0,'require manifest byte count')
 return commit

def check_data_reference(root, require_published=False, actual_commit=None):
 reference=read(Path(__file__).resolve().parents[1]/'DATA-REFERENCE.json')
 need(reference['repository']=='jiying2007/kws-data' and reference['candidate_path']=='research/2026-10-04-domain-decoder-diagnostics','data reference scope')
 manifest=(root/'MANIFEST.json').read_bytes()
 need(len(manifest)==reference['manifest_bytes'] and sha(manifest)==reference['manifest_sha256'],'DATA-REFERENCE manifest binding')
 commit=reference['new_record_commit']
 if require_published or actual_commit is not None:
  commit=publication_pin(reference)
  need(type(actual_commit) is str and actual_commit==commit,'actual data checkout commit mismatch or missing')
 return {'manifest_sha256':sha(manifest),'data_commit':commit,'published_commit_gate_required':require_published,'actual_checkout_commit_checked':actual_commit is not None}

def provenance(root):
 p=read(root/'PROVENANCE.json');seen=set()
 for e in p['entries']:
  rel=safe_relative(e['public_path']).relative_to('research/2026-10-04-domain-decoder-diagnostics')
  need(str(rel) not in seen,'duplicate provenance target');seen.add(str(rel));b=(root/rel).read_bytes()
  need(len(b)==e['public_bytes'] and sha(b)==e['public_sha256'],'public provenance identity')
  if e['transformation']=='EXACT_BYTES_UNCHANGED':
   need(e['original_bytes']==e['public_bytes'] and e['original_sha256']==e['public_sha256'],'exact original/public identity')
  elif e['transformation']=='EXPLICIT_PUBLIC_PROJECTION_NOT_HISTORICAL_RAW':
   o=parse(b.decode());need(o['original_evidence_sha256']==e['original_sha256'],'projection origin binding')
   need(o['projection_was_executed_as_experiment'] is False,'projection execution claim')
 return {'bound_payloads':len(seen),'projections_are_new_not_historical_execution':True}

def validate_schema(rows,schema):
 need(len(rows)==schema['records'],'raw record count')
 counts=collections.Counter()
 for row in rows:
  kind=row.get('kind',row.get('type'));need(kind in schema['kinds'],'unknown record kind');counts[kind]+=1;g=schema['kinds'][kind]
  def walk(v,p='$'):
   need(p in g['fields'] and type(v).__name__ in g['fields'][p],'unknown field/type: '+p)
   if isinstance(v,dict):
    need(sorted(v) in g['object_keys'][p],'unknown object keys: '+p)
    for k,w in v.items():walk(w,p+'.'+k)
   elif isinstance(v,list):
    for w in v:walk(w,p+'[]')
   elif isinstance(v,float):need(math.isfinite(v),'nonfinite numeric value')
   elif isinstance(v,str):
    s=schema['string_classes'][p]
    need(bool(re.fullmatch(s['pattern'],v)) if 'pattern' in s else v in s['values'],'unknown raw string: '+p)
  walk(row)
 need(dict(counts)=={k:v['records'] for k,v in schema['kinds'].items()},'kind count mismatch')
 return dict(counts)

def natural(rows,expected):
 need(rows[0]['kind']=='run_start' and rows[-1]['kind']=='run_end','run boundaries')
 clips={};active=None;events=[]
 for r in rows:
  k=r['kind']
  if k=='clip_start':
   need(active is None and r['recording'] not in clips,'overlapping/repeated clip')
   active={'start':r,'callbacks':[],'feeds':[],'finishes':[]};clips[r['recording']]=active
  elif k in ('callback','feed','finish','clip_end'):
   need(active is not None and r['recording']==active['start']['recording'],'clip scope')
   if k=='callback':
    cs=active['callbacks'];need(r['call_index']==len(cs),'callback index')
    n=r['selected_rows'];need(n==len(r['logits'])==len(r['centers']),'callback shape')
    need(all(len(v)==6 and all(type(x) in (int,float) and math.isfinite(x) for x in v) for v in r['logits']),'six finite logits')
    previous=sum(x['selected_rows'] for x in cs)*3
    need(r['centers']==list(range(previous,previous+3*n,3)),'frame centers')
    need(r['decoder_total_frames']==previous+3*n,'full callback clock')
    need(0<=r['decoder_rows_decoded']<=n,'search row bounds')
    need(r['keyword'] in (0,1,2) and r['state'] in (0,1),'event domain')
    if r['state']==1:
     need(r['keyword'] in (1,2) and math.isfinite(r['score']),'event value')
     events.append({'recording':r['recording'],'keyword':r['keyword'],'available_audio_s':r['available_samples']/16000,'score':r['score'],'start_frame':r['start_frame'],'end_frame':r['end_frame']})
    cs.append(r)
   elif k=='feed':
    need(r['feed_index']==len(active['feeds']),'feed index')
    need(r['cumulative_samples']==sum(x['input_samples'] for x in active['feeds'])+r['input_samples'],'feed sum');active['feeds'].append(r)
   elif k=='finish':active['finishes'].append(r)
   else:
    cs=active['callbacks'];need(r['complete'] is True,'incomplete clip')
    for key in ('frames','wav_sha256','pcm_sha256'):need(r[key]==active['start'][key],'clip identity')
    sums={'callbacks':len(cs),'feed_calls':len(active['feeds']),'finish_calls':len(active['finishes']),'model_rows':sum(x['selected_rows'] for x in cs),'decoder_rows_decoded':sum(x['decoder_rows_decoded'] for x in cs),'fbank_rows':sum(x['fbank_rows'] for x in cs),'event_count':sum(x['state']==1 for x in cs)}
    need(all(r[k]==v for k,v in sums.items()),'clip totals');need(sums['finish_calls']==1,'one finish')
    need(sum(x['input_samples'] for x in active['feeds'])==r['frames'],'input completeness');active['end']=r;active=None
 need(active is None,'unclosed clip');end=rows[-1]
 for k in ('frames','callbacks','feed_calls','finish_calls','fbank_rows','model_rows','decoder_rows_decoded','event_count'):
  need(end[k]==sum(c['end'][k] for c in clips.values()),'aggregate '+k)
 need(end['complete'] and end['clips']==len(clips),'run complete')
 need(all(end[k]==v for k,v in expected.items()),'fixed expected observations')
 return {'clips':len(clips),'input_seconds':end['frames']/16000,'callbacks':end['callbacks'],'model_rows':end['model_rows'],'searched_rows':end['decoder_rows_decoded'],'events':events,'qualified_negative_hours':0}

def positive32(bits):
 e,m=bits>>23,bits&0x7fffff
 return Q(m if e==0 else m+2**23)*Q(2)**(-149 if e==0 else e-150)
def strict_cell(lexeme,bits):
 b=int(bits,16);m=b&0x7fffffff;s=-1 if b>>31 else 1
 need(0x800000<=m<0x7f7fffff,'proof requires normal finite nonzero interior')
 v=s*positive32(m);prev=s*positive32(m-1 if s==1 else m+1);nxt=s*positive32(m+1 if s==1 else m-1)
 low=(prev+v)/2;high=(v+nxt)/2;d=Q(lexeme)
 need(low<d<high,'decimal outside strict RNE cell')
 return d-low,high-d,d-v

def qwen(root,rows):
 orig=lines(root/'qwen3case/original-selected-records.jsonl');cs=[r for r in orig if r['kind']=='callback'];out=[r for r in rows if r['kind']=='callback_result']
 need(len(cs)==len(out)==14,'14 callbacks')
 keys={'valid':'valid','state':'state','keyword':'keyword','start_frame':'start_frame','end_frame':'end_frame','decoder_rows_decoded':'rows_decoded','decoder_total_frames':'total_frames'}
 for old,new in zip(cs,out):
  need(old['recording']==new['clip'] and old['call_index']==new['call'],'callback binding')
  need(all(old[k]==new[v] for k,v in keys.items()),'historical callback fields')
  need(f64bits(float(old['score']))==new['score_bits'],'historical score bit identity')
  need(new['exact_fields'] and new['score_within_1e_5'] and new['score_bitwise_equal'],'saved gates')
 tops=[x for x in rows if x['kind']=='top'];need(len(tops)==127,'searched rows')
 for r in tops:
  ps=[bitfloat(s,32) for s in r['probs_f32_bits']];need(len(ps)==6 and all(0<=v<=1 for v in ps),'probabilities')
  need(abs(sum(map(Q,ps))-1)<=Q(1,1000000),'captured probability normalization')
  ts=r['top'];need(len(ts)==len(set(ts))==3 and all(type(i) is int and 0<=i<6 for i in ts),'top IDs')
  need(all(ps[ts[i]]>=ps[ts[i+1]] for i in (0,1)),'top order')
  need(all(ps[i]<=ps[ts[-1]] for i in range(6) if i not in ts),'omitted higher probability')
 for r in rows:
  if 'nodes' not in r:continue
  ns=r['nodes'];hs=r['hyps'];need(len(ns)<=2680 and len(hs)<=120,'state bounds')
  for token,frame,p in ns:need(1<=token<=5 and frame>=0 and frame%3==0 and 0<=bitfloat(p,64)<=1,'node shape')
  for pb,pnb,ids in hs:
   need(bitfloat(pb,64)>=0 and bitfloat(pnb,64)>=0,'nonnegative mass');need(len(ids)<=128 and all(type(i) is int and 0<=i<len(ns) for i in ids),'node index')
 proof=read(root/'qwen3case/decimal-proof.json')['proof'];lex=[]
 for line in (root/'qwen3case/original-selected-records.jsonl').read_text().splitlines():
  r=parse(line,lexical=True)
  if r['kind']=='callback':
   for i,row in enumerate(r['logits']):
    for token,s in enumerate(row):lex.append((s,[r['recording'],int(r['call_index']),i,token]))
 need(len(proof)==len(lex)==768,'768 scalar proof')
 recovered=bytearray()
 for (s,where),p in zip(lex,proof):
  need(s==p['lexeme'] and where==p['where'],'lexeme correspondence')
  low,high,delta=strict_cell(s,p['binary32_bits'])
  need(low==Q(p['lower_margin']) and high==Q(p['upper_margin']) and delta==Q(p['exact_decimal_minus_float']),'rational margins')
  recovered.extend(struct.pack('<I',int(p['binary32_bits'],16)))
 need(sha(recovered)=='d0f40402ab874eb22110ee98dec272ff8811de8e6833aa4a52ed4104ac94669c','reconstructed numeric logit identity')
 need(sum(x['selected_rows'] for x in cs)==128 and sum(x['decoder_rows_decoded'] for x in cs)==127,'original row counts')
 need(evidence(root/'qwen3case/result-summary.json')['original_qwen20_status']=='FAILED_NO_RETRY_UNCHANGED','failure preserved')
 return {'callbacks_bitwise_scores_equal':14,'input_rows':128,'searched_rows':127,'scalar_strict_rational_cells':768,'probabilities':'NEW_RECONSTRUCTION_NOT_HISTORICAL_TRACE','original_qwen20_status':'FAILED_NO_RETRY_UNCHANGED'}

def collapse(raw): return tuple(t for t,_ in groupby(raw) if t)
def enumerate_paths(rows):
 need(1<=len(rows)<=24,'bounded oracle length')
 choices=[]
 for row in rows:
  need(len(row)==6,'six oracle probabilities');vs=list(map(Q,row));need(all(0<=p<=1 for p in vs) and sum(vs)==1,'oracle distribution')
  choices.append([(i,p) for i,p in enumerate(vs) if p])
 need(math.prod(map(len,choices))<=6561,'bounded Cartesian paths')
 return [(tuple(i for i,p in c),math.prod(p for i,p in c)) for c in product(*choices)]
def groups(paths,row,start):
 out={}
 for raw,w in paths:
  seg=raw[start:row+1];t=collapse(seg);g=out.setdefault(t,[Q(0),Q(0)]);g[0 if seg[-1]==0 else 1]+=w
 return out
def has_witness(paths,rows,row,start,prefix,frames,probs):
 if len(frames)!=len(prefix) or len(probs)!=len(prefix) or any(f%3 for f in frames):return False
 for raw,w in paths:
  if not w:continue
  runs=[]
  for j,t in enumerate(raw[start:row+1],start):
   if j>start and t==raw[j-1]:
    if t:runs[-1][1].append(j)
   elif t:runs.append((t,[j]))
  if tuple(t for t,rr in runs)!=tuple(prefix):continue
  if all(f//3 in rr and Q(rows[f//3][t])==Q(p) for (t,rr),f,p in zip(runs,frames,probs)):return True
 return False

def witness(root,raw):
 fs=read(root/'path-witness/fixtures.json')['fixtures'];need(len(fs)==6,'exactly six closed fixtures')
 reference=read(root/'path-witness/exact-ctc-reference.json');comparison=read(root/'path-witness/saved-comparison.json');rr=[x for x in raw if x['type']=='row']
 need(sum(len(f['rows']) for f in fs)==len(rr)==40,'40 rows');need(raw[-1]['events']==5 and raw[-1]['softmax_calls']==raw[-1]['logits_calls']==0,'native historical counters')
 allpaths={};checked=0;observed=[]
 for f in fs:
  fid=f['id'];ps=enumerate_paths(f['rows']);allpaths[fid]=ps
  rs=[x for x in rr if x['fixture']==fid];need(len(rs)==len(f['rows']),'fixture rows')
  ref=next(x for x in reference['fixtures'] if x['id']==fid);starts=[0]+[e['row']+1 for e in f['expected_events'] if e['row']+1<len(rs)]
  for i,r in enumerate(rs):
   need(r['row']==i and r['frame']==3*i and r['clock_after']==3*(i+1),'invented fixture coordinates')
   need(r['observer_api_equal'] is True,'saved observer parity');start=max(s for s in starts if s<=i);g=groups(ps,i,start)
   refg={tuple(x['tokens']):[Q(x['pb']),Q(x['pnb'])] for x in ref['rows_reference'][i]['groups']}
   need(g==refg,'exact saved reference')
   for rank,h in enumerate(r['hyps'],1):
    need(h['rank']==rank,'rank order');nodes=h['nodes'];prefix=tuple(n['token'] for n in nodes)
    need(prefix in g and [Q(h['pb']),Q(h['pnb'])]==g[prefix],'retained pb/pnb exact');checked+=1
    for n in nodes:need(n['frame']%3==0 and 0<=n['frame']<=r['frame'] and Q(n['prob'])==Q(f['rows'][n['frame']//3][n['token']]),'selected local probability')
   if r['result']['state']:observed.append((fid,i,r['result']['keyword'],r['result']['start'],r['result']['end']))
 expected=[(f['id'],e['row'],e['keyword'],e['start'],e['end']) for f in fs for e in f['expected_events']]
 need(observed==expected and checked==91,'91 states and 5 events')
 observations=comparison['observations'];need(len(observations)==7,'seven saved checks')
 for o in observations:
  f=next(f for f in fs if f['id']==o['fixture']);r=next(r for r in rr if r['fixture']==f['id'] and r['row']==o['row']);h=r['hyps'][o['selected_rank']-1]
  nodes=h['nodes'];prefix=[n['token'] for n in nodes];frames=[n['frame'] for n in nodes];probs=[Q(n['prob']) for n in nodes]
  need(prefix==o['whole_prefix'] and frames==o['selected_frames'] and probs==list(map(Q,o['selected_probs_exact'])),'observation binding')
  start=max([0]+[e['row']+1 for e in f['expected_events'] if e['row']<r['row']]);joint=has_witness(allpaths[f['id']],f['rows'],r['row'],start,prefix,frames,probs)
  need(joint==o['selected_tuple_same_path'],'same-path property')
  score=math.sqrt(float(r['prior_hit_score'])*float(math.prod(probs)));need(f64bits(score)==f64bits(o['observed_score'])==f64bits(r['post_detect_hit_score']),'score formula bit identity')
  dur=frames[-1]-frames[0];last=r['prior_last_active'];active=5<=dur<=250 and (last==-1 or frames[-1]-last>=50) and score>=0
  need(active==o['activation']==bool(r['result']['state']),'unchanged guards')
 return {'fixtures':6,'rows':40,'retained_mass_states_exact':checked,'match_reject_checks':7,'events':len(observed),'score_formula_max_ulps':0,'original_port_bug_established':False,'new_native_calls':0}

def text_presence(transcription,raw_transcription):
 cleaned=''.join(c for c in transcription if not c.isspace() and not unicodedata.category(c).startswith('P'))
 return {'exact_keyword_text_hits':{k:k in cleaned for k in ['你好小窝','小窝小窝']},'explicit_nearphrase_text_proxy_hits':[k for k in ['你好','您好','小窝','小我','小沃','小卧','晓我','小屋'] if k in cleaned],'tracked_character_presence':[k for k in ['你','好','小','窝','屋'] if k in cleaned]}
def verify_optional_texts(source, supplied):
 """Optional caller-owned text JSON; no network or audio; never invoked by default."""
 need([x['source_row_index'] for x in supplied]==list(range(20)),'external text fixed rows')
 for expected,actual in zip(source['records'],supplied):
  for k in ('transcription','raw_transcription'):need(sha(actual[k].encode())==expected[k+'_sha256'],'exact external transcript identity')
  need(all(expected[k]==v for k,v in text_presence(actual['transcription'],actual['raw_transcription']).items()),'text-only presence')
 return 'PASS_TEXT_ONLY_NOT_AUDIO_ABSENCE'
def source_metadata(root):
 s=evidence(root/'fleurs20/source-admission.json');d=evidence(root/'fleurs20/pcm16-derivation.json');a=s['summary'];rs=s['records'];ds=d['rows']
 need([r['source_row_index'] for r in rs]==[r['source_row_index'] for r in ds]==list(range(20)),'fixed source rows')
 need(len(set(r['sentence_id'] for r in rs))==19 and [r['source_row_index'] for r in rs if r['sentence_id']==1519]==[7,17],'sentence groups')
 need(sum(r['frames'] for r in rs)==3960960 and a['duration_s']==247.56,'source frames')
 need(sum(any(r['exact_keyword_text_hits'].values()) for r in rs)==a['full_keyword_text_hit_rows']==0,'saved text absence counts')
 need(a['qualified_negative_hours']==0 and not a['spoken_keyword_absence_verified'] and not a['audio_transcript_consistency_verified'],'source evidence boundary')
 for r,q in zip(rs,ds):
  need(r['body_sha256']==q['source_wav_sha256'] and r['decoded_payload_sha256']==q['source_float32_payload_sha256'],'source/derived identities')
  need(r['frames']==q['frames']==q['sample_count'] and q['changed_sample_count']+q['unchanged_sample_count']==q['sample_count'],'derivation counts')
  need(q['lossy'] and q['clipped_sample_count']==q['negative_clip_count']==q['positive_clip_count']==0,'lossy status / clipping')
  need(0<=q['max_absolute_quantization_error']*32768<=0.5,'recorded error bound')
  for k in ('transcription_sha256','raw_transcription_sha256'):need(r[k]==q[k],'text hash preservation')
 need(sum(q['changed_sample_count'] for q in ds)==3905710,'changed sample total')
 starts=[r for r in lines(root/'fleurs20/raw.jsonl') if r['kind']=='clip_start']
 for start,q in zip(starts,ds):need(start['wav_sha256']==q['derived_wav_sha256'] and start['pcm_sha256']==q['derived_payload_sha256'],'A20 derived-input identity')
 return {'source_rows':20,'sentence_groups':19,'duration_s':247.56,'qualified_negative_hours':0,'source_or_derived_audio_reopened':False,'text_presence_recomputed_from_full_text':False,'conversion_reexecuted':False}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--texts',type=Path);p.add_argument('--require-published',action='store_true');p.add_argument('--data-commit',help='Actual data checkout HEAD from git rev-parse; required with --require-published');a=p.parse_args();root=a.data_root
 binding=check_data_reference(root,a.require_published,a.data_commit)
 result={'data_reference':binding,'status':'PASS_SAVED_ONLY_OFFLINE_VALIDATION','manifest':check_manifest(root),'new_native_audio_model_softmax_replay_probe_calls':0}
 result['provenance']=provenance(root)
 schema=read(root/'RAW-SCHEMA-REVIEW.json')['schemas'];raw={}
 for rel,s in schema.items():raw[rel]=lines(root/rel);validate_schema(raw[rel],s)
 result['demand']=natural(raw['demand/raw.jsonl'],{'clips':1,'frames':4800064,'callbacks':1001,'model_rows':9999,'decoder_rows_decoded':9999,'fbank_rows':29998,'event_count':0})
 result['fleurs20']=natural(raw['fleurs20/raw.jsonl'],{'clips':20,'frames':3960960,'callbacks':835,'model_rows':8232,'decoder_rows_decoded':8226,'fbank_rows':24716,'event_count':2})
 ev=result['fleurs20']['events'];need([(r['recording'],r['keyword'],r['available_audio_s'],r['score']) for r in ev]==[('FLEURS-cmn20-row16',1,15.3,0.12134267955397253),('FLEURS-cmn20-row17',2,6.3,0.14145915542700957)],'both events preserved')
 result['source_metadata']=source_metadata(root);result['qwen3case']=qwen(root,raw['qwen3case/raw.jsonl']);result['path_witness']=witness(root,raw['path-witness/raw.jsonl'])
 if a.texts:result['optional_texts']=verify_optional_texts(evidence(root/'fleurs20/source-admission.json'),read(a.texts))
 print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
