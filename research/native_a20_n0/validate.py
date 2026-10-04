#!/usr/bin/env python3
"""Offline, standard-library validation of complete saved N0 evidence.
No model/audio/native loading, DSP, probability reconstruction, process spawn or /proc reads.
The temporary files contain only exact decompressed JSON bytes for the unchanged scorer.
"""
import argparse,collections,hashlib,importlib.util,json,math
from pathlib import Path,PurePosixPath
import re,sys,tempfile,zlib
RAW_BYTES=4617273
RAW_SHA='ac881e0407496a92dbcbe5ed54adc2f2367bd36478240ed093f1433c4c6be998'
GEOMETRY_BYTES=1947411
GEOMETRY_SHA='46c2a74c276a472167b4f80e04aa7950efa3c64e4805633f9643c9ead04e5c75'
PROTOCOL_SHA='4c98537a997a1451626ceef47d032d0f27257473d971974578002fc5d5aada6f'
SCORER_SHA='48c0fc5bdb94a70e748ce86f2b1a4bb4915594605c9549f0187f0a5f1c425139'
MANIFEST_SHA='3063d60ecb973ab2e6f1c4de7b25c8db5ef5cbc95c268e4fc3511dec0a52919b'
class Invalid(ValueError):pass
def require(x,msg):
 if not x:raise Invalid(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def unique(pairs):
 out={}
 for k,v in pairs:
  require(k not in out,'duplicate JSON key');out[k]=v
 return out
def parse(b):return json.loads(b,object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(Invalid('nonfinite JSON')))
def safe_path(root,name):
 require(isinstance(name,str) and name,'empty path');q=PurePosixPath(name)
 require(not q.is_absolute() and str(q)==name and not any(x in ('','.','..') for x in q.parts) and '\\' not in name and '\0' not in name,'unsafe path')
 p=Path(root)
 for part in q.parts:p=p/part;require(not p.is_symlink(),'symlink path')
 return p
def inventory(root,name):
 root=Path(root);require(root.is_dir() and not root.is_symlink(),'invalid root')
 m=parse(safe_path(root,name).read_bytes());require(m['schema']=='n0-public-file-manifest-v1','wrong manifest schema');names=set()
 for row in m['files']:
  n=row['path'];require(n not in names and n!=name,'duplicate/self manifest');names.add(n);p=safe_path(root,n)
  require(type(row['bytes']) is int and 0<=row['bytes']<=2097152,'unbounded size')
  require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p.read_bytes())==row['sha256'],'inventory identity '+n)
 actual=set()
 for p in root.rglob('*'):
  require(not p.is_symlink(),'symlink inventory')
  if p.is_file():actual.add(p.relative_to(root).as_posix())
 require(actual==names|{name},'unlisted or missing file')
 require(sum(x['bytes'] for x in m['files'])==m['total_bytes'],'manifest byte total');return m
def unpack(blob,size,digest):
 d=zlib.decompressobj(31)
 try:b=d.decompress(blob,size+1)
 except zlib.error as e:raise Invalid('invalid gzip') from e
 require(d.eof and not d.unused_data and not d.unconsumed_tail,'incomplete/extra gzip member')
 require(len(b)==size and sha(b)==digest,'decoded identity');return b
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def validate(code,data,require_published=False):
 code=Path(code);data=Path(data);cm=inventory(code,'CODE-MANIFEST.json');dm=inventory(data,'EVIDENCE-MANIFEST.json')
 read=lambda n:parse(safe_path(data,n).read_bytes())
 ref=parse((code/'DATA-REFERENCE.json').read_bytes())
 require(ref['repository']=='jiying2007/kws-data' and ref['path']=='research/2026-10-04-n0-deterministic-controls','data scope')
 require(ref['manifest_sha256']==sha((data/'EVIDENCE-MANIFEST.json').read_bytes()),'data manifest pin')
 if ref['publication_ready']:require(re.fullmatch('[0-9a-f]{40}',ref['commit']) is not None,'mutable data ref')
 else:require(ref['commit']=='PENDING_REVIEWED_DATA_PUBLICATION' and not require_published,'publication blocked: immutable reviewed data pin required')
 prov=read('PROVENANCE.json');require(prov['original_protocol_sha256']==PROTOCOL_SHA and prov['original_scientific_manifest_sha256']==MANIFEST_SHA,'original identities')
 require(prov['public_projection_was_not_executed'] is True,'projection run claim')
 mapped=set()
 for row in prov['mapping']:
  key=(row['repository'],row['public_path']);require(key not in mapped,'duplicate projection');mapped.add(key)
  require(row['repository'] in ('jiying2007/kws-data','jiying2007/kws-pipeline'),'mapping repository')
  b=safe_path(data if row['repository'].endswith('/kws-data') else code,row['public_path']).read_bytes()
  require({'bytes':len(b),'sha256':sha(b)}==row['public'],'projection byte identity')
  if row['transformation']=='byte-identical':require(row['original']==row['public'],'false exact mapping')
  if row['transformation']=='gzip byte-exact original':unpack(b,row['original']['bytes'],row['original']['sha256'])
 raw=unpack((data/'evidence/raw.jsonl.gz').read_bytes(),RAW_BYTES,RAW_SHA)
 gb=unpack((data/'metadata/geometry.json.gz').read_bytes(),GEOMETRY_BYTES,GEOMETRY_SHA)
 require(raw.endswith(b'\n') and all(raw.splitlines()),'raw truncation')
 records=[parse(x) for x in raw.splitlines()];counts=dict(collections.Counter(x['kind'] for x in records))
 require(counts=={'run_start':1,'model_loaded':1,'clip_start':3,'callback':3000,'feed':3000,'finish':3,'clip_end':3,'run_end':1},'raw record coverage')
 require(len(records)==6012,'raw records');cb=[x for x in records if x['kind']=='callback']
 require(sum(len(x['logits']) for x in cb)==29997 and all(len(v)==6 and all(type(z) in (int,float) and math.isfinite(z) for z in v) for x in cb for v in x['logits']),'complete six-class logits')
 require(sum(x['decoder_rows_decoded'] for x in cb)==29997 and sum(x['state'] for x in cb)==0,'observed search/events')
 require(sha((data/'metadata/verified-inputs.json').read_bytes())==MANIFEST_SHA,'exact input metadata')
 require(sha((code/'src/score_n0.py').read_bytes())==SCORER_SHA,'frozen scorer changed')
 s=module(code/'src/score_n0.py','n0_frozen_scorer');header=records[0]
 require(header['protocol_sha256']==PROTOCOL_SHA,'raw protocol')
 with tempfile.TemporaryDirectory(prefix='n0-saved-json-') as td:
  rp=Path(td)/'raw.jsonl';gp=Path(td)/'geometry.json';rp.write_bytes(raw);gp.write_bytes(gb)
  report=s.score(rp,data/'metadata/verified-inputs.json',gp,{k:header[k] for k in s.HASH_KEYS})
 expected=read('evidence/constructed-domain-score.json');require(report==expected,'complete saved scorer mismatch')
 # Independent pure integer geometry recomputation, not frontend or decoder replay.
 geometry=module(code/'src/geometry.py','n0_integer_geometry');savedg=parse(gb)
 for row in savedg['rows']:require({k:v for k,v in row.items() if k!='recording'}==geometry.geometry(4800000),'source-derived integer geometry')
 resource=read('evidence/resource-record.public.json');live=[x for x in resource['proc_samples'] if x['observation']=='LIVE']
 require(len(resource['proc_samples'])==108 and len(live)==107,'resource sample count')
 require(resource['status']=='RAW_COMPLETE_PENDING_AUDIT' and resource['returncode']==0 and resource['guard_stop_reason'] is None,'original resource status')
 require(resource['children_observation']=='NOT_AVAILABLE' and resource['lifetime_child_absence']=='NOT_PROVEN' and resource['terminal_lifetime_io']=='UNKNOWN','observability boundary')
 for x in live:
  e=x['children_error'];require(x['children'] is None and x['children_count'] is None and x['children_observation']=='NOT_AVAILABLE','missing is not zero')
  require(e['type']=='FileNotFoundError' and e['errno']==2 and e['path']==e['filename']=='/proc/6/task/6/children','exact owned endpoint error')
  require(x['Threads']==1 and type(x['VmRSS']) is int and x['VmRSS']>=0 and x['critical_monitoring']=='OBSERVED_RSS_THREADS_ONLY','mandatory telemetry')
 require(max(x['VmRSS'] for x in live)*1024==2912256==resource['sampled_peak_rss_bytes'],'sampled RSS')
 require(all(x['status']=='UNKNOWN' and x['value'] is None for x in resource['io_final_observation'].values()),'terminal IO unknown')
 require(resource['elapsed_wall_s']==10.84850817300321 and resource['rss_limit_kind']=='SAMPLED_RSS_GUARD_NOT_KERNEL_HARD_RSS_LIMIT','r2 resource labels')
 require(records[-1]['wall_ns']==10672298922 and records[-1]['process_cpu_ns']==10669888429 and records[-1]['maxrss_kib']==39376,'native resource values')
 probe=read('evidence/strict-probe-failure.public.json')
 require(probe['status']=='FAILED_NO_RETRY' and probe['candidate_runs']==0 and probe['probe_processes_attempted']==1 and probe['retries_allowed'] is False and probe['returncode']==-9,'strict probe failure preserved')
 require(probe['proc_samples'][0]['critical_errors']['children']['errno']==2 and probe['owned_live_guard_samples']==0 and probe['cleanup']['reaped'] is True,'strict probe failure detail')
 protocol=read('metadata/protocol.public.json');contract=protocol['supervision_contract']
 require(protocol['status']=='PUBLIC_TECHNICAL_PROJECTION_OF_FROZEN_PROTOCOL_NOT_EXECUTED','protocol execution label')
 require(contract['children_enoent']=='EXACT_OWNED_PID_CHILDREN_PATH_ONLY_NOT_AVAILABLE_NULL' and contract['other_children_errors']==contract['known_nonempty_children']=='STOP' and contract['second_probe']=='DISABLED','revised error gate')
 require(protocol['passes']==1 and protocol['retries']==protocol['warmups']==protocol['threshold_sweeps']==0 and protocol['training'] is False,'run cardinality')
 audit=read('evidence/audit-summary.public.json');require(audit['status']=='PASS_DESCRIPTIVE_WITH_LIMITATIONS' and audit['record_counts']==counts and audit['aggregate']['six_class_raw_logit_values_validated']==179982,'audit numeric summary')
 return {'status':'PASS_SAVED_SCIENTIFIC_EVIDENCE_ONLY','data_files':len(dm['files'])+1,'source_files':len(cm['files'])+1,'records':len(records),'callbacks':3000,'fbank_rows':89994,'model_and_search_rows':29997,'logit_values':179982,'events':0,'strict_probe':'FAILED_NO_RETRY','children':'NOT_AVAILABLE/null','lifetime_children':'NOT_PROVEN','terminal_io':'UNKNOWN','new_acoustic_or_probe_runs':0,'projection_executed':False,'publication_ready':ref['publication_ready']}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True,type=Path);p.add_argument('--root',default=Path(__file__).resolve().parent,type=Path);p.add_argument('--require-published',action='store_true');a=p.parse_args()
 try:print(json.dumps(validate(a.root,a.data_root,a.require_published),sort_keys=True));return 0
 except (ValueError,KeyError,OSError,TypeError,AssertionError) as e:print('INVALID: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
