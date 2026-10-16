"""Saved archive integrity only; no model, optimizer or outer-process replay."""
from pathlib import Path
import hashlib,io,json,zipfile
HERE=Path(__file__).resolve().parent
FLAGS=('human_gold','independent_evaluation','production_qualified','reusable_weights_exported','shipping_qualified','training_admitted')
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
def consumed(c):
 expected=dict(schema='lr-stability-campaign-closure-v1',campaign_id='kws-lr-stability-20261016-v1',consumed=True,execution_enabled=False,retry_authorized=False,remaining_budget=0,remaining_model_forward_budget=0,budget_reusable=False,execution_status='completed',evidence_validation='PASS',scientific_validation='DESCRIPTIVE_ONLY',optimizer_calls_reserved=600,successful_optimizer_steps=600,model_forward_calls_reserved=608,model_forward_calls_returned=608,observations=8,primary_endpoint_step=600,historical_visible_trajectory_through500_matched=True,**{k:False for k in FLAGS})
 for k,v in expected.items():need(exact(c.get(k),v),'closure '+k)
def reconstruct(root,m):
 need(type(m['archive_bytes']) is int and 0<m['archive_bytes']<64*1024**2,'archive bound');need(m['chunk_size_bytes']==1048576 and len(m['parts'])==33,'part count')
 pieces=[]
 for i,p in enumerate(m['parts']):
  need(p['path']==f'raw-result.zip.part{i:03d}','part order');need(type(p['bytes']) is int and 0<p['bytes']<=1048576,'part bound');b=read(root,p['path']);need(len(b)==p['bytes'] and sha(b)==p['sha256'],'part identity');pieces.append(b)
 raw=b''.join(pieces);need(len(raw)==m['archive_bytes'] and sha(raw)==m['archive_sha256'],'ZIP identity');return raw
def payload_hash(raw,digest):need(sha(raw)==digest,'raw result hash')
def roster(z,hashes):
 expected={'lr-stability-result/'+n for n in hashes}|{'lr-stability-result-acceptance/hash-manifest.json','lr-stability-result-acceptance/summary.json'};names=z.namelist()
 need(len(hashes)==3037 and len(names)==len(set(names)) and set(names)==expected,'ZIP roster');need(sum(i.file_size for i in z.infolist())<512*1024**2,'unpacked bound')
 for i in z.infolist():need(not i.is_dir() and i.file_size<=1024**2,'member bound')
def verify(root=HERE):
 root=Path(root);result=root/'result-20261016';c=load(root,'closure.json');consumed(c)
 enabled=load(result,'executed-plan.json');closed=load(root,'plan.json');need(enabled['execution_enabled'] is True and closed['execution_enabled'] is False,'plan flags');enabled['execution_enabled']=False;need(exact(enabled,closed),'plan delta')
 for base,name,key in [(root,'plan.json','closed_plan_sha256'),(result,'executed-plan.json','enabled_plan_sha256'),(result,'result-hash-manifest.json','result_manifest_sha256')]:need(sha(read(base,name))==c[key],'closure identity')
 for base,name in [(root,'source-manifest.json'),(result,'archive-manifest.json')]:
  for path,digest in load(base,name)['files'].items():need(sha(read(base,path))==digest,'manifest '+path)
 freeze=load(result,'reproducibility/trusted-freeze.json');need(freeze['reviewed_source_commit']==c['source_commit'],'freeze commit');refs=load(result,'executed-source-references.json');need(len(refs['files'])==28 and refs['source_commit']==c['source_commit'] and refs['source_tree']==c['source_tree'],'source identity');need(len({x['path'] for x in refs['files']})==28,'source duplicates')
 need({r['path']:r['sha256'] for r in refs['files']}==freeze['reviewed_source_pins'],'source pins');need(freeze['plan_sha256']==c['enabled_plan_sha256'],'freeze plan')
 for row in refs['files']:
  need(row['source_url']=='https://github.com/jiying2007/kws-pipeline/blob/'+c['source_commit']+'/'+row['path'],'immutable source URL')
 for name,key in [('reproducibility/launch_reviewed_lr_stability.py','launcher_sha256'),('reproducibility/lr-stability-result-verification/verify_result.py','verifier_sha256')]:need(sha(read(result,name))==freeze[key],'freeze dependency')
 shared_digest=sha(read(result,'reproducibility/grouped-balance-result-verification/verify_result.py'));need(shared_digest=='cf974939751107b8e6e6c15d34bff1e5830e173a1a45cb3c34d1642681d36bcc','shared verifier pin');need(sha(read(result,'reproducibility/grouped-balance-result-verification/tolerances.json'))==freeze['tolerances_sha256']=='b701856f6bae2c93aa377cece1549a4afcab03921940dbc49ece6b3ddc01cf9b','tolerances pin')
 hashes=load(result,'result-hash-manifest.json');need(hashes['shared_verifier_sha256']==shared_digest,'manifest shared verifier');need(hashes['verifier_sha256']==freeze['verifier_sha256'],'manifest verifier');need(hashes['plan_sha256']==c['enabled_plan_sha256'] and hashes['trusted_freeze_sha256']==sha(read(result,'reproducibility/trusted-freeze.json')),'acceptance freeze')
 acceptance=load(result,'acceptance-summary.json');need(acceptance['externally_supplied_source_commit']==c['source_commit'],'acceptance source commit');need(acceptance['shared_verifier_sha256']==shared_digest,'acceptance shared verifier');need(acceptance['execution_status']=='completed' and acceptance['evidence_validation']=='PASS' and acceptance['historical_visible_trajectory_through500_matched'] is True,'acceptance')
 for key,value in [('optimizer_calls_reserved',600),('successful_step_receipts',600),('model_forward_calls_reserved',608),('model_forward_calls_returned',608),('payloads_verified',608),('observations_verified',8)]:need(exact(acceptance.get(key),value),'acceptance '+key)
 with zipfile.ZipFile(io.BytesIO(reconstruct(result,load(result,'parts-manifest.json')))) as z:
  roster(z,hashes['result_files'])
  for name,digest in hashes['result_files'].items():payload_hash(z.read('lr-stability-result/'+name),digest)
  need(z.read('lr-stability-result-acceptance/hash-manifest.json')==read(result,'result-hash-manifest.json'),'original manifest');need(z.read('lr-stability-result-acceptance/summary.json')==read(result,'acceptance-summary.json'),'original summary')
  terminal=decode(z.read('lr-stability-result/terminal.json'));need(terminal['status']=='completed' and terminal['optimizer_calls_reserved']==600 and terminal['model_forward_calls_reserved']==608,'terminal')
 return {'archive_integrity':'PASS','result_files':3037,'outer_process_reverified':False,'model_calls':0,'optimizer_calls':0}
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
