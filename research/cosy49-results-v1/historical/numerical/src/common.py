"""Candidate identity/release adapter; canonical arithmetic is in unchanged sources."""
import hashlib,importlib.util,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[1];CANONICAL=ROOT/'vendor/canonical';OLD=ROOT/'vendor/historical/kws-native-a20-research-v1'
DIMS=[140,250,250]+[128,128,250,250]*4+[140,6]
LENGTHS=[799,800,4799,4800,4801,9600,9601,24001];RAGGED=[1,17,479,1601,3203,7001]
GEOMETRY_FIELDS=['call_index','available_samples','call_samples','waveform_samples','fbank_rows','splice_rows','selected_rows','is_final_short','wave_samples','feature_count','offset']
RELEASE=None;PHASE=None;OUT=None;REFERENCE=None;PAYLOAD=None;PAYLOAD_SHA=None;CONTRACT_SHA=None

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def read(p):
 p=Path(p);require(p.stat().st_size<=20*1024**2,'bounded JSON input')
 def unique(pairs):
  d={}
  for k,v in pairs:require(k not in d,'duplicate JSON key');d[k]=v
  return d
 obj=json.loads(p.read_text(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
 def finite(v):
  if type(v)is float:require(math.isfinite(v),'finite JSON number')
  elif type(v)is list:
   for x in v:finite(x)
  elif type(v)is dict:
   for x in v.values():finite(x)
 finite(obj);return obj
read_json=read

def write(p,value,compact=False):
 with Path(p).open('x')as f:json.dump(value,f,sort_keys=True,allow_nan=False,ensure_ascii=False,**({'separators':(',',':')}if compact else{'indent':2}));f.write('\n');f.flush();os.fsync(f.fileno())
def write_compact(p,v):write(p,v,True)
def write_json(p,v,*unused):write(p,v)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def entry(e):
 p=Path(e['path']);p=p if p.is_absolute()else WORKSPACE/p;require(p.is_file()and not p.is_symlink()and p.stat().st_size==e['bytes']and sha(p)==e['sha256'],'bound file identity');return p

def source_identity():
 f=read(ROOT/'SOURCE-FREEZE.json')
 for e in f['files']:
  p=ROOT/e['path'];require(p.resolve().is_relative_to(ROOT.resolve())and p.is_file()and sha(p)==e['sha256']and p.stat().st_size==e['bytes'],'numerical frozen source')
 return sha(ROOT/'SOURCE-FREEZE.json')

def identities(release):
 require(sys.flags.optimize==0 and os.environ.get('PYTHONOPTIMIZE','')in ('','0'),'numerical assertions required')
 require(type(release)is dict and release.get('approved')is True and release['schema']=='a20-candidate-numerical-release-v1','explicit numerical release')
 require(release['source_freeze_sha256']==source_identity()and release['protocol_sha256']==sha(ROOT/'PROTOCOL.json'),'numerical release identity')
 require(release['independent_source_review']['status']=='PASS'and release['independent_source_review']['source_freeze_sha256']==release['source_freeze_sha256'],'independent source review')
 require(release['private_source_readback']['status']=='PASS'and release['private_source_readback']['source_freeze_sha256']==release['source_freeze_sha256'],'source persistence barrier')
 require(release['training_saved_output_audit']['status']=='PASS'and release['training_saved_output_audit']['terminal_state_sha256']==release['terminal_state_sha256'],'terminal training audit barrier')
 for name in ('native_bundle','export_receipt'):entry(release[name])
 require(release['native_build_saved_output_audit']['status']=='PASS'and release['native_build_saved_output_audit']['bundle_sha256']==release['native_bundle']['sha256'],'independent native build audit barrier')
 require(sha(release['release_path'])==release['release_file_sha256'],'unchanged exact release bytes')
 return True

def bundle():
 b=read(entry(RELEASE['native_bundle']))
 require(b['variant']=='candidate'and b['state_sha256']==RELEASE['terminal_state_sha256'],'candidate bundle identity')
 for name in ('payload','manifest','identity_header','library','resources','diagnostic_library','ctypes_api'):b[name]['path']=str(entry(b[name]))
 return b

def paths():
 b=bundle();return dict(candidate_library=entry(b['library']),candidate_ctypes=entry(b['ctypes_api']),candidate_resources=entry(b['resources']),diagnostic_library=entry(b['diagnostic_library']))

def reference_prerequisite():
 r=RELEASE['reference_saved_output_audit'];require(r['status']=='PASS','reference independent persistence audit')
 p=REFERENCE/'reference-freeze.json';require(sha(p)==r['reference_freeze_sha256'],'reference freeze identity')
 report=read(REFERENCE/'report.json');require(report['passed']is True and report['terminal_state_sha256']==RELEASE['terminal_state_sha256'],'same-weight reference report')
 require(sha(REFERENCE/'report.json')==r['report_sha256']and r['private_readback_status']=='PASS','reference report/readback barrier')
 for name,h in read(p)['files_sha256'].items():require(sha(REFERENCE/name)==h,'persisted reference file')
 return dict(reference_freeze_sha256=sha(p),reference_report_sha256=sha(REFERENCE/'report.json'))

def allocation_check():
 caps=read(ROOT/'PROTOCOL.json')['phase_budgets'][PHASE];files=[p for p in OUT.rglob('*')if p.is_file()]
 require(all(not p.is_symlink()and p.stat().st_size<=20*1024**2 for p in files),'numerical per-file cap')
 require(sum(p.stat().st_size for p in files)<=caps['output_bytes'],'numerical phase output cap')

def configure(release_path,phase):
 global RELEASE,PHASE,OUT,REFERENCE,PAYLOAD,PAYLOAD_SHA,CONTRACT_SHA
 r=read(release_path);r['release_path']=str(Path(release_path).resolve());r['release_file_sha256']=sha(release_path)
 require(r['phase']==phase and phase in ('reference','native'),'one explicit phase only')
 identities(r);RELEASE=r;PHASE=phase;OUT=ROOT/phase/'work/artifact';REFERENCE=Path(r['reference_directory'])if phase=='native'else OUT;CONTRACT_SHA=sha(ROOT/'PROTOCOL.json')
 b=bundle();PAYLOAD=entry(b['payload']);PAYLOAD_SHA=b['payload']['sha256']
 require(sys.version_info[:3]==(3,12,14)and sys.byteorder=='little','original canonical Python runtime')
 import importlib.metadata
 require(importlib.metadata.version('numpy')=='2.3.5','original canonical NumPy runtime')
 inventory=read(ROOT/'metadata/historical-runtime-inventory.json')
 for n,h in inventory['runtime_bindings'].items():require(Path(n).is_file()and sha(n)==h,'unchanged canonical runtime dependency')
 if phase=='native':reference_prerequisite()
 return r

def load_inputs(np):
 manifest=read(ROOT/'metadata/fixture-manifest.json');cases=manifest['cases'];p=ROOT/'metadata/fixture-inputs.npz'
 require(sha(p)==manifest['inputs_npz_sha256']and len(cases)==8,'existing8 fixture archive')
 with np.load(p,allow_pickle=False)as f:inputs={c['id']:f[c['npz_key']].copy()for c in cases}
 for c in cases:
  a=inputs[c['id']];require(a.dtype==np.dtype('<i2')and list(a.shape)==c['shape']and sha_bytes(a.tobytes())==c['pcm_sha256'],'fixture identity')
 return cases,inputs
# Exact reused geometry functions import this module only after its constants exist.
from geometry_reuse import independent_geometry,verify_geometry,partitions,save_arrays
