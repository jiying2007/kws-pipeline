"""Independent restricted storage reader/exporter: zero Torch/model calls.

Only an audited, hash-pinned Torch ZIP is accepted. Pickle tensor reconstruction
is replaced by inert storage descriptors; all other globals are denied.
"""
import argparse,collections,hashlib,io,json,math,pickle,pickletools,struct,sys,zipfile,resource,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SYMBOLS=['<blank>','你','好','小','窝','屋']
CMVN=('global_cmvn.mean','global_cmvn.istd')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
class Storage:
 def __init__(self,dtype,key,count):self.dtype=dtype;self.key=key;self.count=count
class Tensor:
 def __init__(self,storage,offset,shape,stride,requires_grad,hooks,metadata=None):
  require(type(storage)is Storage and type(offset)is int and offset>=0,'tensor storage descriptor')
  require(type(shape)in (tuple,list)and type(stride)in (tuple,list)and len(shape)==len(stride)<=5,'tensor dimensions')
  require(all(type(x)is int and 0<=x<=1000000 for x in shape+stride),'tensor shape values')
  expected=[];v=1
  for n in reversed(shape):expected.append(v);v*=n
  require(tuple(stride)==tuple(reversed(expected)),'contiguous serialized tensors only')
  require(offset+v<=storage.count,'tensor storage extent')
  self.storage=storage;self.offset=offset;self.shape=list(shape);self.count=v
class Restricted(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
  if (module,name)==('torch._utils','_rebuild_tensor_v2'):return Tensor
  if module=='torch'and name in ('FloatStorage','ByteStorage'):return name
  raise ValueError('unapproved pickle global')
 def persistent_load(self,pid):
  require(type(pid)is tuple and len(pid)==5 and pid[0]=='storage','persistent storage schema')
  _,dtype,key,location,count=pid
  require(dtype in ('FloatStorage','ByteStorage')and type(key)is str and key.isdigit()and location=='cpu'and type(count)is int and 0<=count<=1000000,'storage descriptor')
  return Storage(dtype,key,count)
def load_zip(path,expected_sha):
 require(Path(path).stat().st_size<=6*1024**2 and sha(path)==expected_sha,'audited checkpoint identity')
 with zipfile.ZipFile(path)as z:
  names=z.namelist();require(len(names)==len(set(names))and len(names)<=160,'ZIP entry count')
  roots={n.split('/')[0]for n in names};require(len(roots)==1,'ZIP single root');prefix=next(iter(roots))+'/'
  require(all(n.startswith(prefix)and '..'not in Path(n).parts and not Path(n).is_absolute()for n in names),'ZIP closed paths')
  require(sum(i.file_size for i in z.infolist())<=6*1024**2 and all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist()),'ZIP stored bounded payload')
  require(z.read(prefix+'byteorder')==b'little','little endian storage')
  metadata=z.read(prefix+'data.pkl');require(len(metadata)<=1024**2,'pickle metadata cap')
  require(not any(op.name in {'EXT1','EXT2','EXT4','NEWOBJ','NEWOBJ_EX','INST','OBJ'}for op,arg,pos in pickletools.genops(metadata)),'inert pickle opcodes only')
  stream=io.BytesIO(metadata);obj=Restricted(stream).load();require(stream.tell()==len(metadata),'single pickle metadata object');storages={}
  for n in names:
   if n.startswith(prefix+'data/'):
    key=n[len(prefix+'data/'):];require(key.isdigit(),'storage filename');storages[key]=z.read(n)
  return obj,storages

def tensor_bytes(t,storages,dtype=None,shape=None):
 require(type(t)is Tensor,'inert tensor descriptor')
 if dtype is not None:require(t.storage.dtype==dtype,'storage dtype')
 if shape is not None:require(t.shape==shape,'tensor shape')
 width=4 if t.storage.dtype=='FloatStorage'else 1;raw=storages[t.storage.key]
 require(len(raw)==t.storage.count*width,'storage exact bytes');raw=raw[t.offset*width:(t.offset+t.count)*width]
 if width==4:require(all(math.isfinite(v[0])for v in struct.iter_unpack('<f',raw)),'finite float storage')
 return raw

def extract_model(obj,storages,schema):
 require(type(obj)is dict and type(obj['model'])in (dict,collections.OrderedDict),'checkpoint mapping')
 state=obj['model'];require(list(state)==[e['name']for e in schema['tensors']],'ordered30 model tensors')
 raw=bytearray();canonical=hashlib.sha256();entries=[]
 for e in schema['tensors']:
  b=tensor_bytes(state[e['name']],storages,'FloatStorage',e['shape']);h=hashlib.sha256(b).hexdigest();require(len(b)==e['bytes'],'tensor byte size')
  if e['name']in CMVN:require(h==e['sha256'],'immutable CMVN')
  if e['name']=='global_cmvn.istd':require(all(x[0]>0 for x in struct.iter_unpack('<f',b)),'positive CMVN inverse std')
  entries.append(dict(name=e['name'],shape=e['shape'],offset=len(raw),bytes=len(b),sha256=h));raw.extend(b);canonical.update(e['name'].encode());canonical.update(b)
 require(len(raw)==1565280,'native payload size');return bytes(raw),canonical.hexdigest(),entries

def enforce_export_build_limits(cpu=60,wall=120):
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 limit=math.ceil(time.process_time()+cpu)+1;resource.setrlimit(resource.RLIMIT_CPU,(limit,limit))
 resource.setrlimit(resource.RLIMIT_FSIZE,(20*1024**2,)*2);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 def deadline(*unused):raise RuntimeError('EXPORT_BUILD_WALL_LIMIT')
 signal.signal(signal.SIGALRM,deadline);signal.alarm(wall)
 return time.monotonic(),time.process_time()+resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime

def verify_prepared_source(r):
 f=read(ROOT/'SOURCE-FREEZE.json')
 require(r['source_freeze_sha256']==sha(ROOT/'SOURCE-FREEZE.json')and r['protocol_sha256']==sha(ROOT/'PROTOCOL.json'),'reviewed exporter source and protocol')
 require(r['independent_source_review']['status']=='PASS'and r['independent_source_review']['source_freeze_sha256']==r['source_freeze_sha256'],'reviewed source barrier')
 for e in f['files']:
  p=ROOT/e['path'];require(p.resolve().is_relative_to(ROOT.resolve())and p.is_file()and p.stat().st_size==e['bytes']and sha(p)==e['sha256'],'complete exporter source freeze')

def export(release_path):
 r=read(release_path);require(r.get('schema')=='a20-candidate-export-release-v1'and r.get('approved')is True,'export explicit release')
 verify_prepared_source(r)
 enforce_export_build_limits(30,60)
 require(r['training_saved_output_audit']['status']=='PASS'and r['training_saved_output_audit']['checkpoint_sha256']==r['checkpoint']['sha256']and r['training_saved_output_audit']['state_sha256']==r['state_sha256'],'training terminal audit barrier')
 for e in r['source_bindings']:require(sha(ROOT/e['path'])==e['sha256'],'export source freeze')
 cp=Path(r['checkpoint']['path']);obj,storages=load_zip(cp,r['checkpoint']['sha256'])
 require(obj['schema']=='a20-cosy49-checkpoint-v1'and obj['step']==300 and obj['seed']==610104 and obj['symbols']==SYMBOLS and obj['resume_allowed']is False,'terminal candidate envelope')
 schema=read(ROOT/'metadata/A20-TENSOR-SCHEMA.json');payload,state,entries=extract_model(obj,storages,schema)
 require(state==r['state_sha256']==obj['terminal_state_sha256'],'audited terminal state hash')
 require(hashlib.sha256(payload).hexdigest()==r['payload_sha256'],'audited payload hash')
 require(Path(r['training_payload']['path']).read_bytes()==payload and sha(r['training_payload']['path'])==r['training_payload']['sha256'],'independent checkpoint to saved payload equality')
 out=Path(r['output_directory']);require(out.resolve().is_relative_to(ROOT.resolve()),'export output scope');out.mkdir(exist_ok=False)
 (out/'a20.f32').write_bytes(payload);manifest=dict(schema='a20-native-fp32-v1',checkpoint_sha256=r['checkpoint']['sha256'],state_sha256=state,payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),tensors=entries,symbols=SYMBOLS,forward_calls=0,checkpoint_parser='restricted inert descriptors; no Torch or arbitrary pickle global')
 write(out/'manifest.json',manifest);(out/'a20_identity.h').write_text('#define A20_EXPECTED_SHA "'+manifest['payload_sha256']+'"\n')
 require(sum(p.stat().st_size for p in out.iterdir()if p.is_file())<=2*1024**2,'export total output cap')
 write(out/'export-receipt.json',dict(status='PASS_EXPORT_ONLY',model_forwards=0,checkpoint_sha256=r['checkpoint']['sha256'],state_sha256=state,payload_sha256=manifest['payload_sha256'],source_release_sha256=sha(release_path),files=[dict(path=n,bytes=(out/n).stat().st_size,sha256=sha(out/n))for n in ('a20.f32','manifest.json','a20_identity.h')],native_parity='NOT_RUN'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path);a=p.parse_args()
 if a.release is None:raise SystemExit('PREPARATION_ONLY: no exact candidate export release')
 export(a.release)
