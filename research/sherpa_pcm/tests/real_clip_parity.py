"""Analysis-only ctypes test driver; candidate has no Python dependency."""
import argparse,ctypes as C,json,pathlib,wave,hashlib
parser=argparse.ArgumentParser()
for name in ["library","model-dir","keywords","data-root","reference","output","build-receipt"]:parser.add_argument("--"+name,required=True,type=pathlib.Path)
args=parser.parse_args()
R=pathlib.Path(__file__).resolve().parent
from parity_inputs import validate_inputs
validate_inputs(args,R.parent)
class Files(C.Structure):_fields_=[(x,C.c_char_p) for x in ['encoder','decoder','joiner','tokens','keywords']]
class Event(C.Structure):_fields_=[('keyword',C.c_char_p),('keyword_id',C.c_int),('available_samples',C.c_uint64),('eof_flush',C.c_int)]
CB=C.CFUNCTYPE(None,C.c_void_p,C.POINTER(Event)); lib=C.CDLL(str(args.library));P=C.c_void_p; E=C.c_char_p;N=C.c_size_t
lib.pcm_kws_create.argtypes=[C.POINTER(Files),CB,P,C.POINTER(P),E,N];lib.pcm_kws_create.restype=C.c_int
lib.pcm_kws_feed.argtypes=[P,C.POINTER(C.c_int16),N,E,N];lib.pcm_kws_feed.restype=C.c_int
for name in ['pcm_kws_reset','pcm_kws_finish']:getattr(lib,name).argtypes=[P,E,N];getattr(lib,name).restype=C.c_int
lib.pcm_kws_destroy.argtypes=[P];lib.pcm_kws_destroy.restype=None
m=args.model_dir;files=Files(*[str(p).encode() for p in [m/'encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx',m/'decoder-epoch-12-avg-2-chunk-16-left-64.onnx',m/'joiner-epoch-12-avg-2-chunk-16-left-64.onnx',m/'tokens.txt',args.keywords]])
lock=json.loads((R.parent/'dependencies.lock.json').read_text())
for asset in lock['models']:
 f=m/pathlib.Path(asset['path']).name;assert hashlib.sha256(f.read_bytes()).hexdigest()==asset['expected_sha256']
assert hashlib.sha256(args.keywords.read_bytes()).hexdigest()=='bdbeeb12c2b95b9a71f09994024c8ef86d88bdaa8146b758116d48471fd91e1a'
report=json.loads(args.reference.read_text());assert len(report['recordings'])==42
assert len({r['recording'] for r in report['recordings']})==42
clips=[]
for r in report['recordings']:
 p=args.data_root/r['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['file_sha256']
 with wave.open(str(p)) as w:
  assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
  b=w.readframes(w.getnframes())
 assert hashlib.sha256(b).hexdigest()==r['pcm_sha256'];clips.append((r,(C.c_int16*(len(b)//2)).from_buffer_copy(b)))
seen=[]
@CB
def callback(_,e):
 x=e.contents;seen.append((x.keyword.decode(),x.keyword_id,x.available_samples,bool(x.eof_flush)))
err=C.create_string_buffer(512);x=P();assert lib.pcm_kws_create(None,callback,None,C.byref(x),err,512)==1 and not x.value
assert lib.pcm_kws_feed(None,None,0,err,512)==1;assert lib.pcm_kws_reset(None,err,512)==1;lib.pcm_kws_destroy(None)
checks=[]
for mode,pattern in [('fixed320',[320]),('irregular',[1,159,321,17,640]),('wholeclip',[1000000])]:
 assert lib.pcm_kws_create(C.byref(files),callback,None,C.byref(x),err,512)==0,err.value
 try:
  actual_events=0
  assert lib.pcm_kws_feed(x,None,1,err,512)==1
  assert lib.pcm_kws_feed(x,None,0,err,512)==0
  # Partial unfinalized input must be discarded by reset.
  partial=(C.c_int16*169)();assert lib.pcm_kws_feed(x,partial,169,err,512)==0
  for row,pcm in clips:
   assert lib.pcm_kws_reset(x,err,512)==0;seen.clear();pos=0;i=0
   while pos<len(pcm):
    n=min(pattern[i%len(pattern)],len(pcm)-pos);ptr=C.cast(C.byref(pcm,pos*2),C.POINTER(C.c_int16));assert lib.pcm_kws_feed(x,ptr,n,err,512)==0,err.value;pos+=n;i+=1
   assert lib.pcm_kws_finish(x,err,512)==0,err.value
   expected=[(e['phrase'],e['keyword_id'],e['available_audio_samples'],e['eof_flush']) for e in row['events']];assert seen==expected,(mode,row['recording'],seen,expected)
   actual_events+=len(seen)
   assert lib.pcm_kws_finish(x,err,512)==2;assert lib.pcm_kws_feed(x,pcm,1,err,512)==2
  checks.append({'mode':mode,'clips':42,'events':actual_events,'exact_parity':True})
 finally:lib.pcm_kws_destroy(x);x=P()
result={'passed':True,'create_destroy_cycles':3,'cases':checks,'lifecycle':['null arguments','zero-length feed','partial reset discards','per-clip reset','partial EOF flush','double EOF rejected','feed after EOF rejected'],'same_model_decode_settings':True,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'adapter_source_sha256':hashlib.sha256((R.parent/'pcm_kws.cc').read_bytes()).hexdigest(),'header_sha256':hashlib.sha256((R.parent/'pcm_kws.h').read_bytes()).hexdigest(),'library_sha256':hashlib.sha256((args.library).read_bytes()).hexdigest(),'reference_quality_sha256':hashlib.sha256((args.reference).read_bytes()).hexdigest()}
args.output.open('x',encoding='utf-8').write(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
