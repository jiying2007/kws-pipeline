#!/usr/bin/env python3
"""Pure synthetic differential test. No audio, acoustic model, old once runner."""
import ctypes as C, json, pathlib, subprocess, hashlib, random, argparse, sys
from native_elf import require_native, native_expected
from compiler_flags import CFLAGS, STRICT_FP_FLAGS
ROOT=pathlib.Path(__file__).resolve().parent
class Result(C.Structure):
 _fields_=[('valid',C.c_int32),('state',C.c_int32),('keyword',C.c_int32),('start_frame',C.c_int64),('end_frame',C.c_int64),('score',C.c_double),('rows_decoded',C.c_size_t)]
class Node(C.Structure):
 _fields_=[('frame',C.c_int64),('prob',C.c_double),('token',C.c_int32)]
class Hyp(C.Structure):
 _fields_=[('pb',C.c_double),('pnb',C.c_double),('len',C.c_uint16),('node',C.c_uint16*128)]
class State(C.Structure):
 _fields_=[('magic',C.c_uint32),('hyp_count',C.c_uint16),('node_count',C.c_uint16),('total_frames',C.c_int64),('last_active_pos',C.c_int64),('hit_score',C.c_double),('result',Result),('hyps',Hyp*20),('nodes',Node*2680)]
def tup(r):return tuple(getattr(r,k) for k,_ in r._fields_)
def snapshot(d):
 return (d.magic,d.hyp_count,d.node_count,d.total_frames,d.last_active_pos,d.hit_score,tup(d.result),[(h.pb,h.pnb,h.len,list(h.node[:h.len])) for h in d.hyps[:d.hyp_count]], [tup(n) for n in d.nodes[:d.node_count]])
def main():
 if sys.flags.optimize:raise RuntimeError("optimized Python disables test assertions; refused")
 ap=argparse.ArgumentParser();ap.add_argument("--cc",default="cc");ap.add_argument("--original",type=pathlib.Path,required=True);ap.add_argument("--optimized",type=pathlib.Path,required=True);ap.add_argument("--output",type=pathlib.Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 libs=[];states=[];spaces=[];sizes=[];commands=[];binaries={}
 def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
 def compile_library(source,target):
  command=[args.cc,*CFLAGS,'-shared',str(source),'-lm','-o',str(target)]
  p=subprocess.run(command,capture_output=True,text=True,timeout=120)
  commands.append({'command':command,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
  p.check_returncode()
 for name in ('original','optimized'):
  target=args.output/(name+'-decoder.so');source=getattr(args,name)/'a20_decoder.c';compile_library(source,target)
  require_native(target,(3,))
  binaries[name]={'library_sha256':sha(target),'source_sha256':sha(source),'header_sha256':sha(source.with_suffix('.h'))}
  lib=C.CDLL(str(target));lib.a20d_decoder_bytes.restype=C.c_size_t;lib.a20d_workspace_bytes.restype=C.c_size_t
  sizes.append({'decoder':lib.a20d_decoder_bytes(),'workspace':lib.a20d_workspace_bytes()});assert lib.a20d_decoder_bytes()==C.sizeof(State)
  libs.append(lib);states.append(State());spaces.append(C.create_string_buffer(lib.a20d_workspace_bytes()));lib.a20d_init(C.byref(states[-1]))
 calls=0;activations=0;statuses={};rng=random.Random(20261009)
 def run(rows,logits=False,expected=None):
  nonlocal calls,activations
  n=len(rows);arr=(C.c_float*(n*6))(*[x for r in rows for x in r]);rs=[];codes=[];before=[bytes(d) for d in states]
  for lib,d,w in zip(libs,states,spaces):
   r=Result();r.valid=42
   codes.append(getattr(lib,'a20d_process_logits' if logits else 'a20d_process_probs')(C.byref(d),w,arr,n,C.byref(r)));rs.append(r)
  assert codes[0]==codes[1],codes
  if expected is not None:assert codes[0]==expected,(codes,expected)
  assert tup(rs[0])==tup(rs[1]),(tup(rs[0]),tup(rs[1]))
  assert snapshot(states[0])==snapshot(states[1]),'semantic state mismatch'
  if codes[0]:
   assert all(bytes(d)==b for d,b in zip(states,before));assert rs[0].valid==42
  calls+=1;activations+=rs[0].state;statuses[str(codes[0])]=statuses.get(str(codes[0]),0)+1
  return rs[0]
 def reset():
  for lib,d in zip(libs,states):lib.a20d_reset_all(C.byref(d))
 def one(i):return [float(k==i) for k in range(6)]
 run([],expected=0);run([[0,0,0,0,0,float('nan')]],expected=1)
 run([[0,0,0,0,0,float('inf')]],True,1);run([[.2]*6],expected=1)
 reset();r=run([one(i) for i in [1,2,3,4,5,5]],expected=0);assert r.state and r.rows_decoded==4 and states[0].total_frames==18
 reset();run([one(1)]);run([one(1)]);assert states[0].hyps[0].len==1
 run([one(0)]);run([one(1)]);assert states[0].hyps[0].len==2
 # Full input is checked even if activation would have skipped the tail.
 reset();run([one(i) for i in [1,2,3,4]]+[[float('nan')]*6],expected=1)
 # Capacity error after actual processing starts must preserve original state/result.
 reset()
 for d in states:
  d.hyp_count=1;d.node_count=128;h=d.hyps[0];h.len=128;h.pb=0;h.pnb=1
  for i in range(128):h.node[i]=i;d.nodes[i]=Node(i*3,.8,1+i%2)
 run([one(5)],expected=2)
 reset()
 for d in states:d.total_frames=2**63-2
 run([one(0)],expected=3)
 reset()
 # Deterministic repeated/tied distributions, broad random logits/probs, chunk sizes.
 rows_tested=0
 for case in range(1600):
  if case%20==0:reset()
  n=rng.choice([1,2,3,7,19,33]);logits=case%3==0;rows=[]
  for j in range(n):
   if case%11==0:r=[1/6]*6;logits=False
   elif logits:r=[rng.uniform(-20,20) for _ in range(6)]
   elif case%7==0:r=one(rng.randrange(6))
   else:
    r=[rng.random() for _ in range(6)];s=sum(r);r=[x/s for x in r]
   rows.append(r)
  run(rows,logits);rows_tested+=n
 compiler=subprocess.run([args.cc,'--version'],capture_output=True,text=True,check=True,timeout=120).stdout.splitlines()[0]
 report={'schema':'decoder-scratch-v1-differential','compiler':compiler,'compile_flags':list(CFLAGS),'strict_fp_flags':list(STRICT_FP_FLAGS),'compiler_flags_sha256':sha(ROOT/'compiler_flags.py'),'commands':commands,'binaries':binaries,'pass':True,'compiler_command':args.cc,'native_elf':native_expected(),'synthetic_calls':calls,'random_rows':rows_tested,'activations':activations,'status_counts':statuses,'sizes':dict(zip(('original','optimized'),sizes)),'workspace_saved_bytes':sizes[0]['workspace']-sizes[1]['workspace'],'beam_unchanged':True,'transaction_copy_preserved':True,'scope':'host ABI synthetic decoder only; no model/audio evaluation or SSC305 measurement'}
 (args.output/'decoder-test-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
