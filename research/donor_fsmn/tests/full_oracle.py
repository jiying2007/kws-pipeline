"""Fixed all-layer synthetic oracle/propagation diagnosis; no old-gate waiver."""
import ctypes as C, hashlib, json, math, pathlib, resource, subprocess, sys, shutil, traceback
from fractions import Fraction
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
OLD=pathlib.Path('/workspace/shared/kws-fsmn-ab-evidence')
OUT=pathlib.Path('/workspace/shared/kws-fsmn-full-oracle-v1')
PLAN=pathlib.Path('/workspace/shared/kws-fsmn-native-spec/LOCAL_ORACLE_PROPAGATION_PROPOSAL.md')
F=C.POINTER(C.c_float)
class Model(C.Structure):_fields_=[('weights',F),('cache',C.c_float*5632),('fault',C.c_int)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 if p.exists():raise ValueError('refuse overwrite '+str(p))
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def finite(a):
 if not np.isfinite(a).all():raise ValueError('nonfinite data or bound')
 return a
# Nonnegative outward-rounded bound operations; zero stays exact.
def up(a):
 a=np.asarray(a,dtype=np.float64);finite(a)
 if np.any(a<0):raise ValueError('negative bound operand')
 return finite(np.where(a==0,0,np.nextafter(a,np.inf)))
def add(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 finite(a);finite(b)
 if np.any(a<0) or np.any(b<0):raise ValueError('negative bound operand')
 return up(a+b)
def mul(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 finite(a);finite(b)
 if np.any(a<0) or np.any(b<0):raise ValueError('negative bound operand')
 product=a*b;finite(product)
 return finite(np.where((a==0)|(b==0),0,np.nextafter(product,np.inf)))
def gamma(m,bits):
 exact=Fraction(m,2**bits-m);v=float(exact)
 if Fraction.from_float(v)<exact:v=math.nextafter(v,math.inf)
 return v
ETA=2.0**-150;ETA64=float.fromhex('0x0.0000000000001p-1022')
def round_bound(S,m,bits=24):
 eta=ETA if bits==24 else ETA64
 return add(mul(gamma(m,bits),S),mul(mul(m,eta),add(1,gamma(m,bits))))
def absdiff(a,b):return up(np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)))
def ptr(a):return a.ctypes.data_as(F)
def dot_abs(X,W,b=None):
 r=np.zeros((len(X),len(W)),np.float64)
 for j in range(X.shape[1]):r=add(r,mul(X[:,j,None],np.abs(W[:,j])[None,:]))
 if b is not None:r=add(r,np.abs(b)[None,:])
 return r

def affine_oracle(X,W,b):
 out=np.empty((len(X),len(W)),np.float64)
 for i in range(len(X)):
  for o in range(len(W)):
   terms=[float(a)*float(w) for a,w in zip(X[i],W[o])]
   if b is not None:terms.append(float(b[o]))
   out[i,o]=math.fsum(terms)
 return finite(out)
def shifted(X,lag):
 y=np.zeros_like(X)
 if lag==0:y[:]=X
 elif lag<len(X):y[lag:]=X[:-lag]
 return y

def memory_terms(P,L,R):
 return [shifted(P,2)]+[shifted(P,11-k)*L[:,k] for k in range(10)]+[shifted(P,1)*R[:,0],P*R[:,1]]
def memory_oracle(P,L,R):
 terms=memory_terms(P.astype(np.float64),L.astype(np.float64),R.astype(np.float64))
 out=np.empty_like(P,dtype=np.float64)
 for t in range(len(P)):
  for c in range(128):out[t,c]=math.fsum(float(a[t,c]) for a in terms)
 return finite(out)
def memory_abs(P,L,R):
 # Multiplication and every accumulation rounded outward, not plain numpy.sum.
 r=shifted(P,2).copy()
 for k in range(10):r=add(r,mul(shifted(P,11-k),np.abs(L[:,k])))
 r=add(r,mul(shifted(P,1),np.abs(R[:,0])));return add(r,mul(P,np.abs(R[:,1])))

def identities():
 cc=pathlib.Path(shutil.which('cc')).resolve()
 compiler=dict(path=str(cc),sha256=sha(cc),version=subprocess.check_output([str(cc),'--version'],text=True))
 return dict(compiler_identity=compiler,schema=1,plan_sha256=sha(PLAN),script_sha256=sha(pathlib.Path(__file__)),kernel_sha256=sha(ROOT/'fsmn.c'),header_sha256=sha(ROOT/'fsmn.h'),hooks_sha256=sha(ROOT/'tests/oracle_hooks.c'),reference_sha256=sha(OLD/'reference-receipt.json'),goldens_sha256=sha(OLD/'goldens.npz'),payload_sha256=sha(OLD/'donor-local.f32'),manifest_sha256=sha(OLD/'weights-manifest.json'),python=sys.version,numpy=np.__version__,scope='one fixed diagnostic; three frozen synthetic inputs, six routes, 12 original calls; no speech/training/reduction changes',gamma='exact Fraction(m,2**bits-m), converted upward',bound_arithmetic='nonnegative binary64 multiply/add rounded upward after each operation; finite required',fp32_underflow='m*2^-150*(1+gamma_m)',oracle_underflow='conservative m*min-positive-binary64*(1+gamma_m)',old_B=dict(atol=1e-4,rtol=1e-5),compile_flags=['-std=c11','-O2','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror'])

def execute():
 pre=json.loads((OUT/'preflight.json').read_text());assert pre==identities()
 for f in ['result.json','arrays.npz','oracle.so']:
  if (OUT/f).exists():raise ValueError('existing output '+f)
 resource.setrlimit(resource.RLIMIT_CPU,(600,600))
 cmd=[pre['compiler_identity']['path'],*pre['compile_flags'],'-shared','-fPIC',str(ROOT/'tests/oracle_hooks.c'),'-o',str(OUT/'oracle.so')];subprocess.run(cmd,check=True)
 lib=C.CDLL(str(OUT/'oracle.so'));lib.df_init.argtypes=[C.POINTER(Model),F,C.c_size_t];lib.df_step.argtypes=[C.POINTER(Model),F,F,F];lib.df_cmvn.argtypes=[C.POINTER(Model),F,F];lib.test_affine.argtypes=[F,F,F,F,C.c_int,C.c_int];lib.test_memory.argtypes=[C.POINTER(Model),C.c_int,F,F];lib.test_relu.argtypes=[F,C.c_int]
 weights=np.fromfile(OLD/'donor-local.f32',dtype='<f4');finite(weights)
 g=np.load(OLD/'goldens.npz',allow_pickle=False);receipt=json.loads((OLD/'reference-receipt.json').read_text());assert receipt['goldens_sha256']==sha(OLD/'goldens.npz')
 records=[];arrays={};cache_checks=[]
 def check(key,local,ref,oracle,bound,margin,E,trajectory):
  finite(local);finite(ref);finite(oracle);finite(bound);finite(margin);finite(E);finite(trajectory)
  ce=absdiff(local,oracle);te=absdiff(ref,oracle);allowed=add(bound,margin);observed=absdiff(trajectory,ref)
  # ReLU/copied exact zeros have zero limits; ratios remain finite by reporting separately.
  ratio=np.divide(ce,allowed,out=np.zeros_like(ce),where=allowed>0)
  rec=dict(name=key,local_C_max_abs=float(ce.max()),local_Torch_max_abs=float(te.max()),local_C_bound_failures=int((ce>allowed).sum()),local_Torch_bound_failures=int((te>allowed).sum()),local_error_bound_ratio_max=float(ratio.max()),trajectory_max_abs=float(observed.max()),envelope_max=float(E.max()),envelope_exceedances=int((observed>E).sum()),old_B_failures=int((observed>1e-4+1e-5*np.abs(ref)).sum()))
  records.append(rec)
  for name,a in [('C_local',local),('reference',ref),('oracle',oracle),('local_bound',bound),('oracle_margin',margin),('envelope',E),('C_trajectory',trajectory)]:arrays[key+'_'+name]=a
 dims=[140,250,250]+[128,128,250,250]*4+[140,2599]
 for name in ['zero','impulse','random']:
  X=g[name+'_input']
  for route in ['whole','split']:
   rows=[r for r in receipt['cases'] if r['input']==name and f'_{route}_' in r['prefix']];prefix=name+'_'+route
   refs=[np.concatenate([g[r['prefix']+f'_stage{i}'][0] for r in rows]) for i in range(21)]
   normref=np.concatenate([g[r['prefix']+'_cmvn'][0] for r in rows]);m=Model();assert lib.df_init(C.byref(m),ptr(weights),len(weights))==0
   traj=np.empty((len(X),21,2599),np.float32);norm=np.empty_like(X);ccaches={};t=0
   for row in rows:
    for _ in range(row['frames']):
     out=np.empty(2599,np.float32);assert lib.df_cmvn(C.byref(m),ptr(X[t]),ptr(norm[t]))==0;assert lib.df_step(C.byref(m),ptr(X[t]),ptr(out),ptr(traj[t]))==0;t+=1
    ccaches[t]=np.ctypeslib.as_array(m.cache).reshape(1,128,11,4).copy()
   mean=weights[:400].astype(float);scale=weights[400:800].astype(float);s=np.abs(scale)
   oracle=(X.astype(float)-mean)*scale;A=add(np.abs(X.astype(float)),np.abs(mean));bound=add(mul(gamma(2,24),mul(A,s)),mul(ETA,add(mul(add(1,2**-24),s),1)))
   margin=add(mul(gamma(2,53),mul(A,s)),mul(ETA64,add(mul(add(1,2**-53),s),1)))
   E=add(bound,add(absdiff(normref,oracle),margin));check(prefix+'_cmvn',norm,normref,oracle,bound,margin,E,norm)
   prev=normref;env=E;projected_env={};projected_ref={}
   for stage in range(21):
    ref=refs[stage];trajectory=traj[:,stage,:dims[stage]].copy();key=prefix+f'_stage{stage}'
    is_relu=stage==2 or stage in [6,10,14,18]
    is_mem=stage in [4,8,12,16]
    if is_relu:
     local=prev.copy()
     for a in local:lib.test_relu(ptr(a),len(a))
     oracle=np.maximum(prev.astype(float),0);bound=np.zeros_like(oracle);margin=np.zeros_like(oracle);E=env.copy()
    elif is_mem:
     layer=(stage-4)//4;off=92190+layer*65786;L=weights[off+32000:off+33280].reshape(128,10);R=weights[off+33280:off+33536].reshape(128,2)
     mm=Model();assert lib.df_init(C.byref(mm),ptr(weights),len(weights))==0;local=np.empty_like(prev)
     for ti,(a,b) in enumerate(zip(prev,local)):
      lib.test_memory(C.byref(mm),layer,ptr(a),ptr(b))
      exact=np.zeros((128,11),np.float32);past=prev[max(0,ti-10):ti+1];exact[:,-len(past):]=past.T
      assert np.array_equal(np.ctypeslib.as_array(mm.cache).reshape(128,11,4)[:,:,layer],exact)
     oracle=memory_oracle(prev,L,R);S=memory_abs(np.abs(prev.astype(float)),L,R);bound=round_bound(S,27);margin=round_bound(S,14,53)
     propagated=memory_abs(env,L,R);Cscale=memory_abs(add(np.abs(prev.astype(float)),env),L,R)
     E=add(propagated,add(round_bound(Cscale,27),add(absdiff(ref,oracle),margin)))
     projected_env[layer]=env.copy();projected_ref[layer]=prev.copy()
    else:
     if stage==0:off,ni,no,bias=800,400,140,56800
     elif stage==1:off,ni,no,bias=56940,140,250,91940
     elif stage==19:off,ni,no,bias=355334,250,140,390334
     elif stage==20:off,ni,no,bias=390474,140,2599,754334
     else:
      layer=(stage-3)//4;base=92190+layer*65786
      if (stage-3)%4==0:off,ni,no,bias=base,250,128,None
      else:off,ni,no,bias=base+33536,128,250,base+65536
     W=weights[off:off+ni*no].reshape(no,ni);B=None if bias is None else weights[bias:bias+no];local=np.empty((len(X),no),np.float32)
     for a,b in zip(prev,local):lib.test_affine(ptr(a),ptr(b),ptr(W),None if B is None else ptr(B),ni,no)
     oracle=affine_oracle(prev,W,B);S=dot_abs(np.abs(prev.astype(float)),W,B);bound=round_bound(S,2*ni+1);margin=round_bound(S,ni+1,53)
     propagated=dot_abs(env,W);Cscale=dot_abs(add(np.abs(prev.astype(float)),env),W,B)
     E=add(propagated,add(round_bound(Cscale,2*ni+1),add(absdiff(ref,oracle),margin)))
    check(key,local,ref,oracle,bound,margin,E,trajectory);prev=ref;env=E
   # Validate all original call-end cache contents and propagated cache envelope.
   end=0
   for row in rows:
    end+=row['frames'];refcache=g[row['prefix']+'_cache'];expected=np.zeros_like(refcache);ce=np.zeros(refcache.shape,np.float64)
    for layer in range(4):
     start=max(0,end-11);n=end-start;expected[0,:,11-n:,layer]=projected_ref[layer][start:end].T;ce[0,:,11-n:,layer]=projected_env[layer][start:end].T
    assert np.array_equal(expected,refcache)
    obs=absdiff(ccaches[end],refcache);finite(ce)
    cache_checks.append(dict(name=row['prefix'],source_copy_exact=True,envelope_exceedances=int((obs>ce).sum()),old_B_failures=int((obs>1e-4+1e-5*abs(refcache)).sum()),max_abs=float(obs.max()),envelope_max=float(ce.max())))
    arrays[row['prefix']+'_C_cache']=ccaches[end];arrays[row['prefix']+'_cache_envelope']=ce
 result=dict(preflight_sha256=sha(OUT/'preflight.json'),records=records,cache_checks=cache_checks,mathematical_consistency_passed=all(not r['local_C_bound_failures'] and not r['local_Torch_bound_failures'] and not r['envelope_exceedances'] for r in records) and all(not r['envelope_exceedances'] for r in cache_checks),old_B_still_failed=True,old_B_actual_failing_records=sum(r['old_B_failures']>0 for r in records)+sum(r['old_B_failures']>0 for r in cache_checks),stage_C_executed=False,scope='fixed synthetic diagnosis; loose bounds do not establish fidelity',compiler_command=cmd,compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0])
 np.savez(OUT/'arrays.npz',**arrays);result['arrays_sha256']=sha(OUT/'arrays.npz');save(OUT/'result.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['records','cache_checks','compiler_command']},indent=2))
if __name__=='__main__':
 if sys.argv[1]=='prepare':OUT.mkdir(exist_ok=True);save(OUT/'preflight.json',identities());print('FROZEN',sha(OUT/'preflight.json'))
 elif sys.argv[1]=='run':
  try:execute()
  except Exception as exc:
   failure=dict(status='FAILED',exception=repr(exc),traceback=traceback.format_exc(),preflight_sha256=sha(OUT/'preflight.json'),old_B_still_failed=True,stage_C_executed=False)
   if not (OUT/'failure.json').exists():save(OUT/'failure.json',failure)
   raise
 else:raise SystemExit('prepare or run')
