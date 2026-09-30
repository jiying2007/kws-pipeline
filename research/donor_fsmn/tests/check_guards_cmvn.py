import ctypes as C,json,pathlib,sys,hashlib
import numpy as np
E=pathlib.Path(sys.argv[1]);f=C.POINTER(C.c_float)
class Model(C.Structure):_fields_=[('weights',f),('cache',C.c_float*5632),('fault',C.c_int)]
lib=C.CDLL(str(E/'libfsmn-guarded.so'));lib.df_init.argtypes=[C.POINTER(Model),f,C.c_size_t];lib.df_step.argtypes=[C.POINTER(Model),f,f,f];lib.df_cmvn.argtypes=[C.POINTER(Model),f,f];lib.df_reset.argtypes=[C.POINTER(Model)]
def ptr(x):return x.ctypes.data_as(f)
m=Model();w=np.zeros(756933,np.float32);x=np.zeros(400,np.float32);y=np.zeros(2599,np.float32)
assert lib.df_init(C.byref(m),ptr(w),756932)==-1 and not m.weights
w[0]=np.nan;assert lib.df_init(C.byref(m),ptr(w),len(w))==-2 and not m.weights
w[:]=0;w[400:800]=1;assert lib.df_init(C.byref(m),ptr(w),len(w))==0
x[0]=np.nan;assert lib.df_step(C.byref(m),ptr(x),ptr(y),None)==-2
x[:]=0;x[0]=np.finfo(np.float32).max;w[400]=2
assert lib.df_step(C.byref(m),ptr(x),ptr(y),None)==-3 and m.fault==1
x[:]=0;assert lib.df_step(C.byref(m),ptr(x),ptr(y),None)==-3
lib.df_reset(C.byref(m));assert m.fault==0;assert lib.df_step(C.byref(m),ptr(x),ptr(y),None)==0
w=np.fromfile(E/'donor-local.f32',dtype='<f4');assert lib.df_init(C.byref(m),ptr(w),len(w))==0
g=np.load(E/'goldens.npz');r=json.load(open(E/'reference-receipt.json'));checks=[]
for row in r['cases']:
 inp=g[row['input']+'_input'][row['offset']:row['offset']+row['frames']];out=np.empty_like(inp)
 for a,b in zip(inp,out):assert lib.df_cmvn(C.byref(m),ptr(a),ptr(b))==0
 ref=g[row['prefix']+'_cmvn'][0];d=abs(out.astype(float)-ref);passed=bool(np.all(d<=1e-6+2e-7*abs(ref)));checks.append(dict(case=row['prefix'],passed=passed,max_abs=float(d.max())))
report=dict(guard_checks_passed=True,cmvn_passed=all(t['passed'] for t in checks),cmvn_checks=checks,scope='synthetic guard and frozen CMVN only; no acoustic model rerun',library_sha256=hashlib.sha256((E/'libfsmn-guarded.so').read_bytes()).hexdigest());p=E/'guard-cmvn-result.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print('guards PASS; CMVN',report['cmvn_passed'],max(x['max_abs'] for x in checks))
