"""Read frozen synthetic golden; write honest report, never change gates."""
import ctypes as C,hashlib,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];E=pathlib.Path(sys.argv[1])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((E/'reference-receipt.json').read_text());assert sha(E/'goldens.npz')==r['goldens_sha256'];assert sha(E/'weights-manifest.json')==r['weights_manifest_sha256']
wmanifest=json.loads((E/'weights-manifest.json').read_text());assert sha(E/'donor-local.f32')==wmanifest['payload_sha256']
w=np.fromfile(E/'donor-local.f32',dtype='<f4');assert w.size==756933 and np.isfinite(w).all()
for t in wmanifest['tensors']:
 raw=(E/'donor-local.f32').read_bytes()[t['offset']:t['offset']+t['bytes']];assert hashlib.sha256(raw).hexdigest()==t['sha256']
cmd=['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-shared','-fPIC','-I',str(ROOT),str(E/'fsmn-double-diagnostic.c'),'-o',str(E/'libfsmn.so')];subprocess.run(cmd,check=True)
f=C.POINTER(C.c_float)
class Model(C.Structure):_fields_=[('weights',f),('cache',C.c_float*5632),('fault',C.c_int)]
lib=C.CDLL(str(E/'libfsmn.so'));lib.df_init.argtypes=[C.POINTER(Model),f,C.c_size_t];lib.df_step.argtypes=[C.POINTER(Model),f,f,f]
def ptr(a):return a.ctypes.data_as(f)
model=Model();assert lib.df_init(C.byref(model),ptr(w),len(w))==0
g=np.load(E/'goldens.npz',allow_pickle=False);dims=[140,250,250]+[128,128,250,250]*4+[140,2599];checks=[]
def compare(name,a,b):
 d=np.abs(a.astype(np.float64)-b.astype(np.float64));gate=1e-4+1e-5*np.abs(b.astype(np.float64));checks.append(dict(name=name,passed=bool(np.all(d<=gate)),max_abs=float(d.max()),max_gate_ratio=float((d/gate).max()),fail_elements=int((d>gate).sum()),elements=d.size))
for row in r['cases']:
 if row['reset']:lib.df_reset(C.byref(model))
 traces=[]
 for x in g[row['input']+'_input'][row['offset']:row['offset']+row['frames']]:
  out=np.zeros(2599,np.float32);trace=np.zeros((21,2599),np.float32);assert lib.df_step(C.byref(model),ptr(x),ptr(out),ptr(trace))==0;traces.append(trace)
 traces=np.stack(traces)
 for s,d in enumerate(dims):compare(row['prefix']+f'_stage{s}',traces[:,s,:d],g[row['prefix']+f'_stage{s}'][0])
 compare(row['prefix']+'_cache',np.ctypeslib.as_array(model.cache).reshape(1,128,11,4),g[row['prefix']+'_cache'])
report=dict(passed=all(c['passed'] for c in checks),checks=checks,reference_receipt_sha256=sha(E/'reference-receipt.json'),source_sha256=sha(E/'fsmn-double-diagnostic.c'),header_sha256=sha(ROOT/'fsmn.h'),comparison_sha256=sha(pathlib.Path(__file__)),compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0],command=cmd,scope='stage B synthetic only; no speech inference')
p=E/'network-result-double-diagnostic.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print('PASS' if report['passed'] else 'FAIL',len(checks),'checks',sum(not x['passed'] for x in checks),'failed',max(x['max_abs'] for x in checks),'max abs')
