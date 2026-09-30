"""One approved fixed stage5 oracle; no network/corpus run or tolerance change."""
import ctypes as C,hashlib,json,math,pathlib,subprocess,sys
import numpy as np
import torch
E=pathlib.Path('/workspace/shared/kws-fsmn-ab-evidence');ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
g=np.load(E/'goldens.npz');x=g['random_whole_0_stage4'][0].copy();historic=g['random_whole_0_stage5'][0].copy()
weights=np.fromfile(E/'donor-local.f32',dtype='<f4');w=weights[125726:157726].reshape(250,128).copy();b=weights[157726:157976].copy()
config=dict(stage='random_whole_0_stage5',shape=[37,128,250],source_sha256=sha(ROOT/'fsmn.c'),driver_sha256=sha(pathlib.Path(__file__)),goldens_sha256=sha(E/'goldens.npz'),payload_sha256=sha(E/'donor-local.f32'),input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(),bias_sha256=hashlib.sha256(b.tobytes()).hexdigest(),oracle='math.fsum(exact binary64 products and bias)',fp32_bound='gamma_257(2^-24)*S + 257*2^-150',oracle_margin='gamma_129(2^-53)*S',S='sum abs(input*weight)+abs(bias)',assumptions='IEEE finite operands, normal rounding, faithful math.fsum; explicit FP32 subnormal absolute allowance',purpose='local operator diagnosis only; no B gate change')
p=E/'local-affine-prereg.json'
if sys.argv[1]=='prepare':
 assert not p.exists();p.write_text(json.dumps(config,indent=2)+'\n');print('FROZEN',sha(p));raise SystemExit
assert sys.argv[1]=='run' and json.loads(p.read_text())==config
source=E/'local-affine-wrapper.c';source.write_text('#include "'+str(ROOT/'fsmn.c')+'"\nvoid local_affine(const float*x,float*y,const float*w,const float*b){affine(x,y,w,b,128,250);}\n')
cmd=['cc','-std=c11','-O2','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(E/'local-affine.so')];subprocess.run(cmd,check=True)
f=C.POINTER(C.c_float);lib=C.CDLL(str(E/'local-affine.so'));lib.local_affine.argtypes=[f,f,f,f]
def ptr(a):return a.ctypes.data_as(f)
y=np.empty((37,250),np.float32)
for inp,out in zip(x,y):lib.local_affine(ptr(inp),ptr(out),ptr(w),ptr(b))
torch.set_num_threads(1);torch.set_num_interop_threads(1)
with torch.inference_mode():t=torch.nn.functional.linear(torch.from_numpy(x),torch.from_numpy(w),torch.from_numpy(b)).numpy()
oracle=np.empty_like(y,dtype=np.float64);S=np.empty_like(oracle)
for i in range(37):
 for o in range(250):
  terms=[float(a)*float(v) for a,v in zip(x[i],w[o])]+[float(b[o])];oracle[i,o]=math.fsum(terms);S[i,o]=math.fsum(abs(a) for a in terms)
gamma=lambda n,u:n*u/(1-n*u)
bound=gamma(257,2**-24)*S+257*2**-150;oracle_margin=gamma(129,2**-53)*S
reports=[]
for name,a in [('C_4way',y),('Torch_recomputed',t),('Torch_frozen',historic)]:
 d=abs(a.astype(float)-oracle);gate=1e-4+1e-5*abs(oracle)
 reports.append(dict(name=name,max_abs=float(d.max()),max_error_bound_ratio=float((d/(bound+oracle_margin)).max()),bound_failures=int((d>bound+oracle_margin).sum()),original_B_formula_vs_oracle_failures=int((d>gate).sum())))
d=abs(y.astype(float)-historic);bg=1e-4+1e-5*abs(historic)
result=dict(prereg_sha256=sha(p),reports=reports,C_vs_frozen=dict(max_abs=float(d.max()),B_failures=int((d>bg).sum())),torch_recomputed_matches_frozen=bool(np.array_equal(t,historic)),bound_min=float(bound.min()),bound_max=float(bound.max()),oracle_margin_max=float(oracle_margin.max()),scope='one fixed local affine only; original network B remains FAILED',compiler_command=cmd,torch=torch.__version__,python=sys.version)
np.savez(E/'local-affine-arrays.npz',input=x,C=y,Torch=t,frozen=historic,oracle=oracle,S=S,bound=bound,oracle_margin=oracle_margin);result['arrays_sha256']=sha(E/'local-affine-arrays.npz');out=E/'local-affine-result.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
