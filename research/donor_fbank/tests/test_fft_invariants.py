"""Independent FFT gates; only runs after an explicitly approved v2 contract exists."""
import argparse,ctypes as C,hashlib,json,math,pathlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--library',required=True,type=pathlib.Path);p.add_argument('--contract',required=True,type=pathlib.Path);p.add_argument('--goldens',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args();contract=json.loads(a.contract.read_text())
assert contract['status']=='approved-for-v2-acceptance', 'contract needs independent review first'
assert contract['fft']['N']==512;N=512;u=2.0**-24
def gamma(n):return n*u/(1-n*u)
lib=C.CDLL(str(a.library));F=C.c_float;lib.donor_fbank_state_bytes.restype=C.c_size_t;lib.donor_fbank_init.argtypes=[C.c_void_p];lib.donor_fbank_fft_power.argtypes=[C.c_void_p,C.POINTER(F),C.POINTER(F)];lib.donor_fbank_fft_power.restype=C.c_int
state=(C.c_uint64*((lib.donor_fbank_state_bytes()+7)//8))();assert lib.donor_fbank_init(state)==0
from fft_reference import reference,cases,input_manifest
assert input_manifest()==contract["fft_input_manifest"]
def call(x):
 x=np.array(x,dtype=np.float32);out=np.zeros(257,dtype=np.float32);assert lib.donor_fbank_fft_power(state,x.ctypes.data_as(C.POINTER(F)),out.ctypes.data_as(C.POINTER(F)))==0;assert np.isfinite(out).all();return x,out
vectors=cases();results=[]
manifest_path=a.goldens/'manifest.json';assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==contract['golden_manifest_sha256']
manifest=json.loads(manifest_path.read_text())
for case in manifest['cases']:
 if case['frames']:
  path=a.goldens/'cases'/case['name']/'fft_input.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==case['layers']['fft_input']['sha256'];window=np.load(path)
  assert window.shape==(case['frames'],512) and window.dtype==np.float32
  vectors += [('golden:'+case['name']+':'+str(i),row,None) for i,row in enumerate(window)]
for name,input,peak in vectors:
 x,out=call(input);ref=reference(x);ref_power=np.abs(ref)**2;L1=math.fsum(abs(float(v)) for v in x);B=(gamma(contract['fft']['amplitude_operation_budget'])+8*(2.0**-53)/(1-8*(2.0**-53)))*L1
 error=np.abs(out.astype(np.float64)-ref_power)
 # Updated power-rounding term and Parseval budget require review in contract.
 bound=2*np.abs(ref)*B+B*B+gamma(contract['fft']['power_rounding_operations'])*(np.abs(ref)+B)**2+8*(2.0**-53)*ref_power
 ratio=np.divide(error,bound,out=np.zeros_like(error),where=bound>0);bin_ok=bool(np.all(error<=bound))
 energy_in=N*math.fsum(float(v)**2 for v in x);energy_out=float(out[0])+float(out[256])+2*math.fsum(float(v) for v in out[1:256]);relative=abs(energy_out-energy_in)/energy_in if energy_in else abs(energy_out)
 parseval_ok=(relative<=gamma(contract['fft']['parseval_operation_budget'])+contract['fft']['double_bookkeeping_allowance']) if energy_in else not out.any()
 special_ok=True
 if name=='zero':special_ok=not out.any()
 if name.startswith(('dc_','nyquist_')):
  expected=np.zeros(257,dtype=np.float32);expected[peak]=(512*float(x[0]))**2;special_ok=bool(np.array_equal(out,expected))
 if name.startswith('impulse_'):
  power=float(x[np.flatnonzero(x)[0]])**2;special_ok=bool(np.all(np.abs(out.astype(np.float64)-power)<=gamma(contract['fft']['impulse_operation_budget'])*power))
 if name.startswith(('sin_','cos_')):special_ok=int(out.argmax())==peak
 results.append({'name':name,'bin_gate':bin_ok,'parseval_gate':parseval_ok,'special_gate':special_ok,'max_bin_tolerance_ratio':float(ratio.max()),'parseval_relative':relative,'passed':bin_ok and parseval_ok and special_ok})
x,p1=call(next(x for name,x,_ in vectors if name=='dyadic_noise'))
for scale in [2.,.5]:
 _,p2=call(x*scale);scale_ok=bool(np.array_equal(p2,p1*(scale*scale)));results.append({'name':'binary_scale_'+str(scale),'passed':scale_ok})
report={'contract_sha256':hashlib.sha256(a.contract.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'scope':'FFT512power independent scalar/doubleDFT invariants; no feature/model/target claim','passed':all(r['passed'] for r in results),'cases':results}
a.output.open('x').write(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
