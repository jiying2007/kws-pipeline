"""Compare C layers with independently frozen official-source synthetic goldens."""
import argparse,ctypes as C,hashlib,json,pathlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--library',required=True,type=pathlib.Path);p.add_argument('--goldens',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args()
if a.output.exists():raise ValueError('refuse to overwrite numerical evidence')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a.goldens/'tolerances.json')=='b1f898b183916b0bd1cef789cbb75be02e7e5052256b4a74f00f411dc5946ab1'
spec=json.loads((pathlib.Path(__file__).resolve().parents[1]/'spec.json').read_text());assert sha(a.goldens/'manifest.json')==spec['golden_manifest_sha256']
m=json.loads((a.goldens/'manifest.json').read_text());tol=json.loads((a.goldens/'tolerances.json').read_text());lib=C.CDLL(str(a.library));F=C.c_float
class Trace(C.Structure):_fields_=[('dc',F*400),('preemphasis',F*400),('windowed',F*512),('power',F*257),('mel',F*80),('logfbank',F*80)]
lib.donor_fbank_state_bytes.restype=C.c_size_t;lib.donor_fbank_init.argtypes=[C.c_void_p];lib.donor_fbank_analyze_frame.argtypes=[C.c_void_p,C.POINTER(C.c_int16),C.POINTER(Trace)];lib.donor_cmvn400.argtypes=[C.POINTER(F)]*4
state=(C.c_uint64*((lib.donor_fbank_state_bytes()+7)//8))();assert lib.donor_fbank_init(state)==0
for name,h in m['constant_files'].items():
 assert pathlib.Path(name).name==name and sha(a.goldens/'constants'/name)==h
mean=np.load(a.goldens/'constants/mean400.npy');std=np.load(a.goldens/'constants/istd400.npy');results=[];failures=[]
def compare(name,layer,actual,ref,tolerance):
 assert actual.shape==ref.shape and np.isfinite(actual).all()
 d=np.abs(actual.astype(np.float64)-ref.astype(np.float64));bound=tolerance['atol']+tolerance['rtol']*np.abs(ref.astype(np.float64));bad=d>bound
 idx=np.unravel_index(int(d.argmax()),d.shape) if d.size else None
 ratio=np.divide(d,bound,out=np.zeros_like(d),where=bound>0);ri=np.unravel_index(int(ratio.argmax()),ratio.shape) if ratio.size else None
 out={'max_tolerance_ratio':float(ratio.max()) if ratio.size else 0,'worst_ratio_index':[int(z) for z in ri] if ri else None,'case':name,'layer':layer,'shape':list(ref.shape),'max_abs':float(d.max()) if d.size else 0,'violations':int(bad.sum()),'worst_index':[int(z) for z in idx] if idx else None,'worst_actual':float(actual[idx]) if idx else None,'worst_reference':float(ref[idx]) if idx else None};results.append(out)
 if bad.any():failures.append(out)
for case in m['cases']:
 name=case['name'];assert pathlib.Path(name).name==name;root=a.goldens/'cases'/name;pcm=np.load(root/'pcm.npy');assert pcm.dtype==np.int16 and pcm.shape==(case['samples'],)
 expected_frames=max(0,1+(case['samples']-400)//160);assert case['frames']==expected_frames
 assert case['frame_start_samples']==[160*i for i in range(expected_frames)]
 assert case['frame_end_samples_exclusive']==[400+160*i for i in range(expected_frames)]
 for layer,info in case['layers'].items():
  assert pathlib.Path(layer).name==layer and sha(root/(layer+'.npy'))==info['sha256']
  arr=np.load(root/(layer+'.npy'));assert arr.dtype==np.float32 and list(arr.shape)==info['shape'] and np.isfinite(arr).all()
 assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==case['pcm_sha256']
 if not case['frames']:continue
 layers={x:[] for x in ['dc','preemphasis','windowed','fft_input','power','mel_power','logfbank']}
 for pos in case['frame_start_samples']:
  tr=Trace();frame=np.ascontiguousarray(pcm[pos:pos+400],dtype=np.int16);assert lib.donor_fbank_analyze_frame(state,frame.ctypes.data_as(C.POINTER(C.c_int16)),C.byref(tr))==0
  for layer,field,count in [('dc','dc',400),('preemphasis','preemphasis',400),('windowed','windowed',400),('fft_input','windowed',512),('power','power',257),('mel_power','mel',80),('logfbank','logfbank',80)]:layers[layer].append(np.ctypeslib.as_array(getattr(tr,field))[:count].copy())
 for layer,arr in layers.items():
  path=root/(layer+'.npy');assert sha(path)==case['layers'][layer]['sha256'];ref=np.load(path);actual=np.array(arr);compare(name,layer,actual,ref,{'atol':2e-6,'rtol':0} if case['input']['family'] in ['silence','dc'] and layer=='logfbank' else tol['layers'][layer])
  if case['input']['family'] in ['silence','dc'] and layer not in ['logfbank']:assert not actual.any(),(name,layer,'strict zero invariant')
 # Isolated C affine on actual chronological selected donor context reference.
 for label,refinput,refoutput in [('cmvn400','skip3_splice400','cmvn400'),('cmvn400_repeat','cmvn400_repeat_input','cmvn400_repeat_output')]:
  inp=np.load(root/(refinput+'.npy'));out=np.empty_like(inp)
  for row,dest in zip(inp,out):assert lib.donor_cmvn400(row.ctypes.data_as(C.POINTER(F)),mean.ctypes.data_as(C.POINTER(F)),std.ctypes.data_as(C.POINTER(F)),dest.ctypes.data_as(C.POINTER(F)))==0
  compare(name,label+'_isolated',out,np.load(root/(refoutput+'.npy')),tol['isolated_cmvn400_given_golden_input'])
 # Test-only splice of C fbank, preserving donor chronological order/left replication.
 cf=np.array(layers['logfbank']);spliced=np.array([np.concatenate([cf[max(0,j)] for j in range(center-2,center+3)]) for center in range(max(0,len(cf)-2))],dtype=np.float32).reshape(-1,400);selected=spliced[::3].copy();out=np.empty_like(selected)
 for row,dest in zip(selected,out):assert lib.donor_cmvn400(row.ctypes.data_as(C.POINTER(F)),mean.ctypes.data_as(C.POINTER(F)),std.ctypes.data_as(C.POINTER(F)),dest.ctypes.data_as(C.POINTER(F)))==0
 compare(name,'splice400',spliced,np.load(root/'splice400.npy'),tol['layers']['splice400']);compare(name,'skip3_splice400',selected,np.load(root/'skip3_splice400.npy'),tol['layers']['skip3_splice400']);compare(name,'cmvn400',out,np.load(root/'cmvn400.npy'),{'atol':1e-6,'rtol':0} if case['input']['family'] in ['silence','dc'] else tol['layers']['cmvn400'])
report={'passed':not failures,'library_sha256':sha(a.library),'comparison_script_sha256':sha(pathlib.Path(__file__)),'golden_manifest_sha256':sha(a.goldens/'manifest.json'),'tolerance_sha256':sha(a.goldens/'tolerances.json'),'cases':len(m['cases']),'frames':sum(c['frames'] for c in m['cases']),'failures':failures,'layers':results,'c_context_implemented':False}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':not failures,'failures':failures,'max_abs_by_layer':{layer:max(x['max_abs'] for x in results if x['layer']==layer) for layer in {x['layer'] for x in results}}},indent=2));raise SystemExit(0 if not failures else 1)
