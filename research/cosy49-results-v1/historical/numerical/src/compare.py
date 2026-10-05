"""Pure typed comparisons; no source model or candidate evaluation."""
from common import *
def exact(np,a,b,name):
 assert a.shape==b.shape and a.dtype==b.dtype,name+': shape/type'
 assert np.isfinite(a).all()and np.isfinite(b).all(),name+': finite'
 assert a.tobytes()==b.tobytes(),name+': exact'
def numeric(np,checks,name,a,b,limit,gate='hard_accuracy',outward=False):
 assert a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all()
 error=np.abs(a.astype(np.float64)-b.astype(np.float64))
 if outward:error=np.nextafter(error,np.inf)
 limits=np.broadcast_to(np.asarray(limit,np.float64),error.shape);assert np.isfinite(limits).all()and(limits>0).all()
 fail=error>limits
 checks.append(dict(name=name,gate=gate,passed=not bool(fail.any()),elements=int(error.size),failed_elements=int(fail.sum()),max_abs=float(error.max(initial=0)),max_ratio=float((error/limits).max(initial=0)),observed_error_rounding='upward_binary64'if outward else'binary64_difference'))
def cache_sequence(np,before,stages):
 previous=[];after=[];cache=before.copy()
 for row in range(len(stages[3])):
  previous.append(cache.copy());cache[:,:-1,:]=cache[:,1:,:].copy()
  for layer in range(4):cache[:,-1,layer]=stages[3+4*layer][row]
  after.append(cache.copy())
 shape=(0,128,11,4)
 return np.stack(previous)if previous else np.empty(shape,np.float32),np.stack(after)if after else np.empty(shape,np.float32),cache

def certificate(checks,name,result,gate):
 checks.append(dict(name=name,gate=gate,passed=result['passed'],elements=result['elements'],failed_elements=result.get('failed_elements',len(result.get('failures',[]))),interval_certificate=result))
def typed_accuracy(np,checks,name,actual,call,declared,authority,canonical_api,softmax):
 authority.validate_metadata(declared)
 for key,tol in [('fbank',1e-3),('spliced',1e-3),('cmvn',2e-4)]:numeric(np,checks,name+'_'+key,actual[key],authority.interface_reference(call,key,declared),tol)
 rows=len(actual['logits']);logits=authority.interval_reference(call,'raw_logits',declared);probs=authority.interval_reference(call,'probabilities',declared)
 assert len(probs)==rows and len(logits['logits'])==rows
 certificate(checks,name+'_raw_interval',canonical_api.compare_raw_logits(actual['logits'],logits),'hard_raw_interval')
 if rows:certificate(checks,name+'_probability_interval',softmax.compare_absolute(actual['probabilities'].reshape(-1),[v for row in probs for v in row['probability_intervals']],'1e-5'),'hard_probability_interval')
 for label,value in [('native',actual['probabilities']),('canonical_decoder_interface',call['probabilities'])]:
  assert value.dtype==np.float32 and value.shape==(rows,6)and np.isfinite(value).all()and((value>=0)&(value<=1)).all()
  numeric(np,checks,name+'_'+label+'_rowsum',value.astype(np.float64).sum(axis=1),np.ones(rows),1e-5,'hard_probability_structure')
