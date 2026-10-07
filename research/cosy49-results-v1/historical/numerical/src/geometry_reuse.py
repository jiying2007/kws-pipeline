"""Exact geometry and array-serialization functions; no model calls."""
from common import *
def partitions(n):
 result=[];offset=0
 while offset<n:
  count=min(n-offset,RAGGED[len(result)%len(RAGGED)]);result.append(count);offset+=count
 return result

def independent_geometry(samples):
 wave=features=offset=fb_total=rows=0;calls=[]
 for index,start in enumerate(range(0,samples,4800)):
  count=min(4800,samples-start);combined=wave+count
  fb=0 if combined<800 else(combined-400)//160+1
  splice=(fb+(2 if features==0 else features)-4)if fb else 0
  local=list(range(offset,splice,3));centers=[j if features==0 else fb_total-features+j+2 for j in local]
  wave=combined-fb*160
  if fb:offset=(3-((splice+(0 if offset==0 else 3-offset))%3))%3;features=min(fb,4)
  calls.append(dict(call_index=index,available_samples=start+count,call_samples=count,waveform_samples=combined,fbank_rows=fb,splice_rows=splice,selected_rows=len(local),is_final_short=int(count<4800),wave_samples=wave,feature_count=features,offset=offset,centers=centers))
  fb_total+=fb;rows+=len(local)
 return dict(samples=samples,calls=calls,fbank_rows=fb_total,model_rows=rows,canonical_calls=len(calls),acoustic_calls=sum(c['selected_rows']>0 for c in calls))

def verify_geometry(schedules):
 assert len(schedules)==8 and[s['samples']for s in schedules]==LENGTHS
 for n,s in zip(LENGTHS,schedules):
  other=independent_geometry(n)
  for k in ['samples','fbank_rows','model_rows','canonical_calls','acoustic_calls']:assert s[k]==other[k]
  assert len(s['calls'])==len(other['calls'])
  for a,b in zip(s['calls'],other['calls']):
   for k in GEOMETRY_FIELDS+['centers']:assert a[k]==b[k],k
 assert(sum(s['fbank_rows']for s in schedules),sum(s['model_rows']for s in schedules),sum(s['canonical_calls']for s in schedules))==(351,115,17)
 return True

def save_arrays(np,p,arrays):
 assert all(a.dtype!=object and np.isfinite(a).all()for a in arrays.values())
 with Path(p).open('xb')as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
 assert Path(p).stat().st_size<=20*1024**2
