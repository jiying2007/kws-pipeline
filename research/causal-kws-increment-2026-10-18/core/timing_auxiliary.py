"""Fixed weak teacher-relative auxiliary. No teacher computation or alignment."""
import math
AUX_CONFIG={'scale':0.1,'hinge_margin':0.5,'completion_offsets':[0,1,2,3],'prefix_offsets':[-1,0,1],'initial80_policy':'EMPTY_WITH_ORIGINAL_MASK_ABSTAIN','denominator':'original batch Bx2 supervised confidence sum','teacher_time_is_acoustic_boundary':False}
def make_windows(nodes,frames,samples):
 if len(nodes)!=4 or not all(isinstance(n,int) for n in nodes) or not all(a<b for a,b in zip(nodes,nodes[1:])):raise ValueError('four distinct ordered nodes required')
 def valid(t):return 8<=t<frames and 480*t+720<=samples
 t4=nodes[-1]
 return dict(completion=[t for t in range(t4,t4+4) if valid(t)],prefix=sorted({t+d for t in nodes[:3] for d in [-1,0,1] if t+d<t4 and valid(t+d)}),initial80=[t for t in range(frames) if valid(t) and 480*t+720<=1280])
def reference_term(completion,prefix,confidence,denominator):
 if not completion:return 0.
 c=max(completion);bce=max(-c,0)+math.log1p(math.exp(-abs(c)));hinge=max(0.,.5+max(prefix)-c) if prefix else 0.
 return .1*confidence*(bce+hinge)/denominator

def auxiliary(torch,logits,rows,weights,timing):
 denominator=weights.sum() # Current frozen dataset has exactly two known labels per row.
 terms=[];logs=[]
 for b,row in enumerate(rows):
  evidence=timing[row['recording']]
  if not evidence['eligible']:continue
  k=evidence['keyword_index'];cids=evidence['completion'];pids=evidence['prefix']
  if not cids:continue
  c=logits[b,cids,k].max();p=logits[b,pids,k].max() if pids else None
  bce=torch.nn.functional.softplus(-c);hinge=torch.relu(AUX_CONFIG['hinge_margin']+p-c) if p is not None else c*0
  weighted=row['confidence']*(bce+hinge);terms.append(weighted)
  logs.append(dict(recording=row['recording'],cache=row['cache'],source_group=row['source_group'],confidence=row['confidence'],completion_max=float(c.detach()),prefix_max=None if p is None else float(p.detach()),completion_bce=float(bce.detach()),prefix_hinge=float(hinge.detach()),weighted_unscaled_numerator=float(weighted.detach()),denominator=float(denominator.detach())))
 result=AUX_CONFIG['scale']*torch.stack(terms).sum()/denominator if terms else logits.sum()*0
 return result,logs

def positive_diagnostics(torch,logits,rows,mask,timing):
 result=[]
 for b,row in enumerate(rows):
  if sum(row['labels'])!=1:continue
  k=row['labels'].index(1);ids=mask[b].nonzero().flatten();v=logits[b,ids,k];t=int(ids[v.argmax()]);e=timing[row['recording']]
  result.append(dict(recording=row['recording'],cache=row['cache'],source_group=row['source_group'],time_eligible=e['eligible'],keyword_index=k,full_proxy_peak=float(v.max().detach().sigmoid()),full_proxy_positive_loss=float(torch.nn.functional.softplus(-v.max()).detach()),argmax_frame=t,argmax_available_ms=(480*t+720)/16,argmax_after_eof=480*t+720>row['samples']))
 return result
