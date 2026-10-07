"""Hard gamma/mixed checks on saved native actual inputs. Zero C hooks."""
from common import *
from compare import numeric
GAMMA=OLD/'v2/harness/fixtures.py';MIXED=OLD/'oracle-corrected/harness/fixtures.py'
class LocalAudit:
 def __init__(self,np,state):
  self.np=np;self.state=state;self.gamma=module('recovery_gamma',GAMMA);self.mixed=module('recovery_mixed',MIXED)
  self.counts=dict(affine_rows=0,memory_rows=0,weighted_products=0,candidate_calls=0)
 def check(self,prefix,arrays,checks):
  np=self.np;state=self.state;output={};count=len(arrays[prefix+'_logits'])
  for stage,stem,previous,ni,no in self.gamma.AFFINES:
   x=arrays[prefix+'_'+previous].astype(np.float64);w=state[stem+'.linear.weight'].astype(np.float64);bias=state.get(stem+'.linear.bias');bias=np.zeros(no,np.float64)if bias is None else bias.astype(np.float64)
   assert x.shape==(count,ni)and w.shape==(no,ni)and bias.shape==(no,)
   oracle,l1,bound,_=self.gamma.oracle_bounds(np,x[:,None,:]*w[None,:,:],np.broadcast_to(bias,(count,no)),ni+1,2*ni+1)
   tight=self.mixed.mixed_bound(np,oracle,l1,ni+1);actual=arrays[prefix+f'_stage{stage}']
   numeric(np,checks,prefix+f'_affine{stage}_gamma',actual,oracle,bound,'hard_local_gamma',True);numeric(np,checks,prefix+f'_affine{stage}_mixed',actual,oracle,tight,'hard_local_mixed',True)
   for suffix,value in [('oracle',oracle),('l1_upper',l1),('gamma_bound',bound),('mixed_bound',tight)]:output[prefix+f'_affine{stage}_'+suffix]=value
   self.counts['affine_rows']+=count;self.counts['weighted_products']+=count*ni*no
  cache=arrays[prefix+'_cache_previous_rows'].astype(np.float64)
  for layer in range(4):
   projection=arrays[prefix+f'_stage{3+4*layer}'].astype(np.float64);left=state[f'backbone.fsmn.{layer}.1.conv_left.weight'].reshape(128,10).astype(np.float64);right=state[f'backbone.fsmn.{layer}.1.conv_right.weight'].reshape(128,2).astype(np.float64)
   products=np.concatenate([cache[:,:,:10,layer]*left[None,:,:],(cache[:,:,10,layer]*right[None,:,0])[:,:,None],(projection*right[None,:,1])[:,:,None]],axis=-1)
   oracle,l1,bound,_=self.gamma.oracle_bounds(np,products,cache[:,:,9,layer],14,27);tight=self.mixed.mixed_bound(np,oracle,l1,14);actual=arrays[prefix+f'_stage{4+4*layer}']
   numeric(np,checks,prefix+f'_memory{layer}_gamma',actual,oracle,bound,'hard_local_gamma',True);numeric(np,checks,prefix+f'_memory{layer}_mixed',actual,oracle,tight,'hard_local_mixed',True)
   for suffix,value in [('oracle',oracle),('l1_upper',l1),('gamma_bound',bound),('mixed_bound',tight)]:output[prefix+f'_memory{layer}_'+suffix]=value
   self.counts['memory_rows']+=count;self.counts['weighted_products']+=count*128*12
  return output
 def finish(self):assert self.counts==dict(affine_rows=2760,memory_rows=920,weighted_products=89466320,candidate_calls=0)
