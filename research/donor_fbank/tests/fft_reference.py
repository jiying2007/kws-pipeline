"""Independent high-precision roots/direct-DFT input specification; never imports C."""
from decimal import Decimal,localcontext
import hashlib,json,math
import numpy as np
N=512
PI=Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628')
def roots():
 result=[]
 with localcontext() as ctx:
  ctx.prec=80
  for k in range(N):
   angle=-2*PI*Decimal(k if k<=256 else k-512)/N
   st=angle;ct=Decimal(1);ss=st;cc=ct
   for j in range(1,60):
    st*=-angle*angle/((2*j)*(2*j+1));ct*=-angle*angle/((2*j-1)*(2*j));ss+=st;cc+=ct
   result.append((float(cc),float(ss)))
 for k,value in [(0,(1.,0.)),(128,(0.,-1.)),(256,(-1.,0.)),(384,(0.,1.))]:result[k]=value
 return result
ROOTS=roots()
def reference(x):
 return np.array([complex(math.fsum(float(v)*ROOTS[(k*n)%512][0] for n,v in enumerate(x)),math.fsum(float(v)*ROOTS[(k*n)%512][1] for n,v in enumerate(x))) for k in range(257)],dtype=np.complex128)
def cases():
 result=[('zero',np.zeros(512,dtype=np.float32),None)]
 for value in [1.,-2.,16.]:result.append(('dc_'+str(value),np.full(512,value,dtype=np.float32),0));result.append(('nyquist_'+str(value),np.array([value*(-1)**i for i in range(512)],dtype=np.float32),256))
 for pos in [0,1,127,255,511]:
  for amplitude in [1.,32767.]:
   x=np.zeros(512,dtype=np.float32);x[pos]=amplitude;result.append(('impulse_'+str(pos)+'_'+str(amplitude),x,None))
 for k in [1,7,53,123,211,255]:
  for kind,index in [('sin',1),('cos',0)]:result.append((kind+'_'+str(k),np.array([ROOTS[(k*n)%512][index] for n in range(512)],dtype=np.float32),k))
 seed=81173;values=[]
 for i in range(512):seed=(1664525*seed+1013904223)&0xffffffff;values.append(((seed>>16)%4096-2048)/1024)
 result.append(('dyadic_noise',np.array(values,dtype=np.float32),None));return result

def input_manifest():return {'roots_f64le_sha256':hashlib.sha256(np.array(ROOTS,dtype='<f8').tobytes()).hexdigest(),'root_precision_decimal_digits':80,'taylor_terms':60,'root_index_policy':'modulo512; independent Decimal roots rounded tofloat64; cardinal roots exact','cases':[{'name':name,'f32le_sha256':hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),'expected_peak':peak} for name,x,peak in cases()]}
if __name__=='__main__':print(json.dumps(input_manifest(),indent=2))
