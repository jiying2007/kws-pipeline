"""Independent Decimal80 roots, complex sums, power, mel and log; no native/Torch math."""
from decimal import Decimal,localcontext,ROUND_FLOOR,ROUND_CEILING,ROUND_HALF_EVEN
import math
import numpy as np
D=Decimal

def roots():
 # Machin's identity with 110-digit arithmetic and alternating series residual <1e-108.
 with localcontext()as c:
  c.prec=110;c.rounding=ROUND_HALF_EVEN
  def atan_inverse(q):
   x=D(1)/q;s=t=x;j=1
   while True:
    t*=-x*x;a=t/(2*j+1);s+=a
    if abs(a)<D('1e-108'):return s
    j+=1
  pi=16*atan_inverse(5)-4*atan_inverse(239)
  out=[]
  for k in range(512):
   a=-2*pi*(k if k<=256 else k-512)/512;st=ss=a;ct=cc=D(1)
   for j in range(1,100):
    st*=-a*a/((2*j)*(2*j+1));ct*=-a*a/((2*j-1)*(2*j));ss+=st;cc+=ct
   with localcontext()as low:
    low.prec=80;low.rounding=ROUND_HALF_EVEN;out.append((+cc,+ss))
  for k,v in [(0,(1,0)),(128,(0,-1)),(256,(-1,0)),(384,(0,1))]:out[k]=tuple(map(D,v))
 return out

def evaluate(x,weights,offsets,bins,rs):
 assert isinstance(x,np.ndarray)and x.dtype==np.float32 and x.shape==(512,)
 assert np.isfinite(x).all()and np.all(np.abs(x.astype(np.float64))<=65536)and np.all(x[400:]==0)
 assert isinstance(weights,np.ndarray)and weights.dtype==np.float32 and weights.shape==(501,)and np.isfinite(weights).all()and np.all((weights>0)&(weights<=1))
 assert len(offsets)==81 and int(offsets[0])==0 and int(offsets[-1])==501 and all(int(a)<=int(b)for a,b in zip(offsets,offsets[1:]))
 assert len(bins)==501 and all(int(v)==v and 0<=int(v)<257 for v in bins)and all(int(v)==v and 0<=int(v)<=501 for v in offsets)
 assert len(rs)==512 and all(a.is_finite()and b.is_finite()for a,b in rs)
 with localcontext()as c:
  c.prec=80;c.rounding=ROUND_HALF_EVEN
  xd=[D.from_float(float(v))for v in x[:400]];re=[];im=[]
  for k in range(257):
   re.append(sum((v*rs[(k*n)%512][0]for n,v in enumerate(xd)),D(0)))
   im.append(sum((v*rs[(k*n)%512][1]for n,v in enumerate(xd)),D(0)))
  power=[a*a+b*b for a,b in zip(re,im)]
  wd=[D.from_float(float(v))for v in weights]
  mel=[sum((power[int(bins[j])]*wd[j]for j in range(int(offsets[m]),int(offsets[m+1]))),D(0))for m in range(80)]
  logs=[max(v,D(2)**-23).ln()for v in mel]
 return dict(fft_re=re,fft_im=im,power=power,mel=mel,logfbank=logs)

UNCERTAINTY={
 'meaning':'Conservative absolute bookkeeping bound for this diagnostic, not a candidate acceptance tolerance',
 'root_absolute_error':'1e-78',
 'complex_component_absolute_error':'1e-60',
 'basis':'110-digit Machin atan alternating residual <1e-108; 100 Taylor terms at |angle|<=pi; roots rounded once to Decimal80. For 400 float32 inputs bounded by65536, root+80digit sum roundoff is <<1e-60. Power error <=2*(|re|+|im|)*1e-60+2e-120 plus <1e-60 arithmetic rounding. Weighted positive mel error propagates those power bounds with1e-55 arithmetic allowance. Saved outward80 per-output intervals propagate the2^-23 floor through monotone logarithm, with1e-75 log allowance. These arithmetic allowances safely dominate <=501-term80digit sums under the executable amplitude/coefficient guards.',
 'not_included':'Original float32 DC/preemphasis/window rounding is deliberately fixed, not part of DFT oracle.'}


def output_intervals(ideal,weights,offsets,bins):
 """Outward bounds for the mathematical DFT of fixed FP32 input; not acceptance gates."""
 def outward(x,direction):
  with localcontext()as c:
   c.prec=80;c.rounding=direction;return +x
 def lower_sub(a,b):
  with localcontext()as c:
   c.prec=110;c.rounding=ROUND_FLOOR;return a-b
 def pair(lo,hi):return [str(outward(lo,ROUND_FLOOR)),str(outward(hi,ROUND_CEILING))]
 with localcontext()as c:
  c.prec=110;c.rounding=ROUND_CEILING
  e=D('1e-60');eps=D(2)**-23
  radii=[2*(abs(a)+abs(b))*e+2*e*e+D('1e-60')for a,b in zip(ideal['fft_re'],ideal['fft_im'])]
  power_intervals=[pair(max(D(0),lower_sub(v,r)),v+r)for v,r in zip(ideal['power'],radii)]
  mel_radii=[sum((radii[int(bins[j])]*D.from_float(float(weights[j]))for j in range(int(offsets[m]),int(offsets[m+1]))),D(0))+D('1e-55')for m in range(80)]
  mel_intervals=[pair(max(D(0),lower_sub(v,r)),v+r)for v,r in zip(ideal['mel'],mel_radii)]
  # ln inputs are floored before logarithm. 1e-75 covers rounded Decimal80 reported
  # log and Decimal110 endpoint ln arithmetic; outward80 serialization follows.
  log_intervals=[pair(lower_sub(max(eps,lower_sub(v,r)).ln(),D('1e-75')),max(eps,v+r).ln()+D('1e-75'))for v,r in zip(ideal['mel'],mel_radii)]
  log_radii=[max(abs(v-D(lo)),abs(D(hi)-v))for v,(lo,hi)in zip(ideal['logfbank'],log_intervals)]
  assert max(radii)<D('1.1e-52')and max(mel_radii)<D('5.6e-50')and max(log_radii)<D('5e-43')
  return dict(global_absolute_caps=dict(root='1e-78',complex_component='1e-60',power='1.1e-52',mel='5.6e-50',clamped_log='5e-43'),complex_component_absolute_radius=str(e),power_intervals=power_intervals,mel_intervals=mel_intervals,log_intervals=log_intervals,maximum_power_absolute_radius=str(outward(max(radii),ROUND_CEILING)),maximum_mel_absolute_radius=str(outward(max(mel_radii),ROUND_CEILING)),maximum_log_absolute_radius=str(outward(max(log_radii),ROUND_CEILING)),power_rounding_allowance='1e-60',mel_rounding_allowance='1e-55',log_rounding_allowance='1e-75',outward_serialization_digits=80)
