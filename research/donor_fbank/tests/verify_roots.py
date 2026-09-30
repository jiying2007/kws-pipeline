"""Check the declared direct-rounded-root assumption against independent Decimal roots."""
import pathlib,re,struct
from fft_reference import ROOTS
root=pathlib.Path(__file__).resolve().parents[1];text=(root/'fft_twiddles.h').read_text();arrays=[]
for name in ['cos','sin']:
 body=re.search(r'donor_fft_'+name+r'\[256\] = \{(.*?)\};',text,re.S).group(1)
 vals=[float.fromhex(x.strip()[:-1]) for x in body.split(',') if x.strip()];assert len(vals)==256
 for i,x in enumerate(vals):assert struct.pack('<f',x)==struct.pack('<f',ROOTS[i][0 if name=='cos' else 1])
 arrays.append(vals)
u=2.0**-24;error=max(abs(complex(arrays[0][i]-ROOTS[i][0],arrays[1][i]-ROOTS[i][1])) for i in range(256));assert error<=u
assert arrays[0][0]==1 and arrays[1][0]==0 and arrays[0][128]==0 and arrays[1][128]==-1
print('direct root coefficients: exact float32 rounding; max complex error',error,'<=',u)
