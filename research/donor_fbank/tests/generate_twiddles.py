"""Explicit float32 roots of unity; no repeated float rotation accumulation."""
import argparse,math,pathlib,struct
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args()
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
arrays=[]
for kind in ['cos','sin']:
 values=[f32(math.cos(-2*math.pi*i/512) if kind=='cos' else math.sin(-2*math.pi*i/512)) for i in range(256)]
 values[0]=1.0 if kind=='cos' else 0.0;values[128]=0.0 if kind=='cos' else -1.0
 lines=['static const float donor_fft_'+kind+'[256] = {']
 for i in range(0,256,8):lines.append('  '+', '.join(v.hex()+'f' for v in values[i:i+8])+',')
 arrays.append('\n'.join(lines+['};']))
a.output.write_text('/* Float32 radix2 twiddles from mathematical roots of unity,\n *rounded once; exact cardinal values. Avoid accumulated float rotation drift. */\n'+'\n'.join(arrays)+'\n')
