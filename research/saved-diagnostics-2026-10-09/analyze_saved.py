#!/usr/bin/env python3
"""Saved arrays/algebra only: no framework import, audio, model forward, gate edits."""
import argparse, pathlib, json, hashlib
import numpy as np
EXPECTED_SOURCES={'local-certificate-v2-failed/qualification/run-once/D20/pcm9600.npz': '2bff23791b80147968fb7e5b4986f1dc612a088f7310d17f352092de2e69cdc3', 'local-certificate-v2-failed/qualification/run-once/D20/pcm9600-C.npz': '2b02fbb3179d3c25a65e6881539b1bf4bbb5d7048e73b8f6528e342bf8b73992', 'host-runtime-ab/exports/D20/manifest.json': 'bd7b256f5def4ad2c211b47789ada0c8f07ced107c8add0afea79561ff96d6eb'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bitexact(a,b):
 return a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C')
def diff(a,b):
 a=a.astype(np.float64);b=b.astype(np.float64);d=a-b;bad=np.abs(d)>1e-4+1e-5*np.abs(b)
 return {'max_abs':float(np.max(np.abs(d),initial=0)), 'l1':float(np.abs(d).sum()),'bad_elements_at_frozen_raw_tolerance':int(bad.sum()),'bad_indices':np.argwhere(bad).tolist()}
def main():
 import sys
 if sys.flags.optimize:raise RuntimeError("optimized Python disables evidence assertions; refused")
 ap=argparse.ArgumentParser();ap.add_argument('archive',type=pathlib.Path);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('output already exists; evidence will not be overwritten')
 for name,expected in EXPECTED_SOURCES.items():
  if sha(a.archive/name)!=expected:raise ValueError('saved source identity mismatch: '+name)
 p=a.archive/'local-certificate-v2-failed/qualification/run-once/D20';paths=[p/'pcm9600.npz',p/'pcm9600-C.npz'];t,c=[np.load(q,allow_pickle=False) for q in paths]
 ex=a.archive/'host-runtime-ab/exports/D20';mp=ex/'manifest.json';bp=ex/'a20.f32';m=json.loads(mp.read_text());assert sha(bp)==m['payload_sha256'];raw=bp.read_bytes();weights={}
 for x in m['tensors']:
  dat=raw[x['offset']:x['offset']+x['bytes']];assert hashlib.sha256(dat).hexdigest()==x['sha256'];weights[x['name']]=np.frombuffer(dat,dtype='<f4').reshape(x['shape'])
 w=weights['backbone.out_linear2.linear.weight'].astype(np.float64);b=weights['backbone.out_linear2.linear.bias'].astype(np.float64)
 report={'scope':'saved-only algebra; no new qualification; original FAIL preserved; D90 NOT_RUN','sources':{str(q.relative_to(a.archive)):sha(q) for q in paths+[mp,bp]},'counts':t['counts'].tolist(),'parts':[]}
 reconstructed={name:np.zeros((128,11,4),np.float32) for name in ('C','Torch')}
 rowoffset=0
 for part,n in enumerate(t['counts']):
  n=int(n);r={'part':part,'stages':{},'cache_layers':{},'final_affine':[],'cache_checks':[]}
  for k in ['cmvn'+str(part)]+[f'stage{part}_{i}' for i in range(21)]+['cache'+str(part),'logits'+str(part),'probabilities'+str(part)]:r['stages'][k]=diff(c[k],t[k])
  for row in range(n):
   for name,data in [('C',c),('Torch',t)]:
    cache=reconstructed[name]
    if name=='C':assert bitexact(cache,c[f'cache_before_{part}'][row])
    cache[:,:-1,:]=cache[:,1:,:].copy()
    for layer in range(4):cache[:,-1,layer]=data[f'stage{part}_{3+4*layer}'][row]
    if name=='C':assert bitexact(cache,c[f'cache_after_{part}'][row])
   xt=t[f'stage{part}_19'][row].astype(np.float64);xc=c[f'stage{part}_19'][row].astype(np.float64)
   ct=(xt[None,:]*w).sum(axis=1)+b;cc=(xc[None,:]*w).sum(axis=1)+b
   yt=t[f'logits{part}'][row].astype(np.float64);yc=c[f'logits{part}'][row].astype(np.float64)
   observed=yc-yt;input_term=cc-ct;c_res=yc-cc;t_res=yt-ct
   assert np.max(np.abs(observed-(input_term+c_res-t_res)))<1e-12
   r['final_affine'].append({'global_row':rowoffset+row,'observed_C_minus_Torch':observed.tolist(),'saved_input_center_difference':input_term.tolist(),'C_local_residual':c_res.tolist(),'Torch_local_residual':t_res.tolist()})
  for name,data in [('C',c),('Torch',t)]:
   assert bitexact(reconstructed[name],data[f'cache{part}'][0]);r['cache_checks'].append({'backend':name,'saved_chunk_end_matches_saved_projection_shift_register':True,'actual_per_row_cache_recorded':name=='C'})
  for layer in range(4):r['cache_layers'][str(layer)]=diff(c[f'cache{part}'][0,:,:,layer],t[f'cache{part}'][0,:,:,layer])
  rowoffset+=n;report['parts'].append(r)
 report['frontend_input']=diff(c['input'],t['input']);report['interpretation']='Cache mismatch is exactly carried saved projection values, not evidence of a cache shift/index bug. Most failing final-logit difference is already present in distinct saved stage19 inputs. Upstream numerical accumulation/input differences remain plausible; saved observations do not establish a unique causal root.'
 a.output.open('x').write(json.dumps(report,indent=2)+'\n');print(a.output)
if __name__=='__main__':main()
