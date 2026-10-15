#!/usr/bin/env python3
"""Post-hoc descriptive analysis of saved probabilities; no model imports or forwards."""
import json,math
from pathlib import Path
import argparse,hashlib
TOLERANCE=2e-6
FRAMES={'KW1':71,'KW2':83}
PREFIX={'KW1':23,'KW2':10}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def logsumexp(values):
 top=max(values)
 return top if top==-math.inf else top+math.log(math.fsum(math.exp(x-top) for x in values))
def ctc_nll(lp,target):
 ext=[0]
 for token in target:ext.extend((token,0))
 prev=[0.]+[-math.inf]*(len(ext)-1)
 for row in lp:
  nxt=[]
  for i,token in enumerate(ext):
   paths=[prev[i]]
   if i:paths.append(prev[i-1])
   if i>1 and token and token!=ext[i-2]:paths.append(prev[i-2])
   nxt.append(logsumexp(paths)+row[token])
  prev=nxt
 return -logsumexp(prev[-2:] if target else prev)
def check_log_probs(lp,frames):
 if type(lp) is not list or len(lp)!=frames:raise ValueError('frame shape')
 for row in lp:
  if type(row) is not list or len(row)!=5 or any(type(x) not in (float,int) or not math.isfinite(x) for x in row):raise ValueError('finite five-value row required')
  if abs(logsumexp(row))>TOLERANCE:raise ValueError('normalization')
 return [max(range(5),key=lambda j:r[j]) for r in lp]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--result-dir',type=Path,required=True)
p.add_argument('--hash-manifest',type=Path,required=True)
a=p.parse_args()
MANIFEST_SHA='21ad25ced1706f3dc93c2be08d7238a28bb09b6ab0a4d5ad809914a6a97bfede'
manifest_raw=a.hash_manifest.read_bytes()
if sha(manifest_raw)!=MANIFEST_SHA:raise ValueError('frozen result manifest mismatch')
manifest=json.loads(manifest_raw)
out={'schema':'prefix-anchor-posthoc-saved-logprob-analysis-v1','posthoc_descriptive_only':True,'input_hash_manifest_sha256':MANIFEST_SHA,'changes_primary_endpoint':False,'model_or_optimizer_executed':False,'token_names':{'0':'blank','1':'ni','2':'hao','3':'xiao','4':'wo'},'frame_time_definition':'zero-based frame j: analysis window [20*j,20*j+25] milliseconds; not forced phonetic boundaries','checkpoints':[]}
for stage in ('C','T'):
 for update in (0,50,100,200,300):
  name=f'{stage}-{update:04d}.json';raw=(a.result_dir/name).read_bytes()
  if sha(raw)!=manifest['result_files'][name]:raise ValueError('observation hash mismatch')
  d=json.loads(raw)
  for r in d['samples']:
   lp=r['log_probs'];path=check_log_probs(lp,FRAMES[r['target_class']]);nll={k:ctc_nll(lp,t) for k,t in {'KW1':[1,2,3,4],'KW2':[3,4,3,4],'empty':[],'missing_ni':[2,3,4]}.items()}
   runs=[]
   for i,t in enumerate(path):
    if not runs or runs[-1]['token']!=t:runs.append(dict(token=t,start_frame=i,end_frame=i,start_ms=20*i,last_window_end_ms=20*i+25))
    else:runs[-1].update(end_frame=i,last_window_end_ms=20*i+25)
   ni_i=max(range(len(lp)),key=lambda i:lp[i][1]);rival=max((j for j in range(5) if j!=1),key=lambda j:lp[ni_i][j])
   row=dict(stage=stage,update=update,target_class=r['target_class'],source_sha256=sha(raw),decode=r['decode_tokens'],raw_argmax_tokens=path,prefix_nonblank_argmax_frames=r['prefix_nonblank_argmax_frames'],ctc_nll_float64=nll,target_vs_other_nll_margin=nll['KW2' if r['target_class']=='KW1' else 'KW1']-nll[r['target_class']],token_runs=runs,ni_max_frame=ni_i,ni_max_probability=math.exp(lp[ni_i][1]),ni_argmax_frames=path.count(1),ni_max_frame_top_competitor=rival,ni_max_frame_top_competitor_probability=math.exp(lp[ni_i][rival]))
   if stage=='T' and update==300 and r['target_class']=='KW1':
    row['ni_peak_frame_all_probabilities']=[math.exp(x) for x in lp[ni_i]]
    row['missing_ni_probability_over_full_target']=math.exp(nll['KW1']-nll['missing_ni'])
    forced=[[row[0]]+[-math.inf]*4 if i<PREFIX['KW1'] else row[:] for i,row in enumerate(lp)]
    constrained_nll=ctc_nll(forced,[1,2,3,4])
    row['posthoc_prefix_forced_blank_target_nll']=constrained_nll
    row['target_alignment_mass_with_all_prefix_frames_blank_fraction']=math.exp(nll['KW1']-constrained_nll)
    row['conditioning_note']='This ratio conditions on the correct target sequence. It is not an unconditional error probability. Prefix forcing is analytical path restriction only, never modified inference or a new experimental endpoint.'
   out['checkpoints'].append(row)
print(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
