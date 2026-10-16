"""Post-hoc saved-evidence analysis only; no model/runtime imports or source edits."""
import argparse,json,math,itertools,pathlib,hashlib
# Log-domain CTC recurrence independently reproduced from frozen verifier cf974939751107b8e6e6c15d34bff1e5830e173a1a45cb3c34d1642681d36bcc.
class DP:
 @staticmethod
 def logsumexp(xs):
  m=max(xs)
  return m if m==-math.inf else m+math.log(math.fsum(math.exp(x-m) for x in xs))
 @staticmethod
 def collapse(path):
  out=[];old=None
  for x in path:
   if x and x!=old:out.append(x)
   old=x
  return out
 @staticmethod
 def ctc_nll(lp,target):
  ext=[0]
  for t in target:ext.extend((t,0))
  prev=[0.]+[-math.inf]*(len(ext)-1)
  for row in lp:
   new=[]
   for j,t in enumerate(ext):
    paths=[prev[j]]
    if j:paths.append(prev[j-1])
    if j>1 and t and t!=ext[j-2]:paths.append(prev[j-2])
    new.append(DP.logsumexp(paths)+row[t])
   prev=new
  return -DP.logsumexp(prev[-2:] if target else prev)
v=DP
def la(*xs):return v.logsumexp(xs)
def beam(lp,width):
 d={():[0.,-math.inf]}
 for row in lp:
  nxt={}
  def add(pre,ix,x):
   val=nxt.setdefault(pre,[-math.inf,-math.inf]);val[ix]=la(val[ix],x)
  for pre,(pb,pn) in d.items():
   total=la(pb,pn);add(pre,0,total+row[0])
   for c in range(1,5):
    if pre and pre[-1]==c:
     add(pre,1,pn+row[c]);add(pre+(c,),1,pb+row[c])
    else:add(pre+(c,),1,total+row[c])
  d=dict(sorted(nxt.items(),key=lambda kv:(-la(*kv[1]),kv[0]))[:width])
 pre,(pb,pn)=min(d.items(),key=lambda kv:(-la(*kv[1]),kv[0]));return list(pre),la(pb,pn)
def validate_beam():
 rows=[[math.log(x) for x in r] for r in [[.3,.2,.1,.25,.15],[.4,.1,.2,.2,.1],[.2,.3,.2,.1,.2],[.2,.1,.2,.2,.3]]];probs={}
 for path in itertools.product(range(5),repeat=4):
  p=tuple(v.collapse(list(path)));probs[p]=probs.get(p,0)+math.exp(sum(rows[i][t] for i,t in enumerate(path)))
 best=min(probs,key=lambda x:(-probs[x],x));actual,logp=beam(rows,625)
 if list(best)!=actual or abs(logp-math.log(probs[best]))>1e-12:raise RuntimeError('beam brute force disagreement')

MANIFEST_SHA='4260c936d8249c59f2d5b270d40274b5f2b29bb45d930dd31c338b240d666b25'

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--result",type=pathlib.Path,required=True)
 parser.add_argument("--accepted-hash-manifest",type=pathlib.Path,required=True)
 parser.add_argument("--output",type=pathlib.Path,required=True)
 args=parser.parse_args();R=args.result;OUTPUT=args.output
 if OUTPUT.resolve()==R.resolve() or R.resolve() in OUTPUT.resolve().parents:raise ValueError("Output must be outside original evidence")
 raw=args.accepted_hash_manifest.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:raise ValueError("Wrong accepted evidence manifest")
 accepted=json.loads(raw)
 validate_beam();report={'analysis_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'accepted_hash_manifest_sha256':MANIFEST_SHA,'verifier_sha256':accepted['verifier_sha256'],'beam_widths':[32,64],'top_k':1,'status':'POSTHOC_DESCRIPTIVE_ONLY','new_model_forwards':0,'new_optimizer_steps':0,'scope':'saved training data only, no probe decoder selection','beam_validation':'4-frame exhaustive 625 raw paths agrees with unpruned prefix recurrence','observations':{},'step_summaries':{},'input_hashes':{}}
 def read(name):
  raw=(R/name).read_bytes();digest=hashlib.sha256(raw).hexdigest()
  if digest!=accepted['result_files'].get(name):raise ValueError('saved evidence hash mismatch: '+name)
  report['input_hashes'][name]=digest;return json.loads(raw)
 for a in ['A','B']:
  steps=[read(f'step-{(0 if a=="A" else 600)+u:04d}.json') for u in range(1,601)]
  report['step_summaries'][a]={'max_grad':max(steps,key=lambda x:x['finite_grad_norm']),'clip_trigger_updates':[x['update'] for x in steps if x['finite_grad_norm']>5],'last100_min_loss':min(x['finite_loss'] for x in steps[500:]),'last_pre_step_loss':steps[-1]['finite_loss'],'largest_positive_loss_jumps':sorted([{'update':steps[i]['update'],'increase':steps[i]['finite_loss']-steps[i-1]['finite_loss'],'loss':steps[i]['finite_loss'],'grad':steps[i]['finite_grad_norm']} for i in range(1,600)],key=lambda x:x['increase'],reverse=True)[:5]}
  for u in [0,50,100,200,300,400,500,600]:
   o=read(f'{a}-{u:04d}.json');rows=[]
   for r in o['samples']:
    lp=r['log_probs'];target=r['target_tokens'];gt=v.ctc_nll(lp,target);gg=v.ctc_nll(lp,r['decode_tokens']);short=v.ctc_nll(lp,[3,4]);rec={'id':r['record_id'],'class':r['target_class'],'frames':r['input_frames'],'prefix':r['prefix_frames'],'target':target,'greedy':r['decode_tokens'],'greedy_correct':r['complete_match'],'target_nll_float64':gt,'greedy_sequence_nll_float64':gg,'short_34_nll_float64':short,'target_minus_greedy_nll':gt-gg,'target_probability':math.exp(-gt)}
    if (a,u) in [('A',500),('A',600),('B',600)]:
     for width in [32,64]:
      dec,score=beam(lp,width);rec['beam'+str(width)]={'tokens':dec,'correct':dec==target,'pruned_log_probability':score,'exact_nll_for_returned_sequence':v.ctc_nll(lp,dec)}
    rows.append(rec)
   report['observations'][f'{a}{u}']={'ctc_loss':o['ctc_loss'],'anchor_loss':o['anchor_loss'],'per_class_greedy_rate':o['per_class_complete_rate'],'rows':rows}
 for u in [594,595,596,597,598,599,600]:
  step=read(f'step-{u:04d}.json');p=read(step['payload_file']);report.setdefault('A_instability',[]).append({'pre_update':u,'loss':p['finite_loss'],'grad':step['finite_grad_norm'],'sample_losses':[{'id':x['record_id'],'ctc_nll':x['ctc_nll'],'weighted_loss':x['weighted_total_loss'],'anchor_nll':x['anchor_nll']} for x in p['samples']]})
 with OUTPUT.open('x') as stream:stream.write(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 for key in ['A500','A600','B600']:
  rows=report['observations'][key]['rows'];print(key,{str(w):{c:sum(x['beam'+str(w)]['correct'] for x in rows if x['class']==c) for c in ['KW1','KW2']} for w in [32,64]})
 print('Saved post-hoc report; primary endpoints unchanged.')

if __name__=="__main__":main()
