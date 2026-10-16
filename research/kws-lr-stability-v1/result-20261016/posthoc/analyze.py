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

MANIFESTS={'historical':'4260c936d8249c59f2d5b270d40274b5f2b29bb45d930dd31c338b240d666b25','new':'e96f9f8d7fdd9a55126ee8a61bf084024daace069328be39dcdea0d4d8649940'}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('historical-result','new-result','historical-manifest','new-manifest','output'):p.add_argument('--'+name,type=pathlib.Path,required=True)
 args=p.parse_args();roots={'historical':args.historical_result,'new':args.new_result}
 if any(args.output.resolve()==r.resolve() or r.resolve() in args.output.resolve().parents for r in roots.values()):raise ValueError('output must be outside raw evidence')
 manifests={}
 for label,path in [('historical',args.historical_manifest),('new',args.new_manifest)]:
  raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=MANIFESTS[label]:raise ValueError('manifest identity mismatch')
  manifests[label]=json.loads(raw)
 validate_beam()
 report={'status':'POSTHOC_DESCRIPTIVE_ONLY','script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'accepted_manifest_sha256':MANIFESTS,'beam_widths':[32,64],'top_k':1,'new_model_forwards':0,'new_optimizer_steps':0,'input_hashes':{},'observations':{},'curves':{},'beam_fixture':'four frames, exhaustive625 paths; unchanged reviewed prefix recurrence','limitations':['Finite beam is not global MAP proof.','No checkpoint or decoder selection; original fixed600 rawgreedy endpoint unchanged.','Only12 exposed machine-labelled positives; no generalization or false-activation evidence.','Same-seed historical contrast, not independent replicated effect.']}
 def read(label,name):
  raw=(roots[label]/name).read_bytes();h=hashlib.sha256(raw).hexdigest()
  if h!=manifests[label]['result_files'].get(name):raise ValueError('input identity mismatch '+label+'/'+name)
  report['input_hashes'][label+'/'+name]=h;return json.loads(raw)
 for name,label,arm,offset in [('historical_A','historical','A',0),('historical_B','historical','B',600),('new_A','new','A',0)]:
  steps=[read(label,f'step-{offset+u:04d}.json') for u in range(1,601)];late=steps[500:];jumps=[{'update':steps[i]['update'],'delta':steps[i]['finite_loss']-steps[i-1]['finite_loss']} for i in range(500,600)]
  report['curves'][name]={'last100_preupdate_loss_min':min(x['finite_loss'] for x in late),'last100_preupdate_loss_max':max(x['finite_loss'] for x in late),'last100_max_preclip_grad':max(x['finite_grad_norm'] for x in late),'last100_clip_trigger_updates':[x['update'] for x in late if x['finite_grad_norm']>5],'last100_largest_loss_jump':max(jumps,key=lambda x:x['delta']),'last100_positive_loss_jump_count':sum(x['delta']>0 for x in jumps),'last100_first_preupdate_loss':late[0]['finite_loss'],'last100_last_preupdate_loss':late[-1]['finite_loss']}
 for name,label,arm,step in [('historical_A500','historical','A',500),('historical_A600','historical','A',600),('historical_B600','historical','B',600),('new_A600','new','A',600)]:
  o=read(label,f'{arm}-{step:04d}.json');rows=[]
  for r in o['samples']:
   lp=r['log_probs'];target=r['target_tokens'];target_nll=v.ctc_nll(lp,target);short_nll=v.ctc_nll(lp,[3,4]);row={'record_id':r['record_id'],'class':r['target_class'],'target':target,'greedy':r['decode_tokens'],'greedy_correct':r['complete_match'],'target_nll':target_nll,'short_34_nll':short_nll,'log_probability_full_over_short':short_nll-target_nll if r['target_class']=='KW2' else None}
   for width in (32,64):
    dec,score=beam(lp,width);row['beam'+str(width)]={'tokens':dec,'correct':dec==target,'pruned_log_probability':score,'exact_sequence_nll':v.ctc_nll(lp,dec)}
   rows.append(row)
  report['observations'][name]={'ctc_loss':o['ctc_loss'],'anchor_loss':o['anchor_loss'],'finite_loss':o['finite_loss'],'prefix_success':all(r['prefix_nonblank_argmax_frames']==0 for r in o['samples']),'greedy_class_counts':{c:sum(r['greedy_correct'] for r in rows if r['class']==c) for c in ('KW1','KW2')},'beam_class_counts':{str(w):{c:sum(r['beam'+str(w)]['correct'] for r in rows if r['class']==c) for c in ('KW1','KW2')} for w in (32,64)},'kw2_full_probability_exceeds_short_count':sum(r['log_probability_full_over_short']>0 for r in rows if r['class']=='KW2'),'rows':rows}
 oldstep=read('historical','step-0501.json');newstep=read('new','step-0501.json');oldlp=read('historical',oldstep['payload_file']);newlp=read('new',newstep['payload_file']);report['update501_preupdate_payload_equal_except_schema']={k:x for k,x in oldlp.items() if k!='schema'}=={k:x for k,x in newlp.items() if k!='schema'};report['update501_preclip_gradient_equal']=oldstep['finite_grad_norm']==newstep['finite_grad_norm']
 with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 for name,o in report['observations'].items():print(name,o['finite_loss'],o['beam_class_counts'],o['kw2_full_probability_exceeds_short_count'])
if __name__=='__main__':main()
