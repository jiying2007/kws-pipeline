"""Public stdlib integrity/receipt/token verification. No model or CTC execution."""
from pathlib import Path
import hashlib
import io
import json
import math
import zipfile
HERE = Path(__file__).resolve().parent
RESULT = HERE / 'result-20261016'
FLAGS = ('human_gold', 'independent_evaluation', 'production_qualified', 'reusable_weights_exported', 'shipping_qualified', 'training_admitted')
def need(condition, label):
    if not condition: raise ValueError(label)
def exact(a, b):
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a == b
def strict_json(raw):
    def pairs(items):
        out = {}
        for k,v in items:
            need(k not in out, 'duplicate JSON key'); out[k] = v
        return out
    def bad(value): raise ValueError('nonfinite JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def load(path): return strict_json(path.read_bytes())
def verify_consumed(c):
    for k,v in {'consumed':True,'execution_enabled':False,'retry_authorized':False,'remaining_budget':0,'remaining_model_forward_budget':0,'budget_reusable':False,'execution_status':'completed','evidence_validation':'PASS','scientific_validation':'DESCRIPTIVE_ONLY','optimizer_calls_reserved':1200,'successful_optimizer_steps':1200,'model_forward_calls_reserved':1218,'model_forward_calls_returned':1218,'observations':18,'primary_endpoint_step':600,**{f:False for f in FLAGS}}.items(): need(exact(c.get(k),v), 'closure '+k)
def reconstruct(directory=RESULT):
    m=load(directory/'parts-manifest.json'); pieces=[]
    need(type(m['archive_bytes']) is int and 0<m['archive_bytes']<100*1024*1024,'archive bound')
    need(len(m['parts'])==66,'part count')
    for i,p in enumerate(m['parts']):
        need(p['path']==f'raw-result.zip.part{i:03d}','part path/order')
        need(type(p['bytes']) is int and 0<p['bytes']<=1048576,'part bound')
        f=directory/p['path'];need(not f.is_symlink(),'part symlink'); b=f.read_bytes()
        need(len(b)==p['bytes'] and sha(b)==p['sha256'],'part identity');pieces.append(b)
    raw=b''.join(pieces);need(len(raw)==m['archive_bytes'] and sha(raw)==m['archive_sha256'],'archive identity')
    return raw
def statistics(row):
    counts={};prefix=True
    for s in row['samples']:
        argmax=[max(range(len(frame)),key=lambda i:frame[i]) for frame in s['log_probs']]
        need(exact(argmax,s['raw_argmax_tokens']),'raw argmax')
        decoded=[];last=None
        for token in argmax:
            if token!=last and token!=0:decoded.append(token)
            last=token
        need(exact(decoded,s['decode_tokens']),'greedy collapse')
        complete=decoded==s['target_tokens'];need(exact(complete,s['complete_match']),'complete match')
        n=sum(t!=0 for t in argmax[:s['prefix_frames']]);need(exact(n,s['prefix_nonblank_argmax_frames']),'prefix count');prefix=prefix and n==0
        c=counts.setdefault(s['target_class'],{'complete':0,'total':0});c['complete']+=int(complete);c['total']+=1
    rates={k:c['complete']/c['total'] for k,c in counts.items()};total=sum(c['total'] for c in counts.values());complete=sum(c['complete'] for c in counts.values())
    return {'fit_success':complete==total,'prefix_success':prefix,'macro_complete_rate':sum(rates.values())/len(rates),'per_class_complete_rate':rates,'per_class_counts':counts,'complete':complete,'total':total,**{k:row[k] for k in ('finite_loss','ctc_loss','anchor_loss')}}
def verify():
    c=load(HERE/'closure.json');verify_consumed(c)
    enabled=load(RESULT/'enabled-plan.json');closed=load(HERE/'plan.json')
    need(enabled['execution_enabled'] is True and closed['execution_enabled'] is False,'plan flags')
    enabled['execution_enabled']=False;need(exact(enabled,closed),'only enable flag may change')
    for path,key in [(RESULT/'enabled-plan.json','enabled_plan_sha256'),(HERE/'plan.json','closed_plan_sha256'),(RESULT/'result-hash-manifest.json','result_manifest_sha256')]:need(sha(path.read_bytes())==c[key],'closure identity')
    for manifest,root in [(load(HERE/'source-manifest.json'),HERE),(load(RESULT/'archive-manifest.json'),RESULT)]:
        for name,digest in manifest['files'].items():
            need(not Path(name).is_absolute() and '..' not in Path(name).parts and all(not (root/Path(*Path(name).parts[:i])).is_symlink() for i in range(1,len(Path(name).parts)+1)), 'manifest safe relative path');need(sha((root/name).read_bytes())==digest,'manifest hash '+name)
    hashes=load(RESULT/'result-hash-manifest.json')['result_files'];need(len(hashes)==6077,'raw file count')
    acceptance=load(RESULT/'acceptance-summary.json');curve=load(RESULT/'learning-curve.json')['observations'];public=load(RESULT/'public-summary.json')
    verify_consumed(public)
    need(public['scientific_scope']=='exposed machine-labelled preset-group-disjoint diagnostic only','scientific scope')
    need(public['public_verification_scope']=='archive integrity, receipts and saved-token statistics; not independent model or CTC recomputation or process authenticity','verification scope')
    need(len(public['limitations'])>=6,'limitations');need(public['probe_used_for_selection'] is False,'probe selection')
    with zipfile.ZipFile(io.BytesIO(reconstruct())) as z:
        names=z.namelist();expected={'grouped-balance-result/'+n for n in hashes}|{'grouped-balance-result-acceptance/summary.json','grouped-balance-result-acceptance/hash-manifest.json'}
        need(len(names)==len(set(names)) and set(names)==expected,'archive roster')
        need(sum(i.file_size for i in z.infolist())<1024**3,'unpacked bound')
        for i in z.infolist():need(i.file_size<1024*1024 and not i.is_dir(),'member bound')
        for name,digest in hashes.items():need(sha(z.read('grouped-balance-result/'+name))==digest,'result hash '+name)
        for name,out in [('summary.json','acceptance-summary.json'),('hash-manifest.json','result-hash-manifest.json')]:need(z.read('grouped-balance-result-acceptance/'+name)==(RESULT/out).read_bytes(),'original acceptance bytes')
        def raw(name):return strict_json(z.read('grouped-balance-result/'+name+'.json'))
        for i in range(1,1201):
            stage='A' if i<=600 else 'B';step=(i-1)%600+1
            call=raw(f'call-{i:04d}');done=raw(f'step-{i:04d}')
            for row in [call,done]:need(exact(row['stage'],stage) and exact(row['update'],step),'optimizer order')
            need(exact(call['optimizer_call_reserved'],i) and call['status']=='step_reserved','reservation')
            need(done['status']=='step_succeeded' and done['updated_weights_finite'] is True and done['weights_changed'] is True,'step success')
            need(sha(z.read('grouped-balance-result/'+done['payload_file']))==done['payload_sha256'],'step payload')
        schedule=[]
        for arm in ['A','B']:
            schedule.append(('observation',arm,0))
            for step in range(1,601):
                schedule.append(('training',arm,step))
                if step in [50,100,200,300,400,500,600]:schedule.append(('observation',arm,step))
        schedule.extend([('probe','A',600),('probe','B',600)])
        for i,(kind,arm,step) in enumerate(schedule,1):
            for pre,status in [('forward','forward_reserved'),('forward-return','forward_returned')]:
                row=raw(f'{pre}-{i:04d}');need(exact([row['kind'],row['stage'],row['update'],row['model_forward_reserved'],row['status']],[kind,arm,step,i,status]),'forward sequence')
        need(set(curve)==set(acceptance['statistics']),'observation roster')
        for name in curve:
            stats=statistics(raw(name));need(exact(stats,curve[name]),'learning curve '+name)
            need(exact({k:stats[k] for k in acceptance['statistics'][name]},acceptance['statistics'][name]),'acceptance statistics')
        for name in ['A-0600','B-0600','probe-A-0600','probe-B-0600']:need(exact(curve[name],public['endpoints'][name]),'endpoint projection')
        need(exact(public['fixed_endpoint_B_minus_A'],acceptance['fixed_endpoint_B_minus_A']),'delta projection')
        need(raw('terminal')['status']=='completed','terminal')
    return {'result_files':6077,'observations':18,'optimizer_calls':1200,'model_forward_calls':1218,'scientific_validation':'DESCRIPTIVE_ONLY'}
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
