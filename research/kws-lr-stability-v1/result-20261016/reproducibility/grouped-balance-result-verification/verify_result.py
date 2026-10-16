#!/usr/bin/env python3
"""Independent stdlib-only grouped-balance evidence validator. Never imports model code."""
import argparse, hashlib, json, math, os, re, stat
from pathlib import Path
MAX_BYTES=16*1024**2
TOLERANCES={'ctc_nll':{'absolute':1e-3,'relative':1e-5},'logsumexp_absolute':2e-6,'derived':{'absolute':1e-6,'relative':2e-6}}
DENIALS=dict.fromkeys(('training_admitted','production_qualified','shipping_qualified','reusable_weights_exported','human_gold','independent_evaluation'),False)
CAMPAIGN='kws-grouped-balance-20261015-v1'
CHECKPOINTS=[0,50,100,200,300,400,500,600]
TARGETS={'KW1':[1,2,3,4],'KW2':[3,4,3,4]}
class Invalid(ValueError): pass

def require(condition, code):
    if not condition: raise Invalid(code)

def sha(raw): return hashlib.sha256(raw).hexdigest()

def exact(a,b): return type(a) is type(b) and (set(a)==set(b) and all(exact(a[k],b[k]) for k in a) if isinstance(a,dict) else len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b)) if isinstance(a,list) else a==b)

def same(a,b,code): require(exact(a,b),code)

def integer(x, low=0, high=1500): require(type(x) is int and low<=x<=high,'integer_type_or_range'); return x

def number(x, low=0, high=float('inf')):
    require(type(x) in (int,float) and math.isfinite(x) and low<=x<=high,'number_type_or_range'); return x

def keys(x, expected): require(type(x) is dict and set(x)==set(expected),'object_fields')

def denials(x):
    for k,v in DENIALS.items(): same(x.get(k),v,'research_denials')

def digest(x): require(type(x) is str and re.fullmatch('[0-9a-f]{64}',x) is not None,'digest_format')

def strict_json(raw):
    def pairs(items):
        obj={}
        for k,v in items:
            require(k not in obj,'duplicate_json_key');obj[k]=v
        return obj
    def constant(_): raise Invalid('nonfinite_json_constant')
    try:
        value=json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_constant=constant)
        def finite_tree(x):
            if type(x) is float:require(math.isfinite(x),'nonfinite_json_number')
            elif type(x) is dict:
                for item in x.values():finite_tree(item)
            elif type(x) is list:
                for item in x:finite_tree(item)
        finite_tree(value);return value
    except (UnicodeError,json.JSONDecodeError):raise Invalid('invalid_json') from None

def safe_dir(path):
    path=Path(path).absolute()
    require('..' not in path.parts and path.resolve(strict=True)==path,'unsafe_path')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        return fd
    except BaseException:os.close(fd);raise

def safe_read(path):
    path=Path(path).absolute(); directory=safe_dir(path.parent)
    try:
        fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        with os.fdopen(fd,'rb') as f:
            before=os.fstat(f.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=MAX_BYTES,'unsafe_file')
            raw=f.read(MAX_BYTES+1);after=os.fstat(f.fileno())
            require(len(raw)<=MAX_BYTES and (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'file_changed')
            return raw
    finally:os.close(directory)

def pinned(path, expected):
    raw=safe_read(path);same(sha(raw),expected,'pinned_source_hash');return raw

def collapse(tokens):
    out=[]; previous=None
    for t in tokens:
        integer(t,0,4)
        if t and t!=previous:out.append(t)
        previous=t
    return out

def logsumexp(values):
    top=max(values)
    return top if top==-math.inf else top+math.log(math.fsum(math.exp(x-top) for x in values))

def ctc_nll(log_probs,target):
    """Independent log-domain forward recurrence, blank=0, repeated-label skip guard."""
    ext=[0]
    for token in target:integer(token,1,4);ext.extend((token,0))
    previous=[-math.inf]*len(ext);previous[0]=0.0
    for row in log_probs:
        current=[]
        for s,token in enumerate(ext):
            paths=[previous[s]]
            if s:paths.append(previous[s-1])
            if s>1 and token and token!=ext[s-2]:paths.append(previous[s-2])
            current.append(logsumexp(paths)+row[token])
        previous=current
    return -logsumexp(previous[-2:] if target else previous)

def close(actual,expected,code,ctc=False):
    number(actual,-1e100,1e100)
    t=TOLERANCES['ctc_nll' if ctc else 'derived']
    require(abs(actual-expected)<=t['absolute']+t['relative']*abs(expected),code)

def check_log_probs(rows,frames):
    require(type(rows) is list and len(rows)==frames,'log_probs_frames')
    for row in rows:
        require(type(row) is list and len(row)==5,'log_probs_vocab')
        for value in row:number(value,-1e100,1e-6)
        require(abs(logsumexp(row))<=TOLERANCES['logsumexp_absolute'],'log_probs_normalization')
    return [max(range(5),key=lambda i:row[i]) for row in rows]

SAMPLE_KEYS={'record_id','target_class','input_frames','target_length','raw_argmax_tokens','target_tokens','decode_tokens','complete_match','ctc_nll','ctc_nll_per_input_frame','blank_argmax_fraction','blank_mean_probability','log_probs','ctc_nll_by_target','prefix_frames','prefix_nonblank_argmax_frames','prefix_blank_probability_min','prefix_blank_probability_max','anchor_nll','anchor_nll_per_input_frame','first_nonblank_frame','preset_voice_id','pcm_sha256','loss_weight','total_loss_per_input_frame','weighted_total_loss'}
def sample(row,meta,weight,training=False):
    keys(row,SAMPLE_KEYS);frames=integer(meta['input_frames'],1,10000);prefix=integer(meta['prefix_frames'],0,frames);target=TARGETS[meta['target_class']]
    for k in ('record_id','target_class','input_frames','prefix_frames','preset_voice_id','pcm_sha256'):same(row[k],meta[k],'sample_'+k)
    same(row['target_length'],4,'target_length');same(row['target_tokens'],target,'target_tokens')
    path=check_log_probs(row['log_probs'],frames);decoded=collapse(path)
    same(row['raw_argmax_tokens'],path,'argmax');same(row['decode_tokens'],decoded,'collapse');same(row['complete_match'],decoded==target,'complete_match')
    same(row['prefix_nonblank_argmax_frames'],sum(t!=0 for t in path[:prefix]),'prefix_count')
    same(row['first_nonblank_frame'],next((i for i,t in enumerate(path) if t),None),'first_nonblank')
    close(row['blank_argmax_fraction'],path.count(0)/frames,'blank_fraction')
    blank=[math.exp(r[0]) for r in row['log_probs']];close(row['blank_mean_probability'],math.fsum(blank)/frames,'blank_mean')
    for k,fn in [('prefix_blank_probability_min',min),('prefix_blank_probability_max',max)]:
        if prefix:close(row[k],fn(blank[:prefix]),k)
        else:same(row[k],None,'empty_prefix_statistic')
    keys(row['ctc_nll_by_target'],{meta['target_class']} if training else {'KW1','KW2','empty'})
    computed={label:ctc_nll(row['log_probs'],tokens) for label,tokens in {**TARGETS,'empty':[]}.items()}
    for label,nll in row['ctc_nll_by_target'].items():close(nll,computed[label],'ctc_'+label,True)
    close(computed['empty'],-math.fsum(r[0] for r in row['log_probs']),'empty_direct',True)
    close(row['ctc_nll'],computed[meta['target_class']],'target_ctc',True)
    close(row['ctc_nll_per_input_frame'],row['ctc_nll']/frames,'ctc_normalization')
    anchor=-math.fsum(r[0] for r in row['log_probs'][:prefix])
    close(row['anchor_nll'],anchor,'anchor_nll');close(row['anchor_nll_per_input_frame'],anchor/frames,'anchor_normalization')
    close(row['loss_weight'],weight,'sample_weight')
    total=(row['ctc_nll']+anchor)/frames
    close(row['total_loss_per_input_frame'],total,'total_loss');close(row['weighted_total_loss'],total*weight,'weighted_loss')
    return dict(record_id=meta['record_id'],target_class=meta['target_class'],complete_match=decoded==target,prefix_nonblank_argmax_frames=sum(t!=0 for t in path[:prefix]),decode_tokens=decoded,joint_success=decoded==target and all(t==0 for t in path[:prefix]))

def rates(summaries):
    per_class={c:sum(r['complete_match'] for r in summaries if r['target_class']==c)/sum(r['target_class']==c for r in summaries) for c in TARGETS}
    return per_class,math.fsum(per_class.values())/2

def forward_events():
    events=[]
    for arm in ('A','B'):
        events.append((arm,'observation',0))
        for u in range(1,601):
            events.append((arm,'training',u))
            if u in CHECKPOINTS:events.append((arm,'observation',u))
    return events+[('A','probe',600),('B','probe',600)]

FIXED_PLAN={'schema': 'kws-grouped-balance-plan-v1', 'purpose': 'exposed-preset-group-disjoint-fixed-horizon-diagnostic', 'production_qualified': False, 'seed': 20261015, 'model': {'feature_dim': 32, 'hidden_dim': 48, 'vocab_size': 5}, 'targets_by_text': {'你好小窝': [1, 2, 3, 4], '小窝小窝': [3, 4, 3, 4]}, 'optimizer': {'name': 'Adam', 'lr': 0.003, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'eps': 1e-08, 'gradient_clip': 5.0}, 'ctc': {'blank': 0, 'reduction': 'none', 'zero_infinity': False, 'normalization': 'per-sample-input-frame-count'}, 'max_optimizer_calls': 1200, 'retries': 0, 'new_tts_calls': 0, 'new_asr_calls': 0, 'warm_start': False, 'augmentation': False, 'vad': False, 'export_weights': False, 'registry_sha256': '28033edbbc1e444eae27c33af2e060d1654c4b7a072e65401b055fc27fc3b1b0', 'source_verifier_sha256': '9d37fb852cdd5048a4d5c88f9e21efe2b3c1e18f25244babb2d811675ea9fbae', 'runtime_lock_sha256': 'b132c2c202e5ff00b6cbd1b2b630fa92dfe9ba46fcf0423dc0633381bbe80984', 'code_sha256': {'training/model.py': '768800e25537367da770ef44178249208956d7c3c33f1eb7d60b1b9c98fe59c5', 'training/frontend.py': 'd8ae687303d5c9c6bfdf6539c94f4ae463d910a4837ed77633d7656ff02a582e', 'training/frontend_spec.py': 'fbd17236e1449101bed6df8f105baad476edb0b005e6e45cb83ebc7971d166e6'}, 'campaign_id': 'kws-grouped-balance-20261015-v1', 'campaign_claim_path': 'research-budget/kws-grouped-balance-20261015-v1.claim.json', 'input_pins_sha256': '64c77a270c7f13d8252d10282191aaeb207f425c10fb89a2f01e390fa994e37c', 'frontend': {'frontend': 'logmel', 'feature_dim': 32, 'frame_len': 400, 'hop': 320}, 'stages': [{'name': 'A', 'updates': 600, 'checkpoints': [0, 50, 100, 200, 300, 400, 500, 600], 'blank_lambda': 1}, {'name': 'B', 'updates': 600, 'checkpoints': [0, 50, 100, 200, 300, 400, 500, 600], 'blank_lambda': 1}], 'observation_timing': 'post-step; step 0 before any optimizer call', 'loss_timing': 'pre-step', 'max_model_forward_calls': 1218, 'cohort_sha256': 'fe23e633925573e2b02ac8530e066214879a11cbccc2e4ef5875aa2c38006e87', 'prefix_rule': {'window_samples': 400, 'hop_samples': 320, 'threshold': 0.0001, 'rule': 'maximal initial run of complete unwindowed PCM RMS strictly below threshold; zero allowed'}, 'probe_timing': 'after both arms finish 600 updates; A then B; once per arm', 'train_weights': {'A': {'KW1': 0.08333333333333333, 'KW2': 0.08333333333333333}, 'B': {'KW1': 0.125, 'KW2': 0.0625}}, 'early_stop': False, 'extension': False, 'resume': False, 'probe_selection': False, 'train_record_ids': ['kw1-preset-supplement-20261014-v1:kw1s-05', 'kw1-preset-supplement-20261014-v1:kw1s-06', 'kw1-preset-supplement-20261014-v1:kw1s-07', 'tts-quality-remaining-20261014-v1:tqp-qwen06-05', 'tts-quality-remaining-20261014-v1:tqp-qwen06-06', 'tts-quality-remaining-20261014-v1:tqp-qwen06-10', 'tts-quality-remaining-20261014-v1:tqp-qwen06-11', 'tts-quality-remaining-20261014-v1:tqp-qwen06-14', 'tts-quality-remaining-20261014-v1:tqp-qwen06-15', 'tts-quality-remaining-20261014-v1:tqp-qwen17-06', 'tts-quality-remaining-20261014-v1:tqp-qwen17-14', 'tts-quality-remaining-20261014-v1:tqp-qwen17-15'], 'probe_record_ids': ['tts-quality-remaining-20261014-v1:tqp-qwen06-03', 'tts-quality-remaining-20261014-v1:tqp-qwen17-01', 'tts-quality-remaining-20261014-v1:tqp-qwen17-02', 'tts-quality-remaining-20261014-v1:tqp-qwen17-03']}
def check_plan(plan):
    keys(plan,set(FIXED_PLAN)|{'execution_enabled','diagnostic_code_sha256'})
    for k,val in FIXED_PLAN.items():same(plan[k],val,'fixed_plan_'+k)
    same(plan['execution_enabled'],True,'execution_enabled')
    keys(plan['diagnostic_code_sha256'],{'run_grouped_balance.py','contract.py','numeric_preflight.py','requirements.research.lock'})
    for h in plan['diagnostic_code_sha256'].values():digest(h)

def check_cohort(cohort,plan,registry):
    keys(cohort,{'schema','samples','input_archives'});same(cohort['input_archives'],registry['input_archives'],'cohort_archives')
    same(cohort['schema'],'grouped-cohort-prefix-v1','cohort_schema')
    rows=cohort['samples'];require(type(rows) is list and len(rows)==16,'cohort_count')
    ids=[r['record_id'] for r in rows];require(len(set(ids))==16,'duplicate_sample')
    byid={r['record_id']:r for r in registry['records']}
    for r in rows:
        keys(r,{'record_id','slot_id','target_class','target_text','preset_voice_id','pcm_sha256','frames','role','input_frames','prefix_frames','rms'})
        source=byid[r['record_id']];c=r['target_class'];require(c in TARGETS,'class')
        for k in ('slot_id','target_text'):same(r[k],source[k],'registry_'+k)
        same(r['preset_voice_id'],source['conservative_group_id'],'registry_group')
        same(r['pcm_sha256'],source['derived_pcm']['sha256'],'registry_pcm')
        same(r['frames'],source['derived_pcm']['frames'],'pcm_frames')
        same(r['target_class'],source['target_class'],'registry_class')
        frames=1+(integer(r['frames'],400,10000000)-400)//320
        same(r['input_frames'],frames,'frontend_frames')
        require(type(r['rms']) is list and len(r['rms'])==frames,'rms_count')
        for rms in r['rms']:number(rms,0,1)
        prefix=next((i for i,x in enumerate(r['rms']) if x>=1e-4),frames)
        same(r['prefix_frames'],prefix,'prefix_rule')
    train=[r for r in rows if r['role']=='train'];probe=[r for r in rows if r['role']=='probe']
    same([r['record_id'] for r in train],plan['train_record_ids'],'train_ids');same([r['record_id'] for r in probe],plan['probe_record_ids'],'probe_ids')
    for cohort_rows,expected_counts,groups in [(train,{'KW1':4,'KW2':8},{'qwen-preset:serena','qwen-preset:uncle_fu','qwen-preset:dylan'}),(probe,{'KW1':1,'KW2':3},{'qwen-preset:vivian'})]:
        same({c:sum(r['target_class']==c for r in cohort_rows) for c in TARGETS},expected_counts,'class_count')
        same({r['preset_voice_id'] for r in cohort_rows},groups,'group_split')
    require(not ({r['pcm_sha256'] for r in train}&{r['pcm_sha256'] for r in probe}),'pcm_leakage')
    return train,probe

def load_sources(repo,freeze):
    module=Path(repo)/'research/kws-grouped-balance-v1'
    plan=strict_json(pinned(module/'plan.json',freeze['plan_sha256']));check_plan(plan)
    for name,h in plan['diagnostic_code_sha256'].items():pinned(module/name,h)
    for name,h in plan['code_sha256'].items():pinned(Path(repo)/name,h)
    runtime=strict_json(pinned(module/'runtime-candidate.lock.json',plan['runtime_lock_sha256']))
    cohort=strict_json(pinned(module/'cohort-prefix-freeze.json',plan['cohort_sha256']))
    registry_dir=Path(repo)/'research/kws-candidate-registry-v1'
    registry=strict_json(pinned(registry_dir/'candidates.json',plan['registry_sha256']))
    pinned(registry_dir/'build_registry.py',plan['source_verifier_sha256'])
    inputs=strict_json(pinned(registry_dir/'input-pins.json',plan['input_pins_sha256']))
    same(inputs,registry['repository_inputs'],'registry_inputs')
    for name,h in inputs.items():pinned(Path(repo)/name,h)
    train,probe=check_cohort(cohort,plan,registry)
    return plan,train,probe,runtime

PAYLOAD_KEYS={'schema','kind','stage','update','role','timing','model_forward_reserved','optimizer_calls_reserved','samples','ctc_loss','anchor_loss','finite_loss','blank_lambda','per_class_complete_rate','macro_complete_rate'}|set(DENIALS)
STEP_KEYS={'status','stage','update','payload_file','payload_sha256','finite_loss','ctc_loss','anchor_loss','blank_lambda','finite_grad_norm','updated_weights_finite','weights_changed'}
ALLOWED_FILES={'intent.json','binding.json','terminal.json','failure.json','A-initialization.json','B-initialization.json','probe-A-0600.json','probe-B-0600.json'}|{f'{a}-{u:04d}.json' for a in ('A','B') for u in CHECKPOINTS}|{f'{k}-{i:04d}.json' for k in ('call','step') for i in range(1,1201)}|{f'{k}-{i:04d}.json' for k in ('forward','forward-return','payload') for i in range(1,1219)}

def payload(obj,arm,kind,u,idx,train,probe,cache):
    keys(obj,PAYLOAD_KEYS);denials(obj)
    prior=(0 if arm=='A' else 600)+u-(kind=='training') if kind!='probe' else 1200
    for k,val in dict(schema='grouped-forward-payload-v1',kind=kind,stage=arm,update=u,role='probe' if kind=='probe' else 'train',timing='pre-step' if kind=='training' else 'post-step' if u else 'initial',model_forward_reserved=idx,optimizer_calls_reserved=prior,blank_lambda=1).items():same(obj[k],val,'payload_'+k)
    metas=probe if kind=='probe' else train
    require(type(obj['samples']) is list and len(obj['samples'])==len(metas),'payload_samples')
    summaries=[];ctcs=[];anchors=[]
    for row,meta in zip(obj['samples'],metas):
        c=meta['target_class'];weight=(1/2 if c=='KW1' else 1/6) if kind=='probe' else 1/12 if arm=='A' else 1/8 if c=='KW1' else 1/16
        key=(id(row),meta['record_id'],weight,kind=='training')
        if key not in cache:cache[key]=sample(row,meta,weight,kind=='training')
        summaries.append(cache[key]);ctcs.append(row['ctc_nll']/meta['input_frames']*weight);anchors.append(row['anchor_nll']/meta['input_frames']*weight)
    close(obj['ctc_loss'],math.fsum(ctcs),'payload_ctc');close(obj['anchor_loss'],math.fsum(anchors),'payload_anchor');close(obj['finite_loss'],obj['ctc_loss']+obj['anchor_loss'],'payload_loss')
    per_class,macro=rates(summaries);same(obj['per_class_complete_rate'],per_class,'per_class_rate');same(obj['macro_complete_rate'],macro,'macro_rate')
    return dict(fit_success=all(r['complete_match'] for r in summaries),prefix_success=all(r['prefix_nonblank_argmax_frames']==0 for r in summaries),macro_complete_rate=macro,per_class_complete_rate=per_class)

def encoded(obj):return (json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()

def verify_objects(data,plan,train,probe,runtime,claim_raw,result_path,process,plan_sha,raws=None):
    require(type(data) is dict and set(data)<=ALLOWED_FILES,'unexpected_result_file')
    same(data.get('intent.json'),dict(schema='grouped-intent-v1',campaign_id=CAMPAIGN,max_optimizer_calls=1200,max_model_forward_calls=1218,no_resume=True,**DENIALS),'intent')
    same(strict_json(claim_raw),dict(schema='grouped-exclusive-claim-v1',campaign_id=CAMPAIGN,output=str(Path(result_path).absolute()),max_optimizer_calls=1200,max_model_forward_calls=1218,**DENIALS),'exclusive_claim')
    def indices(prefix):return sorted(int(n[len(prefix)+1:-5]) for n in data if re.fullmatch(prefix+r'-[0-9]{4}\.json',n))
    calls=indices('call');steps=indices('step');forwards=indices('forward');returns=indices('forward-return');payloads=indices('payload')
    for label,seq in [('call',calls),('step',steps),('forward',forwards),('forward_return',returns),('payload',payloads)]:same(seq,list(range(1,len(seq)+1)),label+'_gap')
    require(len(steps)<=len(calls)<=len(steps)+1,'reservation_success_gap')
    require(len(payloads)<=len(returns)<=len(forwards)<=len(returns)+1,'forward_return_reservation_gap')
    require(len(forwards)<=len(payloads)+1,'continued_after_missing_payload')
    observations={};covered=set();events=forward_events();training_index={};cache={}
    for i in forwards:
        arm,kind,u=events[i-1]
        expected=dict(status='forward_reserved',stage=arm,update=u,kind=kind,model_forward_reserved=i)
        same(data[f'forward-{i:04d}.json'],expected,'forward_entry')
        if i in returns:same(data[f'forward-return-{i:04d}.json'],{**expected,'status':'forward_returned'},'forward_return_entry')
        total=(0 if arm=='A' else 600)+u
        receipt=f'step-{total:04d}.json' if kind=='training' else f'{arm}-{u:04d}.json' if kind=='observation' else f'probe-{arm}-0600.json'
        if receipt not in data:require(i==len(forwards),'continued_after_unfinished_event')
        else:
            require(i in returns and i in payloads,'receipt_without_forward_return_or_payload');covered.add(receipt)
        if kind=='training':
            training_index[total]=i;require(total<=len(calls)+1,'forward_ahead_of_optimizer')
        elif kind=='observation':require(total<=len(steps),'observation_before_success')
        else:require(len(steps)==1200,'probe_before_training_end')
        if i in payloads:
            p=data[f'payload-{i:04d}.json'];keys(p,PAYLOAD_KEYS)
            # Cheap closed metadata validation precedes independent numerical pass.
            for k,val in dict(kind=kind,stage=arm,update=u,model_forward_reserved=i).items():same(p[k],val,'payload_position')
            if kind!='training' and receipt in data:same(data[receipt],p,'observation_payload');observations[receipt]=(i,arm,kind,u)
    for total in calls:
        arm='A' if total<=600 else 'B';u=(total-1)%600+1
        same(data[f'call-{total:04d}.json'],dict(status='step_reserved',stage=arm,update=u,optimizer_call_reserved=total),'call_entry')
        require(total in training_index and training_index[total] in returns and training_index[total] in payloads,'call_before_forward_return')
    for total in steps:
        name=f'step-{total:04d}.json';require(name in covered,'step_without_forward')
        s=data[name];keys(s,STEP_KEYS);arm='A' if total<=600 else 'B';u=(total-1)%600+1;pn=f'payload-{training_index[total]:04d}.json'
        for k,val in dict(status='step_succeeded',stage=arm,update=u,payload_file=pn,payload_sha256=sha(raws[pn] if raws else encoded(data[pn])),blank_lambda=1,updated_weights_finite=True,weights_changed=True).items():same(s[k],val,'step_'+k)
        for k in ('finite_loss','ctc_loss','anchor_loss'):same(s[k],data[pn][k],'step_payload_'+k)
        number(s['finite_grad_norm'])
    obs_names={n for n in data if re.fullmatch(r'(?:probe-)?[AB]-[0-9]{4}\.json',n)}
    same(obs_names,set(observations),'observation_without_forward')
    if 'binding.json' in data:
        b=data['binding.json'];digest(b.get('initial_weight_sha256'))
        expected=dict(schema='grouped-input-binding-v1',plan_sha256=plan_sha,initial_weight_sha256=b['initial_weight_sha256'],registry_sha256=plan['registry_sha256'],cohort_sha256=plan['cohort_sha256'],runtime_lock_sha256=plan['runtime_lock_sha256'],seed=20261015,threads={'intraop':1,'interop':1},campaign_id=CAMPAIGN,campaign_claim_path=plan['campaign_claim_path'],campaign_claim_sha256=sha(claim_raw),max_model_forward_calls=1218,**DENIALS)
        same(b,expected,'binding')
    for arm in ('A','B'):
        name=f'{arm}-initialization.json'
        if name in data:
            require('binding.json' in data,'initialization_without_binding')
            same(data[name],dict(initial_weight_sha256=data['binding.json']['initial_weight_sha256'],seed=20261015),'initialization')
            if arm=='B':require(len(steps)>=600 and 'A-0600.json' in data,'B_initialization_before_A_complete')
    if forwards:
        for arm in {events[i-1][0] for i in forwards}:require(f'{arm}-initialization.json' in data,'forward_without_initialization')
    if 'A-0000.json' in data and 'B-0000.json' in data:
        left=data['A-0000.json']['samples'];right=data['B-0000.json']['samples']
        require(len(left)==len(right),'initial_sample_count')
        for a,b in zip(left,right):
            same({k:val for k,val in a.items() if k not in ('loss_weight','weighted_total_loss')},{k:val for k,val in b.items() if k not in ('loss_weight','weighted_total_loss')},'initial_observation_exact_match')
    status='incomplete';science='NOT_EVALUATED'
    if 'terminal.json' in data:
        require(len(calls)==len(steps)==1200 and len(forwards)==len(returns)==len(payloads)==1218 and len(observations)==18 and 'failure.json' not in data,'terminal_completeness')
    if 'failure.json' in data:
        f=data['failure.json'];keys(f,{'schema','status','stage','update','optimizer_calls_reserved','model_forward_calls_reserved','error_type','context','error'}|set(DENIALS));denials(f)
        for k,val in dict(schema='grouped-failure-v1',status='failed_no_retry',optimizer_calls_reserved=len(calls),model_forward_calls_reserved=len(forwards),error='Stopped on numerical, evidence, budget or output invariant; no retry or resume').items():same(f[k],val,'failure_'+k)
        require(type(f['error_type']) is str and f['error_type'] and type(f['context']) is dict,'failure_types');integer(f['update'],0,600)
        if f['stage']=='preparation':require(not forwards and not calls and f['update']==0,'preparation_failure')
        else:
            require(f['stage'] in ('A','B'),'failure_stage')
            if forwards:
                arm,kind,u=events[len(forwards)-1]
                eligible_events=[(arm,kind,u)]
                last_total=(600 if arm=='B' else 0)+u
                last_receipt=f'step-{last_total:04d}.json' if kind=='training' else f'{arm}-{u:04d}.json' if kind=='observation' else f'probe-{arm}-0600.json'
                # A failed reservation may leave no new forward file. Only a
                # fully completed prior event permits the immediately next one.
                if last_receipt in covered and len(forwards)<len(events):eligible_events.append(events[len(forwards)])
                require((f['stage'],f['update']) in {(a,u) for a,k,u in eligible_events},'failure_position')
            else:
                eligible_events=[events[0]]
                same((f['stage'],f['update']),('A',0),'failure_initial_position')
        ctx=f['context'];phase=ctx.get('phase')
        if phase=='forward':
            keys(ctx,{'phase','kind','record_ids'});require(ctx['kind'] in ('training','observation','probe'),'failure_kind')
            require(f['stage']!='preparation' and (f['stage'],ctx['kind'],f['update']) in eligible_events,'failure_forward_event')
            same(ctx['record_ids'],[r['record_id'] for r in (probe if ctx['kind']=='probe' else train)],'failure_records')
        elif phase=='loss':
            keys(ctx,{'phase','nonfinite_log_probs','ctc_nll'});integer(ctx['nonfinite_log_probs'],0,1000000)
            require(type(ctx['ctc_nll']) is list and len(ctx['ctc_nll']) in (4,12),'failure_nlls')
            for n in ctx['ctc_nll']:
                if n is not None:number(n,-1e100,1e100)
        elif phase in ('backward','optimizer_step'):keys(ctx,{'phase'})
        else:same(ctx,{},'failure_context')
        status='failed_partial'
    # All metadata/ordering must pass before spending time on numerical replay.
    summaries={}
    for i in payloads:
        arm,kind,u=events[i-1];s=payload(data[f'payload-{i:04d}.json'],arm,kind,u,i,train,probe,cache)
        if kind!='training':summaries[f'{arm}-{u:04d}' if kind=='observation' else f'probe-{arm}-0600']=s
    if 'terminal.json' in data:
        train_endpoints={a:summaries[f'{a}-0600'] for a in ('A','B')};probe_endpoints={a:summaries[f'probe-{a}-0600'] for a in ('A','B')}
        same(data['terminal.json'],dict(schema='grouped-terminal-v1',status='completed',primary_endpoint_step=600,train_endpoints=train_endpoints,probe_endpoints=probe_endpoints,optimizer_calls_reserved=1200,model_forward_calls_reserved=1218,probe_used_for_selection=False,**DENIALS),'terminal')
        status='completed';science='DESCRIPTIVE_ONLY'
    require(type(process.get('exit_code')) is int and type(process.get('wall_timeout')) is bool,'process_types')
    if process['exit_code']!=0 or process['wall_timeout']:status='failed_partial' if 'failure.json' in data else 'incomplete';science='NOT_EVALUATED'
    return dict(schema='grouped-balance-independent-verification-v1',evidence_validation='PASS',scientific_validation=science,execution_status=status,optimizer_calls_reserved=len(calls),successful_step_receipts=len(steps),model_forward_calls_reserved=len(forwards),model_forward_calls_returned=len(returns),payloads_verified=len(payloads),observations_verified=len(observations),primary_endpoint_step=600,statistics=summaries,tolerances=TOLERANCES,**DENIALS)

def verify_outer(process_path, verified_source_commit, repo, freeze):
    PLAN_SHA=freeze["plan_sha256"];LAUNCHER_SHA=freeze["launcher_sha256"]
    require(type(verified_source_commit) is str and re.fullmatch('[0-9a-f]{40}',verified_source_commit) is not None,'source_commit_format')
    parent=Path(process_path).parent
    process=strict_json(safe_read(process_path))
    intent=strict_json(safe_read(parent/'launch-intent.json'))
    started=strict_json(safe_read(parent/'process-start.json'))
    keys(intent,{'schema','campaign_id','parent_verified_source_commit','plan_sha256','runner_sha256','launcher_sha256','command_sha256','max_optimizer_updates','new_tts_calls','new_asr_calls','retry','actual_optimizer_updates','wall_timeout_seconds','cpu_limit_seconds','virtual_address_space_limit_bytes','per_file_limit_bytes','cpu_affinity','started_at_utc','production_qualified','reusable_weights_exported','reviewed_source_pins','independent_new_budget','previous_campaign_budget_reused','claim_relative','core_dump_limit_bytes','max_model_forward_calls','runner_evidence_directory_limit_bytes'})
    expected=dict(schema='research-grouped-balance-local-launch-intent-v1',campaign_id=CAMPAIGN,parent_verified_source_commit=verified_source_commit,plan_sha256=PLAN_SHA,runner_sha256=freeze['runner_sha256'],launcher_sha256=LAUNCHER_SHA,max_optimizer_updates=1200,max_model_forward_calls=1218,new_tts_calls=0,new_asr_calls=0,retry=False,actual_optimizer_updates='UNKNOWN_UNTIL_RESULT_VERIFICATION',wall_timeout_seconds=3600,cpu_limit_seconds=3600,virtual_address_space_limit_bytes=8*1024**3,per_file_limit_bytes=16*1024**2,runner_evidence_directory_limit_bytes=512*1024**2,production_qualified=False,reusable_weights_exported=False,reviewed_source_pins=freeze['reviewed_source_pins'],independent_new_budget=True,previous_campaign_budget_reused=False,claim_relative='research-budget/kws-grouped-balance-20261015-v1.claim.json',core_dump_limit_bytes=0)
    for k,val in expected.items():same(intent[k],val,'launch_binding_'+k)
    digest(intent['command_sha256'])
    base=Path(repo).absolute().parent
    pinned(base/'launch_reviewed_grouped_balance.py',LAUNCHER_SHA)
    command=[str(base/'fit-venv/bin/python'),'-I','-B',str(base/'grouped-balance-integrated/research/kws-grouped-balance-v1/run_grouped_balance.py'),'--repo',str(base/'grouped-balance-integrated'),'--artifacts',str(base.parent/'recovery-kws-20261015/artifacts'),'--wheelhouse',str(base/'runtime-work/wheels'),'--output',str(base/'grouped-balance-result'),'--claim-root',str(base/'grouped-balance-campaign-ledger'),'--plan-sha256',PLAN_SHA]
    same(intent['command_sha256'],sha(json.dumps(command,separators=(',',':')).encode()),'fixed_launch_command')
    require(type(intent['cpu_affinity']) is list and len(intent['cpu_affinity'])==1,'cpu_affinity');integer(intent['cpu_affinity'][0],0,2**20)
    require(type(intent['started_at_utc']) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00',intent['started_at_utc']) is not None,'start_time')
    keys(started,{'pid','command_sha256'});integer(started['pid'],1,2**31);same(started['command_sha256'],intent['command_sha256'],'process_command')
    keys(process,{'schema','exit_code','wall_timeout','wall_seconds','user_cpu_seconds','system_cpu_seconds','child_max_rss_kib','max_rss_scope','stdout_sha256','stderr_sha256','launcher_sha256','command_sha256','actual_optimizer_updates','production_qualified'})
    expected_process=dict(schema='research-grouped-balance-local-process-result-v1',max_rss_scope='Linux RUSAGE_CHILDREN, not cgroup or SSC305',launcher_sha256=LAUNCHER_SHA,command_sha256=intent['command_sha256'],actual_optimizer_updates='VERIFY_CALL_AND_STEP_LEDGER',production_qualified=False)
    for k,val in expected_process.items():same(process[k],val,'process_binding_'+k)
    integer(process['exit_code'],-255,255);require(type(process['wall_timeout']) is bool,'timeout_bool')
    for k in ('wall_seconds','user_cpu_seconds','system_cpu_seconds','child_max_rss_kib'):number(process[k])
    for name in ('stdout','stderr'):same(sha(safe_read(parent/(name+'.log'))),process[name+'_sha256'],'process_log_hash')
    # These raw receipts and logs stay local. Only the supplied public commit is summarized.
    return process


def verify(repo,result,claim_path,process_path,freeze_path):
    freeze_raw=safe_read(freeze_path);freeze=strict_json(freeze_raw)
    keys(freeze,{'schema','plan_sha256','launcher_sha256','runner_sha256','reviewed_source_commit','reviewed_source_pins','tolerances_sha256','verifier_sha256'})
    same(freeze['schema'],'grouped-balance-verifier-trusted-freeze-v1','freeze_schema')
    for k in ('plan_sha256','launcher_sha256','runner_sha256','tolerances_sha256','verifier_sha256'):digest(freeze[k])
    pinned(Path(__file__),freeze['verifier_sha256'])
    tolerance_raw=pinned(Path(__file__).with_name('tolerances.json'),freeze['tolerances_sha256'])
    same(strict_json(tolerance_raw),dict(schema='grouped-balance-independent-tolerances-v1',**TOLERANCES,frozen_before_model_execution=True,posthoc_relaxation_allowed=False),'tolerance_constants')
    require(type(freeze['reviewed_source_pins']) is dict and freeze['reviewed_source_pins'],'freeze_pins')
    for name,h in freeze['reviewed_source_pins'].items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts,'source_path');digest(h);pinned(Path(repo)/name,h)
    prefix='research/kws-grouped-balance-v1/'
    same(freeze['reviewed_source_pins'].get(prefix+'plan.json'),freeze['plan_sha256'],'freeze_plan');same(freeze['reviewed_source_pins'].get(prefix+'run_grouped_balance.py'),freeze['runner_sha256'],'freeze_runner')
    base=Path(repo).absolute().parent
    for actual,expected in ((repo,base/'grouped-balance-integrated'),(result,base/'grouped-balance-result'),(claim_path,base/'grouped-balance-campaign-ledger/research-budget/kws-grouped-balance-20261015-v1.claim.json'),(process_path,base/'grouped-balance-execution-evidence/process-result.json')):same(Path(actual).absolute(),expected,'fixed_path')
    plan,train,probe,runtime=load_sources(repo,freeze)
    fd=safe_dir(result)
    try:names=sorted(os.listdir(fd))
    finally:os.close(fd)
    require(set(names)<=ALLOWED_FILES,'unexpected_result_file')
    raws={};total_bytes=0
    for n in names:
        raw=safe_read(Path(result)/n);total_bytes+=len(raw);require(total_bytes<=512*1024**2,'total_evidence_size');raws[n]=raw
    data={n:strict_json(raw) for n,raw in raws.items()}
    process=verify_outer(process_path,freeze['reviewed_source_commit'],repo,freeze)
    summary=verify_objects(data,plan,train,probe,runtime,safe_read(claim_path),result,process,freeze['plan_sha256'],raws)
    summary['externally_supplied_source_commit']=freeze['reviewed_source_commit']
    summary['limitations']=['All saved training and observation log probabilities have independent float64 CTC checks under pre-frozen tolerances; no model or optimizer reexecution.','A and B use the same source-defined initialization, seed, optimizer and update budget; receipts bind declarations, not independent hardware attestation.','Only fixed step 600 endpoints are compared; probe never selects a checkpoint or model.','The four Vivian probe items are exposed, machine-labelled, non-test diagnostic samples. Even 4/4 is not generalization, recall, FA/h, human-gold evaluation or production qualification.','Preset groups are conservative identity proxies, not proof of independent speakers.']
    if summary['execution_status']=='completed':
        st=summary['statistics'];summary['fixed_endpoint_B_minus_A']={role:{'macro_complete_rate':st[f'{prefix}B-0600']['macro_complete_rate']-st[f'{prefix}A-0600']['macro_complete_rate'],'per_class_complete_rate':{c:st[f'{prefix}B-0600']['per_class_complete_rate'][c]-st[f'{prefix}A-0600']['per_class_complete_rate'][c] for c in TARGETS}} for role,prefix in [('train',''),('probe','probe-')]}
    fd=safe_dir(result)
    try:same(sorted(os.listdir(fd)),names,'directory_changed')
    finally:os.close(fd)
    for n,raw in raws.items():same(sha(safe_read(Path(result)/n)),sha(raw),'evidence_changed')
    manifest=dict(schema='grouped-balance-evidence-hash-manifest-v1',trusted_freeze_sha256=sha(freeze_raw),plan_sha256=freeze['plan_sha256'],result_files={n:sha(raw) for n,raw in raws.items()},verifier_sha256=sha(safe_read(__file__)))
    return summary,manifest

def write_reports(report_dir,result,summary,manifest):
    report_dir=Path(report_dir).absolute();result=Path(result).absolute()
    require(report_dir!=result and result not in report_dir.parents,'reports_must_be_separate')
    parent=safe_dir(report_dir.parent)
    try:
        os.mkdir(report_dir.name,dir_fd=parent)
        directory=os.open(report_dir.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
        try:
            for name,obj in [('hash-manifest.json',manifest),('summary.json',summary)]:
                fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
                with os.fdopen(fd,'w') as f:
                    json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
            os.fsync(directory)
        finally:os.close(directory)
    finally:os.close(parent)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('repo','result','campaign-claim','process-result','report-dir','trusted-freeze'):p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args()
    try:
        summary,manifest=verify(a.repo,a.result,a.campaign_claim,a.process_result,a.trusted_freeze)
        write_reports(a.report_dir,a.result,summary,manifest)
    except Exception:
        print(json.dumps(dict(schema='grouped-balance-independent-verification-v1',evidence_validation='FAIL',scientific_validation='NOT_EVALUATED',execution_status='incomplete',error='evidence_contract_rejected',retry_authorized=False)));return 2
    print(json.dumps({k:v for k,v in summary.items() if k not in ('statistics','limitations')},sort_keys=True))
    return 0 if summary['execution_status']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
