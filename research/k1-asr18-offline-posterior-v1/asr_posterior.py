"""Root-pinned, offline-only ASR evidence acceptance. Never runs a recognizer or writer."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
MANIFEST_SHA = '8674003b2a741bf2e055b6ea2df4d62b0a70f8b2c0af1f3fc687cadb8d30ffd5'
ASR_REL = 'research/k1-punctuation6-asr18-v1/'
MAX_LOG = 8 * 1024 ** 2
SAFE_FAILURE = '{"schema":"k1-asr18-offline-acceptance-failure-v1","status":"blocked_or_unconfirmed","native_success_established":false,"consumed_attempts":null,"retry_authorized":false}'
INPUT_KEYS = {'tts_authority_sha256','screening_receipt_sha256','job_sha256',
              'execution_plan_sha256','worker_input_binding_sha256'}

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def decode(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key');result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def regular(path, cap):
    path=Path(path)
    require(not any(p.is_symlink() for p in (path,*path.parents)), 'unaliased local input')
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    try:
        info=os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and 0<=info.st_size<=cap,'bounded ordinary local input')
        chunks=[];left=cap+1
        while left:
            raw=os.read(fd,min(left,1024*1024))
            if not raw:break
            chunks.append(raw);left-=len(raw)
        raw=b''.join(chunks);after=os.fstat(fd)
        require(len(raw)==info.st_size and after.st_size==info.st_size and after.st_mtime_ns==info.st_mtime_ns,'stable complete local bytes')
        return raw
    finally:os.close(fd)

def _load_validators():
    raw=regular(HERE/'validator-dependencies.json',1024*1024)
    require(sha(raw)==MANIFEST_SHA,'fixed validator manifest')
    for row in decode(raw)['files']:
        body=regular(SOURCE/row['path'],8*1024**2)
        require(len(body)==row['bytes'] and sha(body)==row['sha256'],'fixed validator source bytes')
    for path in (SOURCE/'tools',SOURCE/ASR_REL):sys.path.insert(0,str(path))
    import evidence_transport as t
    import evidence_publisher as p
    import campaign_contract as c
    import transfer_recovery as r
    import failure_observer as f
    import speech_asr_consensus as s
    for module,name in ((t,'evidence_transport.py'),(p,'evidence_publisher.py'),(c,'campaign_contract.py'),(r,'transfer_recovery.py'),(f,'failure_observer.py')):
        require(Path(module.__file__).resolve()==SOURCE/ASR_REL/name,'fixed validator module origin')
    require(Path(s.__file__).resolve()==SOURCE/'tools/speech_asr_consensus.py','fixed normalization origin')
    return t,p,c,r,f,s

try:
    t,p,c,r,f,consensus=_load_validators()
except Exception:
    if __name__!='__main__':raise
    print(SAFE_FAILURE)
    raise SystemExit(2) from None

def exact_fields(value, keys, label):require(type(value) is dict and set(value)==set(keys),label)
def digest_field(value):require(type(value) is str and t.SHA.fullmatch(value),'exact SHA256 pin')
def git_field(value):require(type(value) is str and t.HEAD.fullmatch(value),'exact Git pin')

def validate_expected(e):
    require(type(e) is dict, 'exact acceptance request object')
    bound = e.get('schema') == 'k1-asr18-offline-acceptance-v2'
    fields = {'schema','mode','producer','archive','result','input_authorities','recovery'}
    if bound:
        fields.add('claim_owner')
    exact_fields(e,fields,'exact acceptance request')
    require(e['schema'] in ('k1-asr18-offline-acceptance-v1','k1-asr18-offline-acceptance-v2') and e['mode'] in ('normal_publication','recovered_checkpoint'),'fixed acceptance mode')
    if bound:
        claim = e['claim_owner']
        exact_fields(claim, {'ref','commit','ref_response_sha256','commit_response_sha256','source_branch','workflow','result_branch'}, 'independently pinned claim custody')
        require(type(claim['ref']) is str and claim['ref'].startswith('refs/heads/'), 'fixed external owner ref')
        p._branch(claim['ref'][11:]);git_field(claim['commit'])
        p._branch(claim['source_branch']);p._branch(claim['result_branch'])
        require(type(claim['workflow']) is str and re.fullmatch(r'\.github/workflows/[A-Za-z0-9_-]+\.ya?ml', claim['workflow']), 'root-pinned producer workflow')
        for key in ('ref_response_sha256','commit_response_sha256'):digest_field(claim[key])
        scope = c.current_scope()
        require(claim['ref'] == scope['claim_ref'] and all(claim[key] == scope[key]
            for key in ('source_branch','workflow','result_branch')), 'independent owner custody matches compiled source scope')
    v=e['producer'];exact_fields(v,{'identity','numeric_job_id','source_tree','source_closure_sha256','source_commit_timestamp','terminal'},'exact producer pins')
    t.identity(v['identity']);require(type(v['numeric_job_id']) is int and v['numeric_job_id']>0,'numeric producer job')
    git_field(v['source_tree']);digest_field(v['source_closure_sha256'])
    require(p._utc_timestamp(v['source_commit_timestamp'])==v['source_commit_timestamp'],'canonical source timestamp')
    term=v['terminal'];exact_fields(term,{'run_conclusion','job_conclusion','host_step_conclusion','publisher_step_conclusion','host_receipt_status'},'exact original terminal pins')
    for key in ('run_conclusion','job_conclusion','host_step_conclusion','publisher_step_conclusion'):require(type(term[key]) is str and term[key] in r.CONCLUSIONS,'observed producer terminal')
    require(term['host_receipt_status'] in ('all_planned_native_complete','stopped_no_retry','cleanup_unconfirmed'),'observed host receipt status')
    require(term['host_step_conclusion']!='success' or term['host_receipt_status']=='all_planned_native_complete','host success requires complete receipt')
    require(term['run_conclusion']!='success' or term['job_conclusion']=='success','consistent run/job success')
    require(term['job_conclusion']!='success' or term['host_step_conclusion']==term['publisher_step_conclusion']=='success','consistent job/step success')
    exact_fields(e['archive'],{'bytes','sha256','outer_manifest_sha256'},'exact archive pins')
    require(type(e['archive']['bytes']) is int and 0<e['archive']['bytes']<=t.MAX_ARCHIVE,'bounded original archive')
    for key in ('sha256','outer_manifest_sha256'):digest_field(e['archive'][key])
    exact_fields(e['result'],{'commit','tree'},'exact result pins')
    for value in e['result'].values():git_field(value)
    exact_fields(e['input_authorities'],INPUT_KEYS,'exact original input authorities')
    for value in e['input_authorities'].values():digest_field(value)
    if e['mode']=='normal_publication':
        require(e['recovery'] is None and term['publisher_step_conclusion']=='success','normal publication requires original publisher success')
    else:
        a=authority(e);r.validate_authority(a)
        rec=e['recovery'];exact_fields(rec,{'identity','numeric_job_id','source_tree','source_closure_sha256','source_commit_timestamp','step_name'},'exact recovery pins')
        who=rec['identity'];exact_fields(who,{'repository','run_id','run_attempt','job_key','executed_head_sha'},'exact recovery identity')
        require(who['repository']==t.REPOSITORY and who['job_key']==r.JOB_KEY and type(who['run_id']) is int and who['run_id']>0 and type(who['run_attempt']) is int and who['run_attempt']==1,'fixed recovery identity')
        git_field(who['executed_head_sha']);git_field(rec['source_tree']);digest_field(rec['source_closure_sha256'])
        require(type(rec['numeric_job_id']) is int and rec['numeric_job_id']>0 and who['run_id']!=v['identity']['run_id'] and who['executed_head_sha']!=v['identity']['executed_head_sha'],'distinct original and recovery identities')
        require(p._utc_timestamp(rec['source_commit_timestamp'])==rec['source_commit_timestamp'],'canonical recovery source timestamp')
        require(type(rec['step_name']) is str and 1<=len(rec['step_name'])<=100 and '\n' not in rec['step_name'],'root-pinned recovery step')
    return e

def authority(e):
    v=e['producer'];return {'schema':'k1-asr18-transfer-recovery-authority-v1','producer_identity':v['identity'],'numeric_job_id':v['numeric_job_id'],'source_tree':v['source_tree'],'source_closure_sha256':v['source_closure_sha256'],'source_commit_timestamp':v['source_commit_timestamp'],'archive_sha256':e['archive']['sha256'],'archive_bytes':e['archive']['bytes'],'outer_manifest_sha256':e['archive']['outer_manifest_sha256'],'checkpoint_ref':'refs/heads/'+t.POLICY['result_branch']+'-checkpoint','checkpoint_commit':e['result']['commit'],'checkpoint_tree':e['result']['tree'],'terminal':v['terminal']}

def producer_terminal(e, records, *, source_claim=None):
    v=e['producer'];base=p.BASE+'/actions/runs/'+str(v['identity']['run_id'])
    routes={'producer_run':base,'producer_history':p.BASE+'/actions/workflows/'+t.POLICY['workflow'].split('/')[-1]+'/runs?event=push&branch='+t.POLICY['source_branch']+'&per_page=100','producer_jobs':base+'/attempts/1/jobs?per_page=100'}
    policy = t.LEGACY_POLICY if source_claim is None else c.require_current_scope(source_claim['identity'])
    if source_claim is not None:
        require(e['schema'] == 'k1-asr18-offline-acceptance-v2', 'independent external owner custody required')
        routes['producer_history'] = p.producer_history_route(source_claim['identity'])
        locator = records.get('_response_locators', {}).get('producer_history')
        if locator is not None:
            require(locator in locators(e)['producer_history'], 'independently pinned producer history resource')
            routes['producer_history'] = locator.removeprefix('https://api.github.com')
        external_claim_custody(e, records, source_claim)
    else:
        require(e['schema'] == 'k1-asr18-offline-acceptance-v1', 'v2 request cannot promote legacy source')
    by_route={route:key for key,route in routes.items()}
    def get(method, route):
        require(method=='GET' and route in by_route,'read-only original terminal predicate')
        value = records[by_route[route]]
        return decode(value) if type(value) is bytes else value
    get.response_bodies = {route:(records[key] if type(records[key]) is bytes else records.get('_response_bodies',{}).get(key)) for route,key in by_route.items()}
    observations = {}
    get.history_custody_observations = {}
    number=r._verify_run(get,routes,'producer',v['identity'],policy['source_branch'],policy['workflow'],'asr18',v['terminal'],
        source_claim=source_claim, observations=observations)
    require(number==v['numeric_job_id'],'same original numeric job')
    if get.history_custody_observations:
        observations['producer_history_custody'] = get.history_custody_observations[routes['producer_history']]
    return observations

def external_claim_custody(e, records, proof):
    """Independent original read-only snapshots, outside producer API budgets."""
    pins = e['claim_owner'];owner = proof['admission']['owner'];identity = proof['identity']
    c.require_current_scope(identity)
    require(pins['ref'] == owner['claim_ref'] and pins['commit'] == proof['admission']['commit']['sha'] and
        all(pins[key] == identity[key] for key in ('source_branch','workflow','result_branch')),
        'source-bound immutable owner equals independently pinned custody scope')
    for role,key,validator in (('claim_ref','ref_response_sha256',c.validate_claim_ref),
                               ('claim_commit','commit_response_sha256',c.verify_owner_commit)):
        raw = records.get(role)
        require(type(raw) is bytes and 0 < len(raw) <= c.MAX_HTTP_RESPONSE and sha(raw) == pins[key],
            'original independently pinned external claim response bytes')
        validator(decode(raw), owner)
    return True


def source_bound_producer(e, records, payload):
    v=e['producer']
    require(type(payload) is bytes and len(payload)==e['archive']['bytes'] and sha(payload)==e['archive']['sha256'],
        'original archive body mandatory before v2 policy selection')
    rows=source(records,'producer',v['identity'],v['source_tree'],v['source_commit_timestamp'])
    proof=t.validate_source_binding(payload,v['identity'],v['source_closure_sha256'],rows,v['numeric_job_id'],v['source_tree'])
    require((proof is not None) == (e['schema']=='k1-asr18-offline-acceptance-v2'), 'actual source selects independent acceptance policy')
    if proof is not None:
        require(proof['identity']['source_timestamp']==v['source_commit_timestamp'], 'same independently observed source timestamp')
    return rows,proof


def terminal_recovery(rec, records):
    who=rec['identity'];run=records['recovery_run']
    fixed={'id':who['run_id'],'run_attempt':1,'head_sha':who['executed_head_sha'],'head_branch':r.SOURCE_BRANCH,'event':'push','path':r.WORKFLOW}
    p._equal(run,fixed,'same terminal recovery run');p._equal(run.get('repository'),{'full_name':t.REPOSITORY,'private':False},'public recovery run')
    history=records['recovery_history'];p._equal(history,{'total_count':1},'one complete recovery history')
    require(type(history.get('workflow_runs')) is list and len(history['workflow_runs'])==1,'one recovery run only')
    p._equal(history['workflow_runs'][0],{**fixed,'status':run.get('status'),'conclusion':run.get('conclusion')},'same terminal history observation')
    jobs=records['recovery_jobs'];require(type(jobs.get('total_count')) is int and 1<=jobs['total_count']<=100 and type(jobs.get('jobs')) is list and len(jobs['jobs'])==jobs['total_count'],'complete recovery jobs')
    matches=[j for j in jobs['jobs'] if type(j) is dict and j.get('name')==r.JOB_KEY];require(len(matches)==1,'unique recovery job');job=matches[0]
    p._equal(job,{'id':rec['numeric_job_id'],'run_id':who['run_id'],'head_sha':who['executed_head_sha'],'head_branch':r.SOURCE_BRANCH,'run_url':'https://api.github.com'+p.BASE+'/actions/runs/'+str(who['run_id'])},'same recovery job')
    if 'run_attempt' in job:p._equal(job,{'run_attempt':1},'recovery job attempt one')
    steps=job.get('steps');require(type(steps) is list and 1<=len(steps)<=100 and all(type(x) is dict for x in steps),'complete recovery steps')
    matches=[s for s in steps if s.get('name')==rec['step_name']];require(len(matches)==1,'one pinned recovery step');step=matches[0]
    for item in (run,job,step):
        require(item.get('status') in ('queued','in_progress','completed','waiting','requested','pending') and (item.get('conclusion') is None or type(item.get('conclusion')) is str and item['conclusion'] in r.CONCLUSIONS),'bounded observed recovery terminal enums')
    return all(x.get('status')=='completed' and x.get('conclusion')=='success' for x in (run,job,step)), {'run':run.get('conclusion'),'job':job.get('conclusion'),'step':step.get('conclusion')}

def source(records, prefix, who, tree, stamp):
    commit=records[prefix+'_source_commit'];p._equal(commit,{'sha':who['executed_head_sha']},'pinned source commit')
    p._equal(commit.get('tree'),{'sha':tree},'pinned source tree')
    require(type(commit.get('committer')) is dict and p._utc_timestamp(commit['committer'].get('date'))==stamp,'pinned source UTC')
    return p.verify_tree(records[prefix+'_source_tree'],tree)

def recovery_source(e, records):
    rec=e['recovery'];rows=source(records,'recovery',rec['identity'],rec['source_tree'],rec['source_commit_timestamp']);by={x['path']:x for x in rows}
    raw=records['recovery_source_freeze'];require(type(raw) is bytes and sha(raw)==rec['source_closure_sha256'],'actual recovery freeze bytes')
    freeze=t.decode(raw);require(t.canonical(freeze)==raw and freeze.get('schema')=='k1-asr18-source-freeze-v1' and type(freeze.get('host_code')) is dict,'canonical recovery source freeze')
    bodies={ASR_REL+'execution-freeze.json':raw,ASR_REL+'transfer-recovery-release.json':records['recovery_control'],r.WORKFLOW:records['recovery_workflow']}
    for leaf in ('transfer_recovery.py','evidence_publisher.py','evidence_transport.py','campaign_contract.py'):
        bodies[ASR_REL+leaf]=regular(SOURCE/ASR_REL/leaf,8*1024**2)
    for name,body in bodies.items():
        require(type(body) is bytes and 0<len(body)<=8*1024**2,'original bounded recovery source body')
        require(by.get(name)=={'path':name,'mode':'100644','type':'blob','sha':t.git_sha(body)},'recovery source body Git binding')
        if name!=ASR_REL+'execution-freeze.json':require(freeze['host_code'].get(name)=={'bytes':len(body),'sha256':sha(body)},'recovery authority and shared runtime frozen')
    control=t.decode(records['recovery_control']);require(t.canonical(control)==records['recovery_control'],'canonical recovery control')
    require(c.exact_json(r.recovery_gate(control),authority(e)),'same original authority in actual recovery source')
    workflow=records['recovery_workflow'].decode('utf-8');require(rec['step_name'] in workflow and 'transfer_recovery.py' in workflow,'pinned recovery step source present')

def inspect_stdout(logs, e):
    require(logs is None or type(logs) is dict and set(logs)<= {'producer','recovery'},'fixed optional log roles')
    answer={'attributed':[],'unattributed_ci_or_precontext':0,'recovery_receipts':[]}
    for role,raw in (logs or {}).items():
        require(type(raw) is bytes and len(raw)<=MAX_LOG,'bounded original log')
        for line in raw.decode('utf-8').splitlines():
            if line[:1]!='{':
                if not re.match(r'^20\d\d-\d\d-\d\dT\S+Z ',line):continue
                line=line.split(' ',1)[1]
            try:value=decode(line.encode())
            except (ValueError,UnicodeError):continue
            if type(value) is not dict:continue
            schema=value.get('schema')
            if schema=='k1-asr18-transfer-recovery-receipt-v1':
                require(e['recovery'] is not None and role=='recovery' and c.exact_json(value.get('recovery_identity'),e['recovery']['identity']) and c.exact_json(value.get('producer_identity'),e['producer']['identity']),'exact attributed recovery receipt')
                answer['recovery_receipts'].append(value);continue
            if schema=='k1-asr18-independent-process-diagnostic-envelope-v1':
                record=value.get('record');require(type(record) is dict,'diagnostic record body')
                body=c.canonical(record);require(value.get('record_bytes')==len(body) and value.get('record_sha256')==sha(body),'diagnostic record original digest')
            elif schema in ('k1-asr18-safe-checkpoint-v1','k1-asr18-independent-diagnostic-fallback-v1'):record=value
            else:continue
            who=record.get('identity')
            if who is None:answer['unattributed_ci_or_precontext']+=1;continue
            wanted=e['producer']['identity'] if role=='producer' else (e['recovery'] or {}).get('identity')
            require(c.exact_json(who,wanted),'foreign diagnostic identity rejected')
            require(record.get('stage') in f.STAGES,'fixed diagnostic stage')
            answer['attributed'].append({'role':role,'schema':schema,'stage':record['stage'],'record_sha256':sha(c.canonical(record))})
    return answer

def recovery_receipt(e, value):
    a=authority(e);rec=e['recovery']
    wanted={'schema':'k1-asr18-transfer-recovery-receipt-v1','status':'exact_original_checkpoint_final_ref_verified','producer_identity':a['producer_identity'],'producer_numeric_job_id':a['numeric_job_id'],'original_terminal':a['terminal'],'original_run_status':'completed','original_job_status':'completed','original_publisher_reclassified':False,'recovery_identity':rec['identity'],'recovery_numeric_job_id':rec['numeric_job_id'],'recovery_source_closure_sha256':rec['source_closure_sha256'],'checkpoint_ref':a['checkpoint_ref'],'checkpoint_commit':a['checkpoint_commit'],'checkpoint_tree':a['checkpoint_tree'],'final_ref':'refs/heads/'+t.POLICY['result_branch'],'archive_sha256':a['archive_sha256'],'archive_bytes':a['archive_bytes'],'outer_manifest_sha256':a['outer_manifest_sha256'],'native_success_established':False,'new_native_calls_authorized':False,'retry_authorized':False}
    if e['schema']=='k1-asr18-offline-acceptance-v2':
        wanted['producer_history']=value.get('producer_history')
        require(type(wanted['producer_history']) is dict, 'original recovery producer history observation')
    require(c.exact_json(value,wanted),'exact original-byte transfer-only receipt')

def phase_resource_consistency(files, host, terminal):
    """Cross-member success implications from the unchanged host/supervisor writers."""
    # hosted_run writes this durable proof before every decode/started claim.
    attempted=any(row['attempts']==1 for row in terminal['slots'])
    if attempted:require('policy/budget-proof.json' in files,'started slot requires original predecode capacity proof')
    full=host['status']=='all_planned_native_complete'
    if not full:return {'complete_resource_success_established':False,'scope':'partial observations; missing future/final receipts are not synthesized'}
    required={'cpu-receipt.json','cleanup-receipt.json','supervision-receipt.json','policy/budget-proof.json'}
    require(required<=set(files),'complete host requires original resource and capacity receipts')
    cpu=t.decode(files['cpu-receipt.json']);cleanup=t.decode(files['cleanup-receipt.json']);supervision=t.decode(files['supervision-receipt.json'])
    from cpu_budget import CpuLedger
    ledger=CpuLedger();ledger.entries={name:dict(row) for name,row in cpu['containers'].items()}
    require(ledger.check()==[],'original cumulative CPU threshold predicate')
    snapshot=ledger.receipt()
    for key in ('total_usage_usec','thresholds_seconds','total_threshold_seconds','configured_poll_seconds','kill_command_timeout_seconds','hard_cumulative_kernel_quota'):
        require(c.exact_json(cpu[key],snapshot[key]),'original CPU accounting and limit fields')
    require(cpu['all_final_usage_verified'] is True and cpu['threshold_exceeded'] is False,'complete host requires final CPU verification without threshold failure')
    require(cleanup['all_cleanup_verified'] is True and cleanup['all_removed'] is True,'complete host requires acknowledged cleanup')
    require(all(row['closed'] is True and row['final_verified'] is True for row in cpu['containers'].values()),'all recorded CPU containers finally verified')
    require(all(row['cleanup_ok'] is True and row['status']=='removed' and row['ownership']=='verified' and not row['receipt_errors'] for row in cleanup['containers']),'complete cleanup rows consistent')
    stages=supervision['stages'];expected=[('setup',None)]+[(family,'prepare') for family in c.FAMILIES]+[(family,'decode') for family in c.FAMILIES]
    require([(row['family'],row['mode']) for row in stages]==expected,'complete fixed setup prepare decode supervision roster')
    names=[row['name'] for row in stages];require(len(set(names))==7,'unique completed supervision names')
    cleanup_by={row['name']:row for row in cleanup['containers']}
    require(len(cleanup_by)==len(cleanup['containers']),'unique cleanup names')
    prefix=c.EXPERIMENT+'-'+str(host['identity']['run_id'])+'-'
    for row in stages:
        suffix='setup' if row['family']=='setup' else row['family']+'-'+row['mode']
        require(row['name']==prefix+suffix,'completed supervision uses original producer writer name')
        stage=row['family'] if row['mode']=='decode' else 'setup'
        require(cpu['containers'].get(row['name'],{}).get('stage')==stage and cleanup_by.get(row['name'],{}).get('stage')==stage,'completed worker uses original CPU and cleanup stage')
        require(row['status']=='complete' and row['worker_exit_success'] is True and row['primary_error'] is None and row['cleanup_error'] is None,'complete supervised workers have no failure')
        require(row['gate_result']=={'schema':'qwen12-gated-command-v1','status':'valid','exit_code':0},'complete worker gate acknowledged')
        require(row['name'] in cpu['containers'] and row['name'] in cleanup_by,'supervised workers retain matching resource receipts')
        require(cleanup_by[row['name']]['accounting_status']=='final_usage_verified','completed supervised worker final accounting')
        cleaned=cleanup_by[row['name']]
        require(cleaned['host_deadline_expired_at_start'] is False and cleaned['host_deadline_overrun'] is False and cleaned['pause_acknowledged'] is True and cleaned['cleanup_overrun_seconds']==0 and cleaned['cleanup_seconds']<=cleaned['cleanup_budget_seconds'],'completed supervised cleanup satisfies original pause and deadline conditions')
    return {'complete_resource_success_established':True,'supervised_stages':len(stages),'scope':'original host success implications only; no resource limits changed'}

def scientific_summary(files, e):
    get=lambda name:t.decode(files[name])
    host=get('host-receipt.json');terminal=get('terminal-outcomes.json');projection=get('new-cell-projection.json')
    resource=phase_resource_consistency(files,host,terminal)
    binding_raw=files.get('order/worker-input-binding.json')
    if binding_raw is None:
        require(host['input_verified'] is False,'verified input requires original worker binding body')
        binding_raw=c.canonical(get('input-freeze.json')['actual_worker_binding'])
    binding=t.decode(binding_raw);actual={k:binding[k] for k in INPUT_KEYS if k!='worker_input_binding_sha256'};actual['worker_input_binding_sha256']=sha(binding_raw)
    require(c.exact_json(actual,e['input_authorities']),'five actual six-input authority pins')
    require(host['status']==e['producer']['terminal']['host_receipt_status'],'API and archive host status binding')
    blind=c.metadata.blind_metadata(projection);slots=c.fixed_slots(projection,blind['job'],blind['execution_plan'])
    rows=[]
    for slot,outcome in zip(slots,terminal['slots']):
        row={'slot_id':slot['slot_id'],'cell_id':slot['cell_id'],'family':slot['family'],'audio_id':slot['audio_id'],'wav_sha256':slot['wav_sha256'],'pcm_sha256':slot['pcm_sha256'],'audio_frames':slot['audio_frames'],'status':outcome['status'],'attempts':outcome['attempts'],'raw_transcript':None,'normalized_transcript':None}
        if outcome['status']=='complete':
            raw=get('output/'+slot['family']+'/'+slot['audio_id']+'/raw-transcript.json')['actual_text'];row.update(raw_transcript=raw,normalized_transcript=consensus.normalized_transcript(raw))
        rows.append(row)
    candidates=[]
    for cell in c.CELLS:
        group=[x for x in rows if x['cell_id']==cell];texts=[x['normalized_transcript'] for x in group]
        unanimous=len(group)==3 and all(x['status']=='complete' for x in group) and len(set(texts))==1 and bool(texts[0])
        candidates.append({'cell_id':cell,'three_complete':all(x['status']=='complete' for x in group),'unanimous_machine_candidate':unanimous,'normalized_text':texts[0] if unanimous else None,'training_admitted':False,'human_gold':False})
    complete=host['status']=='all_planned_native_complete' and terminal['accounting_valid'] is True and len(rows)==18 and all(x['status']=='complete' for x in rows)
    return {'status':'complete_native_verified' if complete else 'partial_native_verified','native_success_established':complete,'host_status':host['status'],'input_verified_by_original_host':host['input_verified'],'accounting_valid':terminal['accounting_valid'],'verified_complete_slots':sum(x['status']=='complete' for x in rows),'ledger_reported_attempts':sum(x['attempts'] for x in rows) if terminal['accounting_valid'] else None,'consumed_attempts':None,'rows':rows,'machine_candidates':candidates,'resource_verification':resource,'normalization':consensus.NORMALIZATION,'training_admitted':False,'human_gold':False,'acoustic_verbatim_completeness':'unknown'}

def _accept(expected, records, outer_raw, chunks, payload, logs=None):
    e=validate_expected(expected);require(type(records) is dict,'original API records required')
    source_rows,proof=(None,None)
    if e['schema']=='k1-asr18-offline-acceptance-v2':
        source_rows,proof=source_bound_producer(e,records,payload)
    producer_observations=producer_terminal(e,records,source_claim=proof)
    observation=inspect_stdout(logs,e)
    v=e['producer'];base={'schema':'k1-asr18-offline-acceptance-result-v1','custody_mode':e['mode'],'producer_identity':v['identity'],'producer_terminal':v['terminal'],'recovery_identity':None if e['recovery'] is None else e['recovery']['identity'],'original_publisher_reclassified':False,'diagnostics':{k:x for k,x in observation.items() if k!='recovery_receipts'},'original_writer_receipt_observed':False,'original_publisher_http_telemetry':'unavailable','network_calls':0,'model_calls':0,'new_attempts_authorized':False,**producer_observations}
    if e['recovery'] is not None:
        success,terminal=terminal_recovery(e['recovery'],records);base['recovery_terminal']=terminal
        if not success:return {**base,'status':'recovery_not_verified','native_success_established':False,'consumed_attempts':None,'rows':[],'machine_candidates':[],'training_admitted':False,'human_gold':False}
        recovery_source(e,records)
        require(len(observation['recovery_receipts'])==1,'one original successful recovery stdout receipt')
        recovery_receipt(e,observation['recovery_receipts'][0])
        if proof is not None:
            c.validate_bound_history(proof['identity'],observation['recovery_receipts'][0]['producer_history'],'posterior')
        if 'recovery_receipt' in records:require(c.exact_json(records['recovery_receipt'],observation['recovery_receipts'][0]),'derived recovery receipt must match original stdout')
        p._equal(records['checkpoint_ref'],{'ref':'refs/heads/'+t.POLICY['result_branch']+'-checkpoint'},'fixed retained checkpoint ref')
        p._equal(records['checkpoint_ref'].get('object'),{'type':'commit','sha':e['result']['commit']},'checkpoint immutable target')
    if source_rows is None:source_rows,proof=source_bound_producer(e,records,payload)
    require(type(payload) is bytes and len(payload)==e['archive']['bytes'] and sha(payload)==e['archive']['sha256'],'original archive body mandatory')
    require(type(outer_raw) is bytes and sha(outer_raw)==e['archive']['outer_manifest_sha256'],'original outer bytes mandatory')
    t.parse_manifest(outer_raw,v['identity'],v['source_closure_sha256'],v['numeric_job_id'])
    rebuilt=t.reassemble(outer_raw,chunks,v['identity'],v['source_closure_sha256'],v['numeric_job_id']);require(rebuilt==payload,'same original archive bytes')
    manifest,measured=t.validate_archive(payload,v['identity'])
    t.validate_source_binding(payload,v['identity'],v['source_closure_sha256'],source_rows,v['numeric_job_id'],v['source_tree'])
    p._equal(records['result_ref'],{'ref':'refs/heads/'+t.POLICY['result_branch']},'fixed final result ref')
    p._equal(records['result_ref'].get('object'),{'type':'commit','sha':e['result']['commit']},'observed final ref target required')
    release={'source_sha':v['identity']['executed_head_sha'],'source_tree':v['source_tree'],'commit_timestamp':v['source_commit_timestamp']}
    body,commit=p.commit_body(e['result']['tree'],release);require(commit==e['result']['commit'],'same deterministic producer commit')
    p.verify_commit(records['result_commit'],commit,body);result_rows=p.verify_tree(records['result_tree'],e['result']['tree'])
    files={t.manifest_path():outer_raw,**dict(chunks)}
    expected_rows=source_rows+[{'path':name,'type':'blob','mode':'100644','sha':t.git_sha(raw)} for name,raw in sorted(files.items())]
    require(sorted(result_rows,key=lambda x:x['path'])==sorted(expected_rows,key=lambda x:x['path']),'entire original source plus exact evidence tree')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:members={name:archive.read(name) for name in manifest['members']}
    return {**base,**scientific_summary(members,e),'archive_sha256':measured['archive_sha256'],'archive_bytes':measured['archive_bytes'],'archive_members':measured['member_count'],'source_binding_verified':True,'native_ledger_resource_validator':'unchanged evidence_transport.validate_archive and validate_source_binding; adapter additionally checks host-success resource consistency'}

def accept(expected, records, outer_raw, chunks, payload, logs=None):
    try:return _accept(expected,records,outer_raw,chunks,payload,logs)
    except (KeyError,TypeError,IndexError,UnicodeError,zipfile.BadZipFile) as error:
        raise ValueError('malformed or incomplete original evidence') from None


# These locators describe the fixed observed resources. They are not HTTP
# telemetry and do not turn connector content into a raw HTTP response.
def locators(e):
    validate_expected(e)
    v=e['producer'];base='https://api.github.com'+p.BASE
    policy=t.LEGACY_POLICY if e['schema']=='k1-asr18-offline-acceptance-v1' else t.POLICY
    result={}
    for prefix,who,branch,workflow,tree,job in (
        ('producer',v['identity'],policy['source_branch'],policy['workflow'],v['source_tree'],v['numeric_job_id']),
        *([] if e['recovery'] is None else [('recovery',e['recovery']['identity'],r.SOURCE_BRANCH,r.WORKFLOW,e['recovery']['source_tree'],e['recovery']['numeric_job_id'])])):
        run=base+'/actions/runs/'+str(who['run_id'])
        result.update({prefix+'_run':{run},prefix+'_jobs':{run+'/attempts/1/jobs?per_page=100'},
            prefix+'_history':{base+'/actions/workflows/'+workflow.split('/')[-1]+'/runs?event=push&branch='+branch+'&per_page=100',
                base+'/actions/runs?branch='+branch.replace('/','%2F')+'&event=push&per_page=100'},
            prefix+'_source_commit':{base+'/git/commits/'+who['executed_head_sha']},
            prefix+'_source_tree':{base+'/git/trees/'+tree+'?recursive=1'},
            prefix+'_log':{'connector:github_job_log:'+str(job)}})
    if e['schema']=='k1-asr18-offline-acceptance-v2':
        owner=e['claim_owner']
        result['producer_history']={base+'/actions/workflows/'+owner['workflow'].split('/')[-1]+'/runs?branch='+owner['source_branch']+'&per_page=100',base+'/actions/runs?branch='+owner['source_branch'].replace('/','%2F')+'&per_page=100'}
        result.update(claim_ref={base+'/git/ref/'+owner['ref'][5:]},claim_commit={base+'/git/commits/'+owner['commit']})
    commit=e['result']['commit'];tree=e['result']['tree']
    result.update(result_ref={base+'/git/ref/heads/'+policy['result_branch']},result_commit={base+'/git/commits/'+commit},result_tree={base+'/git/trees/'+tree+'?recursive=1'},outer_manifest={'https://raw.githubusercontent.com/'+t.REPOSITORY+'/'+commit+'/'+t.manifest_path()},archive={'derived:ordered-original-chunks:'+e['archive']['sha256']})
    for index in range(t.MAX_CHUNKS):result['chunk_%04d'%index]={'https://raw.githubusercontent.com/'+t.REPOSITORY+'/'+commit+'/'+t.chunk_path(index)}
    if e['recovery'] is not None:
        rec=e['recovery'];raw='https://raw.githubusercontent.com/'+t.REPOSITORY+'/'+rec['identity']['executed_head_sha']+'/'
        result.update(checkpoint_ref={base+'/git/ref/heads/'+t.POLICY['result_branch']+'-checkpoint'},recovery_source_freeze={raw+ASR_REL+'execution-freeze.json'},recovery_control={raw+ASR_REL+'transfer-recovery-release.json'},recovery_workflow={raw+r.WORKFLOW})
    return result

def load_and_accept(root, expected_path, expected_sha256, custody_path, custody_sha256):
    digest_field(expected_sha256);digest_field(custody_sha256)
    eraw=regular(expected_path,64*1024);craw=regular(custody_path,128*1024)
    require(sha(eraw)==expected_sha256 and sha(craw)==custody_sha256,'independently pinned exact input files')
    e=validate_expected(decode(eraw));index=decode(craw)
    exact_fields(index,{'schema','entries'},'exact custody index')
    require(index['schema']==('k1-asr18-offline-custody-v2' if e['schema']=='k1-asr18-offline-acceptance-v2' else 'k1-asr18-offline-custody-v1') and type(index['entries']) is list and 1<=len(index['entries'])<=80,'bounded original custody index')
    wanted=locators(e);loaded={};seen=set();total=0;provenance=[];root=Path(root).absolute()
    require(root.is_dir() and not any(p.is_symlink() for p in (root,*root.parents)),'fixed evidence directory')
    for row in index['entries']:
        exact_fields(row,{'role','path','bytes','sha256','locator'},'exact original custody row')
        role=row['role'];name=row['path']
        require(type(role) is str and role in wanted and role not in loaded,'one fixed original role')
        require(type(name) is str and 0<len(name)<=240 and '\\' not in name and not PurePosixPath(name).is_absolute() and all(x not in ('','.','..') for x in name.split('/')) and name not in seen,'unique contained input path')
        seen.add(name);require(type(row['bytes']) is int and 0<=row['bytes']<=t.MAX_ARCHIVE,'bounded role bytes');digest_field(row['sha256'])
        require(type(row['locator']) is str and row['locator'] in wanted[role],'exact original resource locator')
        cap=t.MAX_ARCHIVE if role=='archive' else t.MAX_CHUNK if role.startswith('chunk_') else MAX_LOG if role.endswith('_log') else t.MAX_MANIFEST if role=='outer_manifest' else t.MAX_MEMBER
        body=regular(root/name,cap);require(len(body)==row['bytes'] and sha(body)==row['sha256'],'all original bytes match root-pinned custody')
        total+=len(body);require(total<=80*1024**2,'bounded aggregate offline original inputs');loaded[role]=body
        provenance.append({'role':role,'sha256':row['sha256'],'locator':row['locator']})
    required={'producer_run','producer_jobs','producer_history','producer_log'}
    if e['recovery'] is not None:required|={'recovery_run','recovery_jobs','recovery_history','recovery_log'}
    if e['schema']=='k1-asr18-offline-acceptance-v2':required|={'claim_ref','claim_commit'}
    require(required<=set(loaded),'original identity terminal and log bodies required')
    raw_roles={'archive','outer_manifest','producer_log','recovery_log','recovery_source_freeze','recovery_control','recovery_workflow'}
    if e['schema']=='k1-asr18-offline-acceptance-v2':raw_roles|={'claim_ref','claim_commit'}
    records={name:(body if name in raw_roles else decode(body)) for name,body in loaded.items() if not name.startswith('chunk_') and name not in {'archive','outer_manifest','producer_log','recovery_log'}}
    if e['schema']=='k1-asr18-offline-acceptance-v2':
        records['_response_bodies']={'producer_history':loaded['producer_history']}
        records['_response_locators']={row['role']:row['locator'] for row in index['entries'] if row['role']=='producer_history'}
    outer=loaded.get('outer_manifest');chunks=[]
    if outer is not None:
        m=t.parse_manifest(outer,e['producer']['identity'],e['producer']['source_closure_sha256'],e['producer']['numeric_job_id'])
        require({name for name in loaded if name.startswith('chunk_')}=={'chunk_%04d'%i for i in range(m['chunk_count'])},'complete original indexed chunks only')
        chunks=[(row['path'],loaded['chunk_%04d'%row['index']]) for row in m['chunks']]
    answer=accept(e,records,outer,chunks,loaded.get('archive'),{key:loaded[key+'_log'] for key in ('producer','recovery') if key+'_log' in loaded})
    return {**answer,'expected_sha256':expected_sha256,'custody_sha256':custody_sha256,'input_roles':provenance,'custody_scope':'root-pinned original local byte snapshots and exact resource locators; no new HTTP observation or publisher telemetry reconstructed'}

def main(argv=None):
    class SafeParser(argparse.ArgumentParser):
        def error(self, message):raise ValueError('fixed offline invocation required')
    parser=SafeParser(description=__doc__)
    parser.add_argument('--evidence-root',required=True);parser.add_argument('--expected',required=True)
    parser.add_argument('--expected-sha256',required=True);parser.add_argument('--custody',required=True);parser.add_argument('--custody-sha256',required=True)
    args=parser.parse_args(argv)
    answer=load_and_accept(args.evidence_root,args.expected,args.expected_sha256,args.custody,args.custody_sha256)
    print(canonical(answer).decode('utf-8'))
    return 0 if answer['status'] in ('complete_native_verified','partial_native_verified') else 2

if __name__=='__main__':
    try:raise SystemExit(main())
    except SystemExit:raise
    except Exception:
        print(SAFE_FAILURE)
        raise SystemExit(2) from None
