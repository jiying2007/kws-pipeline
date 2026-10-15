"""One push admission. Metadata access is host-only; never forwarded to models."""
import json,os,re,urllib.request
from pathlib import Path
from execution_common import require,document,read,sha,claim
REPOSITORY='jiying2007/kws-pipeline'
BRANCH='research/kw1-preset-supplement-20261014-v1'
WORKFLOW='.github/workflows/kw1-preset-supplement-20261014-v1.yml'
HOST_FILES = ('RECONSTRUCTION.md', 'freeze_sources.py', 'IMPLEMENTATION.md', 'README.md', 'admission.py', 'blind_asr_worker.py', 'blind_pcm.py', 'campaign.py', 'derive_native.py', 'docker_backend.py', 'execution_common.py', 'export_public.py', 'launcher.py', 'official-support.json', 'plan.json', 'plan_contract.py', 'public_evidence.py', 'source-bindings.json', 'test_archive_faults.py', 'test_atomic_export.py', 'test_blind_asr.py', 'test_contract.py', 'test_derive_numeric.py', 'test_docker_backend.py', 'test_execution.py', 'test_suite.py', 'test_worker.py', 'verify_public.py', 'worker.py', 'workflow.yml')
def required_sources():
    from docker_backend import dependency_paths
    return set(dependency_paths()) | {"research/kw1-preset-supplement-v1/"+name for name in HOST_FILES} | {WORKFLOW,"research/tts-pilot-screening-v1/numeric-requirements.txt"}
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl): raise ValueError('metadata redirects forbidden')
def api(path):
    require(path.startswith('/repos/'+REPOSITORY+'/actions/') and '..' not in path and '\\' not in path,'fixed official metadata path')
    request=urllib.request.Request('https://api.github.com'+path,headers={'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'kw1-preset-once'})
    # Public metadata needs no credential. Do not read GITHUB_TOKEN.
    with urllib.request.build_opener(NoRedirect).open(request,timeout=30) as response:
        raw=response.read(2*1024**2+1)
    require(len(raw)<=2*1024**2,'metadata size');return json.loads(raw)
def validate(config,env,event,run,workflow,history,repo):
    require(config.get('schema')=='kw1-execution-admission-v1' and config.get('enabled') is True,'execution disabled pending review')
    require(type(config.get('max_tts_calls')) is int and type(config.get('max_asr_calls')) is int and config.get('max_tts_calls')==16 and config.get('max_asr_calls')==48 and config.get('retry') is False,'immutable budget')
    require(env.get('GITHUB_REPOSITORY')==REPOSITORY and env.get('GITHUB_REF')=='refs/heads/'+BRANCH and env.get('GITHUB_EVENT_NAME')=='push' and env.get('GITHUB_RUN_ATTEMPT')=='1' and env.get('GITHUB_RUN_NUMBER')=='1','first attempt route')
    head=env.get('GITHUB_SHA','');require(re.fullmatch('[0-9a-f]{40}',head),'source SHA')
    require(event.get('created') is True and event.get('deleted') is False and event.get('after')==head and event.get('ref')=='refs/heads/'+BRANCH,'first branch creation')
    require(event.get('repository',{}).get('full_name')==REPOSITORY and event['repository'].get('private') is False,'public repository')
    require(all(type(run.get(k)) is int for k in ('id','workflow_id','run_number','run_attempt')) and str(run['id'])==env.get('GITHUB_RUN_ID') and run.get('head_sha')==head and run.get('head_branch')==BRANCH and run.get('event')=='push' and run.get('run_number')==1 and run.get('run_attempt')==1,'server run identity')
    require(run.get('repository',{}).get('full_name')==REPOSITORY and run['repository'].get('private') is False,'server public identity')
    require(type(workflow.get('id')) is int and workflow.get('path')==WORKFLOW and workflow.get('state')=='active' and run.get('workflow_id')==workflow['id'],'registered workflow identity')
    rows=history.get('workflow_runs',[])
    require(type(history.get('total_count')) is int and history.get('total_count')==1 and len(rows)==1 and all(type(rows[0].get(k)) is int for k in ('id','workflow_id','run_number','run_attempt')) and all(rows[0].get(k)==run.get(k) for k in ('id','head_sha','head_branch','workflow_id','run_number','run_attempt','event')),'workflow lifetime first run only')
    files=config.get('source_files');require(type(files) is dict and set(files)==required_sources(),'exact complete reviewed source lock')
    require('research/kw1-preset-supplement-v1/execution-admission.json' not in files,'self-hash cycle forbidden')
    for name,digest in files.items():
        require(type(name) is str and not name.startswith('/') and '..' not in Path(name).parts,'source relative path')
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest),'source digest')
        require(sha(read(Path(repo)/name))==digest,'source lock mismatch: '+name)
    return dict(schema='kw1-admission-receipt-v1',admitted=True,run_id=run['id'],source_sha=head,workflow_id=workflow['id'],config_sha256=sha(read(Path(repo)/'research/kw1-preset-supplement-v1/execution-admission.json')),source_files=files,max_tts_calls=16,max_asr_calls=48,retry=False)
def admit(repo,work,env=None,fetch=api):
    env=dict(os.environ if env is None else env);repo=Path(repo)
    config=document(repo/'research/kw1-preset-supplement-v1/execution-admission.json')
    require(config.get('enabled') is True,'execution disabled pending review')
    event=document(env['GITHUB_EVENT_PATH']);rid=env['GITHUB_RUN_ID']
    require(rid.isdecimal(),'run id')
    run=fetch('/repos/'+REPOSITORY+'/actions/runs/'+rid)
    workflow=fetch('/repos/'+REPOSITORY+'/actions/workflows/'+WORKFLOW.rsplit('/',1)[1])
    history=fetch('/repos/'+REPOSITORY+'/actions/workflows/'+str(workflow['id'])+'/runs?per_page=100')
    receipt=validate(config,env,event,run,workflow,history,repo)
    # Local exclusive claim supplements server lifetime guard; a new directory alone
    # cannot satisfy the first run/attempt/history requirement.
    Path(work).mkdir(exist_ok=False);claim(Path(work)/'campaign.claim.json',receipt)
    return receipt
