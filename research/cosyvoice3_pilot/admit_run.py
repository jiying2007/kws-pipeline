#!/usr/bin/env python3
"""Fail-closed admission for one exact owner-reviewed public CosyVoice pilot."""
import hashlib, json, os, pathlib, re, subprocess, sys, urllib.request
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import runtime_lock
ROOT = pathlib.Path(__file__).resolve().parents[2]
REPOSITORY = 'jiying2007/kws-pipeline'
BRANCH = 'research/cosyvoice3-six-pair-pilot-20261002'
WORKFLOW = '.github/workflows/research-cosyvoice3-pilot.yml'
PREFIX = 'KWS_COSYVOICE3_APPROVAL='
LABEL = 'cosyvoice3-six-pair-once-v1'
BASE = '55a4e23379ad7072db507dbe419b38d8898e17fa'
NONCE = 'cosyvoice3-20261002-six-pairs-v1'

def require(value, message):
    if not value: raise RuntimeError(message)

def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def git(root, *args):
    return subprocess.check_output(['/usr/bin/git', '-C', str(root), *args], timeout=30, text=True).strip()

def identity(root):
    files = [WORKFLOW] + sorted(str(p.relative_to(root)) for p in (root / 'research/cosyvoice3_pilot').rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc'))
    result = {}
    for name in files:
        p = root / name
        require(not p.is_symlink() and p.is_file(), 'source must be regular')
        b=p.read_bytes(); result[name]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
    return hashlib.sha256(canonical(result)).hexdigest()

def validate(event, env, head, source, paths, lock):
    pr=event.get('pull_request', {}); repo=event.get('repository', {})
    require(env.get('GITHUB_EVENT_NAME') == 'pull_request' and event.get('action') == 'labeled', 'only explicit label admission')
    require(event.get('label', {}).get('name') == LABEL, 'wrong activation label')
    require(env.get('GITHUB_RUN_ATTEMPT') == '1', 'reruns never admitted')
    require(repo.get('full_name') == REPOSITORY and repo.get('private') is False, 'public repository mismatch')
    require(pr.get('head', {}).get('repo', {}).get('full_name') == REPOSITORY and pr['head'].get('ref') == BRANCH, 'branch mismatch')
    require(pr.get('user', {}).get('login') == 'jiying2007' and env.get('GITHUB_ACTOR') == 'jiying2007', 'owner mismatch')
    require(pr.get('base', {}).get('sha') == BASE and pr.get('head', {}).get('sha') == head and re.fullmatch('[0-9a-f]{40}', head), 'head/base mismatch')
    require(paths and all(p.startswith('research/cosyvoice3_pilot/') or p == WORKFLOW for p in paths), 'diff exceeds approved scope')
    number=event.get('number'); require(type(number) is int and number>0 and pr.get('number')==number, 'PR mismatch')
    lines=[x[len(PREFIX):] for x in (pr.get('body') or '').splitlines() if x.startswith(PREFIX)]
    require(len(lines)==1 and len(lines[0])<4096, 'exact single approval line required')
    expected={'armed':True,'pr_number':number,'base_sha':BASE,'head_sha':head,'source_sha256':source,'nonce':NONCE,
        'scope':'one_cpu_qualification_then_six_plain_inpaint_pairs','maximum_clips':12,
        'dependency_compressed_bytes':lock['compressed_total_bytes'],'model_bytes':5427029103,
        'post_runtime_free_bytes':6635020288,'new_job_bytes_max':14000000000,'output_bytes_max':134217728,
        'artifact_retention_days':1,'zero_cost_artifact_verified':True}
    require(json.loads(lines[0])==expected, 'exact source/resource/publication approval differs')
    return expected

def api(path, token):
    require(path.startswith('/repos/'+REPOSITORY+'/actions/'), 'API scope mismatch')
    req=urllib.request.Request('https://api.github.com'+path, headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'bounded-cosyvoice-pilot'})
    with urllib.request.urlopen(req,timeout=30) as r: data=r.read(8*1024*1024+1)
    require(len(data)<=8*1024*1024, 'oversized API response')
    return json.loads(data)

def reject_prior_runs(fetch, current):
    """The serialized job consumes its sole admission even if it later fails."""
    for page in range(1,21):
        runs=fetch('/repos/'+REPOSITORY+'/actions/workflows/research-cosyvoice3-pilot.yml/runs?branch='+BRANCH+'&event=pull_request&per_page=100&page='+str(page))['workflow_runs']
        for run in runs:
            if int(run['id'])==current: continue
            require(run.get('head_branch')==BRANCH, 'run branch scope mismatch')
            for jp in range(1,21):
                jobs=fetch('/repos/'+REPOSITORY+'/actions/runs/'+str(run['id'])+'/jobs?filter=all&per_page=100&page='+str(jp))['jobs']
                for job in jobs:
                    if job.get('name')=='one-time-pilot' and job.get('conclusion')!='skipped':
                        raise RuntimeError('A prior one-time pilot job already consumed admission; no automatic retry')
                if len(jobs)<100: break
            else: raise RuntimeError('prior job history exceeds bounded audit')
        if len(runs)<100: return
    raise RuntimeError('prior run history exceeds bounded audit')

def main():
    root=ROOT
    event=json.loads(pathlib.Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    head=git(root,'rev-parse','HEAD')
    require(git(root,'status','--porcelain')=='', 'working tree not clean')
    git(root,'fetch','--no-tags','--depth=1','origin',BASE)
    paths=git(root,'diff','--name-only',BASE,head).splitlines()
    lock=runtime_lock.load()
    approved=validate(event,os.environ,head,identity(root),paths,lock)
    token=os.environ.pop('GH_READ_TOKEN')
    reject_prior_runs(lambda path:api(path,token),int(os.environ['GITHUB_RUN_ID']))
    marker=pathlib.Path(os.environ['RUNNER_TEMP'])/'cosyvoice3-admitted.json'
    with marker.open('x') as f: json.dump(approved,f,sort_keys=True)
    print(json.dumps(approved,sort_keys=True))
if __name__=='__main__': main()
