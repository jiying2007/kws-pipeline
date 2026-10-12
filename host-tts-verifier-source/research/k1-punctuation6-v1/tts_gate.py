"""New TTS-only byte/identity gate. Supplied release is currently disabled."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

from contract import canonical, expected_plan, require, validate_plan
from tts_common import sha, unaliased

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REPOSITORY = "jiying2007/kws-data"
SOURCE_BRANCH = "research/k1-punctuation6-tts-v1"
RESULT_BRANCH = "data/k1-punctuation6-tts-v1"
WORKFLOW = ".github/workflows/k1-punctuation6-tts-v1.yml"
HEAD = re.compile(r"[0-9a-f]{40}\Z")


def read_json(path, maximum=1024*1024):
    p=unaliased(path,file=True)
    require(p.stat().st_size<=maximum,"bounded metadata")
    raw=p.read_bytes()
    def pairs(items):
        out={}
        for key,value in items:
            require(key not in out,"duplicate JSON key")
            out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(ValueError("nonfinite JSON")))


def source_members(root=ROOT):
    root=Path(root)
    files=[]
    for directory in (root/"research/k1-punctuation6-v1",root/"research/qwen6_tts"):
        require(directory.is_dir(),"complete staged source directory")
        for path in directory.rglob("*"):
            require(not path.is_symlink(),"source aliases prohibited")
            if path.is_dir():
                require(path.name != "__pycache__", "source bytecode cache prohibited")
            else:
                unaliased(path,file=True)
                if path != root/"research/k1-punctuation6-v1/tts-source-freeze.json":
                    files.append(path.relative_to(root).as_posix())
    files.append(WORKFLOW)
    return sorted(files)


def verify_source(root=ROOT):
    root=Path(root)
    freeze=read_json(root/"research/k1-punctuation6-v1/tts-source-freeze.json")
    require(set(freeze)=={"schema","files"} and freeze["schema"]=="k1-punctuation6-tts-source-freeze-v1","new TTS source freeze schema")
    require(type(freeze["files"]) is dict and source_members(root)==sorted(freeze["files"]),"full source member closure")
    for name,item in freeze["files"].items():
        require(type(item) is dict and set(item)=={"bytes","sha256"},"source identity fields")
        p=unaliased(root/name,file=True)
        require(p.stat().st_size==item["bytes"] and sha(p)==item["sha256"],"frozen source byte drift")
    validate_plan(read_json(root/"research/k1-punctuation6-v1/plan.proposed.json"))
    return {"source_freeze_sha256":sha(root/"research/k1-punctuation6-v1/tts-source-freeze.json"),"frozen_files":len(freeze["files"])}


def verify_release(root=ROOT):
    root=Path(root)
    value=read_json(root/"research/k1-punctuation6-v1/tts-release.json")
    require(set(value)=={"schema","experiment","enabled","fixture_write_readback_verified","fixture_result_commit",
        "fixture_source_commit","fixture_archive_sha256","fixture_publisher_source_hashes","source_branch","result_branch",
        "maximum_tts_attempts","asr_enabled","training_enabled","independent_source_review","root_execution_release",
        "historical_audio_inventory_frozen"},"exact TTS release fields")
    require(value.get("schema")=="k1-punctuation6-tts-release-v1" and value.get("experiment")=="k1-punctuation6-v1","new release schema")
    require(value.get("enabled") is True,"new TTS execution is disabled")
    require(value.get("fixture_write_readback_verified") is True and
            type(value.get("fixture_result_commit")) is str and HEAD.fullmatch(value["fixture_result_commit"]),
            "actual same-repository writer preflight readback required")
    from contract import SHA
    require(type(value.get("fixture_source_commit")) is str and HEAD.fullmatch(value["fixture_source_commit"]) and
            type(value.get("fixture_archive_sha256")) is str and SHA.fullmatch(value["fixture_archive_sha256"]),"fixture immutable source/output binding")
    writer_names=("data_transport.py","data_publisher.py","tts_publish.py","tts_common.py","tts_gate.py")
    writer_hashes={name:sha(root/"research/k1-punctuation6-v1"/name) for name in writer_names}
    require(value.get("fixture_publisher_source_hashes")==writer_hashes,"same reviewed writer bytes as actual fixture proof")
    require(value.get("source_branch")==SOURCE_BRANCH and value.get("result_branch")==RESULT_BRANCH and
            value.get("maximum_tts_attempts")==6 and type(value.get("maximum_tts_attempts")) is int and
            value.get("asr_enabled") is False and value.get("training_enabled") is False,
            "fixed TTS-only scope")
    require(value.get("independent_source_review") is True and value.get("root_execution_release") is True and
            value.get("historical_audio_inventory_frozen") is True,
            "source/duplicate/history/root execution release required")
    return value,verify_source(root)


def fresh_context(env,event):
    require(env.get("GITHUB_REPOSITORY")==REPOSITORY and env.get("GITHUB_EVENT_NAME")=="push" and
            env.get("GITHUB_REF")=="refs/heads/"+SOURCE_BRANCH and env.get("GITHUB_JOB")=="k1-tts" and
            env.get("GITHUB_RUN_ATTEMPT")=="1" and env.get("RUNNER_ENVIRONMENT")=="github-hosted" and
            env.get("RUNNER_OS")=="Linux" and env.get("RUNNER_ARCH")=="X64", "fixed standard hosted context")
    require(type(event) is dict and event.get("created") is True and event.get("deleted") is False and
            event.get("ref")==env["GITHUB_REF"] and event.get("after")==env.get("GITHUB_SHA") and
            event.get("repository",{}).get("full_name")==REPOSITORY and
            event.get("repository",{}).get("private") is False,"fresh public source branch push required")
    require(type(env.get("GITHUB_SHA")) is str and HEAD.fullmatch(env["GITHUB_SHA"]),"actual source commit")
    require(type(env.get("GITHUB_RUN_ID")) is str and re.fullmatch(r"[1-9][0-9]{0,19}",env["GITHUB_RUN_ID"]),"numeric run identity")
    return {"repository":REPOSITORY,
            "executed_head_sha":env["GITHUB_SHA"],"run_id":int(env["GITHUB_RUN_ID"]),
            "run_attempt":1,"job_key":"k1-tts"}


def verify_clean_checkout(identity,root=ROOT):
    root=Path(root)
    git=lambda *args:subprocess.run(["git","-C",str(root),*args],check=True,capture_output=True,text=True,env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8","LC_ALL":"C.UTF-8","GIT_CONFIG_NOSYSTEM":"1","GIT_CONFIG_GLOBAL":"/dev/null"}).stdout.strip()
    require(git("rev-parse","HEAD")==identity["executed_head_sha"],"actual checkout commit")
    require(not git("status","--porcelain","--untracked-files=all"),"clean complete checkout required before runtime")
    return {"commit":identity["executed_head_sha"],"tree":git("rev-parse","HEAD^{tree}")}


def read_only_preflight(identity, get):
    """Fixed public GET protocol; no token and no claims based only on env strings."""
    base="https://api.github.com/repos/"+REPOSITORY
    run_url=base+"/actions/runs/"+str(identity["run_id"])
    repo=get(base)
    require(repo.get("full_name")==REPOSITORY and repo.get("private") is False,"API public repository")
    run=get(run_url)
    for key,value in (("id",identity["run_id"]),("run_attempt",1),("head_sha",identity["executed_head_sha"]),
                      ("head_branch",SOURCE_BRANCH),("event","push"),("path",WORKFLOW),("status","in_progress")):
        require(type(run.get(key)) is type(value) and run[key]==value,"API current run identity")
    history=get(base+"/actions/workflows/"+WORKFLOW.split("/")[-1]+"/runs?branch="+SOURCE_BRANCH+"&event=push&per_page=100")
    require(type(history.get("total_count")) is int and history["total_count"]==1 and
            type(history.get("workflow_runs")) is list and len(history["workflow_runs"])==1 and
            history["workflow_runs"][0].get("id")==identity["run_id"] and
            history["workflow_runs"][0].get("path")==WORKFLOW and
            history["workflow_runs"][0].get("head_sha")==identity["executed_head_sha"],"fixed source branch consumed before or history incomplete")
    jobs=get(run_url+"/attempts/1/jobs?per_page=100")
    require(type(jobs.get("total_count")) is int and 1<=jobs["total_count"]<=100 and
            type(jobs.get("jobs")) is list and len(jobs["jobs"])==jobs["total_count"],"complete bounded job list")
    selected=[row for row in jobs["jobs"] if row.get("name")=="k1-tts"]
    require(len(selected)==1,"single nonmatrix TTS job")
    job=selected[0]
    require(type(job.get("id")) is int and job["id"]>0 and job.get("status")=="in_progress" and
            job.get("conclusion") is None and job.get("head_sha")==identity["executed_head_sha"] and
            job.get("run_id")==identity["run_id"] and job.get("labels")==["ubuntu-24.04"],"actual standard hosted job identity")
    get(base+"/git/ref/heads/"+RESULT_BRANCH, expected=404)
    return {"preflight_verified":True,"numeric_job_id":job["id"],"source_branch_workflow_runs":1,
            "repository_public":True,"result_ref_absent":True,"read_only_requests":5}


def public_reader(identity,deadline):
    base="https://api.github.com/repos/"+REPOSITORY
    run=base+"/actions/runs/"+str(identity["run_id"])
    allowed={base,run,base+"/actions/workflows/"+WORKFLOW.split("/")[-1]+"/runs?branch="+SOURCE_BRANCH+"&event=push&per_page=100",
             run+"/attempts/1/jobs?per_page=100",base+"/git/ref/heads/"+RESULT_BRANCH}
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs): return None
    opener=urllib.request.build_opener(NoRedirect)
    attempted=set();measurements={"body_bytes":0,"requests":0,"retries":0}
    def get(url,expected=200):
        require(url in allowed and url not in attempted and expected in (200,404),"fixed no-retry public preflight path")
        attempted.add(url);measurements["requests"]+=1
        remaining=deadline-time.monotonic();require(remaining>0,"preflight shared deadline")
        request=urllib.request.Request(url,headers={"Accept":"application/vnd.github+json","User-Agent":"kws-k1-fixed-preflight"})
        try:
            response=opener.open(request,timeout=min(15,remaining))
        except urllib.error.HTTPError as error:
            if expected==404 and error.code==404:
                with error:
                    body=error.read(1024*1024+1)
                    require(len(body)<=1024*1024,"bounded absent-ref response")
                    measurements["body_bytes"]+=len(body)
                return None
            raise
        with response:
            require(response.status==expected and response.geturl()==url and response.headers.get("Content-Encoding","identity")=="identity","exact public response no redirect")
            raw=response.read(1024*1024+1)
        require(len(raw)<=1024*1024,"bounded preflight body")
        measurements["body_bytes"]+=len(raw)
        value=json.loads(raw)
        require(type(value) is dict,"preflight JSON object")
        return value
    return get,measurements


def verify_worker_release(runtime,output):
    _,source=verify_release()
    runtime=unaliased(runtime);output=unaliased(output)
    require(runtime.is_absolute() and output==runtime/"publication-members","fixed child output path")
    claim=read_json(output/"claim.json")
    require(claim.get("schema")=="k1-punctuation6-host-claim-v1" and claim.get("phase")=="tts" and
            claim.get("source_freeze_sha256")==source["source_freeze_sha256"] and
            claim.get("preflight_verified") is True,"source-bound host claim required")
    require(claim.get("identity",{}).get("repository")==REPOSITORY and
            claim.get("source_branch")==SOURCE_BRANCH and
            claim.get("identity",{}).get("run_attempt")==1,"new one-use child identity")
    return claim
