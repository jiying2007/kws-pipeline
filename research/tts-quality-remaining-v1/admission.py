"""Only workflow step receiving the ephemeral read-only GitHub token."""
from pathlib import Path
import hashlib,io,json,os,sys,urllib.request,urllib.error,urllib.parse,zipfile
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import remaining as r
REPO='jiying2007/kws-pipeline'
ROOT=HERE.parents[1]
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def api(path,token):
    req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,headers={
        'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.build_opener(NoRedirect()).open(req,timeout=30) as f:data=f.read(2*1024**2+1)
    r.require(len(data)<=2*1024**2,'API metadata bound')
    return json.loads(data)

def verify_metadata(b,run,job,artifact):
    r.binding_valid(b,b['engine'])
    r.require((run.get('id'),run.get('head_sha'),run.get('path'),run.get('event'),run.get('run_number'),run.get('run_attempt'),run.get('status'),run.get('conclusion'))==
              (b['probe_run_id'],b['probe_source_sha'],b['workflow_path'],'push',1,1,'completed',b['probe_run_conclusion']),'authenticated run mismatch')
    r.require(run.get('repository',{}).get('full_name')==REPO,'authenticated repository mismatch')
    r.require((job.get('id'),job.get('run_id'),job.get('head_sha'),job.get('run_attempt'),job.get('name'),job.get('status'),job.get('conclusion'))==
              (b['probe_job_id'],b['probe_run_id'],b['probe_source_sha'],1,b['probe_job_name'],'completed','success'),'authenticated target job mismatch')
    r.require((artifact.get('id'),artifact.get('name'),artifact.get('expired'),artifact.get('digest'))==
              (b['artifact_id'],b['artifact_name'],False,'sha256:'+b['artifact_archive_sha256']),'authenticated artifact mismatch')
    r.require((artifact.get('workflow_run',{}).get('id'),artifact.get('workflow_run',{}).get('head_sha'))==
              (b['probe_run_id'],b['probe_source_sha']),'artifact run/source mismatch')

def main():
    engine=sys.argv[1];r.require(engine in r.ENGINES,'engine')
    work=ROOT/('tqremaining-'+engine);work.mkdir(exist_ok=False)
    public=work/'artifact';public.mkdir()
    status={'engine':engine,'status':'started','model_calls':0}
    try:
        b=r.document(HERE/('binding-'+engine+'.json'));r.binding_valid(b,engine)
        token=os.environ.get('GH_TOKEN');r.require(bool(token),'read-only GitHub token required only for admission')
        run=api('/actions/runs/'+str(b['probe_run_id']),token)
        job=api('/actions/jobs/'+str(b['probe_job_id']),token)
        artifact=api('/actions/artifacts/'+str(b['artifact_id']),token)
        verify_metadata(b,run,job,artifact)
        inputs=work/'inputs';inputs.mkdir();archive=inputs/'probe.zip'
        retained=ROOT/'research/effect-data-model-v2/evidence/probe-run-38554125374'/(engine+'.zip')
        body=r.raw(retained,32*1024**2);r.require(r.digest(body)==b['artifact_archive_sha256'],'retained archive mismatch')
        archive.write_bytes(body)
        incoming=inputs/'incoming';incoming.mkdir()
        expected=r.filenames(engine,'tqp-'+engine+'-00')
        members=r.archive_members(archive,expected,None)
        for name,body in members.items():(incoming/name).write_bytes(body)
        r.verify(b,engine,incoming,ROOT,archive,HERE/'source-manifest.json')
        status.update(status='admitted',probe_run_id=b['probe_run_id'],probe_job_id=b['probe_job_id'],artifact_id=b['artifact_id'],source_sha=b['probe_source_sha'],archive_sha256=b['artifact_archive_sha256'])
    except BaseException as e:
        status.update(status='stopped_no_retry',error_type=type(e).__name__);raise
    finally:(public/'admission-status.json').write_text(json.dumps(status,indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        print(json.dumps({'status':'stopped_no_retry','error_type':type(e).__name__}),file=sys.stderr);raise SystemExit(1)
