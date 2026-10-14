"""Small campaign-specific first-run gate and exact preset artifact restoration.

Disabled binding is rejected before network access. No inference occurs here.
Model host is a separate token-free step after the fixed lifetime-run check.
"""
import argparse,hashlib,json,os,re,shutil,stat,sys,urllib.error,urllib.parse,urllib.request,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
from campaign import expected_plan
from screen import require,write,third_policy,THIRD_POLICIES
REPO='jiying2007/kws-pipeline'
BRANCH='research/tts-pilot-asr-20261014-v1'
WORKFLOW='.github/workflows/tts-pilot-asr-20261014-v1.yml'
SHA=re.compile(r'[0-9a-f]{64}\Z');HEAD=re.compile(r'[0-9a-f]{40}\Z')
MAX_ZIP=64*1024**2

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def bounded_read(response,cap):
    value=response.read(cap+1);require(len(value)<=cap,'response cap');return value

def api(path,token):
    require(path.startswith('/repos/'+REPO+'/'),'fixed repository API only')
    request=urllib.request.Request('https://api.github.com'+path,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'},method='GET')
    # Metadata must never redirect while carrying the API bearer token.
    with urllib.request.build_opener(NoRedirect).open(request,timeout=45) as r:
        return json.loads(bounded_read(r,4*1024**2))

def binding(path):
    value=json.loads(Path(path).read_bytes())
    third_policy(value)
    require(value.get('enabled') is True,'ASR activation is disabled; real TTS evidence not bound')
    require(value.get('schema')=='tts-pilot-asr-activation-v1' and value.get('repository')==REPO and value.get('branch')==BRANCH and value.get('workflow_path')==WORKFLOW,'fixed new campaign')
    require(value.get('third_policy') in THIRD_POLICIES,'explicit third policy in activation binding')
    require(value.get('max_asr_slots')==144 and value.get('current_qwen_call_cap')==96,'fixed ASR caps')
    artifacts=value.get('input_artifacts');require(type(artifacts) is list and len(artifacts)==4,'two lanes x probe/remaining real artifacts')
    require({(a['engine'],a['phase']) for a in artifacts}=={(e,p) for e in ('qwen06','qwen17') for p in ('probe','remaining')},'exact input phases')
    require(type(value.get('workflow_id')) is int and value['workflow_id']>0,'real installed workflow ID required')
    for a in artifacts:
        if a.get('state')=='not_run':
            require(a['phase']=='remaining' and set(a)=={'engine','phase','state','reason'} and a['reason']=='verified_probe_failed','explicit no-call remaining only')
            probe=next(p for p in artifacts if p['engine']==a['engine'] and p['phase']=='probe')
            require(probe.get('state')=='recorded_run' and probe.get('lane_outcome')=='failed','NOT_RUN must bind actual failed probe')
            continue
        require(a.get('state')=='recorded_run','explicit real run state')
        require(type(a['run_id']) is int and a['run_id']>0 and type(a['artifact_id']) is int and a['artifact_id']>0,'real run/artifact IDs')
        require(HEAD.fullmatch(a['head_sha']),'actual source identity')
        require(SHA.fullmatch(a['zip_sha256']) and type(a['artifact_name']) is str and bool(a['artifact_name']),'actual archive identity')
        require(a['conclusion'] in ('success','failure'),'actual terminal result binding')
        if a['phase']=='probe':require(a.get('lane_outcome') in ('succeeded','failed'),'explicit per-lane probe result distinct from workflow conclusion')
        require(a['run_attempt']==1 and a['event']=='push' and a['workflow_path'].startswith('.github/workflows/') and a['head_branch'].startswith('research/'),'source execution identity')
        require(type(a['members']) is dict and bool(a['members']),'exact archive member inventory')
        for name,row in a['members'].items():
            require(type(name) is str and name and not Path(name).is_absolute() and '\\' not in name and all(p not in ('','.','..') for p in name.split('/')),'safe archive member')
            require(set(row)=={'sha256','bytes','restore_name'} and SHA.fullmatch(row['sha256']) and type(row['bytes']) is int and 0<=row['bytes']<=MAX_ZIP,'member identity')
            allowed={a['engine']+'-receipt.json','host-receipt.json'}|{c['cell_id']+'.raw-float.wav' for c in expected_plan()[a['engine']]}
            require(row['restore_name'] in allowed or (row['restore_name'] is None and Path(name).suffix in ('.json','.claim')),'restore only preset receipt/audio and host status; audited extra metadata hash-checked then discarded')
        require(len({r['restore_name'] for r in a['members'].values() if r['restore_name'] is not None})==sum(r['restore_name'] is not None for r in a['members'].values()),'no duplicate restored names')
    source=value.get('source_files');require(type(source) is dict and bool(source),'reviewed source inventory required')
    for engine in ('qwen06','qwen17'):
        probe=next(a for a in artifacts if a['engine']==engine and a['phase']=='probe')
        remaining=next(a for a in artifacts if a['engine']==engine and a['phase']=='remaining')
        require((probe['lane_outcome']=='failed')==(remaining['state']=='not_run'),'failed probe prohibits remaining; successful probe requires actual remaining')
    for name,digest in source.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and SHA.fullmatch(digest),'source inventory identity')
    required={WORKFLOW,'research/source-screen32-v1/container-lock.json'}
    required.update('research/tts-pilot-screening-v1/'+p.name for p in BASE.iterdir() if p.is_file() and p.suffix in ('.py','.json'))
    required.update('research/tts-pilot-screening-v1/activation-template/'+n for n in ('launcher.py','finalize_input.py','test_activation.py'))
    require(required<=set(source),'complete reviewed new source inventory required')
    require(value.get('numeric_requirements_file') in source,'pinned numeric requirements in source inventory')
    return value

def verify_current(value,env,event,run):
    require(env.get('GITHUB_REPOSITORY')==REPO and env.get('GITHUB_EVENT_NAME')=='push' and env.get('GITHUB_REF')=='refs/heads/'+BRANCH,'fixed first push only')
    require(env.get('GITHUB_RUN_ATTEMPT')=='1' and env.get('GITHUB_RUN_NUMBER')=='1' and event.get('created') is True and event.get('deleted') is False and event['repository']['private'] is False,'fresh public first execution')
    require(run.get('workflow_id')==value['workflow_id'],'installed workflow identity cannot reset')
    require(run['id']==int(env['GITHUB_RUN_ID']) and run['head_sha']==env['GITHUB_SHA'] and run['head_branch']==BRANCH and run['path']==WORKFLOW and run['run_attempt']==1 and run['run_number']==1 and run['event']=='push','server current run binding')

def verify_upstream(a,run,artifact):
    require(run['id']==a['run_id'] and run['head_sha']==a['head_sha'] and run['head_branch']==a['head_branch'] and run['path']==a['workflow_path'] and run['run_attempt']==a['run_attempt'] and run['event']==a['event'],'upstream run/source binding')
    require(run['status']=='completed' and run['conclusion']==a['conclusion'],'upstream terminal state')
    require(artifact['id']==a['artifact_id'] and artifact['name']==a['artifact_name'] and artifact['expired'] is False and artifact['workflow_run']['id']==a['run_id'] and artifact['workflow_run']['head_sha']==a['head_sha'],'artifact/run identity')
    require(artifact['digest']=='sha256:'+a['zip_sha256'] and 0<artifact['size_in_bytes']<=MAX_ZIP,'server archive digest/size')

def download(a,token,dest):
    # Do not forward API Authorization to the returned signed storage URL.
    url='https://api.github.com/repos/'+REPO+'/actions/artifacts/'+str(a['artifact_id'])+'/zip'
    request=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
    try:
        with urllib.request.build_opener(NoRedirect).open(request,timeout=45) as response:raw=bounded_read(response,MAX_ZIP)
    except urllib.error.HTTPError as error:
        require(error.code in (301,302,303,307,308),'artifact download status')
        location=error.headers['Location'];parsed=urllib.parse.urlsplit(location)
        require(parsed.scheme=='https' and parsed.hostname and parsed.username is None and parsed.password is None,'secure artifact handoff')
        with urllib.request.urlopen(urllib.request.Request(location),timeout=90) as response:raw=bounded_read(response,MAX_ZIP)
    require(hashlib.sha256(raw).hexdigest()==a['zip_sha256'],'download ZIP digest')
    with Path(dest).open('xb') as f:f.write(raw)

def restore(a,archive,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();require(len(infos)==len(a['members']) and len({i.filename for i in infos})==len(infos),'exact unique ZIP members')
        require({i.filename for i in infos}==set(a['members']),'archive membership')
        require(sum(i.file_size for i in infos)<=MAX_ZIP,'uncompressed archive bound')
        for i in infos:
            row=a['members'][i.filename];mode=i.external_attr>>16
            require(not i.is_dir() and not stat.S_ISLNK(mode) and i.file_size==row['bytes'],'regular bounded ZIP file')
            with z.open(i) as f:raw=bounded_read(f,row['bytes'])
            require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'member content binding')
            if row['restore_name'] is not None:
                with (dest/row['restore_name']).open('xb') as f:f.write(raw)

def verify_probe_lane(item,restored):
    receipt=json.loads((Path(restored)/'host-receipt.json').read_bytes())
    expected='stopped_no_retry' if item['lane_outcome']=='failed' else 'probe_completed_pending_human_review'
    require(receipt['engine']==item['engine'] and receipt['status']==expected and receipt['sha']==item['head_sha'] and receipt['campaign']=='tts-quality-probe-20261014-v1','actual per-lane probe host receipt')

def combine(work,dest,artifacts):
    plan=expected_plan();dest=Path(dest)
    for engine in ('qwen06','qwen17'):
        state=next(a for a in artifacts if a['engine']==engine and a['phase']=='remaining')
        if state['state']=='not_run':continue
        probe=work/(engine+'-probe');remaining=work/(engine+'-remaining');name=engine+'-receipt.json'
        p=json.loads((probe/name).read_bytes());r=json.loads((remaining/name).read_bytes())
        require(p['engine']==r['engine']==engine and len(p['outcomes'])==1 and p['outcomes'][0]['cell_id']==plan[engine][0]['cell_id'],'single included probe identity')
        require(r['outcomes'] and r['outcomes'][0]==p['outcomes'][0],'remaining cumulative receipt preserves original probe unchanged')
        target=dest/('tqp-'+engine)/'artifact';target.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(remaining/name,target/name)
        for folder in (probe,remaining):
            for path in folder.iterdir():
                if not path.name.endswith('.raw-float.wav'):continue
                existing=target/path.name
                if existing.exists():require(existing.read_bytes()==path.read_bytes(),'duplicate probe audio must be identical')
                else:shutil.copyfile(path,existing)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binding',default=str(HERE/'activation.json'));parser.add_argument('--output',required=True);a=parser.parse_args()
    value=binding(a.binding) # Must fail disabled before any request.
    root=BASE.parents[1];env=os.environ;token=env['GITHUB_TOKEN'];event=json.loads(Path(env['GITHUB_EVENT_PATH']).read_bytes())
    for name,digest in value['source_files'].items():require(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'reviewed source changed')
    current=api('/repos/'+REPO+'/actions/runs/'+env['GITHUB_RUN_ID'],token);verify_current(value,env,event,current)
    history=api('/repos/'+REPO+'/actions/workflows/'+str(value['workflow_id'])+'/runs?per_page=2',token)
    require(history['total_count']==1 and len(history['workflow_runs'])==1,'one lifetime run for this frozen workflow ID')
    verify_current(value,env,event,history['workflow_runs'][0])
    output=Path(a.output);require(not output.exists(),'fresh staging directory');output.mkdir(parents=True)
    for item in value['input_artifacts']:
        if item['state']=='not_run':continue
        run=api('/repos/'+REPO+'/actions/runs/'+str(item['run_id']),token)
        artifact=api('/repos/'+REPO+'/actions/artifacts/'+str(item['artifact_id']),token);verify_upstream(item,run,artifact)
        archive=output/(item['engine']+'-'+item['phase']+'.zip');download(item,token,archive);restored=output/(item['engine']+'-'+item['phase']);restore(item,archive,restored)
        if item['phase']=='probe':verify_probe_lane(item,restored)
    (output/'inputs').mkdir()
    combine(output,output/'inputs',value['input_artifacts'])
    failed=[a['engine'] for a in value['input_artifacts'] if a['phase']=='probe' and a['lane_outcome']=='failed']
    write(output/'admitted.json',{'campaign':BRANCH,'workflow_id':value['workflow_id'],'run_id':int(env['GITHUB_RUN_ID']),'head_sha':env['GITHUB_SHA'],'failed_probe_lanes':failed,'third_policy':value['third_policy'],'max_asr_slots':144,'current_qwen_call_cap':96})
if __name__=='__main__':main()
