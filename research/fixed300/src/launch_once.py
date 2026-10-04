"""Fail-closed one-release CPU training supervisor. No local/default execution."""
import argparse,math,os,platform,resource,shutil,signal,subprocess,sys,time
from pathlib import Path
from common import require,safe,read_json,write_json,digest
from guard_reuse import phase,accounted_cpu,TOTAL_CPU,TOTAL_WALL,AS_RSS,OUTPUT_CAP
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'src/deps'))
import exact_wheels
ATTEMPT_KEY='a20-fixed300-single-candidate-v1'
BRANCH='research/a20-fixed300-single-candidate-v1'
WORKFLOW='.github/workflows/a20-fixed300.yml'
REPOSITORY='jiying2007/kws-pipeline'

def github_admission(release,env):
    expected={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':REPOSITORY,'GITHUB_EVENT_NAME':'push',
        'GITHUB_REF':'refs/heads/'+BRANCH,'GITHUB_RUN_ATTEMPT':'1',
        'GITHUB_WORKFLOW_REF':REPOSITORY+'/'+WORKFLOW+'@refs/heads/'+BRANCH}
    require(all(env.get(k)==v for k,v in expected.items()),'exact GitHub event/ref/workflow/attempt')
    head=env.get('GITHUB_SHA','');run_id=env.get('GITHUB_RUN_ID','')
    require(len(head)==40 and all(c in '0123456789abcdef'for c in head)and run_id.isdigit(),'GitHub identity')
    require(env.get('GITHUB_WORKFLOW_SHA')==head,'workflow code head')
    event=read_json(Path(env['GITHUB_EVENT_PATH']),4*1024*1024)
    require(event.get('after')==head and event.get('ref')=='refs/heads/'+BRANCH and event.get('deleted')is False,'exact pushed head')
    require(event.get('repository',{}).get('full_name')==REPOSITORY and event['repository'].get('private')is False,'public exact repository')
    # Source content is authorized before publication. This single push may add
    # only the reviewed package/workflow + freeze + release atop the exact base.
    require(type(release['approved_parent_head'])is str and len(release['approved_parent_head'])==40,'approved publication base')
    git_env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'}
    parent=subprocess.check_output(['/usr/bin/git','rev-parse','HEAD^'],cwd=REPO,env=git_env,text=True,timeout=10).strip()
    got_head=subprocess.check_output(['/usr/bin/git','rev-parse','HEAD'],cwd=REPO,env=git_env,text=True,timeout=10).strip()
    require(parent==release['approved_parent_head']and got_head==head,'published commit and parent')
    changed=subprocess.check_output(['/usr/bin/git','diff-tree','--no-commit-id','--name-only','-r','HEAD'],cwd=REPO,env=git_env,text=True,timeout=10).splitlines()
    freeze=read_json(ROOT/'SOURCE-FREEZE.json')
    allowed={e['path']for e in freeze['files']}|{'research/fixed300/SOURCE-FREEZE.json','research/fixed300/EXECUTION-RELEASE.json'}
    require(set(changed)==allowed,'exact publication file delta')
    return dict(repository=REPOSITORY,head_sha=head,parent_head=parent,workflow=WORKFLOW,event='push',run_id=int(run_id),run_attempt=1,
                operator_once='one preread immutable commit, one branch-ref push; no later pushes or blind retries authorized',attempt_key=ATTEMPT_KEY)

def verify_freeze():
    f=read_json(ROOT/'SOURCE-FREEZE.json');require(f['schema']=='a20-execution-source-freeze-v1','execution freeze')
    seen=set()
    for e in f['files']:
        require(e['path']not in seen,'duplicate frozen path');seen.add(e['path'])
        p=safe(REPO,e['path']);require(p.is_file()and p.stat().st_size==e['bytes']and digest(p)==e['sha256'],'frozen source identity')
    require(WORKFLOW in seen and 'research/fixed300/src/launch_once.py'in seen,'mandatory frozen source')
    return f

def admit(release):
    require(release.get('schema')=='a20-fixed300-release-v1'and release.get('approved')is True and release.get('training_authorized')is True,'explicit reviewed training release')
    require(release.get('attempt_key')==ATTEMPT_KEY,'single fixed attempt key')
    freeze=verify_freeze();h=digest(ROOT/'SOURCE-FREEZE.json');p=digest(ROOT/'PROTOCOL.json')
    require(release['source_freeze_sha256']==h and release['protocol_sha256']==p,'exact release binding')
    for key in ('independent_review','private_freeze_readback'):
        r=release[key];require(r['status']=='PASS'and r['source_freeze_sha256']==h and r['protocol_sha256']==p and len(r['receipt_sha256'])==64,'closed review and readback')
    require(release['public_artifact_rights_privacy_approved']is True and release['artifact_max_bytes']==20*1024**2 and release['artifact_retention_days']==1,'bounded output publication approval')
    prep=read_json(ROOT/'metadata/PREPARED-INPUTS.json');require(prep['status']=='VERIFIED_PUBLIC_PIN'and type(prep['commit'])is str and len(prep['commit'])==40,'verified prepared input pin')
    require(platform.python_implementation()=='CPython'and platform.python_version()=='3.12.3'and platform.machine()=='x86_64'and platform.libc_ver()==('glibc','2.39'),'target Ubuntu CPU runtime')
    require(os.environ.get('ImageOS')=='ubuntu24'and os.environ.get('ImageVersion')=='20260927.320.1','reviewed runner image')
    require(shutil.disk_usage(ROOT).free>=4*1024**3,'free disk floor4GiB')
    return freeze,github_admission(release,os.environ)

def context(name,release_sha):
    obj=dict(schema='a20-fixed300-phase-context-v1',phase=name,source_freeze_sha256=digest(ROOT/'SOURCE-FREEZE.json'),release_sha256=release_sha,training_authorized=True)
    p=ROOT/'work'/(name+'-context.json');write_json(p,obj);return p

def run(release_path):
    release=read_json(release_path);freeze,github=admit(release)
    work=ROOT/'work';work.mkdir(mode=0o700,exist_ok=False);out=work/'artifact';out.mkdir(mode=0o700);(work/'dependencies').mkdir(mode=0o700)
    release_sha=digest(release_path);write_json(work/'EXECUTION-LEDGER.json',dict(schema='a20-fixed300-attempt-v1',status='START_RESERVED_NO_RETRY',release_sha256=release_sha,source_freeze_sha256=digest(ROOT/'SOURCE-FREEZE.json'),github_run=github,once_only=True,retries_allowed=False))
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});start=time.monotonic();cpu_start=accounted_cpu();records=[];failed=False
    resource.setrlimit(resource.RLIMIT_AS,(AS_RSS,AS_RSS));hard=math.ceil(time.process_time()+TOTAL_CPU)+1;resource.setrlimit(resource.RLIMIT_CPU,(hard,hard))
    def deadline(*unused):raise RuntimeError('GLOBAL_WALL_LIMIT')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(TOTAL_WALL)
    def execute(name,argv,**kwargs):
        nonlocal failed
        if failed:return False
        r=phase(name,argv,start=start,cpu_start=cpu_start,**kwargs);records.append(r);failed=r['status']!='PASS';return not failed
    try:
        python=sys.executable
        execute('wheels',[python,'-B',str(ROOT/'src/dependency_phase.py'),'--action','wheels','--execute-context',str(context('wheels',release_sha))],phase_wall=300,file_cap=256*1024**2)
        r=exact_wheels.release_template();r['approved']=True
        if not failed:execute('venv',exact_wheels.build_venv_argv(work/'dependencies',r),max_processes=3,max_threads=3,phase_wall=60,file_cap=256*1024**2)
        if not failed:execute('install',exact_wheels.build_install_argv(work/'dependencies',r),max_processes=2,max_threads=2,phase_wall=180,file_cap=1024**3)
        vp=work/'dependencies/venv/bin/python'
        if not failed:execute('verify_install',[str(vp),'-B',str(ROOT/'src/dependency_phase.py'),'--action','verify_install','--execute-context',str(context('verify_install',release_sha))],phase_wall=30)
        if not failed:execute('inputs',[python,'-B',str(ROOT/'src/fetch_inputs.py'),'--execute-context',str(context('inputs',release_sha))],phase_wall=300,file_cap=8*1024**2)
        if not failed:execute('training',[str(vp),'-B',str(ROOT/'src/train_once.py'),'--execute-context',str(context('training',release_sha))],file_cap=8*1024**2)
        verify_freeze()
    except BaseException as e:
        failed=True
        from safe_failure import describe
        write_json(out/'safe-failure.json',describe(ROOT,e,'orchestration','SUPERVISOR'))
    metrics=dict(wall_ms=round((time.monotonic()-start)*1000),cpu_ms=round((accounted_cpu()-cpu_start)*1000))
    if metrics['wall_ms']>TOTAL_WALL*1000 or metrics['cpu_ms']>TOTAL_CPU*1000:failed=True
    signal.alarm(0);status='FAILED_NO_RETRY'if failed else 'SUCCESS'
    from artifacts import journal_counts,package,collect_phase_logs
    counts=journal_counts(out/'call-journal.jsonl')
    if status=='SUCCESS':require(counts['completed']==dict(forward=301,backward=300,update=300)and not counts['partial_final_record'],'successful exact calls')
    write_json(out/'final-status.json',dict(schema='a20-fixed300-final-status-v1',status=status,counts=counts,native_parity='NOT_RUN',acoustic_endpoint_evaluation='NOT_RUN',retry_allowed=False,resume_allowed=False,failed_before_training=not any(r['name']=='training'for r in records)))
    write_json(out/'resources.json',dict(schema='a20-fixed300-resources-v1',status=status,phases=records,guarded=metrics,limits=dict(cpu_seconds=300,wall_seconds=600,address_space_per_process_bytes=AS_RSS,sampled_child_group_RSS_bytes=AS_RSS,raw_artifact_bytes=OUTPUT_CAP),cpu_scope='controller and reaped owned phases through source recheck; final evidence/packaging excluded',controller_RSS='not sampled;2GiBAS enforced',io_null='UNKNOWN',sampling_may_miss_transients=True),1024**2)
    deps={}
    for key,name in [('wheel_inspection','wheel-inspection.json'),('wheel_progress','wheel-progress.json'),('installation','installed.json')]:
        p=work/name;deps[key]=read_json(p)if p.exists()else dict(status='NOT_COMPLETED')
    write_json(out/'dependencies.json',deps)
    cpuinfo=Path('/proc/cpuinfo').read_text();cpu_models=sorted({s.split(':',1)[1].strip()for s in cpuinfo.splitlines()if s.startswith('model name')})
    write_json(out/'environment.json',dict(python_version=platform.python_version(),architecture=platform.machine(),libc=platform.libc_ver(),runner_image=os.environ.get('ImageVersion'),cpu_models=cpu_models,affinity=sorted(os.sched_getaffinity(0)),thread_variables=1,GPU=False,github=github,system_python_sha256=digest(Path(sys.executable).resolve())))
    write_json(out/'source-identities.json',dict(source_freeze_sha256=digest(ROOT/'SOURCE-FREEZE.json'),protocol_sha256=digest(ROOT/'PROTOCOL.json'),release_sha256=release_sha,dependency_lock_sha256=exact_wheels.LOCK_SHA256,prepared_input_manifest_sha256=digest(ROOT/'metadata/PREPARED-INPUTS.json'),model_input_manifest_sha256=digest(ROOT/'metadata/MODEL-INPUTS.json'),train32_sha256=digest(ROOT/'metadata/TRAIN32.json')))
    rights=read_json(ROOT/'metadata/DERIVED-OUTPUT-RIGHTS-PROVENANCE.json');write_json(out/'rights-and-provenance.json',dict(commercial_output_license='not-established',distribution_scope='bounded-public-research-candidate-only',new_license_grant=False,historical_preparation_source_metadata=rights['source_metadata'],historical_preparation_notices=rights['required_output_notices'],historical_scope_note='Original zero-update restrictions describe the preparation run only; they are not the new training release.',current_scope=read_json(ROOT/'metadata/TRAINING-OUTPUT-SCOPE.json'),release_sha256=release_sha))
    collect_phase_logs(ROOT);package(ROOT,status);write_json(work/'EXECUTION-TERMINAL.json',dict(status=status,release_sha256=release_sha,resume_allowed=False))
    print('Fixed300 '+status+'; terminal qualification not implied');return 1 if failed else 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-release',type=Path);a=p.parse_args()
    if a.execute_release is None:raise SystemExit('PREPARATION_ONLY: no exact reviewed execution release')
    raise SystemExit(run(a.execute_release))
