"""One integrated preparation, reusing reviewed guard primitives. No training route."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import struct
import subprocess
import sys
import time
from common import require,safe,read_json,write_json,digest,write_bytes,replace_json
from supervision_reuse import read_proc,cleanup_owned_process,apply_hard_limits
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'src/deps'))
import exact_wheels
ATTEMPT_KEY='a20-token-preparation-setup-recovery-20261004-v1'
BRANCH='research/a20-token-preparation-metadata-recovery-20261004'
WORKFLOW='.github/workflows/a20-token-preparation.yml'
REPOSITORY='jiying2007/kws-pipeline'
TOTAL_CPU=300
TOTAL_WALL=600
AS_RSS=2*1024**3
OUTPUT_CAP=32*1024**2


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
    allowed={e['path']for e in freeze['files']}|{'research/token_preparation/SOURCE-FREEZE.json','research/token_preparation/EXECUTION-RELEASE.json'}
    require(set(changed)==allowed,'exact publication file delta')
    return dict(repository=REPOSITORY,head_sha=head,parent_head=parent,workflow=WORKFLOW,event='push',run_id=int(run_id),run_attempt=1,
                operator_once='one preread immutable commit, one branch-ref push; no later pushes or blind retries authorized',attempt_key=ATTEMPT_KEY)


def verify_freeze():
    f=read_json(ROOT/'SOURCE-FREEZE.json');require(f['schema']=='a20-execution-source-freeze-v1','execution freeze')
    seen=set()
    for e in f['files']:
        require(e['path']not in seen,'duplicate frozen path');seen.add(e['path'])
        p=safe(REPO,e['path']);require(p.is_file()and p.stat().st_size==e['bytes']and digest(p)==e['sha256'],'frozen source identity')
    require(WORKFLOW in seen and 'research/token_preparation/src/launch_once.py'in seen,'mandatory frozen source')
    return f


def admit(release):
    require(release.get('schema')=='a20-integrated-preparation-release-v1'and release.get('approved')is True,'exact preparation release')
    require(release.get('attempt_key')==ATTEMPT_KEY and release.get('training_authorized')is False,'preparation only')
    freeze=verify_freeze();h=digest(ROOT/'SOURCE-FREEZE.json');p=digest(ROOT/'PROTOCOL.json')
    require(release['source_freeze_sha256']==h and release['protocol_sha256']==p,'release binding')
    for key in ('independent_review','private_freeze_readback'):
        r=release[key]
        require(r['status']=='PASS'and r['source_freeze_sha256']==h and r['protocol_sha256']==p
                and len(r['receipt_sha256'])==64,'closed review/readback attestation')
    require(release['public_artifact_rights_privacy_approved']is True and release['artifact_max_bytes']==8*1024**2
            and release['artifact_retention_days']==1,'public artifact release')
    require(platform.python_implementation()=='CPython'and platform.python_version()=='3.12.3'
            and platform.machine()=='x86_64'and platform.libc_ver()==('glibc','2.39'),'target Ubuntu CPU runtime')
    require(os.environ.get('ImageOS')=='ubuntu24'and os.environ.get('ImageVersion')=='20260927.320.1','reviewed runner image')
    github=github_admission(release,os.environ)
    return freeze,github


def clean_env():
    home=ROOT/'work/home';tmp=ROOT/'work/tmp';home.mkdir(mode=0o700,exist_ok=True);tmp.mkdir(mode=0o700,exist_ok=True)
    return {'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','HOME':str(home),'TMPDIR':str(tmp),
        'PIP_CONFIG_FILE':'/dev/null','PYTHONHASHSEED':'610104','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
        'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TORCH_HOME':str(ROOT/'work/torch-home')}


def cpu_ticks(pid):
    text=Path('/proc')/str(pid)/'stat';parts=text.read_text().rsplit(')',1)[1].split()
    return sum(int(parts[i])for i in (11,12,13,14))/os.sysconf('SC_CLK_TCK')


def observe_group(proc,max_processes):
    if proc.poll()is not None:return None
    pending=[proc.pid];seen=set();states=[];cpu=0.
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        seen.add(pid)
        require(len(seen)<=max_processes,'PROCESS_LIMIT')
        try:
            require(os.getpgid(pid)==proc.pid,'OWNED_CHILD_GROUP_ESCAPE')
            s=read_proc(pid)
            errors=list(s['critical_errors'].values())
            if s.get('children_observation')!='AVAILABLE_AT_SAMPLE' and s.get('children_error'):
                errors.append(s['children_error'])
            if errors:
                safely_missing=all(e['type']in ('FileNotFoundError','ProcessLookupError','MissingProcField')for e in errors)
                ended=proc.poll()is not None if pid==proc.pid else False
                if pid!=proc.pid:
                    try:os.getpgid(pid)
                    except ProcessLookupError:ended=True
                if safely_missing and ended:
                    if pid==proc.pid:return None
                    continue
            require(not s['critical_errors'],'MANDATORY_PROC_FIELDS_UNAVAILABLE')
            require(s['children_observation']=='AVAILABLE_AT_SAMPLE','CHILD_VISIBILITY_REQUIRED')
            cpu+=cpu_ticks(pid);states.append(s)
            pending.extend(int(v)for v in s['children'].split())
        except (FileNotFoundError,ProcessLookupError):
            if pid==proc.pid and proc.poll()is not None:return None
            if pid!=proc.pid:
                try:os.getpgid(pid)
                except ProcessLookupError:continue
            raise
    return dict(rss=sum(s['VmRSS']*1024 for s in states),vmsize=sum(s.get('VmSize',0)*1024 for s in states),
        threads=sum(s['Threads']for s in states),processes=len(states),cpu=cpu,
        io={k:sum(s['io'][k]['value']for s in states)if all(s['io'][k]['status']=='AVAILABLE'for s in states)else None
            for k in ('rchar','wchar','read_bytes','write_bytes')})


def bytes_under(path):
    total=0
    for p in path.rglob('*'):
        require(not p.is_symlink(),'unexpected output symlink')
        if p.is_file():total+=p.stat().st_size
    return total


def child_cpu_total():
    r=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime


def accounted_cpu():
    return child_cpu_total()+time.process_time()


def phase(name,argv,*,start,cpu_start,max_processes=1,max_threads=1,phase_wall=600,file_cap=4*1024**2):
    remaining_wall=min(phase_wall,TOTAL_WALL-(time.monotonic()-start))
    remaining_cpu=TOTAL_CPU-(accounted_cpu()-cpu_start)
    require(remaining_wall>0 and remaining_cpu>0,'GLOBAL_BUDGET_EXHAUSTED')
    record=dict(name=name,status='STARTED',samples=[],sample_fields=['wall_ms','rss_bytes','virtual_bytes','threads','owned_processes','observed_group_cpu_ms','rchar','wchar','read_bytes','write_bytes'],
                mandatory_live_samples=0,optional_io_null_means='UNKNOWN',continuous_lifetime_absence_or_peak_proven=False)
    limits=dict(rss_bytes=AS_RSS,cpu_seconds=max(1,math.ceil(remaining_cpu)),wall_seconds=max(1,math.ceil(remaining_wall)),output_bytes=OUTPUT_CAP,
                collector_threads=max_threads,collector_processes=max_processes)
    own_cpu=min(os.sched_getaffinity(0));t=time.monotonic();before=child_cpu_total();proc=None
    logs=ROOT/'work/logs';logs.mkdir(exist_ok=True)
    try:
        with (logs/(name+'.stdout')).open('xb')as stdout,(logs/(name+'.stderr')).open('xb')as stderr:
            proc=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=clean_env(),start_new_session=True,
                preexec_fn=lambda:apply_hard_limits(own_cpu,limits=limits,file_bytes=file_cap))
            while True:
                elapsed=time.monotonic()-t
                require(elapsed<=remaining_wall and time.monotonic()-start<=TOTAL_WALL,'WALL_LIMIT')
                require(bytes_under(logs)<=512*1024,'LOG_OUTPUT_LIMIT')
                require(bytes_under(ROOT/'work/artifact')<=OUTPUT_CAP,'ARTIFACT_WORK_OUTPUT_LIMIT')
                state=observe_group(proc,max_processes)
                if state is not None:
                    require(state['rss']<=AS_RSS,'RSS_LIMIT');require(state['threads']<=max_threads,'THREAD_LIMIT')
                    require(state['cpu']+accounted_cpu()-cpu_start<=TOTAL_CPU,'CPU_LIMIT')
                    record['mandatory_live_samples']+=1
                    record['samples'].append([round(elapsed*1000),state['rss'],state['vmsize'],state['threads'],state['processes'],round(state['cpu']*1000),
                                               *[state['io'][k]for k in ('rchar','wchar','read_bytes','write_bytes')]])
                rc=proc.poll()
                if rc is not None:break
                time.sleep(.1)
            record['returncode']=rc
            cleanup=cleanup_owned_process(proc,kill=False)
            require(not cleanup['errors']and cleanup['reaped'],'PROCESS_REAP')
            require(rc==0 and record['mandatory_live_samples']>0,'CHILD_FAILURE_OR_NO_LIVE_GUARD_SAMPLE')
            require(accounted_cpu()-cpu_start<=TOTAL_CPU,'AGGREGATE_CPU_LIMIT')
            record['status']='PASS'
    except BaseException as e:
        if proc is not None:
            cleaned=cleanup_owned_process(proc,kill=True);record['returncode']=cleaned.get('returncode')
        from safe_failure import describe
        record.update(status='FAILED_NO_RETRY',safe_supervisor_failure=describe(ROOT,e,name,'OWNED_PROCESS_GUARD'))
        child_receipt=ROOT/'work'/(name+'-safe-error.json')
        if child_receipt.exists():record['safe_child_failure']=read_json(child_receipt,8192)
        else:record['safe_child_failure']=dict(phase=name,stage='CHILD_EXIT',error_code='NO_CHILD_RECEIPT_CAPTURED',exception_type='UNKNOWN')
    record.update(wall_ms=round((time.monotonic()-t)*1000),reaped_child_cpu_ms=round((child_cpu_total()-before)*1000))
    return record


def context(name,release_sha):
    obj=dict(schema='a20-preparation-phase-context-v1',phase=name,source_freeze_sha256=digest(ROOT/'SOURCE-FREEZE.json'),release_sha256=release_sha,training_authorized=False)
    p=ROOT/'work'/(name+'-context.json');write_json(p,obj);return p


def run(release_path):
    release=read_json(release_path);freeze,github=admit(release)
    work=ROOT/'work';work.mkdir(mode=0o700,exist_ok=False)
    out=work/'artifact';out.mkdir(mode=0o700)
    release_sha=digest(release_path)
    write_json(work/'EXECUTION-LEDGER.json',dict(schema='a20-preparation-attempt-v1',status='START_RESERVED_NO_RETRY',release_sha256=release_sha,
        source_freeze_sha256=digest(ROOT/'SOURCE-FREEZE.json'),github_run=github,once_only=True,retries_allowed=False))
    (work/'dependencies').mkdir(mode=0o700)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    start=time.monotonic();cpu_start=accounted_cpu();records=[];failure=False
    # The reused child guards remain primary. Also bound source/ZIP checks in
    # this controller; timeout raises into the same failure/owned cleanup path.
    resource.setrlimit(resource.RLIMIT_AS,(AS_RSS,AS_RSS))
    parent_cpu_hard=math.ceil(time.process_time()+TOTAL_CPU)+1
    resource.setrlimit(resource.RLIMIT_CPU,(parent_cpu_hard,parent_cpu_hard))
    def deadline(signum,frame):raise RuntimeError('GLOBAL_WALL_LIMIT')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(TOTAL_WALL)
    def execute(name,argv,**kwargs):
        nonlocal failure
        if failure:
            records.append(dict(name=name,status='NOT_RUN_AFTER_FAILURE',samples=[]));return False
        r=phase(name,argv,start=start,cpu_start=cpu_start,**kwargs);records.append(r)
        failure=r['status']!='PASS';return not failure
    python=sys.executable
    try:
        execute('wheels',[python,'-B',str(ROOT/'src/dependency_phase.py'),'--action','wheels','--execute-context',str(context('wheels',release_sha))],phase_wall=300,file_cap=256*1024**2)
        r=exact_wheels.release_template();r['approved']=True
        if not failure:execute('venv',exact_wheels.build_venv_argv(work/'dependencies',r),max_processes=3,max_threads=3,phase_wall=60,file_cap=256*1024**2)
        if not failure:execute('install',exact_wheels.build_install_argv(work/'dependencies',r),max_processes=2,max_threads=2,phase_wall=180,file_cap=1024**3)
        vpython=work/'dependencies/venv/bin/python'
        if not failure:execute('verify_install',[str(vpython),'-B',str(ROOT/'src/dependency_phase.py'),'--action','verify_install','--execute-context',str(context('verify_install',release_sha))],phase_wall=30)
        if not failure:execute('inputs',[python,'-B',str(ROOT/'src/fetch_public_inputs.py'),'--execute-context',str(context('inputs',release_sha))],phase_wall=300,file_cap=8*1024**2)
        if not failure:
            build=work/'build';build.mkdir();runtime=work/'inputs/runtime'
            argv=['/usr/bin/gcc','-std=c11','-O2','-shared','-fPIC','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-frounding-math',
                '-ffunction-sections','-fdata-sections','-fvisibility=hidden','-Wl,--gc-sections','-I'+str(runtime/'src'),'-I'+str(ROOT/'src'),
                str(ROOT/'src/feature_collector.c'),str(ROOT/'src/collector_abi.c'),str(runtime/'src/pcm_fft64.c'),str(runtime/'src/donor_fft64.c'),
                str(runtime/'baseline/native/stream/splice.c'),'-lm','-o',str(build/'feature_collector.so')]
            execute('compile',argv,max_processes=4,max_threads=4,phase_wall=60)
        if not failure:
            symbols=subprocess.check_output(['/usr/bin/nm','-D','--defined-only',str(work/'build/feature_collector.so')],env=clean_env(),text=True,timeout=10)
            names={line.split()[-1]for line in symbols.splitlines()};expected={'a20_recipe_workspace_bytes','a20_recipe_rows_count','a20_recipe_calls_count','a20_recipe_rows_pointer','a20_recipe_calls_pointer','a20_recipe_execute'}
            require(names==expected,'frontend-only exported ABI')
            allsymbols=subprocess.check_output(['/usr/bin/nm',str(work/'build/feature_collector.so')],env=clean_env(),text=True,timeout=10)
            require(not any(s in allsymbols.lower()for s in ('cmvn','a20_fsmn','decoder','torch')),'frontend-only link closure')
            write_json(out/'build.json',dict(schema='a20-preparation-frontend-build-v1',sha256=digest(work/'build/feature_collector.so'),exported_symbols=sorted(names),
                source_bytes_verified=True,ffp_contract='off',fno_fast_math=True,model_symbols=0,cmvn_symbols=0,decoder_symbols=0,
                compiler_sha256=digest(Path('/usr/bin/gcc').resolve())))
            execute('features_control',[str(vpython),'-B',str(ROOT/'src/run_features_control.py'),'--execute-context',str(context('features_control',release_sha))])
    except BaseException as e:
        failure=True
        from safe_failure import describe
        records.append(dict(name='orchestration',status='FAILED_NO_RETRY',safe_supervisor_failure=describe(ROOT,e,'orchestration','ARGUMENT_OR_SOURCE_CHECK'),samples=[]))
    # The source freeze and public bindings stay immutable. Never resume a phase.
    try:verify_freeze()
    except BaseException:failure=True;records.append(dict(name='source_recheck',status='FAILED_NO_RETRY',samples=[]))
    guarded_metrics=dict(wall_ms=round((time.monotonic()-start)*1000),accounted_cpu_ms=round((accounted_cpu()-cpu_start)*1000))
    if guarded_metrics['wall_ms']>TOTAL_WALL*1000 or guarded_metrics['accounted_cpu_ms']>TOTAL_CPU*1000:
        failure=True;records.append(dict(name='final_guard_boundary',status='FAILED_NO_RETRY',failure_code='FINAL_GUARDED_BUDGET_EXCEEDED',samples=[]))
    signal.alarm(0)
    status='FAILED'if failure else 'SUCCESS'
    finish_evidence(out,status,records,github,guarded_metrics,release_sha)
    from public_artifact import package_artifact
    packaged=package_artifact(out,ROOT/'publication',status,dict(source_root=REPO,source_freeze_sha256=digest(ROOT/'SOURCE-FREEZE.json'),release_sha256=release_sha,rights_privacy_reviewed=True))
    write_json(work/'PUBLICATION-READY.json',packaged)
    # Ledger terminal append is separate; original exclusive reservation survives.
    write_json(work/'EXECUTION-TERMINAL.json',dict(status=status,training_authorized=False,release_sha256=release_sha))
    print('Integrated preparation '+status+'; training disabled; artifact allowlist checked')
    return 0 if status=='SUCCESS'else 1


def repair_partial_evidence(out):
    """Saved-only partial manifests, never another feature/model calculation.

    Preserve complete finite allowed raw files. Incomplete writes remain in a
    private run-local directory with their hash/size recorded; they are not an
    approved public payload. No missing geometry/CTC metric is fabricated.
    """
    from public_artifact import RAW_SHAPES
    rejected=[];holding=ROOT/'work/incomplete-writes'
    def retain_elsewhere(path,reason):
        holding.mkdir(exist_ok=True)
        relative=str(path.relative_to(out));h=digest(path);n=path.stat().st_size
        destination=holding/relative.replace('/','_')
        os.replace(path,destination)
        rejected.append(dict(file=relative,bytes=n,sha256=h,reason=reason))
    for path in out.glob('*.pending'):retain_elsewhere(path,'INCOMPLETE_ATOMIC_JSON_WRITE')
    for group in ('native','official'):
        rows=[]
        directory=out/group
        if directory.exists():
            for path in sorted(directory.iterdir()):
                relative=group+'/'+path.name
                require(relative in RAW_SHAPES and path.is_file()and not path.is_symlink(),'unexpected partial output')
                raw=path.read_bytes();shape=RAW_SHAPES[relative]
                if len(raw)!=math.prod(shape)*4 or not all(math.isfinite(v[0])for v in struct.iter_unpack('<f',raw)):
                    retain_elsewhere(path,'INCOMPLETE_OR_NONFINITE_FP32');continue
                rows.append(dict(recording=path.stem,file=relative,shape=shape,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                    callback_geometry_status='NOT_RECONSTRUCTED_BY_SAVED_ONLY_FAILURE_REPAIR'))
        if rows or (out/(group+'-features.json')).exists():
            original=out/(group+'-features.json')
            if original.exists():
                try:
                    prior=read_json(original)
                    old={r['file']:r for r in prior.get('rows',[])}
                    rows=[old[r['file']]if r['file']in old and old[r['file']].get('sha256')==r['sha256']else r for r in rows]
                except (ValueError,OSError,KeyError):retain_elsewhere(original,'INCOMPLETE_FEATURE_MANIFEST')
            replace_json(original,dict(schema='a20-saved-only-partial-feature-evidence-v1',status='INCOMPLETE',rows=rows,
                raw_file_count=len(rows),no_new_feature_computation=True))
    rawpath=out/'a20-initial-logits.f32le';control=out/'initial-control.json'
    if rawpath.exists():
        raw=rawpath.read_bytes()
        if len(raw)!=72960 or not all(math.isfinite(v[0])for v in struct.iter_unpack('<f',raw)):
            retain_elsewhere(rawpath,'INCOMPLETE_OR_NONFINITE_CONTROL_LOGITS')
        elif not control.exists():
            write_json(control,dict(status='INCOMPLETE',raw_file=rawpath.name,raw_shape=[32,95,6],raw_sha256=hashlib.sha256(raw).hexdigest(),
                model_forwards=1,count_basis='Complete logits are written only after the one-forward control returns; remaining control metadata was not saved.',
                missing_metrics='UNKNOWN_NOT_RECOMPUTED'))
    return rejected


def finish_evidence(out,status,records,github,guarded_metrics,release_sha):
    rejected=repair_partial_evidence(out)if status=='FAILED'else []
    native=list((out/'native').glob('*.f32le'))if (out/'native').exists()else []
    official=list((out/'official').glob('*.f32le'))if (out/'official').exists()else []
    model_count=read_json(out/'initial-control.json')['model_forwards']if (out/'initial-control.json').exists()else 0
    # If a forward started then failed, its exact count is UNKNOWN; no zero claim.
    feature_phase=next((r for r in records if r['name']=='features_control'),None)
    if feature_phase and feature_phase['status']!='PASS'and not (out/'initial-control.json').exists():model_count=None
    def summary(names,complete_file=None):
        rs=[r for r in records if r['name']in names]
        if any(r['status']=='FAILED_NO_RETRY'for r in rs):return 'FAILED'
        if len(rs)==len(names)and all(r['status']=='PASS'for r in rs):
            return 'SUCCEEDED'if complete_file is None or (out/complete_file).exists()else 'INCOMPLETE'
        return 'INCOMPLETE'if rs else 'NOT_RUN'
    phases=dict(input_reconstitution=summary(['inputs'],'input-reconstitution.json'),
        dependencies=summary(['wheels','venv','install','verify_install']),
        build=summary(['compile'],'build.json'),features_control=summary(['features_control'],'preparation-result.json'))
    if status=='FAILED'and all(v=='SUCCEEDED'for v in phases.values()):phases['features_control']='UNKNOWN'
    final=dict(schema='a20-preparation-final-status-v1',status=status,native_feature_count=len(native),
        official_feature_count=len(official),control_logits_count=int((out/'a20-initial-logits.f32le').exists()),
        model_forward_count=model_count,phases=phases,training_update_count=0,
        optimizer_constructions=0,backwards=0,updates=0,decoder_calls=0,training_authorized=False,release_sha256=release_sha,unpublished_incomplete_writes=rejected)
    if status=='FAILED':final['failure_code']='PREPARATION_FAILED_NO_RETRY'
    write_json(out/'final-status.json',final)
    write_json(out/'resources.json',dict(schema='a20-preparation-resources-v1',status=status,phases=records,
        guarded_wall_ms=guarded_metrics['wall_ms'],guarded_accounted_controller_and_reaped_child_cpu_ms=guarded_metrics['accounted_cpu_ms'],
        limits=dict(cpu_seconds=300,wall_seconds=600,address_space_per_process_bytes=AS_RSS,sampled_owned_child_group_RSS_bytes=AS_RSS,artifact_work_output_bytes=OUTPUT_CAP),
        monitor='reused reviewed owned-PID read_proc/cleanup/hard-limit functions; strict current-runner children visibility',
        cpu_scope='controller plus owned phases/reaped descendants through post-phase source recheck; evidence composition and packaging excluded',controller_RSS='not sampled into child-group RSS; hard2GiBAS remains installed',io_null='UNKNOWN',sampling_may_miss_transients=True),1024*1024)
    deps={}
    for key,name in [('wheel_inspection','wheel-inspection.json'),('wheel_progress','wheel-progress.json'),('installation','installed.json')]:
        p=ROOT/'work'/name;deps[key]=read_json(p)if p.exists()else dict(status='NOT_COMPLETED')
    write_json(out/'dependencies.json',deps,512*1024)
    write_json(out/'environment.json',dict(python_version=platform.python_version(),architecture=platform.machine(),libc=platform.libc_ver(),
        runner_image='ubuntu24/20260927.320.1',cpu_affinity_count=len(os.sched_getaffinity(0)),thread_variables=1,GPU=False,github=github,
        system_python_sha256=digest(Path(sys.executable).resolve())))
    write_json(out/'source-identities.json',dict(source_freeze_sha256=digest(ROOT/'SOURCE-FREEZE.json'),protocol_sha256=digest(ROOT/'PROTOCOL.json'),
        public_input_manifest_sha256=digest(ROOT/'metadata/PUBLIC-INPUT-MAP.json'),dependency_lock_sha256=exact_wheels.LOCK_SHA256,
        train32_sha256=digest(ROOT/'metadata/TRAIN32.json'),rights_provenance_sha256=digest(ROOT/'metadata/DERIVED-OUTPUT-RIGHTS-PROVENANCE.json'),release_sha256=release_sha))
    rights=read_json(ROOT/'metadata/DERIVED-OUTPUT-RIGHTS-PROVENANCE.json')
    write_json(out/'rights-and-provenance.json',dict(commercial_output_license='not-established',distribution_scope='bounded-public-research-preparation-only',
        new_license_grant=False,source_provenance_sha256=digest(ROOT/'metadata/DERIVED-OUTPUT-RIGHTS-PROVENANCE.json'),
        source_metadata=rights['source_metadata'],required_output_notices=rights['required_output_notices']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-release',type=Path);a=p.parse_args()
    if a.execute_release is None:raise SystemExit('PREPARATION_ONLY: no exact reviewed release')
    raise SystemExit(run(a.execute_release))
