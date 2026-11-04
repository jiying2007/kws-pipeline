"""One-shot, no-audio real native policy preparation. Never calls recognition."""
from pathlib import Path
import json
import os
import re
import shutil
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
import hashlib
import types
from contextlib import contextmanager
SOURCE_PINS={'research/native-asr-integration-20261104/host.py': 'e29808c4e16aed6eb31dc16513a22dd9c3dc8948a41e456eb5d527eb3bfbda57', 'research/native-asr-integration-20261104/balanced_order.py': '9ac839b3ae6dd75b38e9e015e0a8366cccce966fbf80941550750630a9a3fbad', 'research/native-prepare-capacity-20261104/capacity.py': '1df234418ca584f807119d92b438199e9aef2ce043cc1a665650d2952a6282b6', 'research/native-prepare-fence-v2-20261104/safe_diagnostic.py': 'dd8d95f3aabaa82d4f5bf58ab97e8b67f2b603be8f0a15746db4d20280a64255', 'research/tts-pilot-screening-v1/screen.py': '982582c0e4c7f957276520977ebfa3bad0cbb5753e16d50bbadef0835682523e', 'research/native-prepare-scan-v3-20261104/output_size.py': 'ffa46b23945322467b65b80f4d7913fd70bbd661709d7f404073bcc51cf265da', 'research/native-prepare-scan-v3-20261104/boundary.py': 'b9e61a1e3593038860415cd3141e62b2c18bd6dbb937a5c6e48a7717b8c96848'}

@contextmanager
def aliases(mapping):
    absent=object();old={key:sys.modules.get(key,absent) for key in mapping}
    sys.modules.update(mapping)
    try:yield
    finally:
        for key,value in old.items():
            if value is absent:sys.modules.pop(key,None)
            else:sys.modules[key]=value

def exact_module(relative,name):
    path=ROOT/relative
    if relative not in SOURCE_PINS or path.is_symlink() or path.resolve()!=path.absolute():
        raise ValueError('exact unaliased host module required')
    if not path.is_file() or path.stat().st_size>1024**2:
        raise ValueError('bounded host source file required')
    with path.open('rb') as stream:raw=stream.read(1024**2+1)
    if len(raw)>1024**2 or hashlib.sha256(raw).hexdigest()!=SOURCE_PINS[relative]:
        raise ValueError('host source pin mismatch')
    # Compile the authenticated bytes directly; never trust a cached module/pyc.
    module=types.ModuleType(name);module.__file__=str(path);module.__package__=''
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module

balanced=exact_module('research/native-asr-integration-20261104/balanced_order.py','_scan_v3_balanced')
with aliases({'balanced_order':balanced}):
    host=exact_module('research/native-asr-integration-20261104/host.py','_scan_v3_host_primitives')
if host.ROOT!=ROOT:raise ValueError('retained host source root differs from current workspace')
capacity_module=exact_module('research/native-prepare-capacity-20261104/capacity.py','_scan_v3_capacity')
Capacity,observed=capacity_module.Capacity,capacity_module.observed
diagnostic_module=exact_module('research/native-prepare-fence-v2-20261104/safe_diagnostic.py','_scan_v3_diagnostics')
read_summary,FILENAME=diagnostic_module.read_summary,diagnostic_module.FILENAME

def load_boundary():
    screen=exact_module('research/tts-pilot-screening-v1/screen.py','_scan_v3_screen')
    scanner=exact_module('research/native-prepare-scan-v3-20261104/output_size.py','_scan_v3_output_size')
    with aliases({'screen':screen,'output_size':scanner}):
        boundary=exact_module('research/native-prepare-scan-v3-20261104/boundary.py','_scan_v3_boundary')
    if boundary.observed_output_size is not scanner.observed_output_size or boundary.require is not screen.require:
        raise ValueError('exact boundary dependency identities required')
    return boundary

CAMPAIGN='native-prepare-scan-v3-20261104-v1'
SECONDS=1800
RETAINED_LOCK_SHA256='25e4c3e3409253b2ad41baa41c9e8c1d8694415456d66facfd7a3646bff3cf21'


OVERLAY='research/native-prepare-fence-v2-20261104/'
REPLACEMENTS={
 'research/native-asr-integration-20261104/prepare_worker.py':OVERLAY+'prepare_worker.py',
 'research/native-asr-integration-20261104/zero_forward.py':OVERLAY+'zero_forward.py',
 'research/sense-native-capture-20261104/worker.py':OVERLAY+'sense/worker.py',
 'research/sense-native-capture-20261104/sense_capture.py':OVERLAY+'sense/sense_capture.py',
 'research/whisper-native-capture-20261104/whisper_capture.py':OVERLAY+'whisper_capture.py',
 'research/native-asr-integration-20261104/qwen-source-members.json':OVERLAY+'qwen-source-members.json',
 'research/native-asr-integration-20261104/sense-review-template.json':OVERLAY+'sense-review-template.json'}
ADDITIONS={'research/native-asr-integration-20261104/safe_diagnostic.py':OVERLAY+'safe_diagnostic.py'}

def export_diagnostics(work):
    summaries=[read_summary(work/'prepare'/family/'output'/FILENAME,family) for family in host.FAMILIES]
    host.save(work/'aggregate-only/safe-summary.json',dict(schema='native-prepare-safe-summary-set-v1',
        campaign=CAMPAIGN,models=summaries,diagnostics_do_not_authorize_success=True))

def admission(env):
    event_path=Path(env['GITHUB_EVENT_PATH'])
    host.require(event_path.is_file() and not event_path.is_symlink() and event_path.stat().st_size<=1024**2,'bounded event file')
    event=json.loads(event_path.read_bytes())
    host.require(env['GITHUB_REPOSITORY']=='jiying2007/kws-pipeline' and
        env['GITHUB_REF']=='refs/heads/research/'+CAMPAIGN and env['GITHUB_EVENT_NAME']=='push' and
        env['GITHUB_RUN_NUMBER']=='1' and env['GITHUB_RUN_ATTEMPT']=='1' and
        event.get('created') is True and event.get('deleted') is False and
        event.get('after')==env['GITHUB_SHA'] and event.get('before')=='0'*40 and
        event.get('ref')==env['GITHUB_REF'] and event['repository']['private'] is False and
        env.get('GITHUB_ACTIONS')=='true' and env.get('RUNNER_ENVIRONMENT')=='github-hosted' and
        env.get('RUNNER_OS')=='Linux' and env.get('RUNNER_ARCH')=='X64' and
        env.get('GITHUB_WORKFLOW_SHA')==env['GITHUB_SHA'] and
        env.get('GITHUB_WORKFLOW_REF')=='jiying2007/kws-pipeline/.github/workflows/'+CAMPAIGN+'.yml@'+env['GITHUB_REF'] and
        event['repository']['full_name']=='jiying2007/kws-pipeline' and
        re.fullmatch('[0-9a-f]{40}',env['GITHUB_SHA']) is not None and
        re.fullmatch('[1-9][0-9]{0,19}',env['GITHUB_RUN_ID']) is not None,'exact first new public prepare branch')
    host.require(env.get('GITHUB_WORKSPACE') and Path(env['GITHUB_WORKSPACE']).resolve()==ROOT.resolve(),
                 'exact checked out workspace')

def verify(root,lock):
    fence_path=root/'research/native-prepare-fence-v2-20261104/source-lock.json'
    host.require(host.sha(fence_path)=='1ee10ac2827b76c2b48b49e11eafdcedb8fd0bf816c311f091cdd64d9b2475b7' and
                 lock['retained_fence_lock_sha256']=='1ee10ac2827b76c2b48b49e11eafdcedb8fd0bf816c311f091cdd64d9b2475b7',
                 'exact consumed fence-v2 closure')
    fence=json.loads(fence_path.read_bytes())
    host.require(lock['stage']==fence['stage'] and all(lock['files'].get(p)==h for p,h in fence['files'].items()),
                 'unchanged fence-v2 scientific stage and all original sources')
    host.require(all(lock['files'].get(p)==h for p,h in SOURCE_PINS.items()),'all exact host modules are source-frozen')
    retained_path=root/'research/native-asr-integration-20261104/prepare-source-lock.json'
    host.require(host.sha(retained_path)==RETAINED_LOCK_SHA256 and
                 lock['retained_prepare_lock_sha256']==RETAINED_LOCK_SHA256,'exact reviewed retained source closure')
    retained=json.loads(retained_path.read_bytes())
    host.require(all(lock['files'].get(p)==h for p,h in retained['files'].items()),
                 'all original 88 source identities remain unchanged')
    expected=dict(retained['stage']);expected.update(REPLACEMENTS);expected.update(ADDITIONS)
    host.require(lock['stage']==expected and lock['replacements']==REPLACEMENTS and lock['additions']==ADDITIONS,
                 'exact prospective replacement allowlist only')
    host.require(lock['campaign']==CAMPAIGN and lock['audio_inputs']==0 and lock['inference_calls']==0,
                 'exact no-input preparation contract')
    for relative,digest in lock['files'].items():
        p=Path(relative)
        host.require(not p.is_absolute() and '..' not in p.parts and not (root/p).is_symlink()
                     and host.sha(root/p)==digest,'prepare source freeze')
    for target,source in lock['stage'].items():
        p=Path(target)
        host.require(source in lock['files'] and not p.is_absolute() and '..' not in p.parts
                     and p.suffix not in ('.wav','.flac','.mp3') and 'runtime-policy' not in p.parts,
                     'closed source-only staging')

def launch(boundary,code,runtime,output,image,name,command,deadline,setup=False):
    host.require(deadline-time.monotonic()>=120,'prepare launch reserve')
    boundary.host_capacity()
    host.require(deadline-time.monotonic()>=120,'prepare reserve after capacity')
    output.mkdir(parents=True)
    # Retained tts geometry denotes no-input, network-none runtime-RO worker;
    # no TTS loader or generation command is executed by this new contract.
    kind='setup' if setup else 'tts'
    args=boundary.container_args(name,image,code,runtime,output,'asr' if setup else 'tts',kind,
                                 code if setup else None)
    args[args.index(image)+1:]=command
    host.require(not any('dst=/input' in str(a) or 'target=/input' in str(a) for a in args),'no audio mount')
    host.require(deadline-time.monotonic()>=120,'prepare reserve after argv')
    diagnostic=output.parent/('container-'+kind+'.json')
    host.require(not diagnostic.exists(),'fresh unique diagnostic parent; never delete prior evidence')
    try:
        return boundary.supervise_container(args,name,output,kind,min(deadline-60,time.monotonic()+1200))
    finally:
        diagnostic=output.parent/('container-'+kind+'.json')
        if diagnostic.is_file():
            with (output/'container-private.json').open('xb') as stream:stream.write(diagnostic.read_bytes())

def export(work,status,proofs,policy_sha=None):
    host.require(status in ('prepared_zero_forward','stopped_no_retry'),'fixed prepare status')
    rows=[]
    for family in host.FAMILIES:
        proof=proofs.get(family)
        row=dict(family=family,status=('failed_no_retry' if (work/'prepare'/family/'output').exists() else 'not_run')
                 if proof is None else 'prepared_zero_forward')
        if proof is not None:
            host.require(proof['family']==family and proof['status']=='prepared_zero_forward' and
                         proof['candidate_count']==0 and proof['audio_inputs']==0 and
                         proof['observed_guarded_entrypoint_attempts']==0,'actual guarded no-input proof')
            for key in ('native_config_sha256','source_lock_sha256','asset_lock_sha256'):
                host.require(re.fullmatch('[0-9a-f]{64}',proof[key]) is not None,'bounded public digest')
                row[key]=proof[key]
        rows.append(row)
    host.save(work/'aggregate-only/status.json',dict(schema='native-no-audio-preparation-public-v1',
        campaign=CAMPAIGN,status=status,models=rows,policy_sha256=policy_sha,audio_inputs=0,
        inference_calls=0,training_admitted=False,retry=False,evidence_persistence='runner-only'))

def run(env):
    admission(env)
    lock=json.loads((HERE/'source-lock.json').read_bytes());verify(ROOT,lock)
    work=ROOT/('native-private-'+CAMPAIGN);work.mkdir(exist_ok=False)
    deadline=time.monotonic()+SECONDS;proofs={};status='stopped_no_retry';policy_sha=None
    capacity=Capacity(work,ROOT,env,time.monotonic())
    try:
        initial=capacity.record('initial')
        host.require(initial['work_workspace_same_device'] is True and initial['workspace'].get('free_bytes',-1)>=25*1024**3,'prepare disk reserve')
        code=host.stage(ROOT,work,lock)
        template=json.loads((code/'research/native-asr-integration-20261104/sense-review-template.json').read_bytes())
        host.require(set(template)=={'schema','dependencies','sources'} and
                     template['schema']=='reviewed-sense-runtime-capture-template-v1','reviewed Sense template')
        host.save(code/'runtime-policy/sense-review.json',dict(schema='reviewed-sense-runtime-capture-v1',
            executed_head_sha=env['GITHUB_SHA'],dependencies=template['dependencies'],sources=template['sources']))
        sys.path.insert(0,str(code/'tools'))
        boundary=load_boundary()
        image_lock=json.loads((code/'research/source-screen32-v1/container-lock.json').read_bytes())
        host.require(deadline-time.monotonic()>=120,'prepare image reserve')
        with observed(capacity,'before_image','after_image'):
            boundary.host_capacity();boundary.pull_image(image_lock,work/'image-private.json')
        image=image_lock['image']
        runtimes={k:work/('runtime-'+k) for k in ('primary','whisper')}
        for lane,runtime in runtimes.items():
            runtime.mkdir()
            script='source-screen32-whisper-v1/setup_runtime.py' if lane=='whisper' else 'source-screen32-v1/setup_adapter.py'
            command=['python','-I','-B','/code/research/'+script]
            if lane=='primary':command+=['--profile','asr','--stage','setup','--runtime','/runtime']
            with observed(capacity,'before_setup_'+lane,'after_setup_'+lane):
                launch(boundary,code,runtime,work/'setup'/lane/'output',image,CAMPAIGN+'-setup-'+lane,command,deadline,True)
        identities=[]
        for family in host.FAMILIES:
            output=work/'prepare'/family/'output'
            command=['/runtime/venv/bin/python','-I','-B',
                '/code/research/native-asr-integration-20261104/prepare_worker.py','--family',family,
                '--executed-head',env['GITHUB_SHA']]
            with observed(capacity,'before_prepare_'+family,'after_prepare_'+family):
                launch(boundary,code,runtimes['whisper' if family=='whisper' else 'primary'],output,image,
                       CAMPAIGN+'-'+family,command,deadline)
            summary=read_summary(output/FILENAME,family)
            host.require(summary.get('status')=='complete','verified terminal safe diagnostic required')
            proofs[family]=json.loads((output/'zero-forward-proof.json').read_bytes())
            identities.append(output/'identity.json')
        host.require(deadline-time.monotonic()>=60,'policy freeze within absolute budget')
        _,policy_sha=host.freeze_policy(identities,work/'policy-private.json')
        status='prepared_zero_forward'
    finally:
        primary_failure=sys.exc_info()[0] is not None
        try:
            export(work,status,proofs,policy_sha)
            export_diagnostics(work)
        except Exception:
            if not primary_failure:raise

if __name__=='__main__':run(os.environ)
