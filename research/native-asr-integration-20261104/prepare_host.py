"""One-shot, no-audio real native policy preparation. Never calls recognition."""
from pathlib import Path
import json
import os
import re
import shutil
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import host
ROOT=host.ROOT
CAMPAIGN='native-prepare-20261104-v1'
SECONDS=1800

def admission(env):
    event=json.loads(Path(env['GITHUB_EVENT_PATH']).read_bytes())
    host.require(env['GITHUB_REPOSITORY']=='jiying2007/kws-pipeline' and
        env['GITHUB_REF']=='refs/heads/research/'+CAMPAIGN and env['GITHUB_EVENT_NAME']=='push' and
        env['GITHUB_RUN_NUMBER']=='1' and env['GITHUB_RUN_ATTEMPT']=='1' and
        event.get('created') is True and event.get('deleted') is False and
        event.get('after')==env['GITHUB_SHA'] and event['repository']['private'] is False and
        event['repository']['full_name']=='jiying2007/kws-pipeline' and
        re.fullmatch('[0-9a-f]{40}',env['GITHUB_SHA']) is not None and
        re.fullmatch('[1-9][0-9]{0,19}',env['GITHUB_RUN_ID']) is not None,'exact first new public prepare branch')

def verify(root,lock):
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
    if diagnostic.exists():diagnostic.unlink()  # prior launch already retained its own exclusive copy
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
        row=dict(family=family,status=('failed_no_retry' if (work/'prepare'/family).exists() else 'not_run')
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
    lock=json.loads((HERE/'prepare-source-lock.json').read_bytes());verify(ROOT,lock)
    work=ROOT/('native-private-'+CAMPAIGN);work.mkdir(exist_ok=False)
    deadline=time.monotonic()+SECONDS;proofs={};status='stopped_no_retry';policy_sha=None
    try:
        host.require(shutil.disk_usage(work).free>=25*1024**3,'prepare disk reserve')
        code=host.stage(ROOT,work,lock)
        template=json.loads((code/'research/native-asr-integration-20261104/sense-review-template.json').read_bytes())
        host.require(set(template)=={'schema','dependencies','sources'} and
                     template['schema']=='reviewed-sense-runtime-capture-template-v1','reviewed Sense template')
        host.save(code/'runtime-policy/sense-review.json',dict(schema='reviewed-sense-runtime-capture-v1',
            executed_head_sha=env['GITHUB_SHA'],dependencies=template['dependencies'],sources=template['sources']))
        sys.path[:0]=[str(code/'tools'),str(ROOT/'research/tts-pilot-screening-v1')]
        import boundary
        image_lock=json.loads((code/'research/source-screen32-v1/container-lock.json').read_bytes())
        host.require(deadline-time.monotonic()>=120,'prepare image reserve')
        boundary.host_capacity();boundary.pull_image(image_lock,work/'image-private.json');image=image_lock['image']
        runtimes={k:work/('runtime-'+k) for k in ('primary','whisper')}
        for lane,runtime in runtimes.items():
            runtime.mkdir()
            script='source-screen32-whisper-v1/setup_runtime.py' if lane=='whisper' else 'source-screen32-v1/setup_adapter.py'
            command=['python','-I','-B','/code/research/'+script]
            if lane=='primary':command+=['--profile','asr','--stage','setup','--runtime','/runtime']
            launch(boundary,code,runtime,work/'setup'/lane,image,CAMPAIGN+'-setup-'+lane,command,deadline,True)
        identities=[]
        for family in host.FAMILIES:
            output=work/'prepare'/family
            command=['/runtime/venv/bin/python','-I','-B',
                '/code/research/native-asr-integration-20261104/prepare_worker.py','--family',family,
                '--executed-head',env['GITHUB_SHA']]
            launch(boundary,code,runtimes['whisper' if family=='whisper' else 'primary'],output,image,
                   CAMPAIGN+'-'+family,command,deadline)
            proofs[family]=json.loads((output/'zero-forward-proof.json').read_bytes())
            identities.append(output/'identity.json')
        host.require(deadline-time.monotonic()>=60,'policy freeze within absolute budget')
        _,policy_sha=host.freeze_policy(identities,work/'policy-private.json')
        status='prepared_zero_forward'
    finally:export(work,status,proofs,policy_sha)

if __name__=='__main__':run(os.environ)
