"""Generated6 x two-model supervised source screen; execution requires explicit CLI GO.

The file is prepared for later reviewed execution. Importing performs no work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'core'))
from asr6_contract import (EXPERIMENT,BATCH_IDS,canonical,digest,bind_job,validate_job,validate_input_freeze,MAX_ARCHIVE_BYTES)
from release_gate import verify_release
from supervision import supervise,tree_bytes,GiB
MODELS=('sensevoice','qwen06')
THREADS={'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','BLIS_NUM_THREADS':'1'}
OFFLINE={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','ORT_DISABLE_TELEMETRY':'1','CUDA_VISIBLE_DEVICES':''}

def save(path,value):
    data=canonical(value)
    if len(data)>8*1024**2: raise ValueError('Output file cap')
    with Path(path).open('xb') as f: f.write(data);f.flush();os.fsync(f.fileno())

def unpack_inputs(runtime,incoming,archive_sha256,input_freeze_sha256):
    incoming=Path(incoming)
    if incoming.is_symlink() or not incoming.is_dir() or {p.name for p in incoming.iterdir()}!={'blind-inputs.zip','blind-input-freeze.json'}:
        raise ValueError('Only the two blind producer files are accepted')
    if any(p.is_symlink() or not p.is_file() for p in incoming.iterdir()):raise ValueError('Producer file aliases')
    archive=incoming/'blind-inputs.zip';freeze_raw=(incoming/'blind-input-freeze.json').read_bytes()
    if not 0<archive.stat().st_size<=MAX_ARCHIVE_BYTES or digest(archive.read_bytes())!=archive_sha256:raise ValueError('Producer archive identity/limit')
    freeze=validate_input_freeze(freeze_raw,input_freeze_sha256);expected={r['path']:r for r in freeze['files']}
    target=runtime/'inputs';target.mkdir()
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)!=7 or set(z.namelist())!=set(expected) or z.comment:raise ValueError('Blind archive membership/comments')
        for info in infos:
            row=expected[info.filename]
            if info.is_dir() or info.comment or info.extra or info.file_size!=row['bytes'] or (info.external_attr>>16)&0o170000 not in (0,0o100000):raise ValueError('Blind archive member type/size')
            raw=z.read(info)
            if len(raw)!=row['bytes'] or digest(raw)!=row['sha256']:raise ValueError('Blind archive member hash')
            path=target/info.filename;path.parent.mkdir(exist_ok=True);path.write_bytes(raw)
    job=validate_job((target/'job.json').read_bytes(),freeze['job_sha256'])
    if {'job.json'}|{r['audio_path'] for r in job['clips']}!=set(expected):raise ValueError('Job/archive association')
    (runtime/'blind-input-freeze.json').write_bytes(freeze_raw)
    return bind_job(target,freeze['job_sha256'],input_freeze_sha256)


def make_plans(runtime,decoder):
    prepared=runtime/'prepared';prepared.mkdir();encoded=canonical(decoder);sha=digest(encoded)
    (prepared/'decoder-inputs.json').write_bytes(encoded)
    models=json.loads((ROOT/'model-locks.json').read_bytes());plans={}
    for name in MODELS:
        row=models[name]
        contract={'schema':'asr6-model-execution-v1','armed':True,'model_id':row['model_id'],'run_id':EXPERIMENT+'-'+name,
            'asset_lock_sha256':row['asset_lock_sha256'],'source_lock_sha256':row['source_lock_sha256'],
            'expected_schema_sha256':row['expected_schema_sha256'],'decoder_manifest_sha256':sha,'model_dtype':'float32','batch_size':1,
            'network':'offline_configured_not_kernel_isolated','scope':'single_decode_each_generated6_audio_no_plan_context',
            'experiment_id':EXPERIMENT,'candidate_count':6,'batch':'all6','source_manifest_sha256':decoder['source_manifest_sha256'],'input_freeze_sha256':decoder['input_freeze_sha256']}
        plan={k:row[k] for k in ('model_id','asset_lock','asset_lock_sha256','source_lock','source_lock_sha256')}
        plan.update(schema='asr6-run-plan-v1',model_root=str(runtime/'models'/name),pcm_root=str(runtime/'inputs/audio'),
            contract=contract,contract_sha256=digest(canonical(contract)),decoder_manifest_sha256=sha,clip_seconds=120)
        save(prepared/(name+'.plan.json'),plan);plans[name]=plan
    return plans

def recover_rows(directory,job,termination):
    """Keep all6 despite interrupted writes; retain every original raw byte."""
    directory=Path(directory);snapshots=sorted(directory.glob('outcomes.[0-9]*.json'),reverse=True)
    candidates=[directory/'outcomes.json',*snapshots,directory/'outcomes.initial.json']
    saved=[];warnings=[];selected_source=None
    for source in candidates:
        if not source.exists(): continue
        try:
            candidate=json.loads(source.read_bytes())
            if type(candidate) is not list or len(candidate)!=6: raise ValueError('denominator')
            if [r['opaque_id'] for r in candidate]!=BATCH_IDS['all6']: raise ValueError('IDs')
            for row,expected in zip(candidate,job['clips']):
                if row['wav_sha256']!=expected['wav_sha256'] or row['status'] not in ('not_run','input_error','success','error','timeout'):
                    raise ValueError('row identity/status')
                if row['raw_text'] is not None and type(row['raw_text']) is not str: raise ValueError('raw text')
            saved=candidate;selected_source=source.name;break
        except (OSError,ValueError,KeyError,TypeError):
            warnings.append({'file':source.name,'reason':'unreadable_or_incomplete_snapshot_preserved'})
    by={r['opaque_id']:r for r in saved};result=[]
    for r in job['clips']:
        oid=r['audio_id'];row=dict(by.get(oid,{'opaque_id':oid,'wav_sha256':r['wav_sha256'],'status':'not_run','raw_text':None,'completeness':'unknown','quality_flags':[],'execution_receipt_sha256':None}))
        recovery={'derived':True,'selected_snapshot':selected_source,'source_kind':'aggregate_snapshot' if oid in by else 'no_completed_result',
                  'source_file':selected_source if oid in by else None,'started_without_result_synthesized':False}
        # A fully retained receipt can outlive a torn aggregate snapshot.
        receipt_path=directory/(oid+'.receipt.json')
        if row['status']=='not_run' and receipt_path.exists():
            try:
                raw=receipt_path.read_bytes();receipt=json.loads(raw)
                evidence=directory/(oid+'.decoder.json')
                if receipt['opaque_id']!=oid or receipt['wav_sha256']!=r['wav_sha256'] or digest(evidence.read_bytes())!=receipt['decoder_evidence_sha256']:
                    raise ValueError('receipt/evidence binding')
                row={k:receipt[k] for k in ('opaque_id','wav_sha256','status','raw_text','completeness','quality_flags')}
                row['execution_receipt_sha256']=digest(raw)
                recovery.update(source_kind='per_clip_receipt',source_file=receipt_path.name)
            except (OSError,ValueError,KeyError,TypeError):
                warnings.append({'file':receipt_path.name,'reason':'unreadable_or_unbound_receipt_preserved'})
        if row['status']=='not_run' and (directory/(oid+'.started.json')).exists():
            row.update(status='terminated_during_attempt',termination=termination)
            recovery.update(source_kind='started_without_completed_result',source_file=oid+'.started.json',started_without_result_synthesized=True)
        row['recovery_provenance']=recovery
        result.append(row)
    if warnings:
        for row in result: row['recovery_warnings']=warnings
    return result


def adapt(rows,model,job_sha256):
    """Schema adapter only; preserve original status/raw output and never relabel."""
    out=[]
    for row in rows:
        complete=row['status']=='success' and row['completeness']=='complete'
        status='complete' if complete else ('error' if row['status'] in ('error','timeout','input_error','terminated_during_attempt') else 'incomplete')
        out.append({'audio_id':row['opaque_id'],'wav_sha256':row['wav_sha256'],'raw_text':row['raw_text'] if row['raw_text'] is not None else '',
            'status':status,'quality_flags':row['quality_flags'],'raw_output':row})
    return {'schema':'saved-asr-transcripts-v1','blind_job_sha256':job_sha256,'model':model,'clips':out}

def finalize(runtime,measurements):
    artifact=runtime/'artifact';artifact.mkdir(exist_ok=False)
    raw=runtime/'raw'
    for name in MODELS:
        directory=raw/name
        if not directory.exists():
            directory.mkdir()
            save(directory/'not-run.json',{'schema':'asr6-supervisor-not-run-v1','status':'not_run','model_decode_attempts':0,'record_origin':'supervisor_no_model_load_receipt','requested_ids':BATCH_IDS['all6']})
        freeze=directory/'model-raw-freeze.json'
        if not freeze.exists():save(freeze,{'files':{p.name:digest(p.read_bytes()) for p in directory.iterdir() if p.is_file()},'labels_joined':False})
    if raw.exists(): shutil.copytree(raw,artifact/'raw')
    # Raw completion/snapshot bytes are frozen before adaptation or any private join.
    files={p.relative_to(artifact).as_posix():digest(p.read_bytes()) for p in artifact.rglob('*') if p.is_file()}
    save(artifact/'raw-freeze.json',{'schema':'asr6-raw-freeze-v1','files':files,'labels_joined':False})
    job=validate_job((runtime/'inputs/job.json').read_bytes(),measurements['blind_job_sha256']);statuses={}
    for name in MODELS:
        rows=recover_rows(raw/name,job,measurements.get(name,{}))
        statuses[name]=[{k:row[k] for k in ('opaque_id','wav_sha256','status','completeness','quality_flags','execution_receipt_sha256')}
                        | {'raw_text_present':row['raw_text'] is not None,'raw_text_characters':len(row['raw_text']) if row['raw_text'] is not None else None,
                           'recovery_warnings':row.get('recovery_warnings',[]),'recovery_provenance':row['recovery_provenance']} for row in rows]
    save(artifact/'final-status.json',{'schema':'asr6-final-status-v1','models':statuses,'schema_adaptation':'private_after_raw_hash_verification'})
    shutil.copyfile(runtime/'inputs/job.json',artifact/'blind-job.json')
    shutil.copyfile(runtime/'blind-input-freeze.json',artifact/'blind-input-freeze.json')
    save(artifact/'resources.json',measurements)
    # Setup provenance only; no wheel bodies, weights, tensors or logs in artifact.
    if (runtime/'setup-receipt.json').exists():
        setup=json.loads((runtime/'setup-receipt.json').read_bytes())
        save(artifact/'setup-summary.json',public_setup_summary(setup))
    validate_artifact(artifact)
    save(artifact/'artifact-freeze.json',{'schema':'asr6-artifact-freeze-v1','files':{p.relative_to(artifact).as_posix():digest(p.read_bytes()) for p in artifact.rglob('*') if p.is_file()},'private_labels_joined':False})
    validate_artifact(artifact)
    return artifact

def public_setup_summary(setup):
    # A strict public projection; no command argv, absolute paths, host or environment dump.
    result={k:setup[k] for k in ('schema','python','locks','initial_free_bytes','network_configuration',
        'actual_download_bytes','wall_seconds','disk_checkpoints','bootstrap_pip') if k in setup}
    from setup_diagnostics import public_commands,EXCEPTIONS
    packages={r['name']:r for r in json.loads((ROOT/'runtime-lock.json').read_bytes())['files']}
    result['command_results']=public_commands(setup.get('commands',[]),packages)
    phase=setup.get('phase')
    if type(phase) is dict and set(phase)=={'stage','package'}:
        allowed_stages=set(__import__('setup_diagnostics').STAGES)|{'download_package','download_model_asset','extract_source'}
        if phase['stage'] in allowed_stages and phase['package'] in set(packages)|{'sensevoice','qwen06','pip',None}:result['active_phase']=phase
    result['status']='error' if 'exception' in setup else ('verified' if 'verification' in setup else 'incomplete')
    if 'exception' in setup: result['exception_type']=setup['exception']['type'] if setup['exception']['type'] in EXCEPTIONS else 'OtherException'
    result['wheels']={name:{k:r[k] for k in ('bytes','sha256','name','version','metadata_sha256','metadata_bytes') if k in r}
                      for name,r in setup.get('wheels',{}).items()}
    verification=setup.get('verification',setup.get('package_verification',{}))
    result['source_members']=verification.get('source_members',[])
    result['model_assets']=verification.get('model_assets',{})
    if 'import_preflight' in setup:
        from import_preflight import public_receipt
        try:result['import_preflight']=public_receipt(setup['import_preflight'])
        except (ValueError,TypeError,KeyError):result['import_preflight']={'status':'diagnostic_rejected','code':'unsafe_diagnostic_payload'}
    return result


def validate_artifact(root):
    if not Path(root).is_dir(): raise ValueError('No result artifact was produced')
    for p in Path(root).rglob('*'):
        if p.is_symlink(): raise ValueError('Artifact symlink')
        if p.is_file() and (p.suffix!='.json' or p.stat().st_size>8*1024**2): raise ValueError('Artifact file scope/cap')
    if tree_bytes(root)>20*1024**2: raise ValueError('Artifact aggregate cap')

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--incoming',type=Path,required=True)
    p.add_argument('--archive-sha256',required=True);p.add_argument('--input-freeze-sha256',required=True)
    p.add_argument('--execute-reviewed-asr6',action='store_true',required=True)
    a=p.parse_args();release=verify_release(ROOT,'asr',a.release)
    if platform.python_version()!='3.12.14' or platform.machine()!='x86_64' or sys.platform!='linux':raise ValueError('Exact runtime host required')
    if os.environ.get('GITHUB_RUN_ATTEMPT','1')!='1':raise ValueError('No workflow reruns/retries')
    runtime=ROOT/'runtime'
    if a.incoming.resolve()!=runtime/'incoming':raise ValueError('Blind download must stay in runtime/incoming')
    if set(p.name for p in runtime.iterdir())!={'incoming'}:raise ValueError('Fresh ASR runtime required')
    (runtime/'raw').mkdir()
    began=time.monotonic();provision_elapsed=max(0,time.time()-float(os.environ.get('QWEN6_JOB_STARTED_UNIX_SECONDS',time.time())))
    deadline=began+max(0,2700-provision_elapsed);free=shutil.disk_usage(ROOT).free
    with (runtime/'run.claim').open('x') as f:f.write(release['asr_candidate_sha256']+'\n')
    decoder=unpack_inputs(runtime,a.incoming,a.archive_sha256,a.input_freeze_sha256);plans=make_plans(runtime,decoder)
    measurements={'candidate_sha256':release['asr_candidate_sha256'],'tts_candidate_sha256':release['tts_candidate_sha256'],
        'preregistered_plan_sha256':release['plan_sha256'],'blind_job_sha256':decoder['source_manifest_sha256'],
        'blind_input_freeze_sha256':a.input_freeze_sha256,'blind_archive_sha256':a.archive_sha256,
        'python':platform.python_version(),'platform':platform.platform(),'interpreter_provision_elapsed_seconds':provision_elapsed,
        'interpreter_expanded_bytes':tree_bytes(Path(sys.base_prefix)),
        'runner_image':{k:os.environ.get(k) for k in ('ImageOS','ImageVersion','RUNNER_ARCH','RUNNER_OS')},
        'initial_free_disk_bytes':free,'inference_thread_environment':THREADS,'retries':0,
        'purpose':'generated_source_screen_weak_machine_evidence_only_no_human_gold','requested_clips':6,
        'recognizers':2,'max_asr_calls':12,'raw_frozen_before_plan_join':True}
    failure=None;env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PIP_NO_CACHE_DIR='1')
    try:
        if free<12*GiB: raise RuntimeError('Initial actual free disk below12GiB')
        setup=supervise([sys.executable,str(ROOT/'setup_locked.py'),'--root',str(runtime)],cwd=ROOT,env=env,
            wall_seconds=min(1200,max(0,1500-provision_elapsed)),cpu_seconds=2400,rss_bytes=2*GiB,cores=2,workspace=ROOT,installed=runtime/'venv',buildtmp=runtime/'buildtmp',
            log_path=runtime/'setup.log',job_deadline=deadline,setup=True)
        measurements['setup']=setup
        if (runtime/'setup-receipt.json').exists():
            receipt=json.loads((runtime/'setup-receipt.json').read_bytes())
            if 'actual_download_bytes' in receipt:
                actual=receipt['actual_download_bytes'];measurements['controlled_download_bytes']=actual
                if actual>4294967296:raise RuntimeError('ASR controlled download cap exceeded')
        if setup['returncode']!=0 or setup['termination_reason']!='normal_exit': raise RuntimeError('Setup failed or stopped')
        for name in MODELS:
            child_env=dict(env,**THREADS,**OFFLINE,ASR6_SUPERVISED='1')
            result=supervise([str(runtime/'venv/bin/python'),str(ROOT/'run_model.py'),'--name',name,'--plan-sha256',digest(canonical(plans[name])),
                '--supervisor-pid',str(os.getpid())],cwd=ROOT,env=child_env,wall_seconds=600,cpu_seconds=600,rss_bytes=10*GiB,cores=1,
                workspace=ROOT,installed=runtime/'venv',buildtmp=runtime/'buildtmp',log_path=runtime/(name+'.log'),job_deadline=deadline)
            measurements[name]=result
            model_raw=runtime/'raw'/name
            if model_raw.exists():
                save(model_raw/'model-raw-freeze.json',{'files':{p.name:digest(p.read_bytes()) for p in model_raw.iterdir() if p.is_file()},'labels_joined':False})
            if result['returncode']!=0 or result['termination_reason']!='normal_exit': raise RuntimeError(name+' failed; remaining model not_run')
    except Exception as exc:
        failure={'exception_type':type(exc).__name__,'stage':'supervised_setup_or_model','message':'Stage failed; detailed exception text is not public'};measurements['failure']=failure
    finally:
        measurements['runner_wall_seconds']=time.monotonic()-began;measurements['final_free_disk_bytes']=shutil.disk_usage(ROOT).free
        finalize(runtime,measurements)
    return 1 if failure else 0

def safe_cli(entry=main):
    try:
        return entry()
    except Exception as exc:
        print(json.dumps({'status':'blocked','exception_type':type(exc).__name__,
            'stage':'outer_entry_or_finalization','message':'Stage failed; detailed exception text is not public'}),file=sys.stderr)
        return 2

def artifact_check_cli():
    return safe_cli(lambda: validate_artifact(ROOT/'runtime/artifact') or 0)

if __name__=='__main__': raise SystemExit(safe_cli())
