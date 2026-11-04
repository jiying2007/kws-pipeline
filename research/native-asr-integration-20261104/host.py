"""Concrete retained-boundary three-model prepare/freeze/infer orchestration.

No workflow is enabled. Missing exact generated-input/source/review locks blocks
before Docker. All native payload remains runner-private; public export is one
fixed aggregate object only. Parent review and persistence decision are pending.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from balanced_order import validate_ordinals
CAMPAIGN='native-asr64-20261104-v1'
FAMILIES=('qwen06','sensevoice','whisper')
SECONDS=4800

def require(ok,why):
    if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,allow_nan=False);stream.write('\n')

def verify_lock(root,lock):
    require(lock.get('campaign')==CAMPAIGN,'campaign source freeze')
    require(type(lock.get('input_lock_sha256')) is str and
            re.fullmatch('[0-9a-f]{64}',lock['input_lock_sha256']) is not None,
            'actual 64 generated-input lock is not frozen yet')
    for relative,digest in lock['files'].items():
        path=Path(relative)
        require(not path.is_absolute() and '..' not in path.parts,'relative frozen member')
        source=root/path
        require(source.is_file() and not source.is_symlink() and sha(source)==digest,'source freeze mismatch')
    for target,source in lock['stage'].items():
        require(source in lock['files'] and not Path(target).is_absolute() and '..' not in Path(target).parts,
                'closed stage source mapping')
    require({'research/native-asr-integration-20261104/sense-review-template.json',
             'research/native-asr-integration-20261104/qwen-source-members.json',
             'research/sense-native-capture-20261104/worker.py',
             'research/whisper-native-capture-20261104/worker.py'} <= set(lock['stage']),
            'complete reviewed native prepare dependencies required before setup')
    require(sha(root/lock['input_lock_path'])==lock['input_lock_sha256'],'exact generated input lock')

def stage(root,work,lock):
    code=work/'code';code.mkdir()
    for target,source in lock['stage'].items():
        dest=code/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/source,dest)
    return code

def validate_input(root,input_lock):
    import wave
    require(input_lock['schema']=='exact-generated-asr64-input-v1' and
            len(input_lock['clips'])==64,'actual complete 64 input records')
    validate_ordinals([row.get('producer_ordinal') for row in input_lock['clips']])
    seen=set();pcm_seen=set()
    for i,row in enumerate(input_lock['clips'],1):
        require(set(row)=={'audio_id','path','wav_sha256','pcm_sha256','frames','source_id','request_id','producer_ordinal'}
                and row['audio_id']=='clip-%06d'%i,'exact audio-only input identity fields')
        rel=Path(row['path']);require(not rel.is_absolute() and '..' not in rel.parts,'relative input source')
        p=root/rel
        require(p.is_file() and not p.is_symlink() and p.stat().st_nlink==1 and p.stat().st_size<=384044
                and sha(p)==row['wav_sha256'],'exact derived WAV bytes')
        with wave.open(str(p)) as wav:
            require((wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),wav.getcomptype())==(16000,1,2,'NONE')
                    and wav.getnframes()==row['frames'] and 0<row['frames']<=192000,'whole PCM geometry')
            pcm=hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest()
        require(pcm==row['pcm_sha256'] and pcm not in pcm_seen and row['request_id'] not in seen,
                'unique measured PCM/request identities')
        pcm_seen.add(pcm);seen.add(row['request_id'])

def batch_inputs(root,work,input_lock):
    result=[]
    for batch in range(4):
        directory=work/'inputs'/('batch-%d'%batch);(directory/'audio').mkdir(parents=True)
        # Membership is precommitted by producer ordinal. Only order inside
        # that fixed batch follows the retained content-hash binding contract.
        selected=input_lock['clips'][16*batch:16*(batch+1)]
        rows=[dict(row,global_audio_id=row['audio_id'],audio_id='clip-%06d'%index)
              for index,row in enumerate(sorted(selected,key=lambda r:r['wav_sha256']),1)]
        clips=[];identities=[]
        for row in rows:
            name=row['wav_sha256']+'.wav'
            shutil.copyfile(root/row['path'],directory/'audio'/name)
            clips.append(dict(audio_id=row['audio_id'],audio_path='audio/'+name,wav_sha256=row['wav_sha256']))
            identities.append({k:row[k] for k in ('audio_id','source_id','request_id','wav_sha256')})
        save(directory/'job.json',dict(schema='blind-source-screen32-job-v1',clips=clips))
        result.append((directory,dict(schema='blind-asr-source-request-map-v1',clips=identities),rows))
    return result

def worker_command(family,phase,head,policy_sha=None,batch=0,job_sha=None):
    common=['--phase',phase,'--runtime','/runtime','--executed-head',head]
    if family=='qwen06':
        script='/code/research/native-asr-integration-20261104/qwen_worker.py'
        args=common+['--code','/code','--input','/input','--output','/output',
            '--source-members','/code/research/native-asr-integration-20261104/qwen-source-members.json']
    elif family=='sensevoice':
        script='/code/research/sense-native-capture-20261104/worker.py'
        return ['/runtime/venv/bin/python','-I','-B',script,'--phase',phase]
    else:
        script='/code/research/whisper-native-capture-20261104/worker.py'
        args=common+['--retained-repo','/code','--output','/output/whisper-'+phase,
            '--sink-source','/code/research/qwen-native-capture-20261104/native_sink.py',
            '--consensus-tools','/code/tools']
        if phase=='infer':
            require(type(job_sha) is str and re.fullmatch('[0-9a-f]{64}',job_sha) is not None,'frozen Whisper job bytes')
            args+=['--input','/input','--job-sha256',job_sha]
    if phase=='infer':
        args+=['--policy','/code/runtime-policy/recognizer-policy.json','--policy-sha256',policy_sha,
               '--identity-map','/code/runtime-policy/identity-map-%d.json'%batch]
    return ['/runtime/venv/bin/python','-I','-B',script]+args

def launch(boundary,code,runtime,output,image,name,kind,command,deadline,inputs):
    require(deadline-time.monotonic()>=120,'launch reserve before capacity')
    boundary.host_capacity()
    require(deadline-time.monotonic()>=120,'launch reserve after capacity')
    output.mkdir(parents=True)
    args=boundary.container_args(name,image,code,runtime,output,'asr',kind,inputs)
    args[args.index(image)+1:]=command
    require(deadline-time.monotonic()>=120,'launch reserve after argv')
    diagnostic=output.parent/('container-'+kind+'.json')
    if diagnostic.exists():diagnostic.unlink()  # prior launch already retained its own exclusive copy
    try:
        return boundary.supervise_container(args,name,output,kind,
            min(deadline-60,time.monotonic()+(1200 if kind=='setup' else 1500)))
    finally:
        diagnostic=output.parent/('container-'+kind+'.json')
        if diagnostic.is_file():
            with (output/'container-private.json').open('xb') as stream:stream.write(diagnostic.read_bytes())

def freeze_policy(preparations,destination):
    from speech_asr_consensus import validate_recognizer_policy
    from asr_native_contracts import canonical_sha256
    identities=[json.loads(path.read_bytes()) for path in preparations]
    policy=dict(schema_version=2,evidence_class='speech-three-asr-recognizer-policy-v2',
        normalization='fixed-punctuation-removal-only-v1',frozen_before_recognition=True,
        target_conditioning_allowed=False,recognizers=identities)
    validate_recognizer_policy(policy)
    save(destination,policy)
    return policy,canonical_sha256(policy)

def collect(family,output,rows,policy):
    """Private strict-byte validation. No transcript enters the public result."""
    from asr_native_contracts import canonical_sha256,validate_native_capture
    identities={r['model_family']:r for r in policy['recognizers']}
    identity=identities[{'qwen06':'qwen3-asr','sensevoice':'sensevoice','whisper':'whisper'}[family]]
    root=output/({'qwen06':'qwen06/strict-native','sensevoice':'sensevoice/strict-native',
                  'whisper':'whisper-infer/capture'}[family])
    result=[]
    for row in rows:
        d=root/row['audio_id'];completed=d/'completed.json';started=d/'started.json';raw=d/'raw-output.json'
        legacy_claim=output/family/(row['audio_id']+'.started.json')
        state='failed_no_retry' if d.exists() or legacy_claim.exists() else 'not_run'
        if completed.is_file():
            try:
                receipt=json.loads(completed.read_bytes());text=json.loads(raw.read_bytes());start=json.loads(started.read_bytes())
                require(receipt['schema']=='retained-asr-completed-v1' and receipt['status']=='success'
                        and receipt['decoder_completed'] is True and receipt['started_sha256']==sha(started)
                        and receipt['raw_output_sha256']==sha(raw) and start['schema']=='retained-asr-started-v1',
                        'strict original completion hashes')
                for key,expected in dict(source_id=row['source_id'],request_id=row['request_id'],file_sha256=row['wav_sha256'],
                    pcm_sha256=row['pcm_sha256'],audio_frames=row['frames'],
                    recognizer_policy_sha256=canonical_sha256(policy),
                    recognizer_identity_sha256=canonical_sha256(identity),recognizer_id=identity['recognizer_id'],
                    native_decode_config_sha256=identity['native_decode_config_sha256'],
                    model_lock_sha256=identity['native_decode_config']['asset_lock_sha256'],
                    source_lock_sha256=identity['native_decode_config']['source_lock_sha256']).items():
                    require(receipt[key]==expected and start[key]==expected,'strict original identity binding')
                validate_native_capture(identity['native_decode_config'],receipt['native_capture'])
                require(receipt['processed_start_frame']==0 and receipt['processed_end_frame']==row['frames']
                        and receipt['native_capture']['input']['audio_frames']==row['frames'], 'whole input frame identity')
                require(text['schema']=='retained-asr-raw-transcript-v1' and text['request_id']==row['request_id']
                        and text['output_scope']=='complete-decoder-transcript' and text['truncated'] is False
                        and type(text['actual_text']) is str and text['actual_text'],
                        'complete original raw text binding')
                state='success'
            except (ValueError,KeyError,TypeError,OSError):
                state='failed_no_retry'
        result.append(dict(audio_id=row.get('global_audio_id',row['audio_id']),status=state))
    return result

def export_aggregate(work,states,completed_batches,status):
    require(status in ('complete_screen_only','stopped_no_retry'),'fixed aggregate status')
    counts={f:{s:sum(r['status']==s for r in states[f]) for s in ('success','failed_no_retry','not_run')} for f in FAMILIES}
    public=work/'aggregate-only';public.mkdir()
    save(public/'status.json',dict(schema='native-asr64-aggregate-only-v1',campaign=CAMPAIGN,
        status=status,planned_audio=64,planned_asr_slots=192,families=counts,completed_batches=completed_batches,
        training_admitted=False,human_gold=False,retry=False,evidence_persistence='runner-only-unresolved'))

def run(env):
    require(False,'Full ASR launch remains blocked pending reviewed private evidence persistence and trust roots')
    event=json.loads(Path(env['GITHUB_EVENT_PATH']).read_bytes())
    require(env['GITHUB_REPOSITORY']=='jiying2007/kws-pipeline' and env['GITHUB_REF']=='refs/heads/research/'+CAMPAIGN
            and env['GITHUB_EVENT_NAME']=='push' and env['GITHUB_RUN_NUMBER']=='1' and env['GITHUB_RUN_ATTEMPT']=='1'
            and event.get('created') is True and event.get('deleted') is False and event.get('after')==env['GITHUB_SHA']
            and event['repository']['private'] is False
            and event['repository']['full_name']=='jiying2007/kws-pipeline'
            and re.fullmatch('[0-9a-f]{40}',env['GITHUB_SHA']) is not None
            and re.fullmatch('[1-9][0-9]{0,19}',env['GITHUB_RUN_ID']) is not None,
            'first exact public source branch only')
    lock=json.loads((HERE/'source-lock.json').read_bytes());verify_lock(ROOT,lock)
    inputs=json.loads((ROOT/lock['input_lock_path']).read_bytes());validate_input(ROOT,inputs)
    # The complete capture/dependency source list is frozen before external work.
    work=ROOT/('native-private-'+CAMPAIGN);work.mkdir(exist_ok=False)
    deadline=time.monotonic()+SECONDS
    states={f:[dict(audio_id=r['audio_id'],status='not_run') for r in inputs['clips']] for f in FAMILIES}
    status='stopped_no_retry';completed=0
    try:
        require(shutil.disk_usage(work).free>=25*1024**3,'disk reserve')
        code=stage(ROOT,work,lock)
        template=json.loads((code/'research/native-asr-integration-20261104/sense-review-template.json').read_bytes())
        require(set(template)=={'schema','dependencies','sources'} and
                template['schema']=='reviewed-sense-runtime-capture-template-v1', 'reviewed Sense dependency template')
        save(code/'runtime-policy/sense-review.json',dict(schema='reviewed-sense-runtime-capture-v1',
            executed_head_sha=env['GITHUB_SHA'],dependencies=template['dependencies'],sources=template['sources']))
        sys.path[:0]=[str(code/'tools'),str(ROOT/'research/tts-pilot-screening-v1')]
        import boundary
        jobs=batch_inputs(ROOT,work,inputs)
        image_lock=json.loads((code/'research/source-screen32-v1/container-lock.json').read_bytes())
        boundary.host_capacity();boundary.pull_image(image_lock,work/'image-private.json');image=image_lock['image']
        runtimes={k:work/('runtime-'+k) for k in ('primary','whisper')}
        for lane,runtime in runtimes.items():
            runtime.mkdir()
            script='source-screen32-whisper-v1/setup_runtime.py' if lane=='whisper' else 'source-screen32-v1/setup_adapter.py'
            command=['python','-I','-B','/code/research/'+script]
            if lane=='primary':command+=['--profile','asr','--stage','setup','--runtime','/runtime']
            launch(boundary,code,runtime,work/'setup'/lane,image,CAMPAIGN+'-setup-'+lane,
                   'setup',command,deadline,code)
        preparation_paths=[]
        for family in FAMILIES:
            output=work/'prepare'/family
            launch(boundary,code,runtimes['whisper' if family=='whisper' else 'primary'],output,image,
                CAMPAIGN+'-prepare-'+family,'sensevoice' if family=='whisper' else family,
                worker_command(family,'prepare',env['GITHUB_SHA']),deadline,jobs[0][0])
            preparation_paths.append(output/({'qwen06':'qwen06/prepared/identity.json',
                'sensevoice':'sensevoice/prepare/identity.json','whisper':'whisper-prepare/identity.json'}[family]))
        policy,policy_sha=freeze_policy(preparation_paths,code/'runtime-policy/recognizer-policy.json')
        for batch,(directory,mapping,local_rows) in enumerate(jobs):save(code/'runtime-policy'/('identity-map-%d.json'%batch),mapping)
        for batch,(directory,mapping,local_rows) in enumerate(jobs):
            # Sense's fixed entry reads this exact audio-only map. Replace only
            # between containers; no input labels are ever staged with it.
            target=code/'runtime-policy/identity-map.json'
            temporary=code/'runtime-policy/identity-map.next.json'
            save(temporary,mapping);temporary.replace(target)
            for family in FAMILIES:
                output=work/'inference'/('batch-%d'%batch)/family
                try:
                    launch(boundary,code,runtimes['whisper' if family=='whisper' else 'primary'],output,image,
                        CAMPAIGN+'-batch-%d-'%batch+family,'sensevoice' if family=='whisper' else family,
                        worker_command(family,'infer',env['GITHUB_SHA'],policy_sha,batch,sha(directory/'job.json')),deadline,directory)
                finally:
                    rows=collect(family,output,local_rows,policy)
                    indexed={row['audio_id']:row for row in rows}
                    for global_row in states[family]:
                        if global_row['audio_id'] in indexed:global_row.update(indexed[global_row['audio_id']])
                require(all(r['status']=='success' for r in rows),'incomplete batch stops later work')
            completed+=1
        status='complete_screen_only'
    finally:
        save(work/'terminal-private.json',states)
        export_aggregate(work,states,completed,status)

if __name__=='__main__':run(os.environ)
