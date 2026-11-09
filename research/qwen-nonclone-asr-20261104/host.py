"""Single new four-by-three blind screen. No TTS, retries, training or gold labels."""
from pathlib import Path
import hashlib,importlib.util,json,os,re,shutil,sys,time,unicodedata
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAMPAIGN='qwen-nonclone-asr-20261104-v1'
FAMILIES=('qwen06','sensevoice','whisper')
SECONDS=4800

def require(ok, why):
    if not ok: raise ValueError(why)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def admission(env,event):
    wanted={'GITHUB_REPOSITORY':'jiying2007/kws-pipeline','GITHUB_REF':'refs/heads/research/'+CAMPAIGN,'GITHUB_EVENT_NAME':'push','GITHUB_RUN_NUMBER':'1','GITHUB_RUN_ATTEMPT':'1'}
    require(all(env.get(k)==v for k,v in wanted.items()),'first campaign run only')
    require(re.fullmatch('[0-9a-f]{40}',env.get('GITHUB_SHA','')) is not None and re.fullmatch('[1-9][0-9]{0,19}',env.get('GITHUB_RUN_ID','')) is not None,'source/run identity')
    require(event.get('created') is True and event.get('deleted') is False and event.get('after')==env['GITHUB_SHA'] and event.get('repository',{}).get('full_name')==wanted['GITHUB_REPOSITORY'] and event['repository'].get('private') is False,'first public branch creation')

def verify_sources(root,lock):
    require(lock['campaign']==CAMPAIGN,'campaign source lock')
    for name,expected in lock['files'].items():
        rel=Path(name);p=root/rel
        require(not rel.is_absolute() and '..' not in rel.parts and p.is_file() and not p.is_symlink() and sha(p)==expected,'source/input identity')
    for target,source in lock['stage'].items():
        require(source in lock['files'] and not Path(target).is_absolute() and '..' not in Path(target).parts,'closed stage mapping')

def stage(root,dest,lock):
    dest.mkdir()
    for target,source in lock['stage'].items():
        p=dest/target;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/source,p)

def validate_input(path):
    import wave
    job=json.loads((path/'job.json').read_text())
    require(set(job)=={'schema','clips'} and job['schema']=='blind-source-screen32-job-v1' and len(job['clips'])==4,'exact four blind inputs')
    require([r['audio_id'] for r in job['clips']]==['clip-%06d'%i for i in range(1,5)],'opaque sequence')
    require(len({r['wav_sha256'] for r in job['clips']})==4,'four unique WAVs')
    for row in job['clips']:
        require(set(row)=={'audio_id','audio_path','wav_sha256'} and re.fullmatch('[0-9a-f]{64}',row['wav_sha256']) is not None and row['audio_path']=='audio/'+row['wav_sha256']+'.wav','no label/prompt fields')
        p=path/row['audio_path'];require(p.is_file() and not p.is_symlink() and p.stat().st_size<=384044 and sha(p)==row['wav_sha256'],'input bytes')
        with wave.open(str(p)) as w:require((w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())==(16000,1,2,'NONE') and 0<w.getnframes()<=192000,'complete PCM geometry')
    require({p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}=={'job.json'}|{r['audio_path'] for r in job['clips']},'blind input members')
    return job

def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text) if not c.isspace() and not unicodedata.category(c).startswith('P'))

def score(mapping,results):
    require(len(mapping['rows'])==4 and set(results)==set(FAMILIES),'fixed denominator')
    rows=[]
    for sample in mapping['rows']:
        obs={}
        for family in FAMILIES:
            matched=[r for r in results[family] if r['opaque_id']==sample['opaque_id']]
            require(len(matched)==1 and matched[0]['wav_sha256']==sample['derived_sha256'],'terminal input binding')
            obs[family]=matched[0]
        valid=all(r['status']=='success' and r['completeness']=='complete' and not r.get('quality_flags') and type(r.get('raw_text')) is str for r in obs.values())
        consensus=valid and len({normalize(r['raw_text']) for r in obs.values()})==1
        match=consensus and all(normalize(r['raw_text'])==normalize(sample['requested_text']) for r in obs.values())
        rows.append(dict(sample_id=sample['sample_id'],requested_text=sample['requested_text'],observations=obs,three_model_consensus=bool(consensus),requested_text_match=bool(match),screening_tier='machine_three_asr_candidate' if match else 'review_or_unresolved',human_gold=False,training_admitted=False))
    return dict(campaign=CAMPAIGN,planned_audio=4,planned_asr_slots=12,rows=rows,new_tts_calls=0,new_training_runs=0,human_gold=False,training_admitted=False,confidence_note='Machine screening only; no human truth, acoustic completeness, independent speakers or held-out evaluation claim.')

def launch(b,code,runtime,output,image,family,deadline,input_dir=None):
    require(time.monotonic()+120<deadline,'absolute deadline and launch cleanup reserve')
    setup=family in ('setup-primary','setup-whisper');kind='setup' if setup else ('sensevoice' if family=='whisper' else family)
    output.mkdir();b.host_capacity()
    args=b.container_args(CAMPAIGN+'-'+family,image,code,runtime,output,'asr',kind,code if setup else input_dir)
    pos=args.index(image)+1
    if family=='setup-whisper':args[pos:]=['python','-I','-B','/code/research/source-screen32-whisper-v1/setup_runtime.py']
    if family=='whisper':args[pos:]=['/runtime/venv/bin/python','-I','-B','/code/research/tts-pilot-screening-v1/whisper_worker.py']
    require(time.monotonic()+120<deadline,'remaining launch cleanup reserve')
    return b.supervise_container(args,args[2],output,kind,min(deadline-60,time.monotonic()+(1200 if setup else 1500)))

def export_public(work):
    """Publish a deliberately tiny aggregate-only projection.

    Raw recognizer receipts, target text, per-clip IDs, transcripts, native
    captures, diagnostics and detailed host reports remain runner-local.
    The public artifact is rebuilt from an explicit scalar schema; it never
    recursively copies worker output or an allowlist of worker files.
    """
    work=Path(work)
    receipt=json.loads((work/'host-receipt.json').read_text())
    stages=receipt.get('stages',{})
    require(type(stages) is dict,'bounded aggregate stage map')
    aggregate={
        'schema':'kws-asr-public-aggregate-v1',
        'campaign':CAMPAIGN,
        'status':receipt.get('status','stopped_no_retry'),
        'stage_count':len(stages),
        'completed_stage_count':sum(
            1 for value in stages.values()
            if type(value) is dict and value.get('status') in ('success','complete','completed')
        ),
        'new_tts_calls':0,
        'new_training_runs':0,
        'human_gold':False,
        'training_admitted':False,
        'raw_transcripts_published':False,
        'target_text_published':False,
        'native_evidence_published':False
    }
    dest=work/'artifact'
    dest.mkdir()
    summary=dest/'aggregate-status.json'
    save(summary,aggregate)
    save(dest/'artifact-index.json',{
        'schema':'kws-asr-public-artifact-index-v1',
        'campaign':CAMPAIGN,
        'files':{'aggregate-status.json':sha(summary)}
    })
    names={p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
    require(names=={'aggregate-status.json','artifact-index.json'},
            'public artifact must contain only aggregate status and its index')
    for p in dest.rglob('*'):
        if p.is_file():
            require(not p.is_symlink() and p.stat().st_nlink==1 and p.stat().st_size<=65536,
                    'bounded unaliased aggregate artifact')
    return aggregate

def run(env):
    eventpath=Path(env['GITHUB_EVENT_PATH']);require(eventpath.stat().st_size<=2*1024**2,'event budget');admission(env,json.loads(eventpath.read_text()))
    lock=json.loads((HERE/'source-lock.json').read_text());verify_sources(ROOT,lock)
    inputs=HERE/'inputs'/'blind';job=validate_input(inputs)
    work=ROOT/('asr-'+CAMPAIGN);work.mkdir(exist_ok=False)
    started=time.monotonic();deadline=started+SECONDS
    receipt=dict(campaign=CAMPAIGN,head_sha=env['GITHUB_SHA'],run_id=env['GITHUB_RUN_ID'],input_job_sha256=sha(inputs/'job.json'),source_lock_sha256=sha(HERE/'source-lock.json'),max_asr_calls=12,new_tts_calls=0,new_training_runs=0,retry=False,training_admitted=False,human_gold=False,stages={})
    results={f:[dict(opaque_id=r['audio_id'],wav_sha256=r['wav_sha256'],status='not_run',raw_text=None,completeness='unknown',quality_flags=[]) for r in job['clips']] for f in FAMILIES}
    sys.path.insert(0,str(ROOT/'research/tts-pilot-screening-v1'))
    import boundary as b
    from terminal import whisper_rows
    try:
        require(shutil.disk_usage(work).free>=25*1024**3,'25 GiB disk reserve');receipt['host_capacity']=b.host_capacity()
        code=work/'code';stage(ROOT,code,lock)
        spec=importlib.util.spec_from_file_location('fresh_primary_terminal',code/'research/source-screen32-v1/asr_worker.py');terminal=importlib.util.module_from_spec(spec);spec.loader.exec_module(terminal)
        image_lock=json.loads((ROOT/'research/source-screen32-v1/container-lock.json').read_text());b.pull_image(image_lock,work/'image.json');image=image_lock['image']
        for lane,families in [('primary',FAMILIES[:2]),('whisper',FAMILIES[2:])]:
            lane_root=work/lane;lane_root.mkdir();runtime=work/('runtime-'+lane);runtime.mkdir()
            try: receipt['stages']['setup-'+lane]=launch(b,code,runtime,lane_root/'setup',image,'setup-'+lane,deadline)
            except Exception as e:
                receipt['stages']['setup-'+lane]=dict(status='failed_no_retry',error_type=type(e).__name__);raise
            for family in families:
                output=lane_root/family
                try:receipt['stages'][family]=launch(b,code,runtime,output,image,family,deadline,inputs)
                except Exception as e:
                    receipt['stages'][family]=dict(status='failed_no_retry',error_type=type(e).__name__);raise
                finally:
                    if family=='whisper':results[family]=whisper_rows(inputs,output)
                    else:
                        terminal.terminal_outcomes(inputs,output,family);results[family]=json.loads((output/'terminal-outcomes.json').read_text())
        receipt['status']='screened_pending_review'
    except BaseException as e:
        receipt.update(status='stopped_no_retry',error_type=type(e).__name__);raise
    finally:
        receipt['wall_seconds']=time.monotonic()-started;save(work/'host-receipt.json',receipt)
        try:save(work/'report.json',score(json.loads((HERE/'inputs/mapping.json').read_text()),results))
        finally:export_public(work)

if __name__=='__main__':
    try:run(os.environ)
    except BaseException as e:
        print(json.dumps(dict(campaign=CAMPAIGN,status='stopped_no_retry',error_type=type(e).__name__)));raise SystemExit(1)
