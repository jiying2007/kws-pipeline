"""Bounded stock-voice source screening, adapted from the executed TTS36 runner.

No imports of inference dependencies occur unless --execute-reviewed-six is given.
The caller must use the locked setup and a process-group resource supervisor.
"""
import argparse, ast, hashlib, json, math, os, pathlib, random, signal, time, wave

ROOT=pathlib.Path(__file__).resolve().parent
VOICES=('Ryan','Aiden','Ono_Anna')
TEXTS=('你好小窝','小窝小窝')
STAGES={'entry','verify_setup','import_runtime_gate','validate_plan','verify_model_assets',
        'inspect_model_sidecars','prepare_runtime','import_numpy','import_soundfile',
        'import_scipy_signal','import_qwen_tts','load_model','verify_loaded_model','generate_six_cells'}
DIAGNOSTIC_MODULES={'generate_six.py','runtime_gate.py','setup_locked.py','wheel_identity.py','setup_diagnostics.py'}
def source_functions(path):
    return {n.name for n in ast.walk(ast.parse(path.read_bytes())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}|{'<module>','<lambda>','<listcomp>','<dictcomp>','<setcomp>','<genexpr>','<other>'}
RECIPE={'language':'Chinese','do_sample':True,'top_k':50,'top_p':1.0,
        'temperature':0.9,'repetition_penalty':1.05,'subtalker_dosample':True,
        'subtalker_top_k':50,'subtalker_top_p':1.0,'subtalker_temperature':0.9,
        'max_new_tokens':120,'non_streaming_mode':True}

def require(value, message):
    if not value: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def signal_statistics(samples):
    values=[float(x) for x in samples]
    finite=[x for x in values if math.isfinite(x)]
    return {'nonfinite_count':len(values)-len(finite),
            'peak':max(map(abs,finite)) if finite else None,
            'abs_ge_one_count':sum(abs(x)>=1 for x in finite),
            'saturation_count':sum(x>=1 or x < -1 for x in finite)}

def derivative_signal(samples, expected_frames):
    """Check resampling output before PCM16 conversion can conceal overshoot."""
    require(0<len(samples)<=192000 and len(samples)==expected_frames,'Derivative frame count mismatch')
    statistics=signal_statistics(samples)
    require(statistics['nonfinite_count']==0,'Nonfinite derivative waveform')
    peak=statistics['peak']
    return peak,(['normalized_peak_ge_one'] if peak>=1.0 else [])

def validate_plan(plan):
    rows=plan['generation_requests']
    require(len(rows)==6,'Exactly six requests required')
    expected=[(v,t,1337+2*i+j) for i,v in enumerate(VOICES) for j,t in enumerate(TEXTS)]
    require([(r['voice'],r['intended_text'],r['seed']) for r in rows]==expected,'Frozen six cells changed')
    require([r['source_id'] for r in rows]==[f'qwen6-{i:03d}' for i in range(1,7)],'Opaque IDs changed')
    require(all(r['attempts_max']==1 for r in rows),'One attempt per cell required')
    require(plan['recipe']==RECIPE,'Frozen recipe changed')
    require(plan['model_id']=='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice' and plan['model_revision']=='85e237c12c027371202489a0ec509ded67b5e4b5','Model pin changed')
    require(plan['heldout_voice']=='Sohee' and plan['heldout_generation_allowed'] is False,'Heldout changed')
    require(plan['cpu']=={'dtype':'float32','attention':'eager','threads':4,'interop_threads':1},'CPU recipe changed')
    return rows

def one_attempt_loop(rows, out, function):
    """A failed attempt remains consumed. Existing output directories are refused."""
    out.mkdir(exist_ok=False,parents=True)
    results=[]
    for row in rows:
        claim=out/(row['source_id']+'.started.json')
        with claim.open('x') as f:json.dump({'source_id':row['source_id'],'seed':row['seed'],'attempt':1},f)
        results.append(function(row))
    return results

def run(modeldir, out, stage=lambda value:None):
    stage('verify_setup')
    from setup_locked import verify_setup
    verify_setup(modeldir.resolve().parent.parent)
    stage('import_runtime_gate')
    from runtime_gate import prepare, local_loader_policy, inspect_model_sidecars
    stage('validate_plan')
    plan=json.loads((ROOT/'plan.json').read_text());rows=validate_plan(plan)
    lock=json.loads((ROOT/'model-lock.json').read_text())
    require(lock['model_id']==plan['model_id'] and lock['revision']==plan['model_revision'],'Asset lock mismatch')
    stage('verify_model_assets')
    modeldir=modeldir.resolve()
    require({str(p.relative_to(modeldir)) for p in modeldir.rglob('*') if p.is_file()}=={r['path'] for r in lock['files']},'Model file inventory mismatch')
    for item in lock['files']:
        p=modeldir/item['path'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==item['size_bytes'] and sha(p)==item['sha256'],'Model input identity mismatch')
    stage('inspect_model_sidecars');inspect_model_sidecars(modeldir)
    stage('prepare_runtime')
    torch,transformers,network_events=prepare()
    stage('import_numpy')
    import numpy as np
    stage('import_soundfile')
    import soundfile as sf
    stage('import_scipy_signal')
    from scipy.signal import resample_poly
    stage('import_qwen_tts')
    from qwen_tts import Qwen3TTSModel
    calls=[];start=time.monotonic();receipt={'schema':'qwen6-generation-receipt-v1','status':'started','plan_sha256':sha(ROOT/'plan.json'),'model_lock_sha256':sha(ROOT/'model-lock.json'),'generation_rows':[],'model_loads':calls}
    def save():
        out.mkdir(exist_ok=True,parents=True)
        temp=out/'receipt.tmp';temp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');os.replace(temp,out/'generation-receipt.json')
    require(not out.exists(),'Output exists; no resume or regeneration')
    def timeout(signum,frame):raise TimeoutError('Frozen per-clip time limit exceeded')
    signal.signal(signal.SIGALRM,timeout)
    try:
        stage('load_model')
        with local_loader_policy(modeldir,torch,transformers,calls):
            t=time.monotonic()
            model=Qwen3TTSModel.from_pretrained(str(modeldir),device_map='cpu',dtype=torch.float32,attn_implementation='eager',use_safetensors=True,weights_only=True,local_files_only=True,trust_remote_code=False)
            receipt['model_load_seconds']=time.monotonic()-t
            stage('verify_loaded_model')
            for owner in (model.model,model.model.speech_tokenizer.model):
                require({str(p.device) for p in owner.parameters()}=={'cpu'},'Non-CPU parameters')
                require({str(p.dtype) for p in owner.parameters()}=={'torch.float32'},'Wrong parameter dtype')
                attention=[]
                for name,module in owner.named_modules():
                    if all(hasattr(module,k) for k in ('q_proj','k_proj','v_proj')):
                        require(getattr(module.config,'_attn_implementation',None)=='eager','Wrong attention implementation')
                        attention.append(name)
                require(attention,'Empty attention inventory')
            original=model.model.talker.generate;termination={}
            def capture(*a,**kw):
                result=original(*a,**kw);seq=result.sequences;eos=int(model.model.config.talker_config.codec_eos_token_id)
                last=seq[:,-1].detach().cpu().tolist();iterations=len(result.hidden_states)
                termination.update(iterations=iterations,last_token_ids=last,eos=eos,ended_with_eos=all(x==eos for x in last),hit_token_cap_without_eos=iterations>=120 and not all(x==eos for x in last))
                return result
            model.model.talker.generate=capture
            def generate(row):
                random.seed(row['seed']);np.random.seed(row['seed']);torch.manual_seed(row['seed']);termination.clear();t=time.monotonic()
                record=dict(row,status='attempt_started');receipt['generation_rows'].append(record);save()
                signal.alarm(300)
                try:
                    with torch.inference_mode():wavs,rate=model.generate_custom_voice(text=row['intended_text'],speaker=row['voice'],**RECIPE)
                    require(len(wavs)==1 and rate==24000,'Output geometry changed')
                    arr=np.asarray(wavs[0]);require(arr.ndim==1 and 0<arr.size<=12*rate,'Invalid or oversized waveform')
                    record['source_signal_statistics']=signal_statistics(arr)
                    require(record['source_signal_statistics']['nonfinite_count']==0,'Nonfinite native waveform')
                    raw=out/(row['source_id']+'.raw-float.wav');sf.write(raw,arr,rate,subtype='FLOAT')
                    wav=out/(row['source_id']+'.wav');normalized=resample_poly(arr.astype(np.float64),2,3)
                    require(normalized.ndim==1,'Derivative must be mono')
                    record['normalized_signal_statistics']=signal_statistics(normalized)
                    normalized_peak,flags=derivative_signal(normalized,math.ceil(arr.size*2/3))
                    sf.write(wav,normalized,16000,subtype='PCM_16')
                    with wave.open(str(wav),'rb') as f:
                        require((f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,1,2),'Derivative geometry changed');pcm=f.readframes(f.getnframes())
                    if not np.any(arr):flags.append('silent')
                    if np.max(np.abs(arr))>=1:flags.append('source_peak_ge_one')
                    if not termination.get('ended_with_eos'):flags.append('generation_end_unverified')
                    if termination.get('hit_token_cap_without_eos'):flags.append('token_cap_without_eos')
                    record.update(status='generated_candidate',generation_seconds=time.monotonic()-t,duration_seconds=len(normalized)/16000,source_sample_rate_hz=rate,source_channels=1,source_frames=int(arr.size),source_float_dtype='little_endian_float32',source_float_values_sha256=hashlib.sha256(np.asarray(arr,dtype='<f4').tobytes()).hexdigest(),raw_audio_sha256=sha(raw),audio_sha256=sha(wav),pcm_sha256=hashlib.sha256(pcm).hexdigest(),termination=termination.copy(),signal_flags=flags,peak=float(np.max(np.abs(arr))),normalized_peak=normalized_peak,rms=float(np.sqrt(np.mean(arr.astype(np.float64)**2))))
                    save();return record
                except BaseException as error:
                    record.update(status='failed_no_retry',error_class=type(error).__name__,error_code='TTS_CELL_FAILED');save();raise
                finally:signal.alarm(0)
            stage('generate_six_cells');one_attempt_loop(rows,out,generate)
        receipt.update(status='six_candidates_generated',elapsed_seconds=time.monotonic()-start,python_network_attempts=network_events);save()
    except BaseException as error:
        receipt.update(status='failed_no_retry',error_class=type(error).__name__,error_code='TTS_STAGE_FAILED',elapsed_seconds=time.monotonic()-start);save();raise

def generation_entry(modeldir,out):
    """Persist the earliest failure boundary before any third-party imports."""
    from setup_diagnostics import safe_exception
    diagnostic={'schema':'qwen6-generation-stage-v1','stage':'entry','status':'started',
                'raw_text_disclosed':False}
    target=out.parent/'generation-stage.json'
    require(not target.exists(),'Generation already attempted; no retry')
    def save():
        temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(diagnostic,sort_keys=True,indent=2)+'\n');temporary.replace(target)
    def stage(value):
        require(value in STAGES,'Unknown diagnostic stage')
        diagnostic['stage']=value;save()
    save()
    try:
        run(modeldir,out,stage)
    except BaseException as error:
        frames=[];tb=error.__traceback__
        while tb is not None:
            path=pathlib.Path(tb.tb_frame.f_code.co_filename).resolve()
            if path.name in DIAGNOSTIC_MODULES and path==ROOT/path.name and len(frames)<16:
                function=tb.tb_frame.f_code.co_name
                frames.append({'module':path.name,'function':function if function in source_functions(path) else '<other>',
                               'line':tb.tb_lineno,'sha256':sha(path)})
            tb=tb.tb_next
        diagnostic.update(status='failed_no_retry',exception_type=safe_exception(error),
                          code='GENERATION_STAGE_FAILED',source_frames=frames);save();raise
    else:
        diagnostic.update(status='complete');save()

def public_generation_stage(value):
    """Reject unexpected fields/types instead of publishing arbitrary diagnostics."""
    from setup_diagnostics import EXCEPTIONS
    allowed={'schema','stage','status','raw_text_disclosed','exception_type','code','source_frames'}
    require(type(value) is dict and not set(value)-allowed,'Diagnostic shape')
    require(value.get('schema')=='qwen6-generation-stage-v1' and value.get('stage') in STAGES,'Diagnostic identity')
    require(value.get('status') in {'started','complete','failed_no_retry'} and value.get('raw_text_disclosed') is False,'Diagnostic status')
    if 'exception_type' in value:require(value['exception_type'] in EXCEPTIONS|{'OtherException'},'Diagnostic exception')
    if 'code' in value:require(value['code']=='GENERATION_STAGE_FAILED','Diagnostic code')
    frames=value.get('source_frames',[]);require(type(frames) is list and len(frames)<=16,'Diagnostic frame bound')
    for frame in frames:
        require(type(frame) is dict and set(frame)=={'module','function','line','sha256'},'Diagnostic frame shape')
        require(frame['module'] in DIAGNOSTIC_MODULES and type(frame['line']) is int and 0<frame['line']<100000,'Diagnostic source location')
        require(type(frame['sha256']) is str and len(frame['sha256'])==64 and all(c in '0123456789abcdef' for c in frame['sha256']),'Diagnostic source identity')
        path=ROOT/frame['module']
        require(frame['sha256']==sha(path) and frame['function'] in source_functions(path),'Diagnostic source binding')
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--execute-reviewed-six',action='store_true');a=p.parse_args()
    require(a.execute_reviewed_six,'Explicit reviewed execution flag required')
    require(os.environ.get('ORT_DISABLE_TELEMETRY')=='1','Set ORT telemetry opt-out before imports')
    generation_entry(a.model_dir,a.out)
