"""One isolated 12-clip process; new captures, no historical-result conversion."""
from pathlib import Path
import importlib.util
import os
import re
import signal
import sys
import time
import types
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from campaign_contract import BOUNDARY, canonical, digest, identity, read_json, require, validate_blind, write_once, atomic
from cpu_budget import install_inference_guard
from worker_receipts import bind_predecode,emit_start,emit_completion


# Diagnostic text comes only from fixed public sources or fixed labels. Register
# actual code objects before use; a forged co_filename/function name is not trust.
_DIAGNOSTIC_MODULES={
    __name__:'research/qwen12-native-admission-v1/native_worker.py',
    **{name:'research/qwen12-native-admission-v1/'+name+'.py' for name in
       ('campaign_contract','cpu_budget','worker_receipts','native_locks','whisper_binding','optional_tools')},
    'qwen12_kernel_scope':'research/qwen12-native-admission-v1/runtime_scope.py',
    'qwen12_setup_binding':'research/qwen12-native-admission-v1/setup_adapter.py',
    'qwen12_retained_worker':'research/source-screen32-v1/asr_worker.py',
    'qwen12_native_adapter':'research/source-screen32-v1/native_v2_adapter.py',
    'qwen12_whisper_adapter':'research/source-screen32-whisper-v1/adapter.py',
    **{'asr_stage.'+name:'research/qwen6_asr/core/asr_stage/'+name+'.py' for name in
       ('adapters','architecture','assets','decoding','execution','runtime_support')},
    'pcm.pcm_binding':'research/qwen6_asr/core/pcm/pcm_binding.py',
}
_DIAGNOSTIC_STAGES=frozenset(('startup','pre_model_import','scope','output','guard','provenance','inputs',
    'initial_receipts','import_worker','verify_worker','prepare_caches','verify_sources',
    'import_helpers','bind_inputs','verify_runtime','verify_extensions','read_locks',
    'contract','load_model','load_receipt','import_adapter','frontend','native_locks',
    'construct_adapter','prepare_receipt','compare_frozen','bind_predecode','decoder',
    'bind_audio','start_attempt','infer','complete_attempt','worker_complete'))
_DIAGNOSTIC_ERRORS=(BaseException,Exception,PermissionError,FileNotFoundError,OSError,
    ValueError,TypeError,RuntimeError,TimeoutError,ImportError,ModuleNotFoundError,
    MemoryError,KeyError,IndexError,AttributeError,AssertionError,RecursionError,
    NotImplementedError,KeyboardInterrupt,SystemExit)
_DIAGNOSTIC_CODES={}
_DIAGNOSTIC_REGISTERED={}
_DIAGNOSTIC_PROBE_CODE=None
_FAILURE_CONTEXT=('startup',0)
_FAILURE_FAMILY='unknown'
_FAILURE_MODE='unknown'
MAX_DIAGNOSTIC_FRAMES=16
MAX_DIAGNOSTIC_SCAN=64
MAX_DIAGNOSTIC_BYTES=8192


def _register_subprocess_codes():
    # Snapshot actual objects from the interpreter's already-loaded stdlib, not
    # a filename/name fallback. No import, environment/source read, or process.
    loaded=sys.modules.get('subprocess')
    if type(loaded) is not types.ModuleType or _DIAGNOSTIC_REGISTERED.get('subprocess') is loaded:return
    expected=str(Path(sys.base_prefix)/'lib'/('python'+str(sys.version_info.major)+'.'+
        str(sys.version_info.minor))/'subprocess.py')
    if vars(loaded).get('__file__')!=expected:return
    _DIAGNOSTIC_REGISTERED['subprocess']=loaded
    candidates=[(name,vars(loaded).get(name)) for name in ('run','call','check_call','check_output')]
    popen=vars(loaded).get('Popen')
    if type(popen) is type and popen.__module__=='subprocess':
        candidates.extend((name,vars(popen).get(name)) for name in ('__init__','_execute_child','_posix_spawn'))
    for name,value in candidates:
        if len(_DIAGNOSTIC_CODES)>=2048:break
        if type(value) is not types.FunctionType or value.__module__!='subprocess':continue
        code=value.__code__
        if code.co_filename==expected and code.co_name==name:
            _DIAGNOSTIC_CODES[id(code)]=(code,'stdlib/subprocess.py',name)


def _register_diagnostic_codes():
    # No source, locals, arbitrary module discovery, or filesystem reads. The
    # registry is bounded and retains references so object IDs cannot be reused.
    try:
        _register_subprocess_codes()
        for name,relative in _DIAGNOSTIC_MODULES.items():
            loaded=sys.modules.get(name)
            if type(loaded) is not types.ModuleType or _DIAGNOSTIC_REGISTERED.get(name) is loaded:continue
            expected=str(ROOT/relative)
            if vars(loaded).get('__file__')!=expected:continue
            _DIAGNOSTIC_REGISTERED[name]=loaded
            pending=[]
            for value in vars(loaded).values():
                for _ in range(8):
                    if type(value) is not types.FunctionType:break
                    pending.append(value.__code__);value=vars(value).get('__wrapped__')
            for _ in range(2048):
                if not pending or len(_DIAGNOSTIC_CODES)>=2048:break
                code=pending.pop()
                if code.co_filename!=expected or id(code) in _DIAGNOSTIC_CODES:continue
                if len(code.co_name)>80 or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*|<(?:lambda|listcomp|dictcomp|setcomp|genexpr)>',code.co_name):continue
                _DIAGNOSTIC_CODES[id(code)]=(code,relative,code.co_name)
                pending.extend(value for value in code.co_consts if type(value) is types.CodeType)
    except BaseException:pass


def register_optional_probe_code(expected_code):
    """Callback for optional_tools AFTER its exact source hash verification.

    Compiled locked code survives removal of a failed import from sys.modules.
    The caller supplies authority; arbitrary filename/name matches do not.
    """
    global _DIAGNOSTIC_PROBE_CODE
    try:
        if type(expected_code) is types.CodeType and expected_code.co_name=='is_ffmpeg_installed':
            _DIAGNOSTIC_PROBE_CODE=expected_code
    except BaseException:pass


def _diagnostic_stage(stage,index=0):
    global _FAILURE_CONTEXT
    _FAILURE_CONTEXT=(stage,index)


def _safe_error_type(error):
    for known in _DIAGNOSTIC_ERRORS:
        if type(error) is known:return known.__name__
    return 'OtherException'


def _failure_diagnostic(error):
    context=_FAILURE_CONTEXT
    stage,index=context if type(context) is tuple and len(context)==2 else (None,None)
    stage=stage if type(stage) is str and stage in _DIAGNOSTIC_STAGES else 'unknown'
    index=index if type(index) is int and 0<=index<=12 else None
    result={'schema':'qwen12-bounded-failure-v1','stage':stage,'index':index,
        'family':_FAILURE_FAMILY if type(_FAILURE_FAMILY) is str and _FAILURE_FAMILY in ('qwen06','sensevoice','whisper') else 'unknown',
        'mode':_FAILURE_MODE if type(_FAILURE_MODE) is str and _FAILURE_MODE in ('prepare','decode') else 'unknown',
        'frames':[],'frames_truncated':False,'frame_scope':'first_64_tail_16','chain_followed':False,
        'message_disclosed':False,'source_lines_disclosed':False,'locals_disclosed':False}
    # Use built-in descriptors, never user-overridable exception properties.
    number=OSError.errno.__get__(error) if isinstance(error,OSError) else None
    if type(number) is int and 1<=number<=4095:result['errno']=number
    trace=BaseException.__traceback__.__get__(error)
    for _ in range(MAX_DIAGNOSTIC_SCAN):
        if trace is None:break
        code=trace.tb_frame.f_code;known=_DIAGNOSTIC_CODES.get(id(code))
        frame={'file':'<redacted>','function':'<redacted>','line':None}
        location=(known[1],known[2]) if known is not None and known[0] is code else None
        probe=_DIAGNOSTIC_PROBE_CODE
        if location is None and probe is not None and code.co_filename==probe.co_filename and code==probe:
            location=('runtime/site-packages/funasr/utils/load_utils.py','is_ffmpeg_installed')
        if location is not None:
            line=trace.tb_lineno
            frame={'file':location[0],'function':location[1],
                'line':line if type(line) is int and 1<=line<=1000000 else None}
        result['frames'].append(frame)
        if len(result['frames'])>MAX_DIAGNOSTIC_FRAMES:
            del result['frames'][0];result['frames_truncated']=True
        trace=trace.tb_next
    if trace is not None:result['frames_truncated']=True
    require(len(canonical(result))<=MAX_DIAGNOSTIC_BYTES,'bounded diagnostic')
    return result


def write_failure(error,output):
    record={'error_type':_safe_error_type(error),'status':'failed_no_retry',**BOUNDARY}
    try:
        record['diagnostic']=_failure_diagnostic(error)
        require(len(canonical(record))<=MAX_DIAGNOSTIC_BYTES,'bounded failure record')
    except BaseException:record['diagnostic']={'schema':'qwen12-bounded-failure-v1','status':'unavailable'}
    try:write_once(output/'worker-failure.json',record)
    except BaseException:pass


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result; spec.loader.exec_module(result); _register_diagnostic_codes(); return result


def alarm(signum,frame):raise TimeoutError('clip bound')


def progress(output,stage,index):
    atomic(output/'progress.json',{'stage':stage,'index':index,'started_monotonic':time.monotonic()})


def verify_prefrozen(output):
    for name in ('native-source-lock.json','native-asset-lock.json','native-config.json'):
        require((output/name).read_bytes()==(Path('/frozen')/name).read_bytes(), 'new loaded native lock/config differs from whole-policy freeze')
    for name in ('native-resolved-generation-config.json','native-frontend-config.json'):
        if (output/name).exists():require((output/name).read_bytes()==(Path('/frozen')/name).read_bytes(),'live native dependency drift')


def prepared_or_compare(output,mode):
    if mode=='prepare':
        _diagnostic_stage('prepare_receipt')
        write_once(output/'prepared.json',{'status':'loaded_observed_before_recognition','audio_forward_calls':0,**BOUNDARY})
        return True
    _diagnostic_stage('compare_frozen')
    verify_prefrozen(output)
    return False


def qwen_sense(family,runtime,inputs,output,job,mode):
    _diagnostic_stage('import_worker')
    old=ROOT/'research/source-screen32-v1';sys.path.insert(0,str(old))
    worker=module('qwen12_retained_worker',old/'asr_worker.py')
    _diagnostic_stage('verify_worker')
    worker.verify_scope();os.environ.update(worker.THREAD_ENV);os.environ.update(worker.OFFLINE_ENV)
    _diagnostic_stage('prepare_caches');worker.prepare_caches()
    _diagnostic_stage('verify_sources');worker.verify_sources()
    _diagnostic_stage('import_helpers');helpers,evidence=worker.import_helpers()
    _register_diagnostic_codes()
    _diagnostic_stage('bind_inputs')
    _,job_raw,decoder_raw=worker.bind_inputs(inputs)
    # Explicitly use the new setup binding, including the fresh10GiB/12GiB limits.
    _diagnostic_stage('verify_runtime')
    setup=module('qwen12_setup_binding',HERE/'setup_adapter.py');verified=setup.verify_runtime(runtime,'asr')
    from native_locks import verify_installed_extension_sources,build_native_locks
    _register_diagnostic_codes();_diagnostic_stage('verify_extensions')
    extension=verify_installed_extension_sources(runtime,read_json(worker.RETAINED/'runtime-lock.json')) if family=='qwen06' else None
    _diagnostic_stage('read_locks')
    locks=worker.decode(worker.read_regular(worker.RETAINED/'model-locks.json',65536));lock=locks[family]
    _diagnostic_stage('contract')
    declaration=worker.execution_contract(family,lock,decoder_raw);contract_sha=write_once(output/'contract.json',declaration)
    write_once(output/'decoder-inputs.json',worker.decode(decoder_raw,1024**2))
    loader=worker.load_qwen_after_approval if family=='qwen06' else worker.load_sense_after_approval
    model_root=runtime/'models'/family
    _diagnostic_stage('load_model')
    execution_policy_bytes=None
    if family=='sensevoice':
        from optional_tools import unavailable_ffmpeg,policy_config
        _register_diagnostic_codes()
        with unavailable_ffmpeg(runtime,read_json(output/'cpu-process-guard.json'),
                lambda value:write_once(output/'optional-tool-probes.json',value),
                register_code=register_optional_probe_code):
            loaded=loader(str(model_root),lock['asset_lock'],lock['asset_lock_sha256'],lock['source_lock'],lock['source_lock_sha256'],declaration,contract_sha,decoder_raw)
        execution_policy_bytes=canonical(policy_config())
    else:
        loaded=loader(str(model_root),lock['asset_lock'],lock['asset_lock_sha256'],lock['source_lock'],lock['source_lock_sha256'],declaration,contract_sha,decoder_raw)
    _diagnostic_stage('load_receipt')
    write_once(output/'model-load.json',loaded.receipt)
    _diagnostic_stage('import_adapter')
    native=module('qwen12_native_adapter',old/'native_v2_adapter.py')
    frontend=None
    if family=='sensevoice':
        _diagnostic_stage('frontend')
        write_once(output/'native-frontend-config.json',native.sense_effective(loaded,model_root)[1])
        frontend=(output/'native-frontend-config.json').read_bytes()
    _diagnostic_stage('native_locks')
    source_raw,asset_raw=build_native_locks(runtime,lock,output,family=family,code_root=ROOT,
        loaded_receipt=loaded.receipt,verified_runtime=verified,executed_head_sha=os.environ['KWS_EXECUTED_HEAD_SHA'],
        run_id=os.environ['KWS_RUN_ID'],installed_source_inventory=extension,frontend_config_bytes=frontend,execution_policy_bytes=execution_policy_bytes)
    _diagnostic_stage('construct_adapter')
    adapter=native.NativeCaptureAdapter(loaded,model_root,source_raw,asset_raw,lambda n,v:write_once(output/n,v),frontend_config_bytes=frontend)
    if prepared_or_compare(output,mode):return
    _diagnostic_stage('bind_predecode')
    binding=bind_predecode(recognizer_id=family,policy_path=Path('/policy/policy.json'),
        expected_policy_sha256=os.environ['KWS_POLICY_SHA256'],frozen_dir=Path('/frozen'),current_dir=output,
        run_id=os.environ['KWS_RUN_ID'],executed_head_sha=os.environ['KWS_EXECUTED_HEAD_SHA'])
    _diagnostic_stage('decoder')
    bindings=worker.validate_decoder(decoder_raw,digest(decoder_raw))
    for index,clip in enumerate(job['clips'],1):
        _diagnostic_stage('bind_audio',index)
        worker.verify_scope();descriptor=bindings[clip['audio_id']]['descriptor']
        bound=worker.bind_wave(str(inputs/'audio'),worker.WaveExpectation(clip['audio_id'],clip['wav_sha256']+'.wav',
            clip['wav_sha256'],descriptor['wav']['bytes'],descriptor['raw_pcm16']['frame_count']))
        require(bound.descriptor==descriptor,'full bound PCM unchanged')
        directory=output/clip['audio_id'];directory.mkdir();progress(output,'cell',index)
        _diagnostic_stage('start_attempt',index)
        attempt=emit_start(binding,directory,wav_path=inputs/clip['audio_path'],expected_file_sha256=clip['wav_sha256'],
            request_id='qwen12-'+os.environ['KWS_RUN_ID']+'-'+family+'-'+clip['audio_id'])
        _diagnostic_stage('infer',index)
        signal.setitimer(signal.ITIMER_REAL,120)
        row,raw=adapter.infer(bound,digest(decoder_raw),{})
        signal.setitimer(signal.ITIMER_REAL,0)
        _diagnostic_stage('complete_attempt',index)
        emit_completion(attempt,row=row,raw=raw)


def whisper(runtime,inputs,output,job,mode):
    _diagnostic_stage('import_adapter')
    base=ROOT/'research/source-screen32-whisper-v1';sys.path.insert(0,str(base))
    adapter=module('qwen12_whisper_adapter',base/'adapter.py')
    from native_locks import build_whisper_locks
    _register_diagnostic_codes();_diagnostic_stage('native_locks')
    source_raw,asset_raw=build_whisper_locks(runtime,output,code_root=ROOT,
        official_source_lock=read_json(base/'source-lock.json'),verified_asset_lock=read_json(runtime/'asset-lock.json'),
        executed_head_sha=os.environ['KWS_EXECUTED_HEAD_SHA'],run_id=os.environ['KWS_RUN_ID'])
    from whisper_binding import construct_whisper
    _register_diagnostic_codes();_diagnostic_stage('load_model')
    recognizer=construct_whisper(adapter,runtime,output,source_raw,asset_raw)
    if prepared_or_compare(output,mode):return
    _diagnostic_stage('bind_predecode')
    binding=bind_predecode(recognizer_id='whisper',policy_path=Path('/policy/policy.json'),
        expected_policy_sha256=os.environ['KWS_POLICY_SHA256'],frozen_dir=Path('/frozen'),current_dir=output,
        run_id=os.environ['KWS_RUN_ID'],executed_head_sha=os.environ['KWS_EXECUTED_HEAD_SHA'])
    common=sys.modules['common']
    for index,clip in enumerate(job['clips'],1):
        _diagnostic_stage('bind_audio',index)
        pcm,frames=common.wav_bytes(inputs/clip['audio_path'],clip['wav_sha256'])
        directory=output/clip['audio_id'];directory.mkdir();progress(output,'cell',index)
        _diagnostic_stage('start_attempt',index)
        attempt=emit_start(binding,directory,wav_path=inputs/clip['audio_path'],expected_file_sha256=clip['wav_sha256'],
            request_id='qwen12-'+os.environ['KWS_RUN_ID']+'-whisper-'+clip['audio_id'])
        _diagnostic_stage('infer',index)
        signal.setitimer(signal.ITIMER_REAL,120)
        raw=recognizer.decode(pcm,frames,directory)
        signal.setitimer(signal.ITIMER_REAL,0)
        _diagnostic_stage('complete_attempt',index)
        emit_completion(attempt,row=None,raw=raw)


def main():
    global _FAILURE_FAMILY,_FAILURE_MODE
    _diagnostic_stage('startup')
    require(len(sys.argv)==3 and sys.argv[1] in ('qwen06','sensevoice','whisper') and sys.argv[2] in ('prepare','decode'),'fixed family/mode')
    family=sys.argv[1];mode=sys.argv[2];runtime=Path('/runtime');inputs=Path('/input');output=Path('/output')
    _FAILURE_FAMILY=family;_FAILURE_MODE=mode
    _diagnostic_stage('scope')
    scope=module('qwen12_kernel_scope',HERE/'runtime_scope.py').verify_runtime_scope()
    _diagnostic_stage('output')
    require(not list(output.iterdir()),'fresh isolated worker output')
    _diagnostic_stage('guard')
    guard=install_inference_guard(family)
    _diagnostic_stage('provenance')
    require(re.fullmatch('[0-9a-f]{40}',os.environ.get('KWS_EXECUTED_HEAD_SHA','')) and
        re.fullmatch('[1-9][0-9]{0,19}',os.environ.get('KWS_RUN_ID','')),'real public run provenance')
    write_once(output/'run-identity.json',{'schema':'qwen12-real-actions-identity-v1',
        'run_id':int(os.environ['KWS_RUN_ID']),'executed_head_sha':os.environ['KWS_EXECUTED_HEAD_SHA']})
    _diagnostic_stage('inputs')
    job=validate_blind(inputs)
    _diagnostic_stage('initial_receipts')
    write_once(output/'kernel-scope.json',scope);write_once(output/'cpu-process-guard.json',guard)
    write_once(output/'blind-job.json',job);progress(output,'load',0)
    signal.signal(signal.SIGALRM,alarm)
    (whisper if family=='whisper' else lambda *a:qwen_sense(family,*a))(runtime,inputs,output,job,mode)
    _diagnostic_stage('worker_complete')
    write_once(output/'worker-complete.json',{'calls':0 if mode=='prepare' else 12,'status':'prepared_no_recognition' if mode=='prepare' else 'all12_native_v2_complete',**BOUNDARY})


def run():
    try:main();return 0
    except BaseException as error:
        # Diagnostics and timer cleanup are independently best effort. Neither
        # can replace the original failure with success or a raw traceback.
        try:signal.setitimer(signal.ITIMER_REAL,0)
        except BaseException:pass
        try:write_failure(error,Path('/output'))
        except BaseException:pass
        return 1


_register_diagnostic_codes()
if __name__=='__main__':raise SystemExit(run())
