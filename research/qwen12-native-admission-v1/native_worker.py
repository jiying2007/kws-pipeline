"""One isolated 12-clip process; new captures, no historical-result conversion."""
from pathlib import Path
import importlib.util
import os
import re
import signal
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from campaign_contract import BOUNDARY, canonical, digest, identity, read_json, require, validate_blind, write_once, atomic
from cpu_budget import install_inference_guard
from worker_receipts import bind_predecode,emit_start,emit_completion


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result; spec.loader.exec_module(result); return result


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
        write_once(output/'prepared.json',{'status':'loaded_observed_before_recognition','audio_forward_calls':0,**BOUNDARY})
        return True
    verify_prefrozen(output)
    return False


def qwen_sense(family,runtime,inputs,output,job,mode):
    old=ROOT/'research/source-screen32-v1';sys.path.insert(0,str(old))
    worker=module('qwen12_retained_worker',old/'asr_worker.py')
    worker.verify_scope();os.environ.update(worker.THREAD_ENV);os.environ.update(worker.OFFLINE_ENV)
    worker.prepare_caches();worker.verify_sources();helpers,evidence=worker.import_helpers()
    _,job_raw,decoder_raw=worker.bind_inputs(inputs)
    # Explicitly use the new setup binding, including the fresh10GiB/12GiB limits.
    setup=module('qwen12_setup_binding',HERE/'setup_adapter.py');verified=setup.verify_runtime(runtime,'asr')
    from native_locks import verify_installed_extension_sources,build_native_locks
    extension=verify_installed_extension_sources(runtime,read_json(worker.RETAINED/'runtime-lock.json')) if family=='qwen06' else None
    locks=worker.decode(worker.read_regular(worker.RETAINED/'model-locks.json',65536));lock=locks[family]
    declaration=worker.execution_contract(family,lock,decoder_raw);contract_sha=write_once(output/'contract.json',declaration)
    write_once(output/'decoder-inputs.json',worker.decode(decoder_raw,1024**2))
    loader=worker.load_qwen_after_approval if family=='qwen06' else worker.load_sense_after_approval
    model_root=runtime/'models'/family
    loaded=loader(str(model_root),lock['asset_lock'],lock['asset_lock_sha256'],lock['source_lock'],lock['source_lock_sha256'],declaration,contract_sha,decoder_raw)
    write_once(output/'model-load.json',loaded.receipt)
    native=module('qwen12_native_adapter',old/'native_v2_adapter.py')
    frontend=None
    if family=='sensevoice':
        write_once(output/'native-frontend-config.json',native.sense_effective(loaded,model_root)[1])
        frontend=(output/'native-frontend-config.json').read_bytes()
    source_raw,asset_raw=build_native_locks(runtime,lock,output,family=family,code_root=ROOT,
        loaded_receipt=loaded.receipt,verified_runtime=verified,executed_head_sha=os.environ['KWS_EXECUTED_HEAD_SHA'],
        run_id=os.environ['KWS_RUN_ID'],installed_source_inventory=extension,frontend_config_bytes=frontend)
    adapter=native.NativeCaptureAdapter(loaded,model_root,source_raw,asset_raw,lambda n,v:write_once(output/n,v),frontend_config_bytes=frontend)
    if prepared_or_compare(output,mode):return
    binding=bind_predecode(recognizer_id=family,policy_path=Path('/policy/policy.json'),
        expected_policy_sha256=os.environ['KWS_POLICY_SHA256'],frozen_dir=Path('/frozen'),current_dir=output,
        run_id=os.environ['KWS_RUN_ID'],executed_head_sha=os.environ['KWS_EXECUTED_HEAD_SHA'])
    bindings=worker.validate_decoder(decoder_raw,digest(decoder_raw))
    for index,clip in enumerate(job['clips'],1):
        worker.verify_scope();descriptor=bindings[clip['audio_id']]['descriptor']
        bound=worker.bind_wave(str(inputs/'audio'),worker.WaveExpectation(clip['audio_id'],clip['wav_sha256']+'.wav',
            clip['wav_sha256'],descriptor['wav']['bytes'],descriptor['raw_pcm16']['frame_count']))
        require(bound.descriptor==descriptor,'full bound PCM unchanged')
        directory=output/clip['audio_id'];directory.mkdir();progress(output,'cell',index)
        attempt=emit_start(binding,directory,wav_path=inputs/clip['audio_path'],expected_file_sha256=clip['wav_sha256'],
            request_id='qwen12-'+os.environ['KWS_RUN_ID']+'-'+family+'-'+clip['audio_id'])
        signal.setitimer(signal.ITIMER_REAL,120)
        row,raw=adapter.infer(bound,digest(decoder_raw),{})
        signal.setitimer(signal.ITIMER_REAL,0)
        emit_completion(attempt,row=row,raw=raw)


def whisper(runtime,inputs,output,job,mode):
    base=ROOT/'research/source-screen32-whisper-v1';sys.path.insert(0,str(base))
    adapter=module('qwen12_whisper_adapter',base/'adapter.py')
    from native_locks import build_whisper_locks
    source_raw,asset_raw=build_whisper_locks(runtime,output,code_root=ROOT,
        official_source_lock=read_json(base/'source-lock.json'),verified_asset_lock=read_json(runtime/'asset-lock.json'),
        executed_head_sha=os.environ['KWS_EXECUTED_HEAD_SHA'],run_id=os.environ['KWS_RUN_ID'])
    from whisper_binding import construct_whisper
    recognizer=construct_whisper(adapter,runtime,output,source_raw,asset_raw)
    if prepared_or_compare(output,mode):return
    binding=bind_predecode(recognizer_id='whisper',policy_path=Path('/policy/policy.json'),
        expected_policy_sha256=os.environ['KWS_POLICY_SHA256'],frozen_dir=Path('/frozen'),current_dir=output,
        run_id=os.environ['KWS_RUN_ID'],executed_head_sha=os.environ['KWS_EXECUTED_HEAD_SHA'])
    common=sys.modules['common']
    for index,clip in enumerate(job['clips'],1):
        pcm,frames=common.wav_bytes(inputs/clip['audio_path'],clip['wav_sha256'])
        directory=output/clip['audio_id'];directory.mkdir();progress(output,'cell',index)
        attempt=emit_start(binding,directory,wav_path=inputs/clip['audio_path'],expected_file_sha256=clip['wav_sha256'],
            request_id='qwen12-'+os.environ['KWS_RUN_ID']+'-whisper-'+clip['audio_id'])
        signal.setitimer(signal.ITIMER_REAL,120)
        raw=recognizer.decode(pcm,frames,directory)
        signal.setitimer(signal.ITIMER_REAL,0)
        emit_completion(attempt,row=None,raw=raw)


def main():
    require(len(sys.argv)==3 and sys.argv[1] in ('qwen06','sensevoice','whisper') and sys.argv[2] in ('prepare','decode'),'fixed family/mode')
    family=sys.argv[1];mode=sys.argv[2];runtime=Path('/runtime');inputs=Path('/input');output=Path('/output')
    scope=module('qwen12_kernel_scope',HERE/'runtime_scope.py').verify_runtime_scope()
    require(not list(output.iterdir()),'fresh isolated worker output')
    guard=install_inference_guard(family)
    require(re.fullmatch('[0-9a-f]{40}',os.environ.get('KWS_EXECUTED_HEAD_SHA','')) and
        re.fullmatch('[1-9][0-9]{0,19}',os.environ.get('KWS_RUN_ID','')),'real public run provenance')
    write_once(output/'run-identity.json',{'schema':'qwen12-real-actions-identity-v1',
        'run_id':int(os.environ['KWS_RUN_ID']),'executed_head_sha':os.environ['KWS_EXECUTED_HEAD_SHA']})
    job=validate_blind(inputs)
    write_once(output/'kernel-scope.json',scope);write_once(output/'cpu-process-guard.json',guard)
    write_once(output/'blind-job.json',job);progress(output,'load',0)
    signal.signal(signal.SIGALRM,alarm)
    (whisper if family=='whisper' else lambda *a:qwen_sense(family,*a))(runtime,inputs,output,job,mode)
    write_once(output/'worker-complete.json',{'calls':0 if mode=='prepare' else 12,'status':'prepared_no_recognition' if mode=='prepare' else 'all12_native_v2_complete',**BOUNDARY})


if __name__=='__main__':
    try:main()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:write_once(Path('/output/worker-failure.json'),{'error_type':type(error).__name__,'status':'failed_no_retry',**BOUNDARY})
        except BaseException:pass
        raise SystemExit(1)
