"""No-audio, guarded real model loading and native configuration preparation."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from zero_forward import ForwardFence,primary_loader
from safe_diagnostic import StageTracker
from qwen_worker import module,source_bodies,require

def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n')

def execute(family,head,code=Path('/code'),runtime=Path('/runtime'),output=Path('/output')):
    with StageTracker(output,family) as tracker:
        proof=_execute(family,head,code,runtime,output,tracker)
        tracker.finish()
    return proof

def _execute(family,head,code,runtime,output,tracker):
    require(family in ('qwen06','sensevoice','whisper'),'fixed prepare family')
    require(not Path('/input').exists(),'zero-audio preparation must not mount or create input')
    sys.path[:0]=[str(code/'tools'),str(code/'research/qwen-native-capture-20261104'),
                 str(code/'research/source-screen32-v1')]
    tracker.enter('verify_scope')
    legacy=module(code/'research/source-screen32-v1/asr_worker.py','zero_audio_legacy')
    legacy.verify_scope()
    os.environ.update(legacy.THREAD_ENV);os.environ.update(legacy.OFFLINE_ENV);legacy.prepare_caches()
    tracker.enter('verify_sources')
    legacy.verify_sources();helpers,evidence=legacy.import_helpers()
    write(output/'progress.json',dict(stage='load',index=0,started_monotonic=time.monotonic()))
    # Runtime verification precedes trusted imports and any model construction.
    tracker.enter('verify_runtime')
    if family!='whisper':
        from setup_adapter import verify_runtime
        verify_runtime(runtime,'asr')
    else:
        sys.path.insert(0,str(code/'research/whisper-native-capture-20261104'))
        import whisper_capture
        whisper_capture.verify_prepare_imports(code,runtime)
    tracker.enter('import_framework')
    import torch
    extras=[(evidence,'infer_qwen_once_after_approval'),(evidence,'infer_sense_once_after_approval')]
    tracker.enter('import_model')
    if family=='qwen06':
        from qwen_asr import Qwen3ASRModel
        from transformers.generation.utils import GenerationMixin
        extras.extend([(Qwen3ASRModel,'transcribe'),(GenerationMixin,'generate')])
    elif family=='sensevoice':
        from funasr import AutoModel
        import sense_adapter
        extras.extend([(AutoModel,'generate'),(AutoModel,'inference'),(sense_adapter,'infer_sense_once_after_approval')])
    else:
        import whisper
        extras.extend([(whisper,'decode'),(whisper,'transcribe'),(whisper.decoding.DecodingTask,'run')])
    tracker.enter('install_forward_guard')
    with ForwardFence(torch,extras) as fence:
        require(fence.active,'guard precedes actual loader')
        tracker.enter('review_native_sources')
        if family=='qwen06':
            import capture_worker
            def prepare(loaded):
                tracker.enter('prepare_native_identity')
                bodies=source_bodies(code,code/'research/native-asr-integration-20261104/qwen-source-members.json',loaded)
                return capture_worker.prepare_strict_identity(loaded,output/'prepared',head,bodies)['identity']
            tracker.enter('load_model')
            identity=primary_loader(legacy,family,runtime,prepare)
        elif family=='sensevoice':
            sys.path.insert(0,str(code/'research/sense-native-capture-20261104'))
            sense=module(code/'research/sense-native-capture-20261104/worker.py','zero_audio_sense')
            review=sense.read_object(code/'runtime-policy/sense-review.json')
            bodies=sense.source_closure(review)
            from sense_binding import _contract
            groups=_contract()['SENSE_GROUPS']
            def prepare(loaded):
                tracker.enter('prepare_native_identity')
                return sense.strict_identity(loaded,output/'prepared',review,bodies,
                    code/'research/qwen6_asr/sense_parser.py',groups,
                    forward_metadata=fence.method_metadata)['identity']
            tracker.enter('load_model')
            identity=primary_loader(legacy,family,runtime,prepare)
        else:
            sys.path.insert(0,str(code/'research/whisper-native-capture-20261104'))
            import whisper_capture
            tracker.enter('load_model')
            extras_loaded=whisper_capture.prepare(code,runtime,output/'prepared',head,
                code/'research/qwen-native-capture-20261104/native_sink.py',code/'tools',maximum=16,
                on_loaded=lambda:tracker.enter('prepare_native_identity'))
            identity=extras_loaded.identity
        tracker.enter('verify_zero_forward')
        fence.assert_zero()
        tracker.enter('write_preparation_proof')
        write(output/'identity.json',identity)
        proof=dict(schema='actual-zero-forward-preparation-proof-v1',family=family,status='prepared_zero_forward',
            candidate_count=0,audio_inputs=0,observed_guarded_entrypoint_attempts=len(fence.attempts),
            guard_coverage=['torch.nn.Module._call_impl','existing-module-subclass-direct-entrypoints','reviewed-family-public-inference-entrypoints'],
            guard_limits=['not-all-functional-operators','not-direct-methods-on-future-subclasses'],
            guarded_entrypoints=len(fence.saved),guard_installed_before_model_loader=True,
            native_config_sha256=identity['native_decode_config_sha256'],
            source_lock_sha256=identity['native_decode_config']['source_lock_sha256'],
            asset_lock_sha256=identity['native_decode_config']['asset_lock_sha256'],
            training_admitted=False,human_gold=False)
        write(output/'zero-forward-proof.json',proof)
    return proof

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',required=True,choices=('qwen06','sensevoice','whisper'))
    parser.add_argument('--executed-head',required=True)
    args=parser.parse_args();execute(args.family,args.executed_head)

if __name__=='__main__':main()
