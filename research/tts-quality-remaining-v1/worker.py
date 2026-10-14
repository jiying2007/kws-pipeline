"""Remaining-only worker; unchanged reviewed probe loader and CPU containment."""
from pathlib import Path
import json,sys,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'tts-quality-pilot-v1'))
from pilot import save,scope
from remaining import document,run_verified_loaded
from resource_receipt import sample,worker_final

def main():
    output=Path('/output');start=None
    status={'status':'started','stage':'scope','network_events':[],'phase':'remaining'}
    try:
        scope();start=sample()
        save(output/'progress.json',dict(stage='load',index=0,started_monotonic=time.monotonic()))
        status['stage']='verify_runtime'
        from setup_adapter import verify_runtime,engine
        verify_runtime(Path('/runtime'),'tts')
        gate=sys.modules['runtime_gate'];status['stage']='import_runtime'
        torch,transformers,events=gate.prepare();status['network_events']=events
        import numpy as np
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
        model_dir=Path('/runtime/models/qwen_tts')
        sidecars=gate.inspect_model_sidecars(model_dir);calls=[];status['stage']='load_model'
        with gate.local_loader_policy(model_dir,torch,transformers,calls):
            model=Qwen3TTSModel.from_pretrained(str(model_dir),device_map='cpu',dtype=torch.float32,
                attn_implementation='eager',use_safetensors=True,weights_only=True,
                local_files_only=True,trust_remote_code=False)
            save(output/'runtime-observations.json',dict(sidecars=sidecars,loader_calls=calls))
            status['stage']='generation'
            run_verified_loaded(model,torch,np,sf,document(HERE/('binding-'+engine+'.json')),engine,
                Path('/code/inputs/incoming'),Path('/code/probe-source'),Path('/code/inputs/probe.zip'),HERE/'source-manifest.json',output)
        status['status']='remaining_completed';status['stage']='complete'
    except BaseException as e:
        status.update(status='stopped_no_retry',error_type=type(e).__name__);raise
    finally:
        worker_final(output,start_sample=start,stage=status['stage'],completed=status['status']=='remaining_completed')
        save(output/'worker-status.json',status)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        print(json.dumps({'status':'stopped_no_retry','error_type':type(e).__name__}),file=sys.stderr);raise SystemExit(1)
