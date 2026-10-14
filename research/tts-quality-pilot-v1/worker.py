from pathlib import Path
import os,sys,time,json
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from pilot import run_loaded,save,scope

def main():
    output=Path('/output')
    status={'status':'started','stage':'scope','network_events':[]}
    try:
        scope()
        save(output/'progress.json',dict(stage='load',index=0,started_monotonic=time.monotonic()))
        status['stage']='verify_runtime'
        from setup_adapter import verify_runtime,engine
        verify_runtime(Path('/runtime'),'tts')
        gate=sys.modules['runtime_gate']
        status['stage']='import_runtime'
        torch,transformers,events=gate.prepare()
        status['network_events']=events
        import numpy as np
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
        model_dir=Path('/runtime/models/qwen_tts')
        sidecars=gate.inspect_model_sidecars(model_dir);calls=[]
        status['stage']='load_model'
        with gate.local_loader_policy(model_dir,torch,transformers,calls):
            model=Qwen3TTSModel.from_pretrained(str(model_dir),device_map='cpu',dtype=torch.float32,
                 attn_implementation='eager',use_safetensors=True,weights_only=True,
                 local_files_only=True,trust_remote_code=False)
            save(output/'runtime-observations.json',dict(sidecars=sidecars,loader_calls=calls))
            status['stage']='generation'
            run_loaded(model,torch,np,sf,engine,output,'probe')
        status['status']='probe_completed'
    except BaseException as e:
        status.update(status='stopped_no_retry',error_type=type(e).__name__)
        raise
    finally:
        save(output/'worker-status.json',status)

if __name__=='__main__':
    try: main()
    except BaseException as e:
        print(json.dumps({'status':'stopped_no_retry','error_type':type(e).__name__}),file=sys.stderr)
        raise SystemExit(1)
