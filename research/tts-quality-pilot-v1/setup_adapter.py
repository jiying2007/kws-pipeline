"""Reuse retained setup with only this campaign's explicit model binding."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
path=HERE.parent/'source-screen32-v1/setup_adapter.py'
spec=importlib.util.spec_from_file_location('retained_setup_adapter',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
selected=json.loads((HERE/'qwen-model-lock.json').read_bytes())
identities={
 'qwen06': ('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice','85e237c12c027371202489a0ec509ded67b5e4b5'),
 'qwen17': ('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice','0c0e3051f131929182e2c023b9537f8b1c68adfe'),
}
matched=[key for key,value in identities.items() if value==(selected.get('model_id'),selected.get('revision'))]
if len(matched)!=1: raise ValueError('unknown staged campaign model identity')
engine=matched[0]
if os.environ.get('TQP_ENGINE',engine)!=engine: raise ValueError('engine environment/lock mismatch')
lockpath=HERE/(engine+'-model-lock.json')
lock=json.loads(lockpath.read_bytes())
base.ROOT=HERE
base.MODEL_ID=lock['model_id'];base.MODEL_REVISION=lock['revision']
base.MODEL_LOCK_SHA256=hashlib.sha256(lockpath.read_bytes()).hexdigest()
# The retained binder uses this fixed basename. Select in read-only staged code,
# never alter either original source or an already-created runtime.
if (HERE/'qwen-model-lock.json').read_bytes()!=lockpath.read_bytes():
    raise ValueError('staged engine lock mismatch')
_run_child=base._run_child
bind=base.bind
verify_runtime=base.verify_runtime
if __name__=='__main__': base.main()
