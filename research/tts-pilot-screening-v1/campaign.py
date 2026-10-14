"""New pilot design identity, no execution authorization."""
from pathlib import Path
import hashlib,json
MODELS={'qwen06': ('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', '85e237c12c027371202489a0ec509ded67b5e4b5'), 'qwen17': ('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', '0c0e3051f131929182e2c023b9537f8b1c68adfe')}
PLAN_SHA256='b639367f3d678ca5666220a651b1d9189ec24cb318fb064a9091fae0e0a24c5a'
def expected_plan():
    raw=(Path(__file__).resolve().parent/'expected-plan.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PLAN_SHA256:raise ValueError('pilot design drift')
    return json.loads(raw)
