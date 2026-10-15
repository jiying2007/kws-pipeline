"""Small research-only numerical contract; no production admission authority."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

DENIALS = dict(training_admitted=False, production_qualified=False, shipping_qualified=False,
              reusable_weights_exported=False, human_gold=False, independent_evaluation=False)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def collapse(path):
    result, previous = [], None
    for token in path:
        if type(token) is not int or token not in range(5):
            raise ValueError('invalid token')
        if token != previous and token != 0:
            result.append(token)
        previous = token
    return result

def weights(classes, stage):
    count = Counter(classes)
    if set(count) != {'KW1', 'KW2'}:
        raise ValueError('both classes required')
    if stage == 'B':
        return [1 / (2 * count[c]) for c in classes]
    if stage not in ('pair', 'A'):
        raise ValueError('unknown stage')
    return [1 / len(classes)] * len(classes)

def select_pair(records):
    by_class = {c: sorted((r for r in records if r['target_class'] == c), key=lambda r:r['record_id']) for c in ('KW1','KW2')}
    common = sorted(set(r['preset_voice_id'] for r in by_class['KW1']) & set(r['preset_voice_id'] for r in by_class['KW2']))
    return [next(r for r in by_class[c] if not common or r['preset_voice_id'] == common[0])['record_id'] for c in ('KW1','KW2')]

def atomic_json(path, data):
    """Durable exclusive publication: existing evidence is never overwritten."""
    path = Path(path)
    raw = (json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(temporary)

def strict_equal(a, b):
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(strict_equal(x,y) for x,y in zip(a,b))
    return a == b

FIXED_PLAN = {'schema': 'kws-prefix-blank-anchor-plan-v1', 'purpose': 'exposed-machine-only-prefix-blank-causal-diagnostic', 'production_qualified': False, 'seed': 20261015, 'model': {'feature_dim': 32, 'hidden_dim': 48, 'vocab_size': 5}, 'targets_by_text': {'你好小窝': [1, 2, 3, 4], '小窝小窝': [3, 4, 3, 4]}, 'optimizer': {'name': 'Adam', 'lr': 0.003, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'eps': 1e-08, 'gradient_clip': 5.0}, 'ctc': {'blank': 0, 'reduction': 'none', 'zero_infinity': False, 'normalization': 'per-sample-input-frame-count'}, 'max_optimizer_calls': 600, 'retries': 0, 'new_tts_calls': 0, 'new_asr_calls': 0, 'warm_start': False, 'augmentation': False, 'vad': False, 'export_weights': False, 'pair_record_ids': ['tts-quality-remaining-20261014-v1:tqp-qwen06-05', 'tts-quality-remaining-20261014-v1:tqp-qwen06-06'], 'registry_sha256': '28033edbbc1e444eae27c33af2e060d1654c4b7a072e65401b055fc27fc3b1b0', 'source_verifier_sha256': '9d37fb852cdd5048a4d5c88f9e21efe2b3c1e18f25244babb2d811675ea9fbae', 'runtime_lock_sha256': 'b132c2c202e5ff00b6cbd1b2b630fa92dfe9ba46fcf0423dc0633381bbe80984', 'code_sha256': {'training/model.py': '768800e25537367da770ef44178249208956d7c3c33f1eb7d60b1b9c98fe59c5', 'training/frontend.py': 'd8ae687303d5c9c6bfdf6539c94f4ae463d910a4837ed77633d7656ff02a582e', 'training/frontend_spec.py': 'fbd17236e1449101bed6df8f105baad476edb0b005e6e45cb83ebc7971d166e6'}, 'campaign_id': 'kws-prefix-blank-anchor-20261015-v1', 'campaign_claim_path': 'research-budget/kws-prefix-blank-anchor-20261015-v1.claim.json', 'input_pins_sha256': '64c77a270c7f13d8252d10282191aaeb207f425c10fb89a2f01e390fa994e37c', 'frontend': {'frontend': 'logmel', 'feature_dim': 32, 'frame_len': 400, 'hop': 320}, 'stages': [{'name': 'C', 'updates': 300, 'checkpoints': [0, 50, 100, 200, 300], 'blank_lambda': 0}, {'name': 'T', 'updates': 300, 'checkpoints': [0, 50, 100, 200, 300], 'blank_lambda': 1}], 'prefix_frames': {'KW1': 23, 'KW2': 10}, 'observation_timing': 'post-step; step 0 before any optimizer call', 'loss_timing': 'pre-step', 'max_model_forward_calls': 610, 'control_comparison': 'exact frozen public observables at 0/50/100/200/300; mismatch stops entire campaign', 'energy_sha256': {'prefix-energy-summary.json': 'c165bcd51a8ee2b9dabd975ca396858a2b27b3d5e331caff82c423bb2436dbba', 'prefix-energy-windows.json': '6cf75569e14fef2ec63ab557c750fcc1c1482b5325e8a52ab970cbb229934220'}, 'control_baseline_sha256': '7454a7d86521a52454f80e5bc15e250b4be35a5fcc986d9fc4b92ce0446a3f41'}
CODE_FILES = {'run_anchor.py','contract.py','numeric_preflight.py','requirements.research.lock'}
def validate_plan(plan, require_enabled=True):
    if type(plan) is not dict or set(plan) != set(FIXED_PLAN)|{'execution_enabled','diagnostic_code_sha256'}:
        raise ValueError('closed plan keys required')
    if not strict_equal({k:plan[k] for k in FIXED_PLAN}, FIXED_PLAN):
        raise ValueError('fixed research contract changed')
    if type(plan['execution_enabled']) is not bool or (require_enabled and not plan['execution_enabled']):
        raise ValueError('execution disabled pending independent review and root approval')
    pins=plan['diagnostic_code_sha256']
    if type(pins) is not dict or set(pins)!=CODE_FILES or any(type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in pins.values()):
        raise ValueError('closed code pins required')

def compare_control(samples, frozen):
    return len(samples)==len(frozen) and all(strict_equal({k:a[k] for k in b}, b) for a,b in zip(samples,frozen))

class Budget:
    def __init__(self, output, campaign_claim, campaign_id):
        self.output=Path(output).absolute()
        claim=Path(campaign_claim).absolute()
        for p in (self.output,claim):
            if p.resolve()!=p or '..' in p.parts: raise ValueError('aliases forbidden')
        if campaign_id != FIXED_PLAN['campaign_id'] or claim.name!='kws-prefix-blank-anchor-20261015-v1.claim.json':
            raise ValueError('wrong campaign')
        if self.output.exists(): raise ValueError('output already exists')
        atomic_json(claim,dict(schema='prefix-anchor-exclusive-claim-v1',campaign_id=campaign_id,output=str(self.output),max_optimizer_calls=600,**DENIALS))
        self.output.mkdir(exist_ok=False)
        self.count=0; self.arm_counts={'C':0,'T':0}; self.control_passed=False
        atomic_json(self.output/'intent.json',dict(schema='prefix-anchor-intent-v1',campaign_id=campaign_id,max_optimizer_calls=600,no_resume=True,**DENIALS))
    def before_step(self,stage,update):
        if stage not in self.arm_counts or type(update) is not int or update!=self.arm_counts[stage]+1 or update>300 or self.count>=600:
            raise ValueError('budget/sequence violation')
        if stage=='T' and not self.control_passed: raise ValueError('control must pass first')
        if stage=='C' and self.arm_counts['T']: raise ValueError('control already closed')
        self.count+=1; self.arm_counts[stage]+=1
        atomic_json(self.output/f'call-{self.count:04d}.json',dict(status='step_reserved',stage=stage,update=update,optimizer_call_reserved=self.count))

def strict_json(raw):
    def pairs(rows):
        result={}
        for key,value in rows:
            if key in result: raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    def bad(value): raise ValueError('nonfinite JSON value')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
