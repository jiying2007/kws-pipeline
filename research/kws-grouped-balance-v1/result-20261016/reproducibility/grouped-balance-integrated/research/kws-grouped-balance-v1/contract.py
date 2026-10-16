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

_WRITTEN_BYTES = {}
MAX_EVIDENCE_DIRECTORY_BYTES = 512*1024*1024

def atomic_json(path, data):
    """Durable exclusive publication: existing evidence is never overwritten."""
    path = Path(path)
    raw = (json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if len(raw)>1024*1024:raise ValueError('evidence exceeds fixed 1MiB per-file ceiling')
    directory_key=str(path.parent.absolute())
    # Leave one MiB for a durable failure receipt if the evidence budget is hit.
    ceiling=MAX_EVIDENCE_DIRECTORY_BYTES if path.name=='failure.json' else MAX_EVIDENCE_DIRECTORY_BYTES-1024*1024
    if _WRITTEN_BYTES.get(directory_key,0)+len(raw)>ceiling:raise ValueError('evidence exceeds fixed 512MiB directory ceiling')
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(temporary, path)
        _WRITTEN_BYTES[directory_key]=_WRITTEN_BYTES.get(directory_key,0)+len(raw)
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

FIXED_PLAN = {'schema': 'kws-grouped-balance-plan-v1', 'purpose': 'exposed-preset-group-disjoint-fixed-horizon-diagnostic', 'production_qualified': False, 'seed': 20261015, 'model': {'feature_dim': 32, 'hidden_dim': 48, 'vocab_size': 5}, 'targets_by_text': {'你好小窝': [1, 2, 3, 4], '小窝小窝': [3, 4, 3, 4]}, 'optimizer': {'name': 'Adam', 'lr': 0.003, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'eps': 1e-08, 'gradient_clip': 5.0}, 'ctc': {'blank': 0, 'reduction': 'none', 'zero_infinity': False, 'normalization': 'per-sample-input-frame-count'}, 'max_optimizer_calls': 1200, 'retries': 0, 'new_tts_calls': 0, 'new_asr_calls': 0, 'warm_start': False, 'augmentation': False, 'vad': False, 'export_weights': False, 'registry_sha256': '28033edbbc1e444eae27c33af2e060d1654c4b7a072e65401b055fc27fc3b1b0', 'source_verifier_sha256': '9d37fb852cdd5048a4d5c88f9e21efe2b3c1e18f25244babb2d811675ea9fbae', 'runtime_lock_sha256': 'b132c2c202e5ff00b6cbd1b2b630fa92dfe9ba46fcf0423dc0633381bbe80984', 'code_sha256': {'training/model.py': '768800e25537367da770ef44178249208956d7c3c33f1eb7d60b1b9c98fe59c5', 'training/frontend.py': 'd8ae687303d5c9c6bfdf6539c94f4ae463d910a4837ed77633d7656ff02a582e', 'training/frontend_spec.py': 'fbd17236e1449101bed6df8f105baad476edb0b005e6e45cb83ebc7971d166e6'}, 'campaign_id': 'kws-grouped-balance-20261015-v1', 'campaign_claim_path': 'research-budget/kws-grouped-balance-20261015-v1.claim.json', 'input_pins_sha256': '64c77a270c7f13d8252d10282191aaeb207f425c10fb89a2f01e390fa994e37c', 'frontend': {'frontend': 'logmel', 'feature_dim': 32, 'frame_len': 400, 'hop': 320}, 'stages': [{'name': 'A', 'updates': 600, 'checkpoints': [0, 50, 100, 200, 300, 400, 500, 600], 'blank_lambda': 1}, {'name': 'B', 'updates': 600, 'checkpoints': [0, 50, 100, 200, 300, 400, 500, 600], 'blank_lambda': 1}], 'observation_timing': 'post-step; step 0 before any optimizer call', 'loss_timing': 'pre-step', 'max_model_forward_calls': 1218, 'cohort_sha256': 'fe23e633925573e2b02ac8530e066214879a11cbccc2e4ef5875aa2c38006e87', 'prefix_rule': {'window_samples': 400, 'hop_samples': 320, 'threshold': 0.0001, 'rule': 'maximal initial run of complete unwindowed PCM RMS strictly below threshold; zero allowed'}, 'probe_timing': 'after both arms finish 600 updates; A then B; once per arm', 'train_weights': {'A': {'KW1': 0.08333333333333333, 'KW2': 0.08333333333333333}, 'B': {'KW1': 0.125, 'KW2': 0.0625}}, 'early_stop': False, 'extension': False, 'resume': False, 'probe_selection': False, 'train_record_ids': ['kw1-preset-supplement-20261014-v1:kw1s-05', 'kw1-preset-supplement-20261014-v1:kw1s-06', 'kw1-preset-supplement-20261014-v1:kw1s-07', 'tts-quality-remaining-20261014-v1:tqp-qwen06-05', 'tts-quality-remaining-20261014-v1:tqp-qwen06-06', 'tts-quality-remaining-20261014-v1:tqp-qwen06-10', 'tts-quality-remaining-20261014-v1:tqp-qwen06-11', 'tts-quality-remaining-20261014-v1:tqp-qwen06-14', 'tts-quality-remaining-20261014-v1:tqp-qwen06-15', 'tts-quality-remaining-20261014-v1:tqp-qwen17-06', 'tts-quality-remaining-20261014-v1:tqp-qwen17-14', 'tts-quality-remaining-20261014-v1:tqp-qwen17-15'], 'probe_record_ids': ['tts-quality-remaining-20261014-v1:tqp-qwen06-03', 'tts-quality-remaining-20261014-v1:tqp-qwen17-01', 'tts-quality-remaining-20261014-v1:tqp-qwen17-02', 'tts-quality-remaining-20261014-v1:tqp-qwen17-03']}
CODE_FILES = {"run_grouped_balance.py", "contract.py", "numeric_preflight.py", "requirements.research.lock"}
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

def strict_json(raw):
    def pairs(rows):
        result={}
        for key,value in rows:
            if key in result: raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    def bad(value): raise ValueError('nonfinite JSON value')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)


def forward_schedule():
    events=[]
    for stage in ('A','B'):
        events.append(('observation',stage,0))
        for update in range(1,601):
            events.append(('training',stage,update))
            if update in FIXED_PLAN['stages'][0]['checkpoints']:
                events.append(('observation',stage,update))
    return events+[('probe','A',600),('probe','B',600)]

class Budget:
    """Durable exclusive reservations; a pending event cannot be skipped or reused."""
    def __init__(self,output,campaign_claim,campaign_id):
        self.output=Path(output).absolute(); claim=Path(campaign_claim).absolute()
        for p in (self.output,claim):
            if p.resolve()!=p or '..' in p.parts:raise ValueError('aliases forbidden')
        if campaign_id!=FIXED_PLAN['campaign_id'] or claim.name!=campaign_id+'.claim.json':raise ValueError('wrong campaign')
        if self.output.exists():raise ValueError('output already exists')
        atomic_json(claim,dict(schema='grouped-exclusive-claim-v1',campaign_id=campaign_id,output=str(self.output),max_optimizer_calls=1200,max_model_forward_calls=1218,**DENIALS))
        self.output.mkdir(exist_ok=False)
        self.count=0;self.forward_count=0;self.arm_counts={'A':0,'B':0};self.pending=None;self.step_pending=None;self.returned=None
        atomic_json(self.output/'intent.json',dict(schema='grouped-intent-v1',campaign_id=campaign_id,max_optimizer_calls=1200,max_model_forward_calls=1218,no_resume=True,**DENIALS))
    def before_forward(self,kind,stage,update):
        event=(kind,stage,update)
        if self.pending or self.step_pending or (self.returned and self.returned[0]=='training'):raise ValueError('unfinished event')
        schedule=forward_schedule()
        if type(update) is not int or self.forward_count>=len(schedule) or event!=schedule[self.forward_count]:raise ValueError('forward sequence violation')
        atomic_json(self.output/f'forward-{self.forward_count+1:04d}.json',dict(status='forward_reserved',model_forward_reserved=self.forward_count+1,kind=kind,stage=stage,update=update))
        self.forward_count+=1;self.pending=event;self.returned=None
    def after_forward(self,kind,stage,update):
        event=(kind,stage,update)
        if self.pending!=event:raise ValueError('forward return mismatch')
        atomic_json(self.output/f'forward-return-{self.forward_count:04d}.json',dict(status='forward_returned',model_forward_reserved=self.forward_count,kind=kind,stage=stage,update=update))
        self.pending=None;self.returned=event
    def before_step(self,stage,update):
        if self.pending or self.step_pending or self.returned!=('training',stage,update) or type(update) is not int or update!=self.arm_counts.get(stage,-1)+1 or update>600 or self.count>=1200:raise ValueError('optimizer sequence violation')
        atomic_json(self.output/f'call-{self.count+1:04d}.json',dict(status='step_reserved',stage=stage,update=update,optimizer_call_reserved=self.count+1))
        self.count+=1;self.arm_counts[stage]+=1;self.step_pending=(stage,update)
    def after_step(self,stage,update,evidence):
        if self.step_pending!=(stage,update):raise ValueError('optimizer return mismatch')
        atomic_json(self.output/f'step-{self.count:04d}.json',evidence)
        self.step_pending=None;self.returned=None
    def complete(self):
        if self.pending or self.step_pending or self.count!=1200 or self.forward_count!=1218 or self.arm_counts!={'A':600,'B':600}:raise ValueError('incomplete budget')
