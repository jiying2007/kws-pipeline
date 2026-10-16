"""LR-stability fixed contract. Reuses frozen grouped I/O utilities, never its budget."""
from pathlib import Path
import hashlib
import importlib.util
import os
import stat
HERE=Path(__file__).resolve().parent
SHARED=HERE.parent/'kws-grouped-balance-v1'
def pinned_module(name,path,digest):
    path=Path(path).absolute()
    if path.resolve()!=path or '..' in path.parts:raise ValueError('shared source aliases forbidden')
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());raw=stream.read();after=os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns) or hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('shared source pin changed')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module
_shared=pinned_module('frozen_grouped_contract',SHARED/'contract.py','6dca1cad9dba10f50e439f775b6d04d90f246544cbd5a634e30490589490bc74')
DENIALS,atomic_json,collapse,sha,weights,strict_equal,strict_json=(_shared.DENIALS,_shared.atomic_json,_shared.collapse,_shared.sha,_shared.weights,_shared.strict_equal,_shared.strict_json)
FIXED_PLAN = {'schema': 'kws-lr-stability-plan-v1', 'purpose': 'posthoc-proposed-fixed-lr-stability-diagnostic', 'production_qualified': False, 'seed': 20261015, 'model': {'feature_dim': 32, 'hidden_dim': 48, 'vocab_size': 5}, 'targets_by_text': {'你好小窝': [1, 2, 3, 4], '小窝小窝': [3, 4, 3, 4]}, 'optimizer': {'name': 'Adam', 'lr': 0.003, 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'eps': 1e-08, 'gradient_clip': 5.0}, 'ctc': {'blank': 0, 'reduction': 'none', 'zero_infinity': False, 'normalization': 'per-sample-input-frame-count'}, 'max_optimizer_calls': 600, 'retries': 0, 'new_tts_calls': 0, 'new_asr_calls': 0, 'warm_start': False, 'augmentation': False, 'vad': False, 'export_weights': False, 'registry_sha256': '28033edbbc1e444eae27c33af2e060d1654c4b7a072e65401b055fc27fc3b1b0', 'source_verifier_sha256': '9d37fb852cdd5048a4d5c88f9e21efe2b3c1e18f25244babb2d811675ea9fbae', 'runtime_lock_sha256': 'b132c2c202e5ff00b6cbd1b2b630fa92dfe9ba46fcf0423dc0633381bbe80984', 'code_sha256': {'training/model.py': '768800e25537367da770ef44178249208956d7c3c33f1eb7d60b1b9c98fe59c5', 'training/frontend.py': 'd8ae687303d5c9c6bfdf6539c94f4ae463d910a4837ed77633d7656ff02a582e', 'training/frontend_spec.py': 'fbd17236e1449101bed6df8f105baad476edb0b005e6e45cb83ebc7971d166e6'}, 'campaign_id': 'kws-lr-stability-20261016-v1', 'campaign_claim_path': 'research-budget/kws-lr-stability-20261016-v1.claim.json', 'input_pins_sha256': '64c77a270c7f13d8252d10282191aaeb207f425c10fb89a2f01e390fa994e37c', 'frontend': {'frontend': 'logmel', 'feature_dim': 32, 'frame_len': 400, 'hop': 320}, 'stages': [{'name': 'A', 'updates': 600, 'checkpoints': [0, 50, 100, 200, 300, 400, 500, 600], 'blank_lambda': 1}], 'observation_timing': 'post-step; step 0 before any optimizer call', 'loss_timing': 'pre-step', 'max_model_forward_calls': 608, 'cohort_sha256': 'fe23e633925573e2b02ac8530e066214879a11cbccc2e4ef5875aa2c38006e87', 'prefix_rule': {'window_samples': 400, 'hop_samples': 320, 'threshold': 0.0001, 'rule': 'maximal initial run of complete unwindowed PCM RMS strictly below threshold; zero allowed'}, 'train_weights': {'A': {'KW1': 0.08333333333333333, 'KW2': 0.08333333333333333}}, 'early_stop': False, 'extension': False, 'resume': False, 'probe_selection': False, 'train_record_ids': ['kw1-preset-supplement-20261014-v1:kw1s-05', 'kw1-preset-supplement-20261014-v1:kw1s-06', 'kw1-preset-supplement-20261014-v1:kw1s-07', 'tts-quality-remaining-20261014-v1:tqp-qwen06-05', 'tts-quality-remaining-20261014-v1:tqp-qwen06-06', 'tts-quality-remaining-20261014-v1:tqp-qwen06-10', 'tts-quality-remaining-20261014-v1:tqp-qwen06-11', 'tts-quality-remaining-20261014-v1:tqp-qwen06-14', 'tts-quality-remaining-20261014-v1:tqp-qwen06-15', 'tts-quality-remaining-20261014-v1:tqp-qwen17-06', 'tts-quality-remaining-20261014-v1:tqp-qwen17-14', 'tts-quality-remaining-20261014-v1:tqp-qwen17-15'], 'probe_record_ids': [], 'max_probe_forwards': 0, 'historical_input_pins_sha256': '849d7b768b61d7c8238ffea3f5bd69328b6cfba70d7eb25088338b20b33929cd', 'historical_manifest_sha256': '4260c936d8249c59f2d5b270d40274b5f2b29bb45d930dd31c338b240d666b25', 'initial_weight_sha256': '0eafd98dd20f8c799d1c4d733a47314018ecfdd942c2769fed49cf8ae4fc7b2d', 'historical_replay_through_step': 500, 'learning_rate_schedule': [{'first_update': 1, 'last_update': 500, 'lr': 0.003}, {'first_update': 501, 'last_update': 600, 'lr': 0.0003}], 'shared_source_sha256': {'contract.py': '6dca1cad9dba10f50e439f775b6d04d90f246544cbd5a634e30490589490bc74', 'run_grouped_balance.py': 'f57e0fae61b314769cb1d3f3365b77f9ccb9400071be425b3ce7d04e66d53b6e', 'cohort-prefix-freeze.json': 'fe23e633925573e2b02ac8530e066214879a11cbccc2e4ef5875aa2c38006e87', 'runtime-candidate.lock.json': 'b132c2c202e5ff00b6cbd1b2b630fa92dfe9ba46fcf0423dc0633381bbe80984', 'requirements.research.lock': '7aa51b6eebd2e2db290338345c30b8c7a8e197531746ee77efb7ef8678124dee'}, 'shared_module': 'research/kws-grouped-balance-v1'}
CODE_FILES = {"run_lr_stability.py", "contract.py", "numeric_preflight.py"}
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

def forward_schedule():
    events=[]
    for stage in ('A',):
        events.append(('observation',stage,0))
        for update in range(1,601):
            events.append(('training',stage,update))
            if update in FIXED_PLAN['stages'][0]['checkpoints']:
                events.append(('observation',stage,update))
    return events

class Budget:
    """Durable exclusive reservations; a pending event cannot be skipped or reused."""
    def __init__(self,output,campaign_claim,campaign_id):
        self.output=Path(output).absolute(); claim=Path(campaign_claim).absolute()
        for p in (self.output,claim):
            if p.resolve()!=p or '..' in p.parts:raise ValueError('aliases forbidden')
        if campaign_id!=FIXED_PLAN['campaign_id'] or claim.name!=campaign_id+'.claim.json':raise ValueError('wrong campaign')
        if self.output.exists():raise ValueError('output already exists')
        atomic_json(claim,dict(schema='lr-stability-exclusive-claim-v1',campaign_id=campaign_id,output=str(self.output),max_optimizer_calls=600,max_model_forward_calls=608,**DENIALS))
        self.output.mkdir(exist_ok=False)
        self.count=0;self.forward_count=0;self.arm_counts={'A':0};self.pending=None;self.step_pending=None;self.returned=None
        atomic_json(self.output/'intent.json',dict(schema='lr-stability-intent-v1',campaign_id=campaign_id,max_optimizer_calls=600,max_model_forward_calls=608,no_resume=True,**DENIALS))
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
        if self.pending or self.step_pending or self.returned!=('training',stage,update) or type(update) is not int or update!=self.arm_counts.get(stage,-1)+1 or update>600 or self.count>=600:raise ValueError('optimizer sequence violation')
        atomic_json(self.output/f'call-{self.count+1:04d}.json',dict(status='step_reserved',stage=stage,update=update,optimizer_call_reserved=self.count+1))
        self.count+=1;self.arm_counts[stage]+=1;self.step_pending=(stage,update)
    def after_step(self,stage,update,evidence):
        if self.step_pending!=(stage,update):raise ValueError('optimizer return mismatch')
        atomic_json(self.output/f'step-{self.count:04d}.json',evidence)
        self.step_pending=None;self.returned=None
    def complete(self):
        if self.pending or self.step_pending or self.count!=600 or self.forward_count!=608 or self.arm_counts!={'A':600}:raise ValueError('incomplete budget')

REPLAY_FIELDS=('kind','stage','update','role','timing','samples','ctc_loss','anchor_loss','finite_loss','blank_lambda','per_class_complete_rate','macro_complete_rate')
def replay_projection(payload):
    if payload.get('schema') not in ('grouped-forward-payload-v1','lr-stability-forward-payload-v1'):raise ValueError('unexpected replay payload schema')
    return {key:payload[key] for key in REPLAY_FIELDS}
def verify_replay(observed,historical):
    if not strict_equal(replay_projection(observed),replay_projection(historical)):raise ValueError('historical saved numerical trajectory mismatch')
def learning_rate(update):
    if type(update) is not int or not 1<=update<=600:raise ValueError('invalid schedule update')
    return 0.003 if update<=500 else 0.0003

def prepare_update(optimizer,update,replay_observations):
    """Pure schedule control; never recreates optimizer or reads/writes moment state."""
    expected=learning_rate(update)
    if len(optimizer.param_groups)!=1:raise ValueError('original single Adam group required')
    group=optimizer.param_groups[0]
    if update==501:
        if replay_observations!=[0,50,100,200,300,400,500]:raise ValueError('500 observation replay gate incomplete')
        if group['lr']!=0.003:raise ValueError('premature learning rate change')
        group['lr']=expected
    if group['lr']!=expected:raise ValueError('learning rate schedule mismatch')
    return update==501
