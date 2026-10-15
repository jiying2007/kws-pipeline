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

class Budget:
    def __init__(self, output, campaign_claim, campaign_id):
        self.output = Path(output).absolute()
        if self.output.resolve()!=self.output or '..' in self.output.parts:
            raise ValueError('output aliases/symlinks are forbidden')
        campaign_claim=Path(campaign_claim).absolute()
        if campaign_claim.resolve()!=campaign_claim or '..' in campaign_claim.parts:
            raise ValueError('campaign claim aliases forbidden')
        atomic_json(campaign_claim, {'schema':'research-fit-campaign-claim-v1','campaign_id':campaign_id,'output':str(self.output),'max_optimizer_calls':1500,**DENIALS})
        self.output.mkdir(exist_ok=False)
        self.count = 0
        atomic_json(self.output/'claim.json', {'schema':'research-fit-single-attempt-claim-v1', 'max_optimizer_calls':1500, **DENIALS})

    def before_step(self, stage, update):
        if self.count >= 1500:
            raise ValueError('budget exhausted')
        self.count += 1
        # Reserve before optimizer.step; a crash consumes this slot. Never resume/retry.
        atomic_json(self.output/f'call-{self.count:04d}.json', {'stage':stage, 'update':update, 'optimizer_call_reserved':self.count})
