"""Original-A20 one-forward preparation control. No optimizer/training code."""
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from contracts import CMVN, TOKENS, CHECKPOINT_SHA, STATE_SHA, require, sha, cohort_weights, minimum_frames
ROOT = Path(__file__).resolve().parents[1]

def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def configure_cpu():
    import os
    import random
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(610104); np.random.seed(610104); torch.manual_seed(610104)

def load_a20(workspace, bindings):
    """Deferred strict load from exact A-terminal checkpoint, CPU only."""
    import torch
    import numpy as np
    require(sys.byteorder == 'little', 'canonical tensor byte order')
    require(torch.__version__ == bindings['torch_version'] == '2.12.1+cpu', 'approved CPU Torch build')
    require(torch.version.cuda is None, 'CUDA build excluded')
    require(np.__version__ == bindings['numpy_version'] == '1.26.4', 'approved NumPy version')
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'one CPU thread')
    require(torch.are_deterministic_algorithms_enabled(), 'determinism must be enabled')
    workspace = Path(workspace)
    for relative, digest in bindings['source_sha256'].items():
        require(sha(workspace / relative) == digest, 'source drift: ' + relative)
    require(sha(workspace / bindings['fsmn_path']) == '62265f349ae1dd95580e614557ee58c0ee361fea9e41ac62c6634d5c6ff6c2d3', 'pinned FSMN implementation')
    require(sha(workspace / bindings['cmvn_path']) == '6b25a77adb3b33361798f0d72110915827015fada0ddb0cc22e30e9d70afe05c', 'pinned CMVN implementation')
    require(bindings['historical_training_protocol_sha256'] == 'da6545afa90989d1cab008a0fdfd30cabbfbb612cdf3297b91c132bfe0df036f', 'historical protocol identity')
    source = workspace / bindings['checkpoint_path']
    require(source.stat().st_size == 4732286 and sha(source) == CHECKPOINT_SHA, 'exact A20 checkpoint')
    obj = torch.load(source, map_location='cpu', weights_only=True)
    require(type(obj) is dict and set(obj) == {'model', 'optimizer', 'step', 'torch_rng', 'arm', 'symbols', 'source_protocol_sha256'}, 'checkpoint top-level schema')
    require(type(obj['step']) is int and obj['step'] == 1200 and obj['arm'] == 'A', 'historical endpoint')
    require(obj['symbols'] == list(TOKENS), 'six-class order')
    require(obj['source_protocol_sha256'] == bindings['historical_training_protocol_sha256'], 'historical protocol')
    # Optimizer and RNG payloads are intentionally ignored, never restored.
    state = obj['model']
    schema = json.loads((ROOT / 'metadata/A20-TENSOR-SCHEMA.json').read_text())
    require(sha(ROOT / 'metadata/A20-TENSOR-SCHEMA.json') == bindings['a20_tensor_schema_sha256'] == 'fa10e0a11750e2b3e215c44f340e80bfdec1b374c68cd03eacd23b07b119dcea', 'tensor schema binding')
    require(type(state) in (dict, collections.OrderedDict), 'exact tensor mapping')
    expected = schema['tensors']
    require(list(state) == [t['name'] for t in expected], 'ordered exact key set')
    payload, canonical = hashlib.sha256(), hashlib.sha256()
    for entry in expected:
        key = entry['name']; tensor = state[key]
        require(type(tensor) is torch.Tensor and tensor.dtype == torch.float32 and
                tensor.device.type == 'cpu' and list(tensor.shape) == entry['shape'] and
                torch.isfinite(tensor).all().item(), 'tensor shape/type/finite: ' + key)
        raw = tensor.detach().contiguous().numpy().tobytes(order='C')
        require(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'], 'tensor content: ' + key)
        payload.update(raw); canonical.update(key.encode()); canonical.update(raw)
    require(canonical.hexdigest() == STATE_SHA == schema['state_sha256'], 'canonical state')
    require(payload.hexdigest() == schema['payload_sha256'], 'native tensor payload')
    fsmn = _module('a20_pinned_fsmn', workspace / bindings['fsmn_path'])
    cmvn = _module('a20_pinned_cmvn', workspace / bindings['cmvn_path'])

    class Candidate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.global_cmvn = cmvn.GlobalCMVN(state[CMVN[0]].clone(), state[CMVN[1]].clone(), True)
            self.backbone = fsmn.FSMN(400, 140, 4, 250, 128, 10, 2, 1, 1, 140, 6)
            self.load_state_dict(state, strict=True)

        def forward(self, x, cache=None):
            return self.backbone(self.global_cmvn(x), cache)

    model = Candidate().cpu().float()
    require(sum(p.numel() for p in model.parameters()) == 390520, 'parameter count')
    require(all(p.requires_grad for p in model.parameters()), 'full encoder/head trainability')
    require(list(dict(model.named_buffers())) == list(CMVN) and
            sum(p.numel() for p in model.buffers()) == 800, 'fixed CMVN buffers only')
    require(list(model.state_dict()) == list(state), 'constructed key order')
    require(all(torch.equal(model.state_dict()[k], v) for k, v in state.items()), 'strict loaded state')
    return model

def tensor_ctc_loss(logits, lengths, targets, cohorts):
    import torch
    weights = cohort_weights(cohorts)
    require(tuple(logits.shape) == (32, 95, 6) and logits.dtype == torch.float32 and
            logits.device.type == 'cpu' and torch.isfinite(logits).all().item(), 'logit contract')
    require(len(lengths) == len(targets) == 32, 'source count')
    require(all(type(n) is int and minimum_frames(y) <= n <= 95 and len(y) in (3, 4)
                for n, y in zip(lengths, targets)), 'CTC lengths/feasibility')
    flat = torch.tensor([t for y in targets for t in y], dtype=torch.long)
    tl = torch.tensor([len(y) for y in targets], dtype=torch.long)
    il = torch.tensor(lengths, dtype=torch.long)
    losses = torch.nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0, 1), flat, il, tl,
                                         blank=0, reduction='none', zero_infinity=False)
    require(torch.isfinite(losses).all().item(), 'nonfinite CTC')
    per_source = losses / tl
    return (per_source * torch.tensor(weights, dtype=per_source.dtype)).sum(), per_source

def initialization_control(model, x, rows):
    """Proposed one-forward, zero-update A20 control on the native32 batch.

    No optimizer construction, backward, development data or decoder execution.
    Save returned logits and CTC values without selecting/tuning a candidate.
    """
    import torch
    require(tuple(x.shape) == (32, 95, 400) and x.dtype == torch.float32 and
            x.device.type == 'cpu', 'control input')
    require(len(rows) == 32 and all(r['role'] == 'train' for r in rows), 'control train32')
    def state_digest():
        h = hashlib.sha256()
        for key, value in model.state_dict().items():
            h.update(key.encode()); h.update(value.detach().contiguous().numpy().tobytes())
        return h.hexdigest()
    require(state_digest() == STATE_SHA, 'control original A20 state')
    cmvn_calls = []
    def cmvn_hook(module, args, output):
        require(len(args) == 1 and args[0] is x, 'one PRE-CMVN input to original normalization')
        require(tuple(output.shape) == tuple(x.shape) and torch.isfinite(output).all().item(), 'CMVN output')
        cmvn_calls.append(1)
    hook = model.global_cmvn.register_forward_hook(cmvn_hook)
    # Preserve historical model training mode; dropout modules exist but their
    # calls are commented out in the pinned forward source. Reject any use.
    model.train()
    dropout_calls = []
    def dropout_hook(module, args):
        dropout_calls.append(1)
        raise RuntimeError('unexpected active dropout in unchanged FSMN')
    dropout_hooks = [m.register_forward_pre_hook(dropout_hook) for m in model.modules()
                     if isinstance(m, torch.nn.modules.dropout._DropoutNd)]
    try:
        with torch.no_grad():
            logits, _ = model(x)  # Exactly ONE full-model forward.
            loss, per_source = tensor_ctc_loss(logits, [r['model_rows'] for r in rows],
                [r['target_ids'] for r in rows], [r['cohort'] for r in rows])
    finally:
        hook.remove()
        for handle in dropout_hooks: handle.remove()
    require(not dropout_calls, 'no active dropout')
    require(cmvn_calls == [1] and state_digest() == STATE_SHA, 'CMVN once and original state unchanged')
    return dict(logits=logits.detach().clone(), lengths=[r['model_rows'] for r in rows],
                source_normalized_ctc=per_source.tolist(), cohort_weighted_ctc=float(loss),
                model_forwards=1, optimizer_constructions=0, backwards=0, optimizer_steps=0,
                cmvn_calls=1, dropout_calls=0, model_training_mode=True, state_sha256=STATE_SHA)
