"""One fixed small causal head on cached frozen embeddings, no model sweep.

Prepared library functions only: no CLI execution, source acquisition, admission
or background execution. The calling worker must authenticate and bind its data.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from weak_embedding_contract import EMBED_WIDTH, HEAD_HISTORY, FALSE_FLAGS, need

SPEC = {'history': 16, 'embedding_width': 96, 'hidden': 32, 'classes': 3,
        'seed': 20261116, 'optimizer_updates': 96, 'learning_rate': 0.001,
        'weight_decay': 0.0001, 'cpu_threads': 1,
        'head_parameters': 49283, 'normalization': 'train-windows-only',
        'positive_loss': 'log-mean-probability-on-verified-mel-support-weak-clip-windows',
        'background_loss': 'mean-window-cross-entropy',
        'checkpoint_selection': False}


def head_spec():
    return dict(SPEC)


def train_cached(bags, output_directory, input_binding):
    """Actual fitting entrypoint for a future explicit public Actions worker.

Each bag: id, role, weak_class, windows[N,16,96]. A positive bag may include
only windows selected by mel_supported_clip_ends after ONNX metadata preflight.
This geometric weak target is not pronunciation or word-end truth. Background windows must
contain only train-role or development-role source audio, as appropriate.
"""
    import numpy as np
    import torch
    need(type(input_binding) is dict and input_binding and
         all(type(v) is str and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
             for v in input_binding.values()), 'EXACT_INPUT_BINDING_REQUIRED')
    output = Path(output_directory)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'CREATE_ONLY_TRAIN_OUTPUT')
    need(bags and len({b['id'] for b in bags}) == len(bags), 'UNIQUE_BAGS')
    for bag in bags:
        x = bag['windows']
        need(bag['role'] in ('train', 'development') and type(bag['weak_class']) is int and
             bag['weak_class'] in (0, 1, 2) and type(x) is np.ndarray and x.dtype == np.float32
             and type(bag.get('phone_review_status')) is str and bool(bag['phone_review_status'])
             and x.ndim == 3 and x.shape[1:] == (HEAD_HISTORY, EMBED_WIDTH)
             and 1 <= len(x) <= 1024 and bool(np.isfinite(x).all()), 'FEATURE_BAG')
    need({(b['role'], b['weak_class']) for b in bags} ==
         {(role, label) for role in ('train', 'development') for label in range(3)},
         'SIX_ROLE_INTENT_GROUPS_AFTER_GEOMETRY_REQUIRED')
    train = [b for b in bags if b['role'] == 'train']
    groups = {k: [b for b in train if b['weak_class'] == k] for k in range(3)}
    need(all(groups.values()), 'TRAIN_THREE_WEAK_CLASSES')
    # Statistics use only training windows, never development or future data.
    count = sum(b['windows'].shape[0] * HEAD_HISTORY for b in train)
    total = sum((b['windows'].astype(np.float64).sum(axis=(0, 1)) for b in train))
    square = sum(((b['windows'].astype(np.float64) ** 2).sum(axis=(0, 1)) for b in train))
    mean = (total / count).astype(np.float32)
    scale = np.maximum(np.sqrt(np.maximum(square / count - (total / count) ** 2, 0)), 1e-5).astype(np.float32)
    torch.set_num_threads(1)
    torch.manual_seed(SPEC['seed'])
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Sequential(torch.nn.Linear(HEAD_HISTORY * EMBED_WIDTH, SPEC['hidden']),
                                torch.nn.ReLU(), torch.nn.Linear(SPEC['hidden'], 3))
    need(sum(p.numel() for p in model.parameters()) == SPEC['head_parameters'], 'HEAD_PARAMETER_COUNT')
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC['learning_rate'],
                                  weight_decay=SPEC['weight_decay'])
    prepared = {b['id']: torch.from_numpy(((b['windows'] - mean) / scale).reshape(len(b['windows']), -1))
                for b in bags}
    output.mkdir(mode=0o700)
    losses, draws = [], []
    for step in range(SPEC['optimizer_updates']):
        optimizer.zero_grad(set_to_none=True)
        terms, used = [], []
        for label in range(3):
            bag = groups[label][step % len(groups[label])]
            logp = torch.log_softmax(model(prepared[bag['id']]), dim=-1)
            terms.append(-logp[:, 0].mean() if label == 0 else
                         math.log(len(logp)) - torch.logsumexp(logp[:, label], dim=0))
            used.append(bag['id'])
        loss = torch.stack(terms).mean()
        need(bool(torch.isfinite(loss)), 'NONFINITE_TRAIN_LOSS')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        draws.append(used)
    weights = {'w1': model[0].weight.detach().numpy(), 'b1': model[0].bias.detach().numpy(),
               'w2': model[2].weight.detach().numpy(), 'b2': model[2].bias.detach().numpy(),
               'mean96': mean, 'scale96': scale}
    weight_path = output / 'causal-head.npz'
    np.savez(weight_path, **weights)
    # Independent saved-byte readback; no live model tensors enter NumPy replay.
    with np.load(weight_path, allow_pickle=False) as saved:
        restored = {k: saved[k].copy() for k in weights}
    replay = []
    model.eval()
    with torch.inference_mode():
        for bag in bags:
            z = ((bag['windows'] - restored['mean96']) / restored['scale96']).reshape(len(bag['windows']), -1)
            logits = np.maximum(z @ restored['w1'].T + restored['b1'], 0) @ restored['w2'].T + restored['b2']
            expected = model(prepared[bag['id']]).numpy()
            error = float(np.max(np.abs(logits - expected)))
            need(np.isfinite(logits).all() and error <= 1e-4, 'SAVED_HEAD_REPLAY')
            replay.append({'id': bag['id'], 'role': bag['role'], 'weak_class': bag['weak_class'],
                           'phone_review_status': bag['phone_review_status'],
                           'source_observation': bag.get('source_observation'),
                           'research_selection_sha256': bag.get('research_selection_sha256'),
                           'window_count': len(logits), 'max_abs_logit_error': error,
                           'argmax_classes': logits.argmax(axis=-1).tolist()})
    report = {'schema': 'english-final-frozen-embedding-weak-head-v1', 'status': 'EXECUTED_WEAK_RESEARCH_ONLY',
              'spec': SPEC, 'input_binding': input_binding, 'optimizer_updates': len(losses),
              'optimizer_bag_draws': draws, 'losses': losses, 'saved_replay': replay,
              'weights_sha256': hashlib.sha256(weight_path.read_bytes()).hexdigest(),
              'phone_review_status': 'PRESERVED_PER_BAG_NO_PRONUNCIATION_PASS',
              'independent_frr': 'NOT_RUN', 'continuous_natural_far': 'NOT_RUN',
              'word_end_latency': 'NOT_RUN', 'ssc305': 'NOT_RUN',
              **dict.fromkeys(FALSE_FLAGS, False)}
    with (output / 'RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return report
