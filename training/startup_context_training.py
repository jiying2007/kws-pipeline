#!/usr/bin/env python3
"""Development-only speech/context learnability, reusing the shipping RNN and exporter.

No canonical trainer defaults change. Positive-only scope is a debugging control,
not a detector candidate. Every run records the actual data and recipe; no model
from this module has release authority. CTC uses target-length normalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import shutil
import subprocess
import sys
import time
import wave

# Import model before torch, preserving the repository's CPU numerical topology.
from model import TinyStreamingRNN
import torch
from frontend import features
from frozen_speech_ablation import verify_pool, resolve, sha, write, execute
from synthetic_audio import activity_bounds
from train_ctc import Manifest, training_environment, load_tokens, vocab_fingerprint
from training_state import state_identity
from sequence_margin import keyword_sequence_margin_loss
from objective_contract import SEQUENCE_MARGIN_NEGATIVE_POLICIES
from wake_pressure_balance import keyword_target_sequences

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = 'startup-context-ctc-development-v1'
VARIANTS = ('plain-ctc', 'context-ctc', 'span-ctc', 'blank-ctc', 'grounded-ctc',
            'grounded-negative-margin', 'grounded-sparse-margin', 'grounded-speed')
GROUNDED_VARIANTS = ('grounded-ctc', 'grounded-negative-margin', 'grounded-sparse-margin',
                     'grounded-speed')
MARGIN_VARIANTS = ('grounded-negative-margin', 'grounded-sparse-margin')
PREFIX_HOPS = (0, 10, 25, 50)
TAIL_SAMPLES = 8000
SPEED_FACTORS = (0.85, 1.0, 1.15)


def read_pcm(path: pathlib.Path) -> torch.Tensor:
    with wave.open(str(path), 'rb') as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('context data requires mono PCM16 16kHz')
        raw = f.readframes(f.getnframes())
    return torch.frombuffer(bytearray(raw), dtype=torch.int16).clone()


def respeed_pcm(samples: torch.Tensor, factor: float) -> torch.Tensor:
    """Deterministic PCM speed/pitch perturbation; transcript is unchanged."""
    if samples.dtype != torch.int16 or samples.ndim != 1 or samples.numel() < 400:
        raise ValueError('speed perturbation requires nonempty PCM16 waveform')
    if type(factor) is not float or factor not in SPEED_FACTORS:
        raise ValueError('unsupported speed factor')
    if factor == 1.0:
        return samples.clone()
    target = round(samples.numel() / factor)
    shifted = torch.nn.functional.interpolate(
        samples.float().reshape(1, 1, -1), size=target,
        mode='linear', align_corners=False,
    ).reshape(-1)
    return shifted.round().clamp(-32768, 32767).to(torch.int16)


def active_span(samples: torch.Tensor, frames: int) -> tuple[int, int]:
    """Activity envelope, not forced/phoneme alignment; includes boundary frames.

    The same waveform-only activity detector is used for wake/nonwake rows.
    Blank supervision excludes an additional 40-ms boundary guard band.
    """
    start, end = activity_bounds(samples.tolist())
    lo = max(0, (start - 400) // 320 + 1)
    hi = min(frames, (end + 319) // 320)
    if hi <= lo:
        raise ValueError('no active frames')
    return lo, hi


def select_rows(pool: dict, scope: str) -> list[dict]:
    if scope not in ('positive-only', 'all'):
        raise ValueError('invalid training scope')
    rows = [r for r in pool['rows'] if r['split'] == 'train' and (scope == 'all' or r['expected'])]
    if not rows or len(rows) > 128:
        raise ValueError('bounded train pool must have 1..128 rows')
    if {e['keyword_id'] for r in rows for e in r['expected']} != {1, 2}:
        raise ValueError('both fixed wake targets require training support')
    return rows


def loss_components(log_probs: torch.Tensor, targets: list[torch.Tensor],
                    lengths: list[int], spans: list[tuple[int, int]], variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    if variant not in VARIANTS or not (len(targets) == len(lengths) == len(spans) == log_probs.shape[1]):
        raise ValueError('invalid context loss input')
    parts, ilen = [], []
    for j, (n, (a, b), y) in enumerate(zip(lengths, spans, targets)):
        if type(n) is not int or not 0 <= a < b <= n <= log_probs.shape[0]:
            raise ValueError('invalid frame span')
        if y.numel() == 0 or any(t <= 0 or t >= log_probs.shape[-1] for t in y.tolist()):
            raise ValueError('transcribed rows require nonblank targets')
        start, stop = (a, b) if variant == 'span-ctc' or variant in GROUNDED_VARIANTS else (0, n)
        minimum = len(y) + sum(int(y[k] == y[k - 1]) for k in range(1, len(y)))
        if stop - start < minimum:
            raise ValueError('infeasible CTC alignment')
        parts.append(log_probs[start:stop, j]); ilen.append(stop - start)
    scores = torch.nn.utils.rnn.pad_sequence(parts)
    target_lengths = torch.tensor([len(y) for y in targets])
    raw = torch.nn.functional.ctc_loss(scores, torch.cat(targets), torch.tensor(ilen),
            target_lengths, blank=0, reduction='none', zero_infinity=False)
    if not torch.isfinite(raw).all():
        raise ValueError('non-finite CTC loss')
    ctc = (raw / target_lengths).mean()
    blank = log_probs.sum() * 0.0
    if variant == 'blank-ctc' or variant in GROUNDED_VARIANTS:
        losses = []
        for j, (n, (a, b)) in enumerate(zip(lengths, spans)):
            t = torch.arange(n)
            mask = (t < max(0, a - 2)) | (t >= min(n, b + 2))
            if mask.any():
                losses.append(-log_probs[:n, j, 0][mask].mean())
        if not losses:
            raise ValueError('grounded trial requires non-speech support')
        blank = torch.stack(losses).mean()
    return ctc, blank


def negative_runtime_margin(log_probs: torch.Tensor, targets: list[torch.Tensor],
                            lengths: list[int], keyword_sequences: list[list[int]],
                            *, negative_path_policy: str = 'runtime-executable-v1') -> torch.Tensor:
    """Penalize wake paths on transcribed nonwake rows only."""
    if len(targets) != len(lengths) or len(targets) != log_probs.shape[1]:
        raise ValueError('negative margin batch shape mismatch')
    if negative_path_policy not in SEQUENCE_MARGIN_NEGATIVE_POLICIES:
        raise ValueError('negative margin path policy is invalid')
    wake = {tuple(row) for row in keyword_sequences}
    selected = [i for i, y in enumerate(targets) if tuple(y.tolist()) not in wake]
    if not selected:
        return log_probs.sum() * 0.0
    ys = [targets[i] for i in selected]
    values = keyword_sequence_margin_loss(
        log_probs=log_probs[:, selected, :], targets=torch.cat(ys),
        input_lengths=torch.tensor([lengths[i] for i in selected]),
        target_lengths=torch.tensor([len(y) for y in ys]),
        true_ctc_nll=log_probs.new_zeros(len(selected)),
        keyword_sequences=keyword_sequences,
        negative_path_policy=negative_path_policy,
    )
    if not torch.isfinite(values).all():
        raise ValueError('non-finite negative runtime margin')
    return values.mean()


def validate_recipe(recipe: dict) -> None:
    if (not isinstance(recipe, dict) or recipe.get('policy') != POLICY
            or recipe.get('development_only') is not True or recipe.get('release_authority') is not False
            or recipe.get('variant') not in VARIANTS or recipe.get('scope') not in ('all','positive-only')):
        raise ValueError('invalid development recipe authority')


def bind_development_provenance(checkpoint: pathlib.Path, model: pathlib.Path) -> None:
    """Add explicit experimental semantics to the unchanged canonical exporter.

    Weight bytes are never rewritten. Authenticate the checkpoint/model linkage
    before adding the recipe and the experimental loss column from checkpoint.
    """
    cp=torch.load(checkpoint,weights_only=True,map_location='cpu')
    validate_recipe(cp['development_recipe'])
    provenance_path=model.with_suffix('.kwm.provenance.json')
    provenance=json.loads(provenance_path.read_text())
    identity=state_identity(cp['state_dict'])
    if (cp['float_state_identity'] != identity
            or provenance['training']['float_state_identity'] != identity
            or provenance['checkpoint']['sha256'] != sha(checkpoint)
            or provenance['model']['sha256'] != sha(model)):
        raise ValueError('experimental export readback mismatch')
    provenance['training']['development_recipe']=cp['development_recipe']
    for target,source in zip(provenance['training']['epoch_history'],cp['epoch_history']):
        for column in ('non_speech_blank', 'negative_runtime_margin'):
            value=source[column]
            if not math.isfinite(value) or value<0:
                raise ValueError('invalid experimental loss')
            target[column]=value
    write(provenance_path,provenance)


def train(pool_root: pathlib.Path, pool_sha: str, output: pathlib.Path,
          variant: str, scope: str, epochs: int, seed: int, *, retain_milestones: bool = False,
          initial_checkpoint: pathlib.Path | None = None) -> None:
    if type(retain_milestones) is not bool:
        raise ValueError('retain_milestones must be boolean')
    if variant not in VARIANTS or type(epochs) is not int or not 1 <= epochs <= 800:
        raise ValueError('invalid bounded recipe')
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError('invalid seed')
    pool = verify_pool(pool_root, pool_sha)
    rows = select_rows(pool, scope)
    output.mkdir(parents=True, exist_ok=False)
    tokens = pool_root / 'tokens.example.txt'
    keywords = keyword_target_sequences(tokens, pool_root / 'zh_cn_example.tsv')
    keyword_sequences = [list(keywords[k]) for k in sorted(keywords)]
    manifest = output / 'source-train.tsv'
    manifest.write_text(''.join(f"{resolve(pool_root, r['path'])}\t{' '.join(map(str, r['target_ids']))}\n" for r in rows))
    dataset = Manifest([manifest], 32, 5, 'logmel')
    # Actual PCM is padded then fed to the original frontend, not a feature-only surrogate.
    prefixes = (0,) if variant == 'plain-ctc' else PREFIX_HOPS
    speeds = SPEED_FACTORS if variant == 'grounded-speed' else (1.0,)
    cache = {}
    recipe = {'policy': POLICY, 'development_only': True, 'release_authority': False,
        'variant': variant, 'scope': scope, 'pool_sha256': pool_sha,
        'source_rows': [{'wav_sha256': r['wav_sha256'], 'target_ids': r['target_ids']} for r in rows],
        'prefix_hops': list(prefixes), 'tail_samples': 0 if variant == 'plain-ctc' else TAIL_SAMPLES,
        'speed_factors': list(speeds), 'speed_policy': 'pcm-linear-fixed-v1',
        'ctc_reduction': 'per-target-length-uniform-v1', 'activity_policy': 'waveform-2-percent-peak-min64-v1',
        'blank_boundary_guard_hops': 2,
        'blank_weight': 0.3 if variant == 'blank-ctc' or variant in GROUNDED_VARIANTS else 0.0,
        'negative_runtime_margin_weight': 0.1 if variant in MARGIN_VARIANTS else 0.0,
        'negative_path_policy': ('sparse-chronological-v1' if variant == 'grounded-sparse-margin'
                                 else 'runtime-executable-v1' if variant == 'grounded-negative-margin'
                                 else 'none'),
        'epochs': epochs, 'seed': seed, 'batch_size': 16, 'learning_rate': 0.001,
        'model': 'shipping-rnn-32x64-logmel', 'selection_uses_development_metrics': False,
        'source_tree': subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()}
    source_checkpoint = None
    if initial_checkpoint is not None:
        if scope != 'all' or variant not in GROUNDED_VARIANTS:
            raise ValueError('warm start is limited to paired grounded development trials')
        source_checkpoint = torch.load(initial_checkpoint, map_location='cpu', weights_only=True)
        source_recipe = source_checkpoint.get('development_recipe')
        validate_recipe(source_recipe)
        expected_rows = recipe['source_rows']
        if (source_recipe['variant'] != 'grounded-ctc' or source_recipe['scope'] != 'all'
                or source_recipe['pool_sha256'] != pool_sha or source_recipe['source_rows'] != expected_rows
                or source_checkpoint['float_state_identity'] != state_identity(source_checkpoint['state_dict'])
                or (source_checkpoint['feature_dim'], source_checkpoint['hidden_dim'], source_checkpoint['vocab_size']) != (32,64,5)
                or source_checkpoint['tokens_sha256'] != sha(tokens)):
            raise ValueError('warm-start checkpoint does not match frozen grounded source')
        recipe['warm_start'] = {'checkpoint_sha256': sha(initial_checkpoint),
            'float_state_identity': source_checkpoint['float_state_identity'],
            'source_variant': 'grounded-ctc', 'optimizer_state_restored': False}
    milestones = {'policy': 'context-milestone-retention-v1', 'completed': False,
        'release_authority': False, 'requested_epochs': epochs, 'seed': seed, 'pool_sha256': pool_sha,
        'expected_epochs': list(range(100, epochs + 1, 100)) + ([] if epochs % 100 == 0 else [epochs]),
        'records': []}
    if retain_milestones:
        recipe['checkpoint_retention'] = milestones['policy']
        write(output / 'milestones.json', milestones)
    for i, r in enumerate(rows):
        pcm = read_pcm(resolve(pool_root, r['path']))
        for speed in speeds:
            shifted = respeed_pcm(pcm, speed)
            for prefix in prefixes:
                tail = 0 if variant == 'plain-ctc' else TAIL_SAMPLES
                padded = torch.cat([torch.zeros(prefix * 320, dtype=torch.int16), shifted,
                                    torch.zeros(tail, dtype=torch.int16)])
                x = features(padded.float() / 32768, feature_dim=32)
                cache[i, prefix, speed] = (x, torch.tensor(r['target_ids'], dtype=torch.long),
                                           active_span(padded, len(x)))
    # Independent RNGs avoid changed context draws changing shuffle order between treatments.
    shuffle, context, speed_rng = (random.Random(seed), random.Random(seed + 1901),
                                   random.Random(seed + 2903))
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    model = TinyStreamingRNN(32, 64, 5)
    if source_checkpoint is not None:
        model.load_state_dict(source_checkpoint['state_dict'])
    initial = state_identity(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    environment = training_environment()
    environment['training_code_sha256']['training/startup_context_training.py'] = sha(pathlib.Path(__file__))
    recipe['implementation_sha256'] = dict(environment['training_code_sha256'])
    recipe['source_worktree_clean'] = not subprocess.check_output(
        ['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    write(output / 'recipe.json', recipe)
    batch_digest, context_digest, speed_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    history = []; started = time.monotonic()
    ids = list(range(len(rows)))
    for epoch in range(1, epochs + 1):
        shuffle.shuffle(ids); total = ctc_total = blank_total = margin_total = 0.0; batches = 0
        for offset in range(0, len(ids), 16):
            selected = [(i, context.choice(prefixes)) for i in ids[offset:offset + 16]]
            selected_speeds = [speed_rng.choice(speeds) if len(speeds) > 1 else 1.0
                               for _ in selected]
            batch_digest.update(json.dumps([epoch, [i for i, _ in selected]]).encode())
            context_digest.update(json.dumps([epoch, selected]).encode())
            speed_digest.update(json.dumps([epoch, selected_speeds]).encode())
            group = [cache[i, prefix, speed] for (i, prefix), speed in zip(selected, selected_speeds)]
            xs, ys, spans = zip(*group)
            logits = model(torch.nn.utils.rnn.pad_sequence(xs, batch_first=True))
            log_probs = logits.log_softmax(-1)
            ctc, blank = loss_components(log_probs, list(ys), [len(x) for x in xs], list(spans), variant)
            margin = (negative_runtime_margin(log_probs, list(ys), [len(x) for x in xs], keyword_sequences,
                                              negative_path_policy=recipe['negative_path_policy'])
                      if variant in MARGIN_VARIANTS else log_probs.sum() * 0.0)
            loss = ctc + recipe['blank_weight'] * blank + recipe['negative_runtime_margin_weight'] * margin
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach()); ctc_total += float(ctc.detach())
            blank_total += float(blank.detach()); margin_total += float(margin.detach()); batches += 1
        history.append({'epoch': epoch, 'loss': total / batches, 'ctc': ctc_total / batches,
            'non_speech_blank': blank_total / batches,
            'negative_runtime_margin': margin_total / batches,
            'ordered': 0.0, 'margin': 0.0,
            'completion': 0.0, 'release': 0.0, 'ordered_token_accuracy': 0.0})
        if epoch % 100 == 0 or epoch == epochs:
            zero_weights = {k: 0.0 for k in ('ordered_token_loss_weight','keyword_sequence_margin_loss_weight',
                                             'prefix_completion_loss_weight','recurrent_release_loss_weight')}
            checkpoint = {'state_dict': model.state_dict(), 'float_state_identity': state_identity(model.state_dict()),
                'initial_float_state_identity': initial, 'development_recipe': recipe,
                'batch_order_sha256': batch_digest.hexdigest(), 'context_order_sha256': context_digest.hexdigest(),
                'feature_dim': 32, 'hidden_dim': 64, 'vocab_size': 5,
                'vocab_fingerprint': vocab_fingerprint(load_tokens(tokens)), 'tokens_sha256': sha(tokens),
                'frame_length_samples': 400, 'frame_hop_samples': 320, 'frontend_spec_version': 2,
                'frontend_name': 'logmel', 'frontend_kind': 0, 'training_examples': len(rows),
                'training_manifests': [{'name': manifest.name, 'sha256': sha(manifest)}],
                'training_corpus_identity': dataset.corpus_identity, 'seed': seed, 'epochs': epoch,
                'epoch_history': history, 'batch_size': 16, 'learning_rate': 0.001, 'optimizer': 'AdamW',
                'weight_decay': 0.0001, 'grad_clip_norm': 5.0, 'training_environment': environment,
                'auxiliary_loss_weights': zero_weights, **zero_weights}
            # These are weights-only diagnostic snapshots, not optimizer/RNG resume.
            if epoch == epochs:
                write(output / 'schedule.json', {'batches_sha256': batch_digest.hexdigest(),
                    'contexts_sha256': context_digest.hexdigest(),
                    'speeds_sha256': speed_digest.hexdigest(), 'completed_epochs': epoch})
            temp = output / 'model.pt.tmp'; torch.save(checkpoint, temp); temp.replace(output / 'model.pt')
            if retain_milestones:
                destination = output / 'milestones' / f'epoch-{epoch:04d}'
                destination.mkdir(parents=True, exist_ok=False)
                shutil.copyfile(output / 'model.pt', destination / 'model.pt')
                execute([sys.executable, str(ROOT/'training/export_model.py'),
                    '--checkpoint', str(destination/'model.pt'), '--tokens', str(tokens),
                    '--output', str(destination/'model.kwm')], destination/'export.log')
                bind_development_provenance(destination/'model.pt', destination/'model.kwm')
                milestones['records'].append({'epoch': epoch,
                    'directory': str(destination.relative_to(output)),
                    'files': {name: sha(destination/name) for name in
                        ('model.pt','model.kwm','model.kwm.provenance.json')},
                    'float_state_sha256': checkpoint['float_state_identity']['sha256']})
                write(output / 'milestones.json', milestones)
            write(output / 'progress.json', {'completed_epochs': epoch, 'requested_epochs': epochs,
                'completed': epoch == epochs, 'seconds': time.monotonic()-started,
                'float_state_sha256': checkpoint['float_state_identity']['sha256'], 'last': history[-1]})
            print(json.dumps({'epoch':epoch, 'loss':total/batches,'seconds':round(time.monotonic()-started,2)}),flush=True)
    execute([sys.executable,str(ROOT/'training/export_model.py'),'--checkpoint',str(output/'model.pt'),
             '--tokens',str(tokens),'--output',str(output/'model.kwm')], output/'export.log')
    bind_development_provenance(output/'model.pt',output/'model.kwm')
    provenance = json.loads((output/'model.kwm.provenance.json').read_text())
    if provenance['training'].get('development_recipe') != recipe:
        raise ValueError('export lost development recipe authority')
    verify_pool(pool_root, pool_sha)
    if retain_milestones:
        milestones['completed'] = True
        write(output / 'milestones.json', milestones)
    if initial_checkpoint is not None and sha(initial_checkpoint) != recipe['warm_start']['checkpoint_sha256']:
        raise ValueError('warm-start checkpoint changed during training')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('pool','output'): p.add_argument('--'+k,required=True,type=pathlib.Path)
    p.add_argument('--pool-sha',required=True);p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--scope',choices=('positive-only','all'),default='all')
    p.add_argument('--epochs',type=int,default=600);p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--retain-milestones', action='store_true')
    p.add_argument('--initial-checkpoint', type=pathlib.Path)
    a=p.parse_args(); train(a.pool.resolve(),a.pool_sha,a.output.resolve(),a.variant,a.scope,a.epochs,a.seed,
                           retain_milestones=a.retain_milestones,
                           initial_checkpoint=a.initial_checkpoint.resolve() if a.initial_checkpoint else None)

if __name__ == '__main__':
    main()
