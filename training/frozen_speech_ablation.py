#!/usr/bin/env python3
"""Clean frozen-speech diagnostic; no mining, curriculum, qualification or promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT / 'tools'), str(ROOT / 'eval')]
from external_base_dataset import load_external_base_bundle
from objective_config import auxiliary_loss_weights, optional_objective_cli_args
from objective_contract import (
    ORDERED_TOKEN_SCOPE_DEFAULT, ORDERED_TOKEN_SCOPES,
    SEQUENCE_MARGIN_NEGATIVE_POLICIES,
    SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE,
)
from wake_pressure_balance import derive_wake_pressure_balance, keyword_target_sequences

POLICY = 'clean-frozen-speech-ablation-v1'
SPLITS = ('train', 'calibration', 'test')
VARIANTS = ('current', 'ctc-only')


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected object: {path}')
    return value


def write(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def resolve(root: pathlib.Path, value: str) -> pathlib.Path:
    p = pathlib.Path(value)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('frozen pool paths must be portable and confined')
    result = (root / p).resolve()
    result.relative_to(root.resolve())
    if not result.is_file() or result.is_symlink():
        raise ValueError(f'frozen pool file missing: {value}')
    return result


def materialize_pool(rows: list[dict], output: pathlib.Path,
                     keywords: dict[int, tuple[int, ...]]) -> list[dict]:
    """Retain ONLY train/calibration/feedback audio, never qualification inputs."""
    retained = []
    owners: dict[str, str] = {}
    for row in rows:
        split = row['split']
        if split not in SPLITS:
            continue
        path = pathlib.Path(row['path'])
        digest = sha(path)
        if digest != row['wav_sha256']:
            raise ValueError('source WAV bytes mismatch')
        if digest in owners:
            raise ValueError('duplicate or cross-split audio in frozen pool')
        owners[digest] = split
        target = row['target_ids']
        if (not isinstance(target, list) or any(type(t) is not int or t <= 0 for t in target)):
            raise ValueError('invalid target sequence')
        positive = row['kind'] == 'positive'
        if positive and tuple(target) != keywords.get(row['keyword_id']):
            raise ValueError('positive target does not match configured keyword')
        if not positive and tuple(target) in keywords.values():
            raise ValueError('negative contains exact wake target')
        with wave.open(str(path), 'rb') as f:
            if (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getcomptype()) != (1, 2, 16000, 'NONE'):
                raise ValueError('frozen audio must be mono PCM16 16kHz')
            frames = f.getnframes()
        if frames <= 0:
            raise ValueError('empty frozen audio')
        expected = []
        if positive:
            start, end = row['event_start_frame'], row['event_end_frame']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= frames:
                raise ValueError('invalid wake event bounds')
            expected = [{'keyword_id': row['keyword_id'], 'start_s': start / 16000, 'end_s': end / 16000}]
        relative = f'audio/{digest}.wav'
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        retained.append({'split': split, 'path': relative, 'wav_sha256': digest,
                         'kind': row['kind'], 'target_ids': target, 'expected': expected,
                         'duration_s': frames / 16000,
                         'source_provenance': row.get('speech_like_provenance', {})})
    for split in SPLITS:
        selected = [r for r in retained if r['split'] == split]
        if not selected or not any(not r['expected'] for r in selected):
            raise ValueError(f'{split}: no negative support')
        observed = {e['keyword_id'] for r in selected for e in r['expected']}
        if observed != set(keywords):
            raise ValueError(f'{split}: missing configured keyword support')
        (output / f'{split}.tsv').write_text(''.join(
            f"{r['path']}\t{' '.join(map(str, r['target_ids']))}\n" for r in selected), encoding='utf-8')
        with (output / f'{split}.references.jsonl').open('w', encoding='utf-8') as f:
            for i, r in enumerate(selected):
                f.write(json.dumps({'recording': f'{split}-{i:05d}', 'path': r['path'],
                                    'duration_s': r['duration_s'], 'expected': r['expected']},
                                   sort_keys=True, allow_nan=False) + '\n')
    return retained


def prepare(archive: pathlib.Path, output: pathlib.Path) -> None:
    contract_path = ROOT / 'configs/training/product-speech-like-base-v1.json'
    contract = read(contract_path)
    if sha(archive) != contract['release_archive_sha256']:
        raise ValueError('immutable release archive hash mismatch')
    output.mkdir(parents=True, exist_ok=False)
    source = output / '_source'
    source.mkdir()
    with tarfile.open(archive, 'r:gz') as f:
        for member in f.getmembers():
            if not (member.isfile() or member.isdir()):
                raise ValueError('archive links and special files forbidden')
        f.extractall(source, filter='data')
    corpus = source / 'corpus'
    bundle_path = corpus / 'stage-a-base-bundle.json'
    if sha(bundle_path) != contract['release_bundle_manifest_sha256']:
        raise ValueError('bundle manifest hash mismatch')
    bundle = read(bundle_path)
    cfg = {'generator': {'external_base_dataset': {
        split: {'index': str((corpus / 'bundle' / split / 'dataset-index.jsonl').resolve()),
                'summary': str((corpus / 'bundle' / split / 'dataset-summary.json').resolve()),
                'index_sha256': item['index_sha256'], 'summary_sha256': item['summary_sha256']}
        for split, item in bundle['splits'].items()}}}
    loaded = load_external_base_bundle(output / 'binding.json', cfg)
    if loaded is None or loaded[1]['bundle_sha256'] != contract['external_base_bundle_sha256']:
        raise ValueError('external base identity mismatch')
    rows, _ = loaded
    tokens = ROOT / 'keywords/tokens.example.txt'
    keywords = ROOT / 'keywords/zh_cn_example.tsv'
    retained = materialize_pool(rows, output, keyword_target_sequences(tokens, keywords))
    # Only a portable development pool is handed to training jobs.
    shutil.rmtree(source)
    for p in (tokens, keywords, keywords.with_suffix('.tsv.margin.json')):
        shutil.copyfile(p, output / p.name)
    files = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output / 'pool.json', {'policy': POLICY, 'development_only': True,
          'release_authority': False, 'qualification_scored': False,
          'archive_sha256': sha(archive), 'base_contract_sha256': sha(contract_path),
          'external_base_bundle_sha256': contract['external_base_bundle_sha256'],
          'files': files, 'rows': retained,
          'counts': {s: sum(r['split'] == s for r in retained) for s in SPLITS}})
    print(json.dumps({'prepared': True, 'counts': read(output / 'pool.json')['counts']}))


def verify_pool(root: pathlib.Path, expected_sha: str) -> dict:
    if sha(root / 'pool.json') != expected_sha:
        raise ValueError('frozen pool receipt hash mismatch')
    pool = read(root / 'pool.json')
    if pool.get('policy') != POLICY or pool.get('development_only') is not True or pool.get('release_authority') is not False:
        raise ValueError('invalid frozen pool authority')
    if not pool.get('files') or not pool.get('rows'):
        raise ValueError('empty frozen pool')
    for path, digest in pool['files'].items():
        if sha(resolve(root, path)) != digest:
            raise ValueError(f'frozen pool bytes changed: {path}')
    if any(r['split'] not in SPLITS for r in pool['rows']):
        raise ValueError('qualification is forbidden as a trial input')
    return pool


def loss_settings(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError('unknown ablation variant')
    # Explicit frozen treatment, not helper({}): that helper only returns configured keys.
    weights = {'ordered_token_loss_weight': 0.35,
               'keyword_sequence_margin_loss_weight': 0.10,
               'prefix_completion_loss_weight': 0.10,
               'recurrent_release_loss_weight': 0.05}
    if variant == 'ctc-only':
        weights = {name: 0.0 for name in weights}
    return {**weights, 'path_purity_loss_weight': 0.0,
            'sequence_margin_negative_policy': 'runtime-executable-v1'}


def scoped_loss_settings(
    variant: str, ordered_scope: str,
    negative_policy: str = SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE,
) -> dict:
    if ordered_scope not in ORDERED_TOKEN_SCOPES:
        raise ValueError('unsupported ordered-token scope')
    if negative_policy not in SEQUENCE_MARGIN_NEGATIVE_POLICIES:
        raise ValueError('unsupported sequence-margin negative policy')
    settings = loss_settings(variant)
    settings['sequence_margin_negative_policy'] = negative_policy
    if ordered_scope != ORDERED_TOKEN_SCOPE_DEFAULT:
        settings['ordered_token_scope'] = ordered_scope
    return settings


def clean_source_tree() -> str:
    status = subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=all'],
        cwd=ROOT, text=True,
    )
    if status.strip():
        raise ValueError('frozen-speech trial requires a clean source worktree')
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True,
    ).strip()


def execute(command: list[str], log: pathlib.Path) -> float:
    start = time.monotonic()
    with log.open('w', encoding='utf-8') as f:
        subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    return time.monotonic() - start


def acoustic_diagnostics(pool: dict, root: pathlib.Path, checkpoint: pathlib.Path,
                         posterior_dump: pathlib.Path, out: pathlib.Path) -> list[dict]:
    import struct
    import torch
    from model import TinyStreamingRNN
    from train_ctc import Manifest
    from diagnose_sequence_margin_runtime_gap import read_trace_logits
    from diagnose_acoustic_alignment import greedy_collapse
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = TinyStreamingRNN(cp['feature_dim'], cp['hidden_dim'], cp['vocab_size'])
    model.load_state_dict(cp['state_dict']); model.eval()
    quantized = TinyStreamingRNN(cp['feature_dim'], cp['hidden_dim'], cp['vocab_size'])
    state = {}
    for name, tensor in cp['state_dict'].items():
        if name.endswith('.weight'):
            scale = max(float(tensor.abs().max()), 1e-8) / 127
            packed_scale = struct.unpack('<f', struct.pack('<f', scale))[0]
            tensor = torch.round(tensor / scale).clamp(-127, 127) * packed_scale
        state[name] = tensor
    quantized.load_state_dict(state); quantized.eval()
    result = []
    loss = torch.nn.CTCLoss(blank=0, reduction='sum', zero_infinity=False)
    trace = out / 'probe.kwtr'
    with torch.inference_mode():
        for split in SPLITS:
            dataset = Manifest([root / f'{split}.tsv'], cp['feature_dim'], cp['vocab_size'], cp['frontend_name'])
            for index in range(len(dataset)):
                x, y = dataset[index]
                logits = model(x.unsqueeze(0))
                q_logits = quantized(x.unsqueeze(0))
                nll = loss(logits.log_softmax(2), y, torch.tensor([len(x)]), torch.tensor([len(y)]))
                if not torch.isfinite(nll):
                    raise ValueError('non-finite acoustic CTC NLL')
                subprocess.run([str(posterior_dump), str(checkpoint.with_suffix('.kwm')),
                                str(dataset.rows[index][0]), str(trace)], check=True, stdout=subprocess.DEVNULL)
                c_logits = read_trace_logits(trace)
                if len(c_logits) != len(x):
                    raise ValueError('Python/C frame count mismatch')
                target = tuple(y.tolist())
                result.append({'split': split, 'index': index, 'target': list(target),
                    'float_greedy_exact': greedy_collapse(logits[:, 0].tolist()) == target,
                    'quantized_greedy_exact': greedy_collapse(q_logits[:, 0].tolist()) == target,
                    'c_greedy_exact': greedy_collapse(c_logits) == target,
                    'ctc_nll_per_frame': float(nll) / len(x),
                    'float_quantized_max_logit_error': float((logits - q_logits).abs().max()),
                    'quantized_c_max_logit_error': float((q_logits[:, 0] - torch.tensor(c_logits)).abs().max())})
    trace.unlink(missing_ok=True)
    return result


def trial(root: pathlib.Path, pool_sha: str, variant: str, output: pathlib.Path,
          runner: pathlib.Path, posterior_dump: pathlib.Path, epochs: int, seed: int,
          ordered_scope: str = ORDERED_TOKEN_SCOPE_DEFAULT,
          negative_policy: str = SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE) -> None:
    if not 1 <= epochs <= 72 or not 0 <= seed <= 2147483647:
        raise ValueError('invalid bounded epochs or seed')
    pool = verify_pool(root, pool_sha)
    source_tree = clean_source_tree()
    output.mkdir(parents=True, exist_ok=False)
    tokens, keywords = root / 'tokens.example.txt', root / 'zh_cn_example.tsv'
    weights = scoped_loss_settings(variant, ordered_scope, negative_policy)
    balance = derive_wake_pressure_balance(manifests=[root / 'train.tsv'], tokens=tokens,
                                          keywords=keywords, positive_example_weight=2.0)
    checkpoint = output / 'model.pt'
    command = [sys.executable, str(ROOT / 'training/feature_cached_trainer.py'),
        '--trainer', 'rnn', '--feature-cache-max-items', '512', '--',
        '--manifest', str(root / 'train.tsv'), '--tokens', str(tokens), '--keywords', str(keywords),
        '--feature-dim', '32', '--hidden-dim', '64', '--frontend', 'logmel',
        '--epochs', str(epochs), '--batch-size', '16', '--seed', str(seed), '--lr', '0.001',
        '--positive-example-weight', '2.0', '--wake-keyword-weights', json.dumps(balance['wake_keyword_weights']),
        '--output', str(checkpoint), *optional_objective_cli_args(weights)]
    config = {'policy': POLICY, 'variant': variant, 'pool_sha256': pool_sha, 'epochs': epochs,
              'ordered_token_scope': ordered_scope,
              'sequence_margin_negative_policy': negative_policy,
              'seed': seed, 'train': weights, 'wake_balance': balance,
              'runner_sha256': sha(runner), 'posterior_dump_sha256': sha(posterior_dump),
              'source_tree': source_tree}
    write(output / 'trial-input.json', config)
    train_seconds = execute(command, output / 'train.log')
    execute([sys.executable, str(ROOT / 'training/export_model.py'), '--checkpoint', str(checkpoint),
             '--tokens', str(tokens), '--output', str(checkpoint.with_suffix('.kwm'))], output / 'export.log')
    from verify_training_readback import verify_candidate
    readback = verify_candidate(weights, checkpoint)
    import torch
    checkpoint_payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    provenance = read(pathlib.Path(str(checkpoint.with_suffix('.kwm')) + '.provenance.json'))
    if (checkpoint_payload.get('ordered_token_scope') != ordered_scope or
            provenance.get('training', {}).get('ordered_token_scope') != ordered_scope or
            checkpoint_payload.get('sequence_margin_negative_policy') != negative_policy or
            provenance.get('training', {}).get('sequence_margin_negative_policy') != negative_policy):
        raise ValueError('objective control differs across trial, checkpoint and model provenance')
    readback['ordered_token_scope'] = ordered_scope
    readback['sequence_margin_negative_policy'] = negative_policy
    write(output / 'readback.json', readback)
    execute([sys.executable, str(ROOT / 'tools/compile_keywords.py'), '--tokens', str(tokens),
             '--keywords', str(keywords), '--out-pack', str(output / 'keywords.kwk')], output / 'pack.log')
    runtime = {}
    for split in SPLITS:
        detections = output / f'{split}.detections.jsonl'
        execute([sys.executable, str(ROOT / 'eval/run_corpus.py'), '--runner', str(runner),
            '--model', str(checkpoint.with_suffix('.kwm')), '--keywords', str(output / 'keywords.kwk'),
            '--references', str(root / f'{split}.references.jsonl'), '--audio-root', str(root),
            '--detections', str(detections), '--provenance', str(output / f'{split}.provenance.json')], output / f'{split}-run.log')
        execute([sys.executable, str(ROOT / 'eval/score_events.py'), '--references', str(root / f'{split}.references.jsonl'),
            '--detections', str(detections), '--summary', str(output / f'{split}.summary.json')], output / f'{split}-score.log')
        runtime[split] = read(output / f'{split}.summary.json')
    diagnostics = acoustic_diagnostics(pool, root, checkpoint, posterior_dump, output)
    verify_pool(root, pool_sha)
    write(output / 'summary.json', {**config, 'development_only': True, 'release_authority': False,
        'readback': readback, 'train_seconds': train_seconds, 'runtime': runtime, 'acoustic': diagnostics,
        'limitations': ['Clean synthetic clips with per-recording reset, not far-field or continuous qualification.',
                        'Train-fit metrics are in-sample learnability only; calibration/test are development feedback.',
                        'All auxiliary terms removed together: combined ablation, not individual marginal causality.']})
    print(json.dumps({'variant': variant, 'train_seconds': train_seconds,
                      'test_frr': runtime['test']['frr'], 'test_far': runtime['test']['far_per_hour']}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--archive', type=pathlib.Path, required=True); p.add_argument('--output', type=pathlib.Path, required=True)
    p = sub.add_parser('trial')
    for name in ('pool', 'output', 'runner', 'posterior-dump'):
        p.add_argument('--' + name, type=pathlib.Path, required=True)
    p.add_argument('--pool-sha', required=True); p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--epochs', type=int, default=36); p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--ordered-token-scope', choices=sorted(ORDERED_TOKEN_SCOPES),
                   default=ORDERED_TOKEN_SCOPE_DEFAULT)
    p.add_argument('--sequence-margin-negative-policy',
                   choices=sorted(SEQUENCE_MARGIN_NEGATIVE_POLICIES),
                   default=SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.archive.resolve(), args.output.resolve())
    else:
        trial(args.pool.resolve(), args.pool_sha, args.variant, args.output.resolve(),
              args.runner.resolve(), args.posterior_dump.resolve(), args.epochs, args.seed,
              args.ordered_token_scope, args.sequence_margin_negative_policy)


if __name__ == '__main__':
    main()
