#!/usr/bin/env python3
"""Report-only paired non-silent context checks; no training or threshold edits.

A prior nonwake clip is rejected if its transcript plus the next transcript
creates a configured keyword across their join. Otherwise ordinary expected
windows would incorrectly call a newly formed full keyword a false accept.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import random
import struct
import subprocess
import sys

from frozen_speech_ablation import verify_pool, resolve, sha, write
from startup_context_evaluation import pcm, wav, shifted, collect, measure
from wake_pressure_balance import keyword_target_sequences

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = 'paired-preceding-context-regression-v1'
CONDITIONS = ('noise-low', 'noise-mid', 'prior-nonwake')
GAP_SAMPLES = 3200


def tokens(value) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value or any(type(t) is not int or t <= 0 for t in value):
        raise ValueError('context transcripts require nonempty nonblank integer tokens')
    return tuple(value)


def crosses_keyword_join(prefix, following, keywords: dict) -> bool:
    left, right = tokens(prefix), tokens(following)
    if not isinstance(keywords, dict) or not keywords:
        raise ValueError('configured keywords are required')
    joined = left + right
    for word in keywords.values():
        word = tokens(word)
        for start in range(max(0, len(left) - len(word) + 1), len(left)):
            if start + len(word) > len(left) and joined[start:start + len(word)] == word:
                return True
    return False


def contains_keyword(sequence, keywords: dict) -> bool:
    sequence = tokens(sequence)
    return any(sequence[i:i + len(word)] == word
               for word in map(tokens, keywords.values())
               for i in range(len(sequence) - len(word) + 1))


def select_prior(rows: list[dict], target: dict, index: int, keywords: dict) -> dict:
    if type(index) is not int or index < 0 or target['split'] not in ('train', 'calibration', 'test'):
        raise ValueError('invalid context selection index/split')
    candidates = [row for row in rows if row['split'] == target['split'] and not row['expected']
                  and not contains_keyword(row['target_ids'], keywords)
                  and not crosses_keyword_join(row['target_ids'], target['target_ids'], keywords)]
    if not candidates:
        raise ValueError('no same-split nonwake context without a cross-boundary keyword')
    return candidates[index % len(candidates)]


def noise_prefix(scale: int) -> bytes:
    if type(scale) is not int or scale not in (1, 10):
        raise ValueError('only predeclared noise scales are supported')
    rng = random.Random(7109)
    values = [rng.randint(-57, 57) * scale for _ in range(12800)]
    return struct.pack('<' + 'h' * len(values), *values) + b'\0\0' * GAP_SAMPLES


def speech_prefix(raw: bytes) -> bytes:
    if not isinstance(raw, bytes) or not raw or len(raw) % 2:
        raise ValueError('nonempty PCM16 bytes required')
    return raw + b'\0\0' * ((-(len(raw) // 2)) % 320 + GAP_SAMPLES)


def evaluate(pool_root: pathlib.Path, pool_sha: str, model: pathlib.Path,
             runner: pathlib.Path, output: pathlib.Path) -> dict:
    pool = verify_pool(pool_root, pool_sha)
    keywords = keyword_target_sequences(pool_root / 'tokens.example.txt', pool_root / 'zh_cn_example.tsv')
    model_sha, runner_sha = sha(model), sha(runner)
    output.mkdir(parents=True, exist_ok=False)
    pack = output / 'keywords.kwk'
    subprocess.run([sys.executable, str(ROOT / 'tools/compile_keywords.py'),
                    '--tokens', str(pool_root / 'tokens.example.txt'),
                    '--keywords', str(pool_root / 'zh_cn_example.tsv'), '--out-pack', str(pack)],
                   check=True, stdout=subprocess.DEVNULL, timeout=30)
    report = {'policy': POLICY, 'completed': False, 'development_only': True, 'report_only': True,
              'release_authority': False, 'pool_sha256': pool_sha, 'model_sha256': model_sha,
              'runner_sha256': runner_sha, 'pack_sha256': sha(pack),
              'implementation_sha256': {path: sha(ROOT / path) for path in (
                  'training/preceding_context_evaluation.py', 'training/startup_context_evaluation.py',
                  'training/frozen_speech_ablation.py', 'training/wake_pressure_balance.py', 'eval/score_events.py')},
              'noise_seed': 7109, 'noise_active_samples': 12800, 'gap_samples': GAP_SAMPLES,
              'contexts_overlap_target_audio': False, 'rows': [],
              'limitations': ['Synthetic noise is before speech, not overlapping SNR or physical room noise.',
                             'Prior speech uses same-split development nonwake clips only.',
                             'Cross-boundary lexical wakes are excluded by transcript, not acoustic judgment.',
                             'Short reused development exposure is not product FAR/qualification.']}
    write(output / 'summary.json', report)
    noises = {name: noise_prefix(scale) for name, scale in (('noise-low', 1), ('noise-mid', 10))}
    for split in ('train', 'calibration', 'test'):
        targets = [r for r in pool['rows'] if r['split'] == split]
        for condition in CONDITIONS:
            references, detections, controls, records = [], [], [], []
            for i, row in enumerate(targets):
                raw = pcm(resolve(pool_root, row['path']))
                prior = None
                if condition == 'prior-nonwake':
                    prior = select_prior(pool['rows'], row, i, keywords)
                    before = speech_prefix(pcm(resolve(pool_root, prior['path'])))
                else:
                    before = noises[condition]
                prefix_samples = len(before) // 2
                name = f'{split}-{i}'
                ref = {'recording': name, 'duration_s': (len(before) + len(raw)) / 32000,
                       'expected': shifted(row['expected'], prefix_samples)}
                audio = output / 'scratch.wav'
                wav(audio, before + raw)
                actual = collect(runner, model, pack, audio, name)
                wav(audio, b'\0\0' * prefix_samples + raw)
                control = collect(runner, model, pack, audio, name)
                references.append(ref); detections.extend(actual); controls.extend(control)
                records.append({'source_wav_sha256': row['wav_sha256'], 'target_ids': row['target_ids'],
                                'prefix_pcm_sha256': hashlib.sha256(before).hexdigest(),
                                'prefix_samples': prefix_samples,
                                'prior_wav_sha256': prior['wav_sha256'] if prior else None,
                                'prior_target_ids': prior['target_ids'] if prior else None,
                                'cross_boundary_keyword': False, 'reference': ref,
                                'context_detections': actual, 'silence_detections': control})
            result = {'split': split, 'condition': condition,
                      'context': measure(references, detections),
                      'paired_silence': measure(references, controls)}
            report['rows'].append(result)
            write(output / f'{split}-{condition}.json', {'summary': result, 'records': records})
            write(output / 'summary.json', report)
            print(f"{split} {condition}: silence={result['paired_silence']['matched']}/"
                  f"{result['paired_silence']['expected']},FA={result['paired_silence']['false_accepts']} "
                  f"context={result['context']['matched']}/{result['context']['expected']},"
                  f"FA={result['context']['false_accepts']}", flush=True)
    verify_pool(pool_root, pool_sha)
    if sha(model) != model_sha or sha(runner) != runner_sha:
        raise ValueError('model or runtime changed during context evaluation')
    (output / 'scratch.wav').unlink()
    report['completed'] = True
    write(output / 'summary.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'model', 'runner', 'output'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    parser.add_argument('--pool-sha', required=True)
    args = parser.parse_args()
    evaluate(args.pool.resolve(), args.pool_sha, args.model.resolve(), args.runner.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
