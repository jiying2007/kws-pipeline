#!/usr/bin/env python3
"""Report-only event attribution using three authentic C arms and a counterfactual.

Emission time is not acoustic cause. Repeated predecessor clips are not independent
negative exposure. Prefix blanking edits decoder evidence, not model weights, and
must never be represented as authentic inference or qualification evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import pathlib
import subprocess
import sys

from preceding_context_evaluation import (
    POLICY as SOURCE_POLICY, ROOT, select_prior, speech_prefix, keyword_target_sequences,
    verify_pool, resolve, sha, write, pcm, wav, shifted, collect,
)
from startup_context_evaluation import blank_prefix
from score_events import validate_recordings, validate_detections, score

POLICY = 'paired-context-event-attribution-v1'
SPLITS = ('train', 'calibration', 'test')


def scored(reference: dict, detections: list[dict]) -> tuple[dict, list[dict]]:
    refs = validate_recordings([reference])
    metrics, false_accepts, _ = score(refs, validate_detections(detections, refs), 0.15, 0.5)
    return metrics, false_accepts


def event_key(event: dict) -> tuple[int, float]:
    # Validation belongs to scored(); exact event time/keyword, NOT confidence.
    return event['keyword_id'], event['time_s']


def emission_region(time_s: float, prior_samples: int, prefix_samples: int) -> str:
    if (type(prior_samples) is not int or type(prefix_samples) is not int
            or not 0 < prior_samples < prefix_samples or prefix_samples % 320):
        raise ValueError('invalid predecessor/prefix boundaries')
    if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s) or time_s < 0:
        raise ValueError('invalid emission time')
    if time_s < prior_samples / 16000:
        return 'prior_clip'
    return 'gap' if time_s < prefix_samples / 16000 else 'target_clip'


def attribute_events(false_accepts: list[dict], silence_fa: list[dict],
                     isolated_events: list[dict], prior_samples: int, prefix_samples: int,
                     blanked_fa: list[dict] | None = None) -> list[dict]:
    counts = [Counter(map(event_key, rows)) for rows in (silence_fa, isolated_events, blanked_fa or [])]
    result = []
    for event in false_accepts:
        present = []
        for counter in counts:
            key = event_key(event)
            present.append(counter[key] > 0)
            if counter[key]:
                counter[key] -= 1  # Duplicate events cannot reuse one control event.
        silence, isolated, blanked = present
        label = ('both_controls' if silence and isolated else 'isolated_prior_reproduced' if isolated
                 else 'silence_target_reproduced' if silence else 'combined_only')
        result.append({'event': event, 'emission_region': emission_region(event['time_s'], prior_samples, prefix_samples),
                       'control_relation': label,
                       'survives_decoder_prefix_blanking_at_same_time': blanked if blanked_fa is not None else None})
    return result


def counterfactual_effect(actual: dict, edited: dict) -> dict:
    """Do not mistake removal/replacement of one signature for removal of all FA."""
    return {'false_accept_count_delta': edited['false_accepts'] - actual['false_accepts'],
            'matched_count_delta': edited['matched'] - actual['matched'],
            'all_false_accepts_removed': actual['false_accepts'] > 0 and edited['false_accepts'] == 0}


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('expected object')
    return value


def validate_source(report: dict, pool_sha: str, model_sha: str) -> None:
    if (report.get('policy') != SOURCE_POLICY or report.get('completed') is not True
            or report.get('development_only') is not True or report.get('report_only') is not True
            or report.get('release_authority') is not False or report.get('pool_sha256') != pool_sha
            or report.get('model_sha256') != model_sha):
        raise ValueError('invalid source context report authority or identity')


def evaluate(pool_root: pathlib.Path, pool_sha: str, model: pathlib.Path, runner: pathlib.Path,
             dump: pathlib.Path, replay: pathlib.Path, source: pathlib.Path, output: pathlib.Path) -> dict:
    pool = verify_pool(pool_root, pool_sha)
    source_report = load_object(source / 'summary.json')
    identities = {name: sha(path) for name, path in (('model', model), ('runner', runner), ('dump', dump), ('replay', replay))}
    validate_source(source_report, pool_sha, identities['model'])
    # Deterministic definitions and byte identity are revalidated against the pool.
    keywords = keyword_target_sequences(pool_root / 'tokens.example.txt', pool_root / 'zh_cn_example.tsv')
    pack = source / 'keywords.kwk'
    if sha(pack) != source_report['pack_sha256']:
        raise ValueError('source keyword pack mismatch')
    output.mkdir(parents=True, exist_ok=False)
    original_pack = pack
    pack = output / 'keywords.kwk'
    subprocess.run([sys.executable, str(ROOT / 'tools/compile_keywords.py'),
                    '--tokens', str(pool_root / 'tokens.example.txt'),
                    '--keywords', str(pool_root / 'zh_cn_example.tsv'), '--out-pack', str(pack)],
                   check=True, stdout=subprocess.DEVNULL, timeout=30)
    if sha(pack) != sha(original_pack):
        raise ValueError('source pack differs from frozen configured thresholds')
    traces = output / 'traces'; traces.mkdir()
    report = {'policy': POLICY, 'completed': False, 'development_only': True, 'report_only': True,
              'release_authority': False, 'pool_sha256': pool_sha, 'identities': identities,
              'pack_sha256': sha(pack), 'source_summary_sha256': sha(source / 'summary.json'),
              'source_files_sha256': {}, 'implementation_sha256': {p: sha(ROOT / p) for p in (
                  'training/preceding_context_attribution.py', 'training/preceding_context_evaluation.py',
                  'training/startup_context_evaluation.py', 'eval/score_events.py')}, 'splits': {},
              'limitations': ['Event time regions locate emission, not phoneme origin or causal state.',
                  'Exact-time control matches ignore confidence; absent exact match can include timing shifts.',
                  'Prefix-only arm replaces the entire target with equal-length digital silence.',
                  'Edited logits are counterfactual decoder evidence, not authentic model output.',
                  'Reused prior clips and development feedback sets cannot establish release FAR.']}
    write(output / 'summary.json', report)
    for split in SPLITS:
        path = source / f'{split}-prior-nonwake.json'
        report['source_files_sha256'][path.name] = sha(path)
        previous = load_object(path)
        targets = [r for r in pool['rows'] if r['split'] == split]
        if (previous['summary']['split'] != split or previous['summary']['condition'] != 'prior-nonwake'
                or len(previous['records']) != len(targets)):
            raise ValueError('source record coverage mismatch')
        entries, buckets, relations = [], Counter(), Counter()
        all_refs, all_actual, all_control = [], [], []
        unique_priors, isolated_reproduced = set(), Counter()
        counterfactual_count = 0
        for index, (row, old) in enumerate(zip(targets, previous['records'])):
            prior = select_prior(pool['rows'], row, index, keywords)
            raw = pcm(resolve(pool_root, row['path'])); prior_raw = pcm(resolve(pool_root, prior['path']))
            before = speech_prefix(prior_raw); prefix = len(before) // 2
            name = f'{split}-{index}'
            reference = {'recording': name, 'duration_s': (len(before) + len(raw)) / 32000,
                         'expected': shifted(row['expected'], prefix)}
            expected = {'source_wav_sha256': row['wav_sha256'], 'prior_wav_sha256': prior['wav_sha256'],
                        'target_ids': row['target_ids'], 'prior_target_ids': prior['target_ids'],
                        'prefix_samples': prefix, 'prefix_pcm_sha256': hashlib.sha256(before).hexdigest(),
                        'reference': reference}
            if old.get('cross_boundary_keyword') is not False or any(old.get(k) != v for k, v in expected.items()):
                raise ValueError('source record does not match deterministic frozen inputs')
            arms, metrics, failures = {}, {}, {}
            for arm, audio_bytes in (('context', before + raw), ('silence', b'\0' * len(before) + raw),
                                     ('isolated_prior', before + b'\0' * len(raw))):
                audio = output / 'scratch.wav'; wav(audio, audio_bytes)
                arms[arm] = collect(runner, model, pack, audio, name)
                ref = {**reference, 'expected': []} if arm == 'isolated_prior' else reference
                metrics[arm], failures[arm] = scored(ref, arms[arm])
            # Never silently mix changed inference with retained attribution inputs.
            for arm, field in (('context', 'context_detections'), ('silence', 'silence_detections')):
                old_metrics, _ = scored(reference, old[field])
                if metrics[arm] != old_metrics:
                    raise ValueError('fresh runtime event metrics differ from retained source')
            blanked_fa = None
            if any(e['time_s'] >= prefix / 16000 for e in failures['context']):
                wav(output / 'scratch.wav', before + raw)
                trace = traces / f'{name}.kwtr'
                subprocess.run([str(dump), str(model), str(output / 'scratch.wav'), str(trace)],
                               stdout=subprocess.DEVNULL, check=True, timeout=30)
                authentic_replay = collect(replay, model, pack, trace, name)
                scored(reference, authentic_replay)
                if Counter(map(event_key, authentic_replay)) != Counter(map(event_key, arms['context'])):
                    raise ValueError('authentic posterior replay event mismatch')
                edited = traces / f'{name}-blanked.kwtr'
                blank_prefix(trace, edited, prefix)
                arms['decoder_prefix_blanked'] = collect(replay, model, pack, edited, name)
                metrics['decoder_prefix_blanked'], blanked_fa = scored(reference, arms['decoder_prefix_blanked'])
                counterfactual_count += 1
            attribution = attribute_events(failures['context'], failures['silence'], arms['isolated_prior'],
                                           len(prior_raw)//2, prefix, blanked_fa)
            for event in attribution:
                buckets[event['emission_region']] += 1; relations[event['control_relation']] += 1
                if event['control_relation'] in ('isolated_prior_reproduced', 'both_controls'):
                    unique_priors.add(prior['wav_sha256'])
                    isolated_reproduced[prior['wav_sha256']] += 1
            entries.append({**expected, 'prior_samples': len(prior_raw)//2, 'arms': arms,
                            'metrics': metrics, 'false_accept_attribution': attribution,
                            'context_minus_silence_fa': metrics['context']['false_accepts'] - metrics['silence']['false_accepts'],
                            'counterfactual_effect': counterfactual_effect(metrics['context'], metrics['decoder_prefix_blanked'])
                                if 'decoder_prefix_blanked' in metrics else None})
            all_refs.append(reference); all_actual.extend(arms['context']); all_control.extend(arms['silence'])
        refs = validate_recordings(all_refs)
        context_metrics = score(refs, validate_detections(all_actual, refs), .15, .5)[0]
        control_metrics = score(refs, validate_detections(all_control, refs), .15, .5)[0]
        if sum(buckets.values()) != context_metrics['false_accepts']:
            raise ValueError('attribution count conservation failed')
        summary = {'context': context_metrics, 'paired_silence': control_metrics,
                   'false_accept_emission_regions': dict(buckets), 'control_relations': dict(relations),
                   'unique_reproduced_prior_clips': len(unique_priors),
                   'reproduced_prior_occurrences': dict(isolated_reproduced),
                   'decoder_counterfactual_recordings': counterfactual_count,
                   'counterfactual_recordings_with_no_remaining_fa': sum(
                       r['counterfactual_effect'] is not None and r['counterfactual_effect']['all_false_accepts_removed']
                       for r in entries)}
        write(output / f'{split}.json', {'summary': summary, 'records': entries})
        report['splits'][split] = summary; write(output / 'summary.json', report)
        print(split, dict(buckets), dict(relations), flush=True)
    verify_pool(pool_root, pool_sha)
    if (identities != {name: sha(path) for name, path in (('model',model),('runner',runner),('dump',dump),('replay',replay))}
            or sha(pack) != report['pack_sha256'] or sha(original_pack) != report['pack_sha256'] or sha(source/'summary.json') != report['source_summary_sha256']
            or any(sha(source/name) != digest for name,digest in report['source_files_sha256'].items())):
        raise ValueError('input changed during attribution')
    (output / 'scratch.wav').unlink()
    report['completed'] = True; write(output / 'summary.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'model', 'runner', 'dump', 'replay', 'source', 'output'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    parser.add_argument('--pool-sha', required=True)
    args = parser.parse_args()
    evaluate(args.pool.resolve(), args.pool_sha, args.model.resolve(), args.runner.resolve(),
             args.dump.resolve(), args.replay.resolve(), args.source.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
