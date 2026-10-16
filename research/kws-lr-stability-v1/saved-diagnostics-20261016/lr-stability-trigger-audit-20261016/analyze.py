#!/usr/bin/env python3
"""Stdlib-only saved-LP prefix audit. No model, audio, runtime or training calls."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

PLAN_SHA256 = 'cc947af40d436b1969fb78ae6c2c1c9f5953b4405f57b39949fb52a52ad273e8'
DEFINITIONS_SHA256 = 'b92b4e93d4474ed2bacb7be043b0a7e73f655c90a49218bbd6f8a1cf47e2fa99'
NEG = -math.inf


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_json(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError('SHA256 mismatch: ' + str(path))
    return json.loads(raw)


def logadd(*values):
    maximum = max(values)
    if maximum == NEG:
        return NEG
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


def collapse(path):
    result = []
    previous = None
    for token in path:
        if token != 0 and token != previous:
            result.append(token)
        previous = token
    return tuple(result)


def beam_step(state, row, width):
    candidates = {}

    def add(prefix, branch, value):
        pair = candidates.setdefault(prefix, [NEG, NEG])
        pair[branch] = logadd(pair[branch], value)

    for prefix, (pb, pn) in state.items():
        total = logadd(pb, pn)
        add(prefix, 0, total + row[0])
        for token in range(1, len(row)):
            if prefix and prefix[-1] == token:
                add(prefix, 1, pn + row[token])
                add(prefix + (token,), 1, pb + row[token])
            else:
                add(prefix + (token,), 1, total + row[token])
    candidates = {p: masses for p, masses in candidates.items()
                  if logadd(*masses) != NEG}
    return dict(sorted(candidates.items(), key=lambda pair:
                       (-logadd(*pair[1]), pair[0]))[:width])


def forward_init(target):
    expanded = [0]
    for token in target:
        expanded.extend((token, 0))
    return expanded, [0.0] + [NEG] * (len(expanded) - 1)


def forward_step(expanded, previous, row):
    current = []
    for index, token in enumerate(expanded):
        paths = [previous[index]]
        if index:
            paths.append(previous[index - 1])
        if index > 1 and token != 0 and token != expanded[index - 2]:
            paths.append(previous[index - 2])
        current.append(logadd(*paths) + row[token])
    return current, logadd(*current[-2:])


def intervals(flags):
    result = []
    start = None
    for index, value in enumerate([*flags, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append([start, index - 1])
            start = None
    return result


def occurrences(sequence, target):
    count = index = 0
    while index + len(target) <= len(sequence):
        if tuple(sequence[index:index + len(target)]) == tuple(target):
            count += 1
            index += len(target)
        else:
            index += 1
    return count


def available_ms(frame):
    return None if frame is None else 25 + 20 * frame


def run_stats(flags):
    runs = intervals(flags)
    first = runs[0][0] if runs else None
    stable = runs[-1][0] if runs and runs[-1][1] == len(flags) - 1 else None
    return {'intervals_inclusive': runs, 'frame_count': sum(flags),
            'episode_count': len(runs), 'extra_entries': max(0, len(runs) - 1),
            'first_frame': first, 'first_available_ms': available_ms(first),
            'terminal_stable_start_frame_hindsight': stable,
            'terminal_stable_available_ms_hindsight': available_ms(stable)}


def validate_rows(rows):
    for row in rows:
        if len(row) != 5 or not all(isinstance(x, (int, float)) and
                                    not isinstance(x, bool) and math.isfinite(x)
                                    for x in row):
            raise ValueError('invalid saved log-probability row')
        if abs(logadd(*row)) > 2e-6:
            raise ValueError('saved row is not normalized within accepted tolerance')


def self_test():
    probs = [[.3, .2, .1, .25, .15], [.4, .1, .2, .2, .1],
             [.2, .3, .2, .1, .2], [.2, .1, .2, .2, .3]]
    rows = [[math.log(x) for x in row] for row in probs]
    state = {(): [0., NEG]}
    checked_sequences = 0
    for length in range(1, 5):
        exact = {}
        for path in itertools.product(range(5), repeat=length):
            prefix = collapse(path)
            probability = math.prod(probs[t][token] for t, token in enumerate(path))
            exact[prefix] = exact.get(prefix, 0.) + probability
        state = beam_step(state, rows[length - 1], 625)
        assert set(state) == set(exact)
        for sequence, probability in exact.items():
            assert abs(logadd(*state[sequence]) - math.log(probability)) < 1e-12
            expanded, forward = forward_init(sequence)
            for row in rows[:length]:
                forward, score = forward_step(expanded, forward, row)
            assert abs(score - math.log(probability)) < 1e-12
            checked_sequences += 1
        assert next(iter(state)) == min(exact, key=lambda p: (-exact[p], p))
    assert collapse([1, 1, 0, 1]) == (1, 1)
    assert collapse([3, 4, 3, 4]) == (3, 4, 3, 4)
    assert intervals([False, True, True, False, True]) == [[1, 2], [4, 4]]
    assert intervals([]) == []
    assert run_stats([True, False, True])['extra_entries'] == 1
    assert occurrences([3, 4, 3, 4, 3, 4], [3, 4, 3, 4]) == 1
    assert occurrences([3, 4] * 4, [3, 4, 3, 4]) == 2
    assert available_ms(0) == 25 and available_ms(1) == 45
    for invalid in ([[0.] * 5], [[math.nan] * 5], [[0.]]):
        try:
            validate_rows(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid row accepted')
    return {'exhaustive_prefix_lengths': [1, 2, 3, 4],
            'paths_at_length4': 625, 'sequence_prefix_comparisons': checked_sequences,
            'beam_and_forward_all_sequence_masses_verified': True,
            'collapse_intervals_occurrences_validation': 'PASS'}


def audit_sample(sample, config):
    rows = sample['log_probs']
    validate_rows(rows)
    raw = [max(range(5), key=lambda token: row[token]) for row in rows]
    assert raw == sample['raw_argmax_tokens']
    assert list(collapse(raw)) == sample['decode_tokens']
    targets = {key: tuple(value) for key, value in config['targets'].items()}
    target = targets[sample['target_class']]
    assert target == tuple(sample['target_tokens'])
    assert len(rows) == sample['input_frames']
    state = {(): [0., NEG]}
    forward = {key: forward_init(value) for key, value in targets.items()}
    greedy = []
    previous = None
    trace = []
    for frame, row in enumerate(rows):
        token = raw[frame]
        if token and token != previous:
            greedy.append(token)
        previous = token
        state = beam_step(state, row, config['beam_width'])
        best = next(iter(state))
        scores = {}
        for key, (expanded, values) in forward.items():
            values, score = forward_step(expanded, values, row)
            forward[key] = expanded, values
            scores[key] = None if score == NEG else score
        margin = (scores['KW2'] - scores['SHORT']
                  if scores['KW2'] is not None and scores['SHORT'] is not None
                  else None)
        trace.append({'frame': frame, 'nominal_available_ms': available_ms(frame),
                      'blank_argmax': token == 0, 'raw_argmax': token,
                      'greedy': list(greedy), 'beam32': list(best),
                      'beam32_pruned_log_probability': logadd(*state[best]),
                      'exact_ctc_log_probability': scores,
                      'kw2_full_over_short_log_ratio': margin})
    assert math.isclose(-trace[-1]['exact_ctc_log_probability'][sample['target_class']],
                        sample['ctc_nll'], abs_tol=.001, rel_tol=1e-5)
    result = {'record_id': sample['record_id'], 'class': sample['target_class'],
              'pcm_sha256': sample['pcm_sha256'], 'input_frames': len(rows),
              'frozen_rms_prefix_frames': sample['prefix_frames'],
              'final_frame_nominal_available_ms': available_ms(len(rows) - 1),
              'decoders': {}, 'trace': trace}
    for decoder in ('greedy', 'beam32'):
        hypotheses = [tuple(row[decoder]) for row in trace]
        changes = [{'frame': i, 'tokens': list(tokens)} for i, tokens in enumerate(hypotheses)
                   if i == 0 or tokens != hypotheses[i - 1]]
        occurrence_counts = [occurrences(tokens, target) for tokens in hypotheses]
        result['decoders'][decoder] = {
            'final_tokens': list(hypotheses[-1]), 'final_exact_match': hypotheses[-1] == target,
            'matches': {key: run_stats([tokens == sequence for tokens in hypotheses])
                        for key, sequence in targets.items()},
            'target_nonoverlap_occurrences_final': occurrence_counts[-1],
            'target_nonoverlap_occurrences_max': max(occurrence_counts),
            'transitions': changes}
    if sample['target_class'] == 'KW2':
        margins = [row['kw2_full_over_short_log_ratio'] for row in trace]
        result['kw2_probability_order'] = {
            'strict_full_greater_than_short': run_stats([x is not None and x > 0 for x in margins]),
            'defined_frame_count': sum(x is not None for x in margins),
            'final_log_ratio': margins[-1]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    tests = self_test()
    if args.self_test:
        print(json.dumps(tests, sort_keys=True))
        return
    if args.base is None or args.output is None:
        parser.error('--base and --output are required unless --self-test')
    here = Path(__file__).resolve().parent
    if args.output.resolve().parent != here:
        raise ValueError('output must be a new file in this independent audit directory')
    config = checked_json(here / 'analysis-plan.json', PLAN_SHA256)
    if digest((here / 'DEFINITIONS.md').read_bytes()) != DEFINITIONS_SHA256:
        raise ValueError('definition identity mismatch')
    loaded = {}
    for key, entry in config['inputs'].items():
        path = args.base / entry['relative_path']
        if path.is_symlink() or digest(path.read_bytes()) != entry['sha256']:
            raise ValueError('pinned input mismatch: ' + key)
        if path.suffix == '.json':
            loaded[key] = json.loads(path.read_bytes())
    source_manifest = checked_json(here / 'source-manifest.json', config['source_manifest_sha256'])
    assert source_manifest['commit'] == config['runtime_source_commit']
    for relative, entry in source_manifest['files'].items():
        raw = (here / 'source-contract' / relative).read_bytes()
        assert digest(raw) == entry['sha256']
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == entry['git_blob_sha1']
    manifest, plan, cohort, observation = (loaded[key] for key in
                                          ('accepted_manifest', 'plan', 'cohort', 'observation'))
    assert manifest['result_files']['A-0600.json'] == config['inputs']['observation']['sha256']
    assert manifest['plan_sha256'] == config['inputs']['plan']['sha256']
    assert plan['cohort_sha256'] == config['inputs']['cohort']['sha256']
    for key, source in [('model_source', 'training/model.py'),
                        ('frontend_source', 'training/frontend.py'),
                        ('frontend_spec_source', 'training/frontend_spec.py')]:
        assert plan['code_sha256'][source] == config['inputs'][key]['sha256']
    assert plan['frontend'] == {'frontend': 'logmel', 'feature_dim': 32, 'frame_len': 400, 'hop': 320}
    assert plan['model']['vocab_size'] == 5
    assert observation['update'] == 600 and observation['kind'] == 'observation'
    samples = observation['samples']
    assert len(samples) == config['sample_count']
    assert [s['record_id'] for s in samples] == plan['train_record_ids']
    frozen = {s['record_id']: s for s in cohort['samples'] if s['role'] == 'train'}
    assert set(frozen) == {s['record_id'] for s in samples}
    for sample in samples:
        assert all(sample[key] == frozen[sample['record_id']][key]
                   for key in ('record_id', 'target_class', 'pcm_sha256', 'input_frames', 'prefix_frames'))
    assert {key: sum(s['target_class'] == key for s in samples) for key in config['class_counts']} == config['class_counts']
    rows = [audit_sample(sample, config) for sample in samples]
    summary = {}
    for decoder in ('greedy', 'beam32'):
        summary[decoder] = {}
        for target_class in ('KW1', 'KW2'):
            selected = [r for r in rows if r['class'] == target_class]
            decisions = [r['decoders'][decoder] for r in selected]
            full = [d['matches'][target_class] for d in decisions]
            short = [d['matches']['SHORT'] for d in decisions]
            frames = [s['first_frame'] for s in full if s['first_frame'] is not None]
            stable = [s['terminal_stable_start_frame_hindsight'] for s in full
                      if s['terminal_stable_start_frame_hindsight'] is not None]
            summary[decoder][target_class] = {
                'samples': len(selected), 'final_exact_count': sum(d['final_exact_match'] for d in decisions),
                'ever_full_exact_count': sum(s['episode_count'] > 0 for s in full),
                'full_extra_entries_total': sum(s['extra_entries'] for s in full),
                'ever_short_exact_count': sum(s['episode_count'] > 0 for s in short),
                'final_short_exact_count': sum(d['final_tokens'] == config['targets']['SHORT'] for d in decisions),
                'first_full_frame_range': [min(frames), max(frames)] if frames else None,
                'first_full_nominal_ms_range': [available_ms(min(frames)), available_ms(max(frames))] if frames else None,
                'terminal_stable_frame_range_hindsight': [min(stable), max(stable)] if stable else None,
                'max_target_nonoverlap_occurrences': max(d['target_nonoverlap_occurrences_max'] for d in decisions)}
    report = {'schema': 'saved-prefix-audit-result-v1', 'status': 'POSTHOC_DESCRIPTIVE_ONLY',
              'analysis_plan_sha256': PLAN_SHA256, 'definitions_sha256': DEFINITIONS_SHA256,
              'analyzer_sha256': digest(Path(__file__).read_bytes()),
              'input_hashes': config['inputs'], 'source_manifest_sha256': config['source_manifest_sha256'],
              'validation': tests, 'new_model_forwards': 0, 'new_optimizer_updates': 0,
              'runtime_replay': False, 'summary': summary, 'samples': rows,
              'limitations': ['Twelve exposed machine-labelled positives only; no independent evaluation.',
                              'Prefix exact-match episodes are not runtime wake events.',
                              'Nominal clip-relative window coordinates are not keyword-end latency.',
                              'No false-activation, repeat-wake, generalization or product-readiness claims.',
                              'Finite beam32 is not a global MAP proof; no decoder or threshold selection.',
                              'No audio, weights, model calls, VAD replay, TTS or ASR executed.']}
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
