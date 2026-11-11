#!/usr/bin/env python3
"""Independent post-run audit of complementary saved captures and resources.

No model/head/detector/probe execution and no training. This does not turn the
original strict cross-architecture bit failure into a pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import struct

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

PRECISE = (0, 1027, 1401, 1462, 2093)
KNOWN = list(range(96)) + list(range(100, 134))
HEADER = struct.pack('<8s6I5IIQ', b'R1A113H\0', 1, 64, 1200, 140, 6, 2601, *PRECISE, 113, 0)
DTYPE = np.dtype([('frame', '<u8'), ('source', '<u8'), ('delivery', '<u8'),
                  ('winner', '<u4'), ('other', '<u4'), ('hidden', '<f8', (140,)), ('scores', '<f8', (6,))])
ASSERTIONS = 0


def need(ok, reason):
    global ASSERTIONS
    ASSERTIONS += 1
    if not ok:
        raise ValueError(reason)


def raw(path):
    need(stat.S_ISREG(path.lstat().st_mode), 'regular non-symlink evidence required: ' + str(path))
    return path.read_bytes()


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def load(path):
    def bad(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw(path), parse_constant=bad)


def frame_event_counts(data):
    need(data.endswith(b'\n'), 'incomplete F/E trace')
    frames, events = [], []
    for line in data.decode('ascii').splitlines():
        kind, *values = line.split(',')
        values = tuple(map(int, values))
        need(all(v >= 0 for v in values), 'negative trace field')
        if kind == 'F':
            need(len(values) == 4 and values[0] == len(frames) and values[1] <= values[2] and
                 values[3] < 2601, 'F row schema')
            frames.append(values)
        else:
            need(kind == 'E' and len(values) == 6 and values[0] == 0 and values[1] in (1, 2) and
                 values[3] <= values[2] < len(frames) and values[4] <= values[5], 'E row schema')
            events.append(values)
    words = Counter(row[1] for row in events)
    return frames, {'sdk_model_frames': len(frames), 'keyword_events': len(events),
                    'keyword1_events': words[1], 'keyword2_events': words[2]}


def capture(data, frames):
    need(data[:64] == HEADER and len(data) == 64 + frames * 1200, 'complete capture schema')
    array = np.frombuffer(data, dtype=DTYPE, offset=64)
    need(np.array_equal(array['frame'], np.arange(frames)), 'binary frame sequence')
    need(np.isfinite(array['hidden']).all() and np.isfinite(array['scores']).all(), 'finite captured values')
    return array


def numerical(actual, reference, logits):
    a, r = np.ascontiguousarray(actual), np.ascontiguousarray(reference)
    need(a.shape == r.shape and a.dtype == r.dtype == np.dtype('<f8'), 'numeric array shape/type')
    bit = a.view('<u8') != r.view('<u8')
    delta = a - r
    error = np.abs(delta)
    result = {'values': int(a.size), 'bit_different_values': int(bit.sum()),
              'bit_different_frames': int(bit.any(axis=1).sum()),
              'maximum_absolute_difference': float(error.max(initial=0)),
              'rms_difference': float(np.sqrt(np.mean(delta * delta)))}
    if logits:
        bound = 1e-8 + 1e-8 * np.abs(r)
        need(np.all(error <= bound) and np.all(error <= 1e-6), 'frozen six-score logit bound')
        result.update(atol=1e-8, rtol=1e-8, maximum_absolute_bound=1e-6,
                      maximum_tolerance_ratio=float(np.max(error / bound, initial=0)), contract_passed=True)
    else:
        result['acceptance_bound_applied'] = False
    return result


def same_metrics(saved, actual):
    need(saved.keys() == actual.keys(), 'numerical metric keys')
    for key, value in actual.items():
        if key == 'rms_difference':
            need(math.isclose(saved[key], value, rel_tol=2e-13, abs_tol=1e-30), 'independent RMS reduction')
        else:
            need(saved[key] == value, 'numerical metric mismatch: ' + key)


def receipt_checks(receipt, record, active, capture_mode, base_plan):
    expected = 'r1-active113-offline-resource-v1' if active else 'r1-strong-fsmn-offline-board-resource-v1'
    need(receipt['schema'] == expected and receipt['model_storage'] == 'read_only_mmap' and
         receipt['model_caller_heap_bytes'] == 0, 'resource/functional storage identity')
    need(receipt['pcm_bytes'] == record['pcm_bytes'] and receipt['feed_chunk_bytes'] == 640 and
         receipt['feed_calls'] == (record['pcm_bytes'] + 639) // 640, 'feed denominator')
    need(all(receipt[k] == v for k, v in record['expected_counts'].items()), 'complete event/frame counts')
    need(all(receipt[k] is False for k in ('device_qualified', 'live_afe', 'independent_far_frr')), 'qualification scope')
    if active:
        need(receipt['model_sha256'] == base_plan['active_model']['sha256'] and
             receipt['frontend_sha256'] == base_plan['frontend']['sha256'] and
             receipt['selection_sha256'] == base_plan['selection_sha256'] and
             receipt['model_weight_bytes'] == 1582624 and receipt['capture_instrumented'] is capture_mode,
             'active model/probe identity')
    audio = record['pcm_bytes'] / 32000.0
    need(abs(receipt['audio_seconds'] - audio) <= 5e-10, 'audio duration')
    for key in ('offline_feed_cpu_seconds', 'offline_feed_wall_seconds', 'cpu_seconds_per_audio_second'):
        need(type(receipt[key]) in (int, float) and math.isfinite(receipt[key]) and receipt[key] > 0, 'timing receipt')
    need(abs(receipt['cpu_seconds_per_audio_second'] - receipt['offline_feed_cpu_seconds'] / audio) <= 1e-9,
         'CPU normalization from independent receipt fields')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('continuation', 'strict', 'diagnostic', 'saved', 'output'):
        ap.add_argument('--' + key, required=True, type=Path)
    ap.add_argument('--declared-plan-sha256', required=True)
    ap.add_argument('--aggregate-sha256', required=True)
    args = ap.parse_args()
    os.umask(0o077)
    need(not args.output.exists(), 'fresh independent audit output')
    root, out, old = args.continuation, args.continuation / 'execution', args.strict / 'execution'
    need(sha(root / 'PLAN.json') == args.declared_plan_sha256, 'frozen continuation declaration')
    need(sha(out / 'AGGREGATE.json') == args.aggregate_sha256, 'root-observed final aggregate')
    declared, aggregate = load(root / 'PLAN.json'), load(out / 'AGGREGATE.json')
    preflight, plan = load(root / 'PREFLIGHT.json'), load(out / 'PLAN.json')
    terminal, numeric_pass = load(out / 'TERMINAL.json'), load(out / 'NUMERIC_PASS.json')
    need(aggregate['status'] == terminal['status'] ==
         'PASS_COMPLEMENTARY_SIX_SCORE_CONTRACT_HOST_RESOURCES_STRICT_BIT_FAIL_RETAINED', 'continuation terminal status')
    need(aggregate['declared_plan_sha256'] == plan['declared_plan_sha256'] == args.declared_plan_sha256 and
         aggregate['execution_plan_sha256'] == sha(out / 'PLAN.json') and
         aggregate['terminal_sha256'] == sha(out / 'TERMINAL.json') and
         aggregate['numeric_pass_sha256'] == sha(out / 'NUMERIC_PASS.json'), 'continuation hash chain')
    preview = dict(preflight['execution_plan_preview'])
    preview['preflight_sha256'] = sha(root / 'PREFLIGHT.json')
    need(plan == preview, 'reviewed preflight execution identity')
    for name, digest in preflight['validated_file_sha256'].items():
        need(sha(Path(name)) == digest, 'protected input/source/model/output changed: ' + name)
    for name, value in declared['source_files'].items():
        need(sha(Path(name)) == value['sha256'] and Path(name).stat().st_size == value['bytes'], 'executed source bytes')
    strict_terminal = load(old / 'TERMINAL.json')
    strict_audit = load(args.strict / 'INDEPENDENT_AUDIT.json')
    diagnostic = load(args.diagnostic / 'execution/RESULT.json')
    diagnostic_audit = load(args.diagnostic / 'INDEPENDENT_AUDIT.json')
    need(strict_terminal['status'] == 'STOPPED_ACTIVE113_PCM_VALIDATION' and
         strict_audit['original_strict_bit_gate'] == aggregate['original_strict_bit_gate'] ==
         terminal['original_strict_bit_gate'] == numeric_pass['original_strict_bit_gate'] == 'FAIL_UNCHANGED',
         'original strict failure preserved')
    need(sha(old / 'TERMINAL.json') == aggregate['original_failed_terminal_sha256'] and
         sha(args.strict / 'INDEPENDENT_AUDIT.json') == aggregate['stopped_phase_audit_sha256'] and
         sha(args.diagnostic / 'execution/RESULT.json') == aggregate['diagnostic_result_sha256'] and
         sha(args.diagnostic / 'INDEPENDENT_AUDIT.json') == aggregate['diagnostic_independent_audit_sha256'], 'prior stage binding')
    need(diagnostic_audit['status'] == 'PASS_SAVED_SINGLE_ARM_DIAGNOSTIC_AUDIT' and
         diagnostic['actual_pcm_calls'] == 1 and diagnostic['candidate_reruns'] == 0, 'single diagnostic accounting')
    for name, entry in strict_audit['private_artifact_inventory'].items():
        for directory, suffix, key in (('claims', '.json', 'claim_sha256'), ('receipts', '.json', 'receipt_sha256'),
                                      ('traces', '.txt', 'trace_sha256'), ('scores', '.r1s', 'score_sha256'),
                                      ('comparisons', '.json', 'comparison_sha256')):
            if key in entry:
                need(sha(old / directory / (name + suffix)) == entry[key], 'retained original artifact mutation')
    base_plan = plan['unchanged_probe_model_source_plan']
    need(base_plan == load(old / 'PLAN.json'), 'models/binaries/resource protocol unchanged')
    records = {r['index']: r for r in base_plan['input_records']}
    need(sorted(records) == KNOWN, 'known input denominator')
    old_arm = {load(old / 'claims' / (name + '.json'))['index'] for name in strict_terminal['claimed'] if name.startswith('capture-arm-')}
    new_indices = plan['new_arm_indices']
    need(set(new_indices).isdisjoint(old_arm) and set(new_indices) | old_arm == set(KNOWN) and
         len(new_indices) == len(set(new_indices)) == 125 and len(old_arm) == 5, 'disjoint complete ARM membership')
    expected_names = [f'new-arm-{i:03d}' for i in new_indices]
    resource_rows = []
    for stage in ('cpu', 'memory'):
        orders = base_plan['resource_orders'][stage]
        need(len(orders) == (9 if stage == 'cpu' else 6) and
             orders == [['baseline', 'active113'] if i % 2 == 0 else ['active113', 'baseline'] for i in range(len(orders))],
             'fixed paired resource order')
        for pair, order in enumerate(orders):
            for variant in order:
                name = f'{stage}-{pair:02d}-{variant}'
                expected_names.append(name)
                resource_rows.append((stage, pair, variant, name))
    need(terminal['claimed'] == expected_names and len(expected_names) == len(set(expected_names)) == 155 and
         [r['name'] for r in terminal['completed_invocations']] == expected_names, 'all disjoint calls completed in fixed order')
    completed = {r['name']: r for r in terminal['completed_invocations']}
    for directory, suffix in (('claims', '.json'), ('receipts', '.json'), ('logs', '.stderr')):
        need({p.name for p in (out / directory).iterdir()} == {name + suffix for name in expected_names}, 'exact new invocation file set')
    capture_names = {f'new-arm-{i:03d}' for i in new_indices}
    for directory, suffix in (('scores', '.r1s'), ('traces', '.txt'), ('comparisons', '.json')):
        need({p.name for p in (out / directory).iterdir()} == {name + suffix for name in capture_names}, 'capture-only output file set')
    all_comparisons = {r['index']: r for r in terminal['retained_arm_comparisons'] + terminal['new_arm_comparisons']}
    need(len(all_comparisons) == 130 and list(numeric_pass['arm_comparisons']) ==
         terminal['retained_arm_comparisons'] + terminal['new_arm_comparisons'], 'numeric-pass comparison binding')
    frame_total = event_total = 0
    score_max = hidden_max = tolerance_max = 0.0
    score_changes = hidden_changes = differing_records = 0
    private_comparisons = []
    # All 130 ARM outputs are compared to existing host outputs, never inferred again.
    for i in KNOWN:
        record = records[i]
        name = f'new-arm-{i:03d}' if i in new_indices else f'capture-arm-{i:03d}'
        evidence = out if i in new_indices else old
        host_path = old / 'scores' / (f'capture-host-{i:03d}.r1s')
        arm_path = evidence / 'scores' / (name + '.r1s')
        trace_path = evidence / 'traces' / (name + '.txt')
        baseline_path = args.saved / 'private-traces' / (f'{i:03d}-arm_par4.txt')
        trace, baseline_trace = raw(trace_path), raw(baseline_path)
        need(hashlib.sha256(baseline_trace).hexdigest() == record['baseline_trace_sha256'] and trace == baseline_trace and
             raw(old / 'traces' / (f'capture-host-{i:03d}.txt')) == baseline_trace, 'complete host/ARM/baseline F/E bytes')
        frames, count = frame_event_counts(trace)
        need(count == record['expected_counts'], 'parsed frame/event denominator')
        host = capture(raw(host_path), len(frames))
        arm = capture(raw(arm_path), len(frames))
        for key in ('frame', 'source', 'delivery', 'winner', 'other'):
            need(np.array_equal(arm[key], host[key]), 'exact clocks/global/OTHER IDs')
        for j, frame in enumerate(frames):
            need(frame == (int(arm['frame'][j]), int(arm['source'][j]), int(arm['delivery'][j]), int(arm['winner'][j])), 'F trace/binary coherence')
            ids = (*PRECISE, int(arm['other'][j]))
            scores = arm['scores'][j]
            winner = min(range(6), key=lambda k: (-float(scores[k]), ids[k]))
            need(ids[winner] == int(arm['winner'][j]), 'score/original-ID winner')
        score_metrics = numerical(arm['scores'], host['scores'], True)
        hidden_metrics = numerical(arm['hidden'], host['hidden'], False)
        comparison = all_comparisons[i]
        same_metrics(comparison['six_scores_to_host_active113'], score_metrics)
        same_metrics(comparison['hidden140_diagnostic_to_host_active113'], hidden_metrics)
        need(comparison['score_sha256'] == sha(arm_path) and comparison['trace_sha256'] == sha(trace_path) and
             comparison['host_score_sha256'] == sha(host_path) and comparison['original_strict_bit_gate'] == 'FAIL_UNCHANGED' and
             comparison['hidden140_acceptance_bound_applied'] is False, 'comparison provenance and scope')
        if i in new_indices:
            need(load(out / 'comparisons' / (name + '.json')) == comparison, 'saved comparison receipt')
        receipt = load(evidence / 'receipts' / (name + '.json'))
        receipt_checks(receipt, record, True, True, base_plan)
        frame_total += len(frames)
        event_total += count['keyword_events']
        score_max = max(score_max, score_metrics['maximum_absolute_difference'])
        hidden_max = max(hidden_max, hidden_metrics['maximum_absolute_difference'])
        tolerance_max = max(tolerance_max, score_metrics['maximum_tolerance_ratio'])
        score_changes += score_metrics['bit_different_values']
        hidden_changes += hidden_metrics['bit_different_values']
        differing_records += score_metrics['bit_different_values'] > 0
        private_comparisons.append({'index': i, 'name': name, 'score_sha256': sha(arm_path),
            'trace_sha256': sha(trace_path), 'six_scores': score_metrics, 'hidden_diagnostic': hidden_metrics})
    need(frame_total == 12255 and event_total == 86, 'full known functional totals')
    cpu = {'baseline': [], 'active113': []}
    p95 = {'baseline': [], 'active113': []}
    metrics = ('proc_self_pss_kib', 'proc_self_private_dirty_kib')
    memory = {variant: {metric: [] for metric in metrics} for variant in cpu}
    resource_inventory = []
    for name in expected_names:
        claim = load(out / 'claims' / (name + '.json'))
        i = claim['index']
        record = records[i]
        need(claim['name'] == name and claim['attempt'] == 1 and
             claim['execution_plan_sha256'] == sha(out / 'PLAN.json') and
             claim['expected_counts'] == record['expected_counts'] and claim['pcm_sha256'] == record['pcm_sha256'], 'new claim identity')
        need(raw(out / 'logs' / (name + '.stderr')) == b'', 'child stderr')
        receipt_path = out / 'receipts' / (name + '.json')
        receipt = load(receipt_path)
        need(completed[name]['receipt_sha256'] == sha(receipt_path), 'completed invocation receipt')
        if name in capture_names:
            argv = [base_plan['qemu']['path'], '-L', base_plan['sysroot'],
                    base_plan['active_probes']['arm']['binaries']['capture']['path'],
                    base_plan['active_model']['path'], base_plan['frontend']['path'], record['pcm_path'], '--mmap']
            need(claim['scope'] == 'previously_unattempted_arm_capture' and i in new_indices, 'new ARM scope')
            need(completed[name]['score_sha256'] == sha(out / 'scores' / (name + '.r1s')) and
                 completed[name]['trace_sha256'] == sha(out / 'traces' / (name + '.txt')), 'new capture complete binding')
        else:
            stage, _pair, variant = name.split('-', 2)
            active = variant == 'active113'
            need(i == 49 and claim['scope'] == 'host_plain_' + stage and claim['trace'] is None and claim['scores'] is None,
                 'plain fixed resource input scope')
            binary = base_plan['active_probes']['host']['binaries']['plain']['path'] if active else base_plan['baseline_probe']['path']
            model = base_plan['active_model']['path'] if active else base_plan['baseline_model']['path']
            argv = [binary, model, base_plan['frontend']['path'], record['pcm_path'], '--mmap']
            receipt_checks(receipt, record, active, False, base_plan)
            if stage == 'cpu':
                cpu[variant].append(receipt['cpu_seconds_per_audio_second'])
                p95[variant].append(receipt['feed_p95_histogram_upper_us'])
            else:
                for metric in metrics:
                    value = receipt[metric]
                    need(value is None or (type(value) is int and value >= 0), 'own-process memory metric')
                    memory[variant][metric].append(value)
            resource_inventory.append({'name': name, 'claim_sha256': sha(out / 'claims' / (name + '.json')),
                                       'receipt_sha256': sha(receipt_path)})
        need(claim['argv'] == argv, 'actual executable/model/PCM argument binding')
    savings = [100 * (1 - active / baseline) for baseline, active in zip(cpu['baseline'], cpu['active113'])]
    cpu_medians = {k: statistics.median(v) for k, v in cpu.items()}
    need(terminal['host_cpu'] == aggregate['host_cpu'] and terminal['host_memory'] == aggregate['host_memory'], 'resource terminal/aggregate consistency')
    saved_cpu = aggregate['host_cpu']
    need(saved_cpu['cpu_seconds_per_audio_second'] == cpu and saved_cpu['feed_p95_histogram_upper_us'] == p95 and
         saved_cpu['median_cpu_seconds_per_audio_second'] == cpu_medians and saved_cpu['paired_cpu_savings_pct'] == savings and
         saved_cpu['paired_median_cpu_savings_pct'] == statistics.median(savings) and
         saved_cpu['active_wins'] == sum(a < b for b, a in zip(cpu['baseline'], cpu['active113'])) and saved_cpu['warmup_calls'] == 0,
         'independent paired CPU recomputation')
    memory_medians = {variant: {metric: statistics.median(values) if all(x is not None for x in values) else None
        for metric, values in by.items()} for variant, by in memory.items()}
    deltas = {metric: [b - a for b, a in zip(memory['baseline'][metric], memory['active113'][metric])]
              if all(x is not None for v in memory for x in memory[v][metric]) else None for metric in metrics}
    delta_medians = {metric: statistics.median(values) if values is not None else None for metric, values in deltas.items()}
    saved_memory = aggregate['host_memory']
    need(saved_memory['own_process_observations_kib'] == memory and saved_memory['own_process_medians_kib'] == memory_medians and
         saved_memory['paired_baseline_minus_active113_kib'] == deltas and
         saved_memory['median_baseline_minus_active113_kib'] == delta_medians and
         saved_memory['peak_rss_used_for_savings'] is False and saved_memory['memory_pair_timing_used_for_speedup'] is False,
         'independent own-process memory recomputation')
    need(aggregate['six_score_max_abs_difference'] == score_max and aggregate['six_score_bit_different_values'] == score_changes and
         aggregate['hidden140_diagnostic_max_abs_difference'] == hidden_max and aggregate['hidden140_bit_different_values'] == hidden_changes,
         'full numerical aggregate recomputation')
    need(aggregate['hidden140_acceptance_bound_applied'] is False and
         aggregate['full_2601_logits'] == aggregate['posterior_parity'] == 'NOT_TESTED' and
         aggregate['physical_ssc305'] == aggregate['independent_far_frr'] == 'NOT_RUN' and
         aggregate['model_promotion_authorized'] is False and aggregate['default_model_changed'] is False,
         'acceptance and qualification boundaries')
    need(aggregate['retained_candidate_calls'] == 135 and aggregate['diagnostic_baseline_calls'] == 1 and
         aggregate['new_arm_calls'] == 125 and aggregate['host_cpu_calls'] == 18 and aggregate['host_memory_calls'] == 12 and
         aggregate['new_pcm_calls'] == 155 and aggregate['cumulative_pcm_calls'] == 291 and
         aggregate['unknown_calls'] == aggregate['training'] == aggregate['candidate_reruns'] == 0, 'all-stage accounting')
    result = {
        'schema': 'active113-independent-complementary-audit-v1',
        'status': 'PASS_SAVED_COMPLEMENTARY_AUDIT_STRICT_BIT_FAIL_RETAINED',
        'audit_source_sha256': sha(Path(__file__)),
        'declared_plan_sha256': args.declared_plan_sha256, 'aggregate_sha256': args.aggregate_sha256,
        'preflight_sha256': sha(root / 'PREFLIGHT.json'), 'execution_plan_sha256': sha(out / 'PLAN.json'),
        'terminal_sha256': sha(out / 'TERMINAL.json'), 'numeric_pass_sha256': sha(out / 'NUMERIC_PASS.json'),
        'original_failed_terminal_sha256': aggregate['original_failed_terminal_sha256'],
        'stopped_phase_audit_sha256': aggregate['stopped_phase_audit_sha256'],
        'diagnostic_result_sha256': aggregate['diagnostic_result_sha256'],
        'diagnostic_independent_audit_sha256': aggregate['diagnostic_independent_audit_sha256'],
        'model_sha256': aggregate['model_sha256'], 'selection_sha256': aggregate['selection_sha256'],
        'public_summary': {
            'known_inputs_per_architecture': 130, 'frames_per_architecture': frame_total, 'events_per_architecture': event_total,
            'complete_fe_clocks_global_ids_other_ids_equal': True,
            'retained_candidate_calls': 135, 'new_candidate_arm_calls': 125, 'diagnostic_baseline_calls': 1,
            'host_cpu_calls': 18, 'host_memory_calls': 12, 'new_pcm_calls': 155, 'cumulative_pcm_calls': 291,
            'repeated_candidate_functional_calls': 0, 'unknown_calls': 0, 'training': 0,
            'six_score_contract_passed': True, 'six_score_max_abs_difference': score_max,
            'six_score_max_tolerance_ratio': tolerance_max, 'six_score_bit_different_values': score_changes,
            'six_score_bit_different_records': differing_records,
            'hidden140_diagnostic_max_abs_difference': hidden_max, 'hidden140_bit_different_values': hidden_changes,
            'hidden140_acceptance_bound_applied': False, 'original_strict_bit_gate': 'FAIL_UNCHANGED',
            'full_2601_logits': 'NOT_TESTED', 'posterior_parity': 'NOT_TESTED',
            'model_bytes': aggregate['model_bytes'], 'baseline_model_bytes': aggregate['original_model_bytes'],
            'model_file_reduction_bytes': aggregate['original_model_bytes'] - aggregate['model_bytes'],
            'model_file_reduction_percent': 100 * (1 - aggregate['model_bytes'] / aggregate['original_model_bytes']),
            'host_cpu': {'pairs': 9, 'median_cpu_seconds_per_audio_second': cpu_medians,
                'paired_median_cpu_savings_percent': statistics.median(savings),
                'paired_cpu_savings_percent_range': [min(savings), max(savings)],
                'active_wins': saved_cpu['active_wins'], 'single_fixed_input': True, 'capture_qemu_timings_used': False},
            'host_memory': {'pairs': 6, 'own_process_medians_kib': memory_medians,
                'median_baseline_minus_active113_kib': delta_medians, 'peak_rss_used': False},
            'physical_ssc305': 'NOT_RUN', 'independent_far_frr': 'NOT_RUN', 'model_promotion_authorized': False,
        },
        'private_arm_comparisons': private_comparisons,
        'private_resource_artifact_inventory': resource_inventory,
        'protected_file_posthash_count': len(preflight['validated_file_sha256']),
        'all_protected_identities_unchanged': True,
        'reviewer_execution': {'model_calls': 0, 'head_calls': 0, 'detector_calls': 0, 'pcm_calls': 0,
            'training': 0, 'compilation': 0, 'saved_arrays_and_receipts_only': True},
        'interpretation': 'The separately preregistered six-score/F-E contract passed on the retained set. The original strict bit gate remains failed. Resource observations describe one fixed host input with plain probes; no physical-device, independent FAR/FRR, posterior or full-vocabulary-logit qualification is inferred.',
        'assertions': ASSERTIONS,
    }
    with args.output.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps({'status': result['status'], 'audit_sha256': sha(args.output),
        'audit_source_sha256': result['audit_source_sha256'], 'assertions': result['assertions'],
        'public_summary': result['public_summary']}, sort_keys=True))


if __name__ == '__main__':
    main()
