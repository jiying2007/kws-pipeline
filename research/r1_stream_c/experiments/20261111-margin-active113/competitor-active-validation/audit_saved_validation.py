#!/usr/bin/env python3
"""Audit a stopped ACTIVE113 validation using saved bytes only.

This source never executes a network, head, detector, PCM probe, compiler or
numeric reference. It does not reinterpret the stopped bit-exact gate as PASS.
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
import struct

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np


PRECISE = (0, 1027, 1401, 1462, 2093)
KNOWN = list(range(96)) + list(range(100, 134))
PINS = {
    'DECLARED_PLAN.json': 'a02e36d066526b3f99a2f93296d3debe370760f31ef00b2870f0949f08077e8d',
    'build/BUILD_RECEIPT.json': '0ee8394dea3319f00e7f361d829f8a0a13b20bbed9ddb0d8b5ee6df422bc1391',
    'PREFLIGHT.json': 'b0f299a501828585c4f1b45202fe40347a816d2f2a13d7214375b03a0a3a50c9',
    'execution/PLAN.json': '8c0162fb58c0ad53d49f70e54df9e70f05206cee69137bc019441073be6a1bc4',
    'execution/TERMINAL.json': '8f36a97bfde30ac8eba3fb3a1ec84ba725e49107b29e05d2cb4c1448725896a7',
}
CAPTURE_DTYPE = np.dtype([
    ('frame', '<u8'), ('source', '<u8'), ('delivery', '<u8'),
    ('best', '<u4'), ('other_id', '<u4'),
    ('hidden', '<f8', (140,)), ('scores', '<f8', (6,)),
])
PREDICTION_DTYPE = np.dtype([
    ('frame', '<u8'), ('source', '<u8'), ('delivery', '<u8'),
    ('other_id', '<u4'), ('reserved', '<u4'), ('other', '<f8'),
])
ASSERTIONS = 0


def need(ok, message):
    global ASSERTIONS
    ASSERTIONS += 1
    if not ok:
        raise ValueError(message)


def raw(path):
    need(stat.S_ISREG(path.lstat().st_mode), 'nonregular evidence: ' + str(path))
    return path.read_bytes()


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def load(path):
    def bad(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw(path), parse_constant=bad)


def trace_rows(data):
    need(data.endswith(b'\n'), 'trace missing terminal newline')
    frames, events = [], []
    for line in data.decode('ascii').splitlines():
        row = line.split(',')
        need(row[0] in ('F', 'E'), 'trace row type')
        values = tuple(map(int, row[1:]))
        need(all(value >= 0 for value in values), 'negative trace field')
        if row[0] == 'F':
            need(len(values) == 4 and values[0] == len(frames) and
                 values[1] <= values[2] and values[3] < 2601, 'frame trace shape/order')
            frames.append(values)
        else:
            need(len(values) == 6 and values[0] == 0 and values[1] in (1, 2) and
                 values[3] <= values[2] < len(frames) and values[4] <= values[5], 'event trace shape')
            events.append(values)
    return frames, events


def numeric_difference(actual, reference):
    a = np.ascontiguousarray(actual, dtype='<f8')
    r = np.ascontiguousarray(reference, dtype='<f8')
    if a.ndim == 1:
        a, r = a[:, None], r[:, None]
    need(a.shape == r.shape and np.isfinite(a).all() and np.isfinite(r).all(), 'finite comparison shape')
    changed = a.view('<u8') != r.view('<u8')
    absolute = np.abs(a - r)
    locations = np.argwhere(changed)
    return {
        'values': int(a.size), 'frames': len(a),
        'bit_different_values': int(changed.sum()),
        'bit_different_frames': int(changed.any(axis=1).sum()),
        'maximum_absolute_difference': float(absolute.max(initial=0)),
        'first_different_location': locations[0].tolist() if len(locations) else None,
        'last_different_location': locations[-1].tolist() if len(locations) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('validation', 'teacher', 'screen', 'saved', 'output'):
        ap.add_argument('--' + name, required=True, type=Path)
    args = ap.parse_args()
    os.umask(0o077)
    need(not args.output.exists(), 'exclusive output required')
    out = args.validation / 'execution'
    for relative, digest in PINS.items():
        need(sha(args.validation / relative) == digest, 'stopped evidence binding: ' + relative)
    declared = load(args.validation / 'DECLARED_PLAN.json')
    preflight = load(args.validation / 'PREFLIGHT.json')
    plan = load(out / 'PLAN.json')
    terminal = load(out / 'TERMINAL.json')
    need(terminal['status'] == 'STOPPED_ACTIVE113_PCM_VALIDATION' and terminal['stage'] == 'FUNCTIONAL',
         'stopped functional status required')
    need(terminal['execution_plan_sha256'] == PINS['execution/PLAN.json'] and
         plan['declaration_sha256'] == PINS['DECLARED_PLAN.json'] and
         plan['preflight_sha256'] == PINS['PREFLIGHT.json'], 'stopped plan chain')
    preview = dict(preflight['execution_plan_preview'])
    preview['preflight_sha256'] = PINS['PREFLIGHT.json']
    need(preview == plan, 'preflight/execution plan drift')
    need(terminal['host_cpu'] is None and terminal['host_memory'] is None and
         not (out / 'FUNCTIONAL_PASS.json').exists() and not (out / 'AGGREGATE.json').exists(),
         'stopped gate must not have resource or success aggregate')
    for path, digest in preflight['validated_file_sha256'].items():
        need(sha(Path(path)) == digest, 'protected evidence changed: ' + path)
    # Identity-only hashing above may read PCM bytes; no audio processing occurs.
    records = {r['index']: r for r in plan['input_records']}
    need(sorted(records) == KNOWN, 'known denominator')
    ordered = [f'capture-{arch}-{i:03d}' for arch in ('host', 'arm') for i in KNOWN]
    claimed = terminal['claimed']
    need(claimed == ordered[:len(claimed)] and len(claimed) == len(set(claimed)) == 135, 'exact stopped call prefix')
    completed = {r['name']: r for r in terminal['completed_invocations']}
    comparisons = {r['name']: r for r in terminal['functional']}
    need(list(completed) == claimed[:-1] and list(comparisons) == claimed[:-1] and len(completed) == 134,
         'completed comparison prefix')
    suffixes = {'claims': '.json', 'receipts': '.json', 'logs': '.stderr', 'traces': '.txt', 'scores': '.r1s'}
    for directory, suffix in suffixes.items():
        need({p.name for p in (out / directory).iterdir()} == {name + suffix for name in claimed},
             'exact attempted file set: ' + directory)
    need({p.name for p in (out / 'comparisons').iterdir()} == {name + '.json' for name in completed},
         'comparison files include failed or unattempted call')
    extraction = load(args.teacher / 'extraction/RESULT.json')
    extraction_plan = load(args.teacher / 'extraction/PLAN.json')
    hidden_rows = {r['index']: r for r in extraction['records']}
    predicted = load(args.screen / 'execution/PREDICTIONS.json')
    predicted_rows = {r['index']: r for r in predicted['records']}
    trace_manifest = load(args.saved / 'FILE_MANIFEST.json')
    need('/host/' in extraction_plan['static_sdk']['path'] and
         'x86_64' in extraction_plan['compiler']['resolved_path'], 'numerical reference provenance')
    groups = {arch: {'executed_calls': 0, 'completed_bit_comparisons': 0, 'frames': 0,
                    'events': 0, 'complete_fe_byte_equal_calls': 0,
                    'clock_id_exact_calls': 0, 'hidden_bit_equal_calls': 0,
                    'precise_bit_equal_calls': 0, 'selected_other_bit_equal_calls': 0}
              for arch in ('host', 'arm')}
    private_failures, artifact_inventory = [], {}
    for name in claimed:
        claim = load(out / 'claims' / (name + '.json'))
        i, arch = claim['index'], claim['architecture']
        need(i in records and arch in groups and name == f'capture-{arch}-{i:03d}', 'claimed identity')
        record = records[i]
        argv = [plan['active_probes'][arch]['binaries']['capture']['path'], plan['active_model']['path'],
                plan['frontend']['path'], record['pcm_path'], '--mmap']
        if arch == 'arm':
            argv = [plan['qemu']['path'], '-L', plan['sysroot'], *argv]
        need(claim['argv'] == argv and claim['attempt'] == 1 and
             claim['scope'] == 'candidate_functional_capture' and
             claim['execution_plan_sha256'] == PINS['execution/PLAN.json'] and
             claim['expected_counts'] == record['expected_counts'] and
             claim['pcm_sha256'] == record['pcm_sha256'] and claim['timeout_seconds'] == 30,
             'claim protocol')
        receipt_path = out / 'receipts' / (name + '.json')
        receipt = load(receipt_path)
        need(raw(out / 'logs' / (name + '.stderr')) == b'', 'nonempty stderr')
        need(receipt['schema'] == 'r1-active113-offline-resource-v1' and
             receipt['model_sha256'] == plan['active_model']['sha256'] and
             receipt['frontend_sha256'] == plan['frontend']['sha256'] and
             receipt['selection_sha256'] == plan['selection_sha256'] and receipt['capture_instrumented'] is True,
             'actual execution receipt identity')
        need(receipt['model_storage'] == 'read_only_mmap' and receipt['model_caller_heap_bytes'] == 0 and
             receipt['model_weight_bytes'] == 1582624 and receipt['pcm_bytes'] == record['pcm_bytes'] and
             receipt['feed_chunk_bytes'] == 640 and receipt['feed_calls'] == (record['pcm_bytes'] + 639) // 640,
             'actual feed/storage receipt')
        need(all(receipt[key] == value for key, value in record['expected_counts'].items()), 'actual complete input counts')
        need(all(receipt[key] is False for key in ('device_qualified', 'live_afe', 'independent_far_frr')),
             'scope boundary')
        for key in ('offline_feed_cpu_seconds', 'offline_feed_wall_seconds'):
            need(isinstance(receipt[key], (int, float)) and math.isfinite(receipt[key]) and receipt[key] > 0,
                 'complete receipt timing')
        trace_path = out / 'traces' / (name + '.txt')
        score_path = out / 'scores' / (name + '.r1s')
        need(claim['trace'] == str(trace_path) and claim['scores'] == str(score_path), 'claimed output paths')
        trace = raw(trace_path)
        baseline_name = f'private-traces/{i:03d}-arm_par4.txt'
        baseline = raw(args.saved / baseline_name)
        need(hashlib.sha256(baseline).hexdigest() == trace_manifest[baseline_name]['sha256'] == record['baseline_trace_sha256'],
             'retained event reference identity')
        need(trace == baseline, 'saved F/E bytes differ')
        frames, events = trace_rows(trace)
        counts = Counter(row[1] for row in events)
        need({'sdk_model_frames': len(frames), 'keyword_events': len(events),
              'keyword1_events': counts[1], 'keyword2_events': counts[2]} == record['expected_counts'], 'parsed F/E counts')
        data = raw(score_path)
        hidden_path = args.teacher / 'extraction' / hidden_rows[i]['path']
        pred_path = args.screen / 'execution' / predicted_rows[i]['path']
        hidden_data, pred_data = raw(hidden_path), raw(pred_path)
        need(hashlib.sha256(hidden_data).hexdigest() == hidden_rows[i]['sha256'] == record['hidden_sha256'] and
             hashlib.sha256(pred_data).hexdigest() == predicted_rows[i]['sha256'] == record['cached_prediction_sha256'],
             'numerical reference hashes')
        frame_count = len(frames)
        need(data[:64] == struct.pack('<8s6I5IIQ', b'R1A113H\0', 1, 64, 1200, 140, 6, 2601, *PRECISE, 113, 0) and
             len(data) == len(hidden_data) == 64 + frame_count * 1200 and
             len(pred_data) == 64 + frame_count * 40, 'capture/reference lengths')
        actual = np.frombuffer(data, dtype=CAPTURE_DTYPE, offset=64)
        hidden = np.frombuffer(hidden_data, dtype=CAPTURE_DTYPE, offset=64)
        pred = np.frombuffer(pred_data, dtype=PREDICTION_DTYPE, offset=64)
        for key in ('frame', 'source', 'delivery'):
            need(np.array_equal(actual[key], hidden[key]) and np.array_equal(actual[key], pred[key]), 'clock/reference field')
        need(np.array_equal(actual['best'], hidden['best']) and np.array_equal(actual['other_id'], pred['other_id']), 'original IDs')
        for j, row in enumerate(frames):
            need(row == (int(actual['frame'][j]), int(actual['source'][j]), int(actual['delivery'][j]), int(actual['best'][j])),
                 'binary/F trace coherence')
            ids = (*PRECISE, int(actual['other_id'][j]))
            scores = actual['scores'][j]
            need(all(math.isfinite(float(x)) for x in scores), 'finite actual score')
            winner = min(range(6), key=lambda k: (-float(scores[k]), ids[k]))
            need(ids[winner] == int(actual['best'][j]), 'actual score/original-ID tie policy')
        differences = {
            'hidden140': numeric_difference(actual['hidden'], hidden['hidden']),
            'precise5': numeric_difference(actual['scores'][:, :5], hidden['scores'][:, :5]),
            'selected_other': numeric_difference(actual['scores'][:, 5], pred['other']),
        }
        all_bit_equal = all(value['bit_different_values'] == 0 for value in differences.values())
        artifacts = {'claim_sha256': sha(out / 'claims' / (name + '.json')),
                     'receipt_sha256': sha(receipt_path), 'trace_sha256': sha(trace_path), 'score_sha256': sha(score_path)}
        if name in completed:
            need(all_bit_equal, 'accepted comparison must be bit exact')
            comparison = load(out / 'comparisons' / (name + '.json'))
            need(comparison == comparisons[name] and comparison['receipt_sha256'] == artifacts['receipt_sha256'] and
                 comparison['trace_sha256'] == artifacts['trace_sha256'] and comparison['score_trace_sha256'] == artifacts['score_sha256'],
                 'accepted comparison hash binding')
            need(completed[name]['receipt_sha256'] == artifacts['receipt_sha256'] and
                 completed[name]['trace_sha256'] == artifacts['trace_sha256'] and
                 completed[name]['score_trace_sha256'] == artifacts['score_sha256'], 'completed receipt binding')
            need(all(comparison[key] is True for key in ('fe_trace_byte_equal', 'hidden140_bit_equal',
                 'five_precise_logits_bit_equal', 'selected_other_score_bit_equal', 'selected_other_id_equal', 'argmax_ids_equal')),
                 'accepted comparison flags')
            artifacts['comparison_sha256'] = sha(out / 'comparisons' / (name + '.json'))
        else:
            need(not all_bit_equal and name == claimed[-1] and arch == 'arm' and
                 differences['hidden140']['bit_different_values'] > 0 and
                 terminal['failure'] == 'ValueError: hidden140 bits changed: ' + name, 'actual stop cause')
            host_data = raw(out / 'scores' / (f'capture-host-{i:03d}.r1s'))
            host = np.frombuffer(host_data, dtype=CAPTURE_DTYPE, offset=64)
            private_failures.append({'name': name, 'index': i, 'split': record['split'], 'frames': frame_count,
                'complete_actual_output_present': True, 'fe_bytes_equal': True, 'clocks_ids_equal': True,
                'runner_completion_record_written': False, 'still_counts_as_executed_call': True,
                'differences_to_host_reference': differences,
                'all_six_scores_to_host_active113': numeric_difference(actual['scores'], host['scores']),
                'artifacts': artifacts})
        group = groups[arch]
        group['executed_calls'] += 1
        group['completed_bit_comparisons'] += name in completed
        group['frames'] += frame_count
        group['events'] += len(events)
        group['complete_fe_byte_equal_calls'] += 1
        group['clock_id_exact_calls'] += 1
        for key, section in (('hidden_bit_equal_calls', 'hidden140'), ('precise_bit_equal_calls', 'precise5'),
                             ('selected_other_bit_equal_calls', 'selected_other')):
            group[key] += differences[section]['bit_different_values'] == 0
        artifact_inventory[name] = artifacts
    need(len(private_failures) == 1 and groups['host']['executed_calls'] == 130 and
         groups['host']['frames'] == 12255 and groups['host']['events'] == 86 and
         groups['arm']['executed_calls'] == 5, 'stopped aggregate denominators')
    need(sum(group['executed_calls'] for group in groups.values()) == 135 and
         sum(group['completed_bit_comparisons'] for group in groups.values()) == 134, 'executed/completed distinction')
    failure = private_failures[0]
    difference_summary = {key: {field: value[field] for field in ('bit_different_values', 'bit_different_frames',
        'maximum_absolute_difference')} for key, value in failure['differences_to_host_reference'].items()}
    receipt = {
        'schema': 'active113-independent-stopped-validation-audit-v1',
        'status': 'PASS_AUDIT_OF_STOPPED_VALIDATION',
        'validation_status': terminal['status'], 'original_strict_bit_gate': 'FAIL_UNCHANGED',
        'audit_source_sha256': sha(Path(__file__)), 'evidence_sha256': PINS,
        'model_sha256': plan['active_model']['sha256'], 'frontend_sha256': plan['frontend']['sha256'],
        'selection_sha256': plan['selection_sha256'], 'cached_screen_sha256': plan['cached_screen_sha256'],
        'groups': groups,
        'public_summary': {
            'executed_pcm_calls': 135, 'successful_bit_comparisons': 134,
            'failed_comparison_calls_still_counted_as_executed': 1,
            'planned_capture_calls': 260, 'unattempted_capture_calls': 125,
            'host_complete_fe_equal_inputs': 130, 'arm_complete_fe_equal_inputs': 5,
            'executed_frames': sum(group['frames'] for group in groups.values()),
            'host_cpu_resource_calls': 0, 'host_memory_resource_calls': 0,
            'unattempted_resource_calls': 30, 'unknown_calls': 0, 'baseline_functional_calls': 0,
            'failed_arm_output_difference_summary': difference_summary,
            'all_six_scores_maximum_absolute_difference': failure['all_six_scores_to_host_active113']['maximum_absolute_difference'],
            'strict_cross_architecture_bit_gate_passed': False,
        },
        'reference_provenance': {
            'extraction_plan_sha256': sha(args.teacher / 'extraction/PLAN.json'),
            'compiler': extraction_plan['compiler'], 'static_sdk': extraction_plan['static_sdk'],
            'architecture': 'x86_64 host', 'numeric_precision': 'binary64',
            'arm_hidden_reference_in_original_extraction': False,
        },
        'causal_interpretation': {
            'established': 'The frozen comparator used host numerical references for ARM output. The first failed captured output already differs at pre-head hidden140; all clocks, winner IDs and complete F/E bytes in the 135 executed calls remain equal.',
            'not_established': 'Saved stage19/output values alone do not isolate frontend libm rounding, backbone arithmetic rounding, or an ARM-specific integration defect. No tolerance gate is substituted for the failed bit gate.',
            'consistent_hypothesis': 'Tiny pre-head differences are consistent with inherited cross-architecture floating-point behavior; a separate same-architecture baseline diagnostic is needed for causal attribution.',
        },
        'private_failed_invocations': private_failures,
        'private_artifact_inventory': artifact_inventory,
        'protected_file_posthash_count': len(preflight['validated_file_sha256']),
        'all_protected_sources_models_inputs_references_unchanged': True,
        'reviewer_execution': {'model_calls': 0, 'neural_head_calls': 0, 'detector_calls': 0,
            'pcm_probe_calls': 0, 'training': 0, 'compilation': 0, 'saved_arrays_only': True,
            'pcm_bytes_read_for_identity_hash_only': True},
        'resource_claims': 'No plain CPU or memory measurements were executed. Instrumented capture/QEMU receipts are not used to estimate resource savings.',
        'default_model_changed': False, 'model_promotion_authorized': False,
        'physical_ssc305': 'NOT_RUN', 'independent_far_frr': 'NOT_RUN',
        'full_2601_logits': 'NOT_TESTED', 'assertions': ASSERTIONS,
    }
    with args.output.open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps({'status': receipt['status'], 'audit_sha256': sha(args.output),
        'audit_source_sha256': receipt['audit_source_sha256'], 'assertions': receipt['assertions'],
        'public_summary': receipt['public_summary'], 'model_calls': 0, 'pcm_probe_calls': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
