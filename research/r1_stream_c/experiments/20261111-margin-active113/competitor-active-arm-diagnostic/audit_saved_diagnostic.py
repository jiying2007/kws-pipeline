#!/usr/bin/env python3
"""Independent byte-only audit of the single retained ARM baseline diagnostic."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct

PINS = {
    'PLAN.json': '26a9393d2d3a9a38af46e48069c27b90b468da6c62689da21f600a6810e0fe44',
    'diagnostic.py': 'c0acf3b2a85221e5cbc1c4a87a9390aac7e80f3029d5ea3671995f53eee80069',
    'build/BUILD_RECEIPT.json': '55941a9edd6236ef073ffe7c8a3cfbee9081befdbf585cf23524a4c8c501ad87',
    'execution/TERMINAL.json': 'e21dfedabd21a18f8952bdabf14ee23a7e72db5e50d132c7680a435a629ffadc',
    'execution/RESULT.json': '037113bf0e27364a96354b7415c0d466148db88f5684c2e9aede3458f1d35bc6',
}
CHECKS = 0


def need(ok, message):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise ValueError(message)


def raw(path):
    need(stat.S_ISREG(path.lstat().st_mode), 'regular evidence required')
    return path.read_bytes()


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def load(path):
    def bad(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw(path), parse_constant=bad)


def rows(path, active, frame_count):
    data = raw(path)
    magic, scores, other = (b'R1A113H\0', 6, 113) if active else (b'R1HD140\0', 5, 1)
    need(data[:64] == struct.pack('<8s6I5IIQ', magic, 1, 64, 1200, 140, scores, 2601,
                                  0, 1027, 1401, 1462, 2093, other, 0), 'numeric header')
    need(len(data) == 64 + frame_count * 1200, 'complete numeric capture')
    records = [data[64 + i * 1200:64 + (i + 1) * 1200] for i in range(frame_count)]
    for i, row in enumerate(records):
        need(struct.unpack_from('<Q', row)[0] == i, 'frame sequence')
        need(all(math.isfinite(value[0]) for value in struct.iter_unpack('<d', row[32:])), 'finite values')
    return records


def compare(left, right, offset, count):
    need(len(left) == len(right), 'comparison denominator')
    different_values = different_frames = max_ulp = 0
    max_abs = sum_abs = 0.0
    for a, b in zip(left, right):
        need(a[:24] == b[:24], 'frame/source/delivery clocks')
        any_change = False
        for start in range(offset, offset + count * 8, 8):
            x, y = a[start:start + 8], b[start:start + 8]
            if x == y:
                continue
            any_change = True
            different_values += 1
            absolute = abs(struct.unpack('<d', x)[0] - struct.unpack('<d', y)[0])
            max_abs = max(max_abs, absolute)
            sum_abs += absolute
            xb, yb = int.from_bytes(x, 'little'), int.from_bytes(y, 'little')
            xo = ((1 << 64) - 1 - xb) if xb & (1 << 63) else xb + (1 << 63)
            yo = ((1 << 64) - 1 - yb) if yb & (1 << 63) else yb + (1 << 63)
            max_ulp = max(max_ulp, abs(xo - yo))
        different_frames += any_change
    return {'frames': len(left), 'values': len(left) * count, 'different_values': different_values,
            'different_frames': different_frames, 'byte_exact': different_values == 0,
            'max_absolute_error': max_abs, 'mean_absolute_error': sum_abs / (len(left) * count),
            'max_ulp_distance': max_ulp}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diagnostic', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    os.umask(0o077)
    need(not args.output.exists(), 'exclusive audit output')
    root = args.diagnostic
    for path, digest in PINS.items():
        need(sha(root / path) == digest, 'frozen diagnostic identity: ' + path)
    plan, build = load(root / 'PLAN.json'), load(root / 'build/BUILD_RECEIPT.json')
    result, terminal = load(root / 'execution/RESULT.json'), load(root / 'execution/TERMINAL.json')
    claim = load(root / 'execution/CLAIM.json')
    receipt = load(root / 'execution/receipt.json')
    need(plan['max_pcm_calls'] == 1 and plan['retry'] is False and plan['process_timeout_seconds'] == 30,
         'fixed one-call protocol')
    for name, digest in plan['source_sha256'].items():
        need(sha(root / name) == digest, 'source binding')
    for binding in plan['bindings'].values():
        need(sha(Path(binding['path'])) == binding['sha256'], 'retained reference binding')
    need(sha(Path(build['binary']['path'])) == build['binary']['sha256'] and
         build['plan_sha256'] == claim['plan_sha256'] == terminal['plan_sha256'] == PINS['PLAN.json'] and
         claim['build_receipt_sha256'] == terminal['build_receipt_sha256'] == PINS['build/BUILD_RECEIPT.json'], 'build/claim chain')
    need(claim['attempt'] == claim['max_pcm_calls'] == terminal['pcm_calls_claimed'] == terminal['pcm_calls_completed'] == 1 and
         terminal['returncode'] == 0 and raw(root / 'execution/stderr.log') == b'', 'single successful baseline invocation')
    need(result['candidate_reruns'] == result['resource_calls'] == result['training'] == 0 and
         terminal['candidate_reruns'] == terminal['resource_calls'] == 0 and
         result['strict_cross_arch_gate'] == terminal['strict_cross_arch_gate'] == 'FAILED_UNCHANGED', 'unchanged strict-gate scope')
    need(receipt['frames'] == receipt['teacher_steps'] == plan['frames'] and
         receipt['pcm_bytes'] == plan['pcm']['bytes'], 'single-forward complete capture counts')
    need(sha(root / 'execution/baseline_arm.r1h') == terminal['baseline_hidden_sha256'] and
         sha(root / 'execution/baseline_arm.txt') == terminal['baseline_trace_sha256'] and
         sha(root / 'execution/receipt.json') == terminal['receipt_sha256'] and
         sha(root / 'execution/CLAIM.json') == result['claim_sha256'], 'saved output binding')
    bindings = plan['bindings']
    baseline = rows(root / 'execution/baseline_arm.r1h', False, plan['frames'])
    arm = rows(Path(bindings['candidate_arm_scores']['path']), True, plan['frames'])
    teacher = rows(Path(bindings['host_teacher_hidden']['path']), False, plan['frames'])
    host = rows(Path(bindings['candidate_host_scores']['path']), True, plan['frames'])
    comparison = {}
    for name, left, right in (
        ('baseline_arm_vs_candidate_arm', baseline, arm),
        ('baseline_arm_vs_host_teacher', baseline, teacher),
        ('candidate_host_vs_host_teacher', host, teacher),
    ):
        comparison[name] = {'hidden140': compare(left, right, 32, 140), 'precise5': compare(left, right, 1152, 5)}
    need(comparison == result['comparisons'] == terminal['comparisons'], 'independent numeric summary recomputation')
    need(all(value['byte_exact'] for value in comparison['baseline_arm_vs_candidate_arm'].values()), 'same-ARM hidden/precise parity')
    need(all(value['byte_exact'] for value in comparison['candidate_host_vs_host_teacher'].values()), 'same-host hidden/precise parity')
    need(any(not value['byte_exact'] for value in comparison['baseline_arm_vs_host_teacher'].values()), 'inherited cross-architecture difference')
    need(raw(root / 'execution/baseline_arm.txt') == raw(Path(bindings['candidate_arm_trace']['path'])) ==
         raw(Path(bindings['saved_original_trace']['path'])), 'same-ARM complete F/E bytes')
    need(all(a[24:28] == b[24:28] for a, b in zip(baseline, arm)), 'same-ARM global winner IDs')
    audited = {
        'schema': 'active113-independent-same-arm-diagnostic-audit-v1',
        'status': 'PASS_SAVED_SINGLE_ARM_DIAGNOSTIC_AUDIT', 'audit_source_sha256': sha(Path(__file__)),
        'evidence_sha256': PINS, 'comparisons': comparison,
        'public_summary': {'actual_baseline_pcm_calls': 1, 'candidate_reruns': 0,
            'same_arm_hidden140_bit_equal': True, 'same_arm_precise5_bit_equal': True,
            'same_arm_fe_byte_equal': True, 'same_arm_global_ids_equal': True,
            'inherited_baseline_cross_architecture_difference_observed': True,
            'strict_original_bit_gate': 'FAIL_UNCHANGED', 'resource_calls': 0,
            'single_record_causal_scope': True},
        'causal_scope': 'For the one diagnosed recording, the unmodified baseline ARM implementation reproduces exactly the ACTIVE113 ARM hidden140 and precise5 values, including their differences from the host reference. This rules out an introduced compact-head/trunk numerical difference for those captured values in this recording. It does not locate the difference within frontend/libm/backbone, establish every remaining ARM recording, or assert full-OTHER equivalence.',
        'all_bound_sources_models_saved_outputs_unchanged': True,
        'reviewer_execution': {'model_calls': 0, 'pcm_calls': 0, 'detector_calls': 0, 'training': 0,
                               'compilation': 0, 'saved_bytes_only': True},
        'model_promotion_authorized': False, 'physical_ssc305': 'NOT_RUN', 'independent_far_frr': 'NOT_RUN',
        'assertions': CHECKS,
    }
    with args.output.open('x') as f:
        json.dump(audited, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps({'status': audited['status'], 'audit_sha256': sha(args.output),
                      'audit_source_sha256': audited['audit_source_sha256'], 'assertions': audited['assertions'],
                      'public_summary': audited['public_summary']}, sort_keys=True))


if __name__ == '__main__':
    main()
