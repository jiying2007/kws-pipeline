#!/usr/bin/env python3
"""One new immutable-view candidate; reuse old traces, never retry a model call."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

INPUT_SHA = '63d2f75e5e93e454bdf23624277fa728bba0cef6eee9cce311f037612198d6ab'
MODEL_SHA = '63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4'
FRONTEND_SHA = '49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49'
SAVED_TRACE_MANIFEST_SHA = '8a3d0fbd2822c383b40db2159785419a3b85ab926f06a14c3b3da2a056dd1e44'

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def save_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())

def trace_counts(data):
    frames = events = 0
    words = {'1': 0, '2': 0}
    for line in data.decode('ascii').splitlines():
        fields = line.split(',')
        if fields[0] == 'F':
            if len(fields) != 5:
                raise ValueError('malformed frame trace')
            frame, source, delivery, token = map(int, fields[1:])
            if frame != frames or not (0 <= source <= delivery) or not 0 <= token < 2601:
                raise ValueError('invalid frame trace')
            frames += 1
        elif fields[0] == 'E':
            if len(fields) != 7:
                raise ValueError('malformed event trace')
            segment, word, frame, first, source, delivery = map(int, fields[1:])
            if word not in (1, 2) or not 0 <= first <= frame < frames or not 0 <= source <= delivery:
                raise ValueError('invalid event trace')
            events += 1
            words[str(word)] += 1
        else:
            raise ValueError('unexpected trace line')
    return frames, events, words

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--saved-traces', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--frontend', type=Path, required=True)
    ap.add_argument('--host-trace', type=Path, required=True)
    ap.add_argument('--host-init', type=Path, required=True)
    ap.add_argument('--arm-trace', type=Path, required=True)
    ap.add_argument('--qemu', type=Path, required=True)
    ap.add_argument('--sysroot', type=Path, required=True)
    ap.add_argument('--build-receipt', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    for name in vars(args):
        setattr(args, name, getattr(args, name).absolute())
    if args.output.exists():
        raise ValueError('fresh output required; do not replay a consumed run')
    for name, expected in [('model', MODEL_SHA), ('frontend', FRONTEND_SHA)]:
        if sha(getattr(args, name)) != expected:
            raise ValueError(name + ' identity mismatch')
    manifest_path = args.inputs / 'INPUT_MANIFEST.json'
    if sha(manifest_path) != INPUT_SHA:
        raise ValueError('input manifest identity mismatch')
    manifest = json.loads(manifest_path.read_text())
    records = manifest['records']
    if len(records) != 134 or [r['index'] for r in records] != list(range(134)):
        raise ValueError('fixed input inventory mismatch')
    if sha(args.saved_traces / 'FILE_MANIFEST.json') != SAVED_TRACE_MANIFEST_SHA:
        raise ValueError('saved trace capsule manifest identity mismatch')
    baseline = json.loads((args.saved_traces / 'FILE_MANIFEST.json').read_text())
    build = json.loads(args.build_receipt.read_text())
    if build['status'] != 'HOST_AND_ARM_BUILT_NO_MODEL_EXECUTION':
        raise ValueError('incomplete build receipt')
    for path, arch, name in ((args.host_trace, 'host', 'fsmn_trace_view'),
                             (args.host_init, 'host', 'fsmn_init_bench'),
                             (args.arm_trace, 'arm', 'fsmn_trace_view')):
        expected = build['variants'][arch]['binaries'][name]
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('executable differs from build receipt: ' + name)
    for record in records:
        index = record['index']
        pcm = args.inputs / 'pcm' / f'{index:03}.pcm'
        old_rel = f'private-traces/{index:03}-arm_par4.txt'
        old = args.saved_traces / old_rel
        if pcm.stat().st_size != record['pcm_bytes'] or sha(pcm) != record['pcm_sha256']:
            raise ValueError('PCM identity mismatch')
        if sha(old) != baseline[old_rel]['sha256'] or old.stat().st_size != baseline[old_rel]['bytes']:
            raise ValueError('saved comparison identity mismatch')
        count, _, _ = trace_counts(old.read_bytes())
        if count != record['feature_shape'][0]:
            raise ValueError('saved trace frame inventory mismatch')
    args.output.mkdir(parents=True)
    for directory in ('claims', 'traces', 'logs', 'init'):
        (args.output / directory).mkdir()
    names = ('model', 'frontend', 'host_trace', 'host_init', 'arm_trace', 'qemu', 'build_receipt')
    plan = {
        'schema': 'r1-borrowed-fixed-comparison-plan-v1',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'candidate': 'R1-DEV-FSMN-20261107-head8-par4-borrowed',
        'input_manifest_sha256': INPUT_SHA,
        'input_rows': 134, 'train_role_rows': 100, 'exposed_development_rows': 34,
        'candidate_runtime_arms': ['host', 'arm_qemu'],
        'max_candidate_clip_invocations': 268,
        'max_init_only_invocations': 24,
        'init_modes': ['copy', 'view'], 'init_repeats_per_mode': 12,
        'per_process_timeout_seconds': 45, 'total_deadline_seconds': 900,
        'old_model_inference_invocations': 0,
        'training': 0, 'asr': 0, 'tts': 0, 'new_data_admission': 0,
        'retry': False, 'selection_or_threshold_search': False,
        'comparison': 'byte-exact F/E frame-argmax/event/source/delivery-clock trace against retained original arm_par4',
        'full_logit_parity': 'NOT_TESTED; original arithmetic retained byte-for-byte',
        'physical_ssc305': 'NOT_RUN', 'independent_far_frr': 'NOT_RUN',
        'source_script_sha256': sha(Path(__file__)),
        'identities': {name: {'path': str(getattr(args, name)), 'sha256': sha(getattr(args, name))} for name in names},
        'saved_trace_manifest_sha256': sha(args.saved_traces / 'FILE_MANIFEST.json'),
        'sysroot': str(args.sysroot),
        'host': {'platform': platform.platform(), 'machine': platform.machine()},
    }
    save_new(args.output / 'PLAN.json', plan)
    terminal = {'schema': 'r1-borrowed-fixed-comparison-terminal-v1', 'status': 'RUNNING',
                'plan_sha256': sha(args.output / 'PLAN.json'), 'claimed': [], 'completed': [],
                'init_results': [], 'new_training': 0, 'new_tts': 0, 'new_asr': 0}
    deadline = time.monotonic() + 900

    def invoke(name, argv, stdout_path, stderr_path):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('declared total deadline exhausted')
        claim = {'name': name, 'argv': [str(x) for x in argv], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        save_new(args.output / 'claims' / (name + '.json'), claim)
        terminal['claimed'].append(name)
        with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    timeout=min(45, remaining), check=False)
        if result.returncode:
            raise RuntimeError(name + ' returned ' + str(result.returncode) + '; retained, no retry')

    try:
        # Constructor-only controls are a new paired memory observation, not repeated speech inference.
        for number in range(12):
            modes = ('copy', 'view') if number % 2 == 0 else ('view', 'copy')
            for mode in modes:
                name = f'init-{number:02}-{mode}'
                output = args.output / 'init' / (name + '.json')
                invoke(name, [args.host_init, mode, args.model, args.frontend], output,
                       args.output / 'logs' / (name + '.stderr'))
                value = json.loads(output.read_text())
                if value['mode'] != mode or value['model_forwards'] != 0 or value['pcm_bytes'] != 0 or value['initializations'] != 1:
                    raise ValueError('invalid init-only receipt')
                terminal['init_results'].append(value)
        for arm in ('host', 'arm_qemu'):
            for record in records:
                index = record['index']
                name = f'{arm}-{index:03}'
                target = args.output / 'traces' / (name + '.txt')
                stderr = args.output / 'logs' / (name + '.stderr.json')
                argv = [args.host_trace if arm == 'host' else args.arm_trace, args.model,
                        args.frontend, args.inputs / 'pcm' / f'{index:03}.pcm']
                if arm == 'arm_qemu':
                    argv = [args.qemu, '-L', args.sysroot, *argv]
                invoke(name, argv, target, stderr)
                data = target.read_bytes()
                expected = (args.saved_traces / f'private-traces/{index:03}-arm_par4.txt').read_bytes()
                if data != expected:
                    raise ValueError(name + ' trace mismatch; no retry or threshold adaptation')
                frames, events, words = trace_counts(data)
                receipt = json.loads(stderr.read_text())
                if receipt.get('ok') is not True or receipt['frames'] != frames or receipt['events'] != events or receipt['pcm_bytes'] != record['pcm_bytes']:
                    raise ValueError(name + ' summary mismatch')
                terminal['completed'].append({'arm': arm, 'index': index, 'trace_sha256': sha(target),
                    'frames': frames, 'events': events, 'words': words, 'pcm_sha256': record['pcm_sha256']})
                if (index + 1) % 16 == 0 or index == 133:
                    print(json.dumps({'arm': arm, 'completed': index + 1, 'total': 134}), flush=True)
        stats = {}
        for mode in ('copy', 'view'):
            group = [r for r in terminal['init_results'] if r['mode'] == mode]
            stats[mode] = {'runs': len(group)}
            for metric in ('model_init_wall_ms', 'proc_self_vm_hwm_kib', 'proc_self_vm_rss_kib', 'process_peak_rss_kib'):
                values = [r[metric] for r in group if r[metric] is not None]
                stats[mode][metric] = {'min': min(values), 'median': statistics.median(values), 'max': max(values)} if values else None
        terminal['init_statistics'] = stats
        terminal['status'] = 'PASS_SAVED_TRACE_PARITY_NOT_BOARD_QUALIFICATION'
        terminal['completed_candidate_clip_invocations'] = len(terminal['completed'])
        terminal['runtime_totals'] = {}
        for arm in ('host', 'arm_qemu'):
            group = [r for r in terminal['completed'] if r['arm'] == arm]
            totals = {'clips': len(group), 'frames': sum(r['frames'] for r in group),
                      'events': sum(r['events'] for r in group),
                      'words': {k: sum(r['words'][k] for r in group) for k in ('1', '2')}}
            if totals != {'clips': 134, 'frames': 12534, 'events': 88, 'words': {'1': 44, '2': 44}}:
                raise ValueError('runtime totals differ from fixed saved inventory')
            terminal['runtime_totals'][arm] = totals
    except BaseException as exc:
        terminal['status'] = 'FAILED_RETAINED_NO_RETRY'
        terminal['exception'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        terminal['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save_new(args.output / 'TERMINAL.json', terminal)
    print(json.dumps({'status': terminal['status'], 'candidate_invocations': len(terminal['completed']),
                      'init_only_invocations': len(terminal['init_results']), 'init_statistics': stats}), flush=True)

if __name__ == '__main__':
    main()
