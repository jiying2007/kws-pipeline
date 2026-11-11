#!/usr/bin/env python3
"""One fixed 268-call detector-only screen; zero audio or model computation.

Refuses to start until the matrix experiment has a successful complete terminal.
Never waits, polls, retries, overwrites, changes thresholds or changes a source.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path('/workspace/scratch/fec7ccd5bb4b')
HERE = Path(__file__).resolve().parent
SAVED = Path('/workspace/scratch/1e1fe64498b2/r1-local/evidence-qemu')
SDK = ROOT/'matrix-build/sdk/par4'
SAVED_SHA = '8a3d0fbd2822c383b40db2159785419a3b85ab926f06a14c3b3da2a056dd1e44'
SOURCE_SHA = '5ef3fc5410d30588cb516253b3f40e9299316178d5e54fe46ad22fd720314dcb'
HEADER_SHA = '539f40b0e8c81eed520da12d8256c01cfe54963a200dccb3a79aaee30e4b5098'
MATRIX_PLAN_SHA = '1074eadcdb1783b4f50abc11c26aeaf15f09a3f23775361957864c54c2e801b2'
CRITICAL = {0, 1027, 1401, 1462, 2093}


def need(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    need(path.is_file() and not path.is_symlink(), 'nonregular or linked input')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key'); result[key] = value
        return result
    def invalid(value): raise ValueError('nonfinite JSON')
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def main():
    terminal_path = ROOT/'matrix-execution/TERMINAL.json'
    need(terminal_path.is_file(), 'NOT_STARTED: matrix terminal is not ready; no polling or waiting')
    matrix = load(terminal_path)
    need(matrix.get('status') == 'PASS_FIXED_FE_MATRIX_HOST_SCREEN_ONLY' and
         matrix.get('plan_sha256') == MATRIX_PLAN_SHA and matrix.get('all_claims_completed') is True and
         len(matrix.get('completed_invocations', [])) == 843 and matrix.get('capture_completed') == 804,
         'NOT_STARTED: matrix has not completed successfully')
    out = HERE/'execution'
    need(not out.exists(), 'fresh screen execution required; no retry or overwrite')
    need(sha(SAVED/'FILE_MANIFEST.json') == SAVED_SHA, 'saved trace manifest changed')
    need(sha(SDK/'src/fsmn_detector.c') == SOURCE_SHA and sha(SDK/'include/fsmn_detector.h') == HEADER_SHA,
         'detector source/header changed')
    manifest = load(SAVED/'FILE_MANIFEST.json')
    inputs = []
    for index in range(134):
        name = f'private-traces/{index:03d}-arm_par4.txt'; path = SAVED/name
        data = path.read_bytes(); identity = manifest[name]
        need(0 < len(data) <= 65536 and len(data) == identity['bytes'] and sha(path) == identity['sha256'],
             'retained input trace identity')
        need(data.endswith(b'\n'), 'trace termination')
        frames = events = changed = 0; words = [0, 0]
        expected = bytearray()
        for line in data.splitlines(keepends=True):
            fields = line[:-1].decode('ascii').split(',')
            need(fields[0] in ('F', 'E') and len(fields) == (5 if fields[0] == 'F' else 7), 'trace row shape')
            need(all(x.isascii() and x.isdecimal() for x in fields[1:]), 'trace unsigned integer')
            v = [int(x) for x in fields[1:]]
            if fields[0] == 'F':
                need(v[0] == frames and v[1] <= v[2] and v[3] < 2601, 'frame ordering/values')
                changed += v[3] not in CRITICAL and v[3] != 1
                frames += 1
            else:
                need(v[0] == 0 and v[1] in (1, 2) and v[3] <= v[2] < frames and v[4] <= v[5], 'event ordering')
                events += 1; words[v[1]-1] += 1; expected.extend(line)
        inputs.append({'path': path, 'sha256': identity['sha256'], 'expected': bytes(expected),
                       'frames': frames, 'events': events, 'words': words, 'mapped_frames': changed})
    need(sum(r['frames'] for r in inputs) == 12534 and sum(r['events'] for r in inputs) == 88 and
         [sum(r['words'][k] for r in inputs) for k in (0, 1)] == [44, 44], 'fixed input denominator')
    os.umask(0o077); out.mkdir()
    for name in ('claims', 'events', 'logs'): (out/name).mkdir()
    compiler = Path('/usr/bin/gcc').resolve(); executable = out/'replay_detector'
    command = [str(compiler), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
               '-I'+str(SDK/'include'), str(HERE/'replay_detector.c'), str(SDK/'src/fsmn_detector.c'),
               '-o', str(executable)]
    plan = {'schema': 'r1-saved-other-collapse-screen-plan-v1', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'matrix_terminal_sha256': sha(terminal_path), 'matrix_plan_sha256': MATRIX_PLAN_SHA,
        'saved_manifest_sha256': SAVED_SHA, 'detector_source_sha256': SOURCE_SHA, 'detector_header_sha256': HEADER_SHA,
        'runner_source_sha256': sha(HERE/'replay_detector.c'), 'script_sha256': sha(Path(__file__)),
        'compiler_sha256': sha(compiler), 'build_command': command,
        'inputs': 134, 'frames_per_mode': 12534, 'modes': ['original', 'other'], 'sentinel_id': 1,
        'preserved_ids': sorted(CRITICAL), 'maximum_detector_only_replays': 268,
        'max_seconds': 120, 'per_replay_timeout_seconds': 2, 'retries': 0,
        'pcm_reads': 0, 'neural_model_forwards': 0, 'training_updates': 0, 'asr': 0, 'tts': 0,
        'detector_parameters_changed': False, 'model_promotion_authorized': False,
        'historical_compact_affine_sum_other_experiment': '2026-09-30 FAIL_FIDELITY remains negative; different objective and teacher'}
    save(out/'PLAN.json', plan)
    result = {'schema': 'r1-saved-other-collapse-screen-result-v1', 'status': 'RUNNING',
              'plan_sha256': sha(out/'PLAN.json'), 'claimed': [], 'completed': []}
    begin = time.monotonic()
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        with (out/'logs/build.stdout').open('xb') as so, (out/'logs/build.stderr').open('xb') as se:
            built = subprocess.run(command, stdout=so, stderr=se, stdin=subprocess.DEVNULL, timeout=20, check=False)
        need(built.returncode == 0 and (out/'logs/build.stderr').stat().st_size == 0, 'compile failed; no retry')
        result['executable_sha256'] = sha(executable)
        for index, item in enumerate(inputs):
            for mode in ('original', 'other'):
                need(time.monotonic()-begin < 120, 'fixed screen time budget exhausted')
                name = f'{index:03d}-{mode}'; argv = [str(executable), mode, str(item['path'])]
                save(out/'claims'/(name+'.json'), {'name': name, 'argv': argv, 'attempt': 1,
                     'input_sha256': item['sha256'], 'model_forwards': 0})
                result['claimed'].append(name)
                output = out/'events'/(name+'.txt'); error = out/'logs'/(name+'.stderr')
                with output.open('xb') as so, error.open('xb') as se:
                    executed = subprocess.run(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL,
                                              timeout=2, check=False, preexec_fn=limits)
                need(executed.returncode == 0 and error.stat().st_size == 0 and output.stat().st_size <= 65536,
                     'detector replay failed; no retry')
                result['completed'].append({'name': name, 'mode': mode, 'index': index,
                    'events_equal': output.read_bytes() == item['expected'], 'output_sha256': sha(output),
                    'expected_events_sha256': hashlib.sha256(item['expected']).hexdigest(),
                    'frames': item['frames'], 'events': item['events'], 'words': item['words']})
        need(len(result['claimed']) == 268 and len(set(result['claimed'])) == 268 and
             result['claimed'] == [r['name'] for r in result['completed']], 'complete unique denominator')
        need(sha(SAVED/'FILE_MANIFEST.json') == SAVED_SHA and
             all(sha(r['path']) == r['sha256'] for r in inputs) and
             sha(SDK/'src/fsmn_detector.c') == SOURCE_SHA and sha(SDK/'include/fsmn_detector.h') == HEADER_SHA and
             sha(HERE/'replay_detector.c') == plan['runner_source_sha256'] and sha(Path(__file__)) == plan['script_sha256'],
             'post-screen input/source drift')
        result['post_input_sources_unchanged'] = True
        result['status'] = 'PASS_SAVED_EVENT_EQUIVALENCE_ONLY' if all(r['events_equal'] for r in result['completed']) else 'FAIL_SAVED_EVENT_EQUIVALENCE'
    except BaseException as exc:
        result['status'] = 'STOPPED_NO_RETRY'; result['failure'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        result['wall_seconds'] = time.monotonic()-begin
        save(out/'TERMINAL.json', result)
    modes = {}
    for mode in ('original', 'other'):
        rows = [r for r in result['completed'] if r['mode'] == mode]
        modes[mode] = {'replays': len(rows), 'frames': sum(r['frames'] for r in rows),
            'saved_events': sum(r['events'] for r in rows), 'saved_events_k1': sum(r['words'][0] for r in rows),
            'saved_events_k2': sum(r['words'][1] for r in rows), 'mismatching_recordings': sum(not r['events_equal'] for r in rows)}
    aggregate = {'schema': 'r1-saved-other-collapse-screen-aggregate-v1', 'status': result['status'],
        'plan_sha256': sha(out/'PLAN.json'), 'terminal_sha256': sha(out/'TERMINAL.json'),
        'saved_manifest_sha256': SAVED_SHA, 'detector_source_sha256': SOURCE_SHA,
        'detector_header_sha256': HEADER_SHA, 'unique_inputs': 134, 'completed_detector_only_replays': 268,
        'modes': modes, 'other_mode_token_ids_changed_frames': sum(r['mapped_frames'] for r in inputs),
        'neural_model_forwards': 0, 'pcm_reads': 0, 'training_updates': 0, 'asr': 0, 'tts': 0,
        'detector_parameters_changed': False, 'model_promotion_authorized': False,
        'post_input_sources_unchanged': True, 'historical_compact_affine_sum_other': 'FAIL_FIDELITY_UNCHANGED',
        'interpretation': 'Only saved argmax token mapping was exercised through unchanged detector code.',
        'not_established': ['Accuracy of any approximate or learned max-other head',
            'Equivalence for unseen audio or full logits', 'Probability calibration',
            'Independent FAR/FRR', 'Physical SSC305 runtime or latency']}
    save(out/'AGGREGATE.json', aggregate)
    print(json.dumps(aggregate, sort_keys=True, allow_nan=False))
    need(result['status'] == 'PASS_SAVED_EVENT_EQUIVALENCE_ONLY', 'mapping failed; preserve result, no retry')


if __name__ == '__main__':
    main()
