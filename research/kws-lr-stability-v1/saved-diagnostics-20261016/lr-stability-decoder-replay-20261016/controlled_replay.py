#!/usr/bin/env python3
"""One controlled native decoder pass on saved LP; never runs a model."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

PLAN_SHA256 = '0d36ca92c05b7e369ef8f3da8a7c1f729616e1646c4d3015ef6123a96b90a9c3'
SOURCE_MANIFEST_SHA256 = '45ce555b5984fc6e04d215f85b672b621acd810914c580084376783d7ca72ca9'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def require(condition, text):
    if not condition:
        raise ValueError(text)


def checked(path, digest):
    require(sha(path) == digest, 'hash mismatch: ' + str(path))
    return json.loads(path.read_bytes())


def bits(value):
    return format(struct.unpack('<I', struct.pack('<f', value))[0], '08x')


def validate_inputs(base):
    plan = checked(HERE / 'analysis-plan.json', PLAN_SHA256)
    sources = checked(HERE / 'source-manifest.json', SOURCE_MANIFEST_SHA256)
    require(sources['commit'] == plan['source_commit'], 'source commit mismatch')
    for relative, pin in sources['files'].items():
        raw = (HERE / 'source' / relative).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == pin['sha256'], 'source SHA256 mismatch')
        git_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        require(git_blob == pin['git_blob_sha1'], 'source Git blob mismatch')
    loaded = {}
    pins = dict(plan['original_inputs'], **plan['additional_original_inputs'])
    for name, pin in pins.items():
        path = base / pin['relative_path']
        require(sha(path) == pin['sha256'], 'original evidence changed: ' + name)
        if path.suffix == '.json':
            loaded[name] = json.loads(path.read_bytes())
    observation, experiment, accepted, cohort = (loaded[name] for name in
                                                 ('observation', 'plan', 'accepted_manifest', 'cohort'))
    require(accepted['result_files']['A-0600.json'] == pins['observation']['sha256'], 'accepted LP membership mismatch')
    require(accepted['plan_sha256'] == pins['plan']['sha256'], 'experiment identity mismatch')
    require(experiment['cohort_sha256'] == pins['cohort']['sha256'], 'cohort binding mismatch')
    require(experiment['diagnostic_code_sha256']['run_lr_stability.py'] == pins['actual_lr_observation_writer']['sha256'], 'actual writer mismatch')
    require(observation['update'] == 600 and observation['kind'] == 'observation', 'wrong observation')
    require(observation['role'] == 'train', 'only fixed training traces allowed')
    samples = observation['samples']
    require(len(samples) == 12 and sum(s['input_frames'] for s in samples) == 1276, 'fixed budget mismatch')
    require([s['record_id'] for s in samples] == experiment['train_record_ids'], 'record order changed')
    frozen = {s['record_id']: s for s in cohort['samples'] if s['role'] == 'train'}
    require(len(frozen) == 12, 'frozen cohort count mismatch')
    transport = ['KWSLP_DIAGNOSTIC_V1 12']
    for index, sample in enumerate(samples):
        require(all(sample[key] == frozen[sample['record_id']][key] for key in
                    ('record_id', 'target_class', 'pcm_sha256', 'input_frames', 'prefix_frames')), 'sample metadata mismatch')
        target = [1, 2, 3, 4] if sample['target_class'] == 'KW1' else [3, 4, 3, 4]
        require(sample['target_tokens'] == target, 'target mismatch')
        require(len(sample['log_probs']) == sample['input_frames'], 'frame count mismatch')
        transport.append('CLIP %d %d' % (index, sample['input_frames']))
        for frame, row in enumerate(sample['log_probs']):
            require(len(row) == 5 and all(type(v) in (int, float) and math.isfinite(v) for v in row), 'invalid LP row')
            require(abs(math.log(math.fsum(math.exp(v) for v in row))) <= 2e-6, 'unnormalized LP row')
            require(all(struct.unpack('<f', struct.pack('<f', v))[0] == v for v in row), 'LP not exact saved float32')
            require(max(range(5), key=lambda k: row[k]) == sample['raw_argmax_tokens'][frame], 'saved argmax mismatch')
            transport.append('%d %d 1 %s' % (frame, 400 + 320 * frame, ' '.join(float(v).hex() for v in row)))
    require((HERE / 'source/keywords/tokens.example.txt').read_text().splitlines() ==
            ['<blk> 0', 'ni3 1', 'hao3 2', 'xiao3 3', 'wo1 4'], 'token mapping mismatch')
    contract = json.loads((HERE / 'source/configs/parameter-contract.json').read_bytes())
    require(contract['runtime']['state_retention']['default'] == .94, 'state retention changed')
    require(contract['runtime']['token_boost']['default'] == 1.5, 'compatibility field changed')
    require(contract['runtime']['refractory_ms']['default'] == 1200, 'refractory changed')
    require(all(contract['keyword_pack'][key]['default'] == 0 for key in
                ('min_trailing_blanks', 'priority', 'grace_frames')), 'keyword default changed')
    shipping = json.loads((HERE / 'source/configs/shipping.xiaowo.json').read_bytes())
    require([word['threshold'] for word in shipping['shipping_wake_words']] == [.55, .55], 'example threshold changed')
    return plan, sources, samples, '\n'.join(transport) + '\n'


def command(args, commands, *, cwd=None):
    result = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True, check=False, timeout=120)
    record = {'args': [str(a) for a in args], 'returncode': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr}
    commands.append(record)
    require(result.returncode == 0, 'build/synthetic command failed: ' + json.dumps(record))
    return result.stdout


def preflight(base, build_name):
    require(not (HERE / 'execution-claim.json').exists(), 'saved-data execution already claimed; no new preflight')
    require(not (HERE / 'preexecution-freeze.json').exists(), 'preflight already frozen')
    require(Path(build_name).name == build_name and build_name.startswith('build-'), 'use simple build-* name')
    plan, sources, samples, transport = validate_inputs(base)
    build = HERE / build_name
    build.mkdir()
    commands = []
    cc = shutil.which('cc')
    require(cc is not None, 'C compiler unavailable')
    compiler_version = command([cc, '--version'], commands)
    generated = build / 'generated'
    command([sys.executable, '-B', '-I', HERE / 'source/tools/gen_parameter_limits.py',
             HERE / 'source/configs/parameter-contract.json', generated], commands)
    command([sys.executable, '-B', '-I', HERE / 'source/tools/gen_parameter_limits.py',
             HERE / 'source/configs/parameter-contract.json', generated, '--check'], commands)
    flags = ['-std=c11', '-O2', '-fno-fast-math', '-ffp-contract=off',
             '-Wall', '-Wextra', '-Werror', '-pedantic',
             '-I' + str(HERE / 'source/include'), '-I' + str(HERE / 'source/src'),
             '-I' + str(generated), '-I' + str(HERE)]
    command([cc, *flags, HERE / 'source/tests/test_decoder.c', HERE / 'source/src/decoder.c',
             '-lm', '-o', build / 'test_decoder'], commands)
    original_test = command([build / 'test_decoder'], commands)
    command([cc, *flags, HERE / 'replay_harness.c', HERE / 'source/src/decoder.c',
             '-lm', '-o', build / 'replay_harness'], commands)
    harness_test = json.loads(command([build / 'replay_harness', '--self-test'], commands))
    require(harness_test['status'] == 'PASS', 'harness fixture failed')
    symbols = command(['nm', str(build / 'replay_harness')], commands)
    require('kws_decoder_step' in symbols and 'kws_engine_' not in symbols and
            'kws_model_' not in symbols and 'kws_frontend_' not in symbols,
            'binary includes forbidden pipeline implementation')
    transport_path = build / 'saved-lp-input.txt'
    with transport_path.open('x') as stream:
        stream.write(transport)
    save_new(build / 'build-and-synthetic-tests.json', {'commands': commands,
              'original_decoder_test_stdout': original_test, 'harness_tests': harness_test,
              'compiler_version': compiler_version, 'saved_data_executed': False})
    files = ['analysis-plan.json', 'DEFINITIONS.md', 'source-manifest.json',
             'controlled_replay.py', 'replay_harness.c', 'refractory_mirror.h']
    files.extend('source/' + name for name in sources['files'])
    files.extend(str(path.relative_to(HERE)) for path in sorted(build.rglob('*')) if path.is_file())
    freeze = {'schema': 'controlled-native-replay-preexecution-freeze-v1',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'plan_sha256': PLAN_SHA256, 'source_manifest_sha256': SOURCE_MANIFEST_SHA256,
              'compiler': cc, 'build_directory': build_name,
              'binary_path': build_name + '/replay_harness',
              'input_transport_path': build_name + '/saved-lp-input.txt',
              'file_sha256': {name: sha(HERE / name) for name in files},
              'saved_data_executed': False, 'samples': len(samples), 'frames': 1276}
    save_new(HERE / 'preexecution-freeze.json', freeze)
    print('PREFLIGHT PASS: native source fixtures and suppression fixtures; no saved-data C pass executed')


def verify_freeze(base):
    plan, sources, samples, transport = validate_inputs(base)
    freeze = json.loads((HERE / 'preexecution-freeze.json').read_bytes())
    require(freeze['plan_sha256'] == PLAN_SHA256 and freeze['source_manifest_sha256'] == SOURCE_MANIFEST_SHA256, 'freeze identity mismatch')
    for name, digest in freeze['file_sha256'].items():
        require(sha(HERE / name) == digest, 'preexecution file changed: ' + name)
    require((HERE / freeze['input_transport_path']).read_text() == transport, 'prepared transport changed')
    return freeze, plan, samples


def summarize_output(base):
    freeze, plan, samples = verify_freeze(base)
    lines = [json.loads(line) for line in (HERE / 'native-output.jsonl').read_text().splitlines()]
    require(len(lines) == 1277, 'incomplete native output')
    require(lines[-1]['kind'] == 'terminal' and lines[-1]['status'] == 'completed', 'missing completion')
    frames = lines[:-1]
    cursor = 0
    result_samples = []
    totals = {'native_decoder_hits': 0, 'mirror_accepted': 0, 'mirror_suppressed': 0}
    for index, sample in enumerate(samples):
        events = []
        deadline = 0
        counts = dict.fromkeys(totals, 0)
        for frame, saved_row in enumerate(sample['log_probs']):
            row = frames[cursor]
            cursor += 1
            require(row['kind'] == 'frame' and row['clip_index'] == index and row['frame'] == frame, 'native frame order mismatch')
            require(row['end_sample'] == 400 + 320 * frame and row['speech_active'] == 1, 'time/control mismatch')
            require(row['lp_float32_bits'] == [bits(v) for v in saved_row], 'LP transport bits differ')
            require(row['raw_argmax'] == sample['raw_argmax_tokens'][frame], 'C argmax mismatch')
            require(row['suppress_until_before'] == deadline, 'mirror before mismatch')
            hit = row['decoder_hit']
            require(type(hit) is int and hit in (0, 1), 'invalid native hit')
            if hit:
                require(row['keyword_id'] in (1, 2) and .55 - 1e-7 <= row['confidence'] <= 1 and math.isfinite(row['confidence']), 'invalid native event')
                outcome = 1 if row['end_sample'] >= deadline else 2
                if outcome == 1:
                    deadline = row['end_sample'] + 19200
                counts['native_decoder_hits'] += 1
                counts['mirror_accepted' if outcome == 1 else 'mirror_suppressed'] += 1
                events.append({'frame': frame, 'end_sample': row['end_sample'],
                               'nominal_clip_coordinate_ms': row['end_sample'] / 16,
                               'keyword_id': row['keyword_id'], 'confidence': row['confidence'],
                               'intended_keyword': row['keyword_id'] == (1 if sample['target_class'] == 'KW1' else 2),
                               'mirror_outcome': 'accepted' if outcome == 1 else 'suppressed'})
            else:
                outcome = 0
                require(row['keyword_id'] == 0 and row['confidence'] == 0, 'non-hit carries event data')
            require(row['mirror_outcome'] == outcome and row['suppress_until_after'] == deadline, 'independent suppression recurrence mismatch')
        for key in totals:
            totals[key] += counts[key]
        row_bytes = b''.join(struct.pack('<5f', *row) for row in sample['log_probs'])
        result_samples.append({'record_id': sample['record_id'], 'target_class': sample['target_class'],
                               'input_frames': sample['input_frames'],
                               'LP_float32_bytes_sha256': hashlib.sha256(row_bytes).hexdigest(),
                               'counts': counts, 'events': events})
    terminal = lines[-1]
    require(terminal['clips'] == 12 and terminal['frames'] == 1276, 'terminal count mismatch')
    require(all(terminal[key] == value for key, value in totals.items()), 'terminal event totals mismatch')
    per_class = {}
    for target in ('KW1', 'KW2'):
        selected = [row for row in result_samples if row['target_class'] == target]
        per_class[target] = {
            'sample_count': len(selected),
            **{key: sum(row['counts'][key] for row in selected) for key in totals},
            'samples_with_intended_native_hit': sum(any(e['intended_keyword'] for e in row['events']) for row in selected),
            'samples_with_intended_mirror_output': sum(any(e['intended_keyword'] and e['mirror_outcome'] == 'accepted' for e in row['events']) for row in selected),
            'samples_with_multiple_mirror_outputs': sum(row['counts']['mirror_accepted'] > 1 for row in selected),
            'other_full_keyword_native_hits': sum(not e['intended_keyword'] for row in selected for e in row['events'])}
    return {'schema': 'controlled-native-decoder-replay-result-v1',
            'status': 'COMPLETED_SYNTHETIC_GATE_UNCALIBRATED_TRAINING_ONLY',
            'plan_sha256': PLAN_SHA256, 'preexecution_freeze_sha256': sha(HERE / 'preexecution-freeze.json'),
            'native_output_sha256': sha(HERE / 'native-output.jsonl'),
            'source_commit': plan['source_commit'], 'sample_count': 12, 'frame_count': 1276,
            'saved_data_native_passes': 1, 'speech_gate': 'synthetic-constant-one',
            'threshold': .55, 'threshold_calibrated_for_research_model': False,
            'production_engine_executed': False, 'suppression_mirror_refractory_ms': 1200,
            'new_model_forwards': 0, 'new_optimizer_updates': 0, 'new_TTS_ASR_calls': 0,
            'audio_or_weight_reads': 0, 'LP_transport_bitwise_verified': True,
            'suppression_recurrence_independently_verified': True,
            'totals': totals, 'per_class': per_class, 'samples': result_samples,
            'limitations': ['Exposed machine-labelled training positives only; no independent model evaluation.',
                            'Constant-one speech gate is synthetic and not a performance upper bound.',
                            'Fixed example threshold is uncalibrated for this research model.',
                            'Native decoder hits and suppression-mirror outputs are separate quantities.',
                            'Nominal clip coordinates are not human word-end latency or device time.',
                            'No real false-activation, duplicate-wake, recall, FA/h or production claims.',
                            'Original raw logits and matched model artifact are unavailable; no bitexact original-logit claim.']}


def execute(base):
    freeze, plan, samples = verify_freeze(base)
    claim = {'schema': 'controlled-native-replay-exclusive-claim-v1',
             'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'preexecution_freeze_sha256': sha(HERE / 'preexecution-freeze.json'),
             'native_saved_data_passes_reserved': 1, 'frames_reserved': 1276,
             'no_retry': True, 'new_model_forwards_reserved': 0}
    save_new(HERE / 'execution-claim.json', claim)
    with (HERE / freeze['input_transport_path']).open('rb') as source, \
         (HERE / 'native-output.jsonl').open('xb') as out, \
         (HERE / 'native-stderr.txt').open('xb') as err:
        completed = subprocess.run([str(HERE / freeze['binary_path'])], stdin=source,
                                   stdout=out, stderr=err, check=False, timeout=30)
    if completed.returncode != 0:
        save_new(HERE / 'execution-failure.json', {'native_returncode': completed.returncode,
                 'no_retry': True, 'output_sha256': sha(HERE / 'native-output.jsonl')})
        raise RuntimeError('claimed native run failed; no retry')
    report = summarize_output(base)
    save_new(HERE / 'report.json', report)
    save_new(HERE / 'execution-complete.json', {'schema': 'controlled-native-replay-completion-v1',
              'native_returncode': 0, 'saved_data_passes_completed': 1, 'frames_completed': 1276,
              'report_sha256': sha(HERE / 'report.json'), 'original_inputs_unchanged': True,
              'next_action': 'STOP_NO_ADDITIONAL_EXPERIMENTS'})
    print(json.dumps({'totals': report['totals'], 'per_class': report['per_class']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'execute', 'verify'])
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--build-name', default='build-1')
    args = parser.parse_args()
    if args.action == 'preflight':
        preflight(args.base, args.build_name)
    elif args.action == 'execute':
        execute(args.base)
    else:
        actual = summarize_output(args.base)
        expected = json.loads((HERE / 'report.json').read_bytes())
        require(actual == expected, 'reported result differs from read-only recomputation')
        print('VERIFY PASS: read-only output recomputation; no native saved-data replay')


if __name__ == '__main__':
    main()
