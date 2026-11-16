"""Invented scalar/NPZ/PCM evidence fixtures; never opens retained audio or checkpoints."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ab_contract as c
import run_ab as execute


def require(value, why):
    if not value:
        raise ValueError('AB_FIXTURE_' + why)


def rejected(action, marker):
    try:
        action()
    except (ValueError, KeyError, TypeError) as error:
        require(marker in str(error), 'EXPECTED_' + marker + '_GOT_' + str(error))
    else:
        raise ValueError('AB_FIXTURE_ACCEPTED_' + marker)


def npz(kind, changed=False, bad=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for index, (name, shape) in enumerate(c.exporter.PARAMETERS[kind]):
            header = {'descr': '<f4', 'fortran_order': False, 'shape': shape}
            if index == 0 and bad == 'object':
                header['descr'] = '|O'
            if index == 0 and bad == 'list':
                header['shape'] = list(shape)
            if index == 0 and bad == 'fortran':
                header['fortran_order'] = True
            text = repr(header).encode('ascii')
            text += b' ' * ((64 - (10 + len(text) + 1) % 64) % 64) + b'\n'
            values = [0.0] * math.prod(shape)
            if changed and index == 0:
                values[0] = .125
            if bad == 'nan' and index == 0:
                values[0] = float('nan')
            body = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(text)) + text
            body += b''.join(struct.pack('<f', x) for x in values)
            archive.writestr(name + '.npy', body)
    return buffer.getvalue()


def scalar_cases():
    tape = []
    class Scalar:
        def __init__(self, value, derivative=1):
            self.value, self.derivative = value, derivative
        def detach(self):
            return self.value
        def backward(self):
            tape.append(self.derivative)
        def __mul__(self, coefficient):
            return Scalar(self.value * coefficient, self.derivative * coefficient)
    value = c.accumulate_loss(lambda: Scalar(1), lambda i: Scalar([2, 4, 6, 8][i]), True)
    require(value == {'old30_mil_ce': 1.0, 'new128_mil_ce': 5.0, 'objective': 3.5} and
            tape == [1, .125, .125, .125, .125], 'ACCUMULATED_MEAN_GRADIENT')
    tape.clear()
    def forbidden(_):
        raise ValueError('control reached natural background')
    require(c.accumulate_loss(lambda: Scalar(1), forbidden, False)['objective'] == 1 and tape == [1],
            'CONTROL_OLD30_UNCHANGED')


def metrics(natural=(100, 100), correct=(2, 2), wrong=(0, 0), ordinary=0):
    return {'natural_total': sum(natural), 'natural_by_class': dict(zip(('1', '2'), natural)),
            'correct_trigger_clips': dict(zip(('1', '2'), correct)),
            'wrong_other_keyword_clips': dict(zip(('1', '2'), wrong)), 'ordinary_background_proposals': ordinary}


def gate_cases():
    baseline, control, treatment = metrics(), metrics((80, 80)), metrics((40, 40))
    require(c.benefit(baseline, control, treatment)['decision'] == 'RESEARCH_BENEFIT', 'BENEFIT')
    cases = [metrics((41, 40)), metrics((0, 0), correct=(0, 0)), metrics((40, 40), correct=(1, 2)),
             metrics((40, 40), wrong=(1, 0)), metrics((40, 40), ordinary=1)]
    for row in cases:
        require(c.benefit(baseline, control, row)['decision'] == 'NO_BENEFIT', 'GATE_REGRESSION')
    require(c.benefit(metrics((0, 0)), control, metrics((0, 0)))['decision'] == 'NO_BENEFIT', 'ZERO_BASELINE')
    require(c.benefit(metrics((0, 200)), metrics((0, 160)), metrics((1, 79)))['decision'] == 'NO_BENEFIT',
            'PER_KEYWORD_NATURAL_REGRESSION')
    require(c.benefit(metrics(correct=(4, 2)), control, treatment)['decision'] == 'NO_BENEFIT', 'POSITIVE_BASELINE_FLOOR')


def fake_header():
    text = '#include "en63_stream.h"\n'
    for kind in c.KINDS:
        text += c.exporter.array('en63_' + kind + '_parameters', [0.] * sum(math.prod(s) for _, s in c.exporter.PARAMETERS[kind]))
        text += '/* "' + c.p.c.SOURCE_WEIGHTS[kind] + '" */\n'
    return text.encode('ascii')


def npz_cases():
    for kind in c.KINDS:
        original, changed = c.npz_model(npz(kind), kind), c.npz_model(npz(kind, True), kind)
        require(c.changed_tensors(original[1], changed[1], kind) == 1, 'REAL_PARAMETER_CHANGE')
        same = io.BytesIO(npz(kind))
        with zipfile.ZipFile(same, 'a') as archive:
            archive.comment = b'invented container metadata change'
        meta, payload = c.npz_model(same.getvalue(), kind)
        require(meta['npz_sha256'] != original[0]['npz_sha256'] and c.changed_tensors(original[1], payload, kind) == 0,
                'ZIP_HASH_IS_NOT_WEIGHT_CHANGE')
        for tag in ('object', 'list', 'fortran', 'nan'):
            rejected(lambda: c.npz_model(npz(kind, bad=tag), kind), 'AB_NPY_VALUES' if tag == 'nan' else 'AB_NPY_SCHEMA')
    baseline = fake_header()
    with mock.patch.dict(c.PROFILE, {'baseline_header_sha256': c.digest(baseline)}):
        new = c.replace_models(baseline, {k: c.npz_model(npz(k, True), k) for k in c.KINDS})
        require(new != baseline and b'0x1.0000000000000p-3f' in new, 'EXACT_FP32_EXPORT')
        require(all(c.changed_tensors(c.baseline_payloads(baseline)[k], c.npz_model(npz(k, True), k)[1], k) == 1
                    for k in c.KINDS), 'BASELINE_PAYLOAD_ROUNDTRIP')


def make_segments(samples):
    return [{'source_id': 'invented-' + str(i), 'samples': n, 'weak_wholeclip_class': 0 if i == 0 else (i - 1) // 4,
             'rows': [{'member': 'invented-' + str(i), 'sample_count': n, 'concatenation_start_sample': 0}]}
            for i, n in enumerate(samples)]


def invented_trace(segments, events=False):
    output = bytearray(c.TRACE_HEADER)
    for index, source in enumerate(segments):
        frames = c.geometry(source['samples'])['frames']
        output.extend(c.SEGMENT_HEADER.pack(index, 32, source['samples'], frames))
        for frame in range(1, frames + 1):
            probs, event = (1., 0., 0.), 0
            if events and 50 <= frame <= 52:
                probs = (.1, .9, 0.)
                event = 1 if frame == 52 else 0
            elif events and 152 <= frame <= 154:
                probs = (.1, 0., .9)
                event = 2 if frame == 154 else 0
            output.extend(c.s.TRACE_RECORD.pack(frame, 400 + (frame - 1) * 160, *probs, event))
    return bytes(output)


def trace_cases():
    segments = make_segments([24880] * 13)
    segments[0]['rows'] = [{'member': 'first', 'sample_count': 8560, 'concatenation_start_sample': 0},
                           {'member': 'second', 'sample_count': 16320, 'concatenation_start_sample': 8560}]
    raw = invented_trace(segments, True)
    result = c.trace_observation(raw, segments)
    first = result['segments'][0]['events']
    require(first[0]['frame'] == 52 and first[0]['keyword'] == 'Hey Vosen' and first[0]['source_interval_member'] == 'first'
            and first[0]['exact_splice_boundary'] is True and first[1]['frame'] == 154 and first[1]['keyword'] == 'Hey Mero',
            'DECODER_CLASS_CLOCK_BOUNDARY')
    require(c.event_metrics(result) == metrics((1, 1), (4, 4), (4, 4), 8), 'UNIQUE_CLIP_COVERAGE')
    altered = bytearray(raw)
    struct.pack_into('<I', altered, 16 + 24 + 51 * 32 + 28, 2)
    rejected(lambda: c.trace_observation(bytes(altered), segments), 'LIBRISPEECH_RAW_PROBABILITY_DECODER_EVENT')
    altered = bytearray(raw)
    struct.pack_into('<Q', altered, 16 + 24 + 8, 401)
    rejected(lambda: c.trace_observation(bytes(altered), segments), 'LIBRISPEECH_TRACE_VALUES')
    rejected(lambda: c.trace_observation(raw + b'x', segments), 'AB_TRACE_NO_EXTRA_BYTES')


def write_zip(path, members):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(name, raw)


def saved_forgery_case(root):
    """Only historical input loading is substituted; real saved weight/build/decoder/summary checks run."""
    root.mkdir()
    samples = [11124244] + [32000] * 11 + [42240]  # Twelve independent clips: 2440 frames, 394240 samples.
    segments = make_segments(samples)
    trace = invented_trace(segments)
    observed = c.trace_observation(trace, segments)
    require(observed['frames'] == 71965 and observed['feed_calls'] == 71991, 'INVENTED_FULL_GEOMETRY')
    baseline = fake_header()
    weights = {a + '-' + k + '.npz': npz(k, True) for k, a in c.ARMS}
    weight_meta = {n: c.npz_model(raw, 'dsc_tcn' if 'dsc_tcn' in n else 'gru') for n, raw in weights.items()}
    identity = {'repository': 'jiying2007/kws-data', 'head': '1' * 40, 'run_id': '123', 'run_attempt': '1'}
    hashes = c.source_hashes()
    inputs = {'request_sha256': '2' * 64, 'claim_sha256': '3' * 64, 'trained_files_sha256': {},
              'evaluation_pcm_sha256': {}}
    build = {'schema': 'english63-background-ab-build-v1', 'compiler': '/usr/bin/gcc', 'compiler_realpath': '/usr/bin/gcc',
             'compiler_sha256': '4' * 64, 'compiler_version': 'invented', 'pipeline_head': '5' * 40,
             'source_root': str(c.ROOT), 'build_root': str(root / 'invented-build'), 'output_root': str(root),
             'entrypoint_pid': os.getpid(), 'public_run': identity, 'physical_ssc305': False,
             'source_sha256': hashes, 'variants': []}
    runtime = {}
    with mock.patch.dict(c.PROFILE, {'baseline_header_sha256': c.digest(baseline)}), \
            mock.patch.object(c, 'verify_inputs', return_value=segments):
        for arm in c.VARIANTS:
            header = baseline if arm == 'baseline' else c.replace_models(baseline, {k: weight_meta[arm + '-' + k + '.npz'] for k in c.KINDS})
            binary = b'\x7fELFinvented-nonexecuted'
            runtime[arm + '.h'], runtime[arm + '.bin'] = header, binary
            build['variants'].append({'arm': arm, 'command': c.command_for(build['compiler'], c.ROOT, Path(build['build_root']) / arm),
                                      'header_sha256': c.digest(header), 'binary_sha256': c.digest(binary), 'exit_code': 0})
        training = {'schema': 'english63-background-ab-training-v1', 'status': 'COMPLETED', 'optimizer_updates_started': 192,
                    'optimizer_updates_completed': 192, 'active_training': None, 'group_wall_seconds': 4, 'arms': []}
        for kind, arm in c.ARMS:
            training['arms'].append({'kind': kind, 'arm': arm, 'status': 'COMPLETED', 'optimizer_updates': 48,
                'seed': c.PROFILE['seeds'][kind], 'warm_start_sha256': c.p.c.SOURCE_WEIGHTS[kind], 'optimizer': 'fresh-AdamW',
                'learning_rate': .001, 'weight_decay': .01, 'gradient_clip': 1, 'original_train_rows': 30,
                'natural_train_rows': 128 if arm == 'treatment' else 0, 'natural_coefficient': .5 if arm == 'treatment' else 0,
                'train_development_rows': 0, 'train_nearphone_rows': 0, 'model_promotion_allowed': False,
                'weights': weight_meta[arm + '-' + kind + '.npz'][0], 'warm_start_parameter_payload_sha256':
                c.digest(c.baseline_payloads(baseline)[kind]), 'changed_state_tensors': 1,
                'runtime': {'numpy': '2.3.5', 'torch': '2.10.0+cpu', 'threads': 1, 'mkldnn': False},
                'child_wall_seconds': 1, 'child_exit_code': 0, 'child_peak_rss_kib': 1, 'timed_out': False,
                'updates': [{'step': i, 'old30_mil_ce': 1., 'new128_mil_ce': 2. if arm == 'treatment' else None,
                             'objective': 2. if arm == 'treatment' else 1., 'gradient_norm': 1.} for i in range(1, 49)]})
        integrity = {'source_before': hashes, 'source_after': hashes,
                     'corpus_before': c.s.CORPUS['sha256'], 'corpus_after': c.s.CORPUS['sha256'],
                     'request_before': inputs['request_sha256'], 'request_after': inputs['request_sha256'],
                     'claim_before': inputs['claim_sha256'], 'claim_after': inputs['claim_sha256'],
                     'trained_files_before': {}, 'trained_files_after': {}, 'evaluation_pcm_before': {}, 'evaluation_pcm_after': {},
                     'runtime_before': {n: c.digest(raw) for n, raw in runtime.items()},
                     'runtime_after': {n: c.digest(raw) for n, raw in runtime.items()}}
        measurements = {'schema': 'english63-background-ab-measurements-v1', 'public_run': identity,
                        'pipeline_head': build['pipeline_head'], 'initial_allowed_cpus': [0], 'selected_cpu': 0,
                        'final_allowed_cpus': [0], 'native_group_wall_seconds': 1, 'source_integrity': integrity, 'passes': []}
        models = []
        for arm, kind in c.PASSES:
            body = {'schema': 'english63-background-ab-native-v1', 'kind': kind, 'pcm_samples': 11518484,
                    'frames': 71965, 'feed_calls': 71991, 'proposal_count': 0, 'state_bytes': 38192,
                    'weight_parameter_bytes': (6531 if kind == 'gru' else 4803) * 4, 'frontend_table_bytes': 3894,
                    'pool_time_major': 0, 'affinity_cpu': 0, 'affinity_single_cpu': True, 'initializations': 13,
                    'finish_calls': 13, 'reset_calls': 0, 'discontinuity_calls': 0, 'eof_padding_samples': 0, 'eof_events': 0,
                    'cpu_ns': 1000, 'wall_ns': 1000, 'segments': [{key: r[key] for key in
                     ('segment_index', 'samples', 'frames', 'feed_calls', 'final_feed_samples', 'proposal_count')}
                     for r in observed['segments']]}
            lifetime = {'schema': 'english63-background-ab-child-lifetime-v1', 'child_exit_code': 0,
                        'child_signaled': False, 'process_lifetime_peak_rss_kib': 1}
            record = {'arm': arm, 'kind': kind, 'status': 'COMPLETED', 'exit_code': 0, 'timed_out': False, 'raw_stderr': '',
                      'command': [str(Path(build['build_root']) / arm / 'ab_replay'), '--supervise', kind,
                                  str(Path(build['build_root']) / 'evaluation.txt'), str(root / (arm + '-' + kind + '.trace')),
                                  '0', str(os.getpid())], 'raw_stdout': json.dumps(body) + '\n' + json.dumps(lifetime) + '\n'}
            measurements['passes'].append(record)
            models.append(c.pass_observation(record, arm, kind, build, 0, observed))
            (root / (arm + '-' + kind + '.trace')).write_bytes(trace)
        for name, value in (('PROTOCOL.json', c.protocol()), ('INPUTS.json', inputs), ('BUILD.json', build),
                            ('TRAINING.json', training), ('MEASUREMENTS.json', measurements)):
            c.p.c.write_json(root / name, value)
        write_zip(root / 'WEIGHTS.zip', weights)
        write_zip(root / 'RUNTIME.zip', runtime)
        write_zip(root / 'SOURCE.zip', {n: (c.ROOT / n).read_bytes() for n in c.SOURCES})
        (root / 'native.stderr').write_bytes(b'')
        summary = c.summary_from({**inputs, 'file_sha256': c.s.sha_file(root / 'INPUTS.json')}, build,
                                 {**training, 'file_sha256': c.s.sha_file(root / 'TRAINING.json')},
                                 {**measurements, 'file_sha256': c.s.sha_file(root / 'MEASUREMENTS.json')}, models)
        c.p.c.write_json(root / 'SUMMARY.json', summary)
        c.p.c.write_json(root / 'REPLAY.json', c.saved_receipt(root))
        c.verify_saved(root)
        def rehash_receipt(name):
            receipt = c.p.read_json(root / 'REPLAY.json')
            identity = c.p.artifact_identity((root / name).read_bytes())
            receipt['files'][name] = identity
            receipt['files_sha256'][name] = identity['sha256']
            if name == 'SUMMARY.json':
                receipt['summary_sha256'] = identity['sha256']
            (root / 'REPLAY.json').write_text(json.dumps(receipt))
        training_raw = (root / 'TRAINING.json').read_bytes()
        for key, value in (('timed_out', True), ('child_exit_code', False)):
            forged_training = copy.deepcopy(training)
            forged_training['arms'][0][key] = value
            (root / 'TRAINING.json').write_text(json.dumps(forged_training))
            rehash_receipt('TRAINING.json')
            rejected(lambda: c.verify_saved(root), 'AB_TRAINING_ARM')
        (root / 'TRAINING.json').write_bytes(training_raw)
        rehash_receipt('TRAINING.json')
        summary_raw = (root / 'SUMMARY.json').read_bytes()
        (root / 'SUMMARY.json').write_text(json.dumps({**summary, 'decision': 'RESEARCH_BENEFIT'}))
        rehash_receipt('SUMMARY.json')
        rejected(lambda: c.verify_saved(root), 'AB_SUMMARY_RECOMPUTED')
        (root / 'SUMMARY.json').write_bytes(summary_raw)
        rehash_receipt('SUMMARY.json')
        forged = bytearray(trace)
        struct.pack_into('<I', forged, 16 + 24 + 51 * 32 + 28, 1)
        (root / 'baseline-gru.trace').write_bytes(forged)
        rehash_receipt('baseline-gru.trace')
        rejected(lambda: c.verify_saved(root), 'LIBRISPEECH_RAW_PROBABILITY_DECODER_EVENT')


def not_running(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        return path.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    except (FileNotFoundError, ProcessLookupError):
        # A /proc entry may disappear between enumeration and read.
        return True


def procfs_race_cases():
    with mock.patch.object(Path, 'read_text', side_effect=ProcessLookupError):
        require(not_running(12345), 'PROCFS_DISAPPEARING_PID')


def live_children(parent):
    children = []
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            try:
                state, ppid, *_ = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(ppid) == parent and state != 'Z':
                    children.append(int(path.name))
            except (FileNotFoundError, ProcessLookupError):
                pass
    return children


def await_file(path):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        if path.is_file() and path.stat().st_size:
            return path.read_text()
        time.sleep(.01)
    raise ValueError('AB_FIXTURE_CHILD_DID_NOT_START')


def await_dead(pids):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        if all(not_running(pid) for pid in pids):
            return
        time.sleep(.01)
    raise ValueError('AB_FIXTURE_SURVIVING_CHILD:' + str(pids))


def process_cases(temp, binary):
    procfs_race_cases()
    cpu = min(os.sched_getaffinity(0))
    result = execute.bounded_child(lambda: time.sleep(30), .1, cpu)
    require(result['timed_out'] and result['child_exit_code'] == -signal.SIGKILL and result['child_wall_seconds'] < 2,
            'TRAIN_HARD_TIMEOUT')
    for group_kill in (False, True):
        marker = temp / ('training-pid-' + str(group_kill))
        script = ('import sys,os,time,pathlib;sys.path.insert(0,' + repr(str(HERE)) + ');import run_ab as r;'
                  'r.bounded_child(lambda:(pathlib.Path(' + repr(str(marker)) + ').write_text(str(os.getpid())),time.sleep(30)),20,' + str(cpu) + ')')
        parent = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', script], start_new_session=True)
        try:
            child = int(await_file(marker))
            require(os.getpgid(child) == parent.pid, 'TRAIN_INHERITS_OUTER_GROUP')
            if group_kill:
                os.killpg(parent.pid, signal.SIGKILL)
            else:
                parent.kill()
            parent.wait(timeout=5)
            await_dead([child])
        finally:
            if parent.poll() is None:
                os.killpg(parent.pid, signal.SIGKILL)
                parent.wait(timeout=5)
    fifo = temp / 'invented-blocked.pcm'
    os.mkfifo(fifo)
    manifest = temp / 'blocked-inputs.txt'
    manifest.write_text((str(fifo) + '\n') * 13)
    for group_kill in (False, True):
        marker = temp / ('native-pid-' + str(group_kill))
        trace = temp / ('blocked-' + str(group_kill) + '.trace')
        command = [str(binary), '--supervise', 'gru', str(manifest), str(trace), str(cpu)]
        script = ('import subprocess,os,time,pathlib;p=subprocess.Popen(' + repr(command) + '+[str(os.getpid())]);'
                  'pathlib.Path(' + repr(str(marker)) + ').write_text(str(p.pid));time.sleep(30)')
        parent = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', script], start_new_session=True,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            supervisor = int(await_file(marker))
            end = time.monotonic() + 5
            children = []
            while time.monotonic() < end:
                children = live_children(supervisor)
                if children:
                    break
                time.sleep(.01)
            require(len(children) == 1 and os.getpgid(supervisor) == parent.pid and
                    os.getpgid(children[0]) == parent.pid, 'NATIVE_INHERITS_OUTER_GROUP')
            if group_kill:
                os.killpg(parent.pid, signal.SIGKILL)
            else:
                parent.kill()
            parent.wait(timeout=5)
            await_dead([supervisor, *children])
        finally:
            if parent.poll() is None:
                os.killpg(parent.pid, signal.SIGKILL)
                parent.wait(timeout=5)


def c_cases(compiler, flags, temp):
    def array(kind, name, values):
        return 'static const ' + kind + ' ' + name + '[' + str(len(values)) + ']={' + ','.join(map(str, values)) + '};\n'
    header = '#include "en63_stream.h"\n'
    for name, values in (('w', [1]*400), ('mean', [0]*32), ('std', [1]*32), ('mel', [1]*32)):
        header += array('float', name, values)
    for name, values in (('start', [0]*32), ('count', [1]*32), ('offset', list(range(33)))):
        header += array('uint16_t', name, values)
    for kind, size, tag in (('gru', 6531, 1), ('dsc_tcn', 4803, 2)):
        header += f'static const float {kind}_p[{size}]={{[{size-2}]=4}};\n'
        header += f'static const en63_model en63_{kind}_model={{ {tag}, {size}, {kind}_p, w, mean, std, start, count, offset, mel, 32, "invented" }};\n'
    header += '#define EN63_FRONTEND_TABLE_BYTES 2178u\n'
    (temp / 'en63_generated.h').write_text(header)
    binary = temp / 'invented-ab-replay'
    subprocess.run([compiler, *flags, '-DEN63_POOL_TIME_MAJOR=0', '-I', str(c.s.HERE), '-I', str(temp),
                    str(c.s.HERE / 'en63_stream.c'), str(HERE / 'ab_replay.c'), '-lm', '-o', str(binary)], check=True)
    first, second = temp / 'first.pcm', temp / 'second.pcm'
    first.write_bytes(b'\x01\x00' * 8560)
    second.write_bytes(b'\xff\xff' * 1524)
    combined = temp / 'joined.pcm'
    require(c.s.join_pcm_files([first, second], combined) == 10084 and
            combined.read_bytes() == first.read_bytes() + second.read_bytes(), 'BYTE_PRESERVING_SPLICE')
    samples = [10084] + [9000 + i for i in range(12)]
    paths = [combined]
    for i, count in enumerate(samples[1:]):
        path = temp / ('isolated-' + str(i) + '.pcm')
        path.write_bytes(b'\x01\x00' * count)
        paths.append(path)
    manifest = temp / 'evaluation.txt'
    manifest.write_text(''.join(str(p) + '\n' for p in paths))
    segments = make_segments(samples)
    segments[0]['rows'] = [{'member': 'first', 'sample_count': 8560, 'concatenation_start_sample': 0},
                           {'member': 'second', 'sample_count': 1524, 'concatenation_start_sample': 8560}]
    for kind in c.KINDS:
        trace = temp / (kind + '.trace')
        raw = subprocess.check_output([str(binary), '--supervise', kind, str(manifest), str(trace),
                                       str(min(os.sched_getaffinity(0))), str(os.getpid())], text=True, timeout=30)
        body, lifetime = [json.loads(line) for line in raw.splitlines()]
        observed = c.trace_observation(trace.read_bytes(), segments)
        require(body['initializations'] == body['finish_calls'] == 13 and body['reset_calls'] == 0 and
                body['eof_padding_samples'] == body['eof_events'] == 0 and body['frames'] == observed['frames'] and
                body['feed_calls'] == observed['feed_calls'] and body['pcm_samples'] == sum(samples) and
                lifetime['child_exit_code'] == 0 and all(r['events'][0]['frame'] == 52 for r in observed['segments']),
                'C_REAL_FILE_BOUNDARIES_NO_PADDING')
    process_cases(temp, binary)


def partial_case(temp):
    outer = temp / 'partial-result'
    output = outer / 'background-continuation'
    output.mkdir(parents=True)
    working = temp / 'partial-working'
    working.mkdir()
    completed = c.ARMS[:2]
    for kind, arm in completed:
        (working / (arm + '-' + kind + '.npz')).write_bytes(npz(kind, True))
        execute.preserve_weights(output, working, completed[:completed.index((kind, arm)) + 1])
    shutil.rmtree(working)  # Simulated later-arm failure and TemporaryDirectory cleanup.
    names = {arm + '-' + kind + '.npz' for kind, arm in completed}
    require(set(c.zip_members((output / 'WEIGHTS.zip').read_bytes(), names)) == names and
            {p.name for p in output.iterdir()} == {'WEIGHTS.zip'}, 'COMPLETED_WEIGHTS_SURVIVE_FAILURE')


def run(compiler, flags, temporary):
    temp = temporary / 'background-ab-cases'
    temp.mkdir()
    scalar_cases()
    gate_cases()
    npz_cases()
    trace_cases()
    saved_forgery_case(temp / 'saved-forgery')
    partial_case(temp)
    c_cases(compiler, flags, temp)
    rejected(lambda: execute.main(['--verify-result-dir', str(temp), '--source-tar', '/invented.tar']), 'AB_SAVED_ONLY_ARGUMENTS')
    require(not any(name in sys.modules for name in ('numpy', 'torch', 'soundfile')), 'STDLIB_ONLY_FIXTURES')
    print('PASS background A/B scalar gradients, safety gates, exact NPZ/header, 13-stream C boundaries, '
          'rehash-resistant saved evidence, partial checkpoint retention, hard timeout/parent death; retained-model/audio executions=0')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='en63-background-source-test-') as directory:
        run(shutil.which('cc'), ['-std=c11', '-O2', '-ffp-contract=off', '-fno-fast-math',
                                '-Wall', '-Wextra', '-Werror', '-pedantic'], Path(directory))
