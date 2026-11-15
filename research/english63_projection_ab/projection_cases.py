"""Invented gradient/NPZ/result fixtures; no retained audio, checkpoints, or model execution."""
from __future__ import annotations

import copy
from contextlib import ExitStack
import io
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import projection_contract as c
import run_projection as execute
old = c.load('_en63_projection_old_fixtures', c.ROOT / 'research/english63_background_ab/ab_cases.py')


def require(value, marker):
    if not value:
        raise ValueError('PROJECTION_FIXTURE_' + marker)


def rejected(action, marker):
    try:
        action()
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        require(marker in str(error), 'EXPECTED_' + marker + '_GOT_' + str(error))
    else:
        raise ValueError('PROJECTION_FIXTURE_ACCEPTED_' + marker)


def npy(shape, payload, bad=None):
    header = {'descr': '<f4', 'fortran_order': False, 'shape': shape}
    if bad == 'object':
        header['descr'] = '|O'
    elif bad == 'fortran':
        header['fortran_order'] = True
    elif bad == 'list':
        header['shape'] = list(shape)
    text = repr(header).encode('ascii')
    text += b' ' * ((64 - (10 + len(text) + 1) % 64) % 64) + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(text)) + text + payload


def zip_bytes(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(name, raw)
    return buffer.getvalue()


def npz(kind, first_value):
    members = {}
    for i, (name, shape) in enumerate(c.exporter.PARAMETERS[kind]):
        values = [0.] * math.prod(shape)
        if i == 0:
            values[0] = first_value
        members[name + '.npy'] = npy(shape, c.pack_vector(values))
    return zip_bytes(members)


def projection_cases():
    combined, stats = c.projection_vectors([1., 2.], [-3., 0.])
    require(combined == [c.f32(-1.4), c.f32(3.2)] and stats['projection_applied'] and
            stats['old_gradient_squared_norm'] == 5 and stats['old_background_dot'] == -3 and
            stats['projection_coefficient'] == -.6 and abs(stats['old_projected_background_dot']) < 1e-15,
            'GLOBAL_ONE_SIDED_PROJECTION')
    # Projecting each scalar separately would give [1,2], and is deliberately rejected.
    rejected(lambda: c.gradient_diagnostics([1., 2.], [-3., 0.], [1., 2.], [1., 2.], [0., 0.], 1., 1.),
             'PROJECTION_EXACT_COMBINED')
    combined, stats = c.projection_vectors([1., 0.], [1., 2.])
    require(combined == [2., 2.] and not stats['projection_applied'] and not stats['zero_reference'],
            'NONCONFLICT_UNCHANGED')
    combined, stats = c.projection_vectors([0., 0.], [-1., 2.])
    require(combined == [-1., 2.] and not stats['projection_applied'] and stats['zero_reference'] and
            stats['old_background_cosine'] is None, 'ZERO_ANCHOR_NO_EPSILON')
    tiny = c.f32(2 ** -130)
    combined, stats = c.projection_vectors([tiny, 0.], [-tiny, tiny])
    require(stats['projection_applied'] and not stats['zero_reference'] and combined == [tiny, tiny],
            'SUBNORMAL_ANCHOR_NOT_DISCARDED')
    rejected(lambda: c.projection_vectors([float('nan')], [1.]), 'PROJECTION_GRADIENT_FINITE')
    rejected(lambda: c.projection_vectors([1.], [float('inf')]), 'PROJECTION_GRADIENT_FINITE')
    tape = []
    class Scalar:
        def __init__(self, value, derivative=1):
            self.value, self.derivative = value, derivative
        def detach(self):
            return self.value
        def __mul__(self, coefficient):
            return Scalar(self.value * coefficient, self.derivative * coefficient)
        def backward(self):
            tape.append(self.derivative)
    require(c.accumulate_background(lambda i: Scalar([2, 4, 6, 8][i])) == 5 and tape == [.125] * 4,
            'FOUR_IDENTICAL_WEIGHTED_MICROBATCHES')
    for norm in (0., c.f32(.25), c.f32(1.), c.f32(math.sqrt(5)), c.f32(1234.125)):
        expected = min(1., c.f32(1. / c.f32(norm + c.f32(1e-6))))
        require(c.clip_coefficient(norm) == expected, 'FP32_CLIP_ROUNDING')
    require(c.clip_coefficient(c.f32(1.)) < 1., 'TORCH_CLIP_EPSILON_APPLIES_AT_ONE')


def gradient_fixture(kind):
    count = c.parameter_count(kind)
    payloads = {name: bytearray() for name in c.TAPE_FIELDS}
    old_gradient, background = [0.] * count, [0.] * count
    old_gradient[0], background[0], background[1] = 2., -1., 1.
    combined, _ = c.projection_vectors(old_gradient, background)
    norm = c.f32(math.sqrt(5.))
    factor = c.clip_coefficient(norm)
    clipped = [c.f32(value * factor) for value in combined]
    updates = []
    for i in range(48):
        after, delta = [0.] * count, [0.] * count
        after[0], delta[0] = (i + 1) / 128, 1 / 128
        stats = c.gradient_diagnostics(old_gradient, background, combined, clipped, delta, norm, factor)
        updates.append({'step': i + 1, 'training_forward_calls': 5, 'training_padded_forward_frames': 48644,
                        'old30_mil_ce': 1., 'old30_mil_ce_by_class': {'0': 1., '1': 1., '2': 1.},
                        'new128_mil_ce': 2., 'weighted_loss_value': 2., 'gradient_norm': norm,
                        'clip_factor': factor, **stats})
        for name, row in zip(c.TAPE_FIELDS, (old_gradient, background, combined, clipped, delta, after)):
            payloads[name].extend(c.pack_vector(row))
    raw = zip_bytes({name + '.npy': npy((48, count), value) for name, value in payloads.items()})
    return raw, updates


def fake_export(baseline, models):
    # The old source suite independently tests its immutable exact C exporter.
    return baseline + b''.join(meta['npz_sha256'].encode() + payload for meta, payload in models.values())


def saved_case(root):
    root.mkdir()
    samples = [11124244] + [32000] * 11 + [42240]
    segments = old.make_segments(samples)
    trace = old.invented_trace(segments)
    observation = c.trace_observation(trace, segments)
    baseline = old.fake_header()
    zero_payloads = {kind: c.npz_model(npz(kind, 0.), kind)[1] for kind in c.KINDS}
    weights = {arm + '-' + kind + '.npz': npz(kind, .125) for arm in ('control', 'treatment') for kind in c.KINDS}
    weights.update({arm + '-' + kind + '.npz': npz(kind, .375) for kind, arm in c.ARMS})
    weight_meta = {name: c.npz_model(raw, 'dsc_tcn' if 'dsc_tcn' in name else 'gru') for name, raw in weights.items()}
    frozen_pins = {name: c.digest(weights[name]) for name in c.FROZEN_WEIGHT_FILES}
    identity = {'repository': 'jiying2007/kws-data', 'head': '1' * 40, 'run_id': '123', 'run_attempt': '1'}
    hashes = c.source_hashes()
    inputs = {'request_sha256': '2' * 64, 'claim_sha256': '3' * 64, 'trained_files_sha256': {},
              'background_files_sha256': {}, 'evaluation_pcm_sha256': {}}
    build = {'schema': 'english63-projection-ab-build-v1', 'compiler': '/usr/bin/gcc', 'compiler_realpath': '/usr/bin/gcc',
             'compiler_sha256': '4' * 64, 'compiler_version': 'invented', 'pipeline_head': '5' * 40,
             'source_root': str(c.ROOT), 'build_root': str(root / 'invented-build'), 'output_root': str(root),
             'entrypoint_pid': os.getpid(), 'public_run': identity, 'physical_ssc305': False,
             'source_sha256': hashes, 'variants': []}
    runtime, gradients, tapes = {}, {}, {}
    training = {'schema': 'english63-projection-ab-training-v1', 'status': 'COMPLETED',
                'optimizer_updates_started': 96, 'optimizer_updates_completed': 96,
                'training_forward_calls_started': 480, 'training_forward_calls_completed': 480,
                'training_padded_frames_started': 4669824, 'training_padded_frames_completed': 4669824,
                'active_training': None, 'group_wall_seconds': 4, 'arms': []}
    for kind, arm in c.ARMS:
        tape_raw, updates = gradient_fixture(kind)
        gradients[kind + '-gradients.npz'] = tape_raw
        tapes[kind + '-gradients.npz'] = c.tape_model(tape_raw, kind)
        training['arms'].append({'kind': kind, 'arm': arm, 'status': 'COMPLETED', 'optimizer_updates': 48,
            'seed': c.PROFILE['seeds'][kind], 'warm_start_sha256': c.p.c.SOURCE_WEIGHTS[kind], 'optimizer': 'fresh-AdamW',
            'learning_rate': .001, 'weight_decay': .01, 'gradient_clip': 1, 'original_train_rows': 30,
            'natural_train_rows': 128, 'natural_coefficient': .5, 'train_development_rows': 0, 'train_nearphone_rows': 0,
            'model_promotion_allowed': False, 'parameter_order': c.parameter_order(kind),
            'weights': weight_meta[arm + '-' + kind + '.npz'][0],
            'warm_start_parameter_payload_sha256': c.digest(zero_payloads[kind]), 'changed_state_tensors': 1,
            'gradient_tape': tapes[kind + '-gradients.npz'][0],
            'runtime': {'numpy': '2.3.5', 'torch': '2.10.0+cpu', 'threads': 1, 'mkldnn': False},
            'child_wall_seconds': 1, 'child_exit_code': 0, 'child_peak_rss_kib': 1, 'timed_out': False,
            'updates': updates})
    prior_models = [{**observation, 'arm': arm, 'kind': kind, 'metrics': c.event_metrics(observation)}
                    for arm in ('baseline', 'control', 'treatment') for kind in c.KINDS]
    historical_globals = c.p.canonical(c.ab.FIXED)
    with ExitStack() as patches:
        patches.enter_context(mock.patch.dict(c.PROFILE, {'baseline_header_sha256': c.digest(baseline)}))
        patches.enter_context(mock.patch.dict(c.FIXED['source_artifacts']['background'], {'weights_sha256': frozen_pins}))
        patches.enter_context(mock.patch.object(c, 'verify_inputs', return_value=segments))
        patches.enter_context(mock.patch.object(c, 'verify_background_evidence', return_value=({}, {'models': prior_models}, {})))
        patches.enter_context(mock.patch.object(c, 'baseline_payloads', return_value=zero_payloads))
        patches.enter_context(mock.patch.object(c, 'replace_models', side_effect=fake_export))
        for arm in c.VARIANTS:
            header = baseline if arm == 'baseline' else fake_export(baseline, {kind: weight_meta[arm + '-' + kind + '.npz'] for kind in c.KINDS})
            binary = b'\x7fELFinvented-nonexecuted'
            runtime[arm + '.h'], runtime[arm + '.bin'] = header, binary
            build['variants'].append({'arm': arm, 'command': c.command_for(build['compiler'], c.ROOT, Path(build['build_root']) / arm),
                                     'header_sha256': c.digest(header), 'binary_sha256': c.digest(binary), 'exit_code': 0})
        integrity = {'source_before': hashes, 'source_after': hashes,
                     'corpus_before': c.s.CORPUS['sha256'], 'corpus_after': c.s.CORPUS['sha256'],
                     'request_before': inputs['request_sha256'], 'request_after': inputs['request_sha256'],
                     'claim_before': inputs['claim_sha256'], 'claim_after': inputs['claim_sha256'],
                     'trained_files_before': {}, 'trained_files_after': {}, 'background_files_before': {}, 'background_files_after': {},
                     'evaluation_pcm_before': {}, 'evaluation_pcm_after': {},
                     'runtime_before': {name: c.digest(raw) for name, raw in runtime.items()},
                     'runtime_after': {name: c.digest(raw) for name, raw in runtime.items()}}
        measurements = {'schema': 'english63-projection-ab-measurements-v1', 'public_run': identity,
                        'pipeline_head': build['pipeline_head'], 'initial_allowed_cpus': [0], 'selected_cpu': 0,
                        'final_allowed_cpus': [0], 'native_group_wall_seconds': 1, 'source_integrity': integrity, 'passes': []}
        models, traces = [], {}
        for arm, kind in c.PASSES:
            body = {'schema': 'english63-background-ab-native-v1', 'kind': kind, 'pcm_samples': 11518484,
                    'frames': 71965, 'feed_calls': 71991, 'proposal_count': 0, 'state_bytes': 38192,
                    'weight_parameter_bytes': c.parameter_count(kind) * 4, 'frontend_table_bytes': 3894,
                    'pool_time_major': 0, 'affinity_cpu': 0, 'affinity_single_cpu': True, 'initializations': 13,
                    'finish_calls': 13, 'reset_calls': 0, 'discontinuity_calls': 0, 'eof_padding_samples': 0, 'eof_events': 0,
                    'cpu_ns': 1000, 'wall_ns': 1000, 'segments': [{key: row[key] for key in
                     ('segment_index', 'samples', 'frames', 'feed_calls', 'final_feed_samples', 'proposal_count')}
                     for row in observation['segments']]}
            lifetime = {'schema': 'english63-background-ab-child-lifetime-v1', 'child_exit_code': 0,
                        'child_signaled': False, 'process_lifetime_peak_rss_kib': 1}
            record = {'arm': arm, 'kind': kind, 'status': 'COMPLETED', 'exit_code': 0, 'timed_out': False, 'raw_stderr': '',
                      'command': [str(Path(build['build_root']) / arm / 'ab_replay'), '--supervise', kind,
                                  str(Path(build['build_root']) / 'evaluation.txt'), str(root / (arm + '-' + kind + '.trace')),
                                  '0', str(os.getpid())], 'raw_stdout': json.dumps(body) + '\n' + json.dumps(lifetime) + '\n'}
            measurements['passes'].append(record)
            models.append(c.pass_observation(record, arm, kind, build, 0, observation))
            traces[arm + '-' + kind + '.trace'] = trace
        for name, value in (('PROTOCOL.json', c.protocol()), ('INPUTS.json', inputs), ('BUILD.json', build),
                            ('TRAINING.json', training), ('MEASUREMENTS.json', measurements)):
            c.p.c.write_json(root / name, value)
        for name, members in (('WEIGHTS.zip', weights), ('RUNTIME.zip', runtime), ('GRADIENTS.zip', gradients),
                              ('TRACES.zip', traces), ('SOURCE.zip', {name: (c.ROOT / name).read_bytes() for name in c.SOURCES})):
            (root / name).write_bytes(zip_bytes(members))
        (root / 'native.stderr').write_bytes(b'')
        summary = c.summary_from({**inputs, 'file_sha256': c.s.sha_file(root / 'INPUTS.json')}, build,
                                 {**training, 'file_sha256': c.s.sha_file(root / 'TRAINING.json')},
                                 {**measurements, 'file_sha256': c.s.sha_file(root / 'MEASUREMENTS.json')}, models)
        c.p.c.write_json(root / 'SUMMARY.json', summary)
        c.p.c.write_json(root / 'REPLAY.json', c.saved_receipt(root))
        require(c.verify_saved(root)['model_forward_calls'] == 0 and len(list(root.iterdir())) == 13,
                'COMPLETE_CLOSED_SAVED_REPLAY')
        for field, value, marker in (('old_background_dot', 1., 'PROJECTION_RECOMPUTED_DIAGNOSTIC'),
                                     ('projection_applied', False, 'PROJECTION_RECOMPUTED_DIAGNOSTIC'),
                                     ('clip_factor', 1., 'PROJECTION_CLIP_FACTOR')):
            forged = copy.deepcopy(training)
            forged['arms'][0]['updates'][0][field] = value
            rejected(lambda: c.verify_training(forged, inputs, weight_meta, baseline, tapes), marker)
        forged = copy.deepcopy(training)
        forged['arms'][0]['timed_out'] = True
        rejected(lambda: c.verify_training(forged, inputs, weight_meta, baseline, tapes), 'PROJECTION_TRAINING_ARM')
        changed = dict(weight_meta)
        changed['control-gru.npz'] = c.npz_model(npz('gru', .5), 'gru')
        rejected(lambda: c.verify_training(training, inputs, changed, baseline, tapes), 'PROJECTION_FROZEN_COMPARATOR_WEIGHT')
        # Forge a vector, update its own metadata, and still require recomputed geometry/parameter binding.
        for field, marker in (('combined_preclip', 'PROJECTION_EXACT_COMBINED'),
                              ('actual_clipped_gradient', 'PROJECTION_EXACT_CLIPPED_GRADIENT'),
                              ('actual_parameter_delta', 'PROJECTION_PARAMETER_DELTA_BINDING'),
                              ('parameter_after', 'PROJECTION_PARAMETER_DELTA_BINDING')):
            changed_tapes = copy.deepcopy(tapes)
            changed_payload = bytearray(changed_tapes['gru-gradients.npz'][1][field])
            struct.pack_into('<f', changed_payload, 0, 4.)
            changed_tapes['gru-gradients.npz'][1][field] = bytes(changed_payload)
            rejected(lambda: c.verify_training(training, inputs, weight_meta, baseline, changed_tapes), marker)
        forged_summary = {**summary, 'decision': 'RESEARCH_BENEFIT'}
        (root / 'SUMMARY.json').write_text(json.dumps(forged_summary))
        replay = c.p.read_json(root / 'REPLAY.json')
        identity = c.p.artifact_identity((root / 'SUMMARY.json').read_bytes())
        replay['files']['SUMMARY.json'] = identity
        replay['files_sha256']['SUMMARY.json'] = replay['summary_sha256'] = identity['sha256']
        (root / 'REPLAY.json').write_text(json.dumps(replay))
        rejected(lambda: c.verify_saved(root), 'PROJECTION_SUMMARY_RECOMPUTED')
        # Successful completed artifacts stay available after working-directory loss.
        work = root / 'partial-working'
        work.mkdir()
        for name, raw in weights.items():
            (work / name).write_bytes(raw)
        for name, raw in gradients.items():
            (work / name).write_bytes(raw)
        partial = root.parent / 'partial-result' / 'projection-continuation'
        partial.mkdir(parents=True)
        execute.preserve_weights(partial, work, c.ARMS[:1])
        require(set(c.zip_members((partial / 'WEIGHTS.zip').read_bytes(), c.FROZEN_WEIGHT_FILES | {'projected-gru.npz'})) ==
                c.FROZEN_WEIGHT_FILES | {'projected-gru.npz'} and
                set(c.zip_members((partial / 'GRADIENTS.zip').read_bytes(), {'gru-gradients.npz'}, 8 * 1024 * 1024)) ==
                {'gru-gradients.npz'}, 'PARTIAL_WEIGHTS_AND_TAPES_PRESERVED')
    require(c.p.canonical(c.ab.FIXED) == historical_globals, 'HISTORICAL_PROTOCOL_UNMUTATED')
    raw = gradients['gru-gradients.npz']
    members = c.zip_members(raw, {key + '.npy' for key in c.TAPE_FIELDS}, 2 * 1024 * 1024)
    for tag in ('object', 'fortran', 'list'):
        bad = dict(members)
        bad['old_gradient.npy'] = npy((48, c.parameter_count('gru')), tapes['gru-gradients.npz'][1]['old_gradient'], tag)
        rejected(lambda: c.tape_model(zip_bytes(bad), 'gru'), 'PROJECTION_TAPE_SCHEMA')
    bad = dict(members)
    payload = bytearray(tapes['gru-gradients.npz'][1]['old_gradient'])
    struct.pack_into('<f', payload, 0, float('nan'))
    bad['old_gradient.npy'] = npy((48, c.parameter_count('gru')), payload)
    rejected(lambda: c.tape_model(zip_bytes(bad), 'gru'), 'PROJECTION_TAPE_VALUES')


def run(compiler, flags, temporary):
    del compiler, flags  # C implementation is unchanged and exercised by the existing source suite.
    historical = c.p.canonical(c.ab.FIXED)
    projection_cases()
    saved_case(temporary / 'projection-saved-cases')
    cpu = min(os.sched_getaffinity(0))
    result = execute.bounded_child(lambda: time.sleep(30), .1, cpu)
    require(result['timed_out'] and result['child_exit_code'] == -signal.SIGKILL and result['child_wall_seconds'] < 2,
            'WORKER_HARD_TIMEOUT')
    with mock.patch.dict(os.environ, {}, clear=True):
        rejected(lambda: execute.run(SimpleNamespace()), 'PUBLIC_ACTIONS_FIRST_ATTEMPT_ONLY')
    rejected(lambda: execute.main(['--verify-result-dir', str(temporary), '--background-result-dir', '/invented-inner']),
             'PROJECTION_SAVED_ONLY_ARGUMENTS')
    require(c.p.canonical(c.ab.FIXED) == historical and not any(name in sys.modules for name in ('numpy', 'torch', 'soundfile')),
            'STDLIB_ONLY_AND_IMMUTABLE_HISTORY')
    print('PASS projection global/zero/subnormal geometry, four .125 gradients, exact FP32 clip, six-array '
          'parameter chain, closed 8-trace/2-tape replay, forged evidence rejection, preserved partial artifacts; '
          'retained-model/audio executions=0')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='en63-projection-source-test-') as directory:
        run(None, None, Path(directory))
