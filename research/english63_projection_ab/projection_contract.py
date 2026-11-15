"""Frozen projection continuation and stdlib-only saved-vector replay."""
from __future__ import annotations

import ast
import io
import math
from pathlib import Path
import struct
import zipfile
import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ab = load('_en63_projection_background_contract', ROOT / 'research/english63_background_ab/ab_contract.py')
s, p, need, exact, digest, parse_json = ab.s, ab.p, ab.need, ab.exact, ab.digest, ab.parse_json
exporter = ab.exporter
npz_model, zip_members = ab.npz_model, ab.zip_members
baseline_payloads, replace_models, changed_tensors = ab.baseline_payloads, ab.replace_models, ab.changed_tensors
natural_plan, geometry, trace_observation, pass_observation = ab.natural_plan, ab.geometry, ab.trace_observation, ab.pass_observation
command_for, benefit, event_metrics = ab.command_for, ab.benefit, ab.event_metrics
FIXED = parse_json((HERE / 'protocol.json').read_bytes())
PROFILE = FIXED['profile']
KINDS = ab.KINDS
VARIANTS = ('baseline', 'control', 'treatment', 'projected')
ARMS = [(kind, 'projected') for kind in KINDS]
PASSES = [(arm, kind) for arm in VARIANTS for kind in KINDS]
FALSE_FIELDS = ab.FALSE_FIELDS
SOURCES = (*ab.SOURCES, 'research/english63_projection_ab/run_projection.py',
           'research/english63_projection_ab/projection_contract.py', 'research/english63_projection_ab/protocol.json')
RESULT_FILES = {'PROTOCOL.json', 'INPUTS.json', 'SOURCE.zip', 'TRAINING.json', 'BUILD.json',
                'MEASUREMENTS.json', 'SUMMARY.json', 'REPLAY.json', 'native.stderr',
                'WEIGHTS.zip', 'RUNTIME.zip', 'TRACES.zip', 'GRADIENTS.zip'}
FROZEN_WEIGHT_FILES = set(FIXED['source_artifacts']['background']['weights_sha256'])
WEIGHT_FILES = FROZEN_WEIGHT_FILES | {arm + '-' + kind + '.npz' for kind, arm in ARMS}
TRACE_FILES = {arm + '-' + kind + '.trace' for arm, kind in PASSES}
RUNTIME_FILES = {arm + suffix for arm in VARIANTS for suffix in ('.h', '.bin')}
GRADIENT_FILES = {kind + '-gradients.npz' for kind in KINDS}
TAPE_FIELDS = ('old_gradient', 'weighted_background_gradient', 'combined_preclip',
               'actual_clipped_gradient', 'actual_parameter_delta', 'parameter_after')
TOLERANCE = 1e-6


def source_hashes():
    return {name: digest((ROOT / name).read_bytes()) for name in SOURCES}


def parameter_order(kind):
    rows, offset = [], 0
    for name, shape in exporter.PARAMETERS[kind]:
        count = math.prod(shape)
        rows.append({'name': name, 'shape': list(shape), 'offset': offset, 'count': count})
        offset += count
    return rows


def parameter_count(kind):
    return sum(row['count'] for row in parameter_order(kind))


def protocol():
    return {**FIXED, 'schema': 'english63-projection-ab-protocol-v1', 'classes': p.bridge.CLASSES,
            'training_order': [list(x) for x in ARMS], 'evaluation_order': [list(x) for x in PASSES],
            'roster_sha256': ab.ROSTER_SHA, 'tts_geometry_sha256': ab.TTS_GEOMETRY_SHA,
            'native_children': 8, 'logical_streams': 104, 'native_feed_calls': 575928,
            'native_supplied_samples': 92147872, 'unique_evaluation_seconds': 719.90525,
            'trace_header': 'EN63ABT1,<II version=1,segments=13>',
            'trace_segment': '<IIQQ index,record_bytes=32,samples,frames> then <QQ3fI> rows',
            'gradient_tape_members': sorted(GRADIENT_FILES), 'gradient_arrays': list(TAPE_FIELDS),
            'parameter_order': {kind: parameter_order(kind) for kind in KINDS},
            'source_role': 'already-exposed-read-speech-study-not-independent-holdout',
            'event_clock': 'dependency-end-exclusive;source-interval-is-not-acoustic-attribution',
            'qualification': 'NOT_RUN', 'no_benefit_is_successful_completion': True,
            'no_retry_or_result_driven_extension': True, **dict.fromkeys(FALSE_FIELDS, False)}


def validate_request(raw, head):
    value = parse_json(raw)
    need(type(value) is dict and set(value) == set(FIXED) | {'pipeline_commit', 'reason'} and
         all(exact(value[k], v) for k, v in FIXED.items()) and type(head) is str and
         p.SHA1.fullmatch(head) and value['pipeline_commit'] == head and
         type(value['reason']) is str and 1 <= len(value['reason'].strip()) <= 500, 'PROJECTION_FIXED_REQUEST')
    return value


def claim_expected(inputs, build):
    return {'schema': 'kws-research-v4-projection-continuation-start-claim-v1',
            'request_sha256': inputs['request_sha256'], 'pipeline_commit': build['pipeline_head'],
            'source_checkpoint_artifact_id': 14114695344, 'source_negative_artifact_id': 14090383839,
            'source_background_artifact_id': 14149230666,
            'source_background_request_sha256': FIXED['source_artifacts']['background']['request_sha256'],
            'run_id': build['public_run']['run_id'], 'run_attempt': '1', 'optimizer_updates': 0,
            'optimizer_update_budget': 96, 'status': 'CONSUMED_BEFORE_GRADIENT_PROJECTION'}


def verify_background_evidence(inputs):
    pin = FIXED['source_artifacts']['background']
    raw = inputs['background_raw_utf8']
    need(type(raw) is dict and set(raw) == {'INPUTS.json', 'SUMMARY.json', 'REPLAY.json'},
         'PROJECTION_BACKGROUND_EVIDENCE_CLOSURE')
    for name, key in (('INPUTS.json', 'inputs_sha256'), ('SUMMARY.json', 'summary_sha256'),
                      ('REPLAY.json', 'replay_sha256')):
        need(type(raw[name]) is str and digest(raw[name].encode()) == pin[key], 'PROJECTION_BACKGROUND_EVIDENCE_PIN')
    prior, summary, replay = [parse_json(raw[name]) for name in ('INPUTS.json', 'SUMMARY.json', 'REPLAY.json')]
    identity = {'repository': 'jiying2007/kws-data', 'head': pin['data_commit'],
                'run_id': str(pin['run_id']), 'run_attempt': '1'}
    need(summary['pipeline_head'] == pin['pipeline_commit'] and exact(summary['public_run'], identity) and
         prior['request_sha256'] == pin['request_sha256'] and summary['optimizer_updates'] == 192 and
         replay['schema'] == 'english63-background-ab-replay-v1' and
         replay['status'] == 'VERIFIED_SAVED_BACKGROUND_AB' and
         replay['summary_sha256'] == pin['summary_sha256'] and
         exact(inputs['background_files_sha256'], {**replay['files_sha256'], 'REPLAY.json': pin['replay_sha256']}),
         'PROJECTION_BACKGROUND_PRIOR_IDENTITY')
    ab.verify_inputs(prior, {'public_run': identity, 'pipeline_head': pin['pipeline_commit']})
    return prior, summary, replay


def verify_inputs(inputs, build):
    need(inputs['schema'] == 'english63-projection-ab-inputs-v1' and
         exact(inputs['source_artifacts'], FIXED['source_artifacts']), 'PROJECTION_INPUT_SCHEMA')
    prior, _, _ = verify_background_evidence(inputs)
    independent = {'schema', 'source_artifacts', 'request_raw_utf8', 'request_sha256', 'claim_raw_utf8', 'claim_sha256'}
    common = set(prior) - independent
    need(set(inputs) == set(prior) | {'background_raw_utf8', 'background_files_sha256'} and
         all(exact(inputs[key], prior[key]) for key in common), 'PROJECTION_IDENTICAL_DATA_AND_FEATURES')
    request, claim = inputs['request_raw_utf8'].encode(), inputs['claim_raw_utf8'].encode()
    validate_request(request, build['pipeline_head'])
    need(digest(request) == inputs['request_sha256'] and digest(claim) == inputs['claim_sha256'] and
         exact(parse_json(claim), claim_expected(inputs, build)), 'PROJECTION_CLAIM_BINDING')
    return inputs['evaluation_segments']


def accumulate_background(natural_losses):
    """Accumulate the same four .125-weighted microbatch gradients as the consumed study."""
    mean_value = 0.0
    for index in range(4):
        loss = natural_losses(index)
        value = float(loss.detach())
        need(finite_number(value) and value >= 0, 'PROJECTION_NATURAL_LOSS_FINITE')
        (loss * .125).backward()
        mean_value += value / 4
    return mean_value


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def dot(left, right):
    need(len(left) == len(right), 'PROJECTION_VECTOR_LENGTH')
    return math.fsum(a * b for a, b in zip(left, right))


def close(actual, expected, scale=None):
    if not finite_number(actual) or not finite_number(expected):
        return False
    size = max(abs(actual), abs(expected)) if scale is None else abs(scale)
    return abs(actual - expected) <= TOLERANCE * size


def pack_vector(values):
    return struct.pack('<' + str(len(values)) + 'f', *values)


def projection_vectors(old, weighted_background):
    """Exact frozen scalar arithmetic; one FP32 cast only after FP64 projection and addition."""
    need(len(old) == len(weighted_background) and old and
         all(finite_number(v) for v in (*old, *weighted_background)), 'PROJECTION_GRADIENT_FINITE')
    n, h2, d = dot(old, old), dot(weighted_background, weighted_background), dot(old, weighted_background)
    applied = d < 0 and n > 0
    coefficient = d / n if applied else 0.0
    projected = [h - coefficient * g if applied else h for g, h in zip(old, weighted_background)]
    combined = [f32(g + h) for g, h in zip(old, projected)]
    need(all(math.isfinite(v) for v in combined), 'PROJECTION_COMBINED_FINITE')
    projected_dot = dot(old, projected)
    projected_norm = math.sqrt(dot(projected, projected))
    old_norm, background_norm = math.sqrt(n), math.sqrt(h2)
    need(projected_dot >= -TOLERANCE * old_norm * background_norm and
         projected_norm <= (1 + TOLERANCE) * background_norm, 'PROJECTION_GEOMETRY')
    combined_dot = dot(old, combined)
    need(combined_dot >= n - TOLERANCE * old_norm * (old_norm + background_norm),
         'PROJECTION_COMBINED_FIRST_ORDER')
    stats = {'old_gradient_squared_norm': n, 'old_gradient_l2': old_norm,
             'weighted_background_gradient_l2': background_norm, 'old_background_dot': d,
             'old_background_cosine': d / (old_norm * background_norm) if old_norm > 0 and background_norm > 0 else None,
             'zero_reference': n == 0, 'projection_applied': applied, 'projection_coefficient': coefficient,
             'projected_background_gradient_l2': projected_norm, 'old_projected_background_dot': projected_dot,
             'combined_gradient_l2': math.sqrt(dot(combined, combined)), 'old_combined_gradient_dot': combined_dot}
    return combined, stats


def clip_coefficient(norm):
    need(finite_number(norm) and norm >= 0, 'PROJECTION_CLIP_NORM')
    return min(1.0, f32(1.0 / f32(norm + f32(1e-6))))


def tape_model(raw, kind):
    """Parse numeric-only six-array NPZ using stdlib; never imports NumPy or Torch."""
    count = parameter_count(kind)
    members = zip_members(raw, {key + '.npy' for key in TAPE_FIELDS}, 2 * 1024 * 1024)
    payloads = {}
    for name in TAPE_FIELDS:
        value = members[name + '.npy']
        need(value[:8] == b'\x93NUMPY\x01\x00' and len(value) >= 10, 'PROJECTION_TAPE_NPY_VERSION')
        end = 10 + struct.unpack_from('<H', value, 8)[0]
        need(end <= len(value) and end <= 4096, 'PROJECTION_TAPE_HEADER_BOUND')
        header = ast.literal_eval(value[10:end].decode('latin1'))
        need(type(header) is dict and type(header.get('shape')) is tuple and
             exact(header, {'descr': '<f4', 'fortran_order': False, 'shape': (48, count)}), 'PROJECTION_TAPE_SCHEMA')
        payload = value[end:]
        need(len(payload) == 48 * count * 4 and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', payload)),
             'PROJECTION_TAPE_VALUES')
        payloads[name] = payload
    return {'npz_sha256': digest(raw), 'dtype': '<f4', 'shape': [48, count], 'arrays': list(TAPE_FIELDS),
            'payload_sha256': {name: digest(payload) for name, payload in payloads.items()}}, payloads


def tape_row(payloads, name, index, count):
    return struct.unpack_from('<' + str(count) + 'f', payloads[name], index * count * 4)


def gradient_diagnostics(old, background, combined, clipped, delta, norm, factor):
    expected_combined, stats = projection_vectors(old, background)
    need(pack_vector(combined) == pack_vector(expected_combined), 'PROJECTION_EXACT_COMBINED')
    need(exact(norm, f32(norm)) and close(norm, stats['combined_gradient_l2']) and
         exact(factor, clip_coefficient(norm)),
         'PROJECTION_CLIP_FACTOR')
    expected_clipped = [f32(value * factor) for value in combined]
    need(pack_vector(clipped) == pack_vector(expected_clipped), 'PROJECTION_EXACT_CLIPPED_GRADIENT')
    return {**stats, 'actual_clipped_gradient_l2': math.sqrt(dot(clipped, clipped)),
            'old_parameter_delta_dot': dot(old, delta),
            'weighted_background_parameter_delta_dot': dot(background, delta),
            'parameter_delta_l2': math.sqrt(dot(delta, delta))}


def verify_training(training, inputs, weights, baseline, tapes):
    need(training['schema'] == 'english63-projection-ab-training-v1' and
         exact(training['optimizer_updates_started'], 96) and exact(training['optimizer_updates_completed'], 96) and
         exact(training['training_forward_calls_started'], 480) and exact(training['training_forward_calls_completed'], 480) and
         exact(training['training_padded_frames_started'], 4669824) and exact(training['training_padded_frames_completed'], 4669824) and
         training['active_training'] is None and training['status'] == 'COMPLETED' and
         finite_number(training['group_wall_seconds']) and 0 < training['group_wall_seconds'] <= 360 and
         len(training['arms']) == 2, 'PROJECTION_TRAINING_BUDGET')
    original_payloads = baseline_payloads(baseline)
    for name in FROZEN_WEIGHT_FILES:
        need(weights[name][0]['npz_sha256'] == FIXED['source_artifacts']['background']['weights_sha256'][name],
             'PROJECTION_FROZEN_COMPARATOR_WEIGHT')
    for record, (kind, arm) in zip(training['arms'], ARMS):
        meta, final_payload = weights[arm + '-' + kind + '.npz']
        tape_meta, payloads = tapes[kind + '-gradients.npz']
        fixed = {'kind': kind, 'arm': arm, 'status': 'COMPLETED', 'optimizer_updates': 48,
                 'seed': PROFILE['seeds'][kind], 'warm_start_sha256': p.c.SOURCE_WEIGHTS[kind],
                 'optimizer': 'fresh-AdamW', 'learning_rate': .001, 'weight_decay': .01,
                 'gradient_clip': 1, 'original_train_rows': 30, 'natural_train_rows': 128,
                 'natural_coefficient': .5, 'train_development_rows': 0, 'train_nearphone_rows': 0,
                 'model_promotion_allowed': False, 'parameter_order': parameter_order(kind)}
        need(all(exact(record[k], v) for k, v in fixed.items()) and exact(record['weights'], meta) and
             exact(record['gradient_tape'], tape_meta) and meta['npz_sha256'] != p.c.SOURCE_WEIGHTS[kind] and
             record['runtime']['numpy'] == '2.3.5' and record['runtime']['torch'] == '2.10.0+cpu' and
             record['runtime']['threads'] == 1 and record['runtime']['mkldnn'] is False and
             0 < record['child_wall_seconds'] <= 180 and exact(record['child_exit_code'], 0) and
             record['timed_out'] is False and p.is_int(record['child_peak_rss_kib'], 1) and
             len(record['updates']) == 48, 'PROJECTION_TRAINING_ARM')
        changes = changed_tensors(original_payloads[kind], final_payload, kind)
        need(changes > 0 and exact(record['changed_state_tensors'], changes) and
             record['warm_start_parameter_payload_sha256'] == digest(original_payloads[kind]),
             'PROJECTION_TRUE_WEIGHT_PAYLOAD_CHANGE')
        count = parameter_count(kind)
        previous = struct.unpack('<' + str(count) + 'f', original_payloads[kind])
        for i, update in enumerate(record['updates']):
            need(exact(update['step'], i + 1) and exact(update['training_forward_calls'], 5) and
                 exact(update['training_padded_forward_frames'], 48644) and
                 all(finite_number(update[key]) and update[key] >= 0 for key in
                     ('old30_mil_ce', 'new128_mil_ce', 'weighted_loss_value', 'gradient_norm', 'clip_factor')),
                 'PROJECTION_TRAINING_STEP')
            need(exact(update['weighted_loss_value'], update['old30_mil_ce'] + .5 * update['new128_mil_ce']),
                 'PROJECTION_LOSS_VALUES')
            by_class = update['old30_mil_ce_by_class']
            need(type(by_class) is dict and set(by_class) == {'0', '1', '2'} and
                 all(finite_number(value) and value >= 0 for value in by_class.values()) and
                 close(update['old30_mil_ce'], math.fsum(by_class.values()) / 3), 'PROJECTION_PER_CLASS_LOSS')
            rows = {name: tape_row(payloads, name, i, count) for name in TAPE_FIELDS}
            expected_delta = [f32(after - before) for after, before in zip(rows['parameter_after'], previous)]
            need(pack_vector(rows['actual_parameter_delta']) == pack_vector(expected_delta),
                 'PROJECTION_PARAMETER_DELTA_BINDING')
            stats = gradient_diagnostics(rows['old_gradient'], rows['weighted_background_gradient'],
                                         rows['combined_preclip'], rows['actual_clipped_gradient'],
                                         rows['actual_parameter_delta'], update['gradient_norm'], update['clip_factor'])
            need(set(update) == set(stats) | {'step', 'training_forward_calls', 'training_padded_forward_frames',
                 'old30_mil_ce', 'old30_mil_ce_by_class', 'new128_mil_ce', 'weighted_loss_value',
                 'gradient_norm', 'clip_factor'}, 'PROJECTION_STEP_CLOSURE')
            for key, expected in stats.items():
                need(exact(update[key], expected), 'PROJECTION_RECOMPUTED_DIAGNOSTIC:' + key)
            previous = rows['parameter_after']
        need(pack_vector(previous) == final_payload, 'PROJECTION_FINAL_PARAMETER_TAPE_BINDING')


def summary_from(inputs, build, training, measurements, models):
    lookup = {(r['arm'], r['kind']): r['metrics'] for r in models}
    gates = {kind: benefit(*(lookup[a, kind] for a in ('baseline', 'control', 'projected'))) for kind in KINDS}
    _, prior, _ = verify_background_evidence(inputs)
    previous = {(r['arm'], r['kind']): r for r in prior['models']}
    mechanisms = {}
    for record in training['arms']:
        steps = record['updates']
        mechanisms[record['kind']] = {
            'conflict_updates': sum(row['projection_applied'] for row in steps),
            'zero_reference_updates': sum(row['zero_reference'] for row in steps),
            'actual_old_loss_first_order_increase_updates': sum(row['old_parameter_delta_dot'] > 0 for row in steps),
            'actual_background_loss_first_order_increase_updates': sum(row['weighted_background_parameter_delta_dot'] > 0 for row in steps),
            'old30_mil_ce_initial': steps[0]['old30_mil_ce'], 'old30_mil_ce_last_preupdate': steps[-1]['old30_mil_ce'],
            'new128_mil_ce_initial': steps[0]['new128_mil_ce'], 'new128_mil_ce_last_preupdate': steps[-1]['new128_mil_ce']}
    return {'schema': 'english63-projection-ab-result-v1', 'status': 'MEASURED_EXPOSED_GRADIENT_PROJECTION_NOT_QUALIFICATION',
            'experiment': FIXED['experiment'], 'public_run': build['public_run'], 'pipeline_head': build['pipeline_head'],
            'request_sha256': inputs['request_sha256'], 'source_sha256': build['source_sha256'],
            'source_artifacts': FIXED['source_artifacts'], 'profile': PROFILE, 'classes': p.bridge.CLASSES,
            'inputs_sha256': inputs['file_sha256'], 'training_sha256': training['file_sha256'],
            'measurements_sha256': measurements['file_sha256'], 'models': models, 'benefit_by_kind': gates,
            'mechanism_by_kind': mechanisms,
            'frozen_comparator_replay': {kind: {arm: {
                'metrics_equal_prior': exact(lookup[arm, kind], previous[arm, kind]['metrics']),
                'trace_equal_prior': next(r['trace_sha256'] for r in models if r['arm'] == arm and r['kind'] == kind) ==
                                     previous[arm, kind]['trace_sha256']} for arm in ab.VARIANTS} for kind in KINDS},
            'weighted_background_comparator_role': 'descriptive-mechanism-comparator-only',
            'decision': 'RESEARCH_BENEFIT' if all(v['decision'] == 'RESEARCH_BENEFIT' for v in gates.values()) else 'NO_BENEFIT',
            'optimizer_updates': 96, 'training_forward_calls': 480, 'training_padded_forward_frames': 4669824,
            'reused_frozen_training_arms': 4, 'native_forward_frames': 575720, 'native_feed_calls': 575928,
            'native_supplied_samples': 92147872, 'native_children': 8, 'logical_streams': 104,
            'native_initializations': 104, 'native_finishes': 104,
            'unique_evaluation_source_seconds': 719.90525, 'new_independent_audio_seconds': 0,
            'new_tts_asr_ipa_calls': 0, 'training_group_wall_seconds': training['group_wall_seconds'],
            'native_group_wall_seconds': measurements['native_group_wall_seconds'],
            'independent_english_frr': 'NOT_RUN', 'continuous_negative_far_per_hour': 'NOT_RUN',
            'ssc305_board_profile': 'NOT_RUN', **dict.fromkeys(FALSE_FIELDS, False)}


def saved_receipt(root):
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'PROJECTION_SAVED_ROOT')
    paths = list(root.iterdir())
    need({x.name for x in paths} in (RESULT_FILES, RESULT_FILES - {'REPLAY.json'}) and
         all(x.is_file() and not x.is_symlink() for x in paths) and
         sum(x.stat().st_size for x in paths) < 64 * 1024 * 1024, 'PROJECTION_RESULT_CLOSURE')
    files = {x.name: p.artifact_identity(x.read_bytes()) for x in sorted(paths) if x.name != 'REPLAY.json'}
    need(exact(p.read_json(root / 'PROTOCOL.json'), protocol()), 'PROJECTION_PROTOCOL')
    source = zip_members((root / 'SOURCE.zip').read_bytes(), SOURCES)
    hashes = {name: digest(raw) for name, raw in source.items()}
    need(exact(hashes, source_hashes()), 'PROJECTION_SOURCE_BYTES')
    inputs, build, training, measurements = [p.read_json(root / name) for name in
                                           ('INPUTS.json', 'BUILD.json', 'TRAINING.json', 'MEASUREMENTS.json')]
    p.public_run(build['public_run'])
    need(build['schema'] == 'english63-projection-ab-build-v1' and exact(build['source_sha256'], hashes) and
         p.SHA1.fullmatch(build['pipeline_head']) and build['physical_ssc305'] is False and
         all(type(build[k]) is str and Path(build[k]).is_absolute() for k in
             ('compiler', 'compiler_realpath', 'source_root', 'build_root', 'output_root')) and
         p.sha(build['compiler_sha256']) and type(build['compiler_version']) is str and build['compiler_version'] and
         p.is_int(build['entrypoint_pid'], 2), 'PROJECTION_BUILD_IDENTITY')
    segments = verify_inputs(inputs, build)
    weight_bytes = zip_members((root / 'WEIGHTS.zip').read_bytes(), WEIGHT_FILES)
    weights = {name: npz_model(raw, 'dsc_tcn' if 'dsc_tcn' in name else 'gru') for name, raw in weight_bytes.items()}
    tape_bytes = zip_members((root / 'GRADIENTS.zip').read_bytes(), GRADIENT_FILES, 8 * 1024 * 1024)
    tapes = {kind + '-gradients.npz': tape_model(tape_bytes[kind + '-gradients.npz'], kind) for kind in KINDS}
    runtime = zip_members((root / 'RUNTIME.zip').read_bytes(), RUNTIME_FILES)
    baseline = runtime['baseline.h']
    need(digest(baseline) == PROFILE['baseline_header_sha256'], 'PROJECTION_BASELINE_EXACT_EXPORT')
    verify_training(training, inputs, weights, baseline, tapes)
    need(len(build['variants']) == 4, 'PROJECTION_BUILD_VARIANTS')
    for report, arm in zip(build['variants'], VARIANTS):
        header = baseline if arm == 'baseline' else replace_models(baseline, {k: weights[arm + '-' + k + '.npz'] for k in KINDS})
        need(runtime[arm + '.h'] == header and runtime[arm + '.bin'].startswith(b'\x7fELF') and
             report['arm'] == arm and report['header_sha256'] == digest(header) and
             report['binary_sha256'] == digest(runtime[arm + '.bin']) and exact(report['exit_code'], 0) and
             exact(report['command'], command_for(build['compiler'], build['source_root'], Path(build['build_root']) / arm)),
             'PROJECTION_EXPORT_AND_BUILD_BINDING')
    cpus = measurements['initial_allowed_cpus']
    need(measurements['schema'] == 'english63-projection-ab-measurements-v1' and type(cpus) is list and cpus and
         all(p.is_int(x, 0, 1023) for x in cpus) and cpus == sorted(set(cpus)) and
         exact(measurements['selected_cpu'], cpus[0]) and exact(measurements['final_allowed_cpus'], cpus) and
         0 < measurements['native_group_wall_seconds'] <= 180 and len(measurements['passes']) == 8 and
         exact(measurements['public_run'], build['public_run']) and measurements['pipeline_head'] == build['pipeline_head'],
         'PROJECTION_EVALUATION_BUDGET')
    expected_integrity = {'source_before': hashes, 'source_after': hashes,
                          'corpus_before': s.CORPUS['sha256'], 'corpus_after': s.CORPUS['sha256'],
                          'request_before': inputs['request_sha256'], 'request_after': inputs['request_sha256'],
                          'claim_before': inputs['claim_sha256'], 'claim_after': inputs['claim_sha256'],
                          'trained_files_before': inputs['trained_files_sha256'], 'trained_files_after': inputs['trained_files_sha256'],
                          'background_files_before': inputs['background_files_sha256'], 'background_files_after': inputs['background_files_sha256'],
                          'evaluation_pcm_before': inputs['evaluation_pcm_sha256'], 'evaluation_pcm_after': inputs['evaluation_pcm_sha256'],
                          'runtime_before': {name: digest(raw) for name, raw in runtime.items()},
                          'runtime_after': {name: digest(raw) for name, raw in runtime.items()}}
    need(exact(measurements['source_integrity'], expected_integrity), 'PROJECTION_INPUT_MUTATION')
    traces = zip_members((root / 'TRACES.zip').read_bytes(), TRACE_FILES)
    models = [pass_observation(record, arm, kind, build, cpus[0], trace_observation(traces[arm + '-' + kind + '.trace'], segments))
              for record, (arm, kind) in zip(measurements['passes'], PASSES)]
    need((root / 'native.stderr').read_text() == ''.join(r['raw_stderr'] for r in measurements['passes']),
         'PROJECTION_STDERR_BINDING')
    expected = summary_from({**inputs, 'file_sha256': files['INPUTS.json']['sha256']}, build,
                            {**training, 'file_sha256': files['TRAINING.json']['sha256']},
                            {**measurements, 'file_sha256': files['MEASUREMENTS.json']['sha256']}, models)
    need(exact(p.read_json(root / 'SUMMARY.json'), expected), 'PROJECTION_SUMMARY_RECOMPUTED')
    return {'schema': 'english63-projection-ab-replay-v1', 'status': 'VERIFIED_SAVED_PROJECTION_AB',
            'files': files, 'files_sha256': {n: r['sha256'] for n, r in files.items()},
            'summary_sha256': files['SUMMARY.json']['sha256'], 'gradient_tape_sha256': files['GRADIENTS.zip']['sha256'],
            'model_forward_calls': 0, 'optimizer_updates': 0, 'new_audio_asr_calls': 0,
            'formal_training_admitted': False, 'model_promotion_allowed': False}


def verify_saved(root):
    expected = saved_receipt(root)
    need(exact(p.read_json(root / 'REPLAY.json'), expected), 'PROJECTION_SAVED_RECEIPT')
    return expected
