"""Frozen background continuation contract and stdlib-only saved evidence reader."""
from __future__ import annotations

import ast
from collections import Counter
import io
import math
from pathlib import Path
import re
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


s = load('_en63_background_screen', ROOT / 'research/english63_online_c/librispeech_screen.py')
p, need, exact, digest, parse_json = s.p, s.need, s.exact, s.digest, s.parse_json
exporter = load('_en63_background_export', ROOT / 'research/english63_online_c/export_weights.py')
FIXED = parse_json((HERE / 'protocol.json').read_bytes())
PROFILE = FIXED['profile']
KINDS = ('gru', 'dsc_tcn')
VARIANTS = ('baseline', 'control', 'treatment')
ARMS = [(kind, arm) for kind in KINDS for arm in ('control', 'treatment')]
PASSES = [(arm, kind) for arm in VARIANTS for kind in KINDS]
FALSE_FIELDS = ('formal_training_admitted', 'human_gold', 'trusted_negative', 'model_promotion_allowed',
                'model_selection_made', 'threshold_selection', 'runtime_default_changed', 'new_audio_generated')
SOURCES = (
    'research/english63_background_ab/run_ab.py', 'research/english63_background_ab/ab_contract.py',
    'research/english63_background_ab/protocol.json', 'research/english63_background_ab/ab_replay.c',
    'research/english63_online_pilot/online_pilot.py', 'research/english63_online_pilot/online_contract.py',
    'research/english63_weak_pilot/english63_weak_train.py', 'research/english63_weak_pilot/RESULT.json',
    'research/english63_weak_pilot/PCM_FRONTEND_PARITY.json',
    'research/english63_weak_pilot/integrated_pcm_encoder.py',
    'research/english63_weak_pilot/pcm_frontend_parity.py', 'research/english63_weak_pilot/streaming_parity.py',
    'research/english63_weak_pilot/librispeech_negative_events.py',
    'research/english63_weak_pilot/pretrained_quant_events.py',
    'training/english_research_label_bridge.py', 'training/audit_dataset.py',
    'tools/corpus_identity.py', 'tools/source_identity.py',
    'research/english63_online_c/en63_stream.c', 'research/english63_online_c/en63_stream.h',
    'research/english63_online_c/contract.py', 'research/english63_online_c/pool_profile.py',
    'research/english63_online_c/librispeech_screen.py', 'research/english63_online_c/export_weights.py')
RESULT_FILES = {'PROTOCOL.json', 'INPUTS.json', 'SOURCE.zip', 'TRAINING.json', 'BUILD.json',
                'MEASUREMENTS.json', 'SUMMARY.json', 'REPLAY.json', 'native.stderr',
                'WEIGHTS.zip', 'RUNTIME.zip'} | {a + '-' + k + '.trace' for a, k in PASSES}
WEIGHT_FILES = {a + '-' + k + '.npz' for k, a in ARMS}
RUNTIME_FILES = {a + suffix for a in VARIANTS for suffix in ('.h', '.bin')}
TRACE_HEADER = b'EN63ABT1' + struct.pack('<II', 1, 13)
SEGMENT_HEADER = struct.Struct('<IIQQ')
TTS_GEOMETRY_SHA = '37d5d6f3474d287217e09b5a48810c9c37f047addea884cf73cdd8212a91b8a3'
ROSTER_SHA = {'train_members': 'e1bef5e07944db2b18424885fb6b0868079d0392533fe6cc6c4d26e8d1b6cc42',
              'train_windows': '3e2c49be5ffbcd11c134621a5c633b9d968d94761885600c2ac002644e15abce',
              'dev_members': 'dd9188fdacdd302f5d41ae723176f09bbee14f3aef3c9a65e06c3ddb2f3d89f0'}


def source_hashes():
    return {name: digest((ROOT / name).read_bytes()) for name in SOURCES}


def protocol():
    return {**FIXED, 'schema': 'english63-background-ab-protocol-v1', 'classes': p.bridge.CLASSES,
            'training_order': [list(x) for x in ARMS], 'evaluation_order': [list(x) for x in PASSES],
            'roster_sha256': ROSTER_SHA, 'tts_geometry_sha256': TTS_GEOMETRY_SHA,
            'native_children': 6, 'logical_streams': 78, 'native_feed_calls': 431946,
            'native_supplied_samples': 69110904, 'unique_evaluation_seconds': 719.90525,
            'trace_header': 'EN63ABT1,<II version=1,segments=13>',
            'trace_segment': '<IIQQ index,record_bytes=32,samples,frames> then <QQ3fI> rows',
            'source_role': 'already-exposed-read-speech-study-not-independent-holdout',
            'event_clock': 'dependency-end-exclusive;source-interval-is-not-acoustic-attribution',
            'qualification': 'NOT_RUN', 'no_benefit_is_successful_completion': True,
            'no_retry_or_result_driven_extension': True, **dict.fromkeys(FALSE_FIELDS, False)}


def validate_request(raw, head):
    value = parse_json(raw)
    need(type(value) is dict and set(value) == set(FIXED) | {'pipeline_commit', 'reason'} and
         all(exact(value[k], v) for k, v in FIXED.items()) and type(head) is str and
         p.SHA1.fullmatch(head) and value['pipeline_commit'] == head and
         type(value['reason']) is str and 1 <= len(value['reason'].strip()) <= 500, 'AB_FIXED_REQUEST')
    return value


def claim_expected(inputs, build):
    return {'schema': 'kws-research-v4-background-continuation-start-claim-v1',
            'request_sha256': inputs['request_sha256'], 'pipeline_commit': build['pipeline_head'],
            'source_checkpoint_artifact_id': 14114695344, 'source_negative_artifact_id': 14090383839,
            'run_id': build['public_run']['run_id'], 'run_attempt': '1', 'optimizer_updates': 0,
            'optimizer_update_budget': 192, 'status': 'CONSUMED_BEFORE_BACKGROUND_CONTINUATION'}


def geometry(samples):
    need(p.is_int(samples, 400, 58646164), 'AB_SAMPLE_GEOMETRY')
    return {'samples': samples, 'frames': 1 + (samples - 400) // 160,
            'feed_calls': (samples + 159) // 160, 'final_feed_samples': (samples - 1) % 160 + 1}


def natural_plan(rows):
    speakers = sorted({r['speaker_source_id'] for r in rows},
                      key=lambda x: (digest((PROFILE['speaker_hash_domain'] + x).encode()), x))
    need(len(speakers) == 40, 'AB_SPEAKERS')
    train_speakers, dev_speakers = speakers[:32], speakers[32:]
    selected = []
    for speaker in train_speakers:
        eligible = [r for r in rows if r['speaker_source_id'] == speaker and r['sample_count'] >= 48000]
        eligible.sort(key=lambda r: (digest((PROFILE['record_hash_domain'] + r['member']).encode()), r['member']))
        need(len(eligible) >= 4, 'AB_FOUR_WINDOWS_PER_SPEAKER')
        for row in eligible[:4]:
            start = (row['sample_count'] - 48000) // 2
            selected.append({**row, 'crop_start_sample': start, 'crop_end_sample_exclusive': start + 48000,
                             'crop_samples': 48000, 'crop_frames': 298, 'weak_wholeclip_class': 0})
    selected.sort(key=lambda r: r['member'])
    dev, offset = [], 0
    for row in sorted(rows, key=lambda r: r['member']):
        if row['speaker_source_id'] in dev_speakers:
            dev.append({**row, 'concatenation_start_sample': offset})
            offset += row['sample_count']
    hashes = {'train_members': digest(''.join(r['member'] + '\n' for r in selected).encode()),
              'train_windows': digest(''.join(f"{r['member']}\t{r['crop_start_sample']}\t{r['crop_end_sample_exclusive']}\n"
                                             for r in selected).encode()),
              'dev_members': digest(''.join(r['member'] + '\n' for r in dev).encode())}
    need(len(selected) == 128 and len(dev) == 102 and offset == 11124244 and
         exact(hashes, ROSTER_SHA), 'AB_FROZEN_NATURAL_ROSTER')
    return {'train_speakers': train_speakers, 'study_dev_speakers': dev_speakers,
            'train_windows': selected, 'study_dev_rows': dev, 'roster_sha256': hashes}


def zip_members(raw, names, max_member=4 * 1024 * 1024):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        items = archive.infolist()
        need(len(items) == len(names) and set(archive.namelist()) == set(names) and
             all(not i.is_dir() and 0 < i.file_size <= max_member and
                 (i.external_attr >> 16) & 0o170000 != 0o120000 for i in items), 'AB_ZIP_CLOSURE')
        return {name: archive.read(name) for name in sorted(names)}


def npz_model(raw, kind):
    """Parse safe numeric NPY without importing NumPy or constructing a model."""
    tensors = exporter.PARAMETERS[kind]
    members = zip_members(raw, {name + '.npy' for name, _ in tensors}, 100000)
    flat = []
    for name, shape in tensors:
        value = members[name + '.npy']
        need(value[:8] == b'\x93NUMPY\x01\x00' and len(value) >= 10, 'AB_NPY_VERSION')
        end = 10 + struct.unpack_from('<H', value, 8)[0]
        need(end <= len(value) and end <= 4096, 'AB_NPY_HEADER_BOUND')
        header = ast.literal_eval(value[10:end].decode('latin1'))
        need(type(header) is dict and type(header.get('shape')) is tuple and
             exact(header, {'descr': '<f4', 'fortran_order': False, 'shape': shape}), 'AB_NPY_SCHEMA')
        payload = value[end:]
        need(len(payload) == math.prod(shape) * 4 and
             all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', payload)), 'AB_NPY_VALUES')
        flat.append(payload)
    payload = b''.join(flat)
    return {'npz_sha256': digest(raw), 'parameter_count': len(payload) // 4,
            'parameter_payload_sha256': digest(payload),
            'tensor_order': [{'name': n, 'shape': list(sh)} for n, sh in tensors]}, payload


def baseline_payloads(header):
    need(digest(header) == PROFILE['baseline_header_sha256'], 'AB_BASELINE_PAYLOAD_IDENTITY')
    output = {}
    for kind in KINDS:
        pattern = r'static const float en63_' + kind + r'_parameters\[\d+\] = \{\n(.*?)\n\};\n'
        matches = re.findall(pattern, header.decode('ascii'), flags=re.DOTALL)
        need(len(matches) == 1, 'AB_BASELINE_PARAMETER_ARRAY')
        values = [float.fromhex(x.strip()[:-1]) for x in matches[0].split(',')]
        output[kind] = b''.join(struct.pack('<f', x) for x in values)
        need(len(output[kind]) == 4 * sum(math.prod(shape) for _, shape in exporter.PARAMETERS[kind]),
             'AB_BASELINE_PARAMETER_LENGTH')
    return output


def changed_tensors(before, after, kind):
    need(len(before) == len(after) == 4 * sum(math.prod(shape) for _, shape in exporter.PARAMETERS[kind]),
         'AB_WEIGHT_PAYLOAD_LENGTH')
    changed, cursor = 0, 0
    for _, shape in exporter.PARAMETERS[kind]:
        size = math.prod(shape) * 4
        changed += before[cursor:cursor + size] != after[cursor:cursor + size]
        cursor += size
    return changed


def replace_models(baseline, models):
    """Restricted exporter: only two parameter arrays and two NPZ identities change."""
    need(digest(baseline) == PROFILE['baseline_header_sha256'], 'AB_BASELINE_HEADER')
    text = baseline.decode('ascii')
    for kind in KINDS:
        meta, payload = models[kind]
        pattern = r'static const float en63_' + kind + r'_parameters\[\d+\] = \{\n.*?\n\};\n'
        replacement = exporter.array('en63_' + kind + '_parameters', [x[0] for x in struct.iter_unpack('<f', payload)])
        text, count = re.subn(pattern, lambda _: replacement, text, flags=re.DOTALL)
        need(count == 1 and text.count('"' + p.c.SOURCE_WEIGHTS[kind] + '"') == 1, 'AB_HEADER_MODEL_SECTION')
        text = text.replace('"' + p.c.SOURCE_WEIGHTS[kind] + '"', '"' + meta['npz_sha256'] + '"')
    return text.encode('ascii')


def accumulate_loss(old_loss, natural_losses, treatment):
    """Backprop each microbatch; sum is old30 + .5 * mean(new128), once per update."""
    old = old_loss()
    old_value = float(old.detach())
    need(math.isfinite(old_value), 'AB_OLD_LOSS_FINITE')
    old.backward()
    natural_value = 0.0
    if treatment:
        for index in range(4):
            loss = natural_losses(index)
            value = float(loss.detach())
            need(math.isfinite(value), 'AB_NATURAL_LOSS_FINITE')
            (loss * (0.5 * 32 / 128)).backward()
            natural_value += value / 4
    return {'old30_mil_ce': old_value, 'new128_mil_ce': natural_value if treatment else None,
            'objective': old_value + (0.5 * natural_value if treatment else 0)}


def trace_observation(raw, segments):
    need(len(segments) == 13 and raw[:16] == TRACE_HEADER, 'AB_TRACE_CONTAINER')
    cursor, output = 16, []
    for index, segment in enumerate(segments):
        need(cursor + 24 <= len(raw), 'AB_TRACE_SEGMENT_HEADER')
        actual_index, record_bytes, samples, frames = SEGMENT_HEADER.unpack_from(raw, cursor)
        g = geometry(segment['samples'])
        need(actual_index == index and record_bytes == 32 and samples == g['samples'] and frames == g['frames'],
             'AB_TRACE_SEGMENT_GEOMETRY')
        cursor += 24
        body = raw[cursor:cursor + frames * 32]
        observation = s.trace_observation(s.TRACE_HEADER + body, segment['rows'], samples)
        peaks = [0.0, 0.0, 0.0]
        for frame, _, p0, p1, p2, _ in s.TRACE_RECORD.iter_unpack(body):
            if frame >= 50:
                peaks = [max(a, b) for a, b in zip(peaks, (p0, p1, p2))]
        output.append({'segment_index': index, 'source_id': segment['source_id'],
                       'weak_wholeclip_class': segment['weak_wholeclip_class'], **g, **observation,
                       'peak_probabilities_after_min_frames': peaks})
        cursor += frames * 32
    need(cursor == len(raw), 'AB_TRACE_NO_EXTRA_BYTES')
    return {'trace_sha256': digest(raw), 'segments': output, 'frames': sum(r['frames'] for r in output),
            'feed_calls': sum(r['feed_calls'] for r in output), 'samples': sum(r['samples'] for r in output)}


def event_metrics(observation):
    streams = observation['segments']
    positives = {str(k): [row for row in streams[1:] if row['weak_wholeclip_class'] == k] for k in (1, 2)}
    background = [row for row in streams[1:] if row['weak_wholeclip_class'] == 0]
    need(all(len(x) == 4 for x in [*positives.values(), background]), 'AB_TTS_GROUP_COUNTS')
    return {'natural_total': streams[0]['proposal_count'], 'natural_by_class': streams[0]['proposals_by_class'],
            'correct_trigger_clips': {str(k): sum(r['proposals_by_class'][str(k)] > 0 for r in positives[str(k)])
                                      for k in (1, 2)},
            'wrong_other_keyword_clips': {str(k): sum(r['proposals_by_class'][str(3-k)] > 0 for r in positives[str(k)])
                                         for k in (1, 2)},
            'ordinary_background_proposals': sum(r['proposal_count'] for r in background)}


def benefit(baseline, control, treatment):
    b, c, t = baseline, control, treatment
    checks = {'natural_total_reduction': b['natural_total'] > 0 and c['natural_total'] > 0 and
              2 * t['natural_total'] <= b['natural_total'] and 2 * t['natural_total'] <= c['natural_total'],
              'ordinary_background_proposals': t['ordinary_background_proposals'] <=
              min(b['ordinary_background_proposals'], c['ordinary_background_proposals'])}
    for k in ('1', '2'):
        checks['natural_class_' + k] = t['natural_by_class'][k] <= min(b['natural_by_class'][k], c['natural_by_class'][k])
        checks['correct_trigger_clips_' + k] = t['correct_trigger_clips'][k] >= max(b['correct_trigger_clips'][k],
                                                                                  c['correct_trigger_clips'][k], 2)
        checks['wrong_other_keyword_clips_' + k] = t['wrong_other_keyword_clips'][k] <= min(
            b['wrong_other_keyword_clips'][k], c['wrong_other_keyword_clips'][k])
    return {'decision': 'RESEARCH_BENEFIT' if all(checks.values()) else 'NO_BENEFIT', 'checks': checks,
            'formal_training_admitted': False, 'model_promotion_allowed': False}


def verify_inputs(inputs, build):
    need(inputs['schema'] == 'english63-background-ab-inputs-v1' and
         exact(inputs['source_artifacts'], FIXED['source_artifacts']), 'AB_INPUT_SCHEMA')
    reference = s.reference_rows(inputs['reference_raw_utf8'].encode())
    s.verify_roster(inputs['natural_rows'], reference)
    planned = natural_plan(inputs['natural_rows'])
    need(exact(inputs['natural_plan'], planned), 'AB_SAVED_NATURAL_PLAN')
    windows = inputs['natural_windows']
    need(len(windows) == 128, 'AB_WINDOWS_COUNT')
    for row, plan in zip(windows, planned['train_windows']):
        need(set(row) == set(plan) | {'window_pcm_sha256', 'feature_payload_sha256'} and
             exact({k: row[k] for k in plan}, plan) and p.sha(row['window_pcm_sha256']) and
             p.sha(row['feature_payload_sha256']), 'AB_WINDOW_IDENTITY')
    report_raw = inputs['review_raw_utf8'].encode()
    need(digest(report_raw) == p.c.REVIEW_SHA, 'AB_REVIEW_IDENTITY')
    rows = p.bridge.diagnostic_plan(parse_json(report_raw))['rows']
    actual_tts = inputs['tts_rows']
    need(len(actual_tts) == 63, 'AB_TTS_INVENTORY')
    for row, source in zip(actual_tts, rows):
        need(set(row) == set(source) | {'source_samples'} and exact({k: row[k] for k in source}, source) and
             p.is_int(row['source_samples'], 8000, 192000), 'AB_TTS_SOURCE_ROW')
    geometry_rows = [{k: r[k] for k in ('cell_id', 'source_samples')} for r in actual_tts]
    need(digest(p.canonical(geometry_rows).encode()) == TTS_GEOMETRY_SHA, 'AB_TTS_FIXED_GEOMETRY')
    old_train = [r for r in actual_tts if r['action'] == 'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY']
    need(exact(inputs['original_train_rows'], old_train) and len(old_train) == 30 and
         Counter(r['weak_wholeclip_class'] for r in old_train) == {0: 10, 1: 10, 2: 10}, 'AB_OLD_TRAIN_ONLY')
    feature_rows = inputs['original_train_features']
    need(len(feature_rows) == 30 and all(exact(r['cell_id'], source['cell_id']) and
         exact(r['frames'], geometry(source['source_samples'])['frames']) and p.sha(r['feature_payload_sha256'])
         for r, source in zip(feature_rows, old_train)), 'AB_OLD_FEATURE_MANIFEST')
    segments = [{'source_id': 'natural-study-development', 'samples': 11124244, 'weak_wholeclip_class': 0,
                 'rows': planned['study_dev_rows']}]
    for row in actual_tts:
        if row['action'] == 'EXPOSED_DEVELOPMENT_NO_TUNE':
            segments.append({'source_id': row['cell_id'], 'samples': row['source_samples'],
                             'weak_wholeclip_class': row['weak_wholeclip_class'],
                             'rows': [{'member': row['cell_id'], 'sample_count': row['source_samples'],
                                       'concatenation_start_sample': 0}]})
    need(exact(inputs['evaluation_segments'], segments) and len(segments) == 13 and
         sum(geometry(r['samples'])['frames'] for r in segments) == 71965 and
         p.sha(inputs['natural_dev_pcm_sha256']), 'AB_EVALUATION_SEGMENTS')
    trained = FIXED['source_artifacts']['trained']
    for key, expected in (('terminal', trained['terminal_sha256']), ('summary', trained['summary_sha256'])):
        need(digest(inputs['trained_' + key + '_raw_utf8'].encode()) == expected, 'AB_CHECKPOINT_EVIDENCE')
    expected_trained = {**parse_json(inputs['trained_terminal_raw_utf8'])['files_sha256'],
                        'TERMINAL.json': trained['terminal_sha256']}
    need(exact(inputs['trained_files_sha256'], expected_trained), 'AB_CHECKPOINT_ALL_FILES')
    expected_pcm = {'natural-study-development.pcm': inputs['natural_dev_pcm_sha256']}
    expected_pcm.update({r['cell_id'] + '.pcm': r['pcm_sha256'] for r in actual_tts
                         if r['action'] == 'EXPOSED_DEVELOPMENT_NO_TUNE'})
    need(exact(inputs['evaluation_pcm_sha256'], expected_pcm), 'AB_EVALUATION_PCM_IDENTITIES')
    request, claim = inputs['request_raw_utf8'].encode(), inputs['claim_raw_utf8'].encode()
    validate_request(request, build['pipeline_head'])
    need(digest(request) == inputs['request_sha256'] and digest(claim) == inputs['claim_sha256'] and
         exact(parse_json(claim), claim_expected(inputs, build)), 'AB_CLAIM_BINDING')
    need(inputs['pcm_decoder']['numpy'] == '2.3.5' and inputs['pcm_decoder']['soundfile'] == '0.13.1' and
         inputs['pcm_decoder']['resampled'] is False and inputs['pcm_decoder']['normalized'] is False,
         'AB_PCM_DECODER')
    return segments


def verify_training(training, inputs, weights, baseline):
    need(training['schema'] == 'english63-background-ab-training-v1' and
         exact(training['optimizer_updates_started'], 192) and exact(training['optimizer_updates_completed'], 192) and
         training['active_training'] is None and training['status'] == 'COMPLETED' and
         type(training['group_wall_seconds']) in (int, float) and 0 < training['group_wall_seconds'] <= 720 and
         len(training['arms']) == 4, 'AB_TRAINING_BUDGET')
    original_payloads = baseline_payloads(baseline)
    for record, (kind, arm) in zip(training['arms'], ARMS):
        name = arm + '-' + kind + '.npz'
        meta, payload = weights[name]
        fixed = {'kind': kind, 'arm': arm, 'status': 'COMPLETED', 'optimizer_updates': 48,
                 'seed': PROFILE['seeds'][kind], 'warm_start_sha256': p.c.SOURCE_WEIGHTS[kind],
                 'optimizer': 'fresh-AdamW', 'learning_rate': .001, 'weight_decay': .01,
                 'gradient_clip': 1, 'original_train_rows': 30, 'natural_train_rows': 128 if arm == 'treatment' else 0,
                 'natural_coefficient': .5 if arm == 'treatment' else 0, 'train_development_rows': 0,
                 'train_nearphone_rows': 0, 'model_promotion_allowed': False}
        need(all(exact(record[k], v) for k, v in fixed.items()) and exact(record['weights'], meta) and
             meta['npz_sha256'] != p.c.SOURCE_WEIGHTS[kind] and
             record['runtime']['numpy'] == '2.3.5' and record['runtime']['torch'] == '2.10.0+cpu' and
             record['runtime']['threads'] == 1 and record['runtime']['mkldnn'] is False and
             0 < record['child_wall_seconds'] <= 180 and exact(record['child_exit_code'], 0) and
             record['timed_out'] is False and
             p.is_int(record['child_peak_rss_kib'], 1) and len(record['updates']) == 48, 'AB_TRAINING_ARM')
        changes = changed_tensors(original_payloads[kind], payload, kind)
        need(changes > 0 and exact(record['changed_state_tensors'], changes) and
             record['warm_start_parameter_payload_sha256'] == digest(original_payloads[kind]),
             'AB_TRUE_WEIGHT_PAYLOAD_CHANGE')
        for i, update in enumerate(record['updates'], 1):
            need(exact(update['step'], i) and all(type(update[k]) in (float, int) and
                 math.isfinite(update[k]) and update[k] >= 0 for k in ('old30_mil_ce', 'objective', 'gradient_norm')),
                 'AB_TRAINING_STEP')
            natural = update['new128_mil_ce']
            need((arm == 'control' and natural is None) or
                 (arm == 'treatment' and type(natural) in (int, float) and math.isfinite(natural) and natural >= 0),
                 'AB_TRAINING_NATURAL_LOSS')
            objective = update['old30_mil_ce'] + (.5 * natural if arm == 'treatment' else 0)
            need(exact(update['objective'], objective), 'AB_LOSS_WEIGHT')


def command_for(compiler, source, output):
    root, out = Path(source), Path(output)
    return [compiler, *s.FLAGS, '-I', str(root / 'research/english63_online_c'), '-I', str(out),
            str(root / 'research/english63_online_c/en63_stream.c'),
            str(root / 'research/english63_background_ab/ab_replay.c'), '-lm', '-o', str(out / 'ab_replay')]


def pass_observation(record, arm, kind, build, cpu, trace):
    need(record['arm'] == arm and record['kind'] == kind and record['status'] == 'COMPLETED' and
         exact(record['exit_code'], 0) and record['timed_out'] is False, 'AB_NATIVE_TERMINAL')
    command = record['command']
    need(type(command) is list and len(command) == 7 and command[:3] ==
         [str(Path(build['build_root']) / arm / 'ab_replay'), '--supervise', kind] and
         command[3] == str(Path(build['build_root']) / 'evaluation.txt') and
         command[4] == str(Path(build['output_root']) / (arm + '-' + kind + '.trace')) and
         command[5:] == [str(cpu), str(build['entrypoint_pid'])],
         'AB_NATIVE_COMMAND')
    lines = record['raw_stdout'].splitlines()
    need(len(lines) == 2, 'AB_NATIVE_STDOUT')
    body, lifetime = [parse_json(line) for line in lines]
    fixed = {'schema': 'english63-background-ab-native-v1', 'kind': kind, 'pcm_samples': 11518484,
             'frames': 71965, 'feed_calls': 71991, 'proposal_count': sum(r['proposal_count'] for r in trace['segments']),
             'state_bytes': 38192, 'weight_parameter_bytes': (6531 if kind == 'gru' else 4803) * 4,
             'frontend_table_bytes': 3894, 'pool_time_major': 0, 'affinity_cpu': cpu, 'affinity_single_cpu': True,
             'initializations': 13, 'finish_calls': 13, 'reset_calls': 0, 'discontinuity_calls': 0,
             'eof_padding_samples': 0, 'eof_events': 0}
    expected_segments = [{key: row[key] for key in ('segment_index', 'samples', 'frames', 'feed_calls',
                                                   'final_feed_samples', 'proposal_count')} for row in trace['segments']]
    need(set(body) == set(fixed) | {'cpu_ns', 'wall_ns', 'segments'} and
         all(exact(body[k], v) for k, v in fixed.items()) and exact(body['segments'], expected_segments) and
         p.is_int(body['cpu_ns'], 1) and p.is_int(body['wall_ns'], 1, 120000000000), 'AB_NATIVE_COUNTERS')
    need(set(lifetime) == {'schema', 'child_exit_code', 'child_signaled', 'process_lifetime_peak_rss_kib'} and
         lifetime['schema'] == 'english63-background-ab-child-lifetime-v1' and
         exact(lifetime['child_exit_code'], 0) and lifetime['child_signaled'] is False and
         p.is_int(lifetime['process_lifetime_peak_rss_kib'], 1), 'AB_NATIVE_LIFETIME')
    return {'arm': arm, 'kind': kind, **trace, 'metrics': event_metrics(trace),
            'cpu_seconds': body['cpu_ns'] / 1e9, 'wall_seconds': body['wall_ns'] / 1e9,
            'process_lifetime_peak_rss_kib': lifetime['process_lifetime_peak_rss_kib']}


def summary_from(inputs, build, training, measurements, models):
    lookup = {(r['arm'], r['kind']): r['metrics'] for r in models}
    gates = {kind: benefit(*(lookup[a, kind] for a in VARIANTS)) for kind in KINDS}
    return {'schema': 'english63-background-ab-result-v1', 'status': 'MEASURED_EXPOSED_BACKGROUND_AB_NOT_QUALIFICATION',
            'experiment': FIXED['experiment'], 'public_run': build['public_run'], 'pipeline_head': build['pipeline_head'],
            'request_sha256': inputs['request_sha256'], 'source_sha256': build['source_sha256'],
            'source_artifacts': FIXED['source_artifacts'], 'profile': PROFILE, 'classes': p.bridge.CLASSES,
            'inputs_sha256': inputs['file_sha256'], 'training_sha256': training['file_sha256'],
            'measurements_sha256': measurements['file_sha256'], 'models': models, 'benefit_by_kind': gates,
            'decision': 'RESEARCH_BENEFIT' if all(v['decision'] == 'RESEARCH_BENEFIT' for v in gates.values()) else 'NO_BENEFIT',
            'optimizer_updates': 192, 'native_forward_frames': 431790, 'native_feed_calls': 431946,
            'native_supplied_samples': 69110904, 'native_children': 6, 'logical_streams': 78,
            'native_initializations': 78, 'native_finishes': 78,
            'unique_evaluation_source_seconds': 719.90525, 'new_independent_audio_seconds': 0,
            'new_tts_asr_ipa_calls': 0, 'training_group_wall_seconds': training['group_wall_seconds'],
            'native_group_wall_seconds': measurements['native_group_wall_seconds'],
            'independent_english_frr': 'NOT_RUN', 'continuous_negative_far_per_hour': 'NOT_RUN',
            'ssc305_board_profile': 'NOT_RUN', **dict.fromkeys(FALSE_FIELDS, False)}


def saved_receipt(root):
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'AB_SAVED_ROOT')
    paths = list(root.iterdir())
    need({x.name for x in paths} in (RESULT_FILES, RESULT_FILES - {'REPLAY.json'}) and
         all(x.is_file() and not x.is_symlink() for x in paths) and
         sum(x.stat().st_size for x in paths) < 64 * 1024 * 1024, 'AB_RESULT_CLOSURE')
    files = {x.name: p.artifact_identity(x.read_bytes()) for x in sorted(paths) if x.name != 'REPLAY.json'}
    need(exact(p.read_json(root / 'PROTOCOL.json'), protocol()), 'AB_PROTOCOL')
    source = zip_members((root / 'SOURCE.zip').read_bytes(), SOURCES)
    hashes = {name: digest(raw) for name, raw in source.items()}
    need(exact(hashes, source_hashes()), 'AB_SOURCE_BYTES')
    inputs, build, training, measurements = [p.read_json(root / name) for name in
                                           ('INPUTS.json', 'BUILD.json', 'TRAINING.json', 'MEASUREMENTS.json')]
    p.public_run(build['public_run'])
    need(build['schema'] == 'english63-background-ab-build-v1' and exact(build['source_sha256'], hashes) and
         p.SHA1.fullmatch(build['pipeline_head']) and build['physical_ssc305'] is False and
         all(type(build[k]) is str and Path(build[k]).is_absolute() for k in
             ('compiler', 'compiler_realpath', 'source_root', 'build_root', 'output_root')) and
         p.sha(build['compiler_sha256']) and type(build['compiler_version']) is str and build['compiler_version'] and
         p.is_int(build['entrypoint_pid'], 2),
         'AB_BUILD_IDENTITY')
    segments = verify_inputs(inputs, build)
    weight_bytes = zip_members((root / 'WEIGHTS.zip').read_bytes(), WEIGHT_FILES)
    weights = {name: npz_model(raw, 'dsc_tcn' if 'dsc_tcn' in name else 'gru') for name, raw in weight_bytes.items()}
    runtime = zip_members((root / 'RUNTIME.zip').read_bytes(), RUNTIME_FILES)
    baseline = runtime['baseline.h']
    need(digest(baseline) == PROFILE['baseline_header_sha256'], 'AB_BASELINE_EXACT_EXPORT')
    verify_training(training, inputs, weights, baseline)
    need(len(build['variants']) == 3, 'AB_BUILD_VARIANTS')
    for report, arm in zip(build['variants'], VARIANTS):
        header = baseline if arm == 'baseline' else replace_models(baseline, {k: weights[arm + '-' + k + '.npz'] for k in KINDS})
        need(runtime[arm + '.h'] == header and runtime[arm + '.bin'].startswith(b'\x7fELF') and
             report['arm'] == arm and report['header_sha256'] == digest(header) and
             report['binary_sha256'] == digest(runtime[arm + '.bin']) and exact(report['exit_code'], 0) and
             exact(report['command'], command_for(build['compiler'], build['source_root'], Path(build['build_root']) / arm)),
             'AB_EXPORT_AND_BUILD_BINDING')
    cpus = measurements['initial_allowed_cpus']
    need(measurements['schema'] == 'english63-background-ab-measurements-v1' and type(cpus) is list and cpus and
         all(p.is_int(x, 0, 1023) for x in cpus) and cpus == sorted(set(cpus)) and
         exact(measurements['selected_cpu'], cpus[0]) and exact(measurements['final_allowed_cpus'], cpus) and
         0 < measurements['native_group_wall_seconds'] <= 180 and len(measurements['passes']) == 6 and
         exact(measurements['public_run'], build['public_run']) and measurements['pipeline_head'] == build['pipeline_head'],
         'AB_EVALUATION_BUDGET')
    expected_integrity = {'source_before': hashes, 'source_after': hashes,
                          'corpus_before': s.CORPUS['sha256'], 'corpus_after': s.CORPUS['sha256'],
                          'request_before': inputs['request_sha256'], 'request_after': inputs['request_sha256'],
                          'claim_before': inputs['claim_sha256'], 'claim_after': inputs['claim_sha256'],
                          'trained_files_before': inputs['trained_files_sha256'],
                          'trained_files_after': inputs['trained_files_sha256'],
                          'evaluation_pcm_before': inputs['evaluation_pcm_sha256'],
                          'evaluation_pcm_after': inputs['evaluation_pcm_sha256'],
                          'runtime_before': {name: digest(raw) for name, raw in runtime.items()},
                          'runtime_after': {name: digest(raw) for name, raw in runtime.items()}}
    need(exact(measurements['source_integrity'], expected_integrity), 'AB_INPUT_MUTATION')
    models = [pass_observation(record, arm, kind, build, cpus[0],
                              trace_observation((root / (arm + '-' + kind + '.trace')).read_bytes(), segments))
              for record, (arm, kind) in zip(measurements['passes'], PASSES)]
    need((root / 'native.stderr').read_text() == ''.join(r['raw_stderr'] for r in measurements['passes']), 'AB_STDERR_BINDING')
    expected = summary_from({**inputs, 'file_sha256': files['INPUTS.json']['sha256']}, build,
                            {**training, 'file_sha256': files['TRAINING.json']['sha256']},
                            {**measurements, 'file_sha256': files['MEASUREMENTS.json']['sha256']}, models)
    need(exact(p.read_json(root / 'SUMMARY.json'), expected), 'AB_SUMMARY_RECOMPUTED')
    return {'schema': 'english63-background-ab-replay-v1', 'status': 'VERIFIED_SAVED_BACKGROUND_AB',
            'files': files, 'files_sha256': {n: r['sha256'] for n, r in files.items()},
            'summary_sha256': files['SUMMARY.json']['sha256'], 'model_forward_calls': 0, 'optimizer_updates': 0,
            'new_audio_asr_calls': 0, 'formal_training_admitted': False, 'model_promotion_allowed': False}


def verify_saved(root):
    expected = saved_receipt(root)
    need(exact(p.read_json(root / 'REPLAY.json'), expected), 'AB_SAVED_RECEIPT')
    return expected
