#!/usr/bin/env python3
"""One bounded public-Actions saved-weight C state experiment; stdlib only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import io
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import zipfile
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load_source('_en63_state_legacy_contract', HERE / 'contract.py')
bridge = load_source('_en63_state_original_label_bridge', ROOT / 'training/english_research_label_bridge.py')

SCHEMA = 'english63-native-c-state-contracts-v1'
STATUS = 'MEASURED_NATIVE_C_STATE_BOUNDARIES_NOT_KWS_QUALIFICATION'
REPLAY_SCHEMA = 'english63-native-c-state-saved-verification-v1'
REPLAY_STATUS = 'VERIFIED_SAVED_NATIVE_C_STATE_CONTRACTS'
BASELINE_RUN = 43605922752
BASELINE_ARTIFACT = 14117638750
BASELINE_ZIP_SHA = 'e85d14c87727f93b6e33edc196af00a6c9ce023d9e7d41d844109617fa5fb397'
BASELINE_REPLAY_SHA = 'e0ec43d08fcccf20828b5c874b63c6f944233829a9ef70f07cc574ff6aee025d'
HEADER_SHA = '912f443a85fddbc77c49f6cc52c2dbde6c395436d7d8d0818a4e6225cd90fc16'
BASELINE_TRACE_SHA = {
    'gru': 'aaace50d91252d4a895961c2d9ec4af5d3971a3ce70024ac897bec4c3953a88b',
    'dsc_tcn': 'fd449807d7508adb1fc1dddcaa4022b293988a4406ab3ceb3f3fb5fa220650d8',
}
PREFIX_SHA = {
    'gru': '17d77c4f17b904f4be14cf137496f4130b3e8da3b6f215d3bea99586e37c76f3',
    'dsc_tcn': 'e055934e8e0d469b7d9f4a06d58498d06fe8345ebd72612a86cffef47195ac62',
}
TEST_SAMPLES = 64123
TEST_UNIQUE_SAMPLES = 2 * TEST_SAMPLES
CUTS = (399, 400, 32401)
FORWARD_FRAMES = 4773
SUPPLIED_SAMPLES = 769475
CASES = ('baseline_saved_prefix', 'reset_midstream', 'reset_after_eof',
         'dual_instance_interleaved', 'discontinuity_399', 'discontinuity_400',
         'discontinuity_32401', 'eof_no_padding', 'invalid_call_atomicity',
         'invalid_model_atomicity')
FALSE_FIELDS = ('formal_training_admitted', 'human_gold', 'model_promotion_allowed',
                'model_selection_made', 'threshold_selection')
FLAGS = ('-std=c11', '-O2', '-ffp-contract=off', '-fno-fast-math',
         '-Wall', '-Wextra', '-Werror', '-pedantic')
TRACE_RECORD = struct.Struct('<IIQQ38fI')
OLD_RECORD = struct.Struct('<IQ38fI')
SOURCES = ('research/english63_online_c/en63_stream.c',
           'research/english63_online_c/en63_stream.h',
           'research/english63_online_c/state_cases.c',
           'research/english63_online_c/state_replay.py',
           'research/english63_online_c/contract.py',
           'training/english_research_label_bridge.py')


def need(condition, message):
    c.require(condition, 'EN63_STATE_' + message)


def frames_for(samples):
    return 0 if samples < 400 else 1 + (samples - 400) // 160


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def protocol():
    return {
        'schema': 'english63-native-c-state-protocol-v1',
        'source_native_run_id': BASELINE_RUN, 'source_native_artifact_id': BASELINE_ARTIFACT,
        'source_native_zip_sha256': BASELINE_ZIP_SHA, 'source_header_sha256': HEADER_SHA,
        'source_tts_zip_sha256': c.TTS_SHA, 'source_review_json_sha256': c.REVIEW_SHA,
        'source_weights_sha256': c.SOURCE_WEIGHTS,
        'source_audio_clips': 63, 'source_pcm_samples': 2101760,
        'input_selection': 'first128246 supplied PCM samples of original sorted-cell-id concatenation',
        'stream_a': {'start_sample': 0, 'samples': TEST_SAMPLES},
        'stream_b': {'start_sample': TEST_SAMPLES, 'samples': TEST_SAMPLES},
        'test_unique_pcm_samples': TEST_UNIQUE_SAMPLES, 'discontinuity_cuts_in_a': list(CUTS),
        'cases_per_model': list(CASES), 'forward_frames_per_model': FORWARD_FRAMES,
        'supplied_samples_including_replays_per_model': SUPPLIED_SAMPLES,
        'baseline_prefix_frames': 399, 'baseline_prefix_sha256': PREFIX_SHA,
        'baseline_bit_identity_required': True, 'numeric_diagnostics_tolerances': c.TOLERANCES,
        'fresh_suffix_reset_and_interleaving_bit_identity_required': True,
        'clock': 'cumulative-supplied-PCM-not-missing-capture-time-or-wall-time',
        'discontinuity': 'discard-acoustic-state-preserve-session-clock-and-counters-restart-warmup',
        'new_session_reset': 'discard-acoustic-state-and-all-session-counters',
        'eof_padding_samples': 0, 'new_independent_audio_samples': 0,
        'new_independent_audio_seconds': 0, 'new_tts_asr_ipa_calls': 0,
        'optimizer_updates': 0, 'acoustic_word_end_labels_created': False,
        **dict.fromkeys(FALSE_FIELDS, False), **dict.fromkeys(c.UNMEASURED, 'NOT_RUN'),
    }


def source_hashes():
    return {name: c.digest((ROOT / name).read_bytes()) for name in SOURCES}


def baseline_inputs(root):
    comparison = root / 'comparison'
    need(root.is_absolute() and comparison.is_dir(), 'BASELINE_ROOT')
    need(c.digest((comparison / 'REPLAY.json').read_bytes()) == BASELINE_REPLAY_SHA,
         'ORIGINAL_BASELINE_RECEIPT')
    c.verify_saved(comparison)  # Saved hashes only; no previous experiment is executed.
    header = (comparison / 'export/en63_generated.h').read_bytes()
    need(c.digest(header) == HEADER_SHA, 'ORIGINAL_EXPORTED_HEADER')
    exported = c.read_json(comparison / 'export/EXPORT.json')
    need(exported['source_training_run_id'] == 43595338774 and
         {m['kind']: m['source_weights_sha256'] for m in exported['models']} == c.SOURCE_WEIGHTS,
         'FINITE_WINDOW_TRAINED_CHECKPOINTS')
    prefixes = {}
    for kind in c.KINDS:
        raw = (comparison / (kind + '-continuous-160.trace')).read_bytes()
        need(c.digest(raw) == BASELINE_TRACE_SHA[kind], 'ORIGINAL_CONTINUOUS_TRACE')
        prefixes[kind] = raw[:16 + 399 * OLD_RECORD.size]
        need(c.digest(prefixes[kind]) == PREFIX_SHA[kind], 'ORIGINAL_TRACE_PREFIX')
    return header, prefixes


def original_pcm(tts, review):
    need(tts.is_absolute() and tts.is_file() and not tts.is_symlink() and
         tts.stat().st_size < 64 * 1024 * 1024 and c.digest(tts.read_bytes()) == c.TTS_SHA,
         'ORIGINAL_TTS_ARCHIVE')
    rows = bridge.diagnostic_plan(bridge.read_verified_report(review))['rows']
    rows = sorted(rows, key=lambda r: r['cell_id'])
    counts = Counter(row['action'] for row in rows)
    need(counts == {'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY': 30,
                    'EXPOSED_DEVELOPMENT_NO_TUNE': 12,
                    'QUARANTINE_AMBIGUOUS_NEARPHONE': 21}, 'ORIGINAL_ROLE_ROSTER')
    chunks, bindings, total = [], [], 0
    with zipfile.ZipFile(tts) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)), 'SOURCE_ZIP_DUPLICATES')
        wav_names = {name for name in names if name.endswith('/audio-16k.wav')}
        need(len(wav_names) == 63, 'ORIGINAL_WAV_ROSTER')
        for row in rows:
            name = 'english-seven-voice-actual/capture/cells/' + row['cell_id'] + '/audio-16k.wav'
            need(name in wav_names, 'SOURCE_WAV_MEMBER')
            wav_names.remove(name)
            raw = archive.read(name)
            need(c.digest(raw) == row['wav_sha256'], 'ORIGINAL_WAV_BYTES')
            with wave.open(io.BytesIO(raw), 'rb') as wav:
                count = wav.getnframes()
                need(wav.getnchannels() == 1 and wav.getsampwidth() == 2 and
                     wav.getframerate() == 16000 and wav.getcomptype() == 'NONE' and
                     8000 <= count <= 192000, 'ORIGINAL_PCM_GEOMETRY')
                pcm = wav.readframes(count)
                need(len(pcm) == count * 2 and not wav.readframes(1), 'ORIGINAL_PCM_LENGTH')
            need(c.digest(pcm) == row['pcm_sha256'], 'ORIGINAL_PCM_BYTES')
            used = min(count, max(0, TEST_UNIQUE_SAMPLES - total))
            chunks.append(pcm[:used * 2])
            bindings.append({'cell_id': row['cell_id'], 'member': name,
                             'wav_sha256': row['wav_sha256'], 'pcm_sha256': row['pcm_sha256'],
                             'action': row['action'], 'source_family_id': row['source_family_id'],
                             'source_samples': count, 'concatenation_start_sample': total,
                             'test_start_within_source': 0, 'test_samples_used': used,
                             'human_gold': False, 'training_admitted': False})
            total += count
        need(not wav_names and total == 2101760, 'FULL_ORIGINAL_PCM_INVENTORY')
    pcm = b''.join(chunks)
    need(len(pcm) == 2 * TEST_UNIQUE_SAMPLES, 'FIXED_INPUT_PREFIX')
    need(pcm[:2 * TEST_SAMPLES] != pcm[2 * TEST_SAMPLES:], 'TWO_DISTINCT_REAL_PCM_WINDOWS')
    return pcm, {'schema': 'english63-state-fixed-input-bindings-v1',
                 'source_tts_zip_sha256': c.TTS_SHA, 'source_review_json_sha256': c.REVIEW_SHA,
                 'source_audio_clips': 63, 'source_pcm_samples': total, 'rows': bindings,
                 'test_unique_pcm_samples': TEST_UNIQUE_SAMPLES,
                 'test_pcm_sha256': c.digest(pcm), 'test_pcm_bytes': len(pcm),
                 'stream_a_pcm_sha256': c.digest(pcm[:2 * TEST_SAMPLES]),
                 'stream_b_pcm_sha256': c.digest(pcm[2 * TEST_SAMPLES:]),
                 'new_audio_generation_or_resampling': False, 'pcm_operation': 'byte-preserving-concatenation-and-prefix-selection',
                 'new_independent_audio_samples': 0, 'new_independent_audio_seconds': 0}


def read_trace(raw):
    need(raw[:16] == b'EN63BND1' + struct.pack('<II', 1, 180) and
         (len(raw) - 16) % TRACE_RECORD.size == 0, 'TRACE_LAYOUT')
    grouped = defaultdict(list)
    for offset in range(16, len(raw), TRACE_RECORD.size):
        record = raw[offset:offset + TRACE_RECORD.size]
        values = TRACE_RECORD.unpack(record)
        group, instance, frame, sample = values[:4]
        need(0 <= group <= 15 and instance == (1 if group in (1, 3, 9, 12, 15) else 0) and
             all(math.isfinite(x) for x in values[4:42]) and values[42] in (0, 1, 2), 'TRACE_VALUES')
        grouped[group].append({'frame': frame, 'sample': sample, 'values': values[4:42],
                               'proposal': values[42], 'payload': record[24:]})
    expected = [399, 399, 399, 399, 201, 399, 399, 0, 396, 396, 1, 396, 396, 201, 196, 196]
    need({key: len(rows) for key, rows in grouped.items()} ==
         {key: count for key, count in enumerate(expected) if count}, 'EXACT_GROUP_FRAME_COUNTS')
    need(sum(map(len, grouped.values())) == FORWARD_FRAMES, 'EXACT_FORWARD_FRAME_BUDGET')
    for group, rows in grouped.items():
        sample_base = CUTS[(group - 8) // 3] if group in (8, 11, 14) else 0
        frame_base = frames_for(sample_base)
        for index, row in enumerate(rows, 1):
            need(row['frame'] == frame_base + index and row['sample'] == sample_base + 400 + (index - 1) * 160,
                 'RAW_SESSION_CLOCK')
            if index < 50 and group not in (4, 7, 10, 13):
                need(row['proposal'] == 0, 'RAW_FRESH_WARMUP')
    return grouped


def identical(a, b, frame_offset=0, sample_offset=0):
    return len(a) == len(b) and all(x['frame'] == y['frame'] + frame_offset and
                                  x['sample'] == y['sample'] + sample_offset and
                                  x['payload'] == y['payload'] for x, y in zip(a, b))


def compare_saved_prefix(kind, rows, old):
    need(c.digest(old) == PREFIX_SHA[kind] and old[:16] == b'EN63TRC1' + struct.pack('<II', 1, 168),
         'SAVED_BASELINE_PREFIX_IDENTITY')
    reconstructed = b'EN63TRC1' + struct.pack('<II', 1, 168)
    maxima = dict.fromkeys(c.TOLERANCES, 0.0)
    events_equal = True
    for index, row in enumerate(rows):
        prior = OLD_RECORD.unpack_from(old, 16 + index * OLD_RECORD.size)
        need(prior[0] == 0 and prior[1] == row['sample'], 'SAVED_BASELINE_CLOCK')
        for field, start, end in (('features', 0, 32), ('logits', 32, 35), ('probabilities', 35, 38)):
            maxima[field] = max(maxima[field], max(abs(row['values'][j] - prior[2 + j]) for j in range(start, end)))
        events_equal = events_equal and prior[40] == row['proposal']
        reconstructed += struct.pack('<IQ', 0, row['sample']) + row['payload']
    return {'frames': len(rows), 'baseline_trace_bit_identical': reconstructed == old,
            'max_absolute_errors': maxima, 'original_tolerance_diagnostic_passed':
            all(maxima[field] <= bound for field, bound in c.TOLERANCES.items()),
            'proposal_class_and_sample_identity': events_equal,
            'source_prefix_sha256': c.digest(old), 'observed_prefix_sha256': c.digest(reconstructed)}


def case_observations(kind, trace_raw, baseline_raw, native):
    groups = read_trace(trace_raw)
    body, lifetime = native['observations'], native['lifetime']
    lines = native.get('raw_stdout', '').splitlines()
    need(len(lines) == 2 and [canonical(json.loads(line)) for line in lines] == [canonical(body), canonical(lifetime)],
         'RAW_NATIVE_STDOUT_BINDING')
    expected = {'schema': 'english63-state-native-case-observations-v1', 'kind': kind,
                'forward_frames': FORWARD_FRAMES, 'supplied_samples_including_replays': SUPPLIED_SAMPLES,
                'finish_calls': 34, 'reentry_rejections': 4, 'finished_rejections': 34,
                'argument_rejections': 6, 'overflow_rejections': 3, 'fault_rejections': 3,
                'model_rejections': 6, 'live_instances': 2, 'input_buffer_bytes': 2 * TEST_UNIQUE_SAMPLES,
                'clock_overflow_fixture_only': True, 'fault_flag_fixture_only': True,
                'eof_padding_samples': 0, 'eof_events': 0, 'gap_padding_samples': 0,
                'gap_callbacks': 0, 'native_checks_passed': True}
    need(all(body.get(key) == value and type(body.get(key)) is type(value) for key, value in expected.items()),
         'NATIVE_FIXED_COUNTERS')
    need(lifetime['schema'] == 'english63-state-native-child-lifetime-v1' and type(lifetime['child_exit_code']) is int and
         lifetime['child_exit_code'] == 0 and
         lifetime['child_signaled'] is False and lifetime['measurement'] == 'linux-wait4-fresh-native-child-after-exit' and
         lifetime['includes_test_harness_and_trace_io'] is True and lifetime['ssc305_measured'] is False and
         lifetime['process_lifetime_peak_rss_kib'] >= body['inference_end_high_water_rss_kib'] > 0,
         'ACTUAL_NATIVE_LIFETIME_RSS_SCOPE')
    baseline = compare_saved_prefix(kind, groups[0], baseline_raw)
    result = [{'name': CASES[0], 'passed': baseline['baseline_trace_bit_identical'] and
               baseline['original_tolerance_diagnostic_passed'] and baseline['proposal_class_and_sample_identity'],
               'observed': baseline},
              {'name': CASES[1], 'passed': identical(groups[5], groups[1]),
               'observed': {'contaminating_prefix_samples': 32401, 'frames': 399, 'new_session_clock': True}},
              {'name': CASES[2], 'passed': identical(groups[6], groups[0]),
               'observed': {'frames': 399, 'new_session_clock': True}},
              {'name': CASES[3], 'passed': identical(groups[2], groups[0]) and identical(groups[3], groups[1]),
               'observed': {'instances': 2, 'frames_per_instance': 399, 'plans': ['mixed', 'mixed-rotated-by-3']}}]
    for index, cut in enumerate(CUTS):
        before, after, reference = 7 + 3 * index, 8 + 3 * index, 9 + 3 * index
        matched = identical(groups[after], groups[reference], frames_for(cut), cut)
        prefix = groups.get(before, [])
        result.append({'name': CASES[4 + index], 'passed': matched and identical(prefix, groups[0][:frames_for(cut)]),
                       'observed': {'cut_samples': cut, 'discarded_carry_samples': cut if cut < 400 else 240 + (cut - 400) % 160,
                                    'preserved_prefix_frames': frames_for(cut), 'suffix_frames': len(groups[after]),
                                    'first_suffix_available_sample': groups[after][0]['sample'],
                                    'fresh_suffix_bit_identical': matched, 'missing_capture_time_inferred': False,
                                    'session_counter_preservation_checked_natively': True,
                                    'warmup_restarted': True, 'padding_samples': 0, 'gap_callbacks': 0}})
    result.extend([
        {'name': CASES[7], 'passed': body['native_checks_passed'] is True and body['finish_calls'] == 34 and
         body['eof_padding_samples'] == body['eof_events'] == 0 and body['finished_rejections'] == 34,
         'observed': {'finish_calls': 34, 'short_tail_lengths': [0, 1, 159, 239, 399],
                                                     'eof_padding_samples': 0, 'eof_callbacks': 0}},
        {'name': CASES[8], 'passed': body['native_checks_passed'] is True and all(body[key] == value for key, value in
         (('argument_rejections', 6), ('overflow_rejections', 3), ('fault_rejections', 3),
          ('reentry_rejections', 4), ('finished_rejections', 34))), 'observed': {key: body[key] for key in
         ('argument_rejections', 'overflow_rejections', 'fault_rejections', 'reentry_rejections', 'finished_rejections',
          'clock_overflow_fixture_only', 'fault_flag_fixture_only')}},
        {'name': CASES[9], 'passed': body['native_checks_passed'] is True and body['model_rejections'] == 6,
         'observed': {'invalid_model_rejections': 6,
                                                     'previous_state_byte_unchanged_checked_natively': True}},
    ])
    for index, case in enumerate(result):
        case['evidence_kind'] = 'numeric_trace_comparison' if index < 7 else 'executed_native_guard_receipt'
        if index >= 7:
            case['observed']['full_state_bytes_retained_for_offline_recomputation'] = False
            case['observed']['native_guard_basis'] = 'frozen-C-return-code-and-memcmp-checks-bound-to-raw-stdout-and-exit'
    need(tuple(row['name'] for row in result) == CASES, 'COMPLETE_CASE_ROSTER')
    return {'kind': kind, 'source_weights_sha256': c.SOURCE_WEIGHTS[kind], 'cases': result,
            'forward_frames': FORWARD_FRAMES, 'supplied_samples_including_replays': SUPPLIED_SAMPLES,
            'native_trace_sha256': c.digest(trace_raw), 'baseline_trace_bit_identical': baseline['baseline_trace_bit_identical'],
            'all_contracts_passed': all(row['passed'] is True for row in result),
            'state_bytes': body['state_bytes'], 'lifetime_resource_observation': lifetime}


def compile_native(output, before):
    compiler = shutil.which('gcc')
    need(compiler is not None and platform.system() == 'Linux', 'LINUX_GCC_REQUIRED')
    binary = output / 'en63_state_cases'
    command = [compiler, *FLAGS, '-I', str(HERE), '-I', str(output), str(HERE / 'en63_stream.c'),
               str(HERE / 'state_cases.c'), '-lm', '-o', str(binary)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    report = {'schema': 'english63-state-native-build-v1', 'command': command, 'exit_code': run.returncode,
              'compiler_version': subprocess.check_output([compiler, '--version'], text=True),
              'source_sha256': before, 'stdout': run.stdout, 'stderr': run.stderr,
              'source_header_sha256': HEADER_SHA, 'machine': platform.machine(), 'libc': list(platform.libc_ver()),
              'data_repository': os.environ['GITHUB_REPOSITORY'], 'data_head': os.environ['GITHUB_SHA'],
              'run_id': os.environ['GITHUB_RUN_ID'], 'run_attempt': os.environ['GITHUB_RUN_ATTEMPT'],
              'pipeline_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'physical_ssc305': False, 'binary_sha256': c.digest(binary.read_bytes()) if binary.exists() else None}
    c.write_json(output / 'BUILD.json', report)
    need(run.returncode == 0, 'COMPILE_FAILED')
    return binary


def native_run(binary, kind, pcm, output):
    command = [str(binary), '--supervise', kind, str(pcm), str(output / (kind + '.trace'))]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate(timeout=10)
    (output / (kind + '.stderr')).write_text(stderr)
    result = {'schema': 'english63-state-native-launch-v1', 'command': command, 'exit_code': process.returncode,
              'timed_out': timed_out, 'raw_stdout': stdout, 'observations': None, 'lifetime': None}
    try:
        lines = stdout.splitlines()
        if len(lines) == 2:
            result['observations'], result['lifetime'] = [json.loads(line) for line in lines]
    finally:
        c.write_json(output / (kind + '-native.json'), result)
    need(not timed_out and process.returncode == 0 and result['observations'] is not None and result['lifetime'] is not None,
         'NATIVE_FAILED_' + kind)
    return result


def artifact_manifest(root):
    need(not (root / 'FAILED.json').exists() and all(not p.is_symlink() for p in root.rglob('*')), 'FAILED_OR_SYMLINK_ARTIFACT')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p != root / 'REPLAY.json')
    expected = {'en63_generated.h', 'en63_state_cases', 'BUILD.json', 'INPUTS.json', 'PROTOCOL.json',
                'SOURCE.zip', 'SUMMARY.json'} | {kind + suffix for kind in c.KINDS for suffix in
                ('.trace', '-native.json', '.stderr', '-baseline-prefix.trace')}
    need({p.relative_to(root).as_posix() for p in files} == expected and len(files) == 15 and
         sum(p.stat().st_size for p in files) < 60 * 1024 * 1024, 'EXACT_ARTIFACT_CLOSURE')
    return {p.relative_to(root).as_posix(): c.digest(p.read_bytes()) for p in files}


def saved_receipt(root):
    files = artifact_manifest(root)
    summary = c.read_json(root / 'SUMMARY.json')
    need(summary['schema'] == SCHEMA and summary['status'] == STATUS and summary['source_audio_clips'] == 63 and
         summary['source_pcm_samples'] == 2101760 and summary['test_unique_pcm_samples'] == TEST_UNIQUE_SAMPLES and
         summary['source_native_run_id'] == BASELINE_RUN and summary['source_header_sha256'] == HEADER_SHA and
         summary['source_tts_zip_sha256'] == c.TTS_SHA and summary['source_review_json_sha256'] == c.REVIEW_SHA and
         summary['source_weights_sha256'] == c.SOURCE_WEIGHTS and summary['source_before_after_verified'] is True and
         all(summary[key] == 0 for key in ('new_independent_audio_samples', 'new_independent_audio_seconds',
                                         'optimizer_updates', 'new_tts_asr_ipa_calls')) and
         all(summary[key] is False for key in FALSE_FIELDS) and
         all(summary[key] == 'NOT_RUN' for key in c.UNMEASURED), 'SUMMARY_SCOPE')
    need(c.read_json(root / 'PROTOCOL.json') == protocol() and c.digest((root / 'en63_generated.h').read_bytes()) == HEADER_SHA,
         'SAVED_PROTOCOL_AND_HEADER')
    inputs, build = c.read_json(root / 'INPUTS.json'), c.read_json(root / 'BUILD.json')
    need(inputs['test_unique_pcm_samples'] == TEST_UNIQUE_SAMPLES and len(inputs['rows']) == 63 and
         sum(row['test_samples_used'] for row in inputs['rows']) == TEST_UNIQUE_SAMPLES and
         sum(row['source_samples'] for row in inputs['rows']) == 2101760 and
         summary['input_bindings_sha256'] == files['INPUTS.json'] and
         build['binary_sha256'] == files['en63_state_cases'] and type(build['exit_code']) is int and
         build['exit_code'] == 0, 'SAVED_INPUT_BUILD_BINDING')
    with zipfile.ZipFile(root / 'SOURCE.zip') as source:
        need(set(source.namelist()) == set(SOURCES) and len(source.namelist()) == len(SOURCES) and
             {name: c.digest(source.read(name)) for name in SOURCES} == build['source_sha256'] == summary['source_sha256'],
             'RETAINED_EXECUTED_SOURCE_BYTES')
    models = []
    for kind in c.KINDS:
        native = c.read_json(root / (kind + '-native.json'))
        need(type(native['exit_code']) is int and native['exit_code'] == 0 and native['timed_out'] is False,
             'SAVED_NATIVE_TERMINAL')
        model = case_observations(kind, (root / (kind + '.trace')).read_bytes(),
                                  (root / (kind + '-baseline-prefix.trace')).read_bytes(), native)
        need(model['all_contracts_passed'], 'SAVED_CASE_FAILURE')
        models.append(model)
    need(models == summary['models'], 'SAVED_CASES_RECOMPUTED_FROM_TRACE_BYTES')
    return {'schema': REPLAY_SCHEMA, 'status': REPLAY_STATUS,
            'files_sha256': files, 'summary_sha256': files['SUMMARY.json'],
            'model_forward_calls': 0, 'optimizer_updates': 0, 'new_audio_asr_calls': 0,
            'formal_training_admitted': False, 'model_promotion_allowed': False}


def verify_saved(root):
    expected = saved_receipt(root)
    need(c.read_json(root / 'REPLAY.json') == expected, 'SAVED_RECEIPT_CHANGED')
    return expected


def require_public_actions():
    need(os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_RUN_ID', '').isdigit() and
         os.environ.get('GITHUB_RUN_ATTEMPT') == '1' and os.environ.get('GITHUB_REPOSITORY') in
         ('jiying2007/kws-data', 'jiying2007/kws-pipeline') and len(os.environ.get('GITHUB_SHA', '')) == 40,
         'PUBLIC_ACTIONS_FIRST_ATTEMPT_ONLY')


def run(args):
    require_public_actions()
    output = args.output_dir
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'CREATE_ONLY_OUTPUT')
    need(all(p is not None and p.is_absolute() for p in (args.baseline_result_dir, args.tts_zip, args.review_json)),
         'ABSOLUTE_INPUTS')
    output.mkdir()
    models = []
    try:
        before = source_hashes()
        c.write_json(output / 'PROTOCOL.json', protocol())
        with zipfile.ZipFile(output / 'SOURCE.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in SOURCES:
                archive.writestr(name, (ROOT / name).read_bytes())
        header, prefixes = baseline_inputs(args.baseline_result_dir)
        (output / 'en63_generated.h').write_bytes(header)
        for kind, raw in prefixes.items():
            (output / (kind + '-baseline-prefix.trace')).write_bytes(raw)
        pcm, bindings = original_pcm(args.tts_zip, args.review_json)
        c.write_json(output / 'INPUTS.json', bindings)
        binary = compile_native(output, before)
        with tempfile.TemporaryDirectory(prefix='english63-state-original-pcm-') as temporary:
            path = Path(temporary) / 'original-prefix.pcm'
            path.write_bytes(pcm)
            for kind in c.KINDS:
                native = native_run(binary, kind, path, output)
                model = case_observations(kind, (output / (kind + '.trace')).read_bytes(), prefixes[kind], native)
                models.append(model)
                need(model['all_contracts_passed'], 'REAL_CASE_FAILURE_' + kind)
            need(c.digest(path.read_bytes()) == bindings['test_pcm_sha256'], 'PCM_BEFORE_AFTER')
        need(source_hashes() == before and baseline_inputs(args.baseline_result_dir) == (header, prefixes) and
             c.digest(args.tts_zip.read_bytes()) == c.TTS_SHA and
             c.digest(args.review_json.read_bytes()) == c.REVIEW_SHA, 'READONLY_SOURCE_BEFORE_AFTER')
        summary = {'schema': SCHEMA, 'status': STATUS, 'source_native_run_id': BASELINE_RUN,
                   'source_native_artifact_id': BASELINE_ARTIFACT, 'source_native_zip_sha256': BASELINE_ZIP_SHA,
                   'source_header_sha256': HEADER_SHA, 'source_training_run_id': 43595338774,
                   'source_weights_sha256': c.SOURCE_WEIGHTS, 'source_tts_zip_sha256': c.TTS_SHA,
                   'source_review_json_sha256': c.REVIEW_SHA, 'source_audio_clips': 63, 'source_pcm_samples': 2101760,
                   'test_unique_pcm_samples': TEST_UNIQUE_SAMPLES, 'source_before_after_verified': True,
                   'source_sha256': before, 'input_bindings_sha256': c.digest((output / 'INPUTS.json').read_bytes()),
                   'models': models, 'new_independent_audio_samples': 0, 'new_independent_audio_seconds': 0,
                   'optimizer_updates': 0, 'new_tts_asr_ipa_calls': 0,
                   'clock_scope': 'supplied-PCM-session-time-not-missing-capture-time-or-acoustic-word-end',
                   'resource_scope': 'Linux-native-test-harness-including-trace-IO-not-core-only-or-SSC305',
                   'per_model_case_evidence': {'numeric_trace_comparisons': 7, 'executed_native_guard_receipts': 3},
                   'actual_capture_discontinuity_measured': False, 'long_duration_24h_soak': 'NOT_RUN',
                   'repeated_diagnostic_audio_is_not_independent_exposure': True,
                   'old_wholeclip_models_or_INT8_chunk8_verdicts_changed': False,
                   **dict.fromkeys(FALSE_FIELDS, False), **dict.fromkeys(c.UNMEASURED, 'NOT_RUN')}
        c.write_json(output / 'SUMMARY.json', summary)
        c.write_json(output / 'REPLAY.json', saved_receipt(output))
        return verify_saved(output)
    except BaseException as error:
        if not (output / 'FAILED.json').exists():
            c.write_json(output / 'FAILED.json', {'schema': SCHEMA, 'status': 'FAILED_NOT_STATE_CONTRACT_EVIDENCE',
                         'error_type': type(error).__name__, 'error': str(error), 'partial_models': models,
                         'forward_budget_per_model': FORWARD_FRAMES, 'unknown_or_failed_calls_not_retried': True,
                         'optimizer_updates': 0, **dict.fromkeys(FALSE_FIELDS, False)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-result-dir', type=Path)
    parser.add_argument('--tts-zip', type=Path)
    parser.add_argument('--review-json', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output-dir', type=Path)
    mode.add_argument('--verify-result-dir', type=Path)
    args = parser.parse_args()
    result = verify_saved(args.verify_result_dir) if args.verify_result_dir else run(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('EN63_STATE_REPLAY_FAILED: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
