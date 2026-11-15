#!/usr/bin/env python3
"""Fixed public-Actions C pool profile; saved verification uses only stdlib."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import wave
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load_source('_en63_pool_original_contract', HERE / 'contract.py')
bridge = load_source('_en63_pool_original_labels', ROOT / 'training/english_research_label_bridge.py')
EXPERIMENT = 'english63-native-c-pool-profile'
SCHEMA = 'english63-pool-profile-result-v1'
REPLAY_SCHEMA = 'english63-pool-profile-replay-v1'
MEASUREMENTS_SCHEMA = 'english63-pool-profile-measurements-v1'
REQUEST_SCHEMA = 'kws-research-v4-native-c-pool-request-v1'
REQUEST_ID = 'en63-native-c-pool-20261115-v1'
HEADER_SHA = '912f443a85fddbc77c49f6cc52c2dbde6c395436d7d8d0818a4e6225cd90fc16'
ANCESTOR = {
    'run_id': 43663731129, 'artifact_id': 14141171320, 'zip_bytes': 1597111,
    'zip_sha256': '2fdd0c951d6cf964d6b73afff5726e7d82b365dae539876822e513d7af0ccaa7',
    'data_commit': '16df89d871f423f8105a22b8bff5556e68685f18',
    'pipeline_commit': '5ed17337558ef1fb48b06f737a2078fcb716a8d3',
    'request_sha256': '5996025f47227d18154fbfa29e7a2bb2fce1322385f53b3d83bebe2e5d6d2e18',
    'terminal_sha256': '00aa4b75d1ca3934992503ac5cd4a698c2ecbc7a48fab51fec660dab70c39e30',
    'header_sha256': HEADER_SHA,
}
PARENT_EVIDENCE = {
    'TERMINAL.json': ANCESTOR['terminal_sha256'],
    'REQUEST.json': ANCESTOR['request_sha256'],
    'state-contracts/REPLAY.json': '0f3fe3bc2057ee37aeeaac16a67a8077658e204295d833dd660576273ed65ae7',
    'state-contracts/INPUTS.json': 'f66635d4f6f4d881c778eec3d1d4b5f6670672123bef7bad01704167c82b7425',
}
BASELINE_TRACE_SHA = {
    'gru': 'aaace50d91252d4a895961c2d9ec4af5d3971a3ce70024ac897bec4c3953a88b',
    'dsc_tcn': 'fd449807d7508adb1fc1dddcaa4022b293988a4406ab3ceb3f3fb5fa220650d8',
}
PROFILE = {
    'macro': 'EN63_POOL_TIME_MAJOR', 'variants': {'baseline': 0, 'candidate': 1},
    'models': ['gru', 'dsc_tcn'], 'source_pcm_samples': 2101760,
    'frames_per_pass': 13134, 'feed_chunk_samples': 160, 'feed_calls_per_pass': 13136,
    'pairs_per_model': {'trace': 1, 'warmup': 1, 'timed': 6},
    'trace_pair_order': ['baseline', 'candidate'], 'warmup_pair_order': ['baseline', 'candidate'],
    'timed_pair_orders': [['baseline', 'candidate'], ['candidate', 'baseline']] * 3,
    'full_passes': 32, 'forward_frames': 420288, 'supplied_samples_including_replays': 67256320,
    'cpu_affinity': 'lowest-allowed-single-cpu', 'native_group_seconds': 180,
    'native_child_seconds': 30, 'state_bytes_max': 38192,
    'acceptance': {'median_paired_cpu_reduction_min': 0.05, 'positive_cpu_pairs_min': 5,
                   'paired_p99_median_ratio_max': 1.10, 'exact_data_equality_required': True},
}
SOURCE_ARTIFACTS = {
    'tts': {'artifact_id': 13954744587, 'zip_sha256': c.TTS_SHA},
    'review': {'artifact_id': 14001747172,
               'zip_sha256': 'f83d6a2f399c20af5bbc920ef6c38c42ac1e44b516d2b7291f4510557491cec8',
               'member': 'english63-three-evidence-review.json', 'member_sha256': c.REVIEW_SHA},
}
FLAGS = ('-std=c11', '-O2', '-ffp-contract=off', '-fno-fast-math',
         '-Wall', '-Wextra', '-Werror', '-pedantic')
SOURCES = ('research/english63_online_c/pool_profile.py',
           'research/english63_online_c/pool_profile.c',
           'research/english63_online_c/en63_stream.c',
           'research/english63_online_c/en63_stream.h',
           'research/english63_online_c/contract.py',
           'training/english_research_label_bridge.py')
RESULT_FILES = {'en63_generated.h', 'en63_pool_baseline', 'en63_pool_candidate', 'BUILD.json',
                'INPUTS.json', 'PROTOCOL.json', 'SOURCE.zip', 'SUMMARY.json', 'REPLAY.json',
                'MEASUREMENTS.json', 'native.stderr'} | {
                    kind + '-' + variant + '.trace' for kind in c.KINDS for variant in PROFILE['variants']}
FALSE_FIELDS = ('formal_training_admitted', 'human_gold', 'model_promotion_allowed',
                'model_selection_made', 'threshold_selection', 'runtime_default_changed',
                'acoustic_word_end_labels_created', 'physical_ssc305_measured',
                'new_audio_generated', 'actual_capture_discontinuity_measured')
TRACE_RECORD = struct.Struct('<IQ38fI')
TRACE_HEADER = b'EN63TRC1' + struct.pack('<II', 1, 168)
TRACE_BYTES = 2206528
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
SHA1 = re.compile(r'[0-9a-f]{40}\Z')
ID = re.compile(r'[1-9][0-9]{0,19}\Z')
STATE_FILES = {'en63_generated.h', 'en63_state_cases', 'BUILD.json', 'INPUTS.json',
               'PROTOCOL.json', 'SOURCE.zip', 'SUMMARY.json', 'REPLAY.json'} | {
                   kind + suffix for kind in c.KINDS for suffix in
                   ('.trace', '-native.json', '.stderr', '-baseline-prefix.trace')}


def need(value, message):
    c.require(value, 'EN63_POOL_' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def exact(value, expected):
    return type(value) is type(expected) and canonical(value) == canonical(expected)


def parse_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('EN63_POOL_NONFINITE_JSON:' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def read_json(path):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size < 32 * 1024 * 1024,
         'JSON_FILE_BOUND')
    return parse_json(path.read_bytes())


def is_int(value, low=0, high=(1 << 63) - 1):
    return type(value) is int and low <= value <= high


def sha(value):
    return type(value) is str and SHA256.fullmatch(value) is not None


def artifact_identity(raw):
    return {'bytes': len(raw), 'sha256': c.digest(raw)}


def source_hashes():
    return {name: c.digest((ROOT / name).read_bytes()) for name in SOURCES}


def schedule():
    rows = []
    for kind in c.KINDS:
        pairs = [('trace', 0, PROFILE['trace_pair_order']), ('warmup', 0, PROFILE['warmup_pair_order'])]
        pairs += [('timed', index, order) for index, order in enumerate(PROFILE['timed_pair_orders'])]
        for phase, pair, order in pairs:
            for variant in order:
                rows.append({'run_index': len(rows), 'kind': kind, 'phase': phase, 'pair': pair,
                             'variant': PROFILE['variants'][variant], 'variant_name': variant})
    return rows


def protocol():
    return {'schema': 'english63-pool-profile-protocol-v1', 'experiment': EXPERIMENT,
            'native_state_ancestor': ANCESTOR, 'profile': PROFILE, 'pass_order': schedule(),
            'baseline_trace_sha256': BASELINE_TRACE_SHA, 'source_header_sha256': HEADER_SHA,
            'source_tts_zip_sha256': c.TTS_SHA, 'source_review_json_sha256': c.REVIEW_SHA,
            'source_weights_sha256': c.SOURCE_WEIGHTS, 'compile_flags': list(FLAGS),
            'input_selection': 'all63-sorted-cell-id-original-PCM-byte-concatenation-no-reset-no-padding',
            'timed_region': 'preloaded-pcm-en63_feed-and-constant-signature-callback',
            'timing_excludes': ['PCM-load', 'allocation', 'initialization', 'trace-file-IO', 'JSON-output'],
            'per_feed_quantile': 'nearest-rank-ceil(p*N)-1-on-sorted-positive-wall-nanoseconds',
            'pair_cpu_reduction': '1-candidate_cpu_ns/baseline_cpu_ns',
            'pair_p99_ratio': 'candidate_feed_p99_ns/baseline_feed_p99_ns',
            'aggregate_decision': 'BENEFIT-only-if-both-models-meet-all-fixed-resource-and-identity-gates',
            'no_benefit_is_valid_completed_experiment': True,
            'no_result_driven_extra_passes_or_retries': True,
            'resource_scope': 'Linux-host-full-feed-chain-with-constant-signature-not-SSC305-or-acoustic-quality',
            'new_independent_audio_samples': 0, 'new_independent_audio_seconds': 0,
            'optimizer_updates': 0, 'new_tts_asr_ipa_calls': 0,
            **dict.fromkeys(FALSE_FIELDS, False), **dict.fromkeys(c.UNMEASURED, 'NOT_RUN')}


def public_run(value):
    need(type(value) is dict and set(value) == {'repository', 'head', 'run_id', 'run_attempt'} and
         value['repository'] == 'jiying2007/kws-data' and type(value['head']) is str and
         SHA1.fullmatch(value['head']) is not None and type(value['run_id']) is str and
         ID.fullmatch(value['run_id']) is not None and value['run_attempt'] == '1', 'PUBLIC_RUN_IDENTITY')
    return value


def require_public_actions():
    need(os.environ.get('GITHUB_ACTIONS') == 'true', 'PUBLIC_ACTIONS_FIRST_ATTEMPT_ONLY')
    return public_run({'repository': os.environ.get('GITHUB_REPOSITORY'),
                       'head': os.environ.get('GITHUB_SHA'), 'run_id': os.environ.get('GITHUB_RUN_ID'),
                       'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT')})


def validate_request(raw, pipeline_head):
    request = parse_json(raw)
    fixed = {'schema': REQUEST_SCHEMA, 'experiment': EXPERIMENT, 'request_id': REQUEST_ID,
             'pipeline_repository': 'jiying2007/kws-pipeline',
             'entrypoint': 'research/english63_online_c/pool_profile.py',
             'source_artifacts': SOURCE_ARTIFACTS, 'native_state_ancestor': ANCESTOR, 'profile': PROFILE,
             'optimizer_updates': 0, 'cpu_threads': 1, 'max_execution_seconds': 600,
             'max_verification_seconds': 180, 'new_audio_generated': False, 'new_tts_asr_ipa_calls': 0,
             'new_independent_audio_seconds': 0, 'formal_training_admitted': False,
             'human_gold': False, 'model_promotion_allowed': False}
    need(type(request) is dict and set(request) == set(fixed) | {'pipeline_commit', 'reason'} and
         all(exact(request[key], value) for key, value in fixed.items()) and
         type(pipeline_head) is str and SHA1.fullmatch(pipeline_head) is not None and
         request['pipeline_commit'] == pipeline_head and type(request['reason']) is str and
         1 <= len(request['reason'].strip()) <= 500, 'FIXED_REQUEST_AND_CHECKOUT')
    return request


def parent_evidence_check(evidence):
    need(type(evidence) is dict and set(evidence) == set(PARENT_EVIDENCE), 'PARENT_EVIDENCE_CLOSURE')
    parsed = {}
    for name, expected in PARENT_EVIDENCE.items():
        record = evidence[name]
        need(type(record) is dict and set(record) == {'raw_utf8', 'bytes', 'sha256'} and
             type(record['raw_utf8']) is str, 'PARENT_EVIDENCE_RECORD')
        raw = record['raw_utf8'].encode('utf-8')
        need(exact(artifact_identity(raw), {key: record[key] for key in ('bytes', 'sha256')}) and
             record['sha256'] == expected, 'PARENT_ORIGINAL_BYTES')
        parsed[name] = parse_json(raw)
    terminal, request = parsed['TERMINAL.json'], parsed['REQUEST.json']
    need(terminal['status'] == 'SUCCEEDED' and terminal['run_id'] == str(ANCESTOR['run_id']) and
         terminal['run_attempt'] == '1' and terminal['pipeline_commit'] == ANCESTOR['pipeline_commit'] and
         terminal['data_commit'] == ANCESTOR['data_commit'] and
         terminal['request_sha256'] == ANCESTOR['request_sha256'] and
         request['pipeline_commit'] == ANCESTOR['pipeline_commit'], 'PARENT_TERMINAL')
    replay = parsed['state-contracts/REPLAY.json']
    need(set(replay['files_sha256']) == STATE_FILES - {'REPLAY.json'} and
         all(terminal['files_sha256']['state-contracts/' + name] == digest
             for name, digest in replay['files_sha256'].items()) and
         terminal['files_sha256']['state-contracts/REPLAY.json'] == PARENT_EVIDENCE['state-contracts/REPLAY.json'] and
         replay['files_sha256']['en63_generated.h'] == HEADER_SHA, 'PARENT_SAVED_HASH_CLOSURE')
    return parsed


def parent_inputs(root):
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'PARENT_RESULT_ROOT')
    evidence = {}
    for name in PARENT_EVIDENCE:
        path = root / name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size < 1024 * 1024,
             'PARENT_METADATA_FILE')
        raw = path.read_bytes()
        evidence[name] = {'raw_utf8': raw.decode('utf-8'), **artifact_identity(raw)}
    parsed = parent_evidence_check(evidence)
    terminal = parsed['TERMINAL.json']
    entries = list(root.rglob('*'))
    need(all(not path.is_symlink() for path in entries) and
         {path.relative_to(root).as_posix() for path in entries if path.is_dir()} == {'state-contracts'},
         'PARENT_RESULT_PATHS')
    files = {path.relative_to(root).as_posix(): c.digest(path.read_bytes())
             for path in entries if path.is_file() and path != root / 'TERMINAL.json'}
    need(exact(files, terminal['files_sha256']), 'PARENT_RESULT_ALL_FILES_UNCHANGED')
    header = (root / 'state-contracts/en63_generated.h').read_bytes()
    need(c.digest(header) == HEADER_SHA, 'PARENT_EXPORTED_WEIGHTS')
    return header, evidence, files


def original_pcm(tts, review, evidence):
    need(tts.is_absolute() and tts.is_file() and not tts.is_symlink() and
         tts.stat().st_size < 64 * 1024 * 1024 and c.digest(tts.read_bytes()) == c.TTS_SHA,
         'ORIGINAL_TTS_ARCHIVE')
    rows = sorted(bridge.diagnostic_plan(bridge.read_verified_report(review))['rows'], key=lambda row: row['cell_id'])
    need(Counter(row['action'] for row in rows) == {'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY': 30,
         'EXPOSED_DEVELOPMENT_NO_TUNE': 12, 'QUARANTINE_AMBIGUOUS_NEARPHONE': 21}, 'ORIGINAL_ROLE_ROSTER')
    old = parent_evidence_check(evidence)['state-contracts/INPUTS.json']['rows']
    need(len(rows) == len(old) == 63, 'ALL63_ROSTER')
    chunks, bindings, total = [], [], 0
    with zipfile.ZipFile(tts) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)), 'SOURCE_ZIP_DUPLICATES')
        wavs = {name for name in names if name.endswith('/audio-16k.wav')}
        need(len(wavs) == 63, 'ORIGINAL_WAV_ROSTER')
        for row, prior in zip(rows, old):
            name = 'english-seven-voice-actual/capture/cells/' + row['cell_id'] + '/audio-16k.wav'
            need(name in wavs, 'SOURCE_WAV_MEMBER')
            wavs.remove(name)
            raw = archive.read(name)
            need(c.digest(raw) == row['wav_sha256'], 'ORIGINAL_WAV_BYTES')
            with wave.open(io.BytesIO(raw), 'rb') as wav:
                count = wav.getnframes()
                need(wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 16000 and
                     wav.getcomptype() == 'NONE' and 8000 <= count <= 192000, 'ORIGINAL_PCM_GEOMETRY')
                pcm = wav.readframes(count)
                need(len(pcm) == count * 2 and not wav.readframes(1), 'ORIGINAL_PCM_LENGTH')
            need(c.digest(pcm) == row['pcm_sha256'], 'ORIGINAL_PCM_BYTES')
            record = {'cell_id': row['cell_id'], 'member': name, 'wav_sha256': row['wav_sha256'],
                      'pcm_sha256': row['pcm_sha256'], 'action': row['action'],
                      'source_family_id': row['source_family_id'], 'source_samples': count,
                      'concatenation_start_sample': total, 'test_start_within_source': 0,
                      'test_samples_used': count, 'human_gold': False, 'training_admitted': False}
            need(exact({k: v for k, v in record.items() if k != 'test_samples_used'},
                       {k: v for k, v in prior.items() if k != 'test_samples_used'}), 'PARENT_SOURCE_ROW_JOIN')
            chunks.append(pcm)
            bindings.append(record)
            total += count
        need(not wavs and total == PROFILE['source_pcm_samples'], 'FULL_PCM_INVENTORY')
    pcm = b''.join(chunks)
    need(len(pcm) == 4203520, 'FULL_PCM_LENGTH')
    return pcm, bindings


def verify_inputs(inputs, pipeline_head, run):
    need(inputs['schema'] == 'english63-pool-profile-inputs-v1' and
         exact(inputs['native_state_ancestor'], ANCESTOR) and inputs['source_header_sha256'] == HEADER_SHA and
         inputs['source_tts_zip_sha256'] == c.TTS_SHA and inputs['source_review_json_sha256'] == c.REVIEW_SHA and
         exact(inputs['source_weights_sha256'], c.SOURCE_WEIGHTS) and
         exact(inputs['source_audio_clips'], 63) and exact(inputs['source_pcm_samples'], 2101760) and
         exact(inputs['unique_input_samples'], 2101760) and exact(inputs['test_pcm_bytes'], 4203520) and
         sha(inputs['test_pcm_sha256']) and inputs['pcm_operation'] == 'byte-preserving-full63-concatenation' and
         inputs['new_audio_generation_or_resampling'] is False, 'SAVED_INPUT_SCOPE')
    old = parent_evidence_check(inputs['parent_evidence'])['state-contracts/INPUTS.json']['rows']
    expected = [{**row, 'test_samples_used': row['source_samples']} for row in old]
    need(exact(inputs['rows'], expected) and sum(row['test_samples_used'] for row in expected) == 2101760,
         'SAVED_FULL63_EXACT_PARENT_ROSTER')
    raw = inputs['request_raw_utf8'].encode('utf-8')
    validate_request(raw, pipeline_head)
    need(c.digest(raw) == inputs['request_sha256'], 'REQUEST_ORIGINAL_BYTES')
    claim_raw = inputs['claim_raw_utf8'].encode('utf-8')
    claim = parse_json(claim_raw)
    expected_claim = {'schema': 'kws-research-v4-native-c-pool-start-claim-v1',
                      'request_sha256': c.digest(raw), 'pipeline_commit': pipeline_head,
                      'source_state_artifact_id': ANCESTOR['artifact_id'], 'run_id': run['run_id'],
                      'run_attempt': '1', 'optimizer_updates': 0,
                      'status': 'CONSUMED_BEFORE_PAIRED_NATIVE_C_POOL_EXPERIMENT'}
    need(exact(claim, expected_claim) and c.digest(claim_raw) == inputs['claim_sha256'], 'CONSUMED_START_CLAIM')
    return inputs['request_sha256']


def signature64(raw):
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return format(value, '016x')


def trace_observation(raw):
    need(type(raw) is bytes and len(raw) == TRACE_BYTES and raw[:16] == TRACE_HEADER, 'FULL_TRACE_LAYOUT')
    candidate, confirmations, blocked_until, proposals = 0, 0, 0, []
    threshold = struct.unpack('<f', struct.pack('<f', 0.8))[0]
    for index, values in enumerate(TRACE_RECORD.iter_unpack(raw[16:]), 1):
        stream, sample = values[:2]
        numbers, actual = values[2:40], values[40]
        need(stream == 0 and sample == 400 + (index - 1) * 160 and
             all(math.isfinite(number) for number in numbers) and actual in (0, 1, 2), 'TRACE_VALUES_AND_CLOCK')
        probabilities = numbers[35:38]
        need(all(0 <= value <= 1 for value in probabilities) and abs(sum(probabilities) - 1) <= 3e-7,
             'TRACE_PROBABILITY_DOMAIN')
        chosen = max(range(3), key=lambda position: probabilities[position])
        expected = 0
        if index < 50 or index < blocked_until or chosen == 0 or probabilities[chosen] < threshold:
            candidate, confirmations = 0, 0
        else:
            confirmations = confirmations + 1 if chosen == candidate else 1
            candidate = chosen
            if confirmations >= 3:
                expected, blocked_until = chosen, index + 100
                candidate, confirmations = 0, 0
        need(actual == expected, 'SAVED_DECODER_FROM_RAW_PROBABILITIES')
        if actual:
            proposals.append({'frame': index, 'sample': sample, 'class': actual})
    need(index == PROFILE['frames_per_pass'], 'TRACE_FRAME_BUDGET')
    return {'sha256': c.digest(raw), 'frames': index, 'signature64': signature64(raw[16:]),
            'proposal_count': len(proposals), 'proposals': proposals,
            'first_sample': 400, 'last_sample': 2101680, 'eof_padding_samples': 0}


def percentile(values, percent):
    need(type(values) is list and values and all(is_int(value, 1) for value in values) and
         is_int(percent, 1, 100), 'POSITIVE_INTEGER_QUANTILE_INPUT')
    return sorted(values)[(percent * len(values) + 99) // 100 - 1]


def pass_observation(record, expected, cpu, trace, binaries):
    need(type(record) is dict and all(exact(record.get(key), value) for key, value in expected.items()) and
         record.get('status') == 'COMPLETED' and type(record.get('exit_code')) is int and record['exit_code'] == 0 and
         record.get('timed_out') is False and record.get('binary_sha256') == binaries[expected['variant_name']] and
         type(record.get('raw_stdout')) is str and type(record.get('raw_stderr')) is str, 'NATIVE_PASS_TERMINAL')
    command = record.get('command')
    need(type(command) is list and len(command) == 7 and all(type(value) is str for value in command) and
         Path(command[0]).is_absolute() and Path(command[0]).name == 'en63_pool_' + expected['variant_name'] and
         command[1:4] == ['--supervise', expected['kind'], expected['phase']] and
         Path(command[4]).is_absolute() and Path(command[4]).name == 'full-original.pcm' and
         command[6] == str(cpu) and
         ((expected['phase'] == 'trace' and Path(command[5]).is_absolute() and
           Path(command[5]).parent == Path(command[0]).parent and
           Path(command[5]).name == expected['kind'] + '-' + expected['variant_name'] + '.trace') or
          (expected['phase'] != 'trace' and command[5] == '-')), 'NATIVE_FIXED_COMMAND')
    lines = record['raw_stdout'].splitlines()
    need(len(lines) == 2, 'TWO_NATIVE_STDOUT_RECORDS')
    body, lifetime = [parse_json(line) for line in lines]
    need(exact(body, record.get('worker')) and exact(lifetime, record.get('lifetime')), 'RAW_NATIVE_STDOUT_BINDING')
    fixed = {'schema': 'english63-pool-native-pass-v1', 'kind': expected['kind'], 'variant': expected['variant'],
             'phase': expected['phase'], 'affinity_cpu': cpu, 'affinity_single_cpu': True,
             'pcm_samples': 2101760, 'frames': 13134, 'feed_calls': 13136,
             'proposal_count': trace['proposal_count'], 'signature64': trace['signature64'],
             'state_bytes': 38192, 'weight_parameter_bytes': (6531 if expected['kind'] == 'gru' else 4803) * 4,
             'frontend_table_bytes': 3894, 'input_pcm_bytes': 4203520,
             'trace_io_enabled': expected['phase'] == 'trace', 'eof_padding_samples': 0, 'eof_events': 0,
             'timed_region': 'preloaded-pcm-en63_feed-and-constant-signature-callback'}
    need(type(body) is dict and set(body) == set(fixed) | {'cpu_ns', 'wall_ns', 'feed_ns',
         'inference_end_high_water_rss_kib'} and all(exact(body[key], value) for key, value in fixed.items()),
         'NATIVE_FIXED_COUNTERS_AND_SIGNATURE')
    need(type(lifetime) is dict and set(lifetime) == {'schema', 'child_exit_code', 'child_signaled',
         'process_lifetime_peak_rss_kib', 'child_user_us', 'child_system_us'} and
         lifetime['schema'] == 'english63-pool-child-lifetime-v1' and
         exact(lifetime['child_exit_code'], 0) and lifetime['child_signaled'] is False and
         is_int(lifetime['process_lifetime_peak_rss_kib'], 1) and
         is_int(body['inference_end_high_water_rss_kib'], 1) and
         lifetime['process_lifetime_peak_rss_kib'] >= body['inference_end_high_water_rss_kib'] and
         is_int(lifetime['child_user_us']) and is_int(lifetime['child_system_us']), 'NATIVE_CHILD_LIFETIME')
    timed = expected['phase'] == 'timed'
    if timed:
        need(is_int(body['cpu_ns'], 1) and is_int(body['wall_ns'], 1) and
             body['wall_ns'] < PROFILE['native_child_seconds'] * 1000000000 and
             type(body['feed_ns']) is list and len(body['feed_ns']) == 13136 and
             all(is_int(value, 1) for value in body['feed_ns']) and sum(body['feed_ns']) <= body['wall_ns'],
             'TIMED_FEED_OBSERVATIONS')
        return {'run_index': expected['run_index'], 'pair': expected['pair'], 'variant': expected['variant'],
                'variant_name': expected['variant_name'], 'cpu_ns': body['cpu_ns'], 'wall_ns': body['wall_ns'],
                'feed_p50_ns': percentile(body['feed_ns'], 50), 'feed_p95_ns': percentile(body['feed_ns'], 95),
                'feed_p99_ns': percentile(body['feed_ns'], 99), 'feed_max_ns': max(body['feed_ns']),
                'state_bytes': body['state_bytes'],
                'inference_end_high_water_rss_kib': body['inference_end_high_water_rss_kib'],
                'process_lifetime_peak_rss_kib': lifetime['process_lifetime_peak_rss_kib']}
    need(exact(body['cpu_ns'], 0) and exact(body['wall_ns'], 0) and exact(body['feed_ns'], []),
         'UNTIMED_PASS_HAS_NO_TIMING_EVIDENCE')
    return None


def model_result(kind, trace, timed):
    need(len(timed) == 12, 'SIX_COMPLETE_TIMED_PAIRS')
    pairs = []
    for pair in range(6):
        rows = {row['variant']: row for row in timed if row['pair'] == pair}
        need(set(rows) == {0, 1} and len([row for row in timed if row['pair'] == pair]) == 2, 'PAIR_ROSTER')
        baseline, candidate = rows[0], rows[1]
        pairs.append({'pair': pair, 'order': PROFILE['timed_pair_orders'][pair],
                      'baseline_cpu_ns': baseline['cpu_ns'], 'candidate_cpu_ns': candidate['cpu_ns'],
                      'baseline_p99_ns': baseline['feed_p99_ns'], 'candidate_p99_ns': candidate['feed_p99_ns'],
                      'cpu_reduction': 1 - candidate['cpu_ns'] / baseline['cpu_ns'],
                      'p99_ratio': candidate['feed_p99_ns'] / baseline['feed_p99_ns']})
    reductions = [row['cpu_reduction'] for row in pairs]
    ratios = [row['p99_ratio'] for row in pairs]
    median, wins, p99 = statistics.median(reductions), sum(value > 0 for value in reductions), statistics.median(ratios)
    benefit = median >= 0.05 and wins >= 5 and p99 <= 1.10
    return {'kind': kind, 'source_weights_sha256': c.SOURCE_WEIGHTS[kind], 'bit_identity': True,
            'baseline_trace_sha256': trace['sha256'], 'candidate_trace_sha256': trace['sha256'],
            'signature64': trace['signature64'], 'proposal_count': trace['proposal_count'],
            'state_bytes': 38192, 'state_bytes_increase': 0, 'timed_passes': timed, 'pairs': pairs,
            'paired_cpu_reductions': reductions, 'paired_cpu_reduction_median': median,
            'paired_cpu_reduction_range': [min(reductions), max(reductions)], 'positive_cpu_pairs': wins,
            'paired_p99_ratios': ratios, 'paired_p99_ratio_median': p99,
            'decision': 'BENEFIT' if benefit else 'NO_BENEFIT',
            'gate_observations': {'median_cpu_reduction': median >= 0.05, 'positive_cpu_pairs': wins >= 5,
                                  'paired_p99': p99 <= 1.10, 'state_not_increased': True, 'bit_identity': True}}


def measurements_result(measurements, traces, binaries):
    need(type(measurements) is dict and measurements['schema'] == MEASUREMENTS_SCHEMA and
         type(measurements['initial_allowed_cpus']) is list and measurements['initial_allowed_cpus'] and
         all(is_int(cpu, 0, 1023) for cpu in measurements['initial_allowed_cpus']) and
         measurements['initial_allowed_cpus'] == sorted(set(measurements['initial_allowed_cpus'])) and
         exact(measurements['selected_cpu'], min(measurements['initial_allowed_cpus'])) and
         exact(measurements['final_allowed_cpus'], measurements['initial_allowed_cpus']) and
         type(measurements['native_group_elapsed_seconds']) in (int, float) and
         math.isfinite(measurements['native_group_elapsed_seconds']) and
         0 < measurements['native_group_elapsed_seconds'] <= 180, 'INITIAL_SINGLE_CPU_AND_BUDGET')
    expected = schedule()
    need(type(measurements['passes']) is list and len(measurements['passes']) == len(expected) == 32,
         'EXACT32_PASS_BUDGET')
    by_kind = {kind: [] for kind in c.KINDS}
    pcm_path = None
    for record, plan in zip(measurements['passes'], expected):
        row = pass_observation(record, plan, measurements['selected_cpu'], traces[plan['kind']], binaries)
        if pcm_path is None:
            pcm_path = record['command'][4]
        need(record['command'][4] == pcm_path, 'SAME_PRELOADED_PCM_PATH')
        if row is not None:
            by_kind[plan['kind']].append(row)
    return [model_result(kind, traces[kind], by_kind[kind]) for kind in c.KINDS]


def native_environment():
    return {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'LANG': 'C', 'LC_ALL': 'C',
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def command_for(compiler, source_root, output_root, variant):
    here = Path(source_root) / 'research/english63_online_c'
    return [compiler, *FLAGS, '-DEN63_POOL_TIME_MAJOR=' + str(PROFILE['variants'][variant]),
            '-I', str(here), '-I', str(output_root), str(here / 'en63_stream.c'),
            str(here / 'pool_profile.c'), '-lm', '-o', str(Path(output_root) / ('en63_pool_' + variant))]


def compile_native(output, before, run, pipeline_head):
    compiler = shutil.which('gcc')
    need(compiler is not None and platform.system() == 'Linux', 'LINUX_GCC_REQUIRED')
    compiler = str(Path(compiler).absolute())
    actual_compiler = Path(compiler).resolve()
    compiler_sha = c.digest(actual_compiler.read_bytes())
    version = subprocess.check_output([compiler, '--version'], text=True, timeout=15, env=native_environment())
    report = {'schema': 'english63-pool-native-build-v1', 'compiler': compiler,
              'compiler_realpath': str(actual_compiler), 'compiler_sha256': compiler_sha,
              'compiler_version': version, 'source_root': str(ROOT), 'output_root': str(output),
              'source_sha256': before, 'source_header_sha256': HEADER_SHA,
              'public_run': run, 'pipeline_head': pipeline_head, 'compile_flags': list(FLAGS),
              'machine': platform.machine(), 'libc': list(platform.libc_ver()),
              'physical_ssc305': False, 'compiles': []}
    for variant in PROFILE['variants']:
        command = command_for(compiler, ROOT, output, variant)
        done = subprocess.run(command, capture_output=True, text=True, timeout=90,
                              check=False, env=native_environment())
        binary = output / ('en63_pool_' + variant)
        report['compiles'].append({'variant_name': variant, 'variant': PROFILE['variants'][variant],
                                   'command': command, 'exit_code': done.returncode,
                                   'stdout': done.stdout, 'stderr': done.stderr,
                                   'binary_sha256': c.digest(binary.read_bytes()) if binary.exists() else None})
        if done.returncode != 0:
            c.write_json(output / 'BUILD.json', report)
            need(False, 'NATIVE_COMPILE_FAILED_' + variant)
    need(c.digest(actual_compiler.read_bytes()) == compiler_sha, 'SAME_COMPILER_BEFORE_AFTER')
    c.write_json(output / 'BUILD.json', report)
    return report


def validate_build(build, files, source):
    need(build['schema'] == 'english63-pool-native-build-v1' and
         type(build['compiler']) is str and Path(build['compiler']).is_absolute() and
         type(build['compiler_realpath']) is str and Path(build['compiler_realpath']).is_absolute() and
         sha(build['compiler_sha256']) and type(build['compiler_version']) is str and build['compiler_version'] and
         type(build['source_root']) is str and Path(build['source_root']).is_absolute() and
         type(build['output_root']) is str and Path(build['output_root']).is_absolute() and
         exact(build['compile_flags'], list(FLAGS)) and build['source_header_sha256'] == HEADER_SHA and
         exact(build['source_sha256'], source) and build['physical_ssc305'] is False and
         type(build['compiles']) is list and len(build['compiles']) == 2, 'SAVED_BUILD_CONTRACT')
    public_run(build['public_run'])
    need(type(build['pipeline_head']) is str and SHA1.fullmatch(build['pipeline_head']) is not None,
         'PIPELINE_CHECKOUT_IDENTITY')
    binaries = {}
    for row, variant in zip(build['compiles'], PROFILE['variants']):
        expected = command_for(build['compiler'], build['source_root'], build['output_root'], variant)
        need(row['variant_name'] == variant and exact(row['variant'], PROFILE['variants'][variant]) and
             exact(row['exit_code'], 0) and exact(row['command'], expected) and
             type(row['stdout']) is str and type(row['stderr']) is str and
             row['binary_sha256'] == files['en63_pool_' + variant]['sha256'], 'PAIRED_FIXED_COMPILER_COMMANDS')
        binaries[variant] = row['binary_sha256']
    return binaries


def artifact_manifest(root):
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'SAVED_RESULT_ROOT')
    entries = list(root.rglob('*'))
    need(all(path.is_file() and not path.is_symlink() for path in entries), 'FLAT_REGULAR_FILE_CLOSURE')
    files = {path.name: path for path in entries if path.name != 'REPLAY.json'}
    need(set(files) == RESULT_FILES - {'REPLAY.json'} and len(files) == 14 and
         all(path.stat().st_size > 0 for name, path in files.items() if name != 'native.stderr') and
         sum(path.stat().st_size for path in files.values()) < 60 * 1024 * 1024, 'EXACT15_RESULT_CLOSURE')
    return {name: artifact_identity(path.read_bytes()) for name, path in sorted(files.items())}


def source_archive(root, build):
    with zipfile.ZipFile(root / 'SOURCE.zip') as archive:
        need(len(archive.infolist()) == len(SOURCES) and set(archive.namelist()) == set(SOURCES) and
             all(not item.is_dir() and item.file_size < 1024 * 1024 and
                 (item.external_attr >> 16) & 0o170000 != 0o120000 for item in archive.infolist()),
             'SOURCE_ARCHIVE_CLOSURE')
        source = {name: c.digest(archive.read(name)) for name in SOURCES}
        need(exact(source, build['source_sha256']) and exact(source, source_hashes()), 'EXECUTED_SOURCE_BODY_HASHES')
    return source


def summary_from(inputs, build, measurements, per_kind, source):
    return {'schema': SCHEMA, 'status': 'SUCCEEDED', 'experiment': EXPERIMENT,
            'decision': 'BENEFIT' if all(row['decision'] == 'BENEFIT' for row in per_kind) else 'NO_BENEFIT',
            'per_kind': per_kind, 'native_state_ancestor': ANCESTOR,
            'source_header_sha256': HEADER_SHA, 'source_training_run_id': 43595338774,
            'source_tts_zip_sha256': c.TTS_SHA, 'source_review_json_sha256': c.REVIEW_SHA,
            'source_weights_sha256': c.SOURCE_WEIGHTS, 'source_audio_clips': 63, 'source_pcm_samples': 2101760,
            'unique_input_samples': 2101760, 'total_supplied_samples': 67256320,
            'total_frames': 420288, 'passes': 32, 'source_before_after_verified': True,
            'source_sha256': source, 'input_bindings_sha256': inputs['file_sha256'],
            'measurements_sha256': measurements['file_sha256'], 'public_run': build['public_run'],
            'pipeline_head': build['pipeline_head'], 'request_sha256': inputs['request_sha256'],
            'selected_cpu': measurements['selected_cpu'], 'initial_allowed_cpus': measurements['initial_allowed_cpus'],
            'native_group_elapsed_seconds': measurements['native_group_elapsed_seconds'],
            'source_integrity': measurements['source_integrity'],
            'new_independent_audio_samples': 0, 'new_independent_audio_seconds': 0,
            'new_tts_asr_ipa_calls': 0, 'optimizer_updates': 0,
            'resource_scope': protocol()['resource_scope'],
            'clock_scope': 'supplied-PCM-input-dependency-not-acoustic-word-end-or-wall-clock-latency',
            'timed_region': protocol()['timed_region'],
            'repeated_diagnostic_audio_is_not_independent_exposure': True,
            'old_wholeclip_models_or_INT8_chunk8_verdicts_changed': False,
            **dict.fromkeys(FALSE_FIELDS, False), **dict.fromkeys(c.UNMEASURED, 'NOT_RUN')}


def saved_receipt(root):
    files = artifact_manifest(root)
    need(exact(read_json(root / 'PROTOCOL.json'), protocol()) and files['en63_generated.h']['sha256'] == HEADER_SHA,
         'SAVED_PROTOCOL_AND_EXPORTED_HEADER')
    build, inputs, measurements = [read_json(root / name) for name in ('BUILD.json', 'INPUTS.json', 'MEASUREMENTS.json')]
    source = source_archive(root, build)
    binaries = validate_build(build, files, source)
    verify_inputs(inputs, build['pipeline_head'], build['public_run'])
    need(exact(measurements['public_run'], build['public_run']) and
         measurements['pipeline_head'] == build['pipeline_head'] and
         measurements['request_sha256'] == inputs['request_sha256'], 'MEASUREMENTS_EXECUTION_IDENTITY')
    need(all(row.get('command', [None])[0] ==
             str(Path(build['output_root']) / ('en63_pool_' + plan['variant_name']))
             for row, plan in zip(measurements['passes'], schedule())), 'EXECUTED_BINARY_PATH_BINDING')
    for variant in PROFILE['variants']:
        need((root / ('en63_pool_' + variant)).read_bytes().startswith(b'\x7fELF'), 'SAVED_NATIVE_ELF')
    traces = {}
    for kind in c.KINDS:
        baseline = (root / (kind + '-baseline.trace')).read_bytes()
        candidate = (root / (kind + '-candidate.trace')).read_bytes()
        need(c.digest(baseline) == BASELINE_TRACE_SHA[kind] and candidate == baseline,
             'COMPLETE_OLD_BASELINE_AND_CANDIDATE_BIT_IDENTITY')
        traces[kind] = trace_observation(baseline)
    need((root / 'native.stderr').read_text() == ''.join(row['raw_stderr'] for row in measurements['passes']),
         'NATIVE_STDERR_ORIGINAL_BYTES')
    integrity = {'source_before': source, 'source_after': source, 'header_before': HEADER_SHA, 'header_after': HEADER_SHA,
                 'tts_before': c.TTS_SHA, 'tts_after': c.TTS_SHA, 'review_before': c.REVIEW_SHA, 'review_after': c.REVIEW_SHA,
                 'pcm_before': inputs['test_pcm_sha256'], 'pcm_after': inputs['test_pcm_sha256'],
                 'parent_terminal_before': ANCESTOR['terminal_sha256'], 'parent_terminal_after': ANCESTOR['terminal_sha256'],
                 'parent_files_before': parent_evidence_check(inputs['parent_evidence'])['TERMINAL.json']['files_sha256'],
                 'parent_files_after': parent_evidence_check(inputs['parent_evidence'])['TERMINAL.json']['files_sha256'],
                 'binaries_before': binaries, 'binaries_after': binaries,
                 'request_before': inputs['request_sha256'], 'request_after': inputs['request_sha256'],
                 'claim_before': inputs['claim_sha256'], 'claim_after': inputs['claim_sha256']}
    need(exact(measurements['source_integrity'], integrity), 'ALL_OBSERVED_BEFORE_AFTER_BINDINGS')
    per_kind = measurements_result(measurements, traces, binaries)
    observed = summary_from({**inputs, 'file_sha256': files['INPUTS.json']['sha256']}, build,
                            {**measurements, 'file_sha256': files['MEASUREMENTS.json']['sha256']}, per_kind, source)
    need(exact(read_json(root / 'SUMMARY.json'), observed), 'SUMMARY_RECOMPUTED_FROM_SAVED_NATIVE_RECORDS')
    return {'schema': REPLAY_SCHEMA, 'status': 'VERIFIED_SAVED_POOL_PROFILE', 'files': files,
            'files_sha256': {name: value['sha256'] for name, value in files.items()},
            'summary_sha256': files['SUMMARY.json']['sha256'], 'model_forward_calls': 0, 'optimizer_updates': 0,
            'new_audio_asr_calls': 0, 'formal_training_admitted': False, 'model_promotion_allowed': False,
            'runtime_default_changed': False}


def verify_saved(root):
    expected = saved_receipt(root)
    need(exact(read_json(root / 'REPLAY.json'), expected), 'SAVED_RECEIPT_CHANGED')
    return expected


def save_progress(path, value):
    temporary = path.with_name(path.name + '.pending')
    c.write_json(temporary, value)
    os.replace(temporary, path)


def native_pass(binary, expected, pcm, trace_path, cpu, deadline):
    remaining = deadline - time.monotonic()
    need(remaining > 0, 'NATIVE_GROUP_WALL_LIMIT')
    command = [str(binary), '--supervise', expected['kind'], expected['phase'], str(pcm),
               str(trace_path) if trace_path else '-', str(cpu)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True, env=native_environment())
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=min(PROFILE['native_child_seconds'], remaining))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate(timeout=10)
    result = {**expected, 'status': 'COMPLETED' if not timed_out and process.returncode == 0 else 'FAILED',
              'command': command, 'exit_code': process.returncode, 'timed_out': timed_out,
              'raw_stdout': stdout, 'raw_stderr': stderr, 'worker': None, 'lifetime': None,
              'binary_sha256': c.digest(binary.read_bytes())}
    lines = stdout.splitlines()
    try:
        if len(lines) == 2:
            result['worker'], result['lifetime'] = [parse_json(line) for line in lines]
    except (ValueError, TypeError):
        result['status'] = 'FAILED'
    return result


def run(args):
    run_identity = require_public_actions()  # Before any input, compiler, PCM or child activity.
    output = args.output_dir
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not any(path.is_symlink() for path in (output, *output.parents)), 'CREATE_ONLY_OUTPUT')
    need(all(path is not None and path.is_absolute() for path in
             (args.state_result_dir, args.tts_zip, args.review_json)), 'ABSOLUTE_INPUTS')
    pipeline_head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    request_path = output.parent / 'REQUEST.json'
    claim_path = output.parent.parent / 'claim/CLAIM.json'
    request_raw, claim_raw = request_path.read_bytes(), claim_path.read_bytes()
    validate_request(request_raw, pipeline_head)
    need(hasattr(os, 'sched_getaffinity') and platform.system() == 'Linux', 'CPU_AFFINITY_REQUIRED')
    allowed = sorted(os.sched_getaffinity(0))
    need(allowed and all(is_int(cpu, 0, 1023) for cpu in allowed), 'INITIAL_ALLOWED_CPU_SET')
    selected_cpu = allowed[0]
    output.mkdir()
    measurements = {'schema': MEASUREMENTS_SCHEMA, 'public_run': run_identity, 'pipeline_head': pipeline_head,
                    'request_sha256': c.digest(request_raw), 'initial_allowed_cpus': allowed,
                    'selected_cpu': selected_cpu, 'passes': []}
    try:
        before = source_hashes()
        c.write_json(output / 'PROTOCOL.json', protocol())
        with zipfile.ZipFile(output / 'SOURCE.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in SOURCES:
                archive.writestr(name, (ROOT / name).read_bytes())
        header, evidence, parent_files = parent_inputs(args.state_result_dir)
        (output / 'en63_generated.h').write_bytes(header)
        pcm, bindings = original_pcm(args.tts_zip, args.review_json, evidence)
        inputs = {'schema': 'english63-pool-profile-inputs-v1', 'native_state_ancestor': ANCESTOR,
                  'source_header_sha256': HEADER_SHA, 'source_tts_zip_sha256': c.TTS_SHA,
                  'source_review_json_sha256': c.REVIEW_SHA, 'source_weights_sha256': c.SOURCE_WEIGHTS,
                  'source_audio_clips': 63, 'source_pcm_samples': 2101760, 'unique_input_samples': 2101760,
                  'test_pcm_bytes': len(pcm), 'test_pcm_sha256': c.digest(pcm), 'rows': bindings,
                  'pcm_operation': 'byte-preserving-full63-concatenation',
                  'new_audio_generation_or_resampling': False, 'parent_evidence': evidence,
                  'request_raw_utf8': request_raw.decode('utf-8'), 'request_sha256': c.digest(request_raw),
                  'claim_raw_utf8': claim_raw.decode('utf-8'), 'claim_sha256': c.digest(claim_raw)}
        verify_inputs(inputs, pipeline_head, run_identity)
        c.write_json(output / 'INPUTS.json', inputs)
        build = compile_native(output, before, run_identity, pipeline_head)
        binaries = {row['variant_name']: row['binary_sha256'] for row in build['compiles']}
        (output / 'native.stderr').touch(exist_ok=False)
        save_progress(output / 'MEASUREMENTS.json', measurements)
        traces = {}
        with tempfile.TemporaryDirectory(prefix='english63-pool-original-pcm-') as temporary:
            path = Path(temporary) / 'full-original.pcm'
            path.write_bytes(pcm)
            start = time.monotonic()
            deadline = start + PROFILE['native_group_seconds']
            for expected in schedule():
                variant, kind = expected['variant_name'], expected['kind']
                binary = output / ('en63_pool_' + variant)
                trace_path = output / (kind + '-' + variant + '.trace') if expected['phase'] == 'trace' else None
                measurements['passes'].append({**expected, 'status': 'STARTED_NO_RETRY'})
                save_progress(output / 'MEASUREMENTS.json', measurements)
                record = native_pass(binary, expected, path, trace_path, selected_cpu, deadline)
                measurements['passes'][-1] = record
                save_progress(output / 'MEASUREMENTS.json', measurements)
                with (output / 'native.stderr').open('a') as stream:
                    stream.write(record['raw_stderr'])
                if trace_path is not None:
                    raw = trace_path.read_bytes()
                    need(c.digest(raw) == BASELINE_TRACE_SHA[kind], 'LIVE_BASELINE_OR_CANDIDATE_TRACE_IDENTITY')
                    if variant == 'baseline':
                        traces[kind] = trace_observation(raw)
                    else:
                        need(raw == (output / (kind + '-baseline.trace')).read_bytes(), 'LIVE_CANDIDATE_BIT_IDENTITY')
                pass_observation(record, expected, selected_cpu, traces[kind], binaries)
                need(time.monotonic() <= deadline, 'NATIVE_GROUP_WALL_LIMIT')
            measurements['native_group_elapsed_seconds'] = time.monotonic() - start
            pcm_after = c.digest(path.read_bytes())
        header_after, evidence_after, parent_after = parent_inputs(args.state_result_dir)
        measurements['final_allowed_cpus'] = sorted(os.sched_getaffinity(0))
        measurements['source_integrity'] = {
            'source_before': before, 'source_after': source_hashes(),
            'header_before': HEADER_SHA, 'header_after': c.digest((output / 'en63_generated.h').read_bytes()),
            'tts_before': c.TTS_SHA, 'tts_after': c.digest(args.tts_zip.read_bytes()),
            'review_before': c.REVIEW_SHA, 'review_after': c.digest(args.review_json.read_bytes()),
            'pcm_before': inputs['test_pcm_sha256'], 'pcm_after': pcm_after,
            'parent_terminal_before': ANCESTOR['terminal_sha256'],
            'parent_terminal_after': c.digest((args.state_result_dir / 'TERMINAL.json').read_bytes()),
            'parent_files_before': parent_files, 'parent_files_after': parent_after,
            'binaries_before': binaries,
            'binaries_after': {variant: c.digest((output / ('en63_pool_' + variant)).read_bytes()) for variant in PROFILE['variants']},
            'request_before': c.digest(request_raw), 'request_after': c.digest(request_path.read_bytes()),
            'claim_before': c.digest(claim_raw), 'claim_after': c.digest(claim_path.read_bytes())}
        need(header_after == header and exact(evidence_after, evidence), 'PARENT_SOURCE_UNCHANGED')
        save_progress(output / 'MEASUREMENTS.json', measurements)
        per_kind = measurements_result(measurements, traces, binaries)
        summary = summary_from({**inputs, 'file_sha256': c.digest((output / 'INPUTS.json').read_bytes())}, build,
                               {**measurements, 'file_sha256': c.digest((output / 'MEASUREMENTS.json').read_bytes())},
                               per_kind, before)
        c.write_json(output / 'SUMMARY.json', summary)
        c.write_json(output / 'REPLAY.json', saved_receipt(output))
        return verify_saved(output)
    except BaseException as error:
        if not (output / 'FAILED.json').exists():
            c.write_json(output / 'FAILED.json', {'schema': SCHEMA, 'status': 'FAILED_NO_RETRY',
                         'error_type': type(error).__name__, 'error': str(error),
                         'passes_started': len(measurements['passes']),
                         'completed_passes': sum(row['status'] == 'COMPLETED' for row in measurements['passes']),
                         'unknown_or_failed_calls_not_retried': True, 'optimizer_updates': 0,
                         **dict.fromkeys(FALSE_FIELDS, False)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-result-dir', type=Path)
    parser.add_argument('--tts-zip', type=Path)
    parser.add_argument('--review-json', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output-dir', type=Path)
    mode.add_argument('--verify-result-dir', type=Path)
    args = parser.parse_args(argv)
    if args.verify_result_dir:
        need(all(value is None for value in (args.state_result_dir, args.tts_zip, args.review_json)),
             'SAVED_VERIFY_ACCEPTS_NO_RUNTIME_INPUTS')
        result = verify_saved(args.verify_result_dir)
    else:
        result = run(args)
    print(canonical(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('EN63_POOL_PROFILE_FAILED: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
