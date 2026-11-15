#!/usr/bin/env python3
"""One frozen two-model LibriSpeech C screen; saved verification runs no model."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import io
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
import tarfile
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p = load('_en63_librispeech_parent', HERE / 'pool_profile.py')
old = load('_en63_librispeech_source', ROOT / 'research/english63_weak_pilot/librispeech_negative_events.py')
need, exact, digest, parse_json = p.need, p.exact, p.c.digest, p.parse_json
EXPERIMENT = 'english63-native-c-librispeech-screen'
REQUEST_ID = 'en63-native-c-librispeech-20261115-v1'
SCHEMA = 'english63-librispeech-screen-result-v1'
STATUS = 'MEASURED_CONTIGUOUS_READ_SPEECH_PROPOSALS_NOT_GOLD_FAR'
REFERENCE = {'run_id': 43538998200, 'artifact_id': 14090383839, 'zip_bytes': 47199,
             'zip_sha256': '7f28b4eddbee87435b9dc5b4c44e47a6d6ed200cbf9f4b8688d9b93cabeb4db4',
             'member': 'english-natural-read-speech-result.json',
             'member_sha256': '788a1b04b9ecb22fb96d4036d19456f460ce3ccecd4da69d53d96fea53b29510'}
CORPUS = {'url': 'https://www.openslr.org/resources/12/dev-clean.tar.gz', 'bytes': 337926286,
          'sha256': '76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3',
          'license': 'CC-BY-4.0'}
PROFILE = {'models': ['gru', 'dsc_tcn'], 'source_clips': 517, 'source_speakers': 40,
           'source_pcm_samples': 58646164, 'source_seconds': 3665.38525, 'order': 'source_member',
           'feed_chunk_samples': 160, 'frames_per_pass': 366537, 'feed_calls_per_pass': 366539,
           'final_feed_samples': 84, 'full_passes': 2, 'forward_frames': 733074,
           'supplied_samples_including_replays': 117292328, 'pool_time_major': 0,
           'native_child_seconds': 180, 'native_group_seconds': 420, 'state_bytes': 38192,
           'cpu_affinity': 'lowest-allowed-single-cpu', 'trace_record_bytes': 32}
SOURCES = ('research/english63_online_c/librispeech_screen.py',
           'research/english63_online_c/librispeech_screen.c',
           'research/english63_online_c/en63_stream.c', 'research/english63_online_c/en63_stream.h',
           'research/english63_online_c/pool_profile.py', 'research/english63_online_c/contract.py',
           'training/english_research_label_bridge.py',
           'research/english63_weak_pilot/librispeech_negative_events.py',
           'research/english63_weak_pilot/pretrained_quant_events.py',
           'research/english63_weak_pilot/pcm_frontend_parity.py')
RESULT_FILES = {'en63_generated.h', 'en63_librispeech', 'SOURCE.zip', 'PROTOCOL.json',
                'INPUTS.json', 'BUILD.json', 'MEASUREMENTS.json', 'SUMMARY.json', 'REPLAY.json',
                'gru.trace', 'dsc_tcn.trace', 'native.stderr'}
TRACE_HEADER = b'EN63LSC1' + struct.pack('<II', 1, 32)
TRACE_RECORD = struct.Struct('<QQ3fI')
CLASS_LABELS = {index: name for name, index in p.bridge.CLASSES.items()}
FLAGS = (*p.FLAGS, '-DEN63_POOL_TIME_MAJOR=0')
FALSE_FIELDS = ('formal_training_admitted', 'human_gold', 'trusted_negative', 'model_promotion_allowed',
                'model_selection_made', 'threshold_selection', 'runtime_default_changed', 'new_audio_generated')
NOT_RUN = ('independent_english_frr', 'continuous_negative_far_per_hour', 'word_end_latency', 'ssc305_board_profile')


def request_fixed():
    return {'schema': 'kws-research-v4-native-c-librispeech-request-v1', 'experiment': EXPERIMENT,
            'request_id': REQUEST_ID, 'pipeline_repository': 'jiying2007/kws-pipeline',
            'entrypoint': SOURCES[0], 'native_state_ancestor': p.ANCESTOR,
            'source_artifacts': {'negative_reference': REFERENCE, 'corpus': CORPUS}, 'profile': PROFILE,
            'optimizer_updates': 0, 'cpu_threads': 1, 'max_execution_seconds': 900,
            'max_verification_seconds': 180, 'new_audio_generated': False, 'new_tts_asr_ipa_calls': 0,
            'new_independent_audio_seconds': 0, 'formal_training_admitted': False,
            'human_gold': False, 'model_promotion_allowed': False}


def validate_request(raw, head):
    request = parse_json(raw)
    fixed = request_fixed()
    need(type(request) is dict and set(request) == set(fixed) | {'pipeline_commit', 'reason'} and
         all(exact(request[key], value) for key, value in fixed.items()) and
         type(head) is str and p.SHA1.fullmatch(head) and request['pipeline_commit'] == head and
         type(request['reason']) is str and 1 <= len(request['reason'].strip()) <= 500, 'LIBRISPEECH_FIXED_REQUEST')
    return request


def protocol():
    return {**request_fixed(), 'schema': 'english63-librispeech-screen-protocol-v1',
            'pass_order': list(p.c.KINDS), 'trace_layout': '<QQ3fI frame,sample,probabilities,class',
            'source_selection': 'SHA256(member_name) first8 big-endian mod5 == 0',
            'source_order': 'lexical-source-member', 'source_reset_calls': 0, 'inserted_samples': 0,
            'classes': p.bridge.CLASSES,
            'eof_padding_samples': 0, 'source_role': 'previously-exposed-untrusted-read-speech-screening',
            'event_clock': 'supplied-PCM-dependency-end-exclusive-not-word-end-or-wall-clock',
            'nonzero_proposals_are_valid_completed_results': True, 'no_result_driven_retries': True,
            **dict.fromkeys(FALSE_FIELDS, False), **dict.fromkeys(NOT_RUN, 'NOT_RUN')}


def source_hashes():
    return {name: digest((ROOT / name).read_bytes()) for name in SOURCES}


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def reference_rows(raw):
    need(type(raw) is bytes and digest(raw) == REFERENCE['member_sha256'], 'LIBRISPEECH_REFERENCE_BYTES')
    report = parse_json(raw)
    need(report['schema'] == old.POLICY and report['status'] ==
         'MEASURED_REAL_READ_SPEECH_SEGMENT_EVENTS_NOT_FIELD_FAR' and
         report['archive_sha256'] == CORPUS['sha256'] and report['license'] == CORPUS['license'] and
         exact(report['sampled_utterances'], 517) and exact(report['speaker_count'], 40) and
         exact(report['source_audio_seconds'], 3665.38525) and report['human_acoustic_gold'] is False and
         report['model_promotion_allowed'] is False, 'LIBRISPEECH_REFERENCE_SCOPE')
    rows = report['records']
    need(type(rows) is list and len(rows) == 517 and len({r['member'] for r in rows}) == 517,
         'LIBRISPEECH_REFERENCE_ROSTER')
    for row in rows:
        match = old.NAME.fullmatch(row['member'])
        need(match and old.selected(row['member']) and row['speaker_source_id'] == match.group(1) and
             all(row[key] is False for key in ('human_gold', 'trusted_negative', 'training_admitted')) and
             p.sha(row['flac_sha256']) and p.sha(row['transcript_sha256']) and
             p.is_int(row['sample_count'], 400, 640000), 'LIBRISPEECH_REFERENCE_ROW')
    need(len({r['speaker_source_id'] for r in rows}) == 40 and
         sum(r['sample_count'] for r in rows) == 58646164, 'LIBRISPEECH_REFERENCE_TOTALS')
    return {row['member']: row for row in rows}


def reference_input(root):
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'LIBRISPEECH_REFERENCE_ROOT')
    entries = list(root.iterdir())
    need(len(entries) == 1 and entries[0].name == REFERENCE['member'] and entries[0].is_file() and
         not entries[0].is_symlink() and entries[0].stat().st_size < 1024 * 1024, 'LIBRISPEECH_REFERENCE_FILE')
    raw = entries[0].read_bytes()
    reference_rows(raw)
    return raw


def join_pcm_files(paths, target):
    """Byte concatenation across arbitrary file boundaries; no sample is inserted."""
    total = 0
    with target.open('xb') as out:
        for path in paths:
            size = path.stat().st_size
            need(size > 0 and size % 2 == 0 and path.is_file() and not path.is_symlink(), 'LIBRISPEECH_PCM_PART')
            with path.open('rb') as stream:
                shutil.copyfileobj(stream, out, 1024 * 1024)
            total += size // 2
    need(target.stat().st_size == total * 2, 'LIBRISPEECH_PCM_JOIN_LENGTH')
    return total


def materialize_pcm(archive, reference, temporary):
    import numpy as np
    import soundfile as sf
    need(np.__version__ == '2.3.5' and sf.__version__ == '0.13.1', 'LIBRISPEECH_PINNED_DECODER')
    old.verified_archive(archive)
    parts, rows = {}, {}
    with tarfile.open(archive, 'r:gz') as tar:
        members, picked = old.source_members(tar)
        need(len({m.name for m in members}) == len(members), 'LIBRISPEECH_TAR_DUPLICATES')
        transcripts = old.extract_transcripts(tar, members)
        need(len(picked) == 517 and {m.name for m in picked} == set(reference), 'LIBRISPEECH_SELECTION_DRIFT')
        for index, member in enumerate(picked):
            prior = reference[member.name]
            speaker, chapter, utterance = old.NAME.fullmatch(member.name).groups()
            text = transcripts[utterance]
            need(old.no_requested_wake(text) and digest(text.encode('utf-8')) == prior['transcript_sha256'] and
                 speaker == prior['speaker_source_id'], 'LIBRISPEECH_TRANSCRIPT_OR_SPEAKER')
            stream = tar.extractfile(member)
            need(stream is not None, 'LIBRISPEECH_FLAC_UNREADABLE')
            raw = stream.read()
            need(digest(raw) == prior['flac_sha256'], 'LIBRISPEECH_FLAC_IDENTITY')
            with sf.SoundFile(io.BytesIO(raw)) as wav:
                need(wav.samplerate == 16000 and wav.channels == 1 and wav.subtype == 'PCM_16' and
                     wav.frames == prior['sample_count'], 'LIBRISPEECH_SOURCE_PCM_GEOMETRY')
                samples = wav.read(dtype='int16', always_2d=False)
            need(type(samples) is np.ndarray and samples.ndim == 1 and samples.dtype == np.dtype('int16') and
                 len(samples) == prior['sample_count'], 'LIBRISPEECH_DECODE_LENGTH')
            pcm = samples.astype('<i2', copy=False).tobytes()
            path = temporary / ('part-' + str(index) + '.pcm')
            path.write_bytes(pcm)
            parts[member.name] = path
            rows[member.name] = {key: prior[key] for key in
                                ('member', 'flac_sha256', 'transcript_sha256', 'speaker_source_id', 'sample_count',
                                 'human_gold', 'trusted_negative', 'training_admitted')}
            rows[member.name]['pcm_sha256'] = digest(pcm)
    roster, offset = [], 0
    for name in sorted(parts):
        roster.append({**rows[name], 'concatenation_start_sample': offset})
        offset += rows[name]['sample_count']
    target = temporary / 'full-original.pcm'
    need(join_pcm_files([parts[name] for name in sorted(parts)], target) == offset == 58646164,
         'LIBRISPEECH_FINAL_PCM_TOTAL')
    return target, roster, {'numpy': np.__version__, 'soundfile': sf.__version__,
                            'libsndfile': sf.__libsndfile_version__, 'sample_type': 'little-endian-PCM16',
                            'resampled': False, 'normalized': False}


def verify_roster(rows, reference):
    need(type(rows) is list and len(rows) == 517 and [r['member'] for r in rows] == sorted(reference),
         'LIBRISPEECH_SAVED_LEXICAL_ROSTER')
    offset = 0
    for row in rows:
        prior = reference[row['member']]
        fields = ('member', 'flac_sha256', 'transcript_sha256', 'speaker_source_id', 'sample_count',
                  'human_gold', 'trusted_negative', 'training_admitted')
        need(set(row) == set(fields) | {'pcm_sha256', 'concatenation_start_sample'} and
             all(exact(row[key], prior[key]) for key in fields) and p.sha(row['pcm_sha256']) and
             exact(row['concatenation_start_sample'], offset), 'LIBRISPEECH_SAVED_SOURCE_IDENTITY')
        offset += row['sample_count']
    need(offset == 58646164, 'LIBRISPEECH_SAVED_SOURCE_TOTAL')


def trace_observation(raw, rows, samples=58646164):
    frames = 1 + (samples - 400) // 160
    need(raw[:16] == TRACE_HEADER and len(raw) == 16 + frames * 32, 'LIBRISPEECH_TRACE_LAYOUT')
    ends = [r['concatenation_start_sample'] + r['sample_count'] for r in rows]
    need(rows and ends[-1] == samples, 'LIBRISPEECH_TRACE_SOURCE_TOTAL')
    threshold = struct.unpack('<f', struct.pack('<f', 0.8))[0]
    candidate, confirmations, blocked_until, events = 0, 0, 0, []
    for index, (frame, sample, *values) in enumerate(TRACE_RECORD.iter_unpack(raw[16:]), 1):
        probabilities, actual = values[:3], values[3]
        need(frame == index and sample == 400 + (index - 1) * 160 and
             all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities) and
             abs(sum(probabilities) - 1) <= 3e-7 and actual in (0, 1, 2), 'LIBRISPEECH_TRACE_VALUES')
        chosen = max(range(3), key=lambda i: probabilities[i])
        expected = 0
        if index < 50 or index < blocked_until or chosen == 0 or probabilities[chosen] < threshold:
            candidate, confirmations = 0, 0
        else:
            confirmations = confirmations + 1 if chosen == candidate else 1
            candidate = chosen
            if confirmations >= 3:
                expected, blocked_until = chosen, index + 100
                candidate, confirmations = 0, 0
        need(actual == expected, 'LIBRISPEECH_RAW_PROBABILITY_DECODER_EVENT')
        if actual:
            # The endpoint is exclusive: sample exactly at a splice still depends
            # on the last sample of the preceding source interval.
            position = bisect.bisect_left(ends, sample)
            source = rows[position]
            events.append({'frame': frame, 'available_sample': sample, 'class': actual,
                           'keyword': CLASS_LABELS[actual],
                           'probabilities': probabilities, 'source_interval_member': source['member'],
                           'exact_splice_boundary': sample in ends[:-1],
                           'near_splice_boundary_200ms': any(abs(sample - edge) <= 3200 for edge in ends[:-1])})
    return {'trace_sha256': digest(raw), 'frames': frames, 'first_available_sample': 400,
            'last_available_sample': 400 + (frames - 1) * 160, 'proposal_count': len(events),
            'proposals_by_class': {str(k): sum(e['class'] == k for e in events) for k in (1, 2)},
            'events': events, 'eof_events': 0, 'eof_padding_samples': 0}


def command_for(compiler, source, output):
    here = Path(source) / 'research/english63_online_c'
    return [compiler, *FLAGS, '-I', str(here), '-I', str(output), str(here / 'en63_stream.c'),
            str(here / 'librispeech_screen.c'), '-lm', '-o', str(Path(output) / 'en63_librispeech')]


def compile_native(output, before, identity, head):
    need(platform.system() == 'Linux' and not (HERE / 'en63_generated.h').exists() and
         not (HERE / 'en63_generated.h').is_symlink(), 'LIBRISPEECH_COMPILER_HEADER_SCOPE')
    compiler = shutil.which('gcc')
    need(compiler is not None, 'LIBRISPEECH_GCC_REQUIRED')
    compiler = str(Path(compiler).absolute())
    real = Path(compiler).resolve()
    compiler_hash = sha_file(real)
    command = command_for(compiler, ROOT, output)
    result = subprocess.run(command, capture_output=True, text=True, timeout=90, env=p.native_environment())
    need(result.returncode == 0, 'LIBRISPEECH_COMPILE_FAILED:' + result.stderr)
    binary = output / 'en63_librispeech'
    build = {'schema': 'english63-librispeech-build-v1', 'compiler': compiler, 'compiler_realpath': str(real),
             'compiler_sha256': compiler_hash, 'compiler_version': subprocess.check_output(
                 [compiler, '--version'], text=True, timeout=15, env=p.native_environment()),
             'command': command, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
             'source_root': str(ROOT), 'output_root': str(output), 'source_sha256': before,
             'binary_sha256': sha_file(binary), 'header_sha256': p.HEADER_SHA,
             'public_run': identity, 'pipeline_head': head, 'physical_ssc305': False,
             'machine': platform.machine(), 'libc': list(platform.libc_ver())}
    need(sha_file(real) == compiler_hash, 'LIBRISPEECH_COMPILER_CHANGED')
    p.c.write_json(output / 'BUILD.json', build)
    return build


def native_pass(output, kind, pcm, cpu, deadline):
    remaining = deadline - time.monotonic()
    need(remaining > 0, 'LIBRISPEECH_NATIVE_GROUP_BUDGET')
    command = [str(output / 'en63_librispeech'), '--supervise', kind, str(pcm), str(output / (kind + '.trace')),
               str(cpu), '58646164']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=p.native_environment(), start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=min(180, remaining))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate(timeout=10)
    return {'kind': kind, 'status': 'COMPLETED' if process.returncode == 0 and not timed_out else 'FAILED',
            'command': command, 'exit_code': process.returncode, 'timed_out': timed_out,
            'raw_stdout': stdout, 'raw_stderr': stderr}


def pass_observation(record, kind, build, cpu, trace):
    need(record['kind'] == kind and record['status'] == 'COMPLETED' and exact(record['exit_code'], 0) and
         record['timed_out'] is False and type(record['raw_stdout']) is str and
         type(record['raw_stderr']) is str, 'LIBRISPEECH_NATIVE_TERMINAL')
    cmd = record['command']
    need(type(cmd) is list and len(cmd) == 7 and cmd[:3] ==
         [str(Path(build['output_root']) / 'en63_librispeech'), '--supervise', kind] and
         Path(cmd[3]).is_absolute() and Path(cmd[3]).name == 'full-original.pcm' and
         cmd[4:] == [str(Path(build['output_root']) / (kind + '.trace')), str(cpu), '58646164'],
         'LIBRISPEECH_NATIVE_COMMAND')
    lines = record['raw_stdout'].splitlines()
    need(len(lines) == 2, 'LIBRISPEECH_NATIVE_STDOUT')
    body, lifetime = [parse_json(line) for line in lines]
    fixed = {'schema': 'english63-librispeech-native-pass-v1', 'kind': kind,
             'pcm_samples': 58646164, 'frames': 366537, 'feed_calls': 366539, 'final_feed_samples': 84,
             'proposal_count': trace['proposal_count'], 'state_bytes': 38192,
             'weight_parameter_bytes': (6531 if kind == 'gru' else 4803) * 4,
             'frontend_table_bytes': 3894, 'pool_time_major': 0, 'affinity_cpu': cpu, 'affinity_single_cpu': True,
             'initializations': 1, 'reset_calls': 0, 'discontinuity_calls': 0, 'finish_calls': 1,
             'eof_padding_samples': 0, 'eof_events': 0}
    need(set(body) == set(fixed) | {'cpu_ns', 'wall_ns'} and
         all(exact(body[key], value) for key, value in fixed.items()) and
         p.is_int(body['cpu_ns'], 1) and p.is_int(body['wall_ns'], 1, 180000000000), 'LIBRISPEECH_NATIVE_COUNTERS')
    need(set(lifetime) == {'schema', 'child_exit_code', 'child_signaled', 'process_lifetime_peak_rss_kib'} and
         lifetime['schema'] == 'english63-librispeech-child-lifetime-v1' and exact(lifetime['child_exit_code'], 0) and
         lifetime['child_signaled'] is False and p.is_int(lifetime['process_lifetime_peak_rss_kib'], 1),
         'LIBRISPEECH_NATIVE_CHILD_LIFETIME')
    return {'kind': kind, 'source_weights_sha256': p.c.SOURCE_WEIGHTS[kind], **trace,
            'pcm_samples': body['pcm_samples'], 'feed_calls': body['feed_calls'], 'final_feed_samples': 84,
            'state_bytes': body['state_bytes'], 'cpu_seconds': body['cpu_ns'] / 1e9,
            'wall_seconds_including_pcm_and_trace_io': body['wall_ns'] / 1e9,
            'process_lifetime_peak_rss_kib': lifetime['process_lifetime_peak_rss_kib']}


def verify_inputs(inputs, build):
    need(inputs['schema'] == 'english63-librispeech-inputs-v1' and
         exact(inputs['native_state_ancestor'], p.ANCESTOR) and exact(inputs['negative_reference'], REFERENCE) and
         exact(inputs['corpus'], CORPUS) and inputs['header_sha256'] == p.HEADER_SHA and
         p.sha(inputs['concatenated_pcm_sha256']), 'LIBRISPEECH_INPUT_IDENTITIES')
    reference = reference_rows(inputs['reference_raw_utf8'].encode('utf-8'))
    verify_roster(inputs['rows'], reference)
    decoder = inputs['pcm_decoder']
    need(set(decoder) == {'numpy', 'soundfile', 'libsndfile', 'sample_type', 'resampled', 'normalized'} and
         decoder['numpy'] == '2.3.5' and decoder['soundfile'] == '0.13.1' and
         type(decoder['libsndfile']) is str and decoder['libsndfile'] and
         decoder['sample_type'] == 'little-endian-PCM16' and decoder['resampled'] is False and
         decoder['normalized'] is False, 'LIBRISPEECH_SAVED_PCM_DECODER')
    p.parent_evidence_check(inputs['parent_evidence'])
    request_raw = inputs['request_raw_utf8'].encode('utf-8')
    validate_request(request_raw, build['pipeline_head'])
    need(digest(request_raw) == inputs['request_sha256'], 'LIBRISPEECH_REQUEST_BYTES')
    claim_raw = inputs['claim_raw_utf8'].encode('utf-8')
    expected = {'schema': 'kws-research-v4-native-c-librispeech-start-claim-v1',
                'request_sha256': inputs['request_sha256'], 'pipeline_commit': build['pipeline_head'],
                'source_state_artifact_id': 14141171320, 'source_negative_artifact_id': 14090383839,
                'run_id': build['public_run']['run_id'], 'run_attempt': '1', 'optimizer_updates': 0,
                'status': 'CONSUMED_BEFORE_NATIVE_C_LIBRISPEECH_SCREEN'}
    need(digest(claim_raw) == inputs['claim_sha256'] and exact(parse_json(claim_raw), expected),
         'LIBRISPEECH_CONSUMED_CLAIM')


def summary_from(inputs, build, measurements, models):
    return {'schema': SCHEMA, 'status': STATUS, 'experiment': EXPERIMENT,
            'public_run': build['public_run'], 'pipeline_head': build['pipeline_head'],
            'request_sha256': inputs['request_sha256'], 'native_state_ancestor': p.ANCESTOR,
            'negative_reference': REFERENCE, 'corpus': CORPUS, 'profile': PROFILE,
            'source_header_sha256': p.HEADER_SHA, 'source_weights_sha256': p.c.SOURCE_WEIGHTS,
            'classes': p.bridge.CLASSES,
            'source_sha256': build['source_sha256'], 'source_manifest_sha256': inputs['file_sha256'],
            'measurements_sha256': measurements['file_sha256'], 'models': models,
            'source_clips': 517, 'source_speakers': 40, 'unique_source_samples': 58646164,
            'unique_source_seconds': 3665.38525, 'passes': 2, 'total_frames': 733074,
            'total_supplied_samples': 117292328, 'new_independent_audio_seconds': 0,
            'source_reset_calls': 0, 'inserted_samples': 0, 'eof_padding_samples': 0,
            'new_tts_asr_ipa_calls': 0, 'optimizer_updates': 0,
            'native_group_elapsed_seconds': measurements['native_group_elapsed_seconds'],
            'source_integrity': measurements['source_integrity'],
            'screening_scope': 'previously-exposed-read-speech-untrusted-negative-not-household-FAR',
            'event_clock': protocol()['event_clock'],
            **dict.fromkeys(FALSE_FIELDS, False), **dict.fromkeys(NOT_RUN, 'NOT_RUN')}


def saved_receipt(root):
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'LIBRISPEECH_SAVED_ROOT')
    entries = list(root.rglob('*'))
    need(all(path.is_file() and not path.is_symlink() for path in entries) and
         {path.name for path in entries} in (RESULT_FILES, RESULT_FILES - {'REPLAY.json'}) and
         sum(path.stat().st_size for path in entries) < 64 * 1024 * 1024, 'LIBRISPEECH_RESULT_CLOSURE')
    files = {path.name: p.artifact_identity(path.read_bytes()) for path in sorted(entries) if path.name != 'REPLAY.json'}
    need(all(row['bytes'] > 0 for name, row in files.items() if name != 'native.stderr') and
         files['en63_generated.h']['sha256'] == p.HEADER_SHA and
         (root / 'en63_librispeech').read_bytes().startswith(b'\x7fELF') and
         exact(p.read_json(root / 'PROTOCOL.json'), protocol()), 'LIBRISPEECH_SAVED_PROTOCOL')
    inputs, build, measurements = [p.read_json(root / name) for name in ('INPUTS.json', 'BUILD.json', 'MEASUREMENTS.json')]
    need(build['schema'] == 'english63-librispeech-build-v1' and
         type(build['pipeline_head']) is str and p.SHA1.fullmatch(build['pipeline_head']) and
         all(type(build[key]) is str and Path(build[key]).is_absolute() for key in
             ('compiler', 'compiler_realpath', 'source_root', 'output_root')) and
         p.sha(build['compiler_sha256']) and type(build['compiler_version']) is str and build['compiler_version'] and
         exact(build['command'], command_for(build['compiler'], build['source_root'], build['output_root'])) and
         exact(build['exit_code'], 0) and build['header_sha256'] == p.HEADER_SHA and
         build['binary_sha256'] == files['en63_librispeech']['sha256'] and
         build['physical_ssc305'] is False, 'LIBRISPEECH_SAVED_BUILD')
    p.public_run(build['public_run'])
    with zipfile.ZipFile(root / 'SOURCE.zip') as archive:
        need(len(archive.infolist()) == len(SOURCES) and set(archive.namelist()) == set(SOURCES) and
             all(not info.is_dir() and info.file_size < 1024 * 1024 and
                 (info.external_attr >> 16) & 0o170000 != 0o120000 for info in archive.infolist()),
             'LIBRISPEECH_SOURCE_ARCHIVE')
        source = {name: digest(archive.read(name)) for name in SOURCES}
    need(exact(source, source_hashes()) and exact(source, build['source_sha256']), 'LIBRISPEECH_SOURCE_BYTES')
    verify_inputs(inputs, build)
    cpus = measurements['initial_allowed_cpus']
    need(measurements['schema'] == 'english63-librispeech-measurements-v1' and
         exact(measurements['public_run'], build['public_run']) and
         measurements['pipeline_head'] == build['pipeline_head'] and
         measurements['request_sha256'] == inputs['request_sha256'] and
         type(cpus) is list and cpus and all(p.is_int(cpu, 0, 1023) for cpu in cpus) and
         cpus == sorted(set(cpus)) and exact(measurements['selected_cpu'], min(cpus)) and
         exact(measurements['final_allowed_cpus'], cpus) and
         type(measurements['native_group_elapsed_seconds']) in (int, float) and
         0 < measurements['native_group_elapsed_seconds'] <= 420 and
         type(measurements['passes']) is list and len(measurements['passes']) == 2,
         'LIBRISPEECH_SAVED_EXECUTION')
    parent = p.parent_evidence_check(inputs['parent_evidence'])['TERMINAL.json']['files_sha256']
    expected_integrity = {'source_before': source, 'source_after': source,
                         'header_before': p.HEADER_SHA, 'header_after': p.HEADER_SHA,
                         'corpus_before': CORPUS['sha256'], 'corpus_after': CORPUS['sha256'],
                         'reference_before': REFERENCE['member_sha256'], 'reference_after': REFERENCE['member_sha256'],
                         'pcm_before': inputs['concatenated_pcm_sha256'], 'pcm_after': inputs['concatenated_pcm_sha256'],
                         'parent_files_before': parent, 'parent_files_after': parent,
                         'binary_before': build['binary_sha256'], 'binary_after': build['binary_sha256'],
                         'request_before': inputs['request_sha256'], 'request_after': inputs['request_sha256'],
                         'claim_before': inputs['claim_sha256'], 'claim_after': inputs['claim_sha256']}
    need(exact(measurements['source_integrity'], expected_integrity), 'LIBRISPEECH_BEFORE_AFTER_BINDINGS')
    need(measurements['passes'][0]['command'][3] == measurements['passes'][1]['command'][3],
         'LIBRISPEECH_SAME_PCM_PATH')
    models = [pass_observation(record, kind, build, measurements['selected_cpu'],
                              trace_observation((root / (kind + '.trace')).read_bytes(), inputs['rows']))
              for kind, record in zip(p.c.KINDS, measurements['passes'])]
    need((root / 'native.stderr').read_text() == ''.join(row['raw_stderr'] for row in measurements['passes']),
         'LIBRISPEECH_STDERR_BINDING')
    summary = summary_from({**inputs, 'file_sha256': files['INPUTS.json']['sha256']}, build,
                           {**measurements, 'file_sha256': files['MEASUREMENTS.json']['sha256']}, models)
    need(exact(p.read_json(root / 'SUMMARY.json'), summary), 'LIBRISPEECH_SUMMARY_RECOMPUTED')
    return {'schema': 'english63-librispeech-screen-replay-v1', 'status': 'VERIFIED_SAVED_LIBRISPEECH_SCREEN',
            'files': files, 'files_sha256': {name: row['sha256'] for name, row in files.items()},
            'summary_sha256': files['SUMMARY.json']['sha256'], 'model_forward_calls': 0, 'optimizer_updates': 0,
            'new_audio_asr_calls': 0, 'formal_training_admitted': False, 'model_promotion_allowed': False,
            'runtime_default_changed': False}


def verify_saved(root):
    expected = saved_receipt(root)
    need(exact(p.read_json(root / 'REPLAY.json'), expected), 'LIBRISPEECH_SAVED_RECEIPT')
    return expected


def run(args):
    identity = p.require_public_actions()  # Before original input, compilation, or model activity.
    output = args.output_dir
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not any(path.is_symlink() for path in (output, *output.parents)), 'LIBRISPEECH_CREATE_ONLY_OUTPUT')
    need(all(path is not None and path.is_absolute() for path in
             (args.state_result_dir, args.negative_evidence_dir, args.source_tar)), 'LIBRISPEECH_ABSOLUTE_INPUTS')
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    request_path, claim_path = output.parent / 'REQUEST.json', output.parent.parent / 'claim/CLAIM.json'
    request_raw, claim_raw = request_path.read_bytes(), claim_path.read_bytes()
    validate_request(request_raw, head)
    cpus = sorted(os.sched_getaffinity(0))
    need(cpus and all(p.is_int(cpu, 0, 1023) for cpu in cpus), 'LIBRISPEECH_AFFINITY')
    output.mkdir()
    measurements = {'schema': 'english63-librispeech-measurements-v1', 'public_run': identity,
                    'pipeline_head': head, 'request_sha256': digest(request_raw),
                    'initial_allowed_cpus': cpus, 'selected_cpu': cpus[0], 'passes': []}
    try:
        before = source_hashes()
        p.c.write_json(output / 'PROTOCOL.json', protocol())
        with zipfile.ZipFile(output / 'SOURCE.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in SOURCES:
                archive.writestr(name, (ROOT / name).read_bytes())
        header, evidence, parent_files = p.parent_inputs(args.state_result_dir)
        (output / 'en63_generated.h').write_bytes(header)
        reference_raw = reference_input(args.negative_evidence_dir)
        reference = reference_rows(reference_raw)
        with tempfile.TemporaryDirectory(prefix='en63-librispeech-original-pcm-') as temporary:
            pcm, rows, decoder = materialize_pcm(args.source_tar, reference, Path(temporary))
            inputs = {'schema': 'english63-librispeech-inputs-v1', 'native_state_ancestor': p.ANCESTOR,
                      'negative_reference': REFERENCE, 'corpus': CORPUS, 'header_sha256': p.HEADER_SHA,
                      'rows': rows, 'concatenated_pcm_sha256': sha_file(pcm), 'pcm_decoder': decoder,
                      'parent_evidence': evidence, 'reference_raw_utf8': reference_raw.decode('utf-8'),
                      'request_raw_utf8': request_raw.decode('utf-8'), 'request_sha256': digest(request_raw),
                      'claim_raw_utf8': claim_raw.decode('utf-8'), 'claim_sha256': digest(claim_raw)}
            verify_inputs(inputs, {'pipeline_head': head, 'public_run': identity})
            p.c.write_json(output / 'INPUTS.json', inputs)
            build = compile_native(output, before, identity, head)
            (output / 'native.stderr').touch(exist_ok=False)
            p.save_progress(output / 'MEASUREMENTS.json', measurements)
            start = time.monotonic()
            deadline = start + 420
            for kind in p.c.KINDS:
                measurements['passes'].append({'kind': kind, 'status': 'STARTED_NO_RETRY'})
                p.save_progress(output / 'MEASUREMENTS.json', measurements)
                record = native_pass(output, kind, pcm, cpus[0], deadline)
                measurements['passes'][-1] = record
                p.save_progress(output / 'MEASUREMENTS.json', measurements)
                with (output / 'native.stderr').open('a') as out:
                    out.write(record['raw_stderr'])
                need(record['status'] == 'COMPLETED', 'LIBRISPEECH_NATIVE_FAILED_NO_RETRY')
                trace = trace_observation((output / (kind + '.trace')).read_bytes(), rows)
                pass_observation(record, kind, build, cpus[0], trace)
                need(time.monotonic() <= deadline, 'LIBRISPEECH_NATIVE_GROUP_LIMIT')
            measurements['native_group_elapsed_seconds'] = time.monotonic() - start
            pcm_after = sha_file(pcm)
        header_after, evidence_after, parent_after = p.parent_inputs(args.state_result_dir)
        need(header_after == header and exact(evidence_after, evidence), 'LIBRISPEECH_PARENT_UNCHANGED')
        measurements['final_allowed_cpus'] = sorted(os.sched_getaffinity(0))
        measurements['source_integrity'] = {
            'source_before': before, 'source_after': source_hashes(),
            'header_before': p.HEADER_SHA, 'header_after': sha_file(output / 'en63_generated.h'),
            'corpus_before': CORPUS['sha256'], 'corpus_after': sha_file(args.source_tar),
            'reference_before': REFERENCE['member_sha256'],
            'reference_after': digest(reference_input(args.negative_evidence_dir)),
            'pcm_before': inputs['concatenated_pcm_sha256'], 'pcm_after': pcm_after,
            'parent_files_before': parent_files, 'parent_files_after': parent_after,
            'binary_before': build['binary_sha256'], 'binary_after': sha_file(output / 'en63_librispeech'),
            'request_before': digest(request_raw), 'request_after': digest(request_path.read_bytes()),
            'claim_before': digest(claim_raw), 'claim_after': digest(claim_path.read_bytes())}
        p.save_progress(output / 'MEASUREMENTS.json', measurements)
        models = [pass_observation(record, kind, build, cpus[0],
                                  trace_observation((output / (kind + '.trace')).read_bytes(), rows))
                  for kind, record in zip(p.c.KINDS, measurements['passes'])]
        p.c.write_json(output / 'SUMMARY.json', summary_from(
            {**inputs, 'file_sha256': sha_file(output / 'INPUTS.json')}, build,
            {**measurements, 'file_sha256': sha_file(output / 'MEASUREMENTS.json')}, models))
        p.c.write_json(output / 'REPLAY.json', saved_receipt(output))
        return verify_saved(output)
    except BaseException as error:
        if not (output / 'FAILED.json').exists():
            p.c.write_json(output / 'FAILED.json', {'schema': SCHEMA, 'status': 'FAILED_NO_RETRY',
                         'error_type': type(error).__name__, 'error': str(error),
                         'passes_started': len(measurements['passes']),
                         'completed_passes': sum(row['status'] == 'COMPLETED' for row in measurements['passes']),
                         'optimizer_updates': 0, **dict.fromkeys(FALSE_FIELDS, False)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('state-result-dir', 'negative-evidence-dir', 'source-tar'):
        parser.add_argument('--' + name, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output-dir', type=Path)
    mode.add_argument('--verify-result-dir', type=Path)
    args = parser.parse_args(argv)
    if args.verify_result_dir:
        need(all(value is None for value in (args.state_result_dir, args.negative_evidence_dir, args.source_tar)),
             'LIBRISPEECH_SAVED_ONLY_MODE')
        result = verify_saved(args.verify_result_dir)
    else:
        result = run(args)
    print(p.canonical(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('EN63_LIBRISPEECH_SCREEN_FAILED: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
